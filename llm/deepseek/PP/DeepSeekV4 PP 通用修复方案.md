# DeepSeekV4 PP 通用修复方案

记录时间：2026-05-13

本文记录 DeepSeekV4 开启 Pipeline Parallel(PP) 后暴露的两类问题，以及一套可支持任意 `pp` 度数的通用修改方案。当前方案先聚焦 `num_mtp_modules = 0` 的 PP 正确性；`MTP + PP` 需要额外传递 `mtp_input_offsets`，建议作为后续独立改造。

## 1. 问题背景

当前 DeepSeekV4 的 forward 有两个和普通 LLM 不同的结构特征：

1. 主干 hidden state 形状是 `[B, S, hc_mult, D]`，最终必须经过 `hc_head` 压成 `[B, S, D]` 后才能进入 `norm/output`。
2. MoE hash routing 依赖真实 token id，即 `input_ids`，而不是只依赖 hidden state。

通用 `pipeline_llm` 默认按普通 LLM 模块结构切分，通常只认识：

```text
tok_embeddings
layers.*
norm
output
```

这会导致两个独立问题：

1. 非首 PP stage 收到的是上一 stage 的 hidden state，却在 `forward()` 中把它误当作原始 token ids，重新生成 `input_ids`。
2. 最后 PP stage 缺少 `hc_head`，导致 `output` 直接作用于 `[B, S, hc_mult, D]`，产生错误 logits shape。

## 2. error_1.log 的直接根因

`deepseek_v4_285b_4layers_debug.toml` 设置：

```toml
pipeline_parallel_degree = 2
pipeline_parallel_schedule = "Interleaved1F1B"
num_mtp_modules = 0
```

该 schedule 下实际会有 4 个 virtual stages。日志中可见：

```text
stage_idx 1: ['layers.1', 'layers.2']
stage_idx 3: ['norm', 'output']
```

`layers.1/layers.2` 仍属于 hash routing 层，`n_hash_layers=3`，因此 MoE router 需要真实 token id：

```python
selected_experts_indices = self.tid2eid[input_ids.flatten()]
```

但当前非首 stage 走了错误逻辑：

```python
input_ids = tokens[:, :seq_len].detach().long()
h = tokens[:, :seq_len]
```

此时 `tokens` 实际是上一 stage 的 hidden state，形状类似 `[B, S, hc_mult, D]`，不是 token ids。日志里的：

```text
67108864 = 1 * 4096 * 4 * 4096
```

正是 hidden state 被 flatten 后误当作 `input_ids` 使用造成的。

因此 `error_1.log` 的直接根因是：`input_ids` 没有作为 sidecar 数据跨 PP stage 传递。

## 3. error.log 的直接根因

`deepseek_v4_285b_43layers_4k_128die.toml` 中最后 stage 日志为：

```text
stage_idx 3: ['layers.33', ..., 'layers.42', 'norm', 'output']
```

缺少 `hc_head`。

DeepSeekV4 正确输出路径应为：

```text
[B, S, hc_mult, D]
  -> hc_head
[B, S, D]
  -> norm
[B, S, D]
  -> output
[B, S, vocab]
```

缺少 `hc_head` 后实际路径变为：

```text
[B, S, hc_mult, D]
  -> norm
[B, S, hc_mult, D]
  -> output
[B, S, hc_mult, vocab]
```

然后 `cross_entropy` 对维度解释错误，最终报：

```text
RuntimeError: Expected target size [4096, 129280], got [4096]
```

因此 `error.log` 的直接根因是：最后 PP stage 缺少 `hc_head`。

## 4. 两类问题的关系

这两类问题不是同一个具体报错，但都来自同一个共性事实：

```text
DeepSeekV4 不能直接复用普通 LLM 的通用 PP 切分和 forward 协议。
```

只修 `hc_head` placement 不能修复 `error_1.log`。

只修 `input_ids` sidecar 不能修复 `error.log`。

必须同时修复：

1. DeepSeekV4 专用 PP 切分。
2. DeepSeekV4 专用 PP forward payload 协议。

## 5. 通用修改方案

### 5.1 新增 DeepSeekV4 专用 PP 切分

需要新增 DeepSeekV4 专用 PP 切分文件：

```text
torchtitan_npu/models/deepseek_v4/infra/pipeline_parallel.py
```

新增这个文件的原因是：通用 `pipeline_llm` 不了解 DeepSeekV4 的模型结构。DeepSeekV4 的末尾不是简单的 `norm/output`，而是必须按下面的顺序执行：

```text
hc_head -> norm -> output
```

同时 `hc_head_fn`、`hc_head_base`、`hc_head_scale` 是挂在 `DeepSeekV4Model` 顶层的 `nn.Parameter`，不是 `nn.Module` child。通用 `_split_module` 只处理 child module，不能正确处理这些顶层参数在 PP stage 之间的归属。因此需要一个 DeepSeekV4 专用入口，在 split 前生成正确 stage 列表，在 split 后清理非 last stage 上残留的无用 `hc_head_*` 参数。

该文件建议包含以下函数：

```python
_get_num_virtual_stages(...)
generate_deepseek_v4_fqn_per_model_part(...)
_validate_deepseek_v4_stage_modules(...)
_cleanup_unused_hc_head_parameters(...)
pipeline_deepseek_v4(...)
```

各函数职责如下：

```text
_get_num_virtual_stages
  根据 pipeline_parallel_degree、pipeline schedule、pipeline_parallel_layers_per_stage
  计算模型应被切成多少个 virtual stages。

generate_deepseek_v4_fqn_per_model_part
  生成每个 virtual stage 应包含的 module fqn 列表。
  first stage 固定包含 tok_embeddings。
  last stage 固定包含 hc_head/norm/output。

_validate_deepseek_v4_stage_modules
  在真正 split 前校验 stage 列表是否正确，避免错误延迟到 FSDP、MoE 或
  pipeline runtime 深处才暴露。

_cleanup_unused_hc_head_parameters
  split 后删除非 last stage 上残留的 hc_head_fn/hc_head_base/hc_head_scale 等顶层参数。

pipeline_deepseek_v4
  DeepSeekV4 的 PP 总入口，负责生成 stage 列表、split model、清理参数、
  调用 parallelize_fn，并构建 pipeline schedule。
```

代码骨架如下：

```python
def pipeline_deepseek_v4(
    model,
    parallel_dims,
    job_config,
    device,
    model_args,
    parallelize_fn,
    loss_fn,
):
    num_virtual_stages = _get_num_virtual_stages(
        parallel_dims,
        job_config,
        model_args.n_layers,
        job_config.parallelism.pipeline_parallel_first_stage_less_layers,
        job_config.parallelism.pipeline_parallel_last_stage_less_layers,
    )

    module_names_per_stage = job_config.parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_deepseek_v4_fqn_per_model_part(
            num_virtual_stages,
            model_args.n_layers,
            job_config.parallelism.pipeline_parallel_first_stage_less_layers,
            job_config.parallelism.pipeline_parallel_last_stage_less_layers,
        )

    _validate_deepseek_v4_stage_modules(
        module_names_per_stage,
        model_args.n_layers,
        num_virtual_stages,
    )

    stages, model_parts = pipeline_module_split(
        model,
        parallel_dims.get_mesh("pp"),
        job_config.parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

    for i, model_part in enumerate(model_parts):
        _cleanup_unused_hc_head_parameters(model_part)
        model_part = parallelize_fn(model_part, parallel_dims, job_config)
        model_parts[i] = model_part
        stages[i].submod = model_part

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)
    return pp_schedule, model_parts, has_first_stage, has_last_stage
```

在：

```text
torchtitan_npu/models/deepseek_v4/__init__.py
```

将：

```python
from torchtitan.distributed.pipeline_parallel import pipeline_llm
...
pipelining_fn=pipeline_llm
```

替换为：

```python
from torchtitan_npu.models.deepseek_v4.infra.pipeline_parallel import (
    pipeline_deepseek_v4,
)
...
pipelining_fn=pipeline_deepseek_v4
```

这一步的目的是让 DeepSeekV4 在开启 PP 时走专用切分逻辑，而不是继续走通用 LLM 切分逻辑。否则即使 `pp=2` 能启动，也可能出现以下问题：

```text
1. last stage 没有正确包含 hc_head。
2. hc_head_fn/hc_head_base/hc_head_scale 残留在所有 stage。
3. 非首 stage 拿不到真实 input_ids。
4. 用户只能靠 toml 手写 module_fqns_per_model_part，无法适配 pp=4/pp=8 等通用场景。
```

### 5.2 module fqn 生成规则

不要在 toml 中手写 `module_fqns_per_model_part`，因为这只适配某个固定 `pp` 配置。

通用规则应为：

1. 根据 `pipeline_parallel_degree` 和 schedule 计算 virtual stage 数。
2. 将 `layers.0 ... layers.N-1` 按 `n_layers` 均匀分到所有 virtual stages。
3. 第一个 virtual stage 固定包含：

```text
tok_embeddings
```

4. 最后一个 virtual stage 固定包含：

```text
hc_head
norm
output
```

例如 4 层、`pp=2 + Interleaved1F1B` 时可形成：

```text
stage0: tok_embeddings, layers.0
stage1: layers.1, layers.2
stage2: layers.3
stage3: hc_head, norm, output
```

即使最后 stage 没有主干 layer，也必须保留 `hc_head/norm/output`。

virtual stage 数的计算规则如下：

```python
def _get_num_virtual_stages(
    parallel_dims,
    job_config,
    num_layers,
    input_weight,
    output_weight,
):
    schedule_class = get_schedule_class(
        job_config.parallelism.pipeline_parallel_schedule
    )
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = job_config.parallelism.pipeline_parallel_layers_per_stage

    if layers_per_stage is None:
        stages_per_rank = 1 if is_single_stage_schedule else 2
        return parallel_dims.pp * stages_per_rank

    num_virtual_stages = math.ceil(
        (num_layers + input_weight + output_weight) / layers_per_stage
    )
    stages_per_rank = num_virtual_stages // parallel_dims.pp
    return num_virtual_stages
```

这里的 `stages_per_rank` 不是单独的 toml 配置项，而是在代码里推导出来的：

```text
single-stage schedule:
  stages_per_rank = 1

multi-stage schedule，例如 pipeline_parallel_schedule = "Interleaved1F1B":
  stages_per_rank = 2
```

如果 toml 中显式配置 `pipeline_parallel_layers_per_stage`，它表示“每个 virtual stage 期望放多少层”，不是 `stages_per_rank`。这种情况下会先根据层数计算 `num_virtual_stages`，再用：

```python
stages_per_rank = num_virtual_stages // parallel_dims.pp
```

反推出每个 PP rank 承载几个 virtual stages。

module fqn 的生成代码应体现 DeepSeekV4 的特殊尾部模块：

```python
def generate_deepseek_v4_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    output_modules = ["hc_head", "norm", "output"]

    if num_stages == 1:
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + output_modules]

    num_effective_layers = num_layers + input_weight + output_weight
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    module_names_per_stage = []
    current_layer = 0
    for stage_idx in range(num_stages):
        stage_modules = []
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        if stage_idx == 0:
            stage_modules.append("tok_embeddings")
            remaining_layers_for_stage = effective_layers_for_stage - input_weight
        elif stage_idx == num_stages - 1:
            remaining_layers_for_stage = effective_layers_for_stage - output_weight
        else:
            remaining_layers_for_stage = effective_layers_for_stage

        for _ in range(remaining_layers_for_stage):
            if current_layer < num_layers:
                stage_modules.append(f"layers.{current_layer}")
                current_layer += 1

        if stage_idx == num_stages - 1:
            stage_modules.extend(output_modules)

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage
```

上面的切分方式会把 `tok_embeddings` 视为 first_stage 的一个权重，`hc_head/norm/output` 视为 last_stage 的一个权重。以 16 层模型为例，当 pp=2，(16 + 1 + 1) // 4 = 4，extra = 2，那么前两个 stage 就会每一个 stage 多一个权重。stage0 包含 `tok_embeddings, layers.0, layers.1, layers.2, layers.3` ，stage1 包含 `layers.4, layers.5, layers.6, layers.7, layers.8` ，stage2 包含 `layers.9, layers.10, layers.11, layers.12` ，stage3 包含 `layers.13, layers.14, layers.15, hc_head, nrom, output` 。

这样做比在 toml 中手写 stage 列表更稳。手写方式只能解决某一个固定 `pp/schedule/n_layers` 场景；一旦 `pp=4`、schedule 变化、层数变化，手写列表就会失效。自动生成函数可以保证：

```text
tok_embeddings 只在 first stage
hc_head/norm/output 只在 last stage
layers.0 ... layers.N-1 全部出现且只出现一次
```

### 5.3 清理 hc_head 顶层参数

`_split_module` 通常只迭代 `model.named_children()`，不会自动删除顶层 `nn.Parameter`。

DeepSeekV4 的这些参数是顶层参数：

```text
hc_head_fn
hc_head_base
hc_head_scale
```

它们不是 child module，因此在切分后会残留在所有 model part 中。必须在 `_split_module` 之后、FSDP 包装之前显式删除非 last stage 的这些参数：

```python
if getattr(model, "hc_head", None) is None:
    for attr in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        if hasattr(model, attr):
            delattr(model, attr)
```

否则 FSDP 会把这些无用参数纳入分片和优化器状态，造成显存浪费，并可能引入错误同步。

这里清理的不是 last stage 的 `hc_head` 参数，而是非 last stage 上残留的无用副本。判断条件必须是：

```python
def _cleanup_unused_hc_head_parameters(model: nn.Module) -> None:
    if getattr(model, "hc_head", None) is not None:
        return

    for attr in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        if hasattr(model, attr):
            delattr(model, attr)
```

也就是说：

```text
last stage:
  model.hc_head is not None
  保留 hc_head_fn/hc_head_base/hc_head_scale

non-last stage:
  model.hc_head is None
  删除 hc_head_fn/hc_head_base/hc_head_scale
```

这样不会影响最后一层输出进入 `hc_head` 的计算。last stage 仍然会执行：

```python
h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
h = self.norm(h)
output = self.output(h.float())
```

`pp=1` 的正常路径不会进行这种清理，因为模型没有被切成多个 PP part，完整模型中 `hc_head` 和 `hc_head_*` 都应保留。只有 PP split 之后的非 last stage 才需要清理。

### 5.4 修改 forward 的 PP payload 协议

当前 `forward()` 的非首 stage 分支错误地从 hidden state 推导 `input_ids`：

```python
input_ids = tokens[:, :seq_len].detach().long()
h = tokens[:, :seq_len]
```

应改成：首 stage 生成真实 `input_ids`，并把它作为 sidecar payload 传给后续 stage（这里的sidecar指的是类似 input_ids/mtp_input_offsets这种不需要经过中间 transformer layer更新的数据）。

forward 协议应区分三类 stage：

```text
first stage:
  输入是真实 tokens，负责生成 h 和真实 input_ids sidecar。

middle stage:
  输入是上游传来的 h 和 input_ids sidecar，继续执行本 stage 的 layers，
  然后继续返回 h 和 input_ids sidecar。

last stage:
  输入是上游传来的 h 和 input_ids sidecar，执行本 stage 的 layers，
  然后执行 hc_head/norm/output，返回 logits。
```

建议新增 helper：

```python
def _normalize_pp_sidecar(self, input_ids: torch.Tensor | None) -> torch.Tensor:
    if input_ids is None or input_ids.ndim != 2:
        raise RuntimeError(
            "DeepSeekV4 PP stage requires input_ids sidecar with shape [B, S]. "
            "The first PP stage must return (hidden_states, input_ids), and "
            "later stages must forward that tuple unchanged."
        )
    return input_ids.detach().long()

def _pp_sidecar_for_send(
    self, input_ids_sidecar: torch.Tensor | None
) -> torch.Tensor:
    if input_ids_sidecar is None:
        raise RuntimeError("DeepSeekV4 PP non-last stage missing input_ids sidecar.")
    if input_ids_sidecar.dtype == torch.float32:
        return input_ids_sidecar
    return input_ids_sidecar.to(torch.float32)
```

首 stage：

```python
seq_len = tokens.shape[1] - self.model_args.num_mtp_modules
input_ids = tokens[:, :seq_len].detach().long()
input_ids_sidecar = input_ids.detach().to(torch.float32)
h = self.tok_embeddings(tokens[:, :seq_len])
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
```

非 last stage 返回时，`input_ids` 建议以 `float32` sidecar 形式传输：

说明：用 float32 不是为了数值计算，而是为了适配 PyTorch Pipeline 的 backward 通信机制。`input_ids` 原本是 token id，正常 dtype 是 `torch.long / int64`。但在 PP 中，非 last stage 如果返回 `return h, input_ids`，PyTorch Pipeline 会把 tuple 里的每个 tensor 都当成 stage output，并为它建立 forward 发送、backward 回传的通信元数据。到了下游 stage，收到的 activation buffer 通常会被设成 `requires_grad_(True)`。`torch.long` tensor 不能 `requires_grad=True`，所以直接传原始 long `input_ids` 可能在 pipeline runtime 建 backward buffer 或设置 requires grad 时出错；即使某些路径不马上报错，反向通信时也会遇到“非浮点 tensor 没有梯度”的问题。

当前不建议把 `input_ids` 直接作为 `torch.long` 的 pipeline output 传递。PyTorch PipelineStage 会把上游 stage 的 tuple output 统一当作 activation tensor，通过位置参数传给下游 stage，并在 backward 场景下为接收 buffer 设置 `requires_grad_(True)`、登记梯度回传信息。理论上可以把 `input_ids` 作为 kwargs 传给每个 stage，这样它不作为 pipeline activation，不需要 `requires_grad`，可以保持 `torch.long`。但 PyTorch 当前的跨 stage activation 传递只走位置参数，kwargs 不会从上游 stage output 自动传到下游 stage；kwargs 只能由 pipeline schedule 外部显式注入到每个 stage。因此该方案需要改调度层和数据分发逻辑，保证每个 PP rank 都能拿到当前 microbatch 的真实 `input_ids`。

所以当前 PP-only 修复选择更小改动的方案：首 stage 将 `input_ids` 转为 `float32` sidecar，随 activation payload 一起传递，后续 stage 使用前再转回 `long`。由于 DeepSeekV4 的 vocab id 范围远小于 float32 可精确表示的整数范围，该转换不会丢失 token id 信息。

```python
return h, input_ids_sidecar.to(torch.float32)
```

中间 stage 和 last stage 接收：

```python
def forward(self, tokens, input_ids=None, attention_masks=None, positions=None):
    if isinstance(tokens, tuple):
        h, input_ids_sidecar = tokens
        input_ids = self._normalize_pp_sidecar(input_ids_sidecar)
    elif self.tok_embeddings is None:
        h = tokens
        input_ids_sidecar = input_ids
        input_ids = self._normalize_pp_sidecar(input_ids_sidecar)
    else:
        ...
```

PyTorch PipelineStage 主路径会将 tuple output 拆成多个位置参数传给下一 stage，因此下一 stage 通常会收到：

```python
forward(h, input_ids, ...)
```

保留 `isinstance(tokens, tuple)` 作为防御性兼容路径即可。后续 stage 使用前再将 sidecar 转回 `long`：

```python
input_ids = input_ids_sidecar.detach().long()
```

这样避免 `torch.long` 出现在非 last stage 的 pipeline output tuple 中，降低 backward 阶段为非浮点 sidecar 建梯度通信缓冲时的 dtype 风险。`float32` 能精确表示当前 vocab 范围内的 token id。

注意 PyTorch `v2.10.0-rc2` 的 `PipelineStage` 会对所有从上游 stage 收到的 tensor 输入登记 backward send 目标，即使这个 tensor 在真实计算中只作为 sidecar 使用。也就是说，非首 stage 的 `input_ids_sidecar` 在 backward 时不能得到 `None` 梯度，否则会触发：

```text
has gradients None and is expecting to send gradients to stage ...
```

因此中间 stage 继续转发 sidecar 时不能 `detach()`；last stage 或任意消费 sidecar 但不把 sidecar 作为数值依赖的 stage，需要给 hidden state 增加零值依赖：

```python
h = h + input_ids_sidecar.to(dtype=h.dtype).sum() * 0.0
```

这不会改变前向数值，但会让 autograd 为 sidecar 生成合法的零梯度，满足 PyTorch 2.10rc2 的 pipeline backward 通信约束。

对应 helper：

```python
def _attach_pp_sidecar_grad(
    self, h: torch.Tensor, input_ids_sidecar: torch.Tensor | None
) -> torch.Tensor:
    if input_ids_sidecar is None or not input_ids_sidecar.is_floating_point():
        return h
    return h + input_ids_sidecar.to(device=h.device, dtype=h.dtype).sum() * 0.0
```

非首 stage 必须校验：

```python
if input_ids is None or input_ids.ndim != 2:
    raise RuntimeError("DeepSeekV4 PP stage requires input_ids sidecar with shape [B, S].")
```

### 5.5 last stage 输出路径

last stage 的判断可基于：

```python
is_last_stage = self.output is not None
```

last stage 必须执行：

```python
h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
h = self.norm(h)
logits = self.output(h.float())
return logits
```

并加清晰断言：

```python
if self.output is not None:
    if self.hc_head is None:
        raise RuntimeError("DeepSeekV4 PP last stage requires hc_head before norm/output.")
    for attr in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        if not hasattr(self, attr):
            raise RuntimeError(f"DeepSeekV4 PP last stage missing {attr}.")
```

这段断言的作用是尽早暴露切分错误。如果 last stage 没有 `hc_head`，模型仍可能继续跑到 `norm/output`，但语义已经错了；更糟的是错误可能表现为 loss 异常而不是明确 crash。因此 last stage 必须显式检查 `hc_head` module 和三个顶层参数都存在。

### 5.6 non-last stage 返回协议

如果当前 stage 没有 `output`，说明它不是最后 stage。

它跑完自己持有的主干层后必须返回：

```python
return h, self._pp_sidecar_for_send(input_ids_sidecar)
```

不能提前执行 `norm/output`，也不能把 `h` 单独返回，否则后续 stage 无法拿到真实 token ids。

完整 forward 主体建议如下：

```python
def forward(self, tokens, input_ids=None, attention_masks=None, positions=None):
    raw_tokens = None
    input_ids_sidecar = None

    if isinstance(tokens, tuple):
        h, input_ids_sidecar = tokens
        input_ids = self._normalize_pp_sidecar(input_ids_sidecar)
        seq_len = h.shape[1]
    elif self.tok_embeddings is not None:
        raw_tokens = tokens
        seq_len = tokens.shape[1] - self.model_args.num_mtp_modules
        input_ids = tokens[:, :seq_len].detach().long()
        input_ids_sidecar = input_ids.detach().to(torch.float32)
        h = self.tok_embeddings(tokens[:, :seq_len])
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
    else:
        h = tokens
        input_ids_sidecar = input_ids
        input_ids = self._normalize_pp_sidecar(input_ids_sidecar)
        seq_len = h.shape[1]

    for layer in self.layers.values():
        if layer.layer_id < self.model_args.n_layers:
            h = layer(
                h,
                input_ids,
                self.freqs_cis
                if self.model_args.compress_ratios[layer.layer_id] > 1
                else self.freqs_cis_wo_compressor,
                self.hadamard_mat,
                attention_masks,
            )

    if raw_tokens is None and input_ids_sidecar is not None:
        h = self._attach_pp_sidecar_grad(h, input_ids_sidecar)

    if self.output is None:
        return h, self._pp_sidecar_for_send(input_ids_sidecar)

    self._validate_last_stage_hc_head()
    h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
    h = self.norm(h)
    output = self.output(h.float())
    return output
```

这里不能继续使用旧逻辑中的本地 `layer_id = 0` 计数。PP split 后每个 stage 的 `self.layers` 只包含部分原始层，例如 stage1 可能包含 `layers.1/layers.2`。如果用本地计数，会把 stage 内的第一层误当成全局 layer0，从而选择错误的 `compress_ratios` 或提前 break。应使用每个 block 自带的全局 `layer.layer_id`。

### 5.6.1 stage 列表校验

DeepSeekV4 专用 PP 切分应在进入 `pipeline_module_split()` 前做强校验：

1. `len(module_fqns_per_model_part)` 必须等于计算出的 virtual stage 数。
2. `layers.0 ... layers.N-1` 必须恰好各出现一次。
3. layer 顺序必须全局递增。
4. 非 last stage 不允许包含 `hc_head/norm/output`。
5. `tok_embeddings` 只能出现在 first stage。
6. 用户手写配置时不允许出现空的中间 stage。

这样可以把错误定位在 PP 配置/切分阶段，而不是等到 FSDP、MoE 或 pipeline runtime 深处才报难读的错误。

建议校验逻辑至少包含：

```python
def _validate_deepseek_v4_stage_modules(
    module_names_per_stage: list[list[str]],
    num_layers: int,
    num_virtual_stages: int,
) -> None:
    if len(module_names_per_stage) != num_virtual_stages:
        raise ValueError(...)

    output_modules = ("hc_head", "norm", "output")
    seen_layers = []

    for stage_idx, modules in enumerate(module_names_per_stage[:-1]):
        for output_module in output_modules:
            if output_module in modules:
                raise ValueError(...)

    for stage_idx, modules in enumerate(module_names_per_stage[1:-1], start=1):
        if not modules:
            raise ValueError(...)

    first_stage = module_names_per_stage[0]
    last_stage = module_names_per_stage[-1]
    if "tok_embeddings" not in first_stage:
        raise ValueError(...)
    if any("tok_embeddings" in stage for stage in module_names_per_stage[1:]):
        raise ValueError(...)
    if [m for m in last_stage if m in output_modules] != list(output_modules):
        raise ValueError(...)

    for stage_idx, modules in enumerate(module_names_per_stage):
        stage_layers = []
        for module_name in modules:
            if module_name.startswith("layers."):
                layer_id = int(module_name.split(".", 1)[1])
                stage_layers.append(layer_id)
            elif module_name not in {"tok_embeddings", *output_modules}:
                raise ValueError(...)

        if stage_layers != sorted(stage_layers):
            raise ValueError(...)
        seen_layers.extend(stage_layers)

    expected_layers = list(range(num_layers))
    if seen_layers != expected_layers:
        raise ValueError(...)
```

### 5.7 TP+PP 下的 parallelize 保护

`apply_non_moe_tp()` 当前对这些模块和参数无条件处理：

```python
parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": ...,
        "norm": ...,
        "output": ...,
        "hc_head": ...,
    },
)
_register_distributed_parameter(model, "hc_head_fn", ...)
_register_distributed_parameter(model, "hc_head_base", ...)
_register_distributed_parameter(model, "hc_head_scale", ...)
```

PP 切分后，不同 stage 只保留一部分模块。因此应改成动态构造 plan：

```python
plan = {}

if getattr(model, "tok_embeddings", None) is not None:
    plan["tok_embeddings"] = RowwiseParallel(...)

if getattr(model, "norm", None) is not None:
    plan["norm"] = SequenceParallel()

if getattr(model, "output", None) is not None:
    plan["output"] = ColwiseParallel(...)

if getattr(model, "hc_head", None) is not None:
    plan["hc_head"] = hc_head_plan

if plan:
    parallelize_module(model, tp_mesh, plan)

if getattr(model, "hc_head", None) is not None:
    for attr in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        if hasattr(model, attr):
            _register_distributed_parameter(model, attr, tp_mesh, [Replicate()])
        else:
            raise RuntimeError(f"DeepSeekV4 last stage missing {attr}.")
```

这可以避免：

1. 非 last stage 注册不存在的 `hc_head_*`。
2. last stage 尝试 parallelize 不存在的 `tok_embeddings`。
3. TP+PP 组合下因为模块为 `None` 或不存在而崩溃。

当前 PP-only 修复阶段可以先显式拒绝 TP+PP：

```python
if parallel_dims.tp_enabled:
    raise NotImplementedError(
        "DeepSeekV4 PP+TP is not supported by the current PP-only fix."
    )
```

后续支持 TP+PP 时，再按上面的动态 plan 方式修改 `apply_non_moe_tp()`。

### 5.8 MTP+PP 当前处理策略

当前方案建议先显式限制：

```python
if pp_enabled and model_args.num_mtp_modules > 0:
    raise NotImplementedError("DeepSeekV4 MTP + PP is not supported yet.")
```

原因是 MTP+PP 还需要额外传递 `mtp_input_offsets` sidecar，其协议类似：

```text
(h, input_ids, mtp_input_offsets)
```

其中 `mtp_input_offsets` 应由首 stage 使用 `tok_embeddings` 生成，并随 pipeline payload 传到最后 stage。这个改造比当前 `num_mtp_modules=0` 的修复多一层状态传递，建议后续单独实现和验证。

## 6. sidecar 数据在 PP 中的生命周期

`input_ids`、未来的 `mtp_input_offsets` 这类 sidecar 数据不是预先保存在每个 rank 上的全局缓存。

它们应作为当前 microbatch 的 pipeline activation payload 的一部分，从首 stage 生成，然后沿 PP stage 逐段发送。

以 `pp=2 + Interleaved1F1B` 为例，实际有 4 个 virtual stages，可能映射为：

```text
rank0: stage0, stage2
rank1: stage1, stage3
```

当 stage0 生成：

```text
(h, input_ids)
```

后：

1. 若下一 stage 在不同 rank 上，PipelineStage 会通过 P2P 通信发送 `h` 和 `input_ids`。
2. 若下一 stage 在同一个 rank 上，则在本进程内传递，不需要跨 rank 通信。
3. 每个 stage 在处理该 microbatch 时临时持有 sidecar，并继续将它返回给下一 stage。
4. sidecar 生命周期跟随该 microbatch 的 forward/backward 调度，不应作为长期状态缓存在模块上。

因此 sidecar 的语义是：

```text
per-microbatch activation payload
```

不是：

```text
per-rank persistent cache
```

对于 `pp=4` 也是同样原则。区别只是 stage 到 rank 的映射变了，sidecar 仍然随 microbatch 从上游 stage 流向下游 stage。

## 7. DSA indexer loss 指标同步

PP 切分后，不是每个 PP rank 都一定拥有会产生 DSA indexer loss 的层。如果某些 rank 的
`DSAIndexerLossLoggingHelper.tracker` 中没有 `"values"` 就直接 `return`，这些 rank 会跳过
DSA loss 的 `all_reduce`，而其他 rank 仍在执行该 collective，最终可能和后续 barrier 的
collective 发生通信序列错位。

修复原则：

1. `apply_distributed_indexer_loss_tracking()` 接收 `n_layers`，通过闭包传给 static tracking 方法。
2. 所有 rank 都参与同形状 collective；没有本地 DSA loss 的 rank 贡献全零 tensor。
3. tracker 同时记录：

```text
values:  每层累计 loss
present: 本 rank 本 step 是否对该层产生过 DSA loss
```

4. 全局使用 `SUM` 聚合：

```python
dist.all_reduce(values, op=dist.ReduceOp.SUM)
dist.all_reduce(present, op=dist.ReduceOp.SUM)
```

5. 只对 `present > 0` 的层求平均：

```python
per_layer_loss = values[valid] / present[valid] / total_acc_steps
loss = per_layer_loss.mean()
```

这里的 `present` 是 layer/rank 级别的存在掩码，不按 microbatch 累加，因此仍需要除以
`total_acc_steps`。同时 DSV4 的 `layer_id` 是 0-based，保存 tracker 时应使用
`tracker["values"][layer_id]`，不能使用 `layer_id - 1`。

## 8. 验证计划

建议按以下顺序验证：

1. `deepseek_v4_285b_4layers_debug.toml`
   - `pipeline_parallel_degree = 2`
   - `num_mtp_modules = 0`
   - 期望不再出现 MoE gather shape mismatch。

2. `deepseek_v4_285b_43layers_4k_128die.toml`
   - `pipeline_parallel_degree = 2`
   - `num_mtp_modules = 0`
   - 期望最后 stage 日志包含 `hc_head/norm/output`。
   - 期望不再出现 `Expected target size [4096, 129280], got [4096]`。

3. TP+PP 组合验证
   - 确认非 last stage 不注册 `hc_head_*`。
   - 确认 last stage 不处理不存在的 `tok_embeddings`。

4. 后续单独验证 MTP+PP
   - 引入 `mtp_input_offsets` sidecar。
   - 验证最后 stage 返回 `[main_logits, mtp_logits, ...]`。
