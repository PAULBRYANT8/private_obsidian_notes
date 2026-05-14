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

新增文件建议：

```text
torchtitan_npu/models/deepseek_v4/infra/pipeline_parallel.py
```

新增函数建议：

```python
pipeline_deepseek_v4(...)
generate_deepseek_v4_fqn_per_model_part(...)
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

### 5.4 修改 forward 的 PP payload 协议

当前 `forward()` 的非首 stage 分支错误地从 hidden state 推导 `input_ids`：

```python
input_ids = tokens[:, :seq_len].detach().long()
h = tokens[:, :seq_len]
```

应改成：首 stage 生成真实 `input_ids`，并把它作为 sidecar payload 传给后续 stage。

首 stage：

```python
seq_len = tokens.shape[1]
input_ids = tokens[:, :seq_len].detach().long()
h = self.tok_embeddings(input_ids)
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
```

非 last stage 返回时，`input_ids` 建议以 `float32` sidecar 形式传输：

```python
return h, input_ids_sidecar.to(torch.float32)
```

中间 stage 和 last stage 接收：

```python
def forward(self, tokens, input_ids=None, attention_masks=None, positions=None):
    if isinstance(tokens, tuple):
        h, input_ids = tokens
    elif self.tok_embeddings is None:
        h = tokens
    else:
        ...
```

PyTorch PipelineStage 主路径会将 tuple output 拆成多个位置参数传给下一 stage，因此下一 stage 通常会收到：

```python
forward(h, input_ids, ...)
```

保留 `isinstance(tokens, tuple)` 作为防御性兼容路径即可。后续 stage 使用前再将 sidecar 转回 `long`：

```python
input_ids = input_ids.detach().long()
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

### 5.6 non-last stage 返回协议

如果当前 stage 没有 `output`，说明它不是最后 stage。

它跑完自己持有的主干层后必须返回：

```python
return h, input_ids
```

不能提前执行 `norm/output`，也不能把 `h` 单独返回，否则后续 stage 无法拿到真实 token ids。

### 5.6.1 stage 列表校验

DeepSeekV4 专用 PP 切分应在进入 `pipeline_module_split()` 前做强校验：

1. `len(module_fqns_per_model_part)` 必须等于计算出的 virtual stage 数。
2. `layers.0 ... layers.N-1` 必须恰好各出现一次。
3. layer 顺序必须全局递增。
4. 非 last stage 不允许包含 `hc_head/norm/output`。
5. `tok_embeddings` 只能出现在 first stage。
6. 用户手写配置时不允许出现空的中间 stage。

这样可以把错误定位在 PP 配置/切分阶段，而不是等到 FSDP、MoE 或 pipeline runtime 深处才报难读的错误。

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
