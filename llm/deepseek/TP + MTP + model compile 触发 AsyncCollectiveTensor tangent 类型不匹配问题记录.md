# TP + MTP + model compile 触发 AsyncCollectiveTensor tangent 类型不匹配问题记录

日期：2026-05-22

## 结论摘要

DeepSeek-V4 在同时开启 Tensor Parallel、MTP 和 `torch.compile` 的 model 编译时，训练在第一步 backward 阶段报错：

```text
Expected type: torch.distributed._functional_collectives.AsyncCollectiveTensor
Runtime type: torch.Tensor
shape: torch.Size([1, 2048, 4096])
```

这个问题不是 OOM、HCCL 超时或 loss compile 本身导致的。直接触发点是 `torch.compile` 对模型子模块做 AOTAutograd 编译后，backward runtime tangent 的实际类型和 trace-time 预测类型不一致：AOTAutograd 预期收到 `AsyncCollectiveTensor`，实际收到的是普通 `torch.Tensor`。

本次采用的代码层修复是新增 `torchtitan_npu/patches/torch/functional_collectives.py`，给 `torch.Tensor` 和 `AsyncCollectiveTensor` 补齐 `__coerce_same_metadata_as_tangent__` 方向的兼容逻辑，并在 `torchtitan_npu/__init__.py` 的 `_apply_patches()` 中加载该 patch。

## 触发条件

问题配置组合：

- DeepSeek-V4。
- `tensor_parallel_degree = 2`，即开启 TP。
- `training.num_mtp_modules = 1`，即开启 MTP。
- `[compile] enable = true` 且 `components = ["model", "loss"]`，即同时编译 model 和 loss。
- `activation_checkpoint.mode = "full"`。
- 训练环境为 `torchtitan_cann_900_022dev`。

失败日志：

```text
/data/z00944403/torchtitan-npu/3x16_v022_deepseek_v4_tp/v022_deepseek_v4_43layers_tp_20260522154613.log
```

对照日志：

```text
/data/z00944403/torchtitan-npu/3x16_v022_deepseek_v4_tp/v022_deepseek_v4_43layers_tp_20260522163103.log
```

对照实验中，仅从 `compile.components` 中删除 `"model"`，保留 loss compile，训练能够跑完 10 step。这说明：

- loss compile 不是直接根因；
- MTP 本身不是直接根因；
- 关键触发组合是 model compile 与 TP/functional collective/activation checkpoint 的交界。

## 错误现象解读

失败日志中的首个失败 rank 是 rank 3，报错位置在 `loss.backward()`：

```text
torchtitan/train.py -> forward_backward_step -> loss.backward()
torch/_functorch/_aot_autograd/runtime_wrappers.py -> process_runtime_tangent()
```

核心错误为：

```text
Expected type: AsyncCollectiveTensor
Runtime type: torch.Tensor
```

这表示 AOTAutograd 在提前 trace backward graph 时，认为某个 backward tangent 应当保持 `AsyncCollectiveTensor` 这个 tensor subclass；但真正执行 backward 时，autograd 传入的是普通 `torch.Tensor`。AOTAutograd 在 `process_runtime_tangent()` 中会比较 expected type/meta 和 runtime type/meta；如果不一致，就尝试调用 tensor 上的 `__coerce_same_metadata_as_tangent__` 做兼容转换。当前 PyTorch 构建里：

- `AsyncCollectiveTensor` 已经有 `__coerce_same_metadata_as_tangent__`，可以处理 `AsyncCollectiveTensor -> torch.Tensor` 的方向；
- `torch.Tensor` 没有该方法，无法处理 `torch.Tensor -> AsyncCollectiveTensor` 的方向；
- 因此当 expected 是 `AsyncCollectiveTensor`、runtime 是 plain Tensor 时，AOTAutograd 无法转换并抛错。

错误信息里提示 `__force_to_same_metadata__`，但在 `torchtitan_cann_900_022dev` 里实际调用的方法名是 `__coerce_same_metadata_as_tangent__`。错误提示沿用了旧文案，不能按文案里的方法名实现。

## 为什么 shape 是 `[1, 2048, 4096]`

失败 shape 是：

```text
torch.Size([1, 2048, 4096])
```

当前训练中 `seq_len = 4096`，`tensor_parallel_degree = 2`。因此按 sequence 维切分后，每个 TP rank 看到的 sequence 长度是 `4096 / 2 = 2048`，hidden size 是 4096，所以这个 shape 与 TP/SequenceParallel 中间激活高度吻合。

这也进一步说明问题不像是 vocab logits 或 cross entropy loss 的输出，而更像是模型中间层 TP 边界处的激活/梯度。

## 相关代码路径

DeepSeek-V4 的 model compile 开关在 `torchtitan_npu/models/deepseek_v4/infra/parallelize.py`：

```python
model_compile_enabled = (
    job_config.compile.enable and "model" in job_config.compile.components
)

if model_compile_enabled:
    apply_compile(model, job_config.compile, parallel_dims.ep_enabled)
```

`apply_compile()` 对 MoE block 中的多个子模块做 `torch.compile`。对于 Attention，代码跳过 `inner_attention`，但会编译 `pre_attention` 和 `post_attention`：

```python
elif isinstance(submod, Attention):
    attention = submod
    for attr_name, submod in attention.named_children():
        if attr_name == "inner_attention":
            continue
        setattr(
            attention,
            attr_name,
            torch.compile(submod, backend=compile_config.backend, fullgraph=True),
        )
```

Attention 的 forward 结构是：

```python
q, kv, kv_compress, q_indexer, k_indexer, weights = self.pre_attention(...)
o, compress_topk_idxs, index_score = self.inner_attention(...)
x = self.post_attention(...)
return x
```

其中 `post_attention` 的最后一行是：

```python
return self.wo_b(o.reshape(bsz, seqlen, -1))
```

TP parallelize plan 中 `attention.post_attention.wo_b` 是 RowwiseParallel，并且明确配置为 `use_local_output=False`：

```python
"attention.post_attention.wo_b": rowwise_parallel(
    input_layouts=Shard(-1),
    output_layouts=Shard(1),
    use_local_output=False,
),
```

这意味着 `wo_b` 的输出会保留 DTensor/functional collective 语义，而不是立即转成普通本地 Tensor。后续 `hc_post` 的 parallelize plan 又会消费 `Shard(1)` 输入，并把它转为本地输入参与计算。

在 eager 或未编译 model 时，这个边界由普通 autograd 路径处理；开启 model compile 后，AOTAutograd 会提前 trace backward，并记录 forward 输出/tangent 的 tensor subclass 信息。一旦运行时某个 tangent 被 materialize 成 plain Tensor，就会触发 expected/runtime 类型不一致。

## 为什么删除 `model` compile 可以跑通

对照日志 `v022_deepseek_v4_43layers_tp_20260522163103.log` 中仍然存在：

```text
Compiling the loss function with torch.compile
```

但没有：

```text
Compiling each TransformerBlock with torch.compile
```

并且最终 `Training completed`。这说明 `loss` 编译路径没有触发这类 functional collective tangent mismatch；问题来自 model 子模块编译后，AOTAutograd 将 TP 边界中的 `AsyncCollectiveTensor` 作为 compiled graph 的 tangent 类型进行预测。

因此临时 workaround 可以是：

```toml
[compile]
enable = true
components = ["loss"]
```

或者彻底关闭 compile：

```toml
[compile]
enable = false
components = ["model", "loss"]
```

但这只是绕过 model compile，并没有修复底层兼容问题。

## 与 pytorch/pytorch#172556 的关系

Issue 链接：

```text
https://github.com/pytorch/pytorch/issues/172556
```

该 issue 标题是：

```text
Forgetting to detach() a DTensor graph output that does not need computed gradients leads to bad error message
```

issue 中的最小复现是一个 `torch.compile(backend="aot_eager")` 函数返回两个 DTensor 输出：

```python
return x.sin(), x.cos()
```

调用方只对第一个输出做 backward：

```python
out1, out2 = f(dt1)
out1.sum().backward()
```

第二个输出虽然不参与最终 loss，但它仍然是 requires-grad 的 graph output。作者解释了这类问题的机制：

1. `torch.compile` 会提前 trace backward graph，而不是等用户真正调用 `.backward()` 时才构造 backward。
2. 为了 trace backward，AOTAutograd 必须为 forward outputs 构造对应的 backward inputs/tangents。
3. 如果 forward output 是 DTensor，AOTAutograd 会假设其 backward tangent 也是 DTensor。
4. 但运行时如果用户只对部分输出做 backward，autograd 会把 unused output 的梯度替换为普通 zero Tensor。
5. 于是 trace-time 预期为 DTensor，runtime 却是 plain Tensor，导致 metadata/type mismatch。

issue 中给出的两个 workaround 是：

- 对不需要梯度的 graph output 显式 `detach()`，这也能降低无用 activation 保留带来的内存浪费；
- 如果可以接受，把 fwd-loss-bwd 编译成一个整体 graph，这样 backward 输入不再需要靠 AOTAutograd 对 graph outputs 进行猜测。

该 issue 当前状态为 closed，定位重点是错误提示不够友好，但它说明了根机制：AOTAutograd 对 backward tangent metadata 的提前预测可能在运行时失效。

与本问题的对应关系：

- #172556 中 expected 是 `DTensor`，runtime 是 `torch.Tensor`；
- 本问题中 expected 是 `AsyncCollectiveTensor`，runtime 是 `torch.Tensor`；
- 二者都是 `torch.compile` + tensor subclass + AOTAutograd backward tangent 预测失效；
- 区别在于本问题触发的 subclass 不是 DTensor 本体，而是 functional collective eager wrapper `AsyncCollectiveTensor`。

## 与 pytorch/pytorch#173123 的关系

Issue 链接：

```text
https://github.com/pytorch/pytorch/issues/173123
```

该 issue 标题是：

```text
[DTensor] Gradient of unused output becomes plain torch.Tensor in backward, causing type mismatch with torch.compile
```

issue 中描述的现象是：一个包含 DTensor 输入的 op 返回多个 DTensor 输出，如果其中一个输出没有被下游计算使用，那么这个 unused output 在 backward 中对应的 gradient 会变成普通 `torch.Tensor` zero，而不是 DTensor。配合 `torch.compile` 后，AOTAutograd 仍然预期该 tangent 是 DTensor，于是报类型不匹配。

issue 报错形态是：

```text
Expected type: torch.distributed.tensor.DTensor
Runtime type: torch.Tensor
```

这与本问题非常接近。本问题的报错形态是：

```text
Expected type: torch.distributed._functional_collectives.AsyncCollectiveTensor
Runtime type: torch.Tensor
```

相同点：

- 都发生在 `torch.compile` 的 AOTAutograd backward prologue；
- 都是 expected tensor subclass 与 runtime plain Tensor 不一致；
- 都与 distributed tensor/collective 相关；
- 都不是数值溢出、通信 hang 或普通 autograd 算子错误。

不同点：

- #173123 聚焦 DTensor 多输出 unused gradient 被 plain zero Tensor 替换；
- 本问题聚焦 functional collective 的 `AsyncCollectiveTensor` wrapper 在 backward tangent 方向与 plain Tensor 的等价性没有被当前 PyTorch 构建识别。

因此 #173123 可以作为同类问题的直接参考：它证明了 `torch.compile` 下 distributed tensor subclass 的 unused/converted tangent 可能变成 plain Tensor，并且这类情况需要由 tensor subclass 或 AOTAutograd coercion 逻辑处理。

## 为什么本次选择 patch AsyncCollectiveTensor coercion

可选修复路径有几类：

1. 配置层绕过：删除 `compile.components` 中的 `"model"`。
2. 编译粒度收窄：保留 model compile，但跳过 `post_attention` 或其它 TP 边界模块。
3. 改 TP layout：修改 `use_local_output`、显式 `wait_tensor()` 或调整 DTensor/local Tensor 边界。
4. 补齐 AOTAutograd tangent coercion：让 plain Tensor 在 expected type 为 `AsyncCollectiveTensor` 时被接受。

第 1 类能跑通但牺牲 model compile，属于 workaround。

第 2 类可能可行，但需要逐个定位所有危险子模块，且后续模型结构变化后仍可能再次踩中。

第 3 类风险较高，因为 `parallelize.py` 中已有注释说明某些 `use_local_output=False` 是为了让中间结果保持 DTensor，使 autograd 正确处理梯度。直接改 TP layout 可能影响分布式语义和数值正确性。

第 4 类最贴近根因：`AsyncCollectiveTensor` 只是 functional collective 的异步等待包装，本身没有额外 metadata。当 backward runtime tangent 已经是 plain Tensor 时，这个 plain Tensor 是合法梯度，不需要强制重新包装为 `AsyncCollectiveTensor`。因此允许 `torch.Tensor -> AsyncCollectiveTensor expected tangent` 通过，是更小且更准确的兼容修复。

## 本次代码修改

新增文件：

```text
torchtitan_npu/patches/torch/functional_collectives.py
```

核心逻辑：

```python
_HOOK = "__coerce_same_metadata_as_tangent__"


def _tensor_coerce_same_metadata_as_tangent(
    self: torch.Tensor,
    expected_metadata: Any,
    expected_type: type | None = None,
):
    if expected_type is AsyncCollectiveTensor and expected_metadata is None:
        return self
    return None


def _async_collective_tensor_coerce_same_metadata_as_tangent(
    self: AsyncCollectiveTensor,
    expected_metadata: Any,
    expected_type: type | None = None,
):
    if expected_type is not torch.Tensor:
        return None
    return self.trigger_wait()
```

并在 `torchtitan_npu/__init__.py` 的 `_apply_patches()` 中加载：

```python
from .patches.torch import (  # noqa: F401
    checkpoint,
    clip_grad,
    functional_collectives,
    micro_pipeline_tp,
    pipelining,
)
```

## patch 的安全边界

该 patch 有几个刻意收窄的边界：

1. `torch.Tensor` 方向只在 `expected_type is AsyncCollectiveTensor` 时生效。

   这意味着如果 expected type 是 DTensor 或其它 tensor subclass，仍然返回 `None`，保留 PyTorch 原有失败路径，不会吞掉其它真实错误。

2. `torch.Tensor` 方向还要求 `expected_metadata is None`。

   当前 `AsyncCollectiveTensor` 没有额外 metadata；如果未来 PyTorch 给它添加了真实 metadata，这个 patch 不会盲目放行。

3. `AsyncCollectiveTensor -> torch.Tensor` 方向使用 `trigger_wait()`。

   这是 `AsyncCollectiveTensor` 自身提供的等待并返回底层 tensor 的安全方法，语义上等价于等待异步 collective 完成后拿到普通 tensor。

4. 使用 `__dict__` 做守卫，而不是 `hasattr()`。

   `AsyncCollectiveTensor` 继承自 `torch.Tensor`。如果先给 `torch.Tensor` 挂 hook，再用 `hasattr(AsyncCollectiveTensor, hook)` 判断，会因为继承关系误以为 ACT 自己已有实现，导致 ACT 自身方向的 hook 没有安装。因此代码使用：

   ```python
   if _HOOK not in AsyncCollectiveTensor.__dict__:
       ...

   if _HOOK not in torch.Tensor.__dict__:
       ...
   ```

5. 未来 PyTorch 版本如果已经提供同名实现，本 patch 不会覆盖。

## 为什么不直接改 `use_local_output`

`attention.post_attention.wo_b` 当前配置为：

```python
use_local_output=False
```

从表面看，把它改成 local output 可能让 compiled graph 少见到 `AsyncCollectiveTensor`，但这会改变 TP/DTensor 输出语义。`parallelize.py` 中相关注释已经说明，保留 DTensor 是为了让 autograd 正确处理中间结果的梯度。

因此直接修改 `use_local_output` 更像是绕开症状，且可能引入分布式梯度错误。本次 patch 不改变模型图、TP layout、通信模式和 loss 计算，只补齐 AOTAutograd 对 functional collective tangent 的类型兼容。

## 验证建议

最直接的验证方式：

1. 在 `torchtitan_cann_900_022dev` 环境中恢复触发配置：

   ```toml
   [compile]
   enable = true
   components = ["model", "loss"]
   ```

2. 保持 TP + MTP：

   ```toml
   tensor_parallel_degree = 2
   num_mtp_modules = 1
   ```

3. 重新运行 `deepseek_v4_285b_4layers_debug.toml` 或等价 debug 配置。

4. 观察是否还出现：

   ```text
   Expected type: AsyncCollectiveTensor
   Runtime type: torch.Tensor
   ```

5. 若能跑完 10 step，则说明本 patch 覆盖了当前 `AsyncCollectiveTensor` tangent mismatch。

辅助验证：

```bash
python3 -m compileall torchtitan_npu/patches/torch/functional_collectives.py torchtitan_npu/__init__.py
```

当前本地已通过语法检查。由于本地 Python 环境没有安装 `torch`，无法在本地直接跑相关 pytest；需要在带 torch/torch_npu 的训练容器内做运行验证。

## 后续维护建议

1. 升级 PyTorch 后检查 `torch.Tensor` 和 `AsyncCollectiveTensor` 是否已经内置 `__coerce_same_metadata_as_tangent__`。

   如果上游已经覆盖两个方向，可以移除本地 monkey patch。

2. 如果以后出现 `Expected type: DTensor / Runtime type: torch.Tensor`，本 patch 不会解决。

   那是 DTensor 自身的 tangent coercion 问题，应参考 #172556/#173123 的 DTensor 路径处理，比如 detach unused graph output，或升级到包含上游修复的 PyTorch。

3. 如果某次报错中 `Expected metadata` 不再是 `None`，不要直接放宽本 patch。

   需要先确认新的 metadata 代表什么，以及 plain Tensor 是否仍然能语义等价。

4. 如果继续追求 model compile 性能，可以在本 patch 验证稳定后，再比较：

   - `components = ["loss"]`
   - `components = ["model", "loss"]`

   重点观察 step time、显存、loss/grad_norm 是否符合预期。

## 参考链接

- PyTorch issue #172556: https://github.com/pytorch/pytorch/issues/172556
- PyTorch issue #173123: https://github.com/pytorch/pytorch/issues/173123

# MTP + TP + Compile 场景下的 wait() 修复方案

## 本次遇到的报错

触发条件：

- DeepSeek-V4
- `tp=2`
- `compile.enable=true`
- `compile.components` 包含 `"model"`
- `num_mtp_modules=1`

旧日志中的核心报错为：

```text
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.

Expected metadata: None, expected type: <class 'torch.distributed._functional_collectives.AsyncCollectiveTensor'>

Runtime metadata: None, runtime type: <class 'torch.Tensor'>

shape: torch.Size([1, 2048, 4096])
```

这个报错发生在 AOTAutograd backward 的 `process_runtime_tangent` 阶段。含义是：编译阶段记录某个 backward tangent 应该是 `AsyncCollectiveTensor`，但运行时实际拿到的是普通 `torch.Tensor`。二者 metadata 都是 `None`，真正不一致的是 tensor subclass 类型。

## 报错位置对应的张量

对比 MTP=0 和 MTP=1 的 debug 日志后，最关键的差异出现在 MTP 额外层 `layers.43`：

```text
layers.43.enorm fwd_in
```

在使用 `_functional_collectives.py` patch 的旧 MTP=1 日志中：

```text
[MTP_TP_COMPILE_DEBUG][fwd_in][layers.43.enorm]
type=torch.distributed._functional_collectives.AsyncCollectiveTensor
shape=(1, 512, 4096)
```

在使用 `wait()` 方案的新 MTP=1 日志中：

```text
[MTP_TP_COMPILE_DEBUG][fwd_in][layers.43.enorm]
type=torch.Tensor
shape=(1, 512, 4096)
```

旧报错中的 shape 是 `[1, 2048, 4096]`，新 debug 配置中是 `[1, 512, 4096]`。两者结构相同，都是 TP=2 后本地 sequence shard 的 hidden tensor，差异来自不同运行配置的 sequence length。

## 为什么 MTP=0 没有触发

MTP=0 时，主干路径在 `tok_embeddings` 后继续进入普通 transformer 主干。日志中没有发现 `AsyncCollectiveTensor` 直接进入 MTP 专属 `enorm` 的情况。

MTP=1 时，模型 forward 多了这一段：

```python
token_offset = tokens[:, token_offset_id:token_end_idx]
input_offset = self.tok_embeddings(token_offset)
h = self.layers[str(layer_id)](
    input_offset,
    prev_embed,
    ...
)
```

其中 `self.layers[str(layer_id)]` 是 MTPModule，第一步就是：

```python
input_offset = self.enorm(input_offset)
```

`tok_embeddings` 在 TP 下使用 RowwiseParallel，输出 layout 是 `Shard(1)`。在 MTP=1 + model compile 场景中，这个输出的本地 shard 可能以 `AsyncCollectiveTensor` 形式进入 `layers.43.enorm` 的 compiled boundary，于是 AOTAutograd 编译期和运行期对 tangent 类型的判断可能不一致。

## 两种修复思路

### 方案一：patch functional_collectives

这个方案在 `torchtitan_npu/patches/torch/functional_collectives.py` 中补齐 `torch.Tensor` 和 `AsyncCollectiveTensor` 的 `__coerce_same_metadata_as_tangent__` 兼容逻辑。

它解决的是：

```text
expected=AsyncCollectiveTensor, runtime=torch.Tensor
```

这种类型不一致出现后，AOTAutograd 如何把二者视为等价 tangent。

优点：

- 直接命中报错本身。
- 不改变模型 forward 中 collective 的等待时机。
- 对已有异步通信 overlap 更友好。

缺点：

- 属于 monkey patch PyTorch tensor subclass 行为。
- 需要持续关注后续 PyTorch 版本是否已经内置该逻辑。

### 方案二：在 MTP + compile 场景提前 wait

这个方案不等报错发生后再做 coercion，而是在 MTP 分支进入 compiled `enorm` 前，让 `tok_embeddings` 的 Rowwise 输出稳定成普通 `torch.Tensor`。

本次修改位置在 `torchtitan_npu/models/deepseek_v4/infra/parallelize.py`。

新增 helper：

```python
_ASYNC_COLLECTIVE_TENSOR_MODULE = "torch.distributed._functional_collectives"
_ASYNC_COLLECTIVE_TENSOR_NAME = "AsyncCollectiveTensor"


def _is_async_collective_tensor(value) -> bool:
    value_type = type(value)
    return (
        value_type.__module__ == _ASYNC_COLLECTIVE_TENSOR_MODULE
        and value_type.__name__ == _ASYNC_COLLECTIVE_TENSOR_NAME
    )


def _wait_async_collective_tensor(value):
    if _is_async_collective_tensor(value):
        return value.wait()
    return value
```

修改 `AwaitRowwiseParallel` 的输出处理：

```python
class AwaitRowwiseParallel(RowwiseParallel):
    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)

        if use_local_output:
            return _wait_async_collective_tensor(outputs.to_local())

        real_tensor = outputs._local_tensor
        if _is_async_collective_tensor(real_tensor):
            real_tensor.wait()
        else:
            torch.ops._c10d_functional.wait_tensor(real_tensor)
        return outputs
```

仅在 MTP + model compile 时启用：

```python
model_compile_enabled = (
    job_config.compile.enable and "model" in job_config.compile.components
)
await_mtp_embedding_output = (
    model_compile_enabled and model.model_args.num_mtp_modules > 0
)
embedding_rowwise_parallel = (
    await_rowwise_parallel if await_mtp_embedding_output else rowwise_parallel
)
if await_mtp_embedding_output:
    logger.info(
        "Waiting tok_embeddings RowwiseParallel output for DeepSeek-V4 MTP compile"
    )
```

然后将 `tok_embeddings` 从固定 `RowwiseParallel` 改为条件选择：

```python
"tok_embeddings": embedding_rowwise_parallel(
    input_layouts=Replicate(),
    output_layouts=Shard(1),
),
```

这样修改后，只有以下场景会走 wait 版本：

```text
compile.enable == true
and "model" in compile.components
and num_mtp_modules > 0
```

MTP=0、不开 compile、只 compile loss 的场景不受影响。

## 为什么使用 wait() 而不是 .elem

`AsyncCollectiveTensor` 是 functional collective 返回的 wrapper subclass。它内部有底层 tensor：

```python
AsyncCollectiveTensor.elem
```

直接使用 `.elem` 确实可以拿到普通 `torch.Tensor`，但它不是推荐修复方式。

风险：

1. `.elem` 只是剥掉 wrapper，不保证显式执行 `wait_tensor`。
2. 它绕过了 `AsyncCollectiveTensor.__torch_dispatch__` 中的正常等待逻辑。
3. `.elem` 是 PyTorch 内部实现细节，不是稳定 API。
4. 日志中可以看到 ACT 外层 `requires_grad=True`，但 `elem` 显示 `requires_grad=False`，直接取 `.elem` 更容易引入 autograd 语义风险。

`wait()` 是该 wrapper 设计出来的 materialize 方式：

```python
def wait(self) -> torch.Tensor:
    return wait_tensor(self.elem)
```

因此更安全的语义是：

```text
AsyncCollectiveTensor
-> wait async collective 完成
-> 返回普通 torch.Tensor
```

而不是：

```text
AsyncCollectiveTensor
-> 直接取内部 elem
-> 绕过 wrapper 等待语义
```

## 两份日志的对比结论

对比日志：

- `_functional_collectives.py` patch 方案：
  `5x16_v022_deepseek_v4_tp_test/v022_deepseek_v4_43layers_tp2_mtp1_af_20260525143021.log`
- `wait()` 方案：
  `5x16_v022_deepseek_v4_tp_test/v022_deepseek_v4_43layers_tp2_mtp1_af_20260525153229.log`

关键信息：

```text
143021.log: AsyncCollectiveTensor 出现 8 次
153229.log: AsyncCollectiveTensor 出现 0 次
```

`layers.43.enorm fwd_in` 对比：

```text
143021.log: type=torch.distributed._functional_collectives.AsyncCollectiveTensor
153229.log: type=torch.Tensor
```

两份日志都训练完成：

```text
Training completed
```

step 10 指标接近：

```text
143021.log loss:      13.37783
153229.log loss:      13.37789

143021.log grad_norm:  8.5247
153229.log grad_norm:  8.5259
```

稳定 step 的性能没有观察到明显变慢。step 2-10 粗略平均：

```text
143021.log: 约 8.95s/step
153229.log: 约 8.73s/step
```

注意：`153229.log` 的 step 1 明显更慢：

```text
step 1 elapsed_time_per_step: 450.259s
```

这更像首次 compile/CANN 编译缓存差异，不应直接归因于 `wait()`。

## 最终建议

当前更推荐保留 `wait()` 方案作为 DeepSeek-V4 MTP + TP + model compile 的局部修复：

1. 修改范围收窄在 `tok_embeddings` 的 Rowwise 输出。
2. 只在 MTP + model compile 场景生效。
3. 不改变 MTP 计算公式、不改变 loss 逻辑、不改变下游 DTensor layout。
4. 从 debug 日志看，能把 `layers.43.enorm` 入口从 ACT 稳定为普通 Tensor。
5. 从短跑日志看，loss/grad_norm 对齐，稳定 step 性能没有明显退化。

如果后续需要追求最大通信 overlap，可以继续评估 `_functional_collectives.py` patch 方案；如果追求更少 monkey patch 和更明确的局部行为，`wait()` 方案更容易维护。
