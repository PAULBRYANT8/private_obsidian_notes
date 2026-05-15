# DeepSeekV4 PP + MTP 报错分析与修复方案

记录时间：2026-05-13

本文整理 `deepseek_v4_285b_43layers_4k_128die.toml` 在开启 Pipeline Parallel(PP) 和 MTP 后遇到的几类报错，包括日志现象、根因分析、当前修改是否有效，以及最终建议的完整修复方案。

## 1. 当前背景

当前模型配置的关键参数如下：

```toml
n_layers = 43
num_mtp_modules = 1
pipeline_parallel_degree = 2
pipeline_parallel_schedule = "Interleaved1F1B"
expert_parallel_degree = 16

[compile]
enable = false
```

DeepSeekV4 模型结构上有两类 layer：

```text
layers.0  - layers.42  # 43 个主干 TransformerBlock
layers.43              # 1 个 MTPModule
```

也就是说：

```python
total_layer_like_modules = n_layers + num_mtp_modules
                         = 43 + 1
                         = 44
```

但是当前通用 PP 切分逻辑只按 `n_layers = 43` 切分，并没有把 `layers.43` 这个 MTP layer 算进去。

从当前 `log/error.log` 可以看到 PP stage 切分结果：

```text
PP rank 1 is building stage_idx 1 with modules
['layers.11', ..., 'layers.21']

PP rank 1 is building stage_idx 3 with modules
['layers.33', ..., 'layers.42', 'norm', 'output']
```

这里最后 stage 只有：

```text
layers.33 - layers.42, norm, output
```

缺了两个关键部分：

```text
layers.43  # MTPModule
hc_head    # DeepSeekV4 的 HC 输出头
```

这就是后续错误的根源。

## 2. 报错一：KeyError: '43'

### 2.1 报错现象

之前开启 compile/autofuse 时，模型在 `DeepSeekV4Model.forward()` 的 MTP 分支中报：

```text
KeyError: '43'
```

触发位置逻辑类似：

```python
layer_id = mtp_layer_id + self.model_args.n_layers
h = self.layers[str(layer_id)](...)
```

当：

```python
self.model_args.n_layers = 43
mtp_layer_id = 0
```

就会访问：

```python
self.layers["43"]
```

### 2.2 为什么会报这个错

完整模型初始化时确实创建了 `layers.43`：

```python
for layer_id in range(model_args.n_layers + model_args.num_mtp_modules):
    if layer_id < model_args.n_layers:
        self.layers[str(layer_id)] = DeepSeekV4TransformerBlock(...)
    else:
        self.layers[str(layer_id)] = MTPModule(...)
```

但是 PP 切分后，每个 stage 只保留自己负责的模块。

当前通用切分逻辑只认识：

```text
tok_embeddings
layers.0 ... layers.42
norm
output
```

它不知道 `num_mtp_modules`，所以最后 stage 里没有 `layers.43`。

于是最后 stage 执行 MTP 分支时：

```python
self.layers["43"]
```

实际不存在，触发 `KeyError: '43'`。

### 2.3 当前临时修改是否解决了它

当前 `model.py` 中加过类似保护：

```python
if (
    str(self.model_args.n_layers) not in self.layers
    or self.tok_embeddings is None
):
    return output
```

这个修改可以绕开 `KeyError: '43'`，但没有真正解决 PP + MTP。

原因是：它只是让缺失 MTP layer 的 stage 直接返回普通 Tensor，等价于在 PP 下跳过了 MTP。可是训练配置里仍然认为 MTP 是开启的，loss 仍然会调用 MTP loss。

因此这个修改只是把错误从 `KeyError: '43'` 推迟到了 loss 阶段。

## 3. 报错二：Expected input batch_size (16372) to match target batch_size (4)

### 3.1 报错现象

将 compile 关闭后，当前 `log/error.log` 中报：

```text
ValueError: Expected input batch_size (16372) to match target batch_size (4).
```

堆栈关键位置：

```text
torch/distributed/pipelining/schedules.py
  _maybe_compute_loss(...)

torchtitan_npu/patches/torchtitan/loss.py
  main_loss = cross_entropy_loss(preds[0], labels[:, :seq_len])

torchtitan/components/loss.py
  torch.nn.functional.cross_entropy(...)
```

PP runtime 日志指出错误发生在：

```text
Step 06: 2SEND_F0     3F0          <-- ERROR HERE
```

`3F0` 表示 stage 3 的 forward，也就是最后 stage 在 forward 后计算 loss 时失败。

### 3.2 直接原因

MTP loss 的输入约定是：

```python
preds = [
    main_logits,  # [B, S, vocab]
    mtp_logits,   # [B, S, vocab]
]
```

loss 中会这样取主 loss：

```python
seq_len = preds[0].shape[1]
main_loss = cross_entropy_loss(preds[0], labels[:, :seq_len])
```

但当前 PP 下最后 stage 实际返回的是普通 Tensor，不是 list：

```python
output  # Tensor
```

于是：

```python
preds[0]
```

并不是取 `main_logits`，而是对 Tensor 取第 0 个 batch 样本。

这会导致 loss 对 shape 的理解完全错位。

### 3.3 为什么会出现 16372 和 4

当前最后 stage 缺少 `hc_head`，所以输出很可能仍然保留 HC 维度：

```text
正常期望:
  [B, S, vocab]

实际可能:
  [B, S, hc_mult, vocab]
```

其中：

```text
hc_mult = 4
```

同时，当前 `DeepSeekV4Model.forward()` 在每个 stage 都执行：

```python
seq_len = tokens.shape[1]
seq_len -= self.model_args.num_mtp_modules
```

这在 first stage 是对的，因为 first stage 收到原始 token ids，比如长度是 4097：

```text
4097 - 1 = 4096
```

但 middle/last stage 收到的是已经裁剪过的 activation，不是原始 token ids。每个 stage 再裁一次会让序列长度不断变短：

```text
stage0: 4097 -> 4096
stage1: 4096 -> 4095
stage2: 4095 -> 4094
stage3: 4094 -> 4093
```

因此最后 stage 的 Tensor 可能形如：

```text
[B, 4093, 4, vocab]
```

loss 中执行：

```python
preds[0]
```

后变成：

```text
[4093, 4, vocab]
```

于是：

```python
seq_len = preds[0].shape[1] = 4
```

target 变成：

```python
labels[:, :4]
```

也就是只有 4 个 label。

而 `cross_entropy` 会把输入的前面维度展平：

```text
input batch = 4093 * 4 = 16372
target batch = 4
```

所以报：

```text
Expected input batch_size (16372) to match target batch_size (4)
```

### 3.4 深层原因

这个错误不是单纯的 loss bug，而是多个问题叠加：

1. MTP 已开启，但最后 stage 没有 `layers.43`，所以 MTP 没有真正参与训练。
2. 当前保护逻辑让模型返回普通 Tensor，而 MTP loss 期望 list。
3. 最后 stage 缺少 `hc_head`，导致输出 shape 保留 `hc_mult` 维度。
4. 每个 stage 都重复执行 `seq_len -= num_mtp_modules`，导致序列长度被反复裁短。
5. non-first stage 把 float activation 强转成 long 当 `input_ids`，会污染 MoE routing。

## 4. 当前修改的有效性判断

### 4.1 `layer.layer_id < n_layers` 的修改

当前将主干 layer 循环从本地计数：

```python
layer_id = 0
for layer in self.layers.values():
    if layer_id < n_layers:
        ...
```

改成：

```python
for layer in self.layers.values():
    if layer.layer_id < n_layers:
        ...
```

这个修改是正确的。

原因是 PP 切分后，每个 stage 的 `self.layers` 只包含局部 layer。例如 stage 3 可能只有：

```text
layers.33 - layers.42
```

如果用本地计数 `layer_id = 0, 1, 2...`，会丢失全局 layer id。用 `layer.layer_id` 才能正确判断这个 layer 是主干层还是 MTP 层。

### 4.2 `MTP layer 不存在就 return output` 的修改

这个修改只能临时绕过 `KeyError: '43'`，不能作为最终方案。

它的问题是：

```text
配置上 MTP 开启
loss 上 MTP 开启
但模型实际没有返回 MTP logits
```

因此会在 loss 阶段继续失败。

最终方案不应该静默跳过 MTP，而应该：

```text
如果 MTP 开启，就必须让最后 stage 拥有 MTP layer，并返回 [main_logits, mtp_logits]
```

如果条件不满足，应抛出清晰错误，而不是返回普通 Tensor。

### 4.3 compile.enable = false 的作用

关闭 compile/autofuse 可以避开编译路径上的报错或干扰，使问题暴露在普通 forward/loss 阶段。

但它不解决 PP + MTP 的模型结构问题。当前新的 batch size mismatch 正说明：compile 不是根因，真正的问题仍然是 PP 切分和 MTP forward 协议不匹配。

## 5. 概念问题汇总

### 5.1 MTP 是什么

MTP 是 Multi-Token Prediction，多 token 预测。

普通语言模型每个位置预测下一个 token：

```text
输入: [A, B, C, D]
预测: [B, C, D, E]
```

MTP 会额外训练模型预测更远的 token。例如 `num_mtp_modules = 1` 时，除了主 loss，还会增加一个 offset 为 1 的 MTP loss：

```text
main branch:
  position A -> predict B
  position B -> predict C
  position C -> predict D
  position D -> predict E

mtp branch:
  使用主干 hidden + offset token embedding
  额外预测下一偏移目标
```

在当前代码里：

```python
output_list[0] = main_logits
output_list[1] = mtp_logits
```

loss 里：

```python
total_loss = main_loss + mtp_loss * mtp_loss_weight
```

### 5.2 stage 指什么

PP 里的 stage 是 pipeline 切分出来的模型片段。

例如完整模型是：

```text
embedding -> layer0 -> layer1 -> ... -> layer42 -> norm -> output
```

PP 会把它切成多个 stage：

```text
stage0: embedding + layers.0-10
stage1: layers.11-21
stage2: layers.22-32
stage3: layers.33-42 + norm + output
```

每个 stage 只持有一部分模型参数，只执行自己负责的 forward。

### 5.3 为什么 pipeline_parallel_degree = 2 会有 4 个 stage

`pipeline_parallel_degree = 2` 表示物理 PP rank 数是 2。

但当前 schedule 是：

```toml
pipeline_parallel_schedule = "Interleaved1F1B"
```

这是 multi-stage schedule。当前通用 PP 逻辑在没有显式设置 `pipeline_parallel_layers_per_stage` 时，默认：

```text
每个 PP rank 放 2 个 virtual stage
```

所以：

```text
物理 PP rank 数 = 2
每个 rank 的 virtual stage 数 = 2
总 stage 数 = 2 * 2 = 4
```

stage 分配方式是 loop 风格：

```text
PP rank 0: stage 0, stage 2
PP rank 1: stage 1, stage 3
```

因此日志里会看到：

```text
stage_idx0
stage_idx1
stage_idx2
stage_idx3
```

### 5.4 tok_embeddings 是什么

`tok_embeddings` 是 token embedding 层：

```python
self.tok_embeddings = nn.Embedding(vocab_size, dim)
```

它负责把 token id 转成向量：

```text
token ids: [B, S]
embedding: [B, S, dim]
```

在 PP 中，`tok_embeddings` 只应该位于 first stage。后面的 stage 收到的是 activation，不再是 token id。

### 5.5 hc_head 是什么

`hc_head` 是 DeepSeekV4 的 HC 输出头，用来把 HC 形态的 hidden state 压回普通 hidden state。

DeepSeekV4 主干层输出形态是：

```text
[B, S, hc_mult, dim]
```

其中：

```text
hc_mult = 4
```

在接 `norm` 和 `output` 前，需要经过：

```python
h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
```

变成：

```text
[B, S, dim]
```

`hc_head` 不是 MTP 引入的。没有 MTP 时 DeepSeekV4 也需要 `hc_head`。只是当前通用 PP 切分没有显式包含它，所以在 PP+MTP 问题里一起暴露出来。

### 5.6 为什么加上 MTP 后感觉才需要 hc_head

不是 MTP 导致了 `hc_head`，而是：

1. 非 PP 或完整模型中，`hc_head` 原本就在模型里，会正常执行。
2. PP 切分后，通用切分函数没有把 `hc_head` 放进最后 stage。
3. 开启 MTP 后，loss 对输出 shape 更敏感，因此 `hc_head` 缺失的问题更明显。

所以正确理解是：

```text
hc_head 是 DeepSeekV4 本身需要的模块；
MTP 只是让这个切分缺陷更快暴露出来。
```

### 5.7 sidecar 是什么

这里说的 sidecar 不是框架固定术语，而是“跟着主数据一起跨 stage 传递的附加信息”。

PP 主数据是：

```text
h  # 当前 stage 要继续加工的 hidden states
```

但 MTP 和 MoE routing 还需要额外信息：

```text
input_ids          # 原始 token id，给 MoE routing 用
mtp_input_offsets  # MTP offset token embedding，给最后 stage 的 MTPModule 用
```

所以 stage 间传递 payload：

```python
(h, input_ids, mtp_input_offsets)
```

这里 `h` 会被每个 stage 更新，但 `input_ids` 和 `mtp_input_offsets` 是从原始 token 派生出来的，不应该被中间 stage 改写。

### 5.8 为什么中间 stage 只更新 h，原样转发 sidecar

中间 stage 的职责是继续跑主干 TransformerBlock：

```python
h = layer(h, input_ids, ...)
```

它需要读取 `input_ids`，因为 MoE routing 可能依赖 token id。但它不应该修改 `input_ids`。

`mtp_input_offsets` 只在最后 stage 的 MTPModule 中使用，中间 stage 不需要处理它。

因此中间 stage 的逻辑应该是：

```python
h, input_ids, mtp_input_offsets = payload

for layer in local_main_layers:
    h = layer(h, input_ids, ...)

return (h, input_ids, mtp_input_offsets)
```

### 5.9 为什么不是每个 stage 都执行 seq_len -= num_mtp_modules

`seq_len -= num_mtp_modules` 的含义是：

```text
从原始 token 序列中，切出主干输入长度，同时为 MTP 保留 offset token。
```

例如：

```text
tokens = [A, B, C, D, E]
num_mtp_modules = 1
```

first stage 应该切成：

```text
main_input_ids    = [A, B, C, D]
mtp_offset_tokens = [B, C, D, E]
```

所以：

```text
seq_len = 5 - 1 = 4
```

但是 stage1、stage2、stage3 收到的已经不是原始 token ids，而是裁剪后的 activation：

```text
h = hidden([A, B, C, D])
```

如果每个 stage 都再裁一次，就会变成：

```text
stage0: 5 -> 4
stage1: 4 -> 3
stage2: 3 -> 2
stage3: 2 -> 1
```

这显然会错误地缩短序列。

因此：

```text
只有 first stage 对原始 token ids 做 seq_len -= num_mtp_modules；
后续 stage 使用 h.shape[1]，不能再减。
```

### 5.10 MTP 没有参与训练时，为什么会说最后 stage 输出和 MTP 不符合

当前确实是 MTP 没有真正参与训练，因为最后 stage 没有 `layers.43`。

之前说“最后 stage 的返回值与 MTP 不符合”，意思不是“最后 stage 的输出已经作为 MTP 输入并失败”，而是：

```text
配置和 loss 认为 MTP 开启；
但模型实际只返回普通 Tensor；
loss 仍然按 MTP 输出 list 去解释这个 Tensor。
```

于是：

```python
preds[0]
```

被错误解释成第一个 logits，而实际是 Tensor 的第 0 个 batch 切片。

正确的 PP+MTP 中，最后 stage 应该先产出主干 hidden：

```text
prev_embed = main hidden after hc_head
```

再把它和 `mtp_input_offsets` 一起送入 MTPModule：

```python
mtp_h = mtp_layer(mtp_input_offsets, prev_embed, input_ids, ...)
```

最后返回：

```python
[main_logits, mtp_logits]
```

## 6. 最终完整修复方案

### 6.1 新增 DeepSeekV4 专用 PP 切分

不要直接改上游通用 `pipeline_llm`，建议在 NPU 仓新增 DeepSeekV4 专用切分逻辑：

```text
torchtitan_npu/models/deepseek_v4/infra/pipeline_parallel.py
```

新增函数：

```python
pipeline_deepseek_v4(...)
generate_deepseek_v4_fqn_per_model_part(...)
```

然后在：

```text
torchtitan_npu/models/deepseek_v4/__init__.py
```

将：

```python
from torchtitan.distributed.pipeline_parallel import pipeline_llm
...
pipelining_fn=pipeline_llm
```

改为：

```python
from torchtitan_npu.models.deepseek_v4.infra.pipeline_parallel import (
    pipeline_deepseek_v4,
)
...
pipelining_fn=pipeline_deepseek_v4
```

DeepSeekV4 专用切分需要满足：

1. layer-like 模块数量使用：

```python
num_layer_like_modules = model_args.n_layers + model_args.num_mtp_modules
```

2. first stage 包含：

```text
tok_embeddings
```

3. last stage 包含：

```text
MTP layers
hc_head
norm
output
```

4. 最后输出模块名必须是：

```text
output
```

不是：

```text
lm_head
```

当前 43 层 + 1 个 MTP layer + 4 virtual stages 推荐切分为：

```python
stage0 = [
    "tok_embeddings",
    "layers.0", "layers.1", ..., "layers.10",
]

stage1 = [
    "layers.11", ..., "layers.22",
]

stage2 = [
    "layers.23", ..., "layers.33",
]

stage3 = [
    "layers.34", ..., "layers.43",
    "hc_head",
    "norm",
    "output",
]
```

这样 `layers.43` 才会进入最后 stage，MTP 才能真正参与训练。

### 6.2 修复 DeepSeekV4Model.forward 的 PP 协议

建议将 `forward` 签名扩展为：

```python
def forward(
    self,
    tokens,
    input_ids=None,
    mtp_input_offsets=None,
    attention_masks=None,
    positions=None,
):
```

并兼容两种 PP 传参方式：

```python
# 情况 1：下游 stage 收到整个 tuple 作为 tokens
if isinstance(tokens, tuple):
    h, input_ids, mtp_input_offsets = tokens

# 情况 2：PP 框架把 tuple unpack 成多个位置参数
else:
    h = tokens
    # input_ids / mtp_input_offsets 从显式参数读取
```

判断 stage 类型：

```python
is_first_stage = self.tok_embeddings is not None
is_last_stage = self.output is not None
has_mtp = self.model_args.num_mtp_modules > 0
```

first stage 处理原始 token ids：

```python
seq_len = tokens.shape[1] - self.model_args.num_mtp_modules

main_input_ids = tokens[:, :seq_len].detach().long()
input_ids = main_input_ids

h = self.tok_embeddings(main_input_ids)
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

mtp_input_offsets = []
for mtp_idx in range(self.model_args.num_mtp_modules):
    offset = mtp_idx + 1
    mtp_tokens = tokens[:, offset : offset + seq_len]
    mtp_input_offsets.append(self.tok_embeddings(mtp_tokens))
```

注意：`mtp_input_offsets` 的每个元素形状应为：

```text
[B, S, dim]
```

不要 expand 成：

```text
[B, S, hc_mult, dim]
```

因为 `MTPModule.forward()` 内部会自己 expand。

non-first stage 处理 PP payload：

```python
h, input_ids, mtp_input_offsets = payload
seq_len = h.shape[1]
```

这里不能再执行：

```python
seq_len -= self.model_args.num_mtp_modules
```

所有 stage 都只执行自己持有的主干层：

```python
for layer in self.layers.values():
    if layer.layer_id < self.model_args.n_layers:
        h = layer(
            h,
            input_ids,
            freqs_cis,
            self.hadamard_mat,
            attention_masks,
        )
```

如果不是最后 stage：

```python
return (h, input_ids, mtp_input_offsets)
```

最后 stage 先产生 main logits：

```python
h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
prev_embed = h

h = self.norm(h)
main_logits = self.output(h.float())
```

如果没有开启 MTP：

```python
return main_logits
```

如果开启 MTP，必须检查：

```python
if str(self.model_args.n_layers) not in self.layers:
    raise RuntimeError("MTP is enabled, but MTP layer is missing from last PP stage")

if mtp_input_offsets is None:
    raise RuntimeError("MTP is enabled, but mtp_input_offsets was not received")
```

然后执行 MTP：

```python
output_list = [main_logits]

for mtp_idx in range(self.model_args.num_mtp_modules):
    layer_id = self.model_args.n_layers + mtp_idx
    mtp_layer = self.layers[str(layer_id)]

    mtp_h = mtp_layer(
        mtp_input_offsets[mtp_idx],
        prev_embed,
        input_ids,
        self.freqs_cis_wo_compressor,
        self.hadamard_mat,
        attention_masks,
    )

    mtp_h = self.hc_head(
        mtp_h,
        self.hc_head_fn,
        self.hc_head_scale,
        self.hc_head_base,
    )
    prev_embed = mtp_h

    mtp_h = self.norm(mtp_h)
    mtp_logits = self.output(mtp_h.float())
    output_list.append(mtp_logits)

return output_list
```

### 6.3 修复 input_ids 传递

当前 non-first stage 中这段是错误的：

```python
input_ids = tokens[:, :seq_len].detach().long()
h = tokens[:, :seq_len]
```

因为 non-first stage 的 `tokens` 实际是 float activation。

必须改成从 sidecar 中取：

```python
h, input_ids, mtp_input_offsets = tokens
```

这样 MoE routing 才会使用真实 token id，而不是 activation 强转出来的垃圾 long 值。

### 6.4 修复 loss 防御

在：

```text
torchtitan_npu/patches/torchtitan/loss.py
```

的 `multi_token_cross_entropy_loss()` 开头加检查：

```python
if not isinstance(preds, (list, tuple)):
    raise RuntimeError(
        "MTP is enabled, but model did not return main/MTP logits. "
        "Expected [main_logits, mtp_logits, ...]."
    )
```

这样如果模型没有真正返回 MTP logits，会直接报清晰错误，不会再出现迷惑性的：

```text
Expected input batch_size (...) to match target batch_size (...)
```

### 6.5 处理 hc_head 顶层参数

`hc_head` 本身是 child module，可以通过 PP 切分保留在最后 stage。

但这些是顶层 `Parameter`：

```python
self.hc_head_fn
self.hc_head_base
self.hc_head_scale
```

当前通用 `pipeline_module_split()` 主要按 `named_children()` 删除模块，不一定会删除这些顶层参数。

因此需要检查非最后 stage 是否还残留这些参数。

如果残留，会导致非最后 stage optimizer 持有无用参数。功能上不一定立刻报错，但会带来参数统计、优化器状态和显存开销问题。

建议在 DeepSeekV4 专用 PP 切分中，如果当前 stage 不包含 `hc_head`，同时将这些顶层参数置为 `None` 或从参数集合中移除。

### 6.6 验证项

修改后需要验证：

1. stage 切分日志应包含：

```text
stage3: ["layers.34", ..., "layers.43", "hc_head", "norm", "output"]
```

2. 不再出现：

```text
KeyError: '43'
```

3. 不再出现：

```text
Expected input batch_size (16372) to match target batch_size (4)
```

4. 最后 stage 输出应是 list：

```python
[
    main_logits,  # [B, S, vocab]
    mtp_logits,   # [B, S, vocab]
]
```

5. `main_logits.shape[1]` 和 `mtp_logits.shape[1]` 应等于主干序列长度：

```text
S = original_seq_len - num_mtp_modules
```

例如输入长度 4097、`num_mtp_modules = 1` 时：

```text
S = 4096
```

6. PP stage 间 tuple 传递需要实测：

```text
float h + int64 input_ids + float mtp_input_offsets
```

理论上 P2P 可以传不同 dtype 的 Tensor，但 PyTorch `PipelineStage` 第一次 forward 会推断 shape，需要确认它能正确处理混合 tuple。若不能，需要改成框架更容易识别的固定位置参数形式。

### 6.7 最终预期

最终 PP + MTP 的数据流应变成：

```text
stage0:
  raw tokens
    -> main_input_ids
    -> h
    -> mtp_input_offsets
  return (h, input_ids, mtp_input_offsets)

stage1:
  receive (h, input_ids, mtp_input_offsets)
  update h
  return (h, input_ids, mtp_input_offsets)

stage2:
  receive (h, input_ids, mtp_input_offsets)
  update h
  return (h, input_ids, mtp_input_offsets)

stage3:
  receive (h, input_ids, mtp_input_offsets)
  run remaining main layers
  hc_head -> norm -> output -> main_logits
  MTPModule(mtp_input_offsets, prev_embed, input_ids)
  hc_head -> norm -> output -> mtp_logits
  return [main_logits, mtp_logits]
```

这样 MTP 才真正参与训练，loss 的输入协议也和模型输出一致。
