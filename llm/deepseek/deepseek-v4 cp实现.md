# DeepSeek-V4 CP 设计文档

## 1. 背景和目标

DeepSeek-V4 的注意力层按 `compress_ratio` 分为多种类型。当前 CP 实现覆盖两类：

- C1A: `compress_ratio == 1`，只使用滑动窗口原始 KV。
- C128A: `compress_ratio == 128`，同时使用滑动窗口原始 KV 和 128 压缩后的 KV。

CP 的目标是把序列维按 `cp_size` 切分，每个 rank 只计算自己的局部 chunk，同时保持与非 CP 全序列计算一致的注意力可见范围、RoPE 位置和反向梯度。

当前入口是：

```text
parallelize_deepseek_v4
  -> patch_deepseek_v4_for_context_parallel(model, cp_mesh)
  -> Attention.forward = attention_forward_with_cp
```

入口处会校验：

```text
seq_len % (cp_size * 128) == 0
```

这个约束保证每个 rank 的 `chunk = seq_len / cp_size` 能被 128 整除，满足 C128A 的压缩 KV 分块要求。

## 2. 核心设计

CP 设计围绕三个问题展开：

1. RoPE 位置必须使用全局 token 坐标。
2. 滑动窗口跨 rank 时，需要从前一个 rank 补齐边界 KV。
3. C128A 的 compressed KV 是跨 rank 全局可见的，但每个 query 只能看见因果范围内的压缩 token。

因此代码里有两个公共通信算子：

| 模块 | Forward | Backward | 用途 |
| --- | --- | --- | --- |
| `BoundaryExchange` | rank `r-1 -> r` | rank `r -> r-1` | 交换滑动窗口需要的前置 KV |
| `AllGatherCompressedKV` | all-gather local compressed KV | reduce-scatter grad | 聚合 C128A 的全局压缩 KV |

`window_size` 表示真实注意力窗口宽度，默认是 128。边界只交换 `window_size - 1` 个 KV，默认 127 个，因为当前 query 对应的 KV 已经在本 rank 的 local KV 中。

为了降低函数参数和返回值复杂度，当前实现把相关数据封装成几个具名结构：

| 结构 | 作用 |
| --- | --- |
| `BoundaryExchangeInfo` | 封装 P2P 边界交换需要的 `rank`、`cp_size`、`group`、`init_value` |
| `CPAttentionModules` | 封装 `pre_attn`、`inner_attn`、`post_attn` 三段 Attention 子模块 |
| `CPForwardContext` | 封装当前 rank 的 CP 上下文，包括 `rank`、`size`、`group`、`chunk_size`、`window_size` |
| `C128ACompressTopkConfig` | 封装 C128A 生成 compressed topk 索引所需的配置 |
| `AttentionForwardOutput` | 用 `NamedTuple` 表达标准 `Attention.forward` 的 10 个返回字段 |

这些封装不改变运行语义，主要是让代码结构更清晰，也避免生产代码里出现过长参数列表、裸 `assert` 和过长 tuple return。

## 3. Patch 流程

`patch_deepseek_v4_for_context_parallel` 做三件事：

1. 从 `cp_mesh` 取出 `cp_group`、`cp_rank`、`cp_size`。
2. 校验序列长度满足 CP 对齐要求。
3. 给 `Attention` 和 `SparseAttention` 类挂上 CP 上下文，并替换 `Attention.forward`。

替换后的 `attention_forward_with_cp` 根据 layer 的 `compress_ratio` 分发：

```text
attention_forward_with_cp
  -> 构造 CPAttentionModules
  -> 构造 CPForwardContext

compress_ratio == 1
  -> c1a_forward_with_cp(modules, x, freqs_cis, attention_masks, context)

compress_ratio == 128
  -> c128a_forward_with_cp(modules, x, freqs_cis, attention_masks, context)

compress_ratio == 4
  -> NotImplementedError
```

C4A 当前不在 CP dispatcher 中支持。

## 4. 公共边界 KV 交换

C1A 和 C128A 都使用 `_exchange_and_concat_boundary_kv`：

```text
context: CPForwardContext(rank, size, group, chunk_size, window_size)
local kv:       [B, chunk, D]
send tensor:    local kv 的最后 window_size - 1 个 token
recv tensor:    前一个 rank 发来的 boundary kv

rank 0:
  kv_full = local_kv

rank > 0:
  kv_full = [boundary_kv, local_kv]
```

rank 0 没有前一个 rank，因此收到的是 0。代码里使用：

```text
kv = kv + boundary_kv.sum(...) * 0.0
```

这个写法不改变数值，但会把 `BoundaryExchange` 保留在 rank 0 的 autograd 图中，使各 rank 的反向通信路径保持一致。

`BoundaryExchange.apply` 的非 Tensor 参数通过 `BoundaryExchangeInfo` 传入：

```text
BoundaryExchangeInfo(
  rank=context.rank,
  cp_size=context.size,
  group=context.group,
  init_value=0.0,
)
```

这样 forward 参数更集中；backward 只需要返回 `grad_send` 和 `None`，其中 `None` 对应 `BoundaryExchangeInfo` 没有梯度。

## 5. C1A 执行流程

C1A 没有压缩 KV，核心是保证滑动窗口跨 rank 正确。

### 5.1 输入布局

每个 rank 输入：

```text
x: [B, chunk, hidden_dim]
freqs_cis_global: [global_seq_len, rope_dim]
```

先根据全局位置切出本 rank 的 RoPE：

```text
global_start = cp_rank * chunk
freqs_local = freqs_cis_global[global_start : global_start + chunk]
```

这里不能使用 `freqs_cis[:chunk]`，否则所有 rank 都会使用从 0 开始的位置编码。

### 5.2 Forward 流程

```text
c1a_forward_with_cp
  输入:
    modules: CPAttentionModules
    context: CPForwardContext

  1. _local_freqs(freqs_cis_global, x, context)
       -> chunk, freqs_local

  2. modules.pre_attn(x, freqs_local, None)
       -> q, kv

  3. _exchange_and_concat_boundary_kv(kv, context)
       rank 0: kv_full = kv
       rank r: kv_full = [boundary_127, kv]

  4. modules.inner_attn.sparse_attn(q, kv_full, attn_sink)
       使用滑动窗口注意力

  5. modules.post_attn(o, freqs_local, ...)
       输出当前 rank 的局部结果

  6. _c1a_output(...)
       返回 AttentionForwardOutput
```

### 5.3 窗口对齐逻辑

以默认 `window_size = 128` 为例，rank `r > 0` 的 `kv_full` 布局是：

```text
kv_full = [前一 rank 的 127 个 KV, 当前 rank 的 chunk 个 KV]
```

对于本 rank 内第 `i` 个 query，它需要看见：

```text
前 127 个历史 token + 当前 token
```

因此跨 rank 只需要补 127 个历史 KV，而不是 128 个。

## 6. C128A 执行流程

C128A 同时有两路 KV：

- `local_window_kv`: 原始窗口 KV，用于局部滑动窗口。
- `local_kv_compress`: 128 压缩后的 KV，用于长程注意力。

### 6.1 Forward 流程

```text
c128a_forward_with_cp
  输入:
    modules: CPAttentionModules
    context: CPForwardContext

  1. _local_freqs(freqs_cis_global, x, context)
       -> chunk, freqs_local

  2. modules.pre_attn(x, freqs_local, None)
       -> q, local_window_kv, local_kv_compress

  3. _exchange_and_concat_boundary_kv(local_window_kv, context)
       rank 0: kv_full = local_window_kv
       rank r: kv_full = [boundary_127, local_window_kv]

  4. offset = kv_full.size(1)
       compressed KV 的索引从 window KV 后面开始

  5. AllGatherCompressedKV(local_kv_compress, context.group)
       -> global_kv_compress

  6. 裁剪因果可见的压缩 KV
       valid_compress_len = (context.rank + 1) * (chunk // 128)
       causal_kv_compress = global_kv_compress[:, :valid_compress_len, :]

  7. get_c128a_compress_topk_idxs(bsz, C128ACompressTopkConfig(...))
       根据全局 query 位置生成 compressed KV 的可见索引

  8. modules.inner_attn.sparse_attn(
       q, kv_full, attn_sink, causal_kv_compress,
       compress_topk_idxs=compress_topk
     )

  9. modules.post_attn(o, freqs_local, ...)
       输出当前 rank 的局部结果

  10. _c128a_output(...)
       返回 AttentionForwardOutput
```

### 6.2 为什么要 all-gather compressed KV

C128A 的压缩 KV 表示长程上下文，当前 rank 的 query 可能需要访问更早 rank 产生的压缩 KV。因此 forward 需要先 all-gather 每个 rank 的 `local_kv_compress`。

反向时，`AllGatherCompressedKV.backward` 使用 reduce-scatter，把全局梯度切回各 rank 对应的 `local_kv_compress` 梯度。

### 6.3 为什么要裁剪 compressed KV

虽然 all-gather 得到了所有 rank 的压缩 KV，但当前 rank 不能看见未来 rank 的 token。因果可见长度是：

```text
valid_compress_len = (cp_rank + 1) * (chunk // 128)
```

所以：

```text
rank 0 只能看 rank 0 的 compressed KV
rank 1 能看 rank 0 和 rank 1 的 compressed KV
rank r 能看 [0, r] 范围内的 compressed KV
```

### 6.4 compressed topk 索引

`get_c128a_compress_topk_idxs` 使用全局 query 坐标计算压缩 KV 的可见范围。当前函数签名为：

```text
get_c128a_compress_topk_idxs(
  bsz,
  C128ACompressTopkConfig(
    chunk,
    ratio,
    cp_rank,
    causal_kv_len,
    offset,
    device,
  ),
)
```

对于本 rank 内第 `i` 个 query：

```text
global_query_pos = cp_rank * chunk + i
可见 compressed token 满足:
j < (global_query_pos + 1) // 128
```

生成索引时还要加上 `offset`：

```text
offset = kv_full.size(1)
```

原因是 `SparseAttention.forward` 的逻辑中，最终 KV 布局等价于：

```text
[window_kv, compressed_kv]
```

所以 compressed KV 的索引必须从 window KV 后面开始。

## 7. SFA converter 下的位置对齐

如果启用了 NPU SFA converter，`SparseAttention.forward` 会替换成 `sdpa_to_sfa_adapter`。这时 CP rank `> 0` 要额外处理 SFA kernel 的位置语义。

当前 `sdpa_to_sfa_adapter` 已拆成几条更清晰的 helper 路径：

```text
sdpa_to_sfa_adapter
  -> _ensure_int32_indices

  cp_rank <= 0:
    -> _run_sfa_with_native_positions

  cp_rank > 0 且 compress_ratio == 128:
    -> _c128a_cp_sfa_with_global_positions

  cp_rank > 0 且 compress_ratio == 1:
    -> _c1a_cp_sfa_fallback

  其他情况:
    -> _c4a_cp_sfa_with_shifted_query
```

这里的拆分只是代码组织变化，核心语义仍然是：rank0/non-CP 使用 kernel 原生位置；rank>0 需要按不同 attention 类型修正位置或 fallback。

### 7.1 C128A

C128A 走 `_c128a_cp_sfa_with_global_positions`。

核心思想是只 padding KV，不 padding query，让 SFA kernel 的有效位置恢复到全局 token 坐标：

```text
global_start = cp_rank * chunk
n_boundary = window_size - 1
kv_prefix_len = global_start - n_boundary

kv_padded = [zero_prefix, boundary_kv, local_kv]
```

这样 SFA 内部的 causal mask 和 band window 都能按全局位置解释。

### 7.2 C1A

C1A 在 CP rank `> 0` 时走 `_c1a_cp_sfa_fallback`，内部调用 `SparseAttention.original_forward`，并显式传入 `window_topk_idxs`：

```text
window_topk[i] = [i, i + 1, ..., i + window_size - 1]
```

这是因为 rank `> 0` 的 `kv_states` 布局是：

```text
[boundary_127, local_chunk]
```

本地第 `i` 个 query 的正确窗口正好对应 `kv_states[i : i + window_size]`。

### 7.3 Rank0 / Non-CP

rank0 和非 CP 场景走 `_run_sfa_with_native_positions`。这条路径不额外 padding，也不修正 query 位置，因为 kernel 看到的 query/KV 位置就是从当前序列开头开始的标准因果位置。

### 7.4 C4A 预留路径

`sdpa_to_sfa_adapter` 中仍保留 `_c4a_cp_sfa_with_shifted_query`，用于把 C4A 的 query 位置向后平移到 boundary KV 后面。不过 DeepSeek-V4 CP dispatcher 当前对 `compress_ratio == 4` 仍然直接抛 `NotImplementedError`，所以这部分不是当前 C1A/C128A 主链路。

## 8. 讲解时可以强调的关键点

1. CP 切分的是序列维，但注意力语义仍然是全局序列语义。
2. RoPE 必须按全局位置切片，这是正确性的第一层保证。
3. 滑动窗口跨 rank 只补 `window_size - 1` 个历史 KV，因为当前 token 的 KV 在本 rank。
4. `BoundaryExchange` 不只是 forward 通信，还手写了 backward 的反向 P2P 梯度路径。
5. C128A 的 compressed KV 需要 all-gather，但使用前必须按 rank 裁剪，避免未来信息泄露。
6. compressed KV 的 topk 索引用全局 query 坐标生成，并通过 `offset` 对齐 `[window_kv, compressed_kv]` 的拼接布局。
7. `CPForwardContext` 和 `CPAttentionModules` 让 C1A/C128A 共享同一套上下文和模块传递方式，减少重复参数。
8. `AttentionForwardOutput` 保持了标准 10 字段返回协议，但用具名字段表达，便于讲解和维护。
9. SFA converter 下，rank `> 0` 要修正 kernel 的位置理解，否则边界 KV 和 compressed KV 的因果关系会错位。

## 9. 当前支持范围

| 类型 | compress_ratio | CP 支持状态 | 主要通信 |
| --- | --- | --- | --- |
| C1A | 1 | 已支持 | BoundaryExchange |
| C128A | 128 | 已支持 | BoundaryExchange + AllGatherCompressedKV |
| C4A | 4 | dispatcher 暂不支持 | 后续需要补齐 |

## 10. 一句话总结

这套 CP 方案的核心是：每个 rank 只算自己的 chunk，但通过全局 RoPE、边界 KV P2P、compressed KV all-gather 和全局坐标索引，把局部计算还原成与非 CP 全序列一致的注意力语义。



因为 RoPE 不是“给当前 chunk 内第几个 token 编码”，而是“给这个 token 在整条序列里的绝对位置编码”。

如果 CP 把序列切成多个 rank：

global seq: [0 ... 4095]
cp_size=4
rank0: token 0...1023
rank1: token 1024...2047
rank2: token 2048...3071
rank3: token 3072...4095
rank1 本地看自己的 chunk 时，第一个 token 的 local index 是 0，但它在原始完整序列里的位置是 1024。RoPE 应该用 1024，而不是 0。

原因是注意力里的相对位置信息来自 Q/K 的旋转相位差。对 RoPE 来说，两个 token 的注意力位置关系依赖它们的全局位置差：

q_pos = 1024
k_pos = 1000
相对距离 = 24
如果 rank1 错误地从 0 开始编码，那么同一个 query 会变成：

q_pos = 0
k_pos = 1000  或者边界 KV 仍带旧位置
相对距离完全错
这样跨 rank 的 boundary KV、C128A 的 compressed KV、以及非 CP 全序列路径就对不上了。结果是同一个 token 在 CP 和非 CP 下看到的 Q/K 相位不一致，attention score 会变，loss/精度也会偏。

所以 CP 里必须这样切：

global_start = cp_rank * chunk
freqs_local = freqs_cis_global[global_start : global_start + chunk]
而不能简单用：

freqs_local = freqs_cis_global[:chunk]
一句话：CP 只是把计算切开，不能把序列位置重新编号；RoPE 要保持和原始完整序列完全一样的位置坐标。