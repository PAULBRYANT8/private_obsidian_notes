# 一、概述

DeepSeek-v4 的每个 Transformer Block 包含两大组件：**MHC（Multi-Head Hyper-Connections）** 和 **Attention**。
CP 切分的挑战主要来自 Attention，不同层的 compress_ratio 决定了序列依赖的范围和通信开销。
`compress_ratios` 的模式是 `(1, 1, 4, 128, 4, 128, ..., 4, 128, 4)`，共三类 Attention 层：

| 层类型                | 序列依赖范围                        | 通信挑战                 |
| ------------------ | ----------------------------- | -------------------- |
| ratio = 1（Window）  | 局部 window_size = 128          | 仅边界 token 通信         |
| ratio = 128（C128A） | 全局（但 token 数极少）               | 压缩 KV 的全局 all-gather |
| ratio = 4（C4A）     | 全局（Lightning Indexer 选 top-k） | 压缩 KV 的全局 all-gather |

**ratio = 1（Window Attention）**：每个 query 只向左看 `window_size=128` 个原始 token，依赖范围纯局部。CP 切分后，rank r 最前面的若干 query 的窗口会跨越到 rank r-1 的末尾，因此只需要一次轻量的 P2P 通信，从前驱 rank 借来末尾 128 个 token 的 KV 即可，通信量极小（约 131 KB/层）。

**ratio = 128（C128A）**：每 128 个原始 token 被 Compressor 压缩为 1 个压缩 token，全局压缩序列极短（例如 seqlen=65536 时仅 512 个压缩 token）。Compressor 使用 `overlap=False`，无跨 group 依赖，不需要 P2P 边界修正。AllGather 的通信量相比 C4A 缩小 32 倍（约 0.5 MB/层），几乎可以忽略。

**ratio = 4（C4A）**：每 4 个原始 token 被 Compressor 压缩为 1 个压缩 token，再由 Lightning Indexer 从全局压缩序列中检索 top-k 个最相关的压缩 token 参与 attention。CP 切分后必须通过 AllGather 将各 rank 的本地压缩 KV 汇聚成全局视图。此外，C4A 的 Compressor 使用 `overlap=True`，相邻压缩 token 之间存在跨 group 依赖，CP 边界处还需要额外的 P2P 边界修正。

**关于 MHC（Multi-Head Hyper-Connections）**
> [!NOTE] 模型用的是 MHC，不是单头 HC
> 代码中 `hc_mult=4`，每个 token 的隐藏状态形状为 `[B, S, 4, D]`，即每个位置
> 维护 4 个并行 stream，这是 MHC（Multi-Head Hyper-Connections）而非单头 HC。

MHC 是纯 **per-token** 的局部操作：HcPre / HcPost 的 Sinkhorn 路由只在同一 token 的 `hc_mult=4` 个 stream 之间做加权混合，**不涉及任何跨 token 的序列交互**。CP 将序列按 token 切分，同一 token 的 4 个 stream 始终在同一 rank 上，因此 MHC **不需要任何 CP 通信**，对 CP 完全透明。

---

# 二、前提与工程修改

## 2.1 Chunk 对齐约束

chunk = seq_len / cp_degree，必须满足：
```python
seq_len % (cp_size * 128) == 0
```

验证：
```
seqlen=65536, cp=8 → chunk=8192, 8192 % 128 = 0 ✓
seqlen=4096, cp=8 → chunk=512, 512 % 128 = 0 ✓
seqlen=4096, cp=32 → chunk=128, 128 % 128 = 0 ✓
seqlen=4096, cp=64 → chunk=64, 64 % 128 = 64 ✗ → NotImplementedError
```

这样所有层的压缩边界都与 CP 切分边界自然对齐，无碎片问题。此约束同时隐含 `chunk % 4 == 0`，因此**一个校验覆盖全部三类层**（ratio=1 / ratio=4 / ratio=128）。

约束校验在**模型初始化时调用一次**，后续所有 CP 专用 forward 函数均不再重复校验：
```python
def validate_cp_alignment(seq_len: int, cp_size: int) -> None:
	"""
	统一的 CP 对齐校验。
	seq_len % (cp_size * 128) == 0 是最强约束，同时隐含：
	- chunk % 128 == 0 （C128A Compressor 无截断）
	- chunk % 4 == 0 （C4A Compressor 无截断）
	- chunk % 1 == 0 （Window Attention 无特殊要求）
	常见训练序列（4096 的整数倍）在常用 CP degree（≤32）下均自然满足。
	"""
	if seq_len % (cp_size * 128) != 0:
		raise NotImplementedError(
			f"CP requires seq_len % (cp_size * 128) == 0. "
			f"Got seq_len={seq_len}, cp_size={cp_size}, "
			f"remainder={seq_len % (cp_size * 128)}."
		)
```

## 2.2 代码修复

### 2.2.1 args.py：attn_type 属性访问 bug

`DeepSeekV4ModelArgs` 未定义 `attn_type` 属性，直接访问会触发 `AttributeError`。

```python
# model/args.py — 当前错误写法
if (
	job_config.parallelism.context_parallel_degree > 1
		and self.attn_type != "sdpa"  # ← AttributeError！
):
```

修复方式（与 `infra/parallelize.py:186` 保持一致）：
```python
# 修复后
attn_type = getattr(self, "attn_type", "sdpa")
if (
	job_config.parallelism.context_parallel_degree > 1
		and attn_type != "sdpa"
):
	raise NotImplementedError("CP support is only supported for SDPA.")
```

### 2.2.2 SparseAttention：增加 window_topk_idxs 参数

CP 模式下三类层（ratio=1 / C128A / C4A）均需向 `SparseAttention` 传递外部计算的修正窗口索引。原 `SparseAttention.forward()`（`model.py:608`）始终调用 `get_window_topk_idxs(window_size, bsz, seqlen)` 生成本地坐标索引（seqlen = chunk）。对 rank r > 0，`kv_full = [boundary_kv(128) || local_kv(chunk)]` 大小为 128+chunk，但生成的索引范围 `0..chunk-1` 根本无法访问前 128 个边界 token，三类层的窗口 attention 计算均错误。

各节计算的 `cp_window_topk_idxs` 是正确的，但只有在 `SparseAttention` 支持外部传入时才能生效。需修改 `model/model.py` 中 `SparseAttention.forward()`，增加可选参数 `window_topk_idxs=None`：

```python
def forward(
    self,
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    attn_sink: torch.Tensor,
    kv_compress: torch.Tensor | None = None,
    compress_topk_idxs: torch.Tensor | None = None,
    window_topk_idxs: torch.Tensor | None = None,   # 新增：CP 模式传入修正后的窗口索引
):
    bsz, seqlen, _, _ = query_states.size()

    # CP 模式：使用外部传入的修正索引；非 CP 模式：按原逻辑生成（向后兼容）
    if window_topk_idxs is not None:
        topk_idxs = window_topk_idxs
    else:
        topk_idxs = self.get_window_topk_idxs(self.window_size, bsz, seqlen)

    if self.compress_ratio > 1:
        offset = kv_states.size(1)
        if compress_topk_idxs is None:
            compress_topk_idxs = self.get_compress_topk_idxs(query_states, offset)
        topk_idxs = torch.cat(
            [topk_idxs.to(kv_states.device), compress_topk_idxs.to(kv_states.device)],
            dim=-1,
        )
    # ... 其余逻辑不变
```

此修改向后兼容：非 CP 路径不传 `window_topk_idxs`，行为与原来完全相同。

---

# 三、公共基础设施

本节定义三类层共同依赖的基础组件。所有后续 CP 前向函数均直接调用，不再重复实现。

## 3.1 freqs_cis 使用约定

> [!NOTE] freqs_cis 使用约定
> 模型维护两套 freqs_cis，CP 各函数的 `freqs_cis_global` 参数需按层类型选择：
> - ratio=1 层：使用 `model.freqs_cis_wo_compressor`（rope_theta=10000）
> - ratio=4 / ratio=128 层：使用 `model.freqs_cis`（compress_rope_theta=40000）
>
> 这与非 CP 的 `DeepSeekV4Model.forward()` 中的选择逻辑完全一致（`compress_ratios[layer_id] > 1` 时用 `freqs_cis`）。各层的 CP 前向函数统一接收一个 `freqs_cis_global` 参数，调用方负责传入正确的那套。

## 3.2 BoundaryExchange

`BoundaryExchange` 是一个自定义的 PyTorch autograd Function，封装了跨 rank P2P 通信，同时保证反向传播梯度方向正确。三类层均依赖它来传递 CP 边界数据。

**前向传播做什么：**

每个 rank 把自己的 `send_tensor`（上一个 group 的边界数据）发给下一个 rank，同时从上一个 rank 接收对应数据存入 `recv_buf`。通信用 `isend`/`irecv` 非阻塞方式挂起，再统一 `wait()` 等待完成。`init_value` 控制 rank 0 的 `recv_buf` 初始值——rank 0 没有前驱，收不到任何数据，`recv_buf` 保持初始值作为默认边界：
- Window KV 和主 Compressor kv：`init_value=0.0`
- C4A Compressor score：`init_value=float("-inf")`（与非 CP `overlap_transform` 初始化行为一致）

**反向传播做什么：**

梯度的流向与前向数据流向相反。前向是 rank r-1 → rank r 发数据，反向就是 rank r 把收到的梯度（`grad_recv`）反向发回给 rank r-1，同时从 rank r+1 接收梯度（`grad_send`）。这样训练时梯度能正确流回产生边界数据的 rank，参数才能得到正确更新。

**为什么要封装成 autograd Function 而不是普通函数：**

普通函数里的 `dist.isend`/`irecv` 只在前向执行，PyTorch 不知道如何为它生成反向梯度。封装成 `autograd.Function` 后，可以手动定义 `backward`，显式控制梯度的反向通信路径，确保整条训练链路的梯度正确传播。

```python
class BoundaryExchange(torch.autograd.Function):
	@staticmethod
	def forward(ctx, send_tensor, rank, cp_size, group, init_value=0.0):
		ctx.rank, ctx.cp_size, ctx.group = rank, cp_size, group
		# init_value 控制 rank 0 收到的默认值
		# kv: init_value=0.0 （与非 CP overlap_transform value=0 一致）
		# score: init_value=-inf （与非 CP overlap_transform value=-inf 一致）
		if init_value == 0.0:
			recv_buf = torch.zeros_like(send_tensor)
		else:
			recv_buf = torch.full_like(send_tensor, init_value)
		
		reqs = []
		
		if rank < cp_size - 1:
			reqs.append(dist.isend(send_tensor.contiguous(), dst=rank+1, group=group))
		if rank > 0:
			reqs.append(dist.irecv(recv_buf, src=rank-1, group=group))
		for req in reqs: req.wait()
		return recv_buf
		
	@staticmethod
	def backward(ctx, grad_recv):
		grad_send = torch.zeros_like(grad_recv)
		reqs = []
		if ctx.rank > 0:
			reqs.append(dist.isend(grad_recv.contiguous(), dst=ctx.rank-1, group=ctx.group))
		if ctx.rank < ctx.cp_size - 1:
			reqs.append(dist.irecv(grad_send, src=ctx.rank+1, group=ctx.group))
			
		for req in reqs: req.wait()
		return grad_send, None, None, None, None  # init_value 无梯度
```

## 3.3 AllGatherCompressedKV

前向 AllGather 将各 rank 的本地压缩 KV 汇聚成全局视图，反向自动变成 ReduceScatter。C128A 和 C4A 均复用此组件。

```python
class AllGatherCompressedKV(torch.autograd.Function):
      """         
      前向：AllGather local_kv_compress → global_kv_compress
      反向：ReduceScatter grad_global   → grad_local
      """
      
      @staticmethod
      def forward(
          ctx,
          local_kv: torch.Tensor,   # [B, chunk//ratio, head_dim]
          group: dist.ProcessGroup,
      ) -> torch.Tensor:            # [B, seq_len//ratio, head_dim]
          ctx.group = group
          ctx.cp_size = dist.get_world_size(group)
          
          B, local_len, D = local_kv.shape
          global_kv = torch.empty(
              B, local_len * ctx.cp_size, D,
              dtype=local_kv.dtype,
              device=local_kv.device,
          )
          dist.all_gather_into_tensor(
              global_kv, local_kv.contiguous(), group=group
          )
          return global_kv

      @staticmethod
      def backward(
          ctx,
          grad_global: torch.Tensor,  # [B, seq_len//ratio, head_dim]
      ) -> tuple:
          B, global_len, D = grad_global.shape
          local_len = global_len // ctx.cp_size

          grad_local = torch.empty(
              B, local_len, D,
              dtype=grad_global.dtype,
              device=grad_global.device,
          )
          dist.reduce_scatter_tensor(
              grad_local, grad_global.contiguous(), group=ctx.group
          )
          return grad_local, None  # group 无梯度
```

## 3.4 cp_window_topk_idxs

CP 切分后，kv_full = [boundary_kv(128) || local_kv(chunk)]，原始 `GetWindowTopkIdxs` 基于本地坐标 `0..chunk-1` 生成的索引无法访问前 128 个边界 token，必须重新计算。

> [!IMPORTANT] 不能沿用原 GetWindowTopkIdxs 生成的索引
> 原 `GetWindowTopkIdxs` 生成的索引基于本地坐标 `0..chunk-1`，拼接边界 token 后
> local_kv 整体偏移到了位置 128，原有索引无法访问边界 token，**必须重新计算**。
> 换成更简单的理解：由于 CP 切分，rank r 需要接收 rank r-1 后 128 个 token，从而组成新的 kv 矩阵，新矩阵不仅大小发生了改变，每个位置的语义也发生了改变。

**rank 0**（无前驱）：
- `kv_full = local_kv`，形状 `[B, chunk, head_dim]`
- `window_topk` = 原 `GetWindowTopkIdxs(window_size=128, bsz, chunk)` 结果（不变）

**rank r > 0**（有边界 token）：
- `kv_full = cat([boundary_kv, local_kv], dim=1)`，形状 `[B, 128+chunk, head_dim]`
- `kv_full[0:128]` = rank r-1 的末尾 128 个全局 token（全局位置 [r*chunk-128, r*chunk-1]）
- `kv_full[128+j]` = rank r 的本地 token j（全局位置 r * chunk + j）
- 对 query i（全局位置 r*chunk+i），所需 KV 全局窗口 [r*chunk+i-127, r*chunk+i] 映射到 kv_full 后：
	- 起点：kv_full[i+1]（对应全局 r*chunk-127+i）
	- 终点：kv_full[i+128]（对应全局 r*chunk+i）
	- 即 window_topk[i] = [i+1, i+2, ..., i+128]，恰好 128 个，无需 clamp

```python
def cp_window_topk_idxs(bsz: int, chunk: int, window_size: int, cp_rank: int):
	"""
	为 CP 模式生成 window attention 的 topk 索引。
	rank 0 无边界 token，沿用原逻辑；rank > 0 需要考虑前置的 128 个边界 token。
	"""
	if cp_rank == 0:
		# kv_full = local_kv [B, chunk, D]，原逻辑不变
		base = torch.arange(chunk).unsqueeze(1)  # [chunk, 1]
		window_topk = (base - window_size + 1).clamp(0) + torch.arange(window_size)
		window_topk = torch.where(window_topk > base, -1, window_topk)
	
	else:
		# kv_full = [boundary(128) || local_kv(chunk)] [B, 128+chunk, D]
		# query i → kv_full 位置 [i+1, i+2, ..., i+128]
		base = torch.arange(chunk).unsqueeze(1)  # [chunk, 1]
		window_topk = base + torch.arange(1, window_size + 1)  # [chunk, 128]
		# 合法性：i ∈ [0, chunk), 取值范围 [1, chunk+127] ⊆ [0, 128+chunk-1] ✓
	
	return window_topk.unsqueeze(0).expand(bsz, -1, -1)
```

拼接完 window_topk_idxs 后，compress 相关的索引 offset 也需随之调整：
- rank 0：`offset = chunk`（与原来相同）
- rank r > 0：`offset = 128 + chunk`（kv_full 多了 128 个边界 token）

---

# 四、三类层 CP 实现

本节按复杂度由低到高介绍三类层的完整 CP 实现，公共组件（BoundaryExchange / AllGatherCompressedKV / cp_window_topk_idxs）均在第三节已定义，此处直接复用。

## 4.1 ratio=1：Window Attention

### 4.1.1 执行流程

起点：每个 rank 持有本地隐状态 x [B, chunk, D]。

第一步：Q 投影（本地）
	对 x 做 wq_a → wq_b 两次线性投影，得到 q [B, chunk, n_heads, head_dim]，施加全局位置坐标的 RoPE（rank r 的起始位置是 r * chunk）。q 投影完后等待最后做 attention。

第二步：Window KV 投影（本地）
	对 x 做 wkv 投影，经 kv_norm 归一化，施加全局位置坐标的 RoPE，得到本地 kv [B, chunk, head_dim]。

第三步：P2P 边界通信
- rank r-1 把自己末尾 128 个 token 的 kv 通过 BoundaryExchange 发给 rank r，rank r 接收后拼接：
	- rank r > 0：kv_full = cat([boundary_kv, local_kv]) → [B, 128+chunk, head_dim]
	- rank 0：无前驱，kv_full = local_kv → [B, chunk, head_dim]

第四步：生成修正后的 window_topk_idxs
- 拼入边界 token 后，kv_full 的索引空间发生变化，使用 §3.4 的 `cp_window_topk_idxs` 重新生成正确索引。

第五步：Attention 计算
	用 q 对 kv_full 做 window attention，window_topk_idxs 告诉每个 query 访问 kv_full 中的哪 128 个位置，输出 o [B, chunk, head_dim]。

> [!NOTE] 为什么需要 128 个边界 token
> Window Attention 中，每个 query 可以向左看 window_size=128 个原始 token。
> ```
> rank r 的第 0 个 query（位置 r*chunk）：
>   需要 KV 来自 [r*chunk - 127, r*chunk - 1]  ← 这 127 个在 rank r-1 上
>
> rank r 的第 127 个 query（位置 r*chunk + 127）：
>   需要 KV 来自 rank r-1 的最后 1 个 token
>
> rank r 的第 128 个 query（位置 r*chunk + 128）：
>   需要 KV 来自 [r*chunk + 1, r*chunk + 128] ← 全部在 rank r 上，不再依赖 rank r-1
>
> 所以发送完整的 window_size=128 个 token 是最简洁的做法。
> ```
> 通信原语：P2P Send/Recv（仅向前一个 rank 借 128 个 token）
> 通信量：128 × head_dim × 2B = 128 × 512 × 2B ≈ 131KB（极小）

### 4.1.2 Window KV 边界通信代码

**使用 `BoundaryExchange`（§3.2）而非普通 isend/irecv**：Window KV 的边界 token 参与 attention 计算后产生梯度，该梯度必须流回 rank r-1 的 `wkv` 参数。普通 `dist.isend/irecv` 不在 autograd 计算图中，梯度会被静默丢弃。`BoundaryExchange` 已封装好前向通信与反向梯度回传，`init_value=0.0` 与非 CP 行为一致：

```python
# Window KV 边界交换：含反向梯度，init_value=0.0
boundary_kv = BoundaryExchange.apply(
    kv[:, -window_size:, :].contiguous(),   # [B, 128, head_dim]，发给 rank r+1
    cp_rank, cp_size, cp_group, 0.0          # rank 0 的 recv_buf 初始化为全 0
)
# 所有 rank 均调用，BoundaryExchange 内部按 rank 判断是否真正收发：
# rank r > 0：boundary_kv = [B, 128, head_dim]（来自 rank r-1 的最后 128 个 token）
# rank 0：    boundary_kv = 全 0 的缓冲区（无前驱，与非 CP 行为一致）

if cp_rank > 0:
    kv_full = torch.cat([boundary_kv, kv], dim=1)  # [B, 128+chunk, head_dim]
else:
    kv_full = kv                                     # [B, chunk, head_dim]

# 修正 window topk 索引（§3.4）
window_topk = cp_window_topk_idxs(B, chunk, window_size, cp_rank)  # [B, chunk, 128]

# SparseAttention（需先按 §2.2.2 修改，增加 window_topk_idxs 参数）
o = inner_attn.sparse_attn(
    q, kv_full, inner_attn.attn_sink,
    window_topk_idxs=window_topk,
)
```

---

## 4.2 ratio=128：C128A

### 4.2.1 执行流程

起点：每个 rank 持有本地隐状态 x [B, chunk, D]。

第一步：Q 投影（本地）
	与 Window Attention 相同，wq_a → wq_b + 全局位置 RoPE，得到 q [B, chunk, n_heads, head_dim]，等待最后使用。

第二步：Window KV 投影（本地）
	对 x 做 wkv 投影 + kv_norm + 全局位置 RoPE，得到本地 window_kv [B, chunk, head_dim]。此路数据用于 window attention，后续还需要做 P2P 边界通信。

第三步：Compressor 压缩（本地，无通信）
	对 x 用 ratio=128 的 Compressor 做压缩（overlap=False，无相邻 group 依赖，不需要 BoundaryExchange）：wkv + wgate 线性投影，每 128 个 token 加权求和压缩成 1 个 token，施加全局位置 RoPE，得到 local_kv_compress [B, chunk//128, 512]。这一步完全本地，无任何跨 rank 通信。

第四步：Window KV P2P 边界通信
- 与 4.1 节相同：用 `BoundaryExchange.apply()` 从 rank r-1 获取末尾 128 个 window_kv token：
	- rank r > 0：kv_full = cat([boundary_kv, window_kv]) → [B, 128+chunk, head_dim]
	- rank 0：kv_full = window_kv → [B, chunk, head_dim]
- 使用 `cp_window_topk_idxs`（§3.4）重新生成修正后的窗口索引，更新 offset：
	- rank 0：offset = chunk；rank r > 0：offset = 128+chunk

第五步：AllGather kv_compress
	各 rank 的 local_kv_compress 通过 `AllGatherCompressedKV`（§3.3）拼成 global_kv_compress [B, seq_len//128, 512]，每个 rank 拿到完整序列的压缩 KV。反向时自动变成 ReduceScatter。

第六步：因果裁剪
	从 global_kv_compress 中只保留 rank 0 到 rank r 的部分，去掉未来 rank 的数据，得到 causal_kv_compress [B, (cp_rank+1)*chunk//128, 512]。

第七步：生成修正后的 compress_topk_idxs
	C128A 不像 C4A 有 Lightning Indexer，top-k 索引直接用位置关系生成。关键修正：用 rank r 的全局 query 坐标作为 base（而非从 0 开始的本地坐标），生成因果掩码，确保每个 query 只访问自己位置之前的压缩 token。索引值以 kv_full 长度为 offset，与 SparseAttention 的寻址方式对齐。

第八步：SparseAttention
- 用 q 同时做两路 attention：
	- Window attention：q 对 kv_full 做注意力，使用修正后的 window_topk_idxs
	- Compress attention：q 对 causal_kv_compress 做注意力，使用修正后的 compress_topk_idxs
- 两路结果合并，输出 o [B, chunk, head_dim]。

### 4.2.2 数据流设计

```
输入 x: [B, chunk, D]，chunk = seq_len / cp_size
每个 CP rank 独立执行：
┌────────────────────────────────-┐   
│  x: [B, chunk, D]                                                                       │
│       │                                                                                            │
│       ├─ Window KV Proj                                                        │
│       │    → local_window_kv: [B,chunk,512]                     │ ← 纯本地，无通信
│       │                                                                                            │ 
│       └─ Compressor (ratio=128)                                         │
│            → 压缩输出：[B, chunk//128, 512]                      │ ← 纯本地    
└────────────────────────────────┘    
              │   
              │  AllGather（仅压缩 KV，通信量极小）
              │  forward:  AllGather 
              │  backward: ReduceScatter（自动）
              ▼      
global_kv_compress: [B, seq_len//128, 512]   
              │      
              ▼
┌────────────────────────────────────────┐
│ kv_full = window_kv (含 P2P 边界修正)                                                │
│ kv_states = cat([kv_full, causal_kv_compress], dim=1)                    │
│                                                                                                                          │
│  SparseAttention(q, kv_full, attn_sink, causal_kv_compress,         │
│                                  compress_topk_idxs, window_topk_idxs)       │
│ （compress_topk_idxs 需用全局坐标，见 4.2.3）                         │
└────────────────────────────────────────┘
              │     
              ▼ 
        输出: [B, chunk, D]
```

### 4.2.3 通信量分析

以 cp=8 为例：
```
local_kv_compress 形状：[B, chunk//128, 512]
                       = [1, 8192//128, 512]  
                       = [1, 64, 512]
AllGather 后：[1, 64*cp_size, 512] = [1, 512, 512]
通信量 = 512 × 512 × 2B(fp16) = 524,288 B ≈ 512 KB/layer（可忽略）

对比：                                                                                                                                                                       
  ratio=4  kv_compress AllGather：16 MB/layer
  ratio=128 kv_compress AllGather：0.5 MB/layer  ← 缩小 32 倍  
```

### 4.2.4 compress_topk_idxs 全局坐标修正

> [!IMPORTANT] 不能沿用原 GetCompressTopkIdxs 生成的索引
> 原 `GetCompressTopkIdxs` 以本地 `seqlen`（即 chunk）为上界生成索引，例如
> chunk=512, ratio=128 → 索引范围 `[0, 3]`。
> 但 CP 模式下 `causal_kv_compress` 对 rank r 有 `(r+1)*chunk//128` 个 token，
> rank r>0 的 query 需要访问更早 rank 的压缩 KV，原有索引完全不够。
> 必须用**全局 query 坐标**重新计算因果掩码。

```python
def get_c128a_compress_topk_idxs_cp(
	bsz: int,
	chunk: int,
	ratio: int,  # = 128
	cp_rank: int,
	causal_kv_len: int,  # = (cp_rank + 1) * chunk // ratio
	offset: int,  # = kv_full.size(1)，window KV 占用的长度
) -> torch.Tensor:  # [B, chunk, causal_kv_len]
	"""
	为 C128A 在 CP 模式下生成 compress_topk_idxs。
	query 在本地位置 i（全局位置 cp_rank*chunk+i）最多看到全局压缩位置
	j < (cp_rank*chunk + i + 1) // ratio 的压缩 token。
	"""
	# 全局 query 位置（以 1 为基，方便计算 ceil(pos/ratio)）
	global_base = cp_rank * chunk + torch.arange(1, chunk + 1)  # [chunk]
	
	# 压缩 token 全局坐标
	compress_pos = torch.arange(causal_kv_len)  # [causal_kv_len]
	
	# causal_mask[i][j] = True 表示 query i 不能看到压缩位置 j（未来）
	causal_mask = compress_pos.unsqueeze(0) >= (global_base // ratio).unsqueeze(1)
	# [chunk, causal_kv_len]
	
	compress_topk = torch.where(causal_mask, -1, compress_pos + offset)
	# [chunk, causal_kv_len]
	
	return compress_topk.unsqueeze(0).expand(bsz, -1, -1)
```

示例验证（seqlen=4096, cp=8, chunk=512, ratio=128）：
```
rank 0：causal_kv_len = 1*512//128 = 4
query i=0 全局位置 1 → compress_pos < 1//128=0 → 无可用压缩 token（全 -1）
query i=127 全局位置 128 → compress_pos < 128//128=1 → 可用 [0]
query i=511 全局位置 512 → compress_pos < 512//128=4 → 可用 [0,1,2,3]

rank 7：causal_kv_len = 8*512//128 = 32
query i=0 全局位置 3585 → compress_pos < 3585//128=28 → 可用 [0..27]
query i=511 全局位置 4096 → compress_pos < 4096//128=32 → 可用 [0..31]（全量）
```

### 4.2.5 完整前向代码

```python
def c128a_forward_with_cp(
    x: torch.Tensor,                 # [B, chunk, D]
    pre_attn,                        # PreAttention 实例（含 wq_a/q_norm/wq_b/wkv/kv_norm/compressor_128）
    inner_attn,                      # InnerAttention 实例（含 attn_sink/sparse_attn）
    cp_rank: int,
    cp_size: int,
    cp_group: dist.ProcessGroup,
    seq_len: int,
    freqs_cis_global: torch.Tensor,  # 全量 freqs_cis（compress_rope_theta），[max_seq_len, rope_head_dim]
    window_size: int = 128,
) -> torch.Tensor:
    B, chunk, D = x.shape
    ratio = 128
    rd = pre_attn.rope_head_dim
    global_start = cp_rank * chunk
    freqs_local = freqs_cis_global[global_start : global_start + chunk]

    # ── 1. Q 投影（本地）────────────────────────────────────────────────
    qr = pre_attn.q_norm(pre_attn.wq_a(x))
    q  = pre_attn.wq_b(qr).unflatten(-1, (pre_attn.n_heads, pre_attn.head_dim))
    q  = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + pre_attn.eps)
    q_nope, q_rope = torch.split(q, [pre_attn.head_dim - rd, rd], dim=-1)
    q_rope = apply_rotary_emb(q_rope, freqs_local)
    q = torch.cat([q_nope, q_rope], dim=-1)   # [B, chunk, n_heads, head_dim]

    # ── 2. Window KV 投影（本地）──────────────────────────────────────────
    kv = pre_attn.kv_norm(pre_attn.wkv(x))
    kv_nope, kv_rope = torch.split(kv, [pre_attn.head_dim - rd, rd], dim=-1)
    kv_rope = apply_rotary_emb(kv_rope, freqs_local)
    local_window_kv = torch.cat([kv_nope, kv_rope], dim=-1)  # [B, chunk, head_dim]

    # ── 3. Window KV 边界通信（BoundaryExchange §3.2，含反向梯度）──────────
    boundary_kv = BoundaryExchange.apply(
        local_window_kv[:, -window_size:, :].contiguous(),
        cp_rank, cp_size, cp_group, 0.0   # init_value=0.0，rank 0 收到全 0
    )
    if cp_rank > 0:
        kv_full = torch.cat([boundary_kv, local_window_kv], dim=1)  # [B, 128+chunk, D]
    else:
        kv_full = local_window_kv                                     # [B, chunk, D]

    # ── 4. Window topk 索引（CP 修正，§3.4）─────────────────────────────
    window_topk = cp_window_topk_idxs(B, chunk, window_size, cp_rank)
    # [B, chunk, 128]

    # ── 5. Compressor 压缩（overlap=False，无 BoundaryExchange，直接调用原始 forward）──
    # 传入 freqs_local，Compressor.forward() 内部会做 [:chunk:128] 切片，正确对应全局位置
    local_kv_compress = pre_attn.compressor_128(x, freqs_local)  # [B, chunk//128, head_dim]

    # ── 6. AllGather 压缩 KV（前向 AllGather，反向自动 ReduceScatter，§3.3）─
    global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
    # [B, seq_len//128, head_dim]

    # ── 7. 因果有效性裁剪──────────────────────────────────────────────────
    valid_compress_len = (cp_rank + 1) * (chunk // ratio)
    causal_kv_compress = global_kv_compress[:, :valid_compress_len, :]
    # rank 0: [B, chunk//128, D]；rank cp_size-1: [B, seq_len//128, D]

    # ── 8. compress_topk_idxs（全局坐标修正，§4.2.4）──────────────────────
    offset = kv_full.size(1)  # rank 0: chunk；rank r>0: 128+chunk
    compress_topk = get_c128a_compress_topk_idxs_cp(
        B, chunk, ratio, cp_rank, valid_compress_len, offset
    )
    # [B, chunk, valid_compress_len]

    # ── 9. SparseAttention（传入修正后的 window_topk_idxs，§2.2.2）─────────
    o = inner_attn.sparse_attn(
        q, kv_full, inner_attn.attn_sink, causal_kv_compress,
        compress_topk_idxs=compress_topk,
        window_topk_idxs=window_topk,   # 外部修正的窗口索引，覆盖内部 get_window_topk_idxs
    )

    return o
```

---

## 4.3 ratio=4：C4A

### 4.3.1 背景：为什么需要分步处理

C4A（ratio=4）在注意力之前需要把全序列的 KV 压缩到 1/4，但压缩后的序列仍然很长（S/4），不能对所有压缩 token 都做 attention，否则计算量仍然很大。因此引入"先检索、再 attention"的两阶段设计：

**第一阶段：Compressor（压缩）+ Indexer 内部压缩**
这两路压缩结果来自**两个独立的 Compressor 实例**，各有自己的 wkv / wgate 参数：

| 实例                             | head_dim | 输出                       | 作用                |
| :----------------------------- | :------- | :----------------------- | :---------------- |
| 主 Compressor（PreAttention 中）   | 512      | kv_compressor [S/4, 512] | 用于实际 attention 计算 |
| Indexer.compressor（Indexer 内部） | 128      | k_indexer [S/4, 128]     | 检索用的轻量 key        |

**第二阶段：Indexer（检索）**
Indexer 用当前 query 与全局的 `k_indexer` 计算相关性分数（低维点积，无 softmax，代价低），找出最相关的 top-k 个压缩 token 的位置索引，再用该索引从 `kv_compressor` 中取出对应行，只对这 top-k 个位置做完整 attention。

```
主 Compressor（head_dim=512）：
	原始隐状态 x [S, dim]
		│ wkv + wgate，每 4 token 压缩为 1
		▼
	kv_compressor [S/4, 512] → 实际 attention 用的 KV

Indexer 内部 Compressor（head_dim=128）：
	原始隐状态 x [S, dim]
		│ wkv + wgate，每 4 token 压缩为 1
		▼
	k_indexer [S/4, 128] → 检索用的轻量 key

Indexer 检索：
	q [chunk, head_dim]
		│
	k_indexer [S/4, 128] ← 低维相关性计算（O(S/4 × 128)）
		│
	top-k 位置索引
		│
	用索引从 kv_compressor 中取出 top-k 行
		│
	只对这 k 个位置做完整 attention（O(k × 512)）
```

<mark style="background:#affad1">k_indexer 只用来做"相关性排序"，不需要携带完整语义信息</mark>。维度低（128）→ 检索更快。
<mark style="background:#affad1">kv_compressor 要参与真正的 attention 计算，需要保留完整表征</mark>。维度高（512）→ 保留语义。

核心权衡：**先均匀压缩（S → S/4），再用低维 k_indexer 快速定位最相关的位置，最后只对 top-k 个位置用高维 kv_compressor 做精确计算**，把选择代价从 O(S × 512) 压缩到 O(S/4 × 128)。

### 4.3.2 问题定位：Compressor overlap=True 的 CP 边界问题

C4A Compressor 使用 `overlap=True`，其 `overlap_transform` 使第 i 个压缩 token 依赖第 i-1 个 group 的数据：
```python
new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]    # group i 混入 group i-1 的数据
```

CP 切分后，rank r 的 `compressed_token[0]` 需要 rank r-1 最后一个 group 的数据，但本地看不到，导致错误地填入全 0：
```
非 CP（正确）：  compressed[0 of rank r] = f(group_k,  group_{k-1})  ✓
CP 无处理：      compressed[0 of rank r] = f(group_k,  0)            ✗
```

同理，Indexer 内部的 Compressor（head_dim=128）也使用 `overlap=True`，存在完全相同的问题。

> [!IMPORTANT] rank 0 的 score 初始化必须用 `-inf`，不能用 `0`
> 原 `overlap_transform` 对 score 使用 `value=float("-inf")` 初始化，使得 rank 0 位置 0 的 overlap 槽权重经 softmax 后接近 0（正确行为）。
> 若 `recv_score` 在 rank 0 时为全 0，softmax 后 overlap 槽会得到非零权重，与非 CP 行为不一致，产生训练偏差。
>
> softmax(-inf) = e^(-inf) / Z ≈ 0    ← 几乎无贡献，✓
> softmax(0)    = e^0      / Z = 1/Z  ← 正常的正数权重，✗
>
> 所以需要修复：kv 和 score 分两次 BoundaryExchange，各自使用正确的初始值：
> - `kv`：`init_value=0.0`（与非 CP `overlap_transform value=0` 一致）
> - `score`：`init_value=float("-inf")`（与非 CP `overlap_transform value=-inf` 一致）

### 4.3.3 带 CP 边界修正的 Compressor 前向

对原始压缩逻辑做三处修正：P2P 边界交换、手动 overlap_transform、全局位置 RoPE，其余与原始函数相同。

> 注：此函数同样被 §4.3.4 的 Indexer 前向复用，Indexer 内部 Compressor 的边界问题与此处完全相同。

```python
def compressor_forward_with_cp(
    compressor,          # Compressor 实例（overlap=True）
    x,                   # [B, chunk, D]
    freqs_cis_global,    # 全量 freqs_cis，[max_seq_len, rope_head_dim]
    cp_rank, cp_size, cp_group,
    chunk_size, ratio=4,
):
    B, chunk, D = x.shape
    d = compressor.head_dim
    dtype = x.dtype
    x_f = x.float()

    # ── Step 1：投影 ─────────────────────────────────────────────────────
    kv    = compressor.wkv(x_f)    # [B, chunk, 2*head_dim]
    score = compressor.wgate(x_f)  # [B, chunk, 2*head_dim]

    # ── Step 2：确定有效长度 ────────────────────────
    cutoff = chunk

    # ── Step 3：reshape 为 group 视图 ────────────────────────────────────
    kv_g    = kv.unflatten(1, (-1, ratio))  # [B, chunk//4, 4, 2*d]
    score_g = score.unflatten(1, (-1, ratio)) + compressor.ape  # [B, chunk//4, 4, 2*d]

    # ── Step 4：P2P 边界交换（kv 和 score 分两次，初始化值不同）──────────
    kv_last    = kv_g[:, -1:, :, :d].contiguous()    # [B, 1, 4, d]
    score_last = score_g[:, -1:, :, :d].contiguous() # [B, 1, 4, d]
    
    # kv: rank 0 接收全 0（与非 CP overlap_transform value=0 一致）
    recv_kv = BoundaryExchange.apply(kv_last, cp_rank, cp_size, cp_group, 0.0)
    # score: rank 0 接收全 -inf（与非 CP overlap_transform value=-inf 一致）
    recv_score = BoundaryExchange.apply(score_last, cp_rank, cp_size, cp_group, float("-inf"))
    # recv_kv/recv_score: [B, 1, 4, d]

    # ── Step 5：手动执行 overlap_transform（带 CP 边界修正）────────────────
    s = chunk // ratio
    new_kv    = kv_g.new_full((B, s, 2*ratio, d), 0.)
    new_score = score_g.new_full((B, s, 2*ratio, d), float("-inf"))

    # 正常部分（不涉及跨 rank）
    new_kv[:, :, ratio:]    = kv_g[:, :, :, d:]
    new_score[:, :, ratio:] = score_g[:, :, :, d:]

    # overlap 本地 shift（位置 1 及之后）
    new_kv[:, 1:, :ratio]    = kv_g[:, :-1, :, :d]
    new_score[:, 1:, :ratio] = score_g[:, :-1, :, :d]

    # CP 边界修正（位置 0）
    new_kv[:, 0, :ratio]    = recv_kv[:, 0]    # rank 0 为全 0 ✓（与非 CP 一致）
    new_score[:, 0, :ratio] = recv_score[:, 0] # rank 0 为全 -inf ✓（与非 CP 一致）

    # ── Step 6：压缩 ─────────────────────────────────────────────────────
    out = (new_kv * new_score.softmax(dim=2)).sum(dim=2)  # [B, chunk//4, d]
    out = compressor.norm(out.to(dtype))

    # ── Step 7：RoPE（使用全局位置） ─────────────────────────────────────
    global_start = cp_rank * chunk_size
    freqs_compress = freqs_cis_global[global_start : global_start + cutoff : ratio]
    kv_rot = apply_rotary_emb(out[..., -compressor.rope_head_dim:], freqs_compress)
    out = torch.cat([out[..., :-compressor.rope_head_dim], kv_rot], dim=-1)

    return out   # [B, chunk//4, head_dim]
```

### 4.3.4 Indexer 前向

`indexer_forward_with_cp` 是非 CP 版 `Indexer.forward()` 的 CP 替代实现，负责产出 LiCompute top-k 检索所需的三个输入：`k_indexer`、`q_indexer`、`weights`。

**第一块：生成 k_indexer（有跨 rank 通信）**

Indexer 内部有自己的 Compressor（head_dim=128），对 x 做压缩得到 `k_indexer`。这个内部 Compressor 同样使用 overlap=True，存在与主 Compressor 完全相同的 group 0 overlap 边界问题。解决方式也完全相同——直接复用 §4.3.3 的 `compressor_forward_with_cp`，只是传入的实例换成 `indexer.compressor`。两个 Compressor 实例各自独立地做 BoundaryExchange，互不干扰。压缩完成后对结果做 Hadamard 旋转（`rotate_activation`）。

**第二块：生成 q_indexer 和 weights（纯本地，无通信）**

- `q_indexer`：由 wq_b 对 qr 投影得到，施加全局位置坐标的 RoPE 后做 Hadamard 旋转。纯本地操作。
- `weights`：由 weights_proj 对 x 投影得到。纯本地操作。

`k_indexer` 是 KV 侧的压缩表示，每个 rank 只有局部序列的 k_indexer，需要 AllGather 才能让所有 rank 看到全局序列；而 `q_indexer` 和 `weights` 是 Query 侧的表示，用本地数据配合全局 k_indexer（AllGather 在 §4.3.5 完成）即可，不需要提前聚合。

```python
def indexer_forward_with_cp(
    indexer,             # Indexer 实例
    x, qr,              # x: [B,chunk,D]，qr: [B,chunk,q_lora_rank]
    freqs_cis_global,
    cp_rank, cp_size, cp_group,
    chunk_size, ratio=4,
):
    B, chunk, _ = x.shape

    # ── 上半部分：k_indexer（indexer 内部的 Compressor，overlap=True）───
    k_indexer = compressor_forward_with_cp(
        indexer.compressor, x, freqs_cis_global,
        cp_rank, cp_size, cp_group, chunk_size, ratio,
    )  # [B, chunk//4, index_head_dim]
    k_indexer = rotate_activation(k_indexer)  # Hadamard 旋转

    # ── 下半部分：q_indexer 和 weights（纯本地，不需要通信）────────────
    rd = indexer.rope_head_dim
    q = indexer.wq_b(qr).view(B, chunk, indexer.n_heads, indexer.head_dim)
    q = q.clone()
    q_nope, q_rope = torch.split(q, [indexer.head_dim - rd, rd], dim=-1)

    global_start = cp_rank * chunk_size
    freqs_local = freqs_cis_global[global_start : global_start + chunk]
    q_rope = apply_rotary_emb(q_rope, freqs_local)
    q_indexer = torch.cat([q_nope, q_rope], dim=-1)
    q_indexer = rotate_activation(q_indexer)

    weights = indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads ** -0.5)

    return q_indexer, k_indexer, weights
    # q_indexer: [B, chunk, n_heads, index_head_dim]（本地，不需要 AllGather）
    # k_indexer: [B, chunk//4, index_head_dim]（需要 AllGather）
    # weights:   [B, chunk, n_heads]（本地）
```

### 4.3.5 AllGather（复用公共基础设施）

```python
# 两路 AllGather，反向自动 ReduceScatter（§3.3）
global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
# [B, seq_len//4, head_dim]

global_k_indexer = AllGatherCompressedKV.apply(local_k_indexer, cp_group)
# [B, seq_len//4, index_head_dim]
```

### 4.3.6 LiCompute CP 修正

`li_compute_with_cp` 是非 CP 版 `LiCompute` 的 CP 替代实现，负责用 `q_indexer` 与 `global_k_indexer` 计算相关性分数，并选出每个 query 最相关的 top-k 个压缩 token 的索引（`compress_topk_idxs`）。

**解决什么问题：**

原始 `LiCompute` 在生成因果掩码时，用本地坐标 `base = arange(seqlen)` 表示 query 的位置，即默认所有 query 的位置从 0 开始。CP 切分后，rank r 的 chunk 在全局序列中的起始位置是 `r * chunk`，本地第 i 个 query 的真实全局位置是 `r * chunk + i`。如果仍用本地坐标 0..chunk-1 作为 base，因果掩码会认为 rank r 的 query 只能看到压缩序列前 chunk//4 个 token，而实际上它能看到 `(r+1) * chunk//4` 个，导致大量本来合法的历史压缩 token 被错误屏蔽，top-k 结果严重偏差。

**三处关键修正：**

1. **全局 query 坐标**：`base = cp_rank * chunk_size + arange(chunk_size)`，用全局位置而非本地位置生成因果掩码
2. **top-k 上限收缩**：rank r 实际能看到的压缩 token 最多只有 `(cp_rank+1) * chunk//ratio` 个，取 top-k 时将 k 限制在这个范围内
3. **offset 对齐**：返回的 `compress_topk_idxs` 索引值加上 `offset`（= kv_full 的长度），与 SparseAttention 内部 `cat([kv_full, causal_kv_compress])` 的寻址方式对齐；对不合法位置（未来 token）填 -1

```python
def li_compute_with_cp(
    li_compute,          # LiCompute 实例
    q_indexer,           # [B, chunk, n_heads, index_head_dim]（本地）
    global_k_indexer,    # [B, seq_len//4, index_head_dim]（AllGather 后）
    weights,             # [B, chunk, n_heads]（本地）
    chunk_size, seq_len, cp_rank, ratio, offset,
):
    # 计算 attention 分数
    index_score = torch.einsum("bshd,btd->bsht", q_indexer, global_k_indexer)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    # index_score: [B, chunk, seq_len//4]

    # ── CP 修正：base 使用全局坐标 ──────────────────────────────────────
    device = index_score.device
    base = (cp_rank * chunk_size
            + torch.arange(chunk_size, device=device)).unsqueeze(1)  # [chunk, 1]

    # matrix 扩展到全局压缩长度
    matrix = torch.arange(seq_len // ratio, device=device).unsqueeze(0)  # [1, seq_len//4]

    causal_mask = matrix >= (base + 1) // ratio   # [chunk, seq_len//4]
    index_score = index_score + torch.where(
        causal_mask, torch.finfo(q_indexer.dtype).min, 0.
    )

    # topk 上限：rank r 最多能看到 (cp_rank+1)*chunk//ratio 个压缩 token
    max_valid = (cp_rank + 1) * chunk_size // ratio
    k = min(li_compute.index_topk, max_valid)
    index_score, topk_idxs = index_score.topk(k, dim=-1)

    # 因果检查 + 加 offset
    mask = topk_idxs >= (base + 1) // ratio
    compress_topk_idxs = torch.where(mask, -1, topk_idxs + offset)

    return compress_topk_idxs, index_score
```

### 4.3.7 整体执行流程

起点：每个 rank 持有本地隐状态 x [B, chunk, D]，chunk = seq_len / cp_size。

第一步：Q 投影（本地）
	对 x 做两次线性投影（wq_a → wq_b），得到主 attention 用的 query q [B, chunk, n_heads, head_dim]。施加 RoPE 时用全局位置坐标（rank r 的起始位置是 r * chunk）。q 投影完之后一直等到最后 SparseAttention 才被使用。

第二步：Window KV 投影 + P2P 边界通信（本地投影 + 跨 rank 通信）
	对 x 做 wkv 投影得到本地 kv [B, chunk, head_dim]，施加全局位置 RoPE。
	投影完成后用 BoundaryExchange 做 P2P 通信：rank r-1 把自己末尾 128 个 token 的 kv 发给 rank r，rank r 收到后拼接到自己 kv 的前面，得到 kv_full [B, 128+chunk, head_dim]。rank 0 没有前驱，kv_full 就等于本地 kv。
	拼接之后用 cp_window_topk_idxs 重新生成窗口注意力索引。

第三步：Indexer 前向（本地 + 跨 rank 通信，不传梯度）
- 这一步并行产出三个结果：
	- k_indexer_local [B, chunk//4, 128]：由 Indexer 内部的 Compressor（head_dim=128）对 x 压缩得到。压缩时同样有 overlap 边界问题，同样通过两次 BoundaryExchange（kv init=0，score init=-inf）+ 手动 overlap_transform 修正。
	- q_indexer [B, chunk, n_heads, 128]：由 wq_b 对 qr 投影得到，纯本地，施加全局位置 RoPE。
	- weights [B, chunk, n_heads]：由 weights_proj 对 x 投影得到，纯本地。

第四步：AllGather k_indexer
	各 rank 的 k_indexer_local 通过 AllGatherCompressedKV 拼接成 global_k_indexer [B, seq_len//4, 128]，每个 rank 都拿到完整序列的检索 key。反向时自动变成 ReduceScatter。

第五步：LiCompute top-k 选择
- 用本地 q_indexer 与 global_k_indexer 做点积，得到每个 query 对所有压缩 token 的相关性分数，再乘以 weights 聚合各头。
- 关键修正：生成因果掩码时用 rank r 的全局 query 坐标，屏蔽掉未来位置的压缩 token。
- 最终 top-k 得到 compress_topk_idxs，索引以 kv_full 的长度为 offset。

第六步：主 Compressor 前向（本地 + 跨 rank 通信）
- 对 x 做 wkv + wgate 投影，reshape 成 group 视图后，做两次 BoundaryExchange：
	- kv：rank r-1 最后一个 group 的前 d 维发给 rank r，rank 0 收到全 0
	- score：同上，rank 0 收到全 -inf
- 然后手动执行 overlap_transform，将收到的跨 rank 数据填入 group 0 的 overlap 槽，其余 group 的 overlap 槽用本地前一个 group 的数据填入。
- 之后做 softmax 加权求和压缩，施加全局位置 RoPE，得到 local_kv_compress [B, chunk//4, 512]。

第七步：AllGather kv_compress + 因果裁剪
- 各 rank 的 local_kv_compress 通过 AllGatherCompressedKV 拼成 global_kv_compress [B, seq_len//4, 512]。
- 然后因果裁剪：rank r 只保留前 (r+1) * chunk//4 个，去掉未来 rank 的数据，得到 causal_kv_compress。

第八步：SparseAttention
- 用第一步的 q，同时做两路 attention：
	- Window attention：q 对 kv_full 做注意力，用修正后的 window_topk_idxs 确定每个 query 访问 kv_full 中的哪 128 个位置
	- Compress attention：q 对 causal_kv_compress 做注意力，只访问 compress_topk_idxs 指定的 top-k 行
- 两路结果合并输出 o [B, chunk, head_dim]。
- LiLoss（辅助损失）由外层 TransformerBlock.cal_index_loss 计算，与非 CP 流程完全相同。

### 4.3.8 完整前向代码

```python
def c4a_attention_forward_with_cp(
    pre_attn,             # PreAttention 实例（含 wq_a/q_norm/wq_b/wkv/kv_norm/compressor/indexer）
    inner_attn,           # InnerAttention 实例（含 attn_sink/sparse_attn/li_compute）
    x,                    # [B, chunk, D]
    freqs_cis_global,     # 全量 freqs_cis（compress_rope_theta），供 Q/KV/Compressor/Indexer 共用
    attention_masks,
    cp_rank, cp_size, cp_group,
    chunk_size, seq_len,
    ratio=4,
    window_size=128,
):
    B, chunk, D = x.shape
    rd = pre_attn.rope_head_dim
    global_start = cp_rank * chunk_size
    freqs_local = freqs_cis_global[global_start : global_start + chunk]

    # ── Q 投影（纯本地）────────────────────────────────────────────────
    qr = pre_attn.q_norm(pre_attn.wq_a(x))
    q  = pre_attn.wq_b(qr).unflatten(-1, (pre_attn.n_heads, pre_attn.head_dim))
    q  = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + pre_attn.eps)
    q_nope, q_rope = torch.split(q, [pre_attn.head_dim - rd, rd], dim=-1)
    q_rope = apply_rotary_emb(q_rope, freqs_local)
    q = torch.cat([q_nope, q_rope], dim=-1)   # [B, chunk, n_heads, head_dim]

    # ── Window KV 投影（纯本地）───────────────────────────────────────────
    kv = pre_attn.kv_norm(pre_attn.wkv(x))
    kv_nope, kv_rope = torch.split(kv, [pre_attn.head_dim - rd, rd], dim=-1)
    kv_rope = apply_rotary_emb(kv_rope, freqs_local)
    kv = torch.cat([kv_nope, kv_rope], dim=-1)   # [B, chunk, head_dim]

    # ── Window KV 边界通信（BoundaryExchange §3.2，含反向梯度）──────────
    boundary_kv = BoundaryExchange.apply(
        kv[:, -window_size:, :].contiguous(),
        cp_rank, cp_size, cp_group, 0.0   # init_value=0.0，rank 0 收到全 0
    )
    if cp_rank > 0:
        kv_full = torch.cat([boundary_kv, kv], dim=1)  # [B, 128+chunk, head_dim]
    else:
        kv_full = kv                                     # [B, chunk, head_dim]

    # ── Window topk 索引（CP 修正，§3.4）────────────────────────────────
    window_topk = cp_window_topk_idxs(B, chunk, window_size, cp_rank)  # [B, chunk, 128]
    offset = kv_full.size(1)  # rank 0: chunk；rank r>0: 128+chunk

    # ── Indexer（detach，不传梯度）─────────────────────────────────────
    with torch.no_grad():
        q_indexer, k_indexer_local, weights = indexer_forward_with_cp(
            pre_attn.indexer, x.detach(), qr.detach(), freqs_cis_global,
            cp_rank, cp_size, cp_group, chunk_size, ratio,
        )

    # ── AllGather k_indexer（§4.3.5）──────────────────────────────────
    global_k_indexer = AllGatherCompressedKV.apply(k_indexer_local, cp_group)
    # [B, seq_len//4, index_head_dim]

    # ── LiCompute：top-k 选择（全局坐标修正，§4.3.6）─────────────────
    compress_topk_idxs, index_score = li_compute_with_cp(
        inner_attn.li_compute, q_indexer, global_k_indexer, weights,
        chunk_size, seq_len, cp_rank, ratio, offset,
    )
    # compress_topk_idxs: [B, chunk, topk]，索引从 offset=kv_full.size(1) 起

    # ── 主 Compressor（overlap=True，需 BoundaryExchange，§4.3.3）───────
    local_kv_compress = compressor_forward_with_cp(
        pre_attn.compressor, x, freqs_cis_global,
        cp_rank, cp_size, cp_group, chunk_size, ratio,
    )   # [B, chunk//4, head_dim]

    # ── AllGather kv_compress（§4.3.5）────────────────────────────────
    global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
    # [B, seq_len//4, head_dim]

    # ── 因果裁剪（去掉未来 rank 的数据）──────────────────────────────
    valid_len = (cp_rank + 1) * chunk_size // ratio
    causal_kv_compress = global_kv_compress[:, :valid_len, :]
    # [B, (cp_rank+1)*chunk//4, head_dim]

    # ── Sparse Attention（传入修正后的 window_topk_idxs，§2.2.2）───────
    o = inner_attn.sparse_attn(
        q, kv_full, inner_attn.attn_sink, causal_kv_compress,
        compress_topk_idxs=compress_topk_idxs,
        window_topk_idxs=window_topk,   # 外部修正的窗口索引，覆盖内部 get_window_topk_idxs
    )

    # LiLoss 由外层 TransformerBlock.cal_index_loss 计算，与非 CP 流程完全相同
    # 返回值对齐非 CP 的 Attention.forward() 输出，供 TransformerBlock 使用
    return (
        o, compress_topk_idxs, offset, q,
        causal_kv_compress, attention_masks,
        index_score, q_indexer, global_k_indexer, weights,
    )
```
