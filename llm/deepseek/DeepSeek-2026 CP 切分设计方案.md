
DeepSeek-v4 CP 切分设计方案

---

# 一、概述

DeepSeek-v4 的每个 Transformer Block 包含两大组件：**MHC（Multi-Head Hyper-Connections）** 和 **Attention**。
CP 切分的挑战主要来自 Attention，不同层的 compress_ratio 决定了序列依赖的范围和通信开销。
`compress_ratios` 的模式是 `(1, 1, 4, 128, 4, 128, ..., 4, 128, 4)`，共三类 Attention 层：

| 层类型                | 序列依赖范围                        | 通信挑战                 |
| ------------------ | ----------------------------- | -------------------- |
| ratio = 1（Window）  | 局部 window_size = 128          | 仅边界 token 通信         |
| ratio = 128（C128A） | 全局（但 token 数极少）               | 压缩 KV 的全局 all-gather |
| ratio = 4（C4A）     | 全局（Lightning Indexer 选 top-k） | 压缩 KV 的全局 all-gather |

**ratio = 1（Window Attention）**：每个 query 只向左看 `window_size=128` 个原始 token，依赖范围纯局部。CP 切分后，rank r 最前面的若干 query 的窗口会跨越到 rank r-1 的末尾，因此只需要一次轻量的 P2P 通信，从前驱 rank 借来末尾 `window_size - 1 = 127` 个 token 的 KV 即可，通信量极小（约 131 KB/层）。

**ratio = 128（C128A）**：每 128 个原始 token 被 Compressor 压缩为 1 个压缩 token，全局压缩序列极短（例如 seqlen=65536 时仅 512 个压缩 token）。Compressor 使用 `overlap=False`，无跨 group 依赖，不需要 P2P 边界修正。AllGather 的通信量相比 C4A 缩小 32 倍（约 0.5 MB/层），几乎可以忽略。

**ratio = 4（C4A）**：每 4 个原始 token 被 Compressor 压缩为 1 个压缩 token，再由 Lightning Indexer 从全局压缩序列中检索 top-k 个最相关的压缩 token 参与 attention。CP 切分后必须通过 AllGather 将各 rank 的本地压缩 KV 汇聚成全局视图。此外，C4A 的 Compressor 使用 `overlap=True`，相邻压缩 token 之间存在跨 group 依赖，CP 边界处还需要额外的 P2P 边界修正。

**关于 MHC（Multi-Head Hyper-Connections）**
> [!NOTE] 模型用的是 MHC，不是单头 HC
> 代码中 `hc_mult=4`，每个 token 的隐藏状态形状为 `[B, S, 4, D]`，即每个位置
> 维护 4 个并行 stream，这是 MHC（Multi-Head Hyper-Connections）而非单头 HC。

MHC 是纯 **per-token** 的局部操作：HcPre / HcPost 的 Sinkhorn 路由只在同一 token 的 `hc_mult=4` 个 stream 之间做加权混合，**不涉及任何跨 token 的序列交互**。CP 将序列按 token 切分，同一 token 的 4 个 stream 始终在同一 rank 上，因此 MHC **不需要任何 CP 通信**，对 CP 完全透明。

---

# 二、前提与工程修改

## 2.0 序列切分方式：必须使用顺序切分

DeepSeek-V4 CP **只支持顺序切分**（rank r 持有全局序列 `[r*chunk, (r+1)*chunk)`），不支持 HeadTail 交错切分。三条 CP 路径的全局位置计算均依赖：

```python
global_start = cp_rank * chunk_size   # _c1a / _c128a / _c4a 三处均如此
```

HeadTail 切分下每个 rank 持有序列的头尾两段，global_start 不再等于 `cp_rank * chunk_size`，会导致 RoPE 位置错误、因果掩码错误，以及 BoundaryExchange 语义错误。

`cp_input_sharding.py` 通过 monkey-patch 强制执行此约束：

```python
# cp_input_sharding.py — 每次 post_dataloading_process 前临时覆盖
parallelism_cfg.context_parallel_load_balancer = None  # None = 顺序切分
```

**运行条件**：训练配置中 `context_parallel_load_balancer` 的实际生效值必须为 `None`；此 patch 已在入口处自动注册，无需手动配置，但任何绕过 patch 的路径都会导致静默的计算错误。

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

### 2.2.2 SparseAttention：增加 window_topk_idxs 参数并保存 _original_forward

原 `SparseAttention.forward()`（`model.py:608`）始终调用 `get_window_topk_idxs(window_size, bsz, seqlen)` 生成本地坐标索引（seqlen = chunk）。对 rank r > 0，`kv_full = [boundary_kv(127) || local_kv(chunk)]` 大小为 127+chunk，但生成的索引范围 `0..chunk-1` 无法访问前 127 个边界 token，窗口 attention 结果错误。

**修改内容**：

1. `SparseAttention.forward()` 增加可选参数 `window_topk_idxs=None`；非 CP 路径不传，行为不变。
2. 在类定义末尾增加 `_original_forward = forward`，保存原始 SDPA 实现。`deepseek_v4_sfa` converter 随后将 `forward` 替换为 `sdpa_to_sfa_adapter`，但 `_original_forward` 已在替换前绑定，之后永久可用作回退路径。

```python
def forward(
    self,
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    attn_sink: torch.Tensor,
    kv_compress: torch.Tensor | None = None,
    compress_topk_idxs: torch.Tensor | None = None,
    window_topk_idxs: torch.Tensor | None = None,   # 新增：外部传入修正后的窗口索引
):
    ...

# 类定义末尾（converter 替换 forward 之前已绑定）
_original_forward = forward
```

**实际路由**：CP forward 函数（`_c1a_forward_with_cp` 等）调用 `inner_attn.sparse_attn()` 时**不显式传 `window_topk_idxs`**，窗口修正完全由 `sdpa_to_sfa_adapter` 内部完成：

```
非 CP / cp_rank=0：               → SFA kernel（band mode，正确）✅
cp_rank>0, compress_ratio ≠ 4：   → 内部计算 window_topk=[i..i+127]，调用 _original_forward ✅
cp_rank>0, compress_ratio == 4：  → SFA dummy-padding，band mode 自然覆盖正确窗口 ✅
```

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
		
		# 使用 group_dst/group_src（组内 local rank），避免多维 mesh 下全局 rank 映射错误
		if rank < cp_size - 1:
			reqs.append(dist.isend(send_tensor.contiguous(), group=group, group_dst=rank+1))
		if rank > 0:
			reqs.append(dist.irecv(recv_buf, group=group, group_src=rank-1))
		for req in reqs: req.wait()
		return recv_buf
		
	@staticmethod
	def backward(ctx, grad_recv):
		grad_send = torch.zeros_like(grad_recv)
		reqs = []
		if ctx.rank > 0:
			reqs.append(dist.isend(grad_recv.contiguous(), group=ctx.group, group_dst=ctx.rank-1))
		if ctx.rank < ctx.cp_size - 1:
			reqs.append(dist.irecv(grad_send, group=ctx.group, group_src=ctx.rank+1))
			
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

CP 切分后，kv_full = [boundary_kv(127) || local_kv(chunk)]（注意：边界 token 数为 `window_size - 1 = 127`，不是 128），原始 `GetWindowTopkIdxs` 基于本地坐标 `0..chunk-1` 生成的索引无法访问边界 token，必须重新计算。

> [!IMPORTANT] 边界 token 数为 127（`window_size - 1`），不是 128
> 发送的边界为 `kv[:, -(window_size-1):, :]`，即末尾 127 个 token。
> 这样 kv_full 大小为 `127 + chunk`，每个 rank 的实现保持一致。

**rank 0**（无前驱）：
- `kv_full = local_kv`，形状 `[B, chunk, head_dim]`
- `window_topk` = 原 `GetWindowTopkIdxs(window_size=128, bsz, chunk)` 结果（不变）

**rank r > 0**（有边界 token）：
- `kv_full = cat([boundary_kv, local_kv], dim=1)`，形状 `[B, 127+chunk, head_dim]`
- `kv_full[0:127]` = rank r-1 的末尾 127 个全局 token（全局位置 [r*chunk-127, r*chunk-1]）
- `kv_full[127+j]` = rank r 的本地 token j（全局位置 r*chunk+j）
- 对 query i（全局位置 r*chunk+i），所需 KV 全局窗口 [r*chunk+i-127, r*chunk+i] 映射到 kv_full：
  - r*chunk+i-127 → kv_full[i]
  - r*chunk+i     → kv_full[i+127]
  - 即 `window_topk[i] = [i, i+1, ..., i+127]`，恰好 128 个，无需 clamp

```python
def cp_window_topk_idxs(bsz: int, chunk: int, window_size: int, cp_rank: int, device=None):
    """
    为 CP 模式生成 window attention 的 topk 索引。
    rank 0 无边界 token，沿用原逻辑；rank > 0 有 (window_size-1)=127 个边界 token。
    """
    if cp_rank == 0:
        # kv_full = local_kv [B, chunk, D]，原逻辑不变
        base = torch.arange(chunk, device=device).unsqueeze(1)
        window_topk = (base - window_size + 1).clamp(0) + torch.arange(window_size, device=device)
        window_topk = torch.where(window_topk > base, -1, window_topk)
    else:
        # kv_full = [boundary(127) || local_kv(chunk)] [B, 127+chunk, D]
        # query i → kv_full 位置 [i, i+1, ..., i+127]
        base = torch.arange(chunk, device=device).unsqueeze(1)
        window_topk = base + torch.arange(window_size, device=device)  # [chunk, 128]
        # 合法性：i ∈ [0, chunk), 最大取值 chunk-1+127 = chunk+126 ≤ 127+chunk-1 ✓

    return window_topk.unsqueeze(0).expand(bsz, -1, -1)
```

> [!NOTE] 窗口索引修正的实际位置
> `cp_window_topk_idxs` 定义在 `deepseek_v4_cp.py` 中，但实际上不由 CP forward 函数
> 显式传入 `sparse_attn`。窗口修正是在 `sdpa_to_sfa_adapter`（`deepseek_v4_sfa.py`）
> 内部完成的：对 cp_rank>0 且 compress_ratio≠4 的层，`sdpa_to_sfa_adapter` 会自动
> 计算 `window_topk=[i..i+127]` 并调用 `_original_forward`，无需 CP forward 函数感知。

拼接完 window_topk_idxs 后，compress 相关的索引 offset 也需随之调整：
- rank 0：`offset = chunk`（与原来相同）
- rank r > 0：`offset = 127 + chunk`（kv_full 多了 127 个边界 token）

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
- rank r-1 把自己末尾 **127 个** token 的 kv（`kv[:, -(window_size-1):, :]`）通过 BoundaryExchange 发给 rank r，rank r 接收后拼接：
  - rank r > 0：kv_full = cat([boundary_kv, local_kv]) → [B, 127+chunk, head_dim]
  - rank 0：无前驱，kv_full = local_kv → [B, chunk, head_dim]
- rank 0 通过 `kv = kv + boundary_kv.sum(...) * 0.0` 建立对 boundary_kv 的 dummy 依赖，确保 BoundaryExchange.backward 被 autograd 调用（见 §5.5）

第四步：Attention 计算
	用 q 对 kv_full 做 window attention。rank r>0 时 `sdpa_to_sfa_adapter` 内部自动生成修正后的 `window_topk=[i..i+127]` 并调用 Python SDPA；rank 0 时走 SFA band mode（自然正确）。输出 o [B, chunk, head_dim]。

> [!NOTE] 为什么发送 127 个边界 token 而非 128
> Window Attention 中，每个 query 可以向左看 window_size=128 个原始 token。
> ```
> rank r 的第 0 个 query（全局位置 r*chunk）：
>   需要 KV 来自 [r*chunk - 127, r*chunk - 1]  ← 这 127 个在 rank r-1 上
>   加上自身 kv_full[127]（local token 0）共 128 个 ✓
>
> rank r 的第 126 个 query（全局位置 r*chunk + 126）：
>   需要 KV 来自 [r*chunk - 1, r*chunk + 126]
>   = kv_full[126..253] ← 仍在边界+本地范围内 ✓
>
> rank r 的第 127 个 query（全局位置 r*chunk + 127）：
>   需要 KV 来自 [r*chunk, r*chunk + 127] = kv_full[127..254] ← 全在本地 ✓
> ```
> 因此只需发送 window_size-1 = 127 个边界 token（而非 128）即可覆盖所有 query 的 window。
> 通信量：127 × 512 × 2B ≈ 130KB/层（极小）

### 4.1.2 Window KV 边界通信代码

**使用 `BoundaryExchange`（§3.2）而非普通 isend/irecv**：Window KV 的边界 token 参与 attention 计算后产生梯度，该梯度必须流回 rank r-1 的 `wkv` 参数。普通 `dist.isend/irecv` 不在 autograd 计算图中，梯度会被静默丢弃。`BoundaryExchange` 已封装好前向通信与反向梯度回传：

```python
q, kv, _, _, _, _, _ = pre_attn(x, freqs_local, None)
# q:  [B, chunk, n_heads, head_dim]
# kv: [B, chunk, head_dim]

# Window KV 边界交换：发送末尾 127 个 token（window_size - 1）
boundary_kv = BoundaryExchange.apply(
    kv[:, -(window_size - 1):, :].contiguous(),  # [B, 127, head_dim]
    cp_rank, cp_size, cp_group, 0.0,
)
# rank r > 0：boundary_kv 来自 rank r-1 的最后 127 个 token
# rank 0：    boundary_kv = 全 0 缓冲区（无前驱）

if cp_rank == 0:
    # 建立 dummy 依赖，防止 autograd 剪枝 BoundaryExchange 节点（见 §5.5）
    kv = kv + boundary_kv.sum(dim=(1, 2), keepdim=True) * 0.0
kv_full = torch.cat([boundary_kv, kv], dim=1) if cp_rank > 0 else kv
# rank r > 0：[B, 127+chunk, head_dim]；rank 0：[B, chunk, head_dim]

# sparse_attn 无需传 window_topk_idxs；sdpa_to_sfa_adapter 内部对 rank>0 自动
# 计算 window_topk=[i..i+127] 并走 _original_forward（Python SDPA）
o = inner_attn.sparse_attn(q, kv_full, inner_attn.attn_sink)
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
- 与 4.1 节相同：用 `BoundaryExchange.apply()` 从 rank r-1 获取末尾 **127 个** window_kv token：
  - rank r > 0：kv_full = cat([boundary_kv, window_kv]) → [B, 127+chunk, head_dim]
  - rank 0：kv_full = window_kv → [B, chunk, head_dim]（含 dummy 依赖）
- 更新 offset：
  - rank 0：offset = chunk；rank r > 0：offset = 127+chunk

第五步：AllGather kv_compress
	各 rank 的 local_kv_compress 通过 `AllGatherCompressedKV`（§3.3）拼成 global_kv_compress [B, seq_len//128, 512]，每个 rank 拿到完整序列的压缩 KV。反向时自动变成 ReduceScatter。

第六步：因果裁剪
	从 global_kv_compress 中只保留 rank 0 到 rank r 的部分，去掉未来 rank 的数据，得到 causal_kv_compress [B, (cp_rank+1)*chunk//128, 512]。

第七步：生成修正后的 compress_topk_idxs
	C128A 不像 C4A 有 Lightning Indexer，top-k 索引直接用位置关系生成。关键修正：用 rank r 的全局 query 坐标作为 base（而非从 0 开始的本地坐标），生成因果掩码，确保每个 query 只访问自己位置之前的压缩 token。索引值以 kv_full 长度为 offset，与 SparseAttention 的寻址方式对齐。

第八步：SparseAttention
- 用 q 同时做两路 attention：
  - Window attention：q 对 kv_full 做注意力
  - Compress attention：q 对 causal_kv_compress 做注意力，使用修正后的 compress_topk_idxs
- 两路结果合并，输出 o [B, chunk, head_dim]
- **注**：C128A 不传 `window_topk_idxs`；`sdpa_to_sfa_adapter` 对 rank>0 自动计算 `window_topk=[i..i+127]` 并走 `_original_forward`（见附录 B）；rank 0 走 SFA band mode

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
def _get_c128a_compress_topk_idxs(
    bsz: int,
    chunk: int,
    ratio: int,  # = 128
    cp_rank: int,
    causal_kv_len: int,  # = (cp_rank + 1) * chunk // ratio
    offset: int,         # = kv_full.size(1)，对 causal_kv_compress 的索引偏移
    device=None,
) -> torch.Tensor:  # [B, chunk, causal_kv_len]
    """
    为 C128A 在 CP 模式下生成 compress_topk_idxs。
    query i（全局位置 cp_rank*chunk+i）最多看到全局压缩位置
    j < (cp_rank*chunk + i + 1) // ratio 的压缩 token。
    返回值以 offset 为基，与 SparseAttention 内部 cat([kv_full, kv_compress]) 的寻址对齐。
    """
    global_base = cp_rank * chunk + torch.arange(1, chunk + 1, device=device)  # [chunk]
    compress_pos = torch.arange(causal_kv_len, device=device)                   # [causal_kv_len]
    causal_mask = compress_pos.unsqueeze(0) >= (global_base // ratio).unsqueeze(1)
    compress_topk = torch.where(causal_mask, -1, compress_pos + offset)
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
def _c128a_forward_with_cp(
    pre_attn, inner_attn, post_attn,
    x: torch.Tensor,                 # [B, chunk, D]
    freqs_cis_global: torch.Tensor,  # 全量 freqs_cis（compress_rope_theta）
    attention_masks,
    cp_rank: int, cp_size: int, cp_group,
    chunk_size: int, seq_len: int,
    window_size: int = 128,
):
    B, chunk, _ = x.shape
    ratio = 128
    global_start = cp_rank * chunk_size
    freqs_local = freqs_cis_global[global_start : global_start + chunk].to(x.device)

    # ── 1. Q / Window KV / Compressor（统一通过 pre_attn 完成）────────────
    # pre_attn.forward() 内部依次做：wq_a/q_norm/wq_b → q；wkv/kv_norm → kv；
    # compressor_128(x, freqs) → kv_compress（ratio=128 时 overlap=False，纯本地）
    q, local_window_kv, local_kv_compress, _, _, _, _ = pre_attn(x, freqs_local, None)
    # q:               [B, chunk, n_heads, head_dim]
    # local_window_kv: [B, chunk, head_dim]
    # local_kv_compress:[B, chunk//128, head_dim]

    # ── 2. Window KV P2P 边界通信（BoundaryExchange §3.2）────────────────
    boundary_kv = BoundaryExchange.apply(
        local_window_kv[:, -(window_size - 1):, :].contiguous(),  # [B, 127, D]
        cp_rank, cp_size, cp_group, 0.0,
    )
    if cp_rank == 0:
        local_window_kv = local_window_kv + boundary_kv.sum(dim=(1, 2), keepdim=True) * 0.0
    kv_full = (
        torch.cat([boundary_kv, local_window_kv], dim=1) if cp_rank > 0 else local_window_kv
    )
    # rank r>0: [B, 127+chunk, D]；rank 0: [B, chunk, D]
    offset = kv_full.size(1)  # rank 0: chunk；rank r>0: 127+chunk

    # ── 3. AllGather 压缩 KV（前向 AllGather，反向自动 ReduceScatter，§3.3）─
    global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
    # [B, seq_len//128, head_dim]

    # ── 4. 因果裁剪──────────────────────────────────────────────────────
    valid_compress_len = (cp_rank + 1) * (chunk // ratio)
    causal_kv_compress = global_kv_compress[:, :valid_compress_len, :]

    # ── 5. compress_topk_idxs（全局坐标修正，§4.2.4）──────────────────────
    compress_topk = _get_c128a_compress_topk_idxs(
        B, chunk, ratio, cp_rank, valid_compress_len, offset, device=x.device,
    )
    # [B, chunk, valid_compress_len]，索引 = compress_pos + offset

    # ── 6. SparseAttention──────────────────────────────────────────────
    # window_topk_idxs 不显式传入；sdpa_to_sfa_adapter 对 rank>0 内部生成
    # window_topk=[i..i+127] 并走 _original_forward（Python SDPA，见附录 B）
    o = inner_attn.sparse_attn(
        q, kv_full, inner_attn.attn_sink, causal_kv_compress,
        compress_topk_idxs=compress_topk,
    )

    n_local_groups = post_attn.n_groups // (pre_attn.n_heads // q.shape[2])
    x_out = post_attn(o, freqs_local, B, chunk, n_local_groups)
    return (x_out, None, offset, q, causal_kv_compress, attention_masks, None, None, None, None)
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

> 注：Indexer 内部的 Compressor（head_dim=128）存在类似的边界问题，但实现中**不复用**此函数——直接调用本地非 CP 版（见 §4.3.4）。

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

**第一块：生成 k_indexer（本地，无跨 rank 通信）**

Indexer 内部有自己的 Compressor（head_dim=128），对 x 做压缩得到 `k_indexer`。虽然这个内部 Compressor 也使用 overlap=True（理论上存在边界问题），但实现中**直接调用原始非 CP 版 `indexer.compressor(x, freqs_local)`**，而非 `compressor_forward_with_cp`。

原因：若对 Indexer 的 Compressor 也做 `BoundaryExchange`，其 backward 与主 KV 的 `BoundaryExchange.backward` 会在相同的 cp_group 上并发 P2P 通信，在 HCCL 内部抢占同一 communicator 的缓冲区/stream，导致梯度被错误混合（tag 碰撞）。使用本地版本相当于对 group 0 的 overlap slot 填零（与非 CP rank 0 行为类似），引入的误差对辅助 LiLoss 可接受。压缩完成后对结果做 Hadamard 旋转（`rotate_activation`）。

**第二块：生成 q_indexer 和 weights（纯本地，无通信）**

- `q_indexer`：由 wq_b 对 qr 投影得到，施加全局位置坐标的 RoPE 后做 Hadamard 旋转。纯本地操作。
- `weights`：由 weights_proj 对 x 投影得到。纯本地操作。

`k_indexer` 是 KV 侧的压缩表示，每个 rank 只有局部序列的 k_indexer，需要 AllGather 才能让所有 rank 看到全局序列；而 `q_indexer` 和 `weights` 是 Query 侧的表示，用本地数据配合全局 k_indexer（AllGather 在 §4.3.5 完成）即可，不需要提前聚合。

```python
def indexer_forward_with_cp(
    indexer,             # Indexer 实例
    x, qr,              # x: [B,chunk,D]，qr: [B,chunk,q_lora_rank]（均已 detach）
    freqs_cis_global,
    hadamard_mat,        # 模型级别的 hadamard_mat，传给 rotate_activation
    cp_rank, cp_size, cp_group,
    chunk_size, ratio=4,
):
    B, chunk, _ = x.shape
    global_start = cp_rank * chunk_size
    freqs_local = freqs_cis_global[global_start : global_start + chunk].to(x.device)

    # ── k_indexer：直接调用非 CP compressor（纯本地，backward 无 P2P）───
    # 原因：若用 compressor_forward_with_cp，其 BoundaryExchange.backward 与
    # 主 KV 的 BoundaryExchange.backward 在同一 cp_group 上并发，引发 HCCL tag 碰撞。
    k_indexer = indexer.compressor(x, freqs_local)  # [B, chunk//4, index_head_dim]
    k_indexer = rotate_activation(k_indexer, hadamard_mat)

    # ── q_indexer 和 weights（纯本地）──────────────────────────────────
    rd = indexer.rope_head_dim
    q = indexer.wq_b(qr).view(B, chunk, indexer.n_heads, indexer.head_dim).clone()
    q_nope, q_rope = torch.split(q, [indexer.head_dim - rd, rd], dim=-1)
    q_rope = apply_rotary_emb(q_rope, freqs_local)
    q_indexer = rotate_activation(torch.cat([q_nope, q_rope], dim=-1), hadamard_mat)

    weights = indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads**-0.5)

    return q_indexer, k_indexer, weights
    # q_indexer: [B, chunk, n_heads, index_head_dim]（本地）
    # k_indexer: [B, chunk//4, index_head_dim]（需要 AllGather）
    # weights:   [B, chunk, n_heads]（本地）
```

### 4.3.5 AllGather（复用公共基础设施）

```python
# k_indexer 使用独立的 cp_group_indexer（相同 ranks 但独立 PG 句柄）
# 原因：kv_compress 和 k_indexer 的 AllGather 分别在 cp_group / cp_group_indexer 上；
# 它们的 backward ReduceScatter 顺序因 rank 而异（causal 裁剪使每个 rank 返回不同量），
# 若共用同一 PG，两个 ReduceScatter 会争抢同一 HCCL communicator 的内部缓冲区，导致梯度损坏。
global_k_indexer = AllGatherCompressedKV.apply(k_indexer_local, cp_group_indexer)
# [B, seq_len//4, index_head_dim]

global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
# [B, seq_len//4, head_dim]
```

### 4.3.6 LiCompute CP 修正

`li_compute_with_cp` 是非 CP 版 `LiCompute` 的 CP 替代实现，负责用 `q_indexer` 与 `global_k_indexer` 计算相关性分数，并选出每个 query 最相关的 top-k 个压缩 token 的索引（`compress_topk_idxs`）。

**解决什么问题：**

原始 `LiCompute` 在生成因果掩码时，用本地坐标 `base = arange(seqlen)` 表示 query 的位置，即默认所有 query 的位置从 0 开始。CP 切分后，rank r 的 chunk 在全局序列中的起始位置是 `r * chunk`，本地第 i 个 query 的真实全局位置是 `r * chunk + i`。如果仍用本地坐标 0..chunk-1 作为 base，因果掩码会认为 rank r 的 query 只能看到压缩序列前 chunk//4 个 token，而实际上它能看到 `(r+1) * chunk//4` 个，导致大量本来合法的历史压缩 token 被错误屏蔽，top-k 结果严重偏差。

**三处关键修正：**

1. **全局 query 坐标**：`base = cp_rank * chunk_size + arange(chunk_size)`，用全局位置而非本地位置生成因果掩码
2. **top-k 上限收缩**：rank r 实际能看到的压缩 token 最多只有 `(cp_rank+1) * chunk//ratio` 个，取 top-k 时将 k 限制在这个范围内
3. **返回 offset-free 索引**：`compress_topk_idxs` 直接指向 `causal_kv_compress[0, max_valid)`，不含 offset。这样对 `cal_index_loss`（`li_loss_adapter`）和 SFA kernel（`cmp_kv` 直接索引）均正确。C128A 的 `_original_forward` 会在内部将 `kv_full ‖ kv_compress` 拼接，此时调用者需叠加 offset——但 C4A 不走 `_original_forward`，直接由 SFA 的 `cmp_kv` 直接寻址，offset-free 正确。

```python
def li_compute_with_cp(
    li_compute,          # LiCompute 实例
    q_indexer,           # [B, chunk, n_heads, index_head_dim]（本地）
    global_k_indexer,    # [B, seq_len//4, index_head_dim]（AllGather 后）
    weights,             # [B, chunk, n_heads]（本地）
    chunk_size, seq_len, cp_rank, ratio,
    # 注意：无 offset 参数；返回 offset-free 索引 [0, max_valid)
):
    index_score = torch.einsum("bshd,btd->bsht", q_indexer, global_k_indexer)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    # index_score: [B, chunk, seq_len//ratio]

    device = index_score.device
    base = (
        cp_rank * chunk_size + torch.arange(chunk_size, device=device)
    ).unsqueeze(1)  # [chunk, 1]
    matrix = torch.arange(seq_len // ratio, device=device).unsqueeze(0)

    causal_mask = matrix >= (base + 1) // ratio
    index_score = index_score + torch.where(
        causal_mask, torch.finfo(q_indexer.dtype).min, 0.0
    )

    max_valid = (cp_rank + 1) * chunk_size // ratio
    k = min(li_compute.index_topk, max_valid)
    index_score, topk_idxs = index_score.topk(k, dim=-1)

    mask = topk_idxs >= (base + 1) // ratio
    # offset-free：直接索引 causal_kv_compress [0, max_valid)
    compress_topk_idxs = torch.where(mask, -1, topk_idxs).to(torch.int32)

    return compress_topk_idxs, index_score
```

### 4.3.7 整体执行流程

起点：每个 rank 持有本地隐状态 x [B, chunk, D]，chunk = seq_len / cp_size。

第一步：Q 投影（本地）
	对 x 做两次线性投影（wq_a → wq_b），得到主 attention 用的 query q [B, chunk, n_heads, head_dim]。施加 RoPE 时用全局位置坐标（rank r 的起始位置是 r * chunk）。q 投影完之后一直等到最后 SparseAttention 才被使用。

第二步：Window KV 投影 + P2P 边界通信（本地投影 + 跨 rank 通信）
	通过 `pre_attn(x, freqs_local, hadamard_mat)` 得到 q 和 kv（其余输出丢弃）。
	投影完成后用 BoundaryExchange 做 P2P 通信：rank r-1 把自己末尾 **127 个** token 的 kv 发给 rank r，rank r 收到后拼接到自己 kv 的前面，得到 kv_full [B, 127+chunk, head_dim]。rank 0 有 dummy 依赖保证 backward 对称。

第三步：Indexer 前向（x / qr 已 detach，indexer 参数仍接梯度）
- 以 detached x 和 qr 为输入调用 `indexer_forward_with_cp`（不套 no_grad），产出三个结果：
  - k_indexer_local [B, chunk//4, 128]：直接调 `indexer.compressor(x, freqs_local)`（非 CP 版），纯本地，backward 无 P2P（避免 HCCL tag 碰撞）。
  - q_indexer [B, chunk, n_heads, 128]：由 wq_b 对 qr 投影得到，施加全局位置 RoPE。
  - weights [B, chunk, n_heads]：由 weights_proj 对 x 投影得到。
- 梯度不经过 x/qr 流回主干，但 **LiLoss backward 仍能通过 indexer 自身参数更新权重**。

第四步：AllGather k_indexer
	各 rank 的 k_indexer_local 通过 AllGatherCompressedKV 拼接成 global_k_indexer [B, seq_len//4, 128]，每个 rank 都拿到完整序列的检索 key。反向时自动变成 ReduceScatter。

第五步：LiCompute top-k 选择
- 用本地 q_indexer 与 global_k_indexer 做点积，得到每个 query 对所有压缩 token 的相关性分数，再乘以 weights 聚合各头。
- 关键修正：生成因果掩码时用 rank r 的全局 query 坐标，屏蔽掉未来位置的压缩 token。
- 最终 top-k 得到 **offset-free** 的 `compress_topk_idxs`（范围 `[0, max_valid)`），直接对应 `causal_kv_compress` 的行索引。

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
  - Window attention：q 对 kv_full 做注意力。rank>0 时 `sdpa_to_sfa_adapter` 走 SFA dummy-padding（C4A compress_ratio=4 不触发 `_original_forward` 回退）
  - Compress attention：q 对 causal_kv_compress 做注意力，使用 offset-free 的 compress_topk_idxs，SFA kernel 直接以此索引 `cmp_kv`
- 两路结果合并输出 o [B, chunk, head_dim]。
- LiLoss（辅助损失）由外层 `TransformerBlock.cal_index_loss` 计算，传入 offset-free 的 `compress_topk_idxs` 和裁剪后的 `causal_k_indexer`（长度 = valid_len，与 `causal_kv_compress` 匹配）。

### 4.3.8 完整前向代码

```python
def _c4a_forward_with_cp(
    pre_attn, inner_attn, post_attn,
    x,                    # [B, chunk, D]
    freqs_cis_global,     # 全量 freqs_cis（compress_rope_theta）
    hadamard_mat,
    attention_masks,
    cp_rank, cp_size, cp_group,
    cp_group_indexer,     # 独立 PG（same ranks，独立句柄），用于 k_indexer AllGather
    chunk_size, seq_len,
    ratio=4, window_size=128,
):
    B, chunk, _ = x.shape
    global_start = cp_rank * chunk_size
    freqs_local = freqs_cis_global[global_start : global_start + chunk].to(x.device)

    # ── Q / KV 投影（通过 pre_attn，kv_compress/indexer 输出丢弃）──────
    q, kv, _, _, _, _, _ = pre_attn(x, freqs_local, hadamard_mat)

    with torch.no_grad():
        qr = pre_attn.q_norm(pre_attn.wq_a(x.detach()))

    # ── Window KV 边界通信─────────────────────────────────────────────
    boundary_kv = BoundaryExchange.apply(
        kv[:, -(window_size - 1):, :].contiguous(),  # [B, 127, D]
        cp_rank, cp_size, cp_group, 0.0,
    )
    if cp_rank == 0:
        kv = kv + boundary_kv.sum(dim=(1, 2), keepdim=True) * 0.0
    kv_full = torch.cat([boundary_kv, kv], dim=1) if cp_rank > 0 else kv
    offset = kv_full.size(1)  # rank 0: chunk；rank r>0: 127+chunk

    # ── Indexer（x/qr detach；indexer 参数仍在 autograd 图中）──────────
    # 不套 no_grad：LiLoss backward 需要通过 indexer 自身参数传梯度。
    # x.detach()/qr.detach() 阻断梯度流回主干，但不影响 indexer 参数更新。
    q_indexer, k_indexer_local, weights = indexer_forward_with_cp(
        pre_attn.indexer, x.detach(), qr.detach(), freqs_cis_global,
        hadamard_mat, cp_rank, cp_size, cp_group, chunk_size, ratio,
    )

    # ── AllGather k_indexer（独立 PG 避免 backward HCCL 竞争，§4.3.5）──
    global_k_indexer = AllGatherCompressedKV.apply(k_indexer_local, cp_group_indexer)
    # [B, seq_len//4, index_head_dim]

    # ── LiCompute（offset-free 索引，§4.3.6）──────────────────────────
    compress_topk_idxs, index_score = li_compute_with_cp(
        inner_attn.li_compute, q_indexer, global_k_indexer, weights,
        chunk_size, seq_len, cp_rank, ratio,
    )
    # compress_topk_idxs: [B, chunk, k]，offset-free，范围 [0, max_valid)

    # ── 主 Compressor（overlap=True，CP-aware，§4.3.3）───────────────
    local_kv_compress = compressor_forward_with_cp(
        pre_attn.compressor, x, freqs_cis_global,
        cp_rank, cp_size, cp_group, chunk_size, ratio,
    )  # [B, chunk//4, head_dim]

    # ── AllGather kv_compress + 因果裁剪──────────────────────────────
    global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
    valid_len = (cp_rank + 1) * chunk_size // ratio
    causal_kv_compress = global_kv_compress[:, :valid_len, :]
    # 同步裁剪 k_indexer，使 size(1) 与 causal_kv_compress 匹配（li_loss kernel 要求）
    causal_k_indexer = global_k_indexer[:, :valid_len, :]

    # ── Sparse Attention（C4A rank>0 走 SFA dummy-padding，rank 0 走 SFA band）─
    # compress_topk_idxs 是 offset-free，SFA kernel 直接以此索引 cmp_kv ✓
    # window_topk_idxs 不传；sdpa_to_sfa_adapter 对 compress_ratio=4 走 dummy-padding
    o = inner_attn.sparse_attn(
        q, kv_full, inner_attn.attn_sink, causal_kv_compress,
        compress_topk_idxs=compress_topk_idxs,
    )

    n_local_groups = post_attn.n_groups // (pre_attn.n_heads // q.shape[2])
    x_out = post_attn(o, freqs_local, B, chunk, n_local_groups)

    return (
        x_out,
        compress_topk_idxs,   # offset-free → cal_index_loss / li_loss_adapter ✓
        offset,
        q, causal_kv_compress, attention_masks, index_score,
        q_indexer,
        causal_k_indexer,     # 裁剪到 valid_len → size(1)==causal_kv_compress.size(1) ✓
        weights,
    )
```

---

# 五、调试记录

## 5.1 错误一：`positions` 意外关键字参数

### 现象

首次运行 CP 训练，第一个 step 立即崩溃：

```
TypeError: DeepSeekV4Model.forward() got an unexpected keyword argument 'positions'
```

### 根因分析

torchtitan 的 `prepare_context_parallel_input` 函数在做序列切分时，会同步创建一个 `positions` 张量（全局位置索引），并将其写入 `extra_kwargs`，最终通过 `model(inputs, **extra_kwargs)` 传入模型：

```python
# torchtitan/distributed/context_parallel.py
positions = torch.arange(0, inputs.shape[1], ...).expand(inputs.shape)
(inputs, labels, positions), _ = cp_shard(cp_mesh, (inputs, labels, positions), ...)
extra_kwargs["positions"] = positions   # ← 写入 extra_kwargs
```

本项目还有 `mtp_context_parallel.py` 补丁，当 `num_mtp_modules=0` 时会直接调用原版 `prepare_context_parallel_input`，同样会产生 `positions`。

`DeepSeekV4Model.forward()` 原始签名只有 `(tokens, input_ids, attention_masks)`，不接受 `positions`，因此报错。

> **注意**：`positions` 对 DeepSeek-V4 CP 没有实际用途。我们的 CP patch 通过 `global_start = cp_rank * chunk` 自行推算每个 rank 的 freqs_cis 偏移，不依赖外部传入的 `positions`。

### 修复

在 `DeepSeekV4Model.forward()` 签名中增加 `positions=None`，接受但忽略（`torchtitan_npu/models/deepseek_v4/model/model.py`）。

---

## 5.2 错误二：`sdpa_to_sfa_adapter` 不接受 `window_topk_idxs`

### 现象

修复错误一后，第一个 step 再次崩溃：

```
TypeError: sdpa_to_sfa_adapter() got an unexpected keyword argument 'window_topk_idxs'
```

调用栈：
```
deepseek_v4_cp.py, _c1a_forward_with_cp
    o = inner_attn.sparse_attn(q, kv_full, inner_attn.attn_sink,
                               window_topk_idxs=window_topk)
→ sdpa_to_sfa_adapter() got an unexpected keyword argument 'window_topk_idxs'
```

### 根因分析

**执行顺序**：模型构建时，`deepseek_v4_sfa` converter 先于 CP patch 运行，将 `SparseAttention.forward` 替换为 `sdpa_to_sfa_adapter`：

```
[SparseAttention forward] Applied 1 replacement(s)   ← SFA converter，先
[DeepSeek-V4 CP] Patched Attention.forward.          ← CP patch，后
```

`sdpa_to_sfa_adapter` 原始签名无 `window_topk_idxs` 参数，`kv_compress`/`compress_topk_idxs` 也无默认值。CP 的 `_c1a_forward_with_cp` 以关键字方式传入 `window_topk_idxs`，Python 在绑定关键字参数阶段遇到未知名称立即抛出 `TypeError`。

**SFA kernel 不支持显式 ori 稀疏索引**：查阅 mindspeed 源码，`SparseAttnSharedKV` 的 metadata 调用中有明确注释 `oriTopk not support now`，即使传入 `ori_sparse_indices` 也会被 kernel 忽略，始终使用 band 模式。

**Band 模式在 CP rank r>0 下结果错误**：band 模式规则为 `query qi → kv[qi−127 .. qi]`，而 rank r>0 的 kv_full 布局为 `[boundary_127 | local_chunk]`，正确的 window 范围是 `kv_full[qi .. qi+127]`，两者完全不同，无法通过调整参数修正。

### 修复

**核心思路**：`sdpa_to_sfa_adapter` 通过 `getattr(self, 'cp_rank', 0)` 从 attention 模块属性读取 CP rank，无需 CP forward 函数显式传入 `window_topk_idxs`。内部按 `compress_ratio` 和 `cp_rank` 分三条路径处理：

```python
cp_rank = getattr(self, 'cp_rank', 0)
if cp_rank > 0:
    n_boundary = self.window_size - 1  # 127

    if self.compress_ratio != 4:
        # C1A / C128A rank>0：SFA band mode 无法覆盖边界窗口，回退到 SDPA。
        # 内部计算 window_topk = [i .. i+127]（覆盖 kv_full 的正确范围）
        B, S, _, _ = query_states.shape
        base = torch.arange(S, device=query_states.device).unsqueeze(1)
        win_off = torch.arange(self.window_size, device=query_states.device)
        window_topk = (base + win_off).unsqueeze(0).expand(B, -1, -1)
        return self._original_forward(
            query_states, kv_states, attn_sink, kv_compress,
            compress_topk_idxs, window_topk,
        )

    # C4A rank>0：预填 127 个 dummy zero query，使实际 query j 落在 tensor index
    # n_boundary+j，SFA band mode 自然覆盖 kv_full[j .. j+127]
    B, S, N, D = query_states.shape
    dummy_q = torch.zeros(B, n_boundary, N, D,
                          dtype=query_states.dtype, device=query_states.device)
    q_padded = torch.cat([dummy_q, query_states], dim=1)
    # compress_topk_idxs 已是 offset-free，直接填充 dummy 行后传给 kernel
    cmp_idx_padded = None
    if compress_topk_idxs is not None:
        topk = compress_topk_idxs.size(-1)
        dummy_idx = torch.zeros(B, n_boundary, topk,
                                dtype=torch.int32, device=compress_topk_idxs.device)
        cmp_idx_padded = torch.cat([dummy_idx, compress_topk_idxs], dim=1)
    output_padded = npu_sparse_attn_shared_kv(
        query=q_padded, ori_kv=kv_states, cmp_kv=kv_compress,
        cmp_sparse_indices=cmp_idx_padded, sinks=attn_sink.float(),
        softmax_scale=self.softmax_scale, cmp_ratio=self.compress_ratio,
    )
    return output_padded[:, n_boundary:].contiguous()

# 非 CP 或 cp_rank=0：标准 causal band mode，直接走 SFA kernel
output = npu_sparse_attn_shared_kv(
    query=query_states, ori_kv=kv_states, cmp_kv=kv_compress,
    cmp_sparse_indices=compress_topk_idxs if self.compress_ratio == 4 else None,
    sinks=attn_sink.float(), softmax_scale=self.softmax_scale, cmp_ratio=self.compress_ratio,
)
return output
```

路由汇总：
```
非 CP / cp_rank=0 (所有 compress_ratio):   SFA kernel（band mode）✅
cp_rank>0, compress_ratio ≠ 4 (C1A/C128A): 内部计算 window_topk=[i..i+127]，_original_forward ✅
cp_rank>0, compress_ratio == 4 (C4A):       SFA dummy-padding，band mode 覆盖正确窗口 ✅
```

涉及文件：
- `torchtitan_npu/models/deepseek_v4/model/model.py`：`SparseAttention` 末尾加 `_original_forward = forward`
- `torchtitan_npu/converters/kernels/deepseek_v4_sfa.py`：`sdpa_to_sfa_adapter` 通过 `cp_rank` 属性内部路由，CP forward 函数无需传入 `window_topk_idxs`

---

## 5.3 错误三：`compress_topk_idxs` 偏移语义混乱导致 HCCL REDUCE_SCATTER 超时

### 现象

修复错误二后，第一个 step 前向传播正常完成，但在反向传播阶段某个 rank 静默崩溃，导致其他 rank 在 FSDP 的 REDUCE_SCATTER 集合通信中永久等待，触发 300 秒看门狗超时：

```
[E420 21:27:54.629452030] [Rank 0] Watchdog caught collective operation timeout:
WorkHCCL(SeqNum=13, OpType=REDUCE_SCATTER, NumelIn=137963680, NumelOut=8622730,
Timeout(ms)=300000) ran for 300444 milliseconds before timing out.
```

关键特征：

- **超时发生在 REDUCE_SCATTER**（不是 AllGather）：前向已完成，反向阶段某 rank 崩溃。
- **NumelIn / NumelOut ≈ 16**：确认 FSDP 规约组包含全部 16 个 rank（dp_shard × cp = 8 × 2），任何一个 rank 在反向崩溃都会导致 rank 0 超时。
- **无显式 Python 报错**：崩溃在 NPU 内核内部，Python 层没有捕获到异常，rank 0 侧只看到超时。

### 根因分析

#### 两条路径对 `compress_topk_idxs` 的索引语义不同

C4A 层 attention 有两条执行路径，对 `compress_topk_idxs` 的索引语义要求截然不同：

| 路径 | 触发条件 | 对 `compress_topk_idxs` 的要求 |
|------|----------|-------------------------------|
| SFA kernel（`npu_sparse_attn_shared_kv`） | rank 0 / 非 CP，`window_topk_idxs=None` | **偏移无关（offset-free）**：直接索引 `cmp_kv`，范围 `[0, valid_len)` |
| `_original_forward`（SDPA 回退） | rank r>0，`window_topk_idxs` 非 None | **偏移相关（offset-inclusive）**：对拼接后的 `kv_full ‖ kv_compress` 做 scatter，范围 `[offset, offset+valid_len)` |

此外，`cal_index_loss` 路径同样对索引有独立要求：

| 路径 | 触发条件 | 对 `compress_topk_idxs` 的要求 |
|------|----------|-------------------------------|
| `li_loss_adapter` → `ms_npu_sparse_lightning_indexer_grad_kl_loss` | 反向传播 | **offset-free**：与非 CP 的 `sdpa_to_li_adapter`（offset=0）保持一致，直接索引 `causal_kv_compress`，范围 `[0, valid_len)` |

#### 原实现的错误

`li_compute_with_cp` 原来在返回 `compress_topk_idxs` 时叠加了 `offset`：

```python
# 错误：返回的是 offset-inclusive 索引 [offset, offset + k)
compress_topk_idxs = torch.where(mask, -1, topk_idxs + offset)
return compress_topk_idxs, index_score
```

这些带偏移的索引被存入 `_c4a_forward_with_cp` 的返回元组，沿调用链传递：

```
_c4a_forward_with_cp 返回元组[1]
  → InnerAttention.forward 的 compress_topk_idxs
    → TransformerBlock.cal_index_loss
      → ComputeIndexerLoss.forward
        → LiLoss.forward（已被 li_loss_adapter 替换）
          → npu_sparse_lightning_indexer_grad_kl_loss NPU kernel
```

NPU kernel 期待 offset-free 的 `[0, valid_len)` 索引，而实际接收的是 `[2048, 3199]`（以 seq=4096、cp=2、rank=1 为例，chunk=2048，offset=2048/4=512，但 topk_idxs 已是全局坐标，叠加更大的偏移）。索引远超 `causal_kv_compress` 的维度，会导致 NPU kernel 非法访存。

#### 为何崩溃在反向而非前向

`SparseLightningIndexerGradKLLossWrapper` 采用"延迟计算"设计：

```python
class SparseLightningIndexerGradKLLossWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):
        ctx.save_for_backward(...)
        # 前向只保存张量，返回 dummy 零值，不做实际计算
        return torch.zeros(1, dtype=torch.float32, device=query.device)[0]

    @staticmethod
    def backward(ctx, grad):
        # 真正的 NPU kernel 在此执行，索引越界在这里才暴露
        d_query_index, d_key_index, d_weights, loss = ms_npu_sparse_lightning_indexer_grad_kl_loss(...)
```

前向传播期间只存储张量，不调用真正的 NPU kernel，因此越界访问不会在前向暴露。真正的 kernel 在反向传播时执行，越界导致 NPU 崩溃，但 Python 层没有对应的异常传播机制，表现为 rank 静默失联，最终触发 rank 0 的 HCCL 超时。

#### 同样受影响的 rank 0 SFA 路径

rank 0 走 SFA kernel 路径时，`_c4a_forward_with_cp` 也直接将带偏移的 `compress_topk_idxs` 传给 `sparse_attn`，而 SFA kernel 期待 offset-free 的 `cmp_sparse_indices`，同样会产生错误结果（但不一定立即崩溃，因为 SFA kernel 在前向执行）。

### 修复

**核心原则**：`li_compute_with_cp` 始终返回 **offset-free** 索引；`_c4a_forward_with_cp` 在调用 attention 前根据路径按需叠加偏移，而返回元组中的 `compress_topk_idxs` 始终保持 offset-free 供 `cal_index_loss` 使用。

#### `li_compute_with_cp` 修改

移除 `offset` 参数，返回 offset-free 的 top-k 索引：

```python
def li_compute_with_cp(
    li_compute,
    q_indexer, global_k_indexer, weights,
    chunk_size, seq_len, cp_rank, ratio,
    # offset 参数已移除
):
    # ... score 计算、causal mask、topk ...
    mask = topk_idxs >= (base + 1) // ratio
    # offset-free：索引 causal_kv_compress 的 [0, max_valid) 区间
    compress_topk_idxs = torch.where(mask, -1, topk_idxs)
    return compress_topk_idxs, index_score
```

#### `_c4a_forward_with_cp` 修改

`li_compute_with_cp` 返回 offset-free 索引，`_c4a_forward_with_cp` 直接传给 `sparse_attn`，不再按 rank 条件叠加 offset。`sdpa_to_sfa_adapter` 内部对 C4A rank>0 采用 dummy-padding 方案，offset-free 索引对所有 rank 均正确：

```python
# LiCompute 返回 offset-free 索引 [0, max_valid)
compress_topk_idxs, index_score = li_compute_with_cp(
    inner_attn.li_compute,
    q_indexer, global_k_indexer, weights,
    chunk_size, seq_len, cp_rank, ratio,
)

# 所有 rank 统一传 offset-free 索引，路由由 sdpa_to_sfa_adapter 内部处理
o = inner_attn.sparse_attn(
    q, kv_full, inner_attn.attn_sink, causal_kv_compress,
    compress_topk_idxs=compress_topk_idxs,
)

# 返回元组携带 offset-free 索引供 cal_index_loss → li_loss_adapter 使用
return (
    x_out,
    compress_topk_idxs,   # offset-free ← li_loss_adapter 期待此语义
    offset,
    q, causal_kv_compress, attention_masks, index_score,
    q_indexer, causal_k_indexer, weights,
)
```

#### 各路径索引语义汇总

```
compress_topk_idxs（offset-free） ──┬──→ cal_index_loss → li_loss_adapter
                                    │      → ms_npu_sparse_lightning_indexer_grad_kl_loss ✅
                                    │
                                    ├──→ [cp_rank=0]  sdpa_to_sfa_adapter
                                    │      → SFA kernel（band mode，直接索引 cmp_kv）✅
                                    │
                                    └──→ [cp_rank>0] sdpa_to_sfa_adapter（C4A dummy-padding）
                                           → SFA kernel（dummy-q 填充后 band mode 正确）✅
```

涉及文件：
- `torchtitan_npu/distributed/context_parallel/deepseek_v4_cp.py`：`li_compute_with_cp` 移除 `offset` 参数，返回 offset-free 索引；`_c4a_forward_with_cp` 直接传 offset-free `compress_topk_idxs`，无条件分叉
- `torchtitan_npu/converters/kernels/deepseek_v4_sfa.py`：`sdpa_to_sfa_adapter` 对 C4A rank>0 做 dummy-padding，offset-free 索引直接可用

---

## 5.4 错误四：cp_rank=0 / cp_rank=1 走不同 attention 路径导致 backward autograd graph 不对称

### 现象

修复错误三后，仍复现完全相同的 REDUCE_SCATTER 超时（SeqNum=13），但 Python 栈显示某些 rank 在反向传播阶段停在 `mhc_triton.py backward → checkpoint recompute → MoE forward → EP AlltoAll`，说明前向已完成，是 backward autograd 图不一致导致 FSDP hook 触发顺序混乱。

### 根因分析

错误三的修复中，C4A 的 attention 路径仍依赖 `cp_rank`：rank 0 走 SFA 融合算子（单个 `autograd.Function`），rank>0 走 `_original_forward`（纯 PyTorch op 链）：

```
cp_rank=0：compress_topk_for_attn = offset-free → SFA kernel（单 Function.backward）
cp_rank=1：compress_topk_for_attn += offset     → _original_forward（op 链 backward）
```

两条路径的 **backward autograd graph 结构不同**：SFA 是集中触发的 `Function.backward`，SDPA 是一串 op 各自 backward。FSDP2 通过对参数输出挂 hook 驱动 per-group pre-backward unshard 和 post-backward REDUCE_SCATTER，两个 CP rank 之间 hook 触发时机不对齐，FSDP 集合通信死锁。

### 修复

**C4A 所有 CP rank 统一走 SFA 路径**，通过 dummy-padding 技巧消除 rank>0 的 band mode 错误：

在 `sdpa_to_sfa_adapter` 中，C4A rank>0 不再调用 `_original_forward`，而是在 query 前填充 `window_size-1 = 127` 个 dummy zero query，使实际 query j 落在 tensor index `127+j`，SFA band mode 自动覆盖 `kv_full[j..j+127]`（正确的边界窗口），取回有效输出时裁掉前 127 行即可。

此方案中两个 CP rank 的 C4A attention backward 均为单个 `SparseAttnSharedKV.backward`（结构相同），FSDP hook 触发顺序一致，死锁消除：

```
cp_rank=0：SFA kernel（正常 band mode）
cp_rank=1：SFA kernel（dummy-padding，band mode 覆盖正确窗口）
→ 两个 rank backward graph 结构完全一致 ✅
```

同步修改：`_c4a_forward_with_cp` 不再按 rank 条件叠加 offset，直接传 offset-free 的 `compress_topk_idxs`（SFA kernel 对所有 rank 均期待 offset-free 索引）。

> [!NOTE] C1A / C128A rank>0 仍走 `_original_forward`
> C4A 的 FSDP 死锁是由 li_compute / li_loss 的复杂 backward 引起的。C1A 无 compress KV，
> C128A 无 li_compute，backward 结构更简单，rank 间差异不会造成 FSDP hook 不对齐，
> 因此 C1A / C128A rank>0 继续在 `sdpa_to_sfa_adapter` 内部调用 `_original_forward`。

涉及文件：
- `torchtitan_npu/converters/kernels/deepseek_v4_sfa.py`：`sdpa_to_sfa_adapter` C4A 路径改为 dummy-padding，不再对 C4A rank>0 调用 `_original_forward`
- `torchtitan_npu/distributed/context_parallel/deepseek_v4_cp.py`：`_c4a_forward_with_cp` 去掉 `compress_topk_for_attn` 的 rank 条件分叉，直接传 offset-free 索引

---

## 5.5 错误五：`BoundaryExchange.backward` P2P 不对称导致 backward 永久阻塞

### 现象

修复错误四后，同一个 REDUCE_SCATTER（SeqNum=13）仍然超时。错误模式：所有偶数 rank（cp_rank=0）SIGABRT，所有奇数 rank（cp_rank=1）SIGTERM。说明 cp_rank=1 的某些 rank 在 backward 中途永久阻塞，从未到达 REDUCE_SCATTER；cp_rank=0 的 rank 等待超时被 HCCL watchdog 杀死；奇数 rank 随后被 elastic launcher 以 SIGTERM 终止。

### 根因分析

`BoundaryExchange.apply` 在前向传播时，rank 0 向 rank 1 发送边界 KV（isend），rank 1 向 rank 0 发送本地边界 KV（isend），双方均 irecv 完成通信。但在**反向传播**时存在不对称：

```
rank 1 (cp_rank=1):
  kv_full = cat([boundary_kv, kv])
  ← boundary_kv 参与 loss 计算，autograd 调用 BoundaryExchange.backward
  ← backward 执行 isend(grad → rank 0)，等待 rank 0 的 irecv 确认

rank 0 (cp_rank=0):
  kv_full = kv  （不含 boundary_kv）
  ← boundary_kv 不在 loss 路径上，autograd 将其剪枝
  ← BoundaryExchange.backward 根本不被调用
  ← 不发出 irecv，rank 1 的 isend 永远等不到对端

→ rank 1 永久阻塞在 BoundaryExchange.backward 的 isend.wait()
→ rank 1 永远到不了 FSDP REDUCE_SCATTER
→ rank 0 等 REDUCE_SCATTER 超时 → SIGABRT
→ elastic launcher 杀掉 rank 1 → SIGTERM
```

P2P 通信不受 HCCL watchdog 监控，因此 rank 1 的阻塞不会触发 watchdog，只有 rank 0 的 REDUCE_SCATTER 超时才会暴露。

### 修复

在 `_c1a_forward_with_cp`、`_c128a_forward_with_cp`、`_c4a_forward_with_cp` 中，对 cp_rank=0 的 `boundary_kv` 添加数值为零的 dummy 依赖，确保 autograd 引擎追踪到 `boundary_kv` 并调用 `BoundaryExchange.backward`：

```python
boundary_kv = BoundaryExchange.apply(
    kv[:, -(window_size - 1):, :].contiguous(),  # 127 tokens，不是 window_size=128
    cp_rank, cp_size, cp_group, 0.0,
)
if cp_rank == 0:
    # boundary_kv 不进入 kv_full，若不做依赖，autograd 会剪枝此节点，
    # 导致 BoundaryExchange.backward 不被调用，rank1 的 isend 永久阻塞。
    kv = kv + boundary_kv.sum(dim=(1, 2), keepdim=True) * 0.0
kv_full = torch.cat([boundary_kv, kv], dim=1) if cp_rank > 0 else kv
```

**正确性**：
- 前向：`* 0.0` 保证 kv 数值不变 ✅
- backward：rank 0 调用 `BoundaryExchange.backward`，执行 `irecv(from rank 1)`，rank 1 的 `isend` 有了对端，双方均正常完成 ✅
- 梯度：rank 0 的 `grad_recv=0`（来自 dummy 项），但通过 `irecv` 正确接收 rank 1 计算出的 `grad_send`，作为 `send_tensor`（即 `kv[:, -window_size:, :]`）的真实梯度返回，梯度数学上正确 ✅

注意：`compressor_forward_with_cp` 内的两个 `BoundaryExchange`（kv 和 score）对 rank 0 无此问题，因为 `recv_kv`/`recv_score` 通过 `new_kv[:, 0, :ratio] = recv_kv[:, 0]` 实际参与了计算，autograd 不会剪枝。

涉及文件：
- `torchtitan_npu/distributed/context_parallel/deepseek_v4_cp.py`：三个 `*_forward_with_cp` 函数的窗口 KV `BoundaryExchange` 之后各加 `if cp_rank == 0: kv = kv + boundary_kv.sum(...) * 0.0`

---

## 5.6 错误六：`key` 与 `key_index` 序列维度不匹配导致 LiLoss backward SIGSEGV

### 现象

修复错误四、五后，HCCL timeout 消失，训练推进到 backward 计算阶段，但立即触发：

```
Fatal Python error: Segmentation fault

Current thread (most recent call first):
  File "torchtitan_npu/converters/kernels/deepseek_v4_sfa.py", line 215 in backward
  File "torch/autograd/function.py", line 317 in apply
```

崩溃在 `SparseLightningIndexerGradKLLossWrapper.backward` 调用 `ms_npu_sparse_lightning_indexer_grad_kl_loss`（line 215）。崩溃模式：偶数 rank（cp_rank=0）全部 SIGSEGV，奇数 rank（cp_rank=1）全部 SIGTERM——crash 只发生在 cp_rank=0。

### 根因分析

`li_loss_adapter` 将 `key`（即 `causal_kv_compress`）和 `key_index`（即 `k_indexer`，来自返回元组位置 [8]）一起传入 NPU kernel：

```python
npu_sparse_lightning_indexer_grad_kl_loss(
    query,
    key.unsqueeze(2),        # causal_kv_compress: [B, valid_len, 1, head_dim]
    query_index,
    key_index.unsqueeze(2),  # global_k_indexer:   [B, seq//ratio, 1, idx_dim]
    weights,
    sparse_indices.unsqueeze(2),
    ...
)
```

NPU kernel 要求 `key.size(1) == key_index.size(1)`，但返回元组中位置 [8] 存的是 **AllGather 后的完整** `global_k_indexer`（序列长度 = `seq_len//ratio`），而 `causal_kv_compress` 已做因果裁剪（序列长度 = `valid_len = (cp_rank+1)*chunk//ratio`）：

| rank | `key`（causal_kv_compress） | `key_index`（global_k_indexer） |
|------|-----------------------------|--------------------------------|
| cp_rank=0 | `[B, 512, head_dim]`（valid_len=512） | `[B, 1024, idx_dim]`（seq//ratio） |
| cp_rank=1 | `[B, 1024, head_dim]` | `[B, 1024, idx_dim]` ✅ |

cp_rank=0 传入 512 vs 1024，NPU kernel 在 C++ 层非法访存，直接 SIGSEGV。cp_rank=1 恰好两者都是 1024，不崩溃——这解释了偶数/奇数 rank 的分裂现象。

### 修复

**主修复**：在 `_c4a_forward_with_cp` 中，将 `global_k_indexer` 裁剪到 `valid_len` 后再放入返回元组：

```python
# 对 causal_kv_compress 做因果裁剪
valid_len = (cp_rank + 1) * chunk_size // ratio
causal_kv_compress = global_kv_compress[:, :valid_len, :]

# 同步裁剪 k_indexer，使序列维度与 causal_kv_compress 匹配
causal_k_indexer = global_k_indexer[:, :valid_len, :]

return (
    x_out,
    compress_topk_idxs,   # offset-free
    offset,
    q,
    causal_kv_compress,
    attention_masks,
    index_score,
    q_indexer,
    causal_k_indexer,     # ← 裁剪后：size(1)=valid_len，与 causal_kv_compress 匹配
    weights,
)
```

**辅助修复**（`li_loss_adapter`）：增加防御性清洗，解决 CP 路径下的 dtype 问题：

```python
# CP 路径的 compress_topk_idxs 来自 torch.topk（int64），
# 非 CP 路径来自 npu_lightning_indexer（int32）。
# NPU kernel 要求 int32，必须转换。
valid_len = key.size(1)
sparse_indices = sparse_indices.to(torch.int32)
sparse_indices = torch.where(
    (sparse_indices < 0) | (sparse_indices >= valid_len),
    torch.full_like(sparse_indices, -1),
    sparse_indices,
).contiguous()
```

涉及文件：
- `torchtitan_npu/distributed/context_parallel/deepseek_v4_cp.py`：`_c4a_forward_with_cp` 返回元组位置 [8] 改为 `global_k_indexer[:, :valid_len, :]`
- `torchtitan_npu/converters/kernels/deepseek_v4_sfa.py`：`li_loss_adapter` 增加 `sparse_indices` 的 int32 转换、越界屏蔽和 contiguous 保证

---

# 附录 B：C128A CP cp_rank>0 attention 输出差异排查与修复

## B.1 问题现象

在 cp=2 配置下，rank=1（`cp_rank=1`）的 attention 输出 `inner_o` 与 cp=1 对应位置的输出存在显著差异（norm 差约 2.5%），而 cp=0 的输出完全匹配。通过日志确认，差异前的所有输入（`q`、`kv_full`、`kv_compress`、`compress_topk`）在 cp=1/cp=2 之间完全一致，问题纯粹在 attention 计算内部。

## B.2 根因分析

### Bug 1：Window topk 索引错误（cp_rank>0）

**背景**：对于 cp_rank>0，`_c128a_forward_with_cp` 将 `kv_full` 拼接为 `[boundary_127 || local_chunk]`，共 `127 + chunk` 个 token。query `i`（0-indexed）正确的 window 应覆盖 `kv_full[i .. i+127]`。

**错误**：`SparseAttention.forward` 在 `window_topk_idxs=None` 时调用 `get_window_topk_idxs(128, bsz, chunk)` 生成 `[max(0, i-127) .. i]`，这是以 `chunk` 为序列长度的局部 causal 窗口。对于 cp_rank=0 正确，但对于 cp_rank>0，kv_full 前 127 个位置是 boundary token，真实 local chunk 从偏移 127 开始，因此生成的索引完全错位。

**修复位置**：在 `sdpa_to_sfa_adapter`（C128A + cp_rank>0 的回退路径，见 Bug 2 修复）中传入正确的 `window_topk_idxs`：

```python
B, S, _, _ = query_states.shape
base = torch.arange(S, device=query_states.device).unsqueeze(1)   # [S, 1]
win_off = torch.arange(self.window_size, device=query_states.device)  # [W]
window_topk = (base + win_off).unsqueeze(0).expand(B, -1, -1)     # [B, S, W]
# 对 query i：window = [i, i+1, ..., i+127]，正确覆盖 kv_full 中的边界+本地窗口
```

### Bug 2：压缩 KV causal mask 错误（C128A + cp_rank>0）

**背景**：`sdpa_to_sfa_adapter` 对 cp_rank>0 采用"dummy-query padding"方案：在 query 前拼接 `window_size-1 = 127` 个全零 dummy query，使真实 query `i` 的 tensor 索引变为 `127+i`，SFA 硬件核的 band mode（`ori_win_left=127`）随即覆盖 `kv_full[i .. i+127]`，正好是 window attention 所需的窗口。

**错误**：SFA 核的压缩 KV 采用 `cmp_mask_mode=3`（因果掩码），其可见性由 query 的 **tensor 位置**决定，而非 **全局序列位置**。对于 cp_rank=1、chunk=2048 的情形：

| | 全局位置 | tensor 位置（padding 后） | 可见压缩 token 数 |
|---|---|---|---|
| query `i`（正确） | `2048 + i` | — | `⌊(2048+i+1)/128⌋ ≈ 16 + i//128` |
| query `i`（实际） | — | `127 + i` | `⌊(127+i+1)/128⌋ = 1 + i//128` |

rank=1 的每个 query 仅能看到约 1～17 个压缩 token，而正确值约为 16～32 个，造成 attention 漏掉大量全局上下文。

## B.3 尝试方案 Option A（失败）

**思路**：绕过 `cmp_mask_mode=3`，改为向 SFA 核传入显式的 `cmp_sparse_indices`（0-indexed），让核直接按索引读取正确的压缩 token，而不依赖位置推算。

修改点：
1. `npu_sparse_attn_shared_kv`：将 `if cmp_ratio != 4` 改为 `if cmp_sparse_indices is None`，允许 C128A 也传入非零 `topk` 和显式索引
2. `sdpa_to_sfa_adapter` cp_rank>0 分支：将 offset-full 的 `compress_topk_idxs`（值为 `kv_full.size(1) + compress_pos`）转换为 0-indexed，拼接 dummy 索引后传入核

**结果**：运行时 NPU 硬件核崩溃：

```
kernelName=SparseAttnSharedkvMetadata, errorCode=0x2a
[aicpu exception] Kernel task happen error, retCode=0x2a
```

**结论**：`SparseAttnSharedkvMetadata` 硬件核在 `cmp_ratio=128` 时不支持显式稀疏索引（`topk > 0`），这是硬件层面的限制，无法在 Python/算子调用层绕过。Option A 不可行。

## B.4 最终方案 Option B（已合入）

**思路**：C128A + cp_rank>0 完全绕开 SFA 硬件核，回退到 Python SDPA 路径（`_original_forward`），在 Python 层使用正确的 `window_topk_idxs` 做 scatter 掩码 attention。

**正确性分析**：
- `_original_forward` 接受显式 `window_topk_idxs` 和 `compress_topk_idxs`
- `window_topk_idxs = [i, i+1, ..., i+127]` 正确覆盖 `kv_full` 中 query `i` 的边界+局部窗口
- `compress_topk_idxs` 为 offset-full 索引（`kv_full.size(1) + compress_pos`），`_original_forward` 内部将 `kv_full` 与 `kv_compress` 拼接后，这些索引恰好指向 `kv_compress` 中的正确位置
- C4A + cp_rank>0 继续使用原有的 dummy-padding + SFA 核路径（硬件支持 C4A 的显式稀疏索引）

**修改内容**（`torchtitan_npu/converters/kernels/deepseek_v4_sfa.py`）：

`npu_sparse_attn_shared_kv`：恢复原始保护逻辑（必须，防止 Option A 残留代码触发硬件崩溃）：

```python
# 恢复原始逻辑：C128A 始终 topk=0，不传显式索引
topk = 0 if cmp_ratio != 4 else cmp_sparse_indices.size(-1)
if cmp_ratio != 4:
    cmp_sparse_indices = None
else:
    cmp_sparse_indices = cmp_sparse_indices.unsqueeze(2).contiguous()
```

`sdpa_to_sfa_adapter`：cp_rank>0 增加 C128A 早退分支：

```python
if cp_rank > 0:
    n_boundary = self.window_size - 1  # 127

    if self.compress_ratio != 4:
        # C128A：硬件核不支持显式 cmp_sparse_indices，回退 Python SDPA
        B, S, _, _ = query_states.shape
        base = torch.arange(S, device=query_states.device).unsqueeze(1)
        win_off = torch.arange(self.window_size, device=query_states.device)
        window_topk = (base + win_off).unsqueeze(0).expand(B, -1, -1)
        return self._original_forward(
            query_states, kv_states, attn_sink, kv_compress,
            compress_topk_idxs, window_topk,
        )

    # C4A：保持原有 dummy-padding + SFA 核路径
    ...
```

**注意**：`_original_forward` 是 `SparseAttention.forward` 在 SFA converter 替换前通过 `_original_forward = forward` 保存的原始方法，可通过 `self._original_forward(...)` 调用。
