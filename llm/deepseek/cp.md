
DeepSeek-2026 CP 切分设计方案
# 零、前提代码修复

`args.py` 中 `attn_type` 属性访问 bug

`DeepSeek2026ModelArgs` 未定义 `attn_type` 属性，直接访问会触发 `AttributeError`。

```python
# model/args.py:65 — 当前错误写法
if (
	job_config.parallelism.context_parallel_degree > 1
		and self.attn_type != "sdpa" # ← AttributeError！
):
```

修复方式（与 `infra/parallelize.py:179` 保持一致）：
```python
# 修复后
attn_type = getattr(self, "attn_type", "sdpa")
if (
	job_config.parallelism.context_parallel_degree > 1
		and attn_type != "sdpa"
):
	raise NotImplementedError("CP support is only supported for SDPA.")
```


---
# 一、问题本质分析

DeepSeek-2026 的每个 Transformer Block 包含两大组件：**MHC（Multi-Head Hyper-Connections）** 和 **Attention**。
CP 切分的挑战主要来自 Attention，不同层的 compress_ratio 决定了序列依赖的范围和通信开销。
`compress_ratios` 的模式是 `(1, 1, 4, 128, 4, 128, ..., 4, 128, 4)`，共三类 Attention 层：

| 层类型                | 序列依赖范围                        | 通信挑战                 |
| ------------------ | ----------------------------- | -------------------- |
| ratio = 1（Window）  | 局部 window_size = 128          | 仅边界 token 通信         |
| ratio = 4（C4A）     | 全局（Lightning Indexer 选 top-k） | 压缩 KV 的全局 all-gather |
| ratio = 128（C128A） | 全局（但 token 数极少）               | 压缩 KV 的全局 all-gather |

**ratio = 1（Window Attention）**：每个 query 只向左看 `window_size=128` 个原始 token，依赖范围纯局部。CP 切分后，rank r 最前面的若干 query 的窗口会跨越到 rank r-1 的末尾，因此只需要一次轻量的 P2P 通信，从前驱 rank 借来末尾 128 个 token 的 KV 即可，通信量极小（约 131 KB/层）。

**ratio = 4（C4A）**：每 4 个原始 token 被 Compressor 压缩为 1 个压缩 token，再由 Lightning Indexer 从全局压缩序列中检索 top-k 个最相关的压缩 token 参与 attention。这意味着每个 query 的 KV 依赖分散在整个序列上，CP 切分后必须通过 AllGather 将各 rank 的本地压缩 KV 汇聚成全局视图。此外，C4A 的 Compressor 使用 `overlap=True`，相邻压缩 token 之间存在跨 group 依赖，CP 边界处还需要额外的 P2P 边界修正。

**ratio = 128（C128A）**：与 C4A 原理相同，但压缩比高达 128:1，全局压缩序列极短（例如 seqlen=65536 时仅 512 个压缩 token）。Compressor 使用 `overlap=False`，无跨 group 依赖，不需要 P2P 边界修正。AllGather 的通信量相比 C4A 缩小 32 倍（约 0.5 MB/层），几乎可以忽略。


**关于 MHC（Multi-Head Hyper-Connections）**
> [!NOTE] 模型用的是 MHC，不是单头 HC
> 代码中 `hc_mult=4`，每个 token 的隐藏状态形状为 `[B, S, 4, D]`，即每个位置
> 维护 4 个并行 stream，这是 MHC（Multi-Head Hyper-Connections）而非单头 HC。

MHC 是纯 **per-token** 的局部操作：HcPre / HcPost 的 Sinkhorn 路由只在同一 token 的 `hc_mult=4` 个 stream 之间做加权混合，**不涉及任何跨 token 的序列交互**。CP 将序列按 token 切分，同一 token 的 4 个 stream 始终在同一 rank 上，因此 MHC **不需要任何 CP 通信**。

CP 按序列维度 S 切分，rank r 持有位置 `[r*chunk, (r+1)*chunk)`。对于这些位置，所有 MHC 运算（HcPre、HcSplitSinkhorn、HcPost，以及跨层的 comb 矩阵链）都完全在本 rank 内自洽完成，不需要从其他 rank 获取任何数据。 CP 面对的是跨 token 数据依赖问题，MHC 本身没有这个问题，所以对 CP 完全透明。
  
  ---
# 二、前提：Chunk 对齐约束（chunk 就是每个 cp rank 持有的本地序列片段）

必要条件：`chunk_size = seqlen / cp_degree` 必须能被 128 整除，等价于：
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
  
  ---
# 三、三类不同层的 CP 通信设计

## 3.1 ratio=1：Window Attention（P2P 边界通信）


> [!NOTE] ratio=1 和 发送 1 个 token 的区别
> ratio=1 表示没有压缩，每个 token 的 KV 就是它本身，不会被合并成更少的 token。
> Window Attention 的计算需求：Window Attention 中，每个 query 可以向左看 window_size=128 个原始 token；query 在全局位置 i -> 需要 KV 的位置来自 [i - 127, i]；

```
  CP 切分后，rank r 的起始 query 位置是 r*chunk：
  
  rank r 的第 0 个 query（位置 r*chunk）：
    需要 KV 来自 [r*chunk - 127,  r*chunk - 1]  ← 这 127 个在 rank r-1 上
    
  rank r 的第 1 个 query（位置 r*chunk + 1）：
    需要 KV 来自 [r*chunk - 126,  r*chunk    ]  ← 126 个在 rank r-1 上 
    
  rank r 的第 k 个 query（位置 r*chunk + k）：
    需要 KV 来自 rank r-1 的最后 (128-k) 个 token
    
  rank r 的第 128 个 query（位置 r*chunk + 128）：
    需要 KV 来自 [r*chunk + 1, r*chunk + 128] ← 全部在 rank r 上，不再依赖 rank r-1
    
  所以需要从 rank r-1 借的 token 数量最多是 128（前 128 个 query 都有不同程度的跨 rank 依赖），发送完整的 window_size=128 个 token 是最简洁的做法。
```

具体需要传送内容的演示：
```
  rank r-1: [..., token_{r*chunk - 128}, ..., token_{r*chunk - 1}]
			                                     ↓  Send last window_size tokens to next the rank
rank r:  [token_{r*chunk}, ...]   ←  Recv (128 tokens)
```

  - 通信原语：P2P Send/Recv（仅向前一个 rank 借 128 个 token）
  - 通信量：128 × head_dim × 2B = 128 × 512 × 2B ≈ 131KB（极小）
- 采用 isend + irecv + wait() 的策略：
```python
  def window_boundary_comm(boundary_tokens, recv_buf, rank, cp_size, group):
      """                                                                   
      boundary_tokens: (B, 128, head_dim)，当前 rank 最后 128 个 token
      recv_buf:        (B, 128, head_dim)，预分配接收缓冲区                    
      """

      reqs = []  

      # 向右邻居发送（非阻塞，立即挂起）
      if rank < cp_size - 1:
          reqs.append(dist.isend(boundary_tokens, dst=rank + 1, group=group)) 
          
      # 从左邻居接收（非阻塞，立即挂起）
      if rank > 0:
          reqs.append(dist.irecv(recv_buf, src=rank - 1, group=group))    

      # ← 这里可以插入本地计算（计算通信 overlap）                                                                                                           
      # 统一等待所有挂起请求
      for req in reqs:
          req.wait()
      return recv_buf if rank > 0 else None
```
采用这个策略不会造成死锁的原因：<mark style="background:#b1ffff">isend/irecv 只是向通信后端注册请求并立即返回，不阻塞</mark>。所有rank在 wait() 之前就已经同时把 send/recv 同时挂起了，后端在内存段完全匹配。

### 3.1.1 KV 拼接后 window_topk_idxs 必须修正

> [!IMPORTANT] 不能沿用原 GetWindowTopkIdxs 生成的索引
> 原 `GetWindowTopkIdxs` 生成的索引基于本地坐标 `0..chunk-1`，拼接边界 token 后
> local_kv 整体偏移到了位置 128，原有索引无法访问边界 token，**必须重新计算**。
> 换成更简单的理解：由于cp切分，rank r 需要接收 rank r-1 后128 个 token，从而组成新的 kv 矩阵，新矩阵不仅大小发生了改变，每个位置的语义也发生了改变。


![[Pasted image 20260416111452.png]]

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
		base = torch.arange(chunk).unsqueeze(1) # [chunk, 1]
		window_topk = (base - window_size + 1).clamp(0) + torch.arange(window_size)
		window_topk = torch.where(window_topk > base, -1, window_topk)
	
	else:
	# kv_full = [boundary(128) || local_kv(chunk)] [B, 128+chunk, D]
	# query i → kv_full 位置 [i+1, i+2, ..., i+128]
		base = torch.arange(chunk).unsqueeze(1) # [chunk, 1]
		window_topk = base + torch.arange(1, window_size + 1) # [chunk, 128]
	# 合法性：i ∈ [0, chunk), 取值范围 [1, chunk+127] ⊆ [0, 128+chunk-1] ✓
	
	return window_topk.unsqueeze(0).expand(bsz, -1, -1)
```

拼接完 window_topk_idxs 后，compress 相关的索引 offset 也需随之调整：
- rank 0：`offset = chunk`（与原来相同）
- rank r > 0：`offset = 128 + chunk`（kv_full 多了 128 个边界 token）

---

## 3.2 C128A CP 切分完整设计方案

### 3.2.2 数据流设计

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
  │                                  compress_topk_idxs)                                                 │
  │ （compress_topk_idxs 需用全局坐标，见 3.2.4）                             │
  └────────────────────────────────────────┘
                │     
                ▼ 
          输出: [B, chunk, D]

### 3.2.3 C128A 的 Window KV 边界通信

C128A 层的 `SparseAttention` 同样调用 `GetWindowTopkIdxs`，因此与 ratio=1 层存在相同的边界依赖问题。

处理方式与 3.1 节完全相同：
- 用 `window_boundary_comm` 获取 boundary_kv（rank r-1 末尾 128 token）
- 构造 `kv_full = cat([boundary_kv, local_window_kv], dim=1)`（rank > 0）
- 用 `cp_window_topk_idxs` 替换 `GetWindowTopkIdxs` 生成 window_topk_idxs
- 更新 `offset = kv_full.size(1)`（rank > 0 时为 128+chunk，rank 0 时为 chunk）

### 3.2.4 通信量分析

以 cp=8 为例：
  local_kv_compress 形状：[B, chunk//128, 512]
                         = [1, 8192//128, 512]  
                         = [1, 64, 512]
  AllGather 后：[1, 64*cp_size, 512] = [1, 512, 512]
  通信量 = 512 × 512 × 2B(fp16) = 524,288 B ≈ 512 KB/layer（可忽略）
  
  对比：                                                                                                                                                                       
    ratio=4  kv_compress AllGather：16 MB/layer
    ratio=128 kv_compress AllGather：0.5 MB/layer  ← 缩小 32 倍  

### 3.2.5 伪代码实现

#### 3.2.5.1 Autograd-aware AllGather(前向 All-Gather，反向自动 Reduce-Scatter)
```python
class AllGatherCompressedKV(torch.autograd.Function):
      """         
      前向：AllGather local_kv_compress → global_kv_compress
      反向：ReduceScatter grad_global   → grad_local
      """
      
      @staticmethod
      def forward(
          ctx,
          local_kv: torch.Tensor,   # [B, chunk//128, 512]
          group: dist.ProcessGroup,
      ) -> torch.Tensor:            # [B, seq_len//128, 512]
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
          grad_global: torch.Tensor,  # [B, seq_len//128, 512]
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
          
def allgather_kv_compress(
      local_kv: torch.Tensor,
      group: dist.ProcessGroup,
  ) -> torch.Tensor:
      return AllGatherCompressedKV.apply(local_kv, group)
```

#### 3.2.5.2 C128A 的 compress_topk_idxs 全局坐标修正

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
	ratio: int, # = 128
	cp_rank: int,
	causal_kv_len: int, # = (cp_rank + 1) * chunk // ratio
	offset: int, # = kv_full.size(1)，window KV 占用的长度
) -> torch.Tensor: # [B, chunk, causal_kv_len]
	"""
	为 C128A 在 CP 模式下生成 compress_topk_idxs。
	query 在本地位置 i（全局位置 cp_rank*chunk+i）最多看到全局压缩位置
	j < (cp_rank*chunk + i + 1) // ratio 的压缩 token。
	
	causal_kv_compress 已经被 AllGather 后裁剪到前 causal_kv_len 个，
	所以这里只需相对于 causal_kv_compress 做因果掩码即可。
	"""
	# 全局 query 位置（以 1 为基，方便计算 ceil(pos/ratio)）
	global_base = cp_rank * chunk + torch.arange(1, chunk + 1) # [chunk]
	
	# 压缩 token 全局坐标
	compress_pos = torch.arange(causal_kv_len) # [causal_kv_len]
	
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

#### 3.2.5.3 C128A 前向主逻辑
```python
  def c128a_forward_with_cp(
      x: torch.Tensor,              # [B, chunk, D]
      compressor,                   # ratio=128 Compressor 实例
      window_proj,                  # Window KV 投影
      attn_sink,
      sparse_attn,                  # SparseAttention 实例（ratio=128）
      cp_rank: int,
      cp_size: int,
      cp_group: dist.ProcessGroup,
      seq_len: int,
      window_size: int = 128,
  ) -> torch.Tensor:
      B, chunk, D = x.shape
      ratio = 128
      
      # ── 1. 本地 Window KV（纯本地投影）────────────────────────────────────
      local_window_kv = window_proj(x) # [B, chunk, head_dim=512]
      
      # ── 2. Window KV 边界通信（复用 3.1 节 window_boundary_comm）──────────
      recv_buf = torch.zeros_like(local_window_kv[:, :window_size, :])
      boundary_kv = window_boundary_comm(
	      local_window_kv[:, -window_size:, :], recv_buf, cp_rank, cp_size, cp_group
	  ) # rank > 0 时为 [B, 128, head_dim]，rank 0 返回 None
	  
	  if cp_rank > 0:
		  kv_full = torch.cat([boundary_kv, local_window_kv], dim=1) # [B, 128+chunk, D]
	  else:
	      kv_full = local_window_kv # [B, chunk, D]
	      
	  # ── 3. Window topk 索引（CP 修正）────────────────────────────────────
	  window_topk = cp_window_topk_idxs(B, chunk, window_size, cp_rank)
	  # [B, chunk, 128]
	  
	  # ── 4. 本地压缩（compressor 在本地 chunk 上独立运行)────────────────────
	  local_kv_compress = compressor(x) # [B, chunk//128, head_dim]
	  
	  # ── 5. AllGather 压缩 KV（通信量 ~512KB，可忽略)───────────────────────
	  global_kv_compress = allgather_kv_compress(local_kv_compress, group=cp_group)
	  # [B, seq_len//128, head_dim]
	  
	  # ── 6. 因果有效性裁剪──────────────────────────────────────────────────
	  valid_compress_len = (cp_rank + 1) * (chunk // ratio)
	  causal_kv_compress = global_kv_compress[:, :valid_compress_len, :]
	  # rank 0: [B, chunk//128, D]；rank 7: [B, seq_len//128, D]
	  
	  # ── 7. compress_topk_idxs（全局坐标修正）──────────────────────────────
	  offset = kv_full.size(1) # rank 0: chunk；rank r>0: 128+chunk
	  compress_topk = get_c128a_compress_topk_idxs_cp(
		  B, chunk, ratio, cp_rank, valid_compress_len, offset
	  )
	  # [B, chunk, valid_compress_len]
	  
	  # ── 8. 合并 topk 索引并调用 SparseAttention ─────────────────────────
	  q = query_proj(x) # [B, chunk, n_heads, head_dim]
	  topk_idxs = torch.cat([window_topk, compress_topk], dim=-1).int()
	  o = sparse_attn(q, kv_full, attn_sink, causal_kv_compress, compress_topk)
	  # sparse_attn 内部会拼接 kv_full 和 causal_kv_compress
	  
	  return o     
```
  

---

## 3.3 ratio=4：C4A（最复杂，需分步处理）

### 3.3.1 C4A 需要分成 Compressor 和 Indexer 的原因

C4A（ratio=4）在注意力之前需要把全序列的 KV 压缩到 1/4，但压缩后的序列仍然很长（S/4），不能对所有压缩 token 都做 attention，否则计算量仍然很大。因此引入"先检索、再 attention"的两阶段设计：这两个阶段都用 wkv 提取特征、wgate提供门控权重，两者逐元素相乘后在 ratio 维度 softmax + sum 完成压缩。

**第一阶段：Compressor（压缩）+ Indexer 内部压缩**
这两路压缩结果来自**两个独立的 Compressor 实例**，各有自己的 wkv / wgate 参数：

| 实例                             | head_dim | 输出                       | 作用                |
| :----------------------------- | :------- | :----------------------- | :---------------- |
| 主 Compressor（PreAttention 中）   | 512      | kv_compressor [S/4, 512] | 用于实际 attention 计算 |
| Indexer.compressor（Indexer 内部） | 128      | k_indexer [S/4, 128]     | 检索用的轻量 key        |


两个实例都是同一个 `Compressor` 类，内部均各自持有独立的 wkv 和 wgate，计算流程相同：`kv = (wkv(x) * wgate(x).softmax()).sum()`，对每 ratio=4 个 token 加权求和压缩成 1 个 token。区别仅在于输出 head_dim 不同（512 vs 128），以及各自独立的参数通过训练学到了不同的投影——一套偏向语义保留，一套偏向检索相关性。

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

<mark style="background:#affad1">k_indexer 只用来做"相关性排序"，不需要携带完整语义信息</mark>。维度低（128）→ 检索更快 → top-k 计算代价小（O(S/4 × 128)，而非 O(S × 512)）。
<mark style="background:#affad1">kv_compressor 要参与真正的 attention 计算，需要保留完整表征</mark>。维度高（512）→ 保留语义 → attention 质量高。

这种设计的核心权衡是：**先均匀压缩（S → S/4），再用低维 k_indexer 快速定位最相关的位置，最后只对 top-k 个位置用高维 kv_compressor 做精确计算**，把选择代价从 O(S × 512) 压缩到 O(S/4 × 128)。
### 3.3.2 C4A CP 方案设计
#### 3.3.2.1 问题定位

  C4A compressor 使用 overlap=True，其 overlap_transform 使第 i 个压缩 token 依赖第 i-1 个 group 的数据：
	  new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]    # group i 混入 group i-1 的数据

  CP 切分后，rank r 的 compressed_token[0] 需要 rank r-1 最后一个 group 的数据，但本地看不到，导致错误地填入全 0：
	  非 CP（正确）：  compressed[0 of rank r] = f(group_k,  group_{k-1})  ✓
	  CP 无处理：      compressed[0 of rank r] = f(group_k,  0)            ✗

![[Pasted image 20260416145328.png]]

#### 3.3.2.2 通用 BoundaryExchange （Compressor 跨 rank 边界修正）

score 是 wgate 的输出，经过 softmax 后作为每个槽的权重，决定 ratio 个 token（含 overlap 槽）各贡献多少到压缩结果。
overlap_transform 的机制是：group i 的 overlap 槽 = group i-1 的 kv/score 的前 d 维。在 CP 模式下，rank r 的 group 0 的overlap 槽应该来自 rank r-1 最后一个 group 的数据。  

如果只做 kv 的 BoundaryExchange，score 不做：                                                                                
	rank r group 0 overlap 槽：
	    kv    = 正确收到 rank r-1 的数据   ✓
	    score = 仍然是 -inf（未更新）     ✗
	最终：kv * softmax(-inf) = kv * 0 = 0
		→ 即使 kv 正确，score 错了，overlap 贡献仍然为 0，计算结果错误
  所以 kv 和 score 必须各自做一次 BoundaryExchange，缺一不可。

> [!IMPORTANT] rank 0 的 score 初始化必须用 `-inf`，不能用 `0`
> 原 `overlap_transform` 对 score 使用 `value=float("-inf")` 初始化，使得 rank 0 位置 0 的 overlap 槽权重经 softmax 后接近 0（正确行为）。
> 若 `recv_score` 在 rank 0 时为全 0，softmax 后 overlap 槽会得到非零权重，与非 CP 行为不一致，产生训练偏差。
>
>softmax(-inf) = e^(-inf) / Z ≈ 0    ← 几乎无贡献，✓    
  softmax(0)    = e^0      / Z = 1/Z  ← 正常的正数权重，✗
>
>所以需要修复：kv 和 score 分两次 BoundaryExchange，各自使用正确的初始值。


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
		return grad_send, None, None, None, None # 新增 init_value 的 None 梯度
```

调用时分别传入正确的 init_value：
```python
# kv 边界交换：rank 0 接收全 0（与非 CP 一致）
recv_kv_slice = BoundaryExchange.apply(kv_last, cp_rank, cp_size, cp_group, 0.0)

# score 边界交换：rank 0 接收全 -inf（与非 CP 一致，避免 softmax 权重错误）
recv_score_slice = BoundaryExchange.apply(score_last, cp_rank, cp_size, cp_group, float("-inf"))
```

#### 3.3.2.3 带 cp 边界修正的 compressor 前向

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
      kv_last = kv_g[:, -1:, :, :d].contiguous() # [B, 1, 4, d]
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
      new_kv[:, 0, :ratio] = recv_kv[:, 0] # rank 0 为全 0 ✓（与非 CP 一致）
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

#### 3.3.2.4 Indexer 前向（两个 Compressor 相互独立 BoundaryExchange ）
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
      # 使用同一个 compressor_forward_with_cp，只是参数实例不同
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

      # freqs_cis 也需要用全局位置
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

#### 3.3.2.5 all-gather （复用 c128）
```python
  # 两路 AllGather，反向自动 ReduceScatter
  global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
  # [B, seq_len//4, head_dim]

  global_k_indexer = AllGatherCompressedKV.apply(local_k_indexer, cp_group)
  # [B, seq_len//4, index_head_dim]
```

#### 3.3.2.6 LiCompute CP 修正
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
      # 原始代码：base = arange(seqlen)
      # CP 修正：rank r 的 query 全局位置是 cp_rank*chunk + local_pos
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

#### 3.3.2.7 完整的 c4a attention 前向（含窗口 KV 边界通信）
> [!IMPORTANT] C4A 的 Window KV 也需要边界通信
> C4A 层的 `SparseAttention` 同样调用 `GetWindowTopkIdxs`（`model.py:518`），
> 存在与 ratio=1 层完全相同的跨 rank 窗口依赖问题。
> 原方案仅处理了 Compressor 的跨 rank 边界，遗漏了 Window KV 的边界通信。
>
> 修复：在 Window KV 投影后，执行与 3.1 节相同的 P2P 通信 + window_topk_idxs 修正。


```python
  def c4a_attention_forward_with_cp(
      attn,               # Attention 实例（compress_ratio=4）
      x,                  # [B, chunk, D]
      freqs_cis_global,
      attention_masks,
      cp_rank, cp_size, cp_group,
      chunk_size, seq_len,
      ratio=4,
      window_size =128,
  ):
      B, chunk, D = x.shape
      rd = attn.rope_head_dim

      # ── Q 投影（纯本地）────────────────────────────────────────────────
      qr = attn.q_norm(attn.wq_a(x))
      q  = attn.wq_b(qr).unflatten(-1, (attn.n_heads, attn.head_dim))
      q  = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)
      q_nope, q_rope = torch.split(q, [attn.head_dim - rd, rd], dim=-1)
      global_start = cp_rank * chunk_size
      freqs_local = freqs_cis_global[global_start : global_start + chunk]
      q_rope = apply_rotary_emb(q_rope, freqs_local)
      q = torch.cat([q_nope, q_rope], dim=-1)   # [B, chunk, n_heads, head_dim]

      # ── Window KV 投影（纯本地）───────────────────────────────────────────
      kv = attn.kv_norm(attn.wkv(x))
      kv_nope, kv_rope = torch.split(kv, [attn.head_dim - rd, rd], dim=-1)
      kv_rope = apply_rotary_emb(kv_rope, freqs_local)
      kv = torch.cat([kv_nope, kv_rope], dim=-1)   # [B, chunk, head_dim]

      # ── Window KV 边界通信（与 ratio=1 层相同，复用 3.1 节）──────────────
      recv_buf = torch.zeros_like(kv[:, :window_size, :])
      boundary_kv = window_boundary_comm(
          kv[:, -window_size:, :], recv_buf, cp_rank, cp_size, cp_group
	  ) # rank > 0: [B, 128, head_dim]；rank 0: None
	  
	  if cp_rank > 0:
		  kv_full = torch.cat([boundary_kv, kv], dim=1) # [B, 128+chunk, head_dim]
	  else:
		  kv_full = kv # [B, chunk, head_dim]
	  # ── Window topk 索引（CP 修正，复用 3.1 节）────────────────────────
	  window_topk = cp_window_topk_idxs(B, chunk, window_size, cp_rank)
	  # [B, chunk, 128]
	  
	  # offset 需要从 kv_full 实际长度算起
	  offset = kv_full.size(1) # rank 0: chunk；rank r>0: 128+chunk

      # ── Indexer（x.detach，不传梯度）───────────────────────────────────
      with torch.no_grad():
          q_indexer, k_indexer_local, weights = indexer_forward_with_cp(
              attn.indexer, x.detach(), qr.detach(), freqs_cis_global,
              cp_rank, cp_size, cp_group, chunk_size, ratio,
          )

      # ── AllGather k_indexer ─────────────────────────────────────────────
      global_k_indexer = AllGatherCompressedKV.apply(k_indexer_local, cp_group)
      # [B, seq_len//4, index_head_dim]

      # ── LiCompute：top-k 选择 （使用全局坐标 offset）──────────────────────
      compress_topk_idxs, index_score = li_compute_with_cp(
          attn.li_compute, q_indexer, global_k_indexer, weights,
          chunk_size, seq_len, cp_rank, ratio, offset,
      )
      # compress_topk_idxs: [B, chunk, topk]，索引从 offset=kv_full.size(1) 起

      # ── 主 Compressor（kv_compress，overlap=True，需 BoundaryExchange）─
      local_kv_compress = compressor_forward_with_cp(
          attn.compressor, x, freqs_cis_global,
          cp_rank, cp_size, cp_group, chunk_size, ratio,
      )   # [B, chunk//4, head_dim]

      # ── AllGather kv_compress ───────────────────────────────────────────
      global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
      # [B, seq_len//4, head_dim]

      # ── 因果裁剪（去掉未来 rank 的数据）────────────────────────────────
      valid_len = (cp_rank + 1) * chunk_size // ratio
      causal_kv_compress = global_kv_compress[:, :valid_len, :]
      # [B, (cp_rank+1)*chunk//4, head_dim]

      # ── Sparse Attention ────────────────────────────────────────────────
      # kv_full 含边界（rank>0）或不含（rank 0），compress_topk_idxs 以 offset 为基准
      o = attn.sparse_attn(q, kv_full, attn.attn_sink, causal_kv_compress, compress_topk_idxs)
      o_nope, o_rope = torch.split(o, [attn.head_dim - rd, rd], dim=-1)
      o_rope = apply_rotary_emb(o_rope, freqs_local, inverse=True)
      o = torch.cat([o_nope, o_rope], dim=-1)

      # ── LiLoss（使用 causal_kv_compress，满足因果）──────────────────────
      loss = attn.li_loss(
          q, causal_kv_compress,
          q_indexer, global_k_indexer,
          weights, compress_topk_idxs, index_score,
          attention_masks, offset,
      )

      return o, loss
```

