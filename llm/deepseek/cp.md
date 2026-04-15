
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

  compress_ratios 的模式是 (1, 1, 4, 128, 4, 128, ..., 4, 128, 4)，三类层对应三种不同的序列依赖范围：

| 层类型                | 序列依赖范围                        | 通信挑战                 |
| ------------------ | ----------------------------- | -------------------- |
| ratio = 1（Window）  | 局部 window_size = 128          | 仅边界 token 通信         |
| ratio = 4（C4A）     | 全局（Lightning Indexer 选 top-k） | 压缩 KV 的全局 all-gather |
| ratio = 128（C128A） | 全局（但 token 数极少）               | 压缩 KV 的全局 all-gather |

  HC（Hyper-Connections）完全是 per-token 的局部操作，不需要任何 CP 通信。
  
  ---
# 二、前提：Chunk 对齐约束（chunk 就是每个 cp rank 持有的本地序列片段）

  必要条件：chunk_size = seqlen / cp_degree 必须能被 128 整除。

  seqlen=65536, cp=8  →  chunk=8192, 8192/128=64 ✓
  seqlen=4096,  cp=8  →  chunk=512,  512/128=4   ✓
  seqlen=4096,  cp=32 →  chunk=128,  128/128=1   ✓

  这样所有层的压缩边界都与 CP 切分边界自然对齐，无碎片问题。
  
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
### 3.2.1 设计前提与约束

  对齐约束（必要条件）
	chunk_size = seq_len / cp_size
  有效压缩要求：chunk_size % 128 == 0
  等价于：      seq_len % (cp_size * 128) == 0                                                                                                            
  验证：                                                     
    seq_len=65536, cp=8  → chunk=8192,  8192%128=0  ✓ 
    seq_len=4096,  cp=4  → chunk=1024,  1024%128=0  ✓ 
    seq_len=4096,  cp=32 → chunk=128,   128%128=0   ✓    
    seq_len=4096,  cp=64 → chunk=64,    64%128=64   ✗ → NotImplementedError
  常见训练序列（4096 的整数倍）在常用 CP degree（≤32）下均自然满足，约束实际不会触发。

  约束 S % (CP * 128) == 0，不满足直接 raise
  
  尾部 token 丢弃规则 （<mark style="background:#b1ffff">理论上应该用不上，我们这里对 seq_len 的大小有明确的约束</mark>）
  seq_len=65536 时，S % 128 = 0，无丢弃
  seq_len=65600 时（假设），65600 % 128 = 64，每 rank 丢弃本地 chunk 末尾 64 个 token                                                                                          
  如果要丢弃必须发生在 compressor 内部，由各 rank 独立处理本地尾部。

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
  │            → 截断尾部：x[:, :chunk//128*128]                     │ ← 本次设计中应该用不上
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

#### 3.2.5.1 对齐校验
```python
def validate_c128a_cp_alignment(seq_len: int, cp_size: int) -> None:
      ratio = 128
      if seq_len % (cp_size * ratio) != 0:
          raise NotImplementedError(
              f"C128A requires seq_len % (cp_size * {ratio}) == 0. "
              f"Got seq_len={seq_len}, cp_size={cp_size}, "
              f"remainder={seq_len % (cp_size * ratio)}. "
              f"Use seq_len that is a multiple of {cp_size * ratio}."
          )
```

#### 3.2.5.2 Autograd-aware AllGather(前向 All-Gather，反向自动 Reduce-Scatter)
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

#### 3.2.5.3 C128A 的 compress_topk_idxs 全局坐标修正

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

```python
  def c128a_forward_with_cp(
      x: torch.Tensor,              # [B, chunk, D]
      compressor,                   # ratio=128 Compressor 实例
      window_proj,                  # Window KV 投影
      attention,                    # 标准 MultiheadAttention
      cp_rank: int,
      cp_size: int,
      cp_group: dist.ProcessGroup,
      seq_len: int,
      window_size: int = 128,
  ) -> torch.Tensor:
      B, chunk, D = x.shape
      ratio = 128
      
      # ── 0. 校验（仅在首次调用或 debug 模式下执行）──────────────────────────  
      # validate_c128a_cp_alignment(seq_len, cp_size)  # 建议在模型初始化时调用
      
      # ── 1. 本地 Window KV（纯本地，无通信）────────────────────────────────
      # 仅需本 rank 的 chunk，借助 ratio=1 层的 P2P 边界 token 处理
      local_window_kv = window_proj(x)   # [B, chunk, head_dim=512]
      
      # ── 2. 本地压缩（compressor 在本地 chunk 上独立运行)────────────────────
      # compressor 内部处理尾部丢弃
      #   valid_len = (chunk // ratio) * ratio  （一般 chunk%128==0，无丢弃）
      #   x_valid   = x[:, :valid_len, :]
      #   压缩后输出: [B, chunk//ratio, head_dim]
      local_kv_compress = compressor(x)  # [B, chunk//128, 512]
      
      # ── 3. AllGather 压缩 KV（通信量 ~512KB，可忽略)───────────────────────
      # 反向时自动执行 ReduceScatter
      global_kv_compress = allgather_kv_compress(
          local_kv_compress, group=cp_group
      )  # [B, seq_len//128, 512]
      
      # ── 4. 因果有效性裁剪（AllGather 包含了未来 rank 的数据）──────────────
      # C128A 无 Lightning Indexer，做完整 attention，但仍需遵守因果约束:
      # rank r 的 query 最远只能看到全局位置 (r+1)*chunk - 1 对应的压缩 KV
      # 即最多看到前 (cp_rank+1)*chunk//128 个压缩 token
      valid_compress_len = (cp_rank + 1) * (chunk // ratio)
      causal_kv_compress = global_kv_compress[:, :valid_compress_len, :]
      # [B, (cp_rank+1)*chunk//128, 512]
      # rank 0: [B, chunk//128, 512]
      # rank 7: [B, seq_len//128, 512]  （最后一个 rank 看到全部）
      
      # ── 5. 拼接 Window KV 与压缩 KV─────────────────────────────────────
      # kv_states: [B, chunk + valid_compress_len, 512]
      kv_states = torch.cat([local_window_kv, causal_kv_compress], dim=1)
      
      # ── 6. 标准 Attention（无稀疏，无 Indexer)────────────────────────────
      # q:         [B, chunk, head_dim]
      # kv_states: [B, chunk + valid_compress_len, head_dim]
      q = query_proj(x)
      out = attention(q, kv_states)  # [B, chunk, D]
      return out     
```


## 3.3 ratio=4：C4A（最复杂，需分步处理）

### 3.3.1 C4A 需要分成 Compressor 和 Indexer 的原因

```
compressor：将原始 kv 压缩；
	输出两路：kv_compressor  [S/4, 512]  -> 实际 attention 用的 KV
			  k_indexer  [S/4, 128]  ->  检索用的轻量 kv

indexer：在压缩 KV 中找出最相关的 top-k；
	q    [chunk, head_dim]
		│
	k_indexer [S/4, 128]  <-  轻量 key
		│
	计算相关性分数（无 softmax，轻量）
		│
	top-k 位置索引
		│
	用索引从  kv_compressor 中取出 top-k 行
		│
	只对这 k 个位置做 attention
```
<mark style="background:#affad1">k_indexer  只用来做“相关性排序”，不需要携带完整语义信息</mark>。维度低 -> 检索更快 -> top-k 计算的代价小
<mark style="background:#affad1">kv_compressor 要参与真正的 attention 计算，需要保留完整表征</mark>。 维度高 -> 保留语义 -> attention 质量高

### 3.3.2 C4A CP 方案设计
#### 3.3.2.1问题定位

  C4A compressor 使用 overlap=True，其 overlap_transform 使第 i 个压缩 token 依赖第 i-1 个 group 的数据：
	  new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]    # group i 混入 group i-1 的数据

  CP 切分后，rank r 的 compressed_token[0] 需要 rank r-1 最后一个 group 的数据，但本地看不到，导致错误地填入全 0：
	  非 CP（正确）：  compressed[0 of rank r] = f(group_k,  group_{k-1})  ✓
	  CP 无处理：      compressed[0 of rank r] = f(group_k,  0)            ✗

#### 3.3.2.2 对齐约束（这里应该可以省略，直接在C128A Compressor 中对齐就可以了）
```python
  def validate_c4a_cp_alignment(seq_len: int, cp_size: int) -> None:
      if seq_len % (cp_size * 4) != 0:
          raise NotImplementedError(
              f"C4A requires seq_len % (cp_size * 4) == 0. "
              f"Got seq_len={seq_len}, cp_size={cp_size}."
          )
```

#### 3.3.2.3 通用 BoundaryExchange （复用）
```python
  class BoundaryExchange(torch.autograd.Function):
      @staticmethod
      def forward(ctx, send_tensor, rank, cp_size, group):
          ctx.rank, ctx.cp_size, ctx.group = rank, cp_size, group
          recv_buf = torch.zeros_like(send_tensor)
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
          return grad_send, None, None, None
```

#### 3.3.2.4 带 cp 边界修正的 compressor 前向
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

      # ── Step 2：截断尾部（chunk%ratio==0 时无截断，这一步还需要吗？前面不是有对齐了吗？对不齐不就直接报错了）───────────────────────
      cutoff = (chunk // ratio) * ratio
      if cutoff < chunk:
          kv    = kv[:, :cutoff]
          score = score[:, :cutoff]

      # ── Step 3：reshape 为 group 视图 ────────────────────────────────────
      kv_g    = kv.unflatten(1, (-1, ratio))  # [B, chunk//4, 4, 2*d]
      score_g = score.unflatten(1, (-1, ratio)) + compressor.ape  # [B, chunk//4, 4, 2*d]

      # ── Step 4：P2P 边界交换（kv + score 拼在一起，一次通信）────────────
      # 发送：本 rank 末尾 group 的前 d 维（将被 shift 给下一 rank 的位置 0）
      kv_last    = kv_g[:, -1:, :, :d]     # [B, 1, 4, d]
      score_last = score_g[:, -1:, :, :d]  # [B, 1, 4, d]
      send_slice = torch.cat([kv_last, score_last], dim=-1).contiguous()  # [B, 1, 4, 2*d]

      recv_slice = BoundaryExchange.apply(send_slice, cp_rank, cp_size, cp_group)
      # recv_slice: [B, 1, 4, 2*d]，rank 0 为全 0（kv 部分）和全 0（score 部分）

      recv_kv    = recv_slice[:, :, :, :d]   # [B, 1, 4, d]
      recv_score = recv_slice[:, :, :, d:]   # [B, 1, 4, d]

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
      new_kv[:, 0, :ratio]    = recv_kv[:, 0]    # rank 0 为全 0 ✓
      new_score[:, 0, :ratio] = recv_score[:, 0]  # rank 0 为全 0（softmax 后趋近均匀，
                                                  # 行为与非 CP rank 0 一致）

      # ── Step 6：压缩 ─────────────────────────────────────────────────────
      out = (new_kv * new_score.softmax(dim=2)).sum(dim=2)  # [B, chunk//4, d]
      out = compressor.norm(out.to(dtype))

      # ── Step 7：RoPE（使用全局位置） ─────────────────────────────────────
      # 非 CP：freqs_cis[:cutoff:ratio]（局部 0~chunk 位置）
      # CP：  freqs_cis[cp_rank*chunk : (cp_rank+1)*chunk : ratio]（全局位置）
      global_start = cp_rank * chunk_size
      freqs_compress = freqs_cis_global[global_start : global_start + cutoff : ratio]
      kv_rot = apply_rotary_emb(out[..., -compressor.rope_head_dim:], freqs_compress)
      out = torch.cat([out[..., :-compressor.rope_head_dim], kv_rot], dim=-1)

      return out   # [B, chunk//4, head_dim]
```

#### 3.3.2.5 Indexer 前向（两个 Compressor 相互独立 BoundaryExchange ）
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

#### 3.3.2.6 all-gather （复用 c128）
```python
  # 两路 AllGather，反向自动 ReduceScatter
  global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
  # [B, seq_len//4, head_dim]

  global_k_indexer = AllGatherCompressedKV.apply(local_k_indexer, cp_group)
  # [B, seq_len//4, index_head_dim]
```

#### 3.3.2.7 LiCompute CP 修正
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

#### 3.3.2.8 完整的 c4a attention 前向
```python
  def c4a_attention_forward_with_cp(
      attn,               # Attention 实例（compress_ratio=4）
      x,                  # [B, chunk, D]
      freqs_cis_global,
      attention_masks,
      cp_rank, cp_size, cp_group,
      chunk_size, seq_len,
      ratio=4,
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

      # ── Window KV（纯本地）─────────────────────────────────────────────
      kv = attn.kv_norm(attn.wkv(x))
      kv_nope, kv_rope = torch.split(kv, [attn.head_dim - rd, rd], dim=-1)
      kv_rope = apply_rotary_emb(kv_rope, freqs_local)
      kv = torch.cat([kv_nope, kv_rope], dim=-1)   # [B, chunk, head_dim]

      offset = kv.size(1)   # = chunk_size，window KV 占据 [0, chunk_size)

      # ── Indexer（x.detach，不传梯度）───────────────────────────────────
      with torch.no_grad():
          q_indexer, k_indexer_local, weights = indexer_forward_with_cp(
              attn.indexer, x.detach(), qr.detach(), freqs_cis_global,
              cp_rank, cp_size, cp_group, chunk_size, ratio,
          )

      # ── AllGather k_indexer ─────────────────────────────────────────────
      global_k_indexer = AllGatherCompressedKV.apply(k_indexer_local, cp_group)
      # [B, seq_len//4, index_head_dim]

      # ── LiCompute：top-k 选择 ───────────────────────────────────────────
      compress_topk_idxs, index_score = li_compute_with_cp(
          attn.li_compute, q_indexer, global_k_indexer, weights,
          chunk_size, seq_len, cp_rank, ratio, offset,
      )
      # compress_topk_idxs: [B, chunk, topk]，索引从 offset 起

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
      # kv_states = [window_kv ∥ causal_kv_compress]
      # compress_topk_idxs 索引从 offset=chunk_size 起，指向 causal_kv_compress 的位置
      o = attn.sparse_attn(q, kv, attn.attn_sink, causal_kv_compress, compress_topk_idxs)
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

