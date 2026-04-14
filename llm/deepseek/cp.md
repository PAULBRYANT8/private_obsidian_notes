
DeepSeek-2026 CP 切分设计方案
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

## 3.1 ratio=1：Window Attention（单向边界通信）


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
    需要 KV 来自 rank r-1 的最后 (128-k) 个 toke
    
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
  - 无需修改 SparseAttention 逻辑，只需在 forward 前拼接边界 token
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
采用这个策略不会造成死锁的原因：isend/irecv 只是向通信后端注册请求并立即返回，不阻塞。所有rank在 wait() 之前就已经同时把 send/recv 同时挂起了，后端在内和段完全比配。

  3.2 C128A CP 切分完整设计方案    
	3.2.1设计前提与约束 
	
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
  
  尾部 token 丢弃规则 
  seq_len=65536 时，S % 128 = 0，无丢弃
  seq_len=65600 时（假设），65600 % 128 = 64，每 rank 丢弃本地 chunk 末尾 64 个 token                                                                                          
  丢弃必须发生在 compressor 内部，由各 rank 独立处理本地尾部                                                                                                                             
	  3.2.2数据流设计
  输入 x: [B, chunk, D]，chunk = seq_len / cp_size
  每个 CP rank 独立执行：
  ┌────────────────────────────────-┐   
  │  x: [B, chunk, D]                                                                       │
  │       │                                                                                            │
  │       ├─ Window KV Proj                                                        │
  │       │    → local_window_kv: [B,chunk,512]                     │ ← 纯本地，无通信
  │       │                                                                                            │ 
  │       └─ Compressor (ratio=128)                                         │
  │            → 截断尾部：x[:, :chunk//128*128]                     │
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
  ┌────────────────────────────────┐
  │  kv_states = cat([local_window_kv,                               │
  │                   global_kv_compress], dim=1)                        │
  │                                                                                                  │
  │  标准 Attention(q, kv_states)                                           │ ← 无稀疏，直接全量 attention
  │  （无 Lightning Indexer）                                                 │
  └────────────────────────────────┘
                │     
                ▼ 
          输出: [B, chunk, D]
	  3.2.3通信量分析 
  local_kv_compress 形状：[B, chunk//128, 512]
                         = [1, 8192//128, 512]  
                         = [1, 64, 512]
  AllGather 后：[1, 64*cp_size, 512] = [1, 512, 512]
  通信量 = 512 × 512 × 2B(fp16) = 524,288 B ≈ 512 KB/layer（可忽略）
  
  对比：                                                                                                                                                                       
    ratio=4  kv_compress AllGather：16 MB/layer
    ratio=128 kv_compress AllGather：0.5 MB/layer  ← 便宜 32 倍  
	
	3.2.4伪代码实现
  3.2.4.1 对齐校验
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
  
  3.2.4.2 Autograd-aware AllGather（前向 AllGather，反向自动 ReduceScatter）                                                                                                         
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
   
  3.2.4.3 C128A 前向主逻辑
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
                  
 3.2.4.4 Compressor 内部尾部丢弃（确认丢弃位置）
```python
  class Compressor(nn.Module):
      def __init__(self, ratio: int = 128, ...):
          self.ratio = ratio
      def forward(self, x: torch.Tensor) -> torch.Tensor:
          # x: [B, seq_len, D]
          B, S, D = x.shape
          
          # 尾部丢弃：在 compressor 内部处理，各 rank 独立截断本地 chunk 尾部
          valid_len = (S // self.ratio) * self.ratio
          if valid_len < S:
              x = x[:, :valid_len, :]   # 丢弃末尾 S%ratio 个 token
              
          # reshape: [B, S//ratio, ratio, D] → 压缩 → [B, S//ratio, head_dim]
          x = x.reshape(B, S // self.ratio, self.ratio, D)
          kv = self.compress_proj(x)     # 线性压缩
          return kv                      # [B, S//ratio, head_dim]
```


## 3.3 ratio=4：C4A（最复杂，需分步处理）

3.2.4.4 C4A compressor 方案
 一、问题定位                                                                                                                                                                 
   
  C4A compressor 使用 overlap=True，其 overlap_transform 使第 i 个压缩 token 依赖第 i-1 个 group 的数据：
  new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]  # group i 混入 group i-1 的数据
  CP 切分后，rank r 的 compressed_token[0] 需要 rank r-1 最后一个 group 的数据，但本地看不到，导致错误地填入全 0：
  非 CP（正确）：  compressed[0 of rank r] = f(group_k,  group_{k-1})  ✓
  CP 无处理：      compressed[0 of rank r] = f(group_k,  0)            ✗

二、对齐约束
```python
  def validate_c4a_cp_alignment(seq_len: int, cp_size: int) -> None:
      if seq_len % (cp_size * 4) != 0:
          raise NotImplementedError(
              f"C4A requires seq_len % (cp_size * 4) == 0. "
              f"Got seq_len={seq_len}, cp_size={cp_size}."
          )
```

三、通用 BoundaryExchange （复用）
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

三、带 cp 边界修正的 compressor 前向

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

      # ── Step 2：截断尾部（chunk%ratio==0 时无截断） ───────────────────────
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

四、indexer 前向（两个 Compressor 相互独立 BoundaryExchange ）
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

五、all-gather （复用 c128）
```python
  # 两路 AllGather，反向自动 ReduceScatter
  global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, cp_group)
  # [B, seq_len//4, head_dim]

  global_k_indexer = AllGatherCompressedKV.apply(local_k_indexer, cp_group)
  # [B, seq_len//4, index_head_dim]
```

六、LiCompute CP 修正

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

七、完整的 c4a attention 前向

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

八、c4a 需要分成 compressor 和 indexer 的原因

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
k_indexer  只用来做“相关性排序”，不需要携带完整语义信息。维度低 -> 检索更快 -> top-k 计算的代价小
kv_compressor 要参与真正的 attention 计算，需要保留完整表征。 维度高 -> 保留语义 -> attention 质量高

3.2.4.5 c4a indexer
数据流
```
输入 x: [B, chunk, D]
          │
          ├──────────────────────────────────────────────────────────────────┐
          │  主 Compressor（overlap=True）                                    │ Indexer
          │  kv_g, score_g = project + reshape                               │（overlap=True）
          │  send = cat[kv_last, score_last]  ──isend──► rank+1              │ 同样逻辑
          │  recv ◄──irecv── rank-1           （BoundaryExchange）            │ 独立 P2P
          │  new_kv[0]    ← recv[:,:,:,:d]                                   │
          │  new_score[0] ← recv[:,:,:,d:]                                   │
          │  compress + norm + RoPE（全局位置）                               │
          ▼                                                                   ▼
  local_kv_compress [B, chunk//4, 512]            k_indexer_local [B, chunk//4, 128]
          │                                                   │
          │          AllGather（反向 ReduceScatter）           │
          ▼                                                   ▼
  global_kv_compress [B, seq//4, 512]         global_k_indexer [B, seq//4, 128]
          │                                                   │
          │                                    q_indexer [B, chunk, n_heads, 128]（本地）
          │                                    weights   [B, chunk, n_heads]（本地）
          │                                               │
          │                                    LiCompute（CP 全局坐标修正）
          │                                    → compress_topk_idxs [B, chunk, topk]
          │
          │ 因果裁剪 valid_len = (cp_rank+1)*chunk//4
          ▼
  causal_kv_compress [B, valid_len, 512]
          │
          ├── SparseAttention(q, window_kv ∥ causal_kv_compress, compress_topk_idxs)
          │
          └── LiLoss(q, causal_kv_compress, q_indexer, global_k_indexer, ...)
```
  ---
  五、各模块 CP 属性汇总

  模块                    CP 通信类型              通信量（seqlen=65536, cp=8）
  ─────────────────────────────────────────────────────────────────────────
  HC (HcPre/HcPost)       无（纯本地混合）          0
  MoE（hash 层）          无（按 token_id 路由）     0
  MoE（普通层）           无（token 独立）           0
  ratio=1 Window KV       P2P（单向边界）            ~131 KB/layer
  ratio=128 Compressor    AllGather                  ~512 KB/layer
  ratio=4 k_indexer       AllGather                  ~4 MB/layer
  ratio=4 kv_compress     AllGather                  ~16 MB/layer

  ---
  六、实现建议

  4. 在 Attention.forward() 注入 CP rank/size 信息，通过 CustomContextParallelContext patch 方式（参考现有 AscendDSAContextParallelContext）注入
  cp_rank, cp_size, cp_mesh，不修改 model 本身签名。

  5. 为 C4A 单独实现 mhc_c4a_forward_with_cp()，复用现有 allgather_sequence() 工具函数。

  6. 为 C128A 实现 mhc_c128a_forward_with_cp()，逻辑更简单（无 Indexer，直接 AllGather）。

  7. LiCompute 需要接受 offset_seqlen（rank offset）参数，用于全局因果 mask 计算，这是与当前 V3.2 CP 实现的主要差异点。

	  1. 反向梯度：AllGather 的反向自动对应 ReduceScatter（参考已有的 AllgatherOnSequence 或 DTensor Shard→Replicate 机制），无需额外处理。x






二、C128A 的 CP 方案可行性分析

  你的方案描述是：
  1. Compressor 在各 rank 本地独立运行，输出 [B, S//128//CP, 512]
  2. 稀疏注意力计算前，AllGather 压缩 KV → [B, S//128, 512]
  3. 反向 ReduceScatter 梯度
  4. 约束 S % (CP * 128) == 0，不满足直接 raise

  整体方向完全可行，但有一个关键细节需要处理。

  ---
  2.1 Compressor 本地运行：✅ 完全正确

  ratio=128 时 overlap=False（代码 model.py:114：self.overlap = compress_ratio == 4），压缩是纯局部操作：

  # Compressor.forward() for ratio=128
  kv = kv.unflatten(1, (-1, 128))          # [B, S//(128*CP), 128, head_dim]
  score = score.unflatten(1, (-1, 128)) + self.ape
  kv = (kv * score.softmax(dim=2)).sum(2)  # [B, S//(128*CP), head_dim]

  每 128 个连续 token 压缩为 1 个，组内操作，无跨 rank 依赖。在 S % (CP*128) == 0 的保证下，每个 rank
  恰好得到 S//(128*CP) 个压缩 token，无尾部截断问题。

  ---
  2.2 AllGather + ReduceScatter：✅ 可行

  local_kv_compress  # [B, S//(128*CP), 512]
  global_kv_compress = all_gather(local_kv_compress, dim=1)  # [B, S//128, 512]

  反向时，global_kv_compress 的梯度经 ReduceScatter 切回到各 rank，与 AllGather/ReduceScatter 对 KV
  矩阵的标准用法一致。

  ---
  2.3 关键细节：GetCompressTopkIdxs 的因果 mask 需要修正

  这是方案中唯一需要额外处理的地方。看当前实现：

  # model.py:435-438
  def forward(self, ratio, bsz, seqlen, offset):
      matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
      mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
      compress_topk = torch.where(mask, -1, matrix + offset)

  非 CP 时，seqlen 就是全局序列长，因果 mask 自然正确。CP 下的问题：

  - seqlen 变成了本地 chunk_size（例如 512）
  - matrix 的范围是 [0, 512//128) = [0, 4)，只有 4 个本地压缩 token
  - AllGather 之后 global_kv_compress 有 S//128 = 32 个 token（对于 seqlen=4096, cp=8）
  - 但因果 mask 仍然只覆盖了 4 个 token 的范围，前序 rank 的压缩 token 全被遗漏了

  修正方案：在 CP forward 中替换 GetCompressTopkIdxs 的调用，使 mask 用全局坐标：

  # CP 模式下的修正逻辑
  global_compress_len = global_seqlen // ratio          # S//128
  cp_rank_offset = cp_rank * chunk_size                 # rank r 的全局 token 起点

  # 本地 query i 的全局位置是 (cp_rank_offset + i)
  # 因果：可访问 global compress token j 当且仅当 j < (cp_rank_offset + i + 1) // ratio
  base_global = cp_rank_offset + torch.arange(chunk_size)  # 全局 query 坐标
  matrix = torch.arange(global_compress_len)
  mask = matrix.unsqueeze(0) >= (base_global.unsqueeze(1) + 1) // ratio
  compress_topk = torch.where(mask, -1, matrix + offset)

  注意两点：
  - offset 依然是 kv.size(1) = chunk_size（window KV 占用了前 chunk_size 个槽位），不变
  - 当 cp_rank > 0 时，前序 rank 的所有压缩 token（j < cp_rank * chunk_size // ratio）对本 rank 所有
  query 都是可见的，这个修正自动处理了这一点

  ---
  2.4 S % (CP * 128) == 0 约束：✅ 合理

  常见训练 seq_len（4096、8192、16384、32768、65536）全都是 4096 的倍数，而 128 整除 4096，因此只要 CP 是
   2 的幂（训练中几乎总是如此），约束自动满足。在 CP 初始化时加一行校验即可：

  if seq_len % (cp_degree * 128) != 0:
      raise NotImplementedError(
          f"For DeepSeek-2026 C128A CP, seq_len must be divisible by "
          f"cp_degree * 128 = {cp_degree * 128}, got seq_len={seq_len}."
      )

  ---
  三、方案总结

  ┌───────────────────────┬────────────────────────────────────────────┬─────────────────────────────┐
  │         步骤          │                    操作                    │        是否需要改动         │
  ├───────────────────────┼────────────────────────────────────────────┼─────────────────────────────┤
  │ Compressor 本地运行   │ 无跨 rank 依赖，直接本地执行               │ 无需改动                    │
  ├───────────────────────┼────────────────────────────────────────────┼─────────────────────────────┤
  │ AllGather kv_compress │ 标准 AllGather on dim=1                    │ 复用现有 allgather_sequence │
  ├───────────────────────┼────────────────────────────────────────────┼─────────────────────────────┤
  │ GetCompressTopkIdxs   │ 因果 mask 改用全局坐标                     │ 需修正（核心改动）          │
  ├───────────────────────┼────────────────────────────────────────────┼─────────────────────────────┤
  │ SparseAttention       │ kv_states = window_kv ∥ global_kv_compress │ 无需改动                    │
  ├───────────────────────┼────────────────────────────────────────────┼─────────────────────────────┤
  │ 反向 ReduceScatter    │ AllGather 的自动反向                       │ 无需改动                    │
  ├───────────────────────┼────────────────────────────────────────────┼─────────────────────────────┤
  │ 约束校验              │ S % (CP * 128) == 0                        │ 加一行 raise                │
  └───────────────────────┴────────────────────────────────────────────┴─────────────────────────────┘

  整体方案是可行且实现代价低的，主要工作量集中在 GetCompressTopkIdxs 的因果 mask
  修正和约束校验，其余复用现有的 allgather/reduce-scatter 基础设施即可。





  ---                                                                                                    
  

[^1]: 
