
DeepSeek-2026 CP 切分设计方案

  一、问题本质分析

  compress_ratios 的模式是 (1, 1, 4, 128, 4, 128, ..., 4, 128, 4)，三类层对应三种不同的序列依赖范围：

| 层类型                | 序列依赖范围                        | 通信挑战                 |
| ------------------ | ----------------------------- | -------------------- |
| ratio = 1（Window）  | 局部 window_size = 128          | 仅边界 token 通信         |
| ratio = 4（C4A）     | 全局（Lightning Indexer 选 top-k） | 压缩 KV 的全局 all-gather |
| ratio = 128（C128A） | 全局（但 token 数极少）               | 压缩 KV 的全局 all-gather |

  HC（Hyper-Connections）完全是 per-token 的局部操作，不需要任何 CP 通信。
  
  ---
  二、前提：Chunk 对齐约束（chunk 就是每个 cp rank 持有的本地序列片段）

  必要条件：chunk_size = seqlen / cp_degree 必须能被 128 整除。

  seqlen=65536, cp=8  →  chunk=8192, 8192/128=64 ✓
  seqlen=4096,  cp=8  →  chunk=512,  512/128=4   ✓
  seqlen=4096,  cp=32 →  chunk=128,  128/128=1   ✓

  这样所有层的压缩边界都与 CP 切分边界自然对齐，无碎片问题。

  ---
  三、三类层的 CP 通信设计

  3.1 ratio=1：Window Attention（单向边界通信）

  rank r-1: [..., token_{r*chunk - 128}, ..., token_{r*chunk - 1}]
                                     ↓  Send last window_size tokens
  rank r:   [token_{r*chunk}, ...]   ←  Recv (128 tokens)

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

  3.2 ratio=128：C128A（AllGather 压缩 KV，廉价）

  每 rank 本地运算:
    local_kv_compress: (B, chunk/128, head_dim)  例如 8192/128=64 个 token

  AllGather:
    global_kv_compress: (B, seqlen/128, head_dim)  例如 65536/128=512 个 token

  通信量: 512 × 512 × 2B = 512KB（可忽略）

  无 Lightning Indexer（代码中 indexer is None when ratio≠4），直接把 global_kv_compress 拼到 window KV 后面做完整 attention。

  3.3 ratio=4：C4A（最复杂，需分步处理）

  Step 1：Overlap 边界通信

  Compressor 在 ratio=4 时 overlap=True，其 overlap_transform 让每个压缩 token 消费相邻两组原始 token（前后各 4 个）。因此 rank r 的第一个压缩 token
   需要 rank r-1 最后 4 个原始 token：

  # overlap_transform: model.py:127-133
  new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]  # 前一组 shift 进来

  解法：P2P Recv 前 rank 最后 compress_ratio=4 个 token，prepend 后运行 Compressor，去除首个压缩 token（它用了借来的边界数据，不属于本 rank
  的有效输出），剩余 chunk/4 个压缩 token 是有效的。

  Step 2：AllGather k_indexer 和 kv_compress

  local_k_indexer:   (B, chunk/4, index_head_dim=128)
  local_kv_compress: (B, chunk/4, head_dim=512)

  AllGather →
  global_k_indexer:   (B, seqlen/4, 128)   通信量: 65536/4×128×2B = 4MB
  global_kv_compress: (B, seqlen/4, 512)   通信量: 65536/4×512×2B = 16MB

  Step 3：修正 LiCompute 的因果偏移

  当前 LiCompute.forward() 中：
  # model.py:554-555
  base = torch.arange(seqlen, device=device).unsqueeze(1)  # 本地 query 位置
  mask = matrix >= (base + 1) // ratio

  在 CP 下，rank r 的 query 全局位置是 r*chunk + local_pos，因此需要将 base 修正为全局坐标：

  # CP 修正：base_global = cp_rank * chunk_size + base_local
  base = (cp_rank * chunk_size + torch.arange(chunk_size)).unsqueeze(1)
  mask = matrix >= (base + 1) // ratio
  # matrix 也需扩展到全局 compressed token 数
  matrix = torch.arange(global_seqlen // ratio)

  Step 4：Sparse Attention 索引对齐

  kv_states = cat([local_window_kv, global_kv_compress], dim=1)

  其中 topk_idxs 索引的是 global_kv_compress 中的位置，offset = chunk_size（window KV 占据 [0, chunk_size) 的槽位，压缩 KV 从 chunk_size 起）。

  Step 5：LiLoss 的因果裁剪

  AllGather 后的 global_kv_compress 包含当前 rank 未来的 token（违反因果），需在 LiLoss 中裁剪：

  # 只传入 current_rank 能看到的 global_kv_compress
  valid_compress_len = (cp_rank + 1) * chunk_size // ratio
  loss = li_loss(q, global_kv_compress[:, :valid_compress_len], ...)

  ---
  四、整体数据流（C4A 层）

  输入 x: (B, chunk, D)
      │
      ├─ [P2P Recv 4 tokens from prev_rank]
      │
      ├─ 1. 计算 window KV       (B, chunk, head_dim)       ← 纯本地
      │
      ├─ 2. Compressor (含边界)  → local_kv_compress (B, chunk/4, head_dim)
      │
      ├─ 3. Indexer              → local_k_indexer  (B, chunk/4, index_head_dim)
      │
      ├─ 4. AllGather ──────────────────────────────────────────────────────┐
      │         global_kv_compress (B, seqlen/4, head_dim)                 │
      │         global_k_indexer   (B, seqlen/4, index_head_dim)           │ 反向: ReduceScatter
      │                                                                      │
      ├─ 5. LiCompute(global_k_indexer, global 因果 mask) → compress_topk_idxs
      │
      ├─ 6. SparseAttention(window_kv ∥ global_kv_compress, compress_topk_idxs)
      │
      └─ 7. LiLoss(global_kv_compress[:valid], ...)

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

  1. 在 Attention.forward() 注入 CP rank/size 信息，通过 CustomContextParallelContext patch 方式（参考现有 AscendDSAContextParallelContext）注入
  cp_rank, cp_size, cp_mesh，不修改 model 本身签名。

  2. 为 C4A 单独实现 mhc_c4a_forward_with_cp()，复用现有 allgather_sequence() 工具函数。

  3. 为 C128A 实现 mhc_c128a_forward_with_cp()，逻辑更简单（无 Indexer，直接 AllGather）。

  4. LiCompute 需要接受 offset_seqlen（rank offset）参数，用于全局因果 mask 计算，这是与当前 V3.2 CP 实现的主要差异点。

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
  压缩 token 的因果性问题
                         
  以 ratio=128 为例，压缩 token j 代表的是原始序列中第 j 组的 128 个 token，即原始位置 [j*128, (j+1)*128 
  - 1]。                                                                                                 
   
  Query 位置 i 要访问压缩 token j，因果性要求这组 128 个 token 必须全部在 i                              
  的过去（不包含未来信息）。判断条件是：
                                                                                                         
  压缩 token j 可见  ⟺  j < (i + 1) // 128
                                                                                                         
  直觉理解：(i+1) // 128 表示"到位置 i 为止，已经完整经过了几组 128 个                                   
  token"。只有已完整经过的组才能作为压缩 token 被访问。                                                  
                                                                                                         
  举例（ratio=128，全局 seqlen=4096）：

  ┌──────────────┬────────────┬─────────────────────────────────┐                                        
  │ Query 位置 i │ (i+1)//128 │      可见的压缩 token 编号      │
  ├──────────────┼────────────┼─────────────────────────────────┤                                        
  │ 0            │ 0          │ 无（第一组还没完成）            │
  ├──────────────┼────────────┼─────────────────────────────────┤
  │ 127          │ 1          │ {0}（tokens 0-127 的压缩）      │                                        
  ├──────────────┼────────────┼─────────────────────────────────┤                                        
  │ 128          │ 1          │ {0}（token 128 还没完成第二组） │                                        
  ├──────────────┼────────────┼─────────────────────────────────┤                                        
  │ 255          │ 2          │ {0, 1}                          │
  ├──────────────┼────────────┼─────────────────────────────────┤                                        
  │ 511          │ 4          │ {0, 1, 2, 3}                    │
  └──────────────┴────────────┴─────────────────────────────────┘                                        
   
  ---                                                                                                    
  原始代码的逻辑（非 CP）

  # model.py:435-438
  def forward(self, ratio, bsz, seqlen, offset):                                                         
      matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
      mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio                                 
      compress_topk = torch.where(mask, -1, matrix + offset)                                             
   
  具体展开（ratio=128, seqlen=4096, offset=4096）：                                                      
                  
  matrix: shape [4096, 32]，每行都是 [0, 1, 2, ..., 31]（32 个压缩 token 编号）                          
                                                                                                         
  torch.arange(1, 4097).unsqueeze(1) // 128 = [[0], [0], ..., [1], [1], ..., [32]]                       
                                                  ^i=0   ^i=127 ^i=128        ^i=4095                    
                                                                                                         
  mask[i][j] = (j >= (i+1)//128)  ← True 表示 token j 对 query i 不可见（未来）                          
                                                                                                         
  compress_topk[i][j] = -1           if mask=True  （不可见，填 -1）                                     
                       = j + offset  if mask=False （可见，填实际 kv_states 中的索引）
                                                                                                         
  offset = kv.size(1) = window KV 的长度，因为 kv_states = cat([window_kv, kv_compress]), 压缩 token j 在
   kv_states 中的实际位置是 offset + j。                                                                 
                                                                                                         
  ---             
  CP 下的问题
                                                                                                         
  以 seqlen=4096, cp=8, chunk_size=512, ratio=128 为例：
                                                                                                         
  - Rank 0 持有 token 0-511，本地位置 i 对应全局位置 i                                                   
  - Rank 1 持有 token 512-1023，本地位置 i 对应全局位置 512+i
                                                                                                         
  非 CP 的代码在 Rank 1 上运行时，seqlen 变成了本地的 chunk_size=512：                                   
                                                                                                         
  matrix = [0, 1, 2, 3]（只有 4 个本地压缩 token）                                                       
  mask[i][j] = (j >= (i+1) // 128)  ← 用的是本地位置 i，不是全局位置                                     
                                                                                                         
  Rank 1 本地位置 i=0 时（实际是全局 token 512）：                                                       
  (0+1) // 128 = 0  →  所有 j>=0 都被掩盖  →  compress_topk = [-1, -1, -1, -1]                           
                                                                                                         
  结果：Rank 1 的第一个 token 看不到任何压缩 token！                                                     
                                                                                                         
  但实际上全局位置 512 的 token 应该能看到：                                                             
  (512+1) // 128 = 4  →  压缩 token {0, 1, 2, 3} 均可见                                                  
                                                                                                         
  也就是说，Rank 0 那 512 个 token 对应的 4 个压缩 token（来自全局 kv_compress 的前 4                    
  项）本应全部可见，却被错误地掩盖了。                                                                   
                                                                                                         
  ---                                                                                                    
  修正方案        
          
  AllGather 之后，global_kv_compress 包含全局 32 个压缩 token，索引 0-31。需要用全局位置来计算因果 mask：
                                                                                                         
  global_compress_len = global_seqlen // ratio      # 4096//128 = 32
  cp_rank_offset      = cp_rank * chunk_size        # rank 1: 512                                        
                                                                                                         
  # 本地 query i 的全局坐标 = cp_rank_offset + i                                                         
  base_global = cp_rank_offset + torch.arange(chunk_size)  # [512, 513, ..., 1023]                       
                                                                                                         
  matrix = torch.arange(global_compress_len)        # [0, 1, ..., 31]
                                                                                                         
  # mask[i][j] = True 表示压缩 token j 对本地 query i 不可见                                             
  mask = matrix.unsqueeze(0) >= (base_global.unsqueeze(1) + 1) // ratio
                                                                                                         
  compress_topk = torch.where(mask, -1, matrix + offset)
                                                                                                         
  Rank 1 本地位置 i=0 时（全局 512）的效果：                                                             
  base_global[0] = 512
  (512 + 1) // 128 = 4                                                                                   
                      
  mask[0] = [0>=4, 1>=4, 2>=4, 3>=4, 4>=4, 5>=4, ...]                                                    
          = [F,    F,    F,    F,    T,    T,   ...]                                                     
                                                                                                         
  compress_topk[0] = [offset+0, offset+1, offset+2, offset+3, -1, -1, ..., -1]                           
                                                                              
  压缩 token 0, 1, 2, 3（来自全局 kv_compress 位置 offset+0 至 offset+3）均可见，后续的（来自 Rank 1     
  自身的 token 组尚未完成）均被掩盖。这正是我们期望的结果。                                              
   
  ---                                                                                                    
  直觉总结        
          
  ┌───────────────────────┬─────────────────┬─────────────────────┬─────────────────────────────────┐
  │                       │      非 CP      │    CP 下错误行为    │            CP 修正后            │    
  ├───────────────────────┼─────────────────┼─────────────────────┼─────────────────────────────────┤
  │ matrix 的范围         │ [0, S//128)     │ [0, chunk//128)     │ [0, S//128) 全局                │    
  │                       │ 全局            │ 本地                │                                 │
  ├───────────────────────┼─────────────────┼─────────────────────┼─────────────────────────────────┤    
  │ 因果 base             │ arange(S)       │ arange(chunk)       │ cp_rank*chunk + arange(chunk)   │    
  │                       │ 全局位置        │ 本地位置            │ 全局位置                        │    
  ├───────────────────────┼─────────────────┼─────────────────────┼─────────────────────────────────┤    
  │ Rank 1 首 token       │ 正确 4          │ 错误 0              │ 正确 4                          │
  │ 可见压缩数            │                 │                     │                                 │    
  └───────────────────────┴─────────────────┴─────────────────────┴─────────────────────────────────┘