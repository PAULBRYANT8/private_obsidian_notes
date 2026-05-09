一、C1A / C4A / C128A 的本质区别

  1.1 层分布

  args.py 的 compress_ratios 决定了每层的类型：

  compress_ratios = (1, 1, 4, 128, 4, 128, 4, 128, ...)
  # 层 0,1 → C1A；层 2,4,6,... → C4A；层 3,5,7,... → C128A

  1.2 三种架构对比（从代码结构出发）

  C1A（compress_ratio=1）— 纯滑动窗口注意力

  PreAttention:
    wq_a/wq_b → Q
    wkv/kv_norm → KV（无 compressor，无 indexer）
    返回: (q, kv, None, None, None, None, offset=0)

  InnerAttention:
    sparse_attn（无 li_compute）

  注意力模式: Q_i 只看最近 window_size=128 个 KV token
  SFA kernel:  ori_mask_mode=4（band 模式），cmp_kv=None

  C128A（compress_ratio=128）— 滑动窗口 + 静态压缩 KV

  PreAttention:
    wq_a/wq_b → Q
    wkv/kv_norm → 窗口 KV
    compressor_128（overlap=False）→ kv_compress
    返回: (q, kv, kv_compress, None, None, None, offset)

  InnerAttention:
    sparse_attn（无 li_compute，压缩 KV 全量可见）

  注意力模式: 窗口注意力 + 因果范围内所有压缩 KV 全部可见
  SFA kernel:  cmp_ratio=128，cmp_sparse_indices=None（无 top-k）

  C4A（compress_ratio=4）— 滑动窗口 + 动态稀疏压缩 KV + Lightning Indexer

  PreAttention:
    wq_a/wq_b → Q
    wkv/kv_norm → 窗口 KV
    compressor（overlap=True，ratio=4）→ kv_compress
    indexer → (q_indexer, k_indexer, weights)
    返回: (q, kv, kv_compress, q_indexer, k_indexer, weights, offset)

  InnerAttention:
    li_compute → compress_topk_idxs, index_score（动态 top-k 选取）
    sparse_attn（只关注 top-k 个压缩 KV）
    → 还有 DSAIndexerLoss 辅助训练

  注意力模式: 窗口注意力 + 动态 top-k 个稀疏压缩 KV
  SFA kernel:  cmp_ratio=4，cmp_sparse_indices=compress_topk_idxs（非 None）

  1.3 C4A 独有的 Overlap Compressor

  这是 C4A 与 C128A 最根本的区别。Compressor 的 overlap=True（仅 ratio=4 时）：

  # Compressor.__init__
  self.overlap = compress_ratio == 4   # 只有C4A才True
  coff = 1 + self.overlap              # C128A: coff=1; C4A: coff=2
  self.wkv   = nn.Linear(dim, coff * head_dim)  # C4A输出 2D，C128A输出 D
  self.wgate = nn.Linear(dim, coff * head_dim)
  self.ape   = nn.Parameter(torch.empty(compress_ratio, coff * head_dim))

  # Compressor.forward() 核心路径
  kv    = self.wkv(x)                           # C4A: [b, s, 2D]; C128A: [b, s, D]
  score = self.wgate(x)
  kv    = kv.unflatten(1, (-1, ratio))          # [b, s//ratio, ratio, 2D or D]
  score = score.unflatten(1, (-1, ratio)) + self.ape
  if overlap:                                    # 只有C4A执行
      kv    = self.overlap_transform(kv, 0)      # → [b, s//r, 2*ratio, D]
      score = self.overlap_transform(score, float("-inf"))
  kv = (kv * score.softmax(dim=2)).sum(dim=2)  # 加权聚合

  def overlap_transform(self, tensor, value=0):
      # tensor: [b, s, ratio, 2D]  →  new_tensor: [b, s, 2*ratio, D]
      new_tensor = tensor.new_full((b, s, 2*ratio, D), value)
      new_tensor[:, :,    ratio:] = tensor[:, :,  :, D:]   # 位置 ratio~2r: 本 block normal 投影
      new_tensor[:, 1:,   :ratio] = tensor[:, :-1, :, :D]  # 位置 0~ratio: 上一 block 的 overlap 投影
      # → block 0 的位置 0~ratio 保持 value（kv=0, score=-inf）

  结果：每个 compressed token 聚合了本 block（ratio=4个token）和上一个 block（4个token）的信息，感受野翻倍。这造成了
  block t 依赖 block t-1 的跨 block 依赖——这正是 C4A CP 切分最难的地方。

  ---
  二、当前 CP 实现（C1A / C128A）的逻辑回顾
  
  Column 1: 核心操作
  C1A CP: P2P BoundaryExchange（127个窗口KV token）       
  C128A CP: BoundaryExchange（窗口KV）+ AllGather（压缩KV）
  ────────────────────────────────────────
  Column 1: 因果索引构建
  C1A CP: 无（SFA band模式自动处理）
  C128A CP: get_c128a_compress_topk_idxs()（全局坐标）
  ────────────────────────────────────────
  Column 1: SFA rank>0处理
  C1A CP: 直接传 [boundary_kv || local_kv]
  C128A CP: _c128a_cp_sfa_with_global_positions()：填零前缀恢复全局坐标

  SFA 内核对全局坐标的要求（来自 _c128a_cp_sfa_with_global_positions 的注释）：

  # SFA内核计算每个query的有效位置为 (S2-S1) + local_i
  # 只需把 ori_kv 填零到 global_start 长度，SFA 会自动把窗口对齐到正确的全局位置
  kv_prefix = torch.zeros(bsz, kv_prefix_len, ...)  # kv_prefix_len = global_start - n_boundary
  kv_padded = torch.cat([kv_prefix, kv_states], dim=1)

  ---
  三、实现 C4A CP 切分需要做什么
  
  相比 C128A，C4A 额外引入了 3 个新问题，加上需要处理的 2 个已有问题的变体：

  3.1 新问题①：Overlap 边界交换（最核心，C4A 独有）

  症结：overlap_transform 在 Compressor.forward() 内部，对 block 0 用 value=0（kv）和 float("-inf")（score）填充。CP
  切分后，rank r（r>0）的 block 0 本应使用 rank r-1 最后 4 个 raw token 的 overlap 投影，而不是 0。

  推荐方案：扩展输入法（不修改 Compressor 主体）

  # 在 c4a_forward_with_cp() 中，调用 pre_attn 之前：

  ratio = 4
  global_start = context.rank * context.chunk_size
  # 1. P2P 交换：rank r-1 的最后 ratio 个 raw hidden states
  boundary_x = BoundaryExchange.apply(
      x[:, -ratio:, :].contiguous(),
      BoundaryExchangeInfo(rank=context.rank, cp_size=context.size,
                           group=context.group, init_value=0.0)
  )
  # boundary_x: [b, ratio, dim]，rank=0 时为全零

  # 2. 拼接成扩展输入
  x_extended = torch.cat([boundary_x, x], dim=1)  # [b, chunk+ratio, dim]

  # 3. 对应的 freqs_cis 也要扩展（从全局表中取）
  freqs_ext_start = max(0, global_start - ratio)
  freqs_ext_end   = global_start + chunk
  freqs_extended  = freqs_cis_global[freqs_ext_start:freqs_ext_end].to(x.device)

  # 4. 用扩展输入运行 compressor
  kv_compress_ext = modules.pre_attn.compressor(x_extended, freqs_extended)
  # shape: [b, (chunk+ratio)//ratio, D] = [b, chunk//ratio+1, D]

  # 5. 丢弃第 0 个（由 boundary 充当"预热"），保留 rank r 自己的
  local_kv_compress = kv_compress_ext[:, 1:, :]   # [b, chunk//ratio, D]

  这样 local_kv_compress 的每一个 compressed token 都有正确的 overlap（来自前驱 block），与非 CP 的计算完全等价。

  同样地，Indexer 中的 compressor 也用了 overlap，也需要相同处理。

  3.2 新问题②：AllGather k_indexer

  LiCompute 需要在全局所有压缩 key vectors 上做 topk 选取。CP 切分后每个 rank 只有 chunk//ratio 个 local k_indexer：

  # 类似 AllGatherCompressedKV，新增：
  global_k_indexer = AllGatherCompressedKV.apply(local_k_indexer, context.group)
  # shape: [b, cp_size * chunk//ratio, n_index_heads, index_head_dim]

  valid_k_len = (context.rank + 1) * (chunk // ratio)
  causal_k_indexer = global_k_indexer[:, :valid_k_len, ...]

  3.3 新问题③：全局因果 LiCompute

  非 CP 时 LiCompute.forward() 的因果 mask 基于 local 序列位置。CP 后需用全局坐标重建 mask（参考
  get_c128a_compress_topk_idxs 的实现思路）：

  def get_c4a_compress_topk_idxs(bsz, chunk, ratio, cp_rank, causal_kv_len, topk, offset, device):
      """
      query i（全局位置 cp_rank*chunk + i）能看到的最大 compressed index 为
      (cp_rank*chunk + i + 1) // ratio - 1（即严格因果）
      """
      global_base = cp_rank * chunk + torch.arange(chunk, device=device)  # [chunk]
      max_visible = (global_base + 1) // ratio  # 每个 query 最多能看的压缩位置数

      # score 矩阵: [chunk, causal_kv_len]
      # 在全局 LiCompute 产出 topk_idxs 后，将超过 max_visible 的置为 -1
      # 并加 offset（= kv_full.size(1) = (window_size-1) + chunk）
      ...
      return compress_topk_idxs  # [b, chunk, topk]

  具体地，在 c4a_forward_with_cp() 内：

  # LiCompute 用 global k_indexer 计算 topk（需修改 causal mask 逻辑）
  compress_topk_idxs, index_score = modules.inner_attn.li_compute(
      q_indexer, causal_k_indexer, weights,
      seqlen=valid_k_len,    # 全局因果 kv 长度
      offset=offset
  )
  # 再用全局坐标修正掉超出因果范围的 idx
  compress_topk_idxs = apply_global_causal_mask_c4a(
      compress_topk_idxs, cp_rank=context.rank, chunk=chunk, ratio=ratio
  )

  3.4 已有问题变体①：AllGather kv_compress（与 C128A 相同）

  global_kv_compress = AllGatherCompressedKV.apply(local_kv_compress, context.group)
  valid_compress_len  = (context.rank + 1) * (chunk // ratio)
  causal_kv_compress  = global_kv_compress[:, :valid_compress_len, :]

  3.5 已有问题变体②：SFA Adapter 补充 C4A rank>0 分支

  deepseek_v4_sfa.py 中 sdpa_to_sfa_adapter 已有 C128A 和 C1A 的 rank>0 分支，但 C4A 缺失：

  def sdpa_to_sfa_adapter(self, query_states, kv_states, attn_sink,
                           kv_compress=None, compress_topk_idxs=None):
      cp_rank = getattr(self, "cp_rank", 0)
      if cp_rank > 0:
          if self.compress_ratio == 128:
              return _c128a_cp_sfa_with_global_positions(...)
          if self.compress_ratio == 1:
              return npu_sparse_attn_shared_kv(...)
          # ← C4A 的 rank>0 分支缺失！需新增：
          if self.compress_ratio == 4:
              return _c4a_cp_sfa_with_global_positions(
                  self, query_states, kv_states, attn_sink,
                  kv_compress, compress_topk_idxs
              )
      # rank=0 的 C4A 已正确处理（走下面的通用路径）

  新增函数 _c4a_cp_sfa_with_global_positions：

  def _c4a_cp_sfa_with_global_positions(self, query_states, kv_states, attn_sink,
                                          kv_compress, compress_topk_idxs):
      """
      C4A CP rank>0：与 C128A 一样，填零前缀恢复全局窗口 KV 坐标。
      区别：compress_topk_idxs 非 None（已是全局偏移索引），直接传入。
      """
      cp_rank = getattr(self, "cp_rank", 0)
      n_boundary = self.window_size - 1
      bsz, seq_len, _n_heads, _head_dim = query_states.shape
      global_start = cp_rank * seq_len

      kv_prefix_len = global_start - n_boundary
      kv_prefix = torch.zeros(bsz, kv_prefix_len, kv_states.size(-1),
                               dtype=kv_states.dtype, device=kv_states.device)
      kv_padded = torch.cat([kv_prefix, kv_states], dim=1)

      return npu_sparse_attn_shared_kv(
          query=query_states,
          ori_kv=kv_padded,
          cmp_kv=kv_compress,
          cmp_sparse_indices=compress_topk_idxs,   # C4A 有 topk 索引
          sinks=attn_sink.float(),
          softmax_scale=self.softmax_scale,
          cmp_ratio=self.compress_ratio,            # =4
      )

  ---
  四、实现总结：需新增/修改的内容

  新增：c4a_forward_with_cp() 函数骨架

  def c4a_forward_with_cp(modules, x, freqs_cis_global, attention_masks, context):
      """ratio=4 C4A CP forward."""
      bsz, chunk, _ = x.shape
      ratio = 4
      global_start = context.rank * context.chunk_size

      # ① Overlap 边界交换：获取前驱 rank 的最后 ratio 个 raw token
      boundary_x = BoundaryExchange.apply(
          x[:, -ratio:, :].contiguous(),
          BoundaryExchangeInfo(context.rank, context.size, context.group, 0.0)
      )

      # ② 构造扩展输入，修正 Compressor 的 overlap 依赖
      x_extended  = torch.cat([boundary_x, x], dim=1)
      freqs_ext   = freqs_cis_global[max(0, global_start-ratio) : global_start+chunk].to(x.device)
      freqs_local = freqs_cis_global[global_start : global_start+chunk].to(x.device)

      # ③ 调用 pre_attn（其内部 compressor 和 indexer 的 compressor 都需要传扩展输入）
      #    需要对 pre_attn 做小改造，支持传入扩展的 x 和 freqs 用于 compressor
      q, local_window_kv, local_kv_compress, q_indexer, local_k_indexer, weights, _ = \
          modules.pre_attn(x, freqs_local, None,
                           x_extended=x_extended, freqs_extended=freqs_ext)

      # ④ 窗口 KV 的边界交换（与 C1A/C128A 相同）
      kv_full = _exchange_and_concat_boundary_kv(local_window_kv, context)
      offset  = kv_full.size(1)  # (window_size - 1) + chunk

      # ⑤ AllGather kv_compress
      global_kv_compress   = AllGatherCompressedKV.apply(local_kv_compress, context.group)
      valid_compress_len   = (context.rank + 1) * (chunk // ratio)
      causal_kv_compress   = global_kv_compress[:, :valid_compress_len, :]

      # ⑥ AllGather k_indexer
      global_k_indexer  = AllGatherCompressedKV.apply(local_k_indexer, context.group)
      causal_k_indexer  = global_k_indexer[:, :valid_compress_len, ...]

      # ⑦ 全局 LiCompute（带全局因果 mask）
      compress_topk_idxs, index_score = modules.inner_attn.li_compute(
          q_indexer, causal_k_indexer, weights,
          seqlen=valid_compress_len, offset=offset
      )
      compress_topk_idxs = _apply_global_causal_mask_c4a(
          compress_topk_idxs, context.rank, chunk, ratio
      )

      # ⑧ Sparse attention（SFA adapter 的 C4A rank>0 分支处理坐标偏移）
      o = modules.inner_attn.sparse_attn(
          q, kv_full, modules.inner_attn.attn_sink,
          causal_kv_compress, compress_topk_idxs=compress_topk_idxs
      )

      n_local_groups = modules.post_attn.n_groups // (modules.pre_attn.n_heads // q.shape[2])
      x_out = modules.post_attn(o, freqs_local, bsz, chunk, n_local_groups)

      return AttentionForwardOutput(
          x=x_out, compress_topk_idxs=compress_topk_idxs, offset=offset,
          q=q, kv_compress=causal_kv_compress, attention_masks=attention_masks,
          index_score=index_score, q_indexer=q_indexer,
          k_indexer=causal_k_indexer, weights=weights,
      )

  改动文件清单

  ┌────────────────────┬───────────────────────────────────────────────────────────────────────────────────┬──────┐
  │        文件        │                                       改动                                        │ 类型 │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────┤
  │ deepseek_v4_cp.py  │ 新增 c4a_forward_with_cp()                                                        │ 新增 │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────┤
  │ deepseek_v4_cp.py  │ attention_forward_with_cp dispatcher 中 else 改为 elif/elif                       │ 修改 │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────┤
  │ deepseek_v4_sfa.py │ 新增 _c4a_cp_sfa_with_global_positions()                                          │ 新增 │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────┤
  │ deepseek_v4_sfa.py │ sdpa_to_sfa_adapter 补充 compress_ratio==4 的 rank>0 分支                         │ 修改 │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────┤
  │ model.py           │ PreAttention.forward() 支持传入 x_extended/freqs_extended 用于 compressor（小改） │ 修改 │
  └────────────────────┴───────────────────────────────────────────────────────────────────────────────────┴──────┘

  最大的实现难点是①（Overlap 边界交换），因为它涉及修改 PreAttention.forward() 的接口，让 Compressor
  接受扩展输入。其余的 AllGather、全局因果 mask、SFA adapter 都可以参照 C128A 的现有模式扩展实现。