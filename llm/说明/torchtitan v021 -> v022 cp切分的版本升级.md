# 一、PyTorch context_parallel CP切分策略
## 1、核心策略：Ring Attention

基本策略：
- 序列切分：将输入序列沿 seq_dim 维度切分，每个 rank 只保留一个分片；
- KV 环形传递：在 size（CP 并行度）次迭代中，KV 在各个 rank 之间旋转传递；
- 局部 SDPA + 合并：每个 rank 用本地的 q 和当前持有的 kv 计算局部的 attention，然后通过 LSE（log-sum-exp）在线合并结果。

KV 传递方式： All-Gather

## 2、参考材料

| 资料                                                                               | 内容                                       |
| -------------------------------------------------------------------------------- | ---------------------------------------- |
| Ring Attention 原论文 https://arxiv.org/abs/2310.01889                              | CP 的理论基础，介绍了如何将序列分布到多设备并使用分块 Transformer |
| Flash Attention 2 https://arxiv.org/abs/2307.08691                               | 底层 SDPA kernel，CP 依赖其返回 LSE 用于合并         |
| ring-flash-attention (zhuzilin) https://github.com/zhuzilin/ring-flash-attention | LSE 合并算法来源，代码注释中直接引用了这个 repo 的 PR #34    |
| Megatron-LM CP 实现 https://github.com/NVIDIA/Megatron-LM                          | NVIDIA 的 CP 参考实现，与 PyTorch 这个实现思路相近      |
| PyTorch CP RFC https://github.com/pytorch/pytorch/issues/109798                  | PyTorch 官方 CP 设计讨论 issue，介绍了设计决策         |


> [!NOTE] 问题：Ring Attention 只解决了 attention 计算本身的分布式问题，但在此之前，整个模型的所有"依赖序列维度"的张量都需要先被正确地切分到对应的 rank 上，而这件事 PyTorch 的 attention kernel 并不管。

# 二、torchtitan v021 cp 切分方案

针对上面的问题，torchtitan v021 cp给出了以下的解决方案：
## （1）问题一：谁来切分输入数据
DataLoader 每次吐出的 inputs 和 labels 的形状是 `[batch, full_seq_len]`，所有 rank 拿到的是完整的序列。在进入模型之前，必须把 seq 维度切分，rank i 只保留属于自己的那一段。
`context_parallel()` `context manager` 在 yield 之前做了这件事：
  
```python
  # pytorch 源码 _attention.py line 1570-1574
  shards = _context_parallel_buffers(mesh, buffers, buffer_seq_dims, load_balancer)、
  for buffer, shard in zip(buffers, shards): 
      shard = shard.clone()
      buffer.resize_(shard.shape)   # 原地修改！把 buffer 缩小成 shard 大小 
      buffer.copy_(shard)           # 把对应分片的数据拷进去
```
 
  关键细节：<mark style="background:#affad1">这是原地修改（in-place resize），而不是返回新 tensor。原因是 inputs 这个变量在进入 context 之前已经被后续代码引用了，必须原地改才能让模型拿到的也是切分后的数据</mark>。
  退出 with 块时再原地 restore 回来（对于 no_restore_buffers 里的不恢复，因为训练时不需要）：
```python
  # line 1580-1583
  for buffer, original_buffer in zip(buffers, original_buffers):
      if original_buffer is not None: 
          buffer.resize_(original_buffer.shape)
          buffer.copy_(original_buffer) 
```
## （2）问题二：freqs_cis 为什么必须也加进去？
  这是 v0.2.1 最核心的复杂之处，也是 v0.2.2 重点解决的问题。
  `freqs_cis` 是 RoPE（旋转位置编码）的预计算缓存，形状是 `[max_seq_len, head_dim//2]`，存储在模型里作为 buffer（不是参数）：
```python
  # llama3 model.py 
  self.freqs_cis = precompute_freqs_cis(...)  # shape: [max_seq_len, dim//2]
```
  在 attention 计算中，`apply_rotary_emb(xq, xk, freqs_cis)` 会用 `freqs_cis[0:seqlen]` 来取位置编码。
  
  问题在于： 如果 rank 1 拿到了 `token [S/N, 2S/N)`，但 freqs_cis 还是完整的 `[0, max_seq_len)`，那么 `freqs_cis[0:seqlen]` 取的是 position 0 到 S/N 的编码，而 rank 1 的 token 实际应该用 position S/N 到  
  2S/N 的编码——位置编码对应关系错误！所以必须把 freqs_cis 也沿 dim=0（序列维度）切分：                                                                                                                                                   
   
```python
  # v0.2.1 train.py 
  cp_buffers = [inputs, labels] 
  cp_seq_dims = [1, 1]
  if hasattr(model_parts[0], "freqs_cis"): 
      for m in model_parts:
          cp_buffers.append(m.freqs_cis)         # 把模型里的 freqs_cis 也加进来
      cp_seq_dims += [0 for _ in model_parts]    # freqs_cis 的序列维度是 dim=0
```

  context manager 会原地把 `m.freqs_cis 从 [max_seq_len, d]` 改成 `[S/N, d]`，只保留当前 rank 对应位置的编码。
  
  PP（Pipeline Parallel）带来的额外复杂性： 当 PP 开启时，模型被切成多个 stage，每个 stage 是独立的 nn.Module，每个 stage 都有自己的 freqs_cis（因为每个 stage 只处理自己负责的那些 transformer layer）。所以代码用 `for m in model_parts` 遍历所有 stage，把每个 stage 的 freqs_cis 都加进 cp_buffer。

## （3）问题三：set_rotate_method 设置的是什么？
   这控制的是 Ring Attention 中 KV 在 rank 间传递的通信原语：
  - allgather 模式（默认）
	  - `_AllGatherRotater`: 一次性 all-gather 所有 rank 的 KV，然后每轮取对应切片 
	  - 优点：只通信一次，但峰值显存更高
- alltoall 模式
	- _AllToAllRotater: 每轮用 permute_tensor（all-to-all）把 KV 传给下一个 rank
	- 优点：显存低（每轮只持有一份 KV），但通信次数多  


# 三、RoPE（Rotaty Position Embedding，旋转位置编码）相接












二、v0.2.2 的 CP 切分方案：基于 DTensor 的显式切分 + 负载均衡

  核心机制

  v0.2.2 完全重构，使用 PyTorch 新的 _ContextParallel DTensor 并行化方案，在 数据加载后显式切分，并引入两种 Load Balancer 解决负载不均衡。

  2.1 SDPA 路径：HeadTail Load Balancer

  HeadTail 切分策略（_HeadTailLoadBalancer）：

  对于 CP degree = N，不再做连续分块，而是将序列的头部和尾部交错配对分配给同一个 rank：

  序列 = [0, 1, 2, 3, 4, 5, 6, 7]，CP degree = 4

  rank 0: token [0, 7]   ← 第1个 + 最后1个
  rank 1: token [1, 6]   ← 第2个 + 倒数第2个
  rank 2: token [2, 5]
  rank 3: token [3, 4]

  对于 causal attention，越靠前的 token 计算量越少，越靠后的越多。头尾配对后，每个 rank 的计算量大致均衡（一轻一重相加）。

  2.2 FlexAttention 路径：PTRR Load Balancer

  v0.2.2 同时新增对 FlexAttention 的 CP 支持，使用 _PTRRLoadBalancer（PTRR = Prefix-aware Token Reordering and Redistribution）。

  PTRR 根据 BlockMask 的实际 attention pattern 来动态计算每个 token 的计算量，然后做负载均衡的分配，而不是依赖 causal 模式的先验假设。这使得它能正确处理各种复杂的 mask 模式（文档分块、prefix sharing
   等）。

  2.3 代码路径重构

  新的切分入口（context_parallel.py，train.py）：

  # v0.2.2 —— 在数据加载后显式切分，freqs_cis 不再需要切分
  def prepare_context_parallel_input(inputs, labels, extra_kwargs,
                                      cp_mesh, device,
                                      load_balancer_type="headtail"):
      # 生成全局 positions 索引
      positions = torch.arange(0, inputs.shape[1]).expand(inputs.shape)
      # 一起切分 inputs / labels / positions
      (inputs, labels, positions), attention_masks = cp_shard(
          cp_mesh, (inputs, labels, positions),
          attention_masks, load_balancer_type
      )
      extra_kwargs["positions"] = positions   # ← positions 传给模型做 RoPE
      return inputs, labels, extra_kwargs

  # train.py 中，无需 context manager，直接切好数据即可
  if parallel_dims.cp_enabled:
      inputs, labels, extra_kwargs = prepare_context_parallel_input(...)

  with self.train_context():   # 不再传 cp_context
      pred = model_parts[0](inputs, ...)

  模型侧的 CP 注入（context_parallel.py）：

  def apply_cp_to_attention_module(attention_modules, cp_mesh, attention_type):
      match attention_type:
          case "sdpa":
              _enable_context_parallel_dispatcher()   # 拦截 SDPA dispatch
              cp_plan = _ContextParallel(seq_dim=2, attention_type=SDPA)
          case "flex":
              cp_plan = _ContextParallel(seq_dim=2, attention_type=FLEX)
      for attn in attention_modules:
          parallelize_module(attn, cp_mesh, cp_plan)  # DTensor 并行化

  配置选项：

  # v0.2.2
  context_parallel_degree = 8
  context_parallel_load_balancer = "headtail"  # "ptrr" 或 None
  context_parallel_rotate_method = "allgather"  # 仍保留，用于底层通信

  ---
  三、核心变更对比

  ┌────────────────┬──────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
  │      维度      │                              v0.2.1                              │                               v0.2.2                               │
  ├────────────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ 切分时机       │ forward 时通过 context manager 隐式切分                          │ 数据加载后显式切分                                                 │
  ├────────────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ 序列分配策略   │ 连续均匀分块（无负载均衡）                                       │ HeadTail 交错配对（SDPA）/ PTRR 动态均衡（Flex）                   │
  ├────────────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ freqs_cis 处理 │ 需要将 RoPE 缓存 buffer 也切分传入                               │ 传入 positions 索引，模型内部用 gather 取 RoPE，无需切分 freqs_cis │
  ├────────────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ attention 支持 │ 仅 SDPA（Flash Attention）                                       │ SDPA + FlexAttention（新增）                                       │
  ├────────────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ 模型侧注入     │ 隐式（context manager 拦截）                                     │ 显式 parallelize_module + DTensor 并行化计划                       │
  ├────────────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ API 层次       │ torch.distributed.tensor.experimental.context_parallel（旧 API） │ _ContextParallel + _context_parallel_shard（新 API）               │
  ├────────────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ 负载均衡配置   │ 无（无此选项）                                                   │ context_parallel_load_balancer = "headtail"/"ptrr"/None            │
  ├────────────────┼──────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ PP 兼容性      │ 需要特殊处理 freqs_cis buffer（每个 PP stage 单独处理）          │ 无需特殊处理，通过 positions 统一解决                              │
  └────────────────┴──────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘

  ---
  四、修改带来的好处

  1. 解决 Causal Attention 的负载不均衡

  这是最核心的改进。原有均匀切分下：

  rank 0 计算量 ∝ S/N × 1       (只 attend to 自身)
  rank N-1 计算量 ∝ S/N × S     (attend to 全序列)

  HeadTail 配对后每个 rank 的计算量近似相等，GPU 利用率和训练吞吐显著提升（尤其是大 CP degree 时效果更明显）。

  2. 解锁 FlexAttention + CP

  通过 PTRR 负载均衡器，v0.2.2 第一次支持了 FlexAttention 与 CP 的组合，这对于需要自定义 attention mask（文档注意力、因果分块、prefix caching 等场景）非常重要。

  3. 消除 freqs_cis 切分的复杂性

  v0.2.1 中必须将模型内部的 RoPE 缓存 freqs_cis 作为 buffer 传入切分，在 PP（Pipeline Parallel）下每个 stage 都要单独处理，逻辑复杂且易出错。v0.2.2 改为切分 positions 索引，模型内部通过
  freqs_cis[positions] 的方式索引取值，完全解耦了 RoPE 与 CP 的交互。

  4. API 更清晰、可组合性更好

  从 context manager 式的隐式黑盒，变成显式的 parallelize_module + DTensor 并行计划，与 TP/FSDP 的组合方式完全一致，更容易理解和扩展。

  ---
  五、参考文献

  HeadTail Load Balancer 的思想来源于：

  ▎ "Ring Attention with Blockwise Transformers for Near-Infinite Context"
  ▎ Liu et al., 2023
  ▎ https://arxiv.org/abs/2310.01889

  以及序列并行负载均衡的更早工作：

  ▎ "Reducing Activation Recomputation in Large Transformer Models" (Megatron-LM Sequence Parallelism)
  ▎ Korthikanti et al., 2022
  ▎ https://arxiv.org/abs/2205.05198

  PTRR（PTRRLoadBalancer） 对应的算法：

  ▎ "DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training"
  ▎ Li et al., 2023
  ▎ https://arxiv.org/abs/2310.03294

  以及 FlexAttention 本身：

  ▎ PyTorch FlexAttention Blog
  ▎ https://pytorch.org/blog/flexattention/

  对应的 PyTorch PR（代码实现层）：

  - _HeadTailLoadBalancer 实现：pytorch/pytorch#170200
  - _PTRRLoadBalancer 实现：pytorch/pytorch#170201