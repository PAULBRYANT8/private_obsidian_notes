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


# 三、RoPE（Rotaty Position Embedding，旋转位置编码）详解

## 1、为什么需要位置编码？
  Transformer 的 self-attention 本质上是一个集合操作（set operation）：Attention(Q, K, V) = softmax(QK^T / √d) · V
  对于序列 `[token_0, token_1, token_2, ...]`，如果把所有 token 打乱顺序，attention 的输出完全不变（因为 QK^T 的每个元素都只是两个向量的点积，与位置无关）。
  但语言显然是有顺序的，"狗咬人" 和 "人咬狗" 意思完全不同。所以必须想办法把位置信息注入模型。
  
## 2、早期方案的问题
- 绝对位置编码（BERT/GPT-1/2 做法）
  在 token embedding 上直接加一个位置 embedding：`input = token_embedding(x) + position_embedding(pos)`
  问题：
    - 最大序列长度固定（`position_embedding` 是查找表，训练时最多 512 个位置）
    - 两个 token 的距离信息被编码在绝对位置里，不直观（位置 3 和位置 5 的"相对距离 2"需要模型自己学）

- 相对位置编码（T5、Shaw et al.）
  在计算 attention score 时，加入相对距离的偏置：`score(q_i, k_j) = q_i · k_j + b(i-j)`
  问题：
    - 需要修改 attention 内部结构
    - 引入大量额外参数 b(·)
    - 难以外推到训练时没见过的序列长度

<mark style="background:#ff4d4f">上面两种方法都不能很好的解决在 llm 中 token 之间的位置问题</mark>。

# 3、RoPE 的核心思想
  RoPE 的目标是找到一种位置编码函数 `f(x, pos)`，满足：<mark style="background:#d3f8b6">两个 token 的内积，只依赖它们各自的内容和相对距离，与绝对位置无关</mark>。
  数学上要求：
  $$\langle f(q, m),\ f(k, n) \rangle = g(q, k, m-n)$$
  即 q 在位置 m、k 在位置 n 的内积，只是 m-n（相对距离）的函数。
  RoPE 的解决方案：用旋转矩阵编码位置。

在标准的 Transformer 模型中，注意力机制通过计算点积 $\langle q, k \rangle$ 来决定一个词（**Query**，查询向量 $q$）应该对另一个词（**Key**，键向量 $k$）分配多少注意力。

然而，这种基础的数学运算本身完全没有包含这些词在句子中“位置”的概念。你提供的等式，实际上是一个完美位置编码系统的理论“愿望清单”：

- $f(q, m)$：这是一个函数，它接收查询向量 $q$，并将其在序列中的**绝对位置** $m$ 注入其中。
    
- $f(k, n)$：同样的函数，将**绝对位置** $n$ 注入到键向量 $k$ 中。
    
- $\langle \dots , \dots \rangle$：用于计算这两个词之间注意力分数的点积（内积）操作。
    
- $g(q, k, m-n)$：这是一个仅依赖于原始向量本身，以及它们之间**相对距离** ($m-n$) 的函数。
    

**通俗来说就是：** “我们希望找到一种方法，把词汇的_绝对位置_编码进去；但当模型计算它们之间的注意力分数时，数学运算的结果却能自然而然地只与它们的_相对距离_有关。”

### 2. RoPE 是如何解决的（“旋转”的奥秘）

RoFormer 论文的作者们发现，如果把向量看作复数，并对它们进行**旋转**，就能完美实现这个等式的要求。

RoPE 不会像早期模型那样直接把位置数值加到向量上，而是将向量在二维空间中旋转一个与其位置成正比的角度 $\theta$：

$$f(q, m) = q \cdot e^{im\theta}$$

$$f(k, n) = k \cdot e^{in\theta}$$

当你计算这两个旋转后向量的内积时，指数上的绝对位置 $m$ 和 $n$ 会因为复数共轭的乘法规律而相互相减，最后只留下相对距离：

$$\langle f(q, m), f(k, n) \rangle = \text{Re}(q k^* e^{i(m-n)\theta})$$

为了直观地理解这一点，你可以尝试下方的交互式演示。请注意：当你改变绝对位置时，向量本身会发生旋转；**但是**，只要两个位置之间的_距离_（即 $m-n$ 的差值）保持不变，这两个向量之间的夹角（即它们算出来的注意力分数）就会完全保持锁定。
  ---
  四、二维情形：直觉理解

  先看最简单的二维向量情形。

  一个二维向量 q = [q_0, q_1]，在位置 m 处，将它旋转 m·θ 角度：

  $$f(q, m) = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} q_0 \ q_1 \end{bmatrix}$$

  两个旋转后向量的内积：

  $$f(q, m)^T \cdot f(k, n) = q^T R_{-m\theta} R_{n\theta} k = q^T R_{(n-m)\theta} k$$

  因为旋转矩阵满足 $R_\alpha^T R_\beta = R_{\beta-\alpha}$，内积结果只依赖 n-m（相对距离）。✓

  ---
  五、高维情形：实际实现

  对于 d 维向量（d 必须是偶数），RoPE 把向量两两配对成 d/2 个二维子空间，每个子空间用不同频率的旋转：

  $$f(q, m) = \begin{bmatrix} q_0 \ q_1 \ q_2 \ q_3 \ \vdots \ q_{d-2} \ q_{d-1} \end{bmatrix} \otimes \begin{bmatrix} \cos(m\theta_0) \ \cos(m\theta_0) \ \cos(m\theta_1) \ \cos(m\theta_1) \ \vdots
  \ \cos(m\theta_{d/2-1}) \ \cos(m\theta_{d/2-1}) \end{bmatrix} + \begin{bmatrix} -q_1 \ q_0 \ -q_3 \ q_2 \ \vdots \ -q_{d-1} \ q_{d-2} \end{bmatrix} \otimes \begin{bmatrix} \sin(m\theta_0) \
  \sin(m\theta_0) \ \sin(m\theta_1) \ \sin(m\theta_1) \ \vdots \ \sin(m\theta_{d/2-1}) \ \sin(m\theta_{d/2-1}) \end{bmatrix}$$

  其中频率按照以下公式设定（继承自 Transformer 原始论文的 sinusoidal encoding）：

  $$\theta_i = \frac{1}{10000^{2i/d}}, \quad i = 0, 1, \ldots, \frac{d}{2}-1$$

  - i=0：旋转最快，每个位置转约 1 弧度，编码近距离关系
  - i=d/2-1：旋转极慢（1/10000 弧度/位置），编码远距离关系

  可以想象成时钟的指针：秒针（高频）转得快，时针（低频）转得慢，组合起来唯一标识时间。

  ---
  六、复数表示（实际代码使用的形式）

  用复数可以更紧凑地表达旋转。把相邻两维配对成复数：

  $$q_{\text{complex}} = q_0 + i \cdot q_1, \quad q_2 + i \cdot q_3, \ldots$$

  旋转位置 m 就是乘以单位复数 $e^{im\theta_k}$：

  $$f(q_{\text{complex}}, m)k = (q{2k} + i \cdot q_{2k+1}) \cdot e^{im\theta_k}$$

  torchtitan 中 freqs_cis 就是预计算好的这些复数旋转因子：

  # torchtitan/models/llama3/model/model.py
  def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
      # 计算频率：θ_i = 1 / 10000^(2i/d)
      freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
      # 计算位置 × 频率
      t = torch.arange(end)          # [0, 1, 2, ..., max_seq_len-1]
      freqs = torch.outer(t, freqs)  # shape: [max_seq_len, dim//2]
      # 转换为复数 e^(i*m*θ)
      freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 模为1，辐角为 m*θ
      return freqs_cis               # shape: [max_seq_len, dim//2], dtype: complex64

  应用 RoPE 时：

  def apply_rotary_emb(xq, xk, freqs_cis):
      # 把 query/key 的最后两维视为复数
      xq_complex = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
      xk_complex = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
      # 乘以旋转因子（广播到 batch 和 head 维度）
      freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim//2]
      xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
      xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
      return xq_out, xk_out

  ---
  七、RoPE 的优势

  ┌─────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐
  │          特性           │                                         说明                                         │
  ├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 相对位置天然涌现        │ 内积 $\langle f(q,m), f(k,n) \rangle$ 只依赖 $m-n$，无需额外设计                     │
  ├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 无额外参数              │ freqs_cis 是固定的，不需要训练，不增加模型参数量                                     │
  ├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 长度外推性              │ 可以在测试时直接用比训练时更大的位置索引（有一定限制，后续被 YaRN 等方法进一步改进） │
  ├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 与 Flash Attention 兼容 │ 在 QK 矩阵乘法之前就完成了位置编码，不改变 attention 内部计算                        │
  ├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 计算效率高              │ 只是逐元素乘法，几乎没有额外计算开销                                                 │
  └─────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘

  ---
  八、与 CP 切分的关系（回到前面的问题）

  现在可以理解为什么 v0.2.1 需要切分 freqs_cis 了：

  完整序列：[token_0, token_1, ..., token_S-1]
  对应位置：[pos_0,   pos_1,   ..., pos_S-1  ]

  freqs_cis[pos_i] = e^(i·pos_i·θ)  ← 位置 pos_i 的旋转因子

  CP 切分后，rank 1 持有 token[S/N : 2S/N]，这些 token 对应的是位置 S/N 到 2S/N-1。如果 freqs_cis 不切分，rank 1 拿到的是 freqs_cis[0:S/N]，即位置 0 到 S/N-1 的旋转因子，对应关系完全错误：

  错误情况（v0.2.1 不切分 freqs_cis 时）：
  rank 1 的 token:  [token_512, token_513, ..., token_1023]  ← 真实位置 512~1023
  rank 1 的 RoPE:   freqs_cis[0:512]                         ← 位置 0~511 的旋转因子
  结果：token_512 被编码成了 position 0，位置编码完全错误！

  正确情况（切分后）：
  rank 1 的 RoPE:   freqs_cis[512:1024]                      ← 位置 512~1023 的旋转因子

  而 v0.2.2 的解决方案更优雅：切分 positions 索引而不是 freqs_cis 本身，让模型内部用 freqs_cis[positions] 按需索引，freqs_cis 始终保持完整，不需要特殊处理。

  ---
  九、参考论文

  ▎ RoFormer: Enhanced Transformer with Rotary Position Embedding
  ▎ Su et al., 2021
  ▎ https://arxiv.org/abs/2104.09864

  这篇是 RoPE 的原始论文，作者苏剑林（苏神）。文章推导严谨，从"什么是好的相对位置编码"出发，一步步推导出旋转的形式，值得精读。














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