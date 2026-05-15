记录时间：2026-05-13

本文整理 `deepseek_v4_285b_43layers_4k_128die.toml` 在开启 Pipeline Parallel(PP) 和 MTP 后遇到的几类报错，包括日志现象、根因分析、当前修改是否有效，以及最终建议的完整修复方案。

---
## 1. 概念问题汇总

### 1.1 MTP 是什么

MTP 是 Multi-Token Prediction，多 token 预测。

普通语言模型每个位置预测下一个 token：

```text
输入: [A, B, C, D]
预测: [B, C, D, E]
```

MTP 会额外训练模型预测更远的 token。例如 `num_mtp_modules = 1` 时，除了主 loss，还会增加一个 offset 为 1 的 MTP loss：

```text
main branch:
  position A -> predict B
  position B -> predict C
  position C -> predict D
  position D -> predict E

mtp branch:
  使用主干 hidden + offset token embedding
  额外预测下一偏移目标
```

在当前代码里：

```python
output_list[0] = main_logits
output_list[1] = mtp_logits
```

loss 里：

```python
total_loss = main_loss + mtp_loss * mtp_loss_weight
```

### 1.2 stage 指什么

PP 里的 stage 是 pipeline 切分出来的模型片段。

例如完整模型是：

```text
embedding -> layer0 -> layer1 -> ... -> layer42 -> norm -> output
```

PP 会把它切成多个 stage：

```text
stage0: embedding + layers.0-10
stage1: layers.11-21
stage2: layers.22-32
stage3: layers.33-42 + norm + output
```

每个 stage 只持有一部分模型参数，只执行自己负责的 forward。

### 1.3 为什么 pipeline_parallel_degree = 2 会有 4 个 stage

`pipeline_parallel_degree = 2` 表示物理 PP rank 数是 2。

但当前 schedule 是：

```toml
pipeline_parallel_schedule = "Interleaved1F1B"
```

这是 multi-stage schedule。当前通用 PP 逻辑在没有显式设置 `pipeline_parallel_layers_per_stage` 时，默认：

```text
每个 PP rank 放 2 个 virtual stage
```

所以：

```text
物理 PP rank 数 = 2
每个 rank 的 virtual stage 数 = 2
总 stage 数 = 2 * 2 = 4
```

stage 分配方式是 loop 风格：

```text
PP rank 0: stage 0, stage 2
PP rank 1: stage 1, stage 3
```

因此日志里会看到：

```text
stage_idx0
stage_idx1
stage_idx2
stage_idx3
```

### 1.4 tok_embeddings 是什么

`tok_embeddings` 是 token embedding 层：

```python
self.tok_embeddings = nn.Embedding(vocab_size, dim)
```

它负责把 token id 转成向量：

```text
token ids: [B, S]
embedding: [B, S, dim]
```

在 PP 中，`tok_embeddings` 只应该位于 first stage。后面的 stage 收到的是 activation，不再是 token id。

### 1.5 hc_head 是什么

`hc_head` 是 DeepSeekV4 的 HC 输出头，用来把 HC 形态的 hidden state 压回普通 hidden state。

DeepSeekV4 主干层输出形态是：

```text
[B, S, hc_mult, dim]
```

其中：

```text
hc_mult = 4
```

在接 `norm` 和 `output` 前，需要经过：

```python
h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
```

变成：

```text
[B, S, dim]
```

`hc_head` 不是 MTP 引入的。没有 MTP 时 DeepSeekV4 也需要 `hc_head`。只是当前通用 PP 切分没有显式包含它，所以在 PP+MTP 问题里一起暴露出来。

### 1.6 sidecar 是什么

这里说的 sidecar 不是框架固定术语，而是“跟着主数据一起跨 stage 传递的附加信息”。

PP 主数据是：

```text
h  # 当前 stage 要继续加工的 hidden states
```

但 MTP 和 MoE routing 还需要额外信息：

```text
input_ids          # 原始 token id，给 MoE routing 用
mtp_input_offsets  # MTP offset token embedding，给最后 stage 的 MTPModule 用
```

所以 stage 间传递 payload：

```python
(h, input_ids, mtp_input_offsets)
```

这里 `h` 会被每个 stage 更新，但 `input_ids` 和 `mtp_input_offsets` 是从原始 token 派生出来的，不应该被中间 stage 改写。

### 1.7 为什么中间 stage 只更新 h，原样转发 sidecar

中间 stage 的职责是继续跑主干 TransformerBlock：

```python
h = layer(h, input_ids, ...)
```

它需要读取 `input_ids`，因为 MoE routing 可能依赖 token id。但它不应该修改 `input_ids`。

`mtp_input_offsets` 只在最后 stage 的 MTPModule 中使用，中间 stage 不需要处理它。

因此中间 stage 的逻辑应该是：

```python
h, input_ids, mtp_input_offsets = payload

for layer in local_main_layers:
    h = layer(h, input_ids, ...)

return (h, input_ids, mtp_input_offsets)
```

### 1.8 为什么不是每个 stage 都执行 seq_len -= num_mtp_modules

`seq_len -= num_mtp_modules` 的含义是：

```text
从原始 token 序列中，切出主干输入长度，同时为 MTP 保留 offset token。
```

例如：

```text
tokens = [A, B, C, D, E]
num_mtp_modules = 1
```

first stage 应该切成：

```text
main_input_ids    = [A, B, C, D]
mtp_offset_tokens = [B, C, D, E]
```

所以：

```text
seq_len = 5 - 1 = 4
```

但是 stage1、stage2、stage3 收到的已经不是原始 token ids，而是裁剪后的 activation：

```text
h = hidden([A, B, C, D])
```

如果每个 stage 都再裁一次，就会变成：

```text
stage0: 5 -> 4
stage1: 4 -> 3
stage2: 3 -> 2
stage3: 2 -> 1
```

这显然会错误地缩短序列。

因此：

```text
只有 first stage 对原始 token ids 做 seq_len -= num_mtp_modules；
后续 stage 使用 h.shape[1]，不能再减。
```

### 1.9 MTP 没有参与训练时，为什么会说最后 stage 输出和 MTP 不符合

当前确实是 MTP 没有真正参与训练，因为最后 stage 没有 `layers.43`。

之前说“最后 stage 的返回值与 MTP 不符合”，意思不是“最后 stage 的输出已经作为 MTP 输入并失败”，而是：

```text
配置和 loss 认为 MTP 开启；
但模型实际只返回普通 Tensor；
loss 仍然按 MTP 输出 list 去解释这个 Tensor。
```

于是：

```python
preds[0]
```

被错误解释成第一个 logits，而实际是 Tensor 的第 0 个 batch 切片。

正确的 PP+MTP 中，最后 stage 应该先产出主干 hidden：

```text
prev_embed = main hidden after hc_head
```

再把它和 `mtp_input_offsets` 一起送入 MTPModule：

```python
mtp_h = mtp_layer(mtp_input_offsets, prev_embed, input_ids, ...)
```

最后返回：

```python
[main_logits, mtp_logits]
```


## 八、TransformerBlock 内部模块结构详解

### 8.1 `attention_norm` 与 `attention` 的关系

这是 **Pre-LN（前置归一化）** 架构的体现。看 `TransformerBlock.forward()`：

```python
x = x + self.attention(self.attention_norm(x), freqs_cis, attention_masks, positions)
```

执行顺序是：

```
原始输入 x
    ↓
attention_norm(x)          ← RMSNorm，先对 x 做归一化
    ↓
attention(归一化后的 x)     ← 再做注意力计算
    ↓
x + attention_output        ← 残差连接，加回的是原始 x（未归一化）
```

`attention_norm` 本身不是注意力的一部分，它是注意力的**前置门卫**：把特征先规范化到合理范围，再送入注意力。`attention` 只负责注意力计算本身，接收的输入已经是归一化后的了。

**与原始 Transformer 的区别**：

| 架构 | 归一化位置 | 写法 |
|---|---|---|
| Post-LN（原始论文） | 残差加法之后 | `x = norm(x + attention(x))` |
| Pre-LN（现代 LLM 主流） | 送入模块之前 | `x = x + attention(norm(x))` |

Pre-LN 训练更稳定，是 DeepSeek V3 及现代大模型的标准做法。

---

### 8.2 `wq_a` / `q_norm` / `wq_b` 的关系（Q 的低秩压缩路径）

这三个模块只有在 `q_lora_rank > 0` 时才存在（默认 `q_lora_rank=0`，走 `wq` 单矩阵路径）：

```python
if self.q_lora_rank == 0:
    self.wq = nn.Linear(2048, 16 * 192)        # 单矩阵直接投影
else:
    self.wq_a = nn.Linear(2048, q_lora_rank)   # 压缩（下投影）
    self.q_norm = nn.RMSNorm(q_lora_rank)      # 归一化低秩表示
    self.wq_b = nn.Linear(q_lora_rank, 16 * 192)  # 展开（上投影）
```

**当 `q_lora_rank > 0` 时，三者构成一条 LoRA 式的 Q 投影链**：

```
输入 x  (B, S, 2048)
    ↓  wq_a: Linear(2048 → q_lora_rank)
低秩表示 c_Q  (B, S, q_lora_rank)     ← 压缩，维度远小于 2048
    ↓  q_norm: RMSNorm(q_lora_rank)
归一化的 c_Q  (B, S, q_lora_rank)     ← 先归一化再展开，防止数值不稳定
    ↓  wq_b: Linear(q_lora_rank → 16*192)
完整 Q  (B, S, 16*192)                ← 展开到全头维度
```

三者关系：`wq_a` 是下投影（压缩），`q_norm` 是中间的稳定器，`wq_b` 是上投影（展开）。不直接用一个大矩阵做投影，而是先压到低维空间再恢复，节省参数量，`q_norm` 解决压缩后数值分布问题。

---

### 8.3 `wkv_a` 是什么，与 `kv_norm` 的关系（MLA 核心）

#### MLA 核心思想

传统 MHA 的 KV 缓存开销：

- K = Linear(dim → n_heads × head_dim)，V 同理
- 推理时 KV Cache = `2 × n_heads × head_dim` per token

DeepSeek V3 使用 MLA（Multi-Head Latent Attention）：**先把 K 和 V 共同压缩成一个低维"潜向量"，推理时只缓存这个潜向量，需要时再展开**，KV Cache 大幅压缩。

#### `wkv_a`：压缩入口，同时分出位置编码分支

```python
self.wkv_a = nn.Linear(2048, 512 + 64)   # kv_lora_rank=512, qk_rope_head_dim=64
```

```
输入 x  (B, S, 2048)
    ↓  wkv_a: Linear(2048 → 576)
输出 (B, S, 576)
    ↓  split([512, 64])
kv    (B, S, 512)   ← K 和 V 的共享潜向量 c_KV
k_pe  (B, S, 64)    ← K 的位置编码分量（RoPE 施加于此）
```

**为什么要分成 512 和 64 两部分？**

K 被设计成两个部分的拼接：
- `k_nope`（No Position Embedding）：从潜向量 `c_KV` 展开，**不含位置信息**
- `k_pe`（Position Embedding）：直接从 `wkv_a` 输出中取，**施加 RoPE 位置编码**

分开的好处是推理时 KV Cache 只需存 `c_KV`（512 维）+ `k_pe`（64 维）= 576 维，而传统 MHA 需缓存 `2 × 16 × (128+128) = 8192` 维，**压缩比约 14 倍**。

#### `kv_norm`：潜向量压缩后的稳定器

```python
self.kv_norm = nn.RMSNorm(512)   # 对 c_KV 潜向量做归一化
```

```
kv (B, S, 512)
    ↓  kv_norm
归一化 kv  (B, S, 512)   ← 数值稳定后再送入 wkv_b 展开
```

`kv_norm` 与 `wkv_a` 的关系，类比于 Q 路径里 `q_norm` 与 `wq_a` 的关系：**`wkv_a` 做了维度压缩，压缩后的数值分布可能不稳定，`kv_norm` 在展开之前先归一化，防止训练发散**。

---

### 8.4 `wkv_b` 是怎么得到的

```python
self.wkv_b = nn.Linear(512, 16 * (128 + 128))   # 512 → 4096
```

```
归一化的 kv  (B, S, 512)
    ↓  wkv_b: Linear(512 → 4096)
联合 KV  (B, S, 4096)
    ↓  view(B, S, 16, 256) + split([128, 128], dim=-1)
k_nope  (B, S, 16, 128)   ← 各头的无位置 K 分量
v       (B, S, 16, 128)   ← 各头的 V 值
```

然后 K 的两个部分合并：

```
k_pe    (B, S, 1, 64)  → expand → (B, S, 16, 64)   ← 所有头共享同一 k_pe
k_nope  (B, S, 16, 128)
    ↓  cat(dim=-1)
k       (B, S, 16, 192)   ← 完整 K = [k_nope ‖ k_pe]
```

`wkv_b` 的本质：**一个 512 维的潜向量 `c_KV` 同时解码出所有 16 个头的 K_nope 和 V**，这就是 "Latent" 的含义。所有头共享同一个压缩表示，参数量大幅减少。

---

### 8.5 `inner_attention` 与 `attention` 的关系

两者是**包含关系**，`inner_attention` 是 `attention` 内部的一个子模块：

```
Attention（外层，整个注意力模块）
├── 投影层：wq_a/q_norm/wq_b 或 wq        ← 产生 Q
│           wkv_a, kv_norm, wkv_b          ← 产生 K、V
│           wo                              ← 输出投影
└── inner_attention（内层，纯计算核）       ← 只做 QKV → 输出的数学运算
    ├── ScaledDotProductAttentionWrapper    （sdpa 模式，默认）
    └── FlexAttentionWrapper                （flex 模式）
```

**`attention` 负责"前处理 + 后处理"**：
1. 把输入 x 投影成 Q、K、V（经过所有 wq/wkv 矩阵）
2. 对 Q 和 K 的 RoPE 分量施加旋转位置编码
3. 调用 `inner_attention` 做核心注意力计算
4. 把 `inner_attention` 的输出通过 `wo` 投影回 `dim` 维度

**`inner_attention` 只负责"纯数学注意力计算"**，接收已经投影好、形状整理好的张量：

```python
output = inner_attention(
    q,   # (B, 16, S, 192)  已含 RoPE
    k,   # (B, 16, S, 192)
    v,   # (B, 16, S, 128)
    scale=softmax_scale
)
# 内部：output = softmax(Q @ K^T / sqrt(d)) @ V
```

这样分层的好处：`inner_attention` 可以被替换成不同的后端实现（FlashAttention、SDPA、FlexAttention），外层投影逻辑完全不变。在 TP 并行化时，也可以对 `inner_attention` 单独指定 `Shard` 输入布局（见 `parallelize_deepseekv3.py` 中的 `attention_kernel_plan`）。

---

### 8.6 `ffn_norm` 以及所有 `*_norm` 命名规律总结

`ffn_norm` 与 `attention_norm` 完全对称，是 FFN/MoE 的**前置归一化**：

```python
x = x + self.feed_forward(self.ffn_norm(x))   # Dense 层
x = x + self.moe(self.ffn_norm(x))            # MoE 层
```

整个模型中所有带 `_norm` 后缀的模块，均遵循同一个模式：**先归一化，再送入对应的主模块，原始张量保留用于残差连接**。

```
TransformerBlock 层面（Pre-LN）：
  attention_norm  →  attention           （归一化后做注意力）
  ffn_norm        →  feed_forward / moe  （归一化后做 FFN）

Attention 内部层面（MLA 低秩路径的数值稳定）：
  q_norm   归一化 Q 低秩表示   →  wq_b 展开   （稳定 LoRA Q 路径）
  kv_norm  归一化 KV 潜向量    →  wkv_b 展开  （稳定 MLA KV 压缩路径）
```

用统一的数据流图表示：

```
TransformerBlock 内部：

  x ──→ [attention_norm] ──→ [attention] ──┐
  x ─────────────────────────────────────── (+) ──→ x'
                                                      │
  x'──→ [ffn_norm] ──→ [feed_forward/moe] ──┐       │
  x'────────────────────────────────────────(+) ──→ 输出


Attention 内部 KV 路径：

  x ──→ [wkv_a] ──┬──→ [kv_norm] ──→ [wkv_b] ──→ k_nope, v
                   └──→ apply_RoPE ──────────────→ k_pe

Attention 内部 Q 路径（q_lora_rank > 0 时）：

  x ──→ [wq_a] ──→ [q_norm] ──→ [wq_b] ──→ q
```

**规律总结**：`xxx_norm` 与 `xxx` 总是成对出现，`_norm` 在数据流中紧靠在对应模块的入口前面。目的统一：**稳定该模块接收到的特征分布，防止训练不稳定或梯度爆炸/消失**。区别仅在于作用层次：TransformerBlock 层面的 norm 服务于残差架构（Pre-LN），Attention 内部的 norm 服务于低秩压缩路径的数值稳定性。

---

## 九、MoE（Mixture of Experts）结构详解

### 9.1 为什么引入 MoE：从 Dense FFN 到稀疏 MoE

`layers.0` 用的是 Dense FFN（每个 token 经过同一个 FFN），`layers.1` 开始换成了 MoE。核心动机：

| | Dense FFN | MoE |
|---|---|---|
| 参数量 | 固定，所有 token 共享 | 参数量可以很大 |
| 每 token 激活参数量 | 全部参数 | 只激活 top-k 个专家 |
| 计算量（FLOPs） | 与参数量成正比 | 远小于总参数量 |

**核心思想**：把一个大的 FFN 拆成 N 个小的"专家 FFN"，每个 token 只路由到其中的 k 个，用少量计算激活大量参数，实现"参数量大但计算量不增加"的效果。

---

### 9.2 MoE 完整结构（以默认 MoEArgs 为例）

**默认超参数**（`MoEArgs` 默认值）：

| 参数 | 值 | 含义 |
|---|---|---|
| num_experts | 8 | 路由专家数 |
| num_shared_experts | 1 | 共享专家数（所有 token 必经） |
| top_k | 1 | 每个 token 路由到几个专家 |
| score_func | "sigmoid" | 路由打分函数 |
| route_scale | 1.0 | 路由得分缩放系数 |
| gate_bias | False | 路由门控是否有偏置 |
| score_before_experts | True | 先加权再送专家（vs 先送专家再加权） |
| num_expert_groups | None | 专家分组数（节点限制路由用） |
| load_balance_coeff | 1e-3 | 负载均衡系数 |
| use_grouped_mm | True | 使用 grouped_mm 还是 for-loop |

结合 `moe_inter_dim=1408`，完整 MoE 结构如下：

```
MoE
├── router        : TokenChoiceTopKRouter
│   └── gate      : Linear(2048 → 8, bias=False)      ← 路由打分矩阵
│
├── experts       : GroupedExperts                      ← 8 个路由专家，参数打包存储
│   ├── w1        : Parameter(8, 1408, 2048)            ← 8 个专家的 gate 投影权重
│   ├── w2        : Parameter(8, 2048, 1408)            ← 8 个专家的输出投影权重
│   └── w3        : Parameter(8, 1408, 2048)            ← 8 个专家的 value 投影权重
│
├── reorderer     : TokenReorderer                      ← token 按专家排序
│
├── shared_experts: FeedForward(2048, 1408)             ← 1 个共享专家（所有 token 必过）
│   ├── w1        : Linear(2048 → 1408)
│   ├── w2        : Linear(1408 → 2048)
│   └── w3        : Linear(2048 → 1408)
│
├── expert_bias   : buffer(8,)  float32                ← 负载均衡偏置（运行时更新）
└── tokens_per_expert: buffer(8,) float32              ← 各专家累计 token 数（统计用）
```

---

### 9.3 MoE Forward：逐步拆解

输入：`x`，shape `(B, S, 2048)`，即一个 batch 内所有 token 的隐状态。

#### Step 1：展平 token 维度

```python
x = x.view(-1, 2048)   # (B*S, 2048)，后文记 N = B*S
```

所有 token 一起参与路由，路由是 per-token 独立的。

---

#### Step 2：路由打分（TokenChoiceTopKRouter.forward）

```python
scores = gate(x)        # Linear(2048→8)，shape (N, 8)
scores = sigmoid(scores.float())   # 转 float32 防止精度溢出，shape (N, 8)
```

**为什么用 sigmoid 而非 softmax？**
- softmax 是归一化竞争：所有专家的得分加和为 1，一个专家高了另一个必然低
- sigmoid 是独立打分：每个专家得分与其他无关，语义是"这个专家对此 token 有多合适"
- DeepSeek V3 选用 sigmoid，路由更灵活，专家之间不强制竞争

```python
scores_for_choice = scores + expert_bias   # 加入负载均衡偏置（初始为全 0）
selected_experts_indices = topk(scores_for_choice, k=1)  # shape (N, 1)
top_scores = scores.gather(selected_experts_indices)      # shape (N, 1)，用原始分，非偏置后的分
num_tokens_per_expert = histc(selected_experts_indices, bins=8)  # shape (8,)，各专家分到的 token 数
```

**节点限制路由（num_expert_groups 不为 None 时）**：
把 8 个专家按节点分成若干组，先选出得分最高的若干组，再从这些组内选 top-k 专家。
好处：减少跨节点 all-to-all 通信量，在大规模 EP 中极其重要。

---

#### Step 3：统计负载，用于后续均衡更新

```python
with torch.no_grad():
    self.tokens_per_expert += num_tokens_per_expert
```

此处只是无梯度的累加统计，不影响前向计算。

---

#### Step 4：token 按专家重新排序（TokenReorderer.forward）

路由结果是"每个 token 去哪个专家"，但 `GroupedExperts` 计算时需要"每个专家连续拿到属于自己的所有 token"，因此需要重排：

```
原始 token 顺序：[t0→E2, t1→E0, t2→E2, t3→E1, t4→E0]

排序后（按专家编号升序）：
  E0 的 token：t1, t4
  E1 的 token：t3
  E2 的 token：t0, t2

token_indices_experts_sorted = argsort([2,0,2,1,0]) = [1,4,3,0,2]
num_tokens_per_expert = [2, 1, 2, 0, 0, 0, 0, 0]
```

```python
token_indices_experts_sorted = argsort(selected_experts_indices.view(-1), stable=True)
# shape (N*top_k,)，记录排序后第 i 位对应原始第几个 token 的第几次路由
top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
```

---

#### Step 5：按专家顺序取出 token，可选先加权

```python
routed_input = x[token_indices_experts_sorted // top_k]
# shape (N*top_k, 2048)，按专家排好序的 token 特征

if score_before_experts:   # 默认 True
    routed_input = routed_input * top_scores_experts_sorted.reshape(-1, 1)
    # 在送入专家之前先乘以路由权重，等价于"带权重的专家输入"
```

**score_before_experts=True 的含义**：先用路由得分缩放输入，再送入专家。
等价于 `output = expert(score * x)`，而非 `output = score * expert(x)`。
两者在数学上有细微差异，前者让路由得分影响专家内部的激活值。

---

#### Step 6：GroupedExperts 计算（8 个专家并行）

每个专家是一个 **SwiGLU FFN**，结构与 Dense FFN 相同：

```
专家 i 的计算（SwiGLU）：
  gate  = w1[i] @ x_i        # shape (tokens_i, 1408)
  value = w3[i] @ x_i        # shape (tokens_i, 1408)
  h     = SiLU(gate) * value  # 逐元素门控，SwiGLU 核心
  out_i = w2[i] @ h           # shape (tokens_i, 2048)
```

**用 grouped_mm 实现并行**（`use_grouped_mm=True`，默认）：

不用 for-loop 逐专家计算，而是用 `torch._grouped_mm`：
```python
offsets = cumsum(num_tokens_per_expert)   # 各专家的 token 段偏移
h = SiLU(grouped_mm(x, w1.T, offs=offsets))   # 所有专家一次性批量矩阵乘
h = h * grouped_mm(x, w3.T, offs=offsets)
out = grouped_mm(h, w2.T, offs=offsets)
```

`grouped_mm` 知道每一段属于哪个专家，对不同段使用对应的权重矩阵，一个 kernel 完成所有专家的计算，GPU 利用率远高于 for-loop。

```python
routed_output = self.experts(routed_input, num_tokens_per_expert)
# shape (N*top_k, 2048)，按专家排序的输出
```

---

#### Step 7：共享专家（shared_experts）计算

```python
out = self.shared_experts(x)   # 所有 token 都过，shape (N, 2048)
```

共享专家是一个普通 FeedForward，**每个 token 无论路由到哪个专家，都必须经过共享专家**。
其作用：
- 保证信息的全局性传播（不依赖路由结果）
- 类似于"公共知识"的存储库，而路由专家存储"领域知识"
- 有助于训练稳定性（不依赖动态路由的梯度）

---

#### Step 8：将路由专家输出还原回 token 原始顺序

```python
routed_output_unsorted = zeros(N*top_k, 2048)
routed_output_unsorted[token_indices_experts_sorted] = routed_output
# 利用排序索引的逆映射，将结果放回原始位置
routed_output_unsorted = routed_output_unsorted.reshape(N, top_k, 2048)
```

---

#### Step 9：合并路由专家输出 + 共享专家输出

```python
# score_before_experts=True 时（默认）：输入已经加权过了，直接求和
out_experts = routed_output_unsorted.sum(dim=1)   # (N, 2048)

# score_before_experts=False 时：在这里加权
out_experts = bmm(top_scores.reshape(N,1,top_k), routed_output_unsorted).squeeze(1)

return (out + out_experts).reshape(B, S, 2048)   # 共享专家 + 路由专家，残差在 TransformerBlock 外加
```

---

#### 完整前向数据流总结

```
输入 x (B, S, 2048)
│
├──→ reshape → (N, 2048)
│        │
│        ├──→ [router]
│        │     gate(x) → sigmoid → topk → selected_indices (N, 1)
│        │                                num_tokens_per_expert (8,)
│        │
│        ├──→ [reorderer]
│        │     argsort → token_indices_sorted (N,)
│        │
│        ├──→ x[sorted_indices] * top_scores   → routed_input (N, 2048)  ← score_before_experts
│        │
│        ├──→ [GroupedExperts]     → routed_output (N, 2048)
│        │     grouped_mm(w1,w2,w3)   8 个专家，SwiGLU，一次并行
│        │
│        ├──→ [shared_experts]    → shared_out (N, 2048)
│        │     FeedForward，所有 token 必过
│        │
│        └──→ unsort + sum + reshape
│
输出 (B, S, 2048)   =  shared_out + routed_out（再经 TransformerBlock 的残差连接加上原始 x）
```

---

### 9.4 负载均衡机制（Auxiliary-Loss-Free）

**问题**：如果不做任何干预，路由网络容易发生"专家坍塌"——少数热门专家接收绝大多数 token，大多数专家几乎闲置，模型退化为等效几个 Dense FFN。

**传统方案**：在 loss 里加辅助损失（auxiliary loss），强制每个专家接收均匀的 token。
**缺点**：辅助 loss 的权重难以调，与主损失存在冲突，影响模型质量。

**DeepSeek V3 方案（`load_balance_coeff=1e-3`，来自论文 https://arxiv.org/abs/2408.15664）**：

不加辅助 loss，而是给路由打分加一个**自适应偏置 expert_bias**：

```
scores_for_choice = sigmoid(gate(x)) + expert_bias
↑ 这个偏置不参与计算路由权重（top_scores 仍用原始 sigmoid 分）
↑ 只用于决定路由到哪个专家
```

**expert_bias 的更新规则**（在 optimizer step 的 pre-hook 里执行，不参与梯度）：

```python
# 若专家 i 比平均值接收更多 token → bias 减小（让它少接收）
# 若专家 i 比平均值接收更少 token → bias 增大（让它多接收）
avg_tokens = tokens_per_expert.mean()
for i in range(num_experts):
    if tokens_per_expert[i] > avg_tokens:
        expert_bias[i] -= load_balance_coeff
    else:
        expert_bias[i] += load_balance_coeff
# 实际代码用 torch.sign() 实现，保证每次更新幅度恒定为 ±load_balance_coeff
```

**优势**：expert_bias 不影响梯度，不干扰主 loss 的优化，完全解耦负载均衡与模型训练。

---

### 9.5 Expert Parallel（EP）与 MoE 的配合

当多机多卡训练时，MoE 的 8 个专家可以分布在不同 GPU 上（Expert Parallel）。

**EP=4 的情况**（4 个 GPU，每卡 2 个专家）：

```
GPU 0: Expert 0, Expert 1
GPU 1: Expert 2, Expert 3
GPU 2: Expert 4, Expert 5
GPU 3: Expert 6, Expert 7

每个 GPU 上的 token 需要路由到不同 GPU 的专家：

Step 1 (Dispatch)：all-to-all 通信
  每个 GPU 把本地 token 发送到对应专家所在的 GPU

Step 2 (Compute)：本地计算
  每个 GPU 只计算自己持有的专家

Step 3 (Combine)：all-to-all 通信
  把计算结果发回 token 原来所在的 GPU

Step 4：unsort + 加权合并
```

EP 的通信量 = 2 × all-to-all（每个 forward 2 次，backward 再 2 次）。
节点限制路由（num_expert_groups + num_limited_groups）的作用正是减少跨节点的 all-to-all 流量。

---

### 9.6 MoE 优缺点总结

#### 优点

| 优点 | 说明 |
|---|---|
| 参数量大、计算量小 | 总参数量 = N 个专家之和，但每 token 只激活 top-k 个，FLOPs 接近 Dense |
| 专家专业化 | 不同专家自然学习不同类型的语言模式或知识领域 |
| 扩展性好 | 增加专家数几乎不增加推理计算量，只增加显存 |
| 与 EP 天然配合 | 专家分布在不同卡，通信开销可接受 |
| 无辅助 loss 均衡 | DeepSeek V3 的 bias 方案不干扰主任务梯度 |

#### 缺点

| 缺点 | 说明 |
|---|---|
| 显存开销大 | 所有专家参数必须加载（推理时 top-k 个专家的参数也需在显存中） |
| all-to-all 通信瓶颈 | EP 时每 step 需 2 次 all-to-all，跨节点带宽成为瓶颈 |
| 路由不稳定风险 | 训练初期路由随机，可能出现专家坍塌或死专家 |
| 动态 shape 难优化 | 每个专家收到的 token 数不固定，grouped_mm 的 padding 浪费算力 |
| 负载不均时效率低 | 某专家收到过多 token 时，该专家成为 pipeline 瓶颈 |
| 调试困难 | 路由是动态的，同一 token 在不同 step 可能走不同专家，复现问题困难 |
