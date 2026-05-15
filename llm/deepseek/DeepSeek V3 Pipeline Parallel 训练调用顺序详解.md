# DeepSeek V3 Pipeline Parallel 训练调用顺序详解

**配置约定**：本文以 `n_layers=16, pp=2（即 pp_degree=2，pp_rank=0 和 pp_rank=1）` 为例。
若未特别说明，采用默认 1F1B 单阶段调度（`PipelineSchedule1F1B`）。

---

## 一、DeepSeek V3 完整模型结构（切分前）

```
DeepSeekV3Model                                  (train.py 中用 meta 设备构建)
├── tok_embeddings : nn.Embedding(102400, 2048)
├── freqs_cis      : buffer，预计算的 RoPE 频率张量 shape=(max_seq_len, qk_rope_head_dim//2)
├── layers         : nn.ModuleDict                (共 16 个 TransformerBlock)
│   ├── "0"  : TransformerBlock(layer_id=0)       ← Dense 层（layer_id < n_dense_layers=1）
│   │   ├── attention_norm : RMSNorm(2048)
│   │   ├── attention      : Attention (MLA)
│   │   │   ├── wq_a / q_norm / wq_b             （q_lora_rank>0 时）或 wq（q_lora_rank=0 时）
│   │   │   ├── wkv_a  : Linear(2048 → 512+64)
│   │   │   ├── kv_norm: RMSNorm(512)
│   │   │   ├── wkv_b  : Linear(512 → 16*(128+128))
│   │   │   ├── wo     : Linear(16*128 → 2048)
│   │   │   └── inner_attention : SDPA / FlexAttention
│   │   ├── ffn_norm      : RMSNorm(2048)
│   │   └── feed_forward  : FeedForward(dim=2048, hidden_dim=inter_dim)
│   │       ├── w1: Linear(2048 → inter_dim)
│   │       ├── w2: Linear(inter_dim → 2048)
│   │       └── w3: Linear(2048 → inter_dim)
│   │
│   ├── "1"  : TransformerBlock(layer_id=1)       ← MoE 层（layer_id >= 1）
│   │   ├── attention_norm : RMSNorm(2048)
│   │   ├── attention      : Attention (MLA)      （结构同上）
│   │   ├── ffn_norm       : RMSNorm(2048)
│   │   └── moe            : MoE                  （替换 feed_forward）
│   │       ├── router     : 门控路由网络
│   │       ├── experts    : 专家 FFN 列表
│   │       └── shared_experts : 共享专家（可选）
│   │
│   ├── "2" ~ "15" : TransformerBlock             ← 全部为 MoE 层
│   │   （结构同 "1"）
│
├── norm   : RMSNorm(2048)
└── output : Linear(2048, 102400, bias=False)
```

**关键参数**（默认值）：
| 参数 | 值 |
|------|-----|
| vocab_size | 102400 |
| dim | 2048 |
| n_layers | 16（本例设定） |
| n_dense_layers | 1（layer.0 是 Dense MLP，其余是 MoE） |
| n_heads | 16 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| v_head_dim | 128 |

---

## 二、PP 初始化阶段：模型切分与 Stage 构建

### Step 1：用 meta 设备构建整体模型

文件：`torchtitan/train.py` → `Trainer.__init__`（约第 147 行）

```python
with torch.device("meta"), utils.set_default_dtype(...):
    model = self.train_spec.model_cls(model_args)  # DeepSeekV3Model
```

此时模型所有参数均在 meta 设备上，**不占用实际 GPU 内存**。整个模型完整存在于所有 rank 的 CPU 内存中（meta tensor 仅保存 shape/dtype，无实际数据）。

### Step 2：进入 PP 分支，调用 pipeline_llm

文件：`torchtitan/train.py` 第 218~239 行

```python
if parallel_dims.pp_enabled:
    (self.pp_schedule, self.model_parts,
     self.pp_has_first_stage, self.pp_has_last_stage) = self.train_spec.pipelining_fn(
        model, parallel_dims, job_config, self.device, model_args,
        self.train_spec.parallelize_fn, self.loss_fn,
    )
```

`pipelining_fn` 对应 `pipeline_llm`（位于 `torchtitan/distributed/pipeline_parallel.py`）。

### Step 3：计算各 Stage 的模块分配

文件：`pipeline_parallel.py` → `generate_llm_fqn_per_model_part`（第 236 行）

**输入参数**：
- `num_stages = 2`（单阶段调度，每 rank 1 个 stage，pp=2）
- `num_layers = 16`
- `input_weight = 1`（tok_embeddings 等价于 1 层）
- `output_weight = 1`（norm+output 等价于 1 层）

**计算过程**：
```
num_effective_layers = 16 + 1 + 1 = 18
layers_per_stage = 18 // 2 = 9
extra_layers = 18 % 2 = 0

Stage 0（rank 0）:
  - tok_embeddings（消耗 input_weight=1）
  - remaining = 9 - 1 = 8 个 transformer 层
  - → layers.0, layers.1, ..., layers.7

Stage 1（rank 1）:
  - remaining = 9 - 1 = 8 个 transformer 层
  - → layers.8, layers.9, ..., layers.15
  - norm（消耗 output_weight=1 中的 0.5，共同）
  - output
```

**最终 module_names_per_stage**：
```python
[
    ["tok_embeddings", "layers.0", "layers.1", ..., "layers.7"],   # Stage 0 → rank 0
    ["layers.8", "layers.9", ..., "layers.15", "norm", "output"],  # Stage 1 → rank 1
]
```

### Step 4：调用 pipeline_module_split 裁剪模型

文件：`pipeline_parallel.py` → `pipeline_module_split`（第 347 行）

**rank 到 stage 的映射**（loop 风格调度）：
```
pp_rank=0  →  stage_idx=0
pp_rank=1  →  stage_idx=1
```

对每个 stage，调用 `_build_stage_from_modules` 做如下操作：

1. `copy.deepcopy(whole_model)`：深拷贝整个模型（含 meta 参数）
2. 遍历顶层子模块，**保留该 stage 需要的，删除不需要的**：
   - 对于 `nn.ModuleDict`（`model.layers`）：删除不在本 stage 的 key
   - 对于普通子模块（`tok_embeddings`, `norm`, `output`）：不需要的设为 `None`
3. 用裁剪后的模型创建 `PipelineStage` 对象

**裁剪后各 rank 持有的模型结构**：

```
rank 0 持有（stage_idx=0）：
DeepSeekV3Model (裁剪版)
├── tok_embeddings : nn.Embedding(102400, 2048)   ✓ 保留
├── freqs_cis      : buffer                        ✓ 保留（buffer 不在 named_children 中，始终保留）
├── layers         : nn.ModuleDict
│   ├── "0"  : TransformerBlock  ← Dense MLP 层    ✓ 保留
│   ├── "1"  : TransformerBlock  ← MoE 层           ✓ 保留
│   ├── ...
│   └── "7"  : TransformerBlock  ← MoE 层           ✓ 保留
│   （"8"~"15" 已被删除）
├── norm   : None                                   ✗ 设为 None
└── output : None                                   ✗ 设为 None
```

```
rank 1 持有（stage_idx=1）：
DeepSeekV3Model (裁剪版)
├── tok_embeddings : None                           ✗ 设为 None
├── freqs_cis      : buffer                        ✓ 保留
├── layers         : nn.ModuleDict
│   ├── "8"  : TransformerBlock  ← MoE 层           ✓ 保留
│   ├── "9"  : TransformerBlock  ← MoE 层           ✓ 保留
│   ├── ...
│   └── "15" : TransformerBlock  ← MoE 层           ✓ 保留
│   （"0"~"7" 已被删除）
├── norm   : RMSNorm(2048)                          ✓ 保留
└── output : Linear(2048, 102400)                  ✓ 保留
```

**注意**：`DeepSeekV3Model.forward()` 中对 `None` 子模块做了容错处理（第 499~504 行）：
```python
h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
for layer in self.layers.values():
    h = layer(h, self.freqs_cis, attention_masks, positions)
h = self.norm(h) if self.norm is not None else h
output = self.output(h) if self.output is not None else h
```
因此 rank 1 接收激活张量直接进入 layers.8，rank 0 的 forward 不计算 norm/output。

### Step 5：对每个 model_part 应用 SPMD 并行化

文件：`pipeline_parallel.py` 第 136~142 行；并行化函数为 `parallelize_deepseekv3`

```python
for i, m in enumerate(model_parts):
    m = parallelize_fn(m, parallel_dims, job_config)
    model_parts[i] = m
    stages[i].submod = m
```

`parallelize_deepseekv3`（位于 `torchtitan/models/deepseek_v3/infra/parallelize.py`）依次做：

1. **TP（Tensor Parallel）**：若启用，对每个 TransformerBlock 的 Attention 和 FFN/MoE 线性层按列/行切分
2. **EP（Expert Parallel）**：若启用，对 MoE 层的 expert 进行切分
3. **CP（Context Parallel）**：若启用，对 Attention 内核进行 sequence 维度切分
4. **Activation Checkpointing**：若启用，对 TransformerBlock 包装 `torch.utils.checkpoint`
5. **torch.compile**：若启用，对模型编译
6. **FSDP/HSDP**：对参数进行 ZeRO 分片

**各 rank 在 SPMD 并行化后**：
- rank 0 的 stage_0 模型：只含 tok_embeddings + layers.0~7，同时可能被 FSDP/TP 进一步切分
- rank 1 的 stage_1 模型：只含 layers.8~15 + norm + output，同样可能被 FSDP/TP 进一步切分

### Step 6：构建 PipelineSchedule

文件：`pipeline_parallel.py` → `build_pipeline_schedule`（第 158 行）

```python
pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)
```

以 1F1B 为例：
- `schedule_class = Schedule1F1B`（`PipelineScheduleSingle`）
- `n_microbatches = local_batch_size // microbatch_size`
- 创建 `Schedule1F1B(stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn)`

### Step 7：模型权重初始化

文件：`train.py` 第 244~248 行

```python
for m in self.model_parts:
    m.to_empty(device=init_device)     # 把 meta tensor 转为真实 GPU tensor（未初始化）
    with torch.no_grad():
        cast(ModelProtocol, m).init_weights(buffer_device=buffer_device)
    m.train()
```

`DeepSeekV3Model.init_weights()` 的执行逻辑：
- rank 0：初始化 `tok_embeddings.weight`（normal init）+ layers.0~7 的所有参数 + 重计算 `freqs_cis` buffer
- rank 1：初始化 layers.8~15 的所有参数 + `norm`（reset_parameters）+ `output.weight`（trunc_normal）+ 重计算 `freqs_cis` buffer
- `norm=None / output=None / tok_embeddings=None` 的情况会被 `if xxx is not None` 跳过

---

## 三、训练循环中的 PP 前向/反向传播

### 总体流程（train_step 视角）

文件：`train.py` → `train_step`（第 547 行）

```
train_step
├── optimizers.zero_grad()
├── 收集 microbatches（CPU 上预加载）
├── 全局 valid token 数统计（all-reduce）
└── for each microbatch:
    └── forward_backward_step(input_dict, labels, global_valid_tokens)
        └── pp_schedule.step(inputs, target=labels, losses=losses)
            └── [1F1B 调度驱动多个 microbatch 的前向+反向]
```

### pp_schedule.step() 内部驱动的执行顺序

以 `n_microbatches=4` 为例，两个 rank 上 1F1B 调度时序：

```
时间 →    T1      T2      T3      T4      T5      T6      T7      T8
rank0:   F(m0)  F(m1)  F(m2)  B(m0)  F(m3)  B(m1)  B(m2)  B(m3)
rank1:          F(m0)  F(m1)  F(m2)  B(m0)  F(m3)  B(m1)  B(m2)  B(m3)

F = 前向传播  B = 反向传播  m0~m3 = 4个微批次
```

### 前向传播详细过程（以 microbatch m0 为例）

#### rank 0 前向（Stage 0）

**触发条件**：`pp_schedule.step()` 调度到 rank 0 执行 F(m0)

**输入**：
- `tokens`：形状 `(microbatch_size, seq_len)` 的 token id 张量（仅 rank 0 从 dataloader 取得）
- `attention_masks`（若为 flex attn）：`BlockMask` 对象
- `positions`（可选）：位置 id

**执行的 forward() 逻辑**（`DeepSeekV3Model.forward`）：

```
Step 0-1: tok_embeddings(tokens)
  输入: tokens, shape=(B, S)  例如 B=microbatch_size, S=seq_len
  输出: h, shape=(B, S, 2048)
  计算: 查找 Embedding 表，将 token id 映射为 2048 维向量

Step 0-2: layers["0"](h, freqs_cis, attention_masks)   ← Dense 层
  内部流程：
    h_norm = attention_norm(h)                          # RMSNorm(2048)
    attn_out = attention(h_norm, freqs_cis, masks)
      ├── q = wq_a(h_norm); q = wq_b(q_norm(q))        # LoRA Q 投影（若 q_lora_rank>0）
      ├── kv = wkv_a(h_norm)                            # shape=(B,S,576=512+64)
      ├── kv, k_pe = split(kv, [512, 64], dim=-1)
      ├── k_pe = apply_rotary_emb(k_pe, freqs_cis)      # RoPE 旋转位置编码
      ├── kv = wkv_b(kv_norm(kv))                       # shape=(B,S,16*(128+128))
      ├── k_nope, v = split(kv, [128, 128], dim=-1)
      ├── k = cat([k_nope, k_pe])                        # shape=(B,S,16,192)
      ├── q/k/v transpose → (B,16,S,192)/(B,16,S,192)/(B,16,S,128)
      ├── output = SDPA(q, k, v, scale=softmax_scale)   # FlashAttention
      └── return wo(output.reshape(B,S,-1))              # shape=(B,S,2048)
    h = h + attn_out                                    # 残差连接
    ffn_out = feed_forward(ffn_norm(h))                 # Dense MLP
      ├── gate = SiLU(w1(x)) * w3(x)
      └── return w2(gate)
    h = h + ffn_out                                     # 残差连接
  输出: h, shape=(B, S, 2048)

Step 0-3 ~ 0-9: layers["1"] ~ layers["7"]              ← MoE 层（各层结构相同）
  各 MoE TransformerBlock 内部：
    h_norm = attention_norm(h)
    attn_out = attention(h_norm, freqs_cis, masks)      # 同上 MLA 计算
    h = h + attn_out
    moe_out = moe(ffn_norm(h))
      ├── router(x) → topk expert indices + weights     # 门控路由
      ├── dispatch tokens to selected experts           # all-to-all（若 EP 启用）
      ├── expert_output = experts[i](token_subset)      # 每个 expert 的 FFN
      ├── combine expert outputs                         # all-to-all（若 EP 启用）
      └── + shared_expert_output（若有共享专家）
    h = h + moe_out
  最后 h shape=(B, S, 2048)

Step 0-10: norm=None, 跳过
Step 0-11: output=None, 跳过
```

**rank 0 前向输出**：激活张量 `h`，shape `(microbatch_size, seq_len, 2048)`

**激活传递**（rank 0 → rank 1）：
- `PipelineStage` 自动调用 `dist.send(h, dst=1)`（PP 通信组内）
- rank 1 通过 `dist.recv(h_recv, src=0)` 接收
- 传输的是一个完整的激活张量，不包含梯度

#### rank 1 前向（Stage 1）

**输入**：从 rank 0 接收的激活张量 `h`，shape `(B, S, 2048)`

**执行的 forward() 逻辑**：

```
Step 1-0: tok_embeddings=None，跳过（直接使用接收到的 h 作为输入）

Step 1-1 ~ 1-8: layers["8"] ~ layers["15"]             ← 全部 MoE 层
  （与 rank 0 的 MoE 层计算完全相同，共 8 层）
  最后 h shape=(B, S, 2048)

Step 1-9: norm(h)                                       ← RMSNorm
  输出: h_normed, shape=(B, S, 2048)

Step 1-10: output(h_normed)                             ← Linear
  输出: logits, shape=(B, S, 102400)
```

**loss 计算**（rank 1 执行，因为 `has_last_stage=True`）：
```python
loss = loss_fn(logits, labels)    # CrossEntropyLoss，labels shape=(B, S)
losses.append(loss)
```

### 反向传播详细过程（以 microbatch m0 为例）

1F1B 调度在 rank 1 完成 F(m0) 后，紧接着在 rank 1 执行 B(m0)。

#### rank 1 反向

```
loss.backward()
  ├── d_logits = dL/d_logits, shape=(B, S, 102400)
  ├── output.weight.grad 累积
  ├── d_h_normed = d_logits @ output.weight.T, shape=(B, S, 2048)
  ├── norm 的梯度
  ├── layers["15"].backward() ... layers["8"].backward()  （逆序）
  │   各 MoE 层：
  │   ├── moe 反向（专家 FFN 梯度）
  │   ├── MLA Attention 反向
  │   ├── 参数梯度累积到 .grad
  │   └── 输入梯度 d_h 传回
  └── 输出：d_h（对 Stage 0 输出激活的梯度），shape=(B, S, 2048)
```

**梯度传递**（rank 1 → rank 0）：
- `PipelineStage` 自动调用 `dist.send(d_h, dst=0)`
- rank 0 通过 `dist.recv(d_h_recv, src=1)` 接收

#### rank 0 反向

```
接收 d_h（来自 rank 1），shape=(B, S, 2048)
  ├── layers["7"].backward() ... layers["0"].backward()  （逆序）
  │   各层 Attention + FFN/MoE 梯度累积
  └── tok_embeddings 反向
      └── tok_embeddings.weight.grad 累积（稀疏梯度，仅被访问的 token id 有梯度）
```

---

## 四、参数更新阶段

文件：`train.py` → `train_step` 第 592~601 行

```python
grad_norm = dist_utils.clip_grad_norm_(
    [p for m in self.model_parts for p in m.parameters()],
    max_norm,
    pp_mesh=parallel_dims.get_optional_mesh("pp"),
)
self.optimizers.step()     # 各 rank 只更新自己持有的 stage 的参数
self.lr_schedulers.step()
```

**各 rank 更新的参数**：
- rank 0：tok_embeddings + layers.0~7 的所有参数（约 = 1/2 模型参数）
- rank 1：layers.8~15 + norm + output 的所有参数（约 = 1/2 模型参数）

---

## 五、完整调用栈总览（一次 train_step）

```
Trainer.train_step(data_iterator)
│
├── optimizers.zero_grad()
│
├── [收集 microbatches 到 CPU]
│
├── dist_sum(local_valid_tokens)  ← DP 维度 all-reduce
│
└── forward_backward_step(input_dict, labels)
    │
    ├── post_dataloading_process()
    │   └── model.get_attention_masks()（若 flex attn）
    │
    └── pp_schedule.step(inputs, target=labels, losses=losses)
        │
        └── [Schedule1F1B 内部，驱动 n_microbatches 次迭代]
            │
            ├── [rank 0 F(m0)]
            │   └── PipelineStage.forward(tokens)
            │       └── DeepSeekV3Model.forward()
            │           ├── tok_embeddings(tokens)
            │           ├── layers["0"].forward()  ← Dense
            │           ├── layers["1"].forward()  ← MoE
            │           ├── ...
            │           └── layers["7"].forward()  ← MoE
            │       └── dist.send(h, dst=rank1)    ← PP 通信
            │
            ├── [rank 1 F(m0)]
            │   └── PipelineStage.forward(recv_h)
            │       └── DeepSeekV3Model.forward()
            │           ├── layers["8"].forward()  ← MoE
            │           ├── ...
            │           ├── layers["15"].forward() ← MoE
            │           ├── norm(h)
            │           └── output(h)              → logits
            │       └── loss_fn(logits, labels)    → loss
            │
            ├── [rank 1 B(m0)]
            │   └── PipelineStage.backward()
            │       └── loss.backward()
            │           ├── output / norm 梯度
            │           ├── layers["15"] ~ layers["8"] 梯度
            │           └── dist.send(d_h, dst=rank0)  ← PP 梯度通信
            │
            ├── [rank 0 B(m0)]
            │   └── PipelineStage.backward()
            │       └── 接收 d_h
            │           ├── layers["7"] ~ layers["0"] 梯度
            │           └── tok_embeddings 梯度
            │
            └── ... [重复上述过程处理 m1, m2, m3]
        │
        └── 返回 losses 列表
    │
    └── loss = sum(losses) / global_valid_tokens
        （仅 rank 1 有意义，rank 0 返回 tensor([-1.0])）
```

---

## 六、业界通用的 Pipeline Parallel 模型执行顺序

### 6.1 朴素流水线（Naive Pipeline / GPipe 变体）

最早的 PP 方案，将 batch 切成 microbatch，顺序执行：

```
时间 →  T1    T2    T3    T4    T5    T6    T7    T8
rank0: F(m0) F(m1) F(m2) F(m3)       B(m0) B(m1) B(m2) B(m3)
rank1:       F(m0) F(m1) F(m2) F(m3)             B(m0) B(m1) B(m2) B(m3)

气泡率 = (pp-1)/(pp+n_microbatch-1)
```

**缺点**：显存峰值高（需保存所有 microbatch 激活），气泡率高。

### 6.2 1F1B（One Forward One Backward）—— GPipe 改进，PipeDream 提出

每完成一个前向后立即安排一个反向，减少激活显存：

```
时间 →  T1    T2    T3    T4    T5    T6    T7    T8
rank0: F(m0) F(m1) F(m2) B(m0) F(m3) B(m1)       B(m2) B(m3)
rank1:       F(m0) F(m1) F(m2) B(m0) F(m3) B(m1)       B(m2) B(m3)
```

**特点**：
- 显存峰值 = pp 个 microbatch 的激活（而非全部）
- 气泡率 = (pp-1)/(n_microbatch+pp-1)
- **torchtitan 默认使用此调度**（`Schedule1F1B`）

### 6.3 Interleaved 1F1B（虚拟流水线，Megatron-LM 提出）

每个 rank 持有多个不连续的 stage（interleaved），进一步减少气泡：

```
配置：pp=2, 每 rank 2 个 stage（virtual_stages_per_rank=2）
rank 0 持有 stage 0 (layers 0-3) + stage 2 (layers 8-11)
rank 1 持有 stage 1 (layers 4-7) + stage 3 (layers 12-15)

时间 →  T1    T2    T3    ...
rank0: F0(m0) F0(m1) ... F2(m0) ... B2(m0) ... B0(m0) ...
rank1:   F1(m0) F1(m1)...  F3(m0)... B3(m0)... B1(m0)...
```

**特点**：
- 气泡率 = 1/n_microbatch（相比 1F1B 的 (pp-1)/(n_microbatch+pp-1) 更小）
- 每 rank 需 2x 通信（send/recv 更频繁）
- **torchtitan 支持** `ScheduleLoopedBFS` / `ScheduleInterleaved1F1B` 等

### 6.4 ZB-H1 / ZB-H2 / ZBV（零气泡流水线）

通过精细化调度（将反向拆分为 B 和 W 两个阶段）实现接近零气泡：

```
配置：ZBV（V 形调度）
rank 0 持有 stage 0 + stage (2*pp-1)
rank 1 持有 stage 1 + stage (2*pp-2)

前半 →      后半
rank 0 → stage 0   AND   stage 3
rank 1 → stage 1   AND   stage 2

执行顺序类似 V 形：0→1→2→3→3→2→1→0
```

**特点**：
- 几乎消除流水线气泡
- 权重梯度（W pass）可延迟到更晚时机执行
- **torchtitan 支持** `ScheduleZBVZeroBubble` 和 `ScheduleDualPipeV`

### 6.5 DualPipeV（DeepSeek 提出）

专为双向流水线设计，与 EP（Expert Parallel）深度结合，支持计算/通信 overlap：

```
同时运行两个方向的流水线（通过注册 OVERLAP_F_B 回调）
适合 MoE 模型中 EP all-to-all 通信与 Dense 计算的重叠
```

torchtitan 实现：`ScheduleDualPipeV` + `overlap_callback`（`dual_pipe_v.py`）

### 6.6 各调度方案对比

| 调度方案 | 气泡率 | 激活显存 | 通信频率 | 适用场景 |
|---|---|---|---|---|
| Naive / GPipe | (pp-1)/(m+pp-1) | O(m*pp) | 低 | 小规模/调试 |
| 1F1B | (pp-1)/(m+pp-1) | O(pp) | 低 | 通用，torchtitan 默认 |
| Interleaved 1F1B | 1/m | O(v*pp) | 高 | m 较小时优势明显 |
| ZB-H1 | 接近 0 | O(pp) | 中 | 追求极致利用率 |
| ZBV | 接近 0 | O(2) | 中 | pp=2 时性能最佳 |
| DualPipeV | 接近 0 | O(2) | 中高 | MoE + EP 场景 |

> **m** = n_microbatches，**v** = virtual stages per rank

---

## 七、torchtitan 中 DeepSeek V3 + PP 的关键设计约束

1. **模型 forward 必须容忍 None 子模块**：`_split_module` 将非本 stage 模块设为 None，模型代码中用 `if xxx is not None` 跳过计算。

2. **freqs_cis buffer 在所有 rank 上都存在**：每个裁剪后的模型副本都保留了完整的 freqs_cis buffer，因为它是 `register_buffer`，不在 `named_children()` 遍历中，不会被删除。

3. **extra_kwargs 跨 stage 传递，extra_inputs 不传递**：`attention_masks` 放在 `extra_kwargs` 中，会被 `PipelineStage` 通过 PP 通信传递给后续 stage；而 `positions` 等非必要输入放在 `extra_inputs` 中，只传给 first stage。

4. **Loss 只在 last stage（rank 1）计算**：`has_last_stage=True` 的 rank 传入 `target=labels`，其他 rank 传入 `target=None`，最终 loss 张量只在 rank 1 上有效。

5. **MoE 的 all-to-all 通信与 PP 通信互不干扰**：EP 使用独立的 process group，PP 使用独立的 pp 通信组，两者可并行发生（DualPipeV overlap 机制）。

6. **参数更新完全局部化**：每个 rank 只拥有自己 stage 的参数，optimizer.step() 只更新本地参数，无需跨 stage 通信。
