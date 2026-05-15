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

---

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
