# DeepSeek V3/V3.2 CP NPU 适配需求文档

## 1. 需求背景

TorchTitan 升级到 0.2.2 后，框架侧 Context Parallel（CP）的接入方式、训练输入切分方式、模型并行化入口和 DeepSeek 系列模型的模块结构都发生了变化。当前 `torchtitan-npu` 代码仓作为 TorchTitan 在 NPU 上执行训练的 patch 仓，需要跟随 TorchTitan 0.2.2 做同步适配，保证已有 DeepSeek V3 和 DeepSeek V3.2 训练能力在 NPU 上不退化。

本需求聚焦 DeepSeek V3 与 DeepSeek V3.2 两个模型开启 CP 后的 NPU 可运行性和精度一致性。两个模型的 CP 路线不同：

- DeepSeek V3 使用标准 SDPA/MLA 注意力，当前适配通过 Ulysses CP 在注意力计算前后插入 all-to-all 通信，使每个 rank 在注意力内部拿到完整序列上下文。
- DeepSeek V3.2 使用 DeepSeek Sparse Attention（DSA）和 NPU sparse attention/indexer 融合算子，当前适配通过 DSA CP 在 CP 域内聚合 KV、Indexer KV 与 RoPE KV，并按因果范围裁剪，保证稀疏注意力和 indexer loss 计算正确。

与 2026_03 旧仓相比，旧实现基于 TorchTitan 0.2.1，主要通过 `custom_context_parallel_path` 配置选择自定义 CP Context。例如 DeepSeek V3 配置 `UlyssesContextParallelContext`，DeepSeek V3.2 配置 `AscendDSAContextParallelContext`，再由 `create_context_parallel_ctx` patch 在进入 CP context 时替换 attention forward。TorchTitan 0.2.2 后，当前仓改为 patch `torchtitan.distributed.context_parallel.apply_cp_to_attention_module`，由模型并行化阶段根据模型和 converter 自动选择 `ulysses` 或 `dsa` 分支，配置侧不再需要写自定义 CP 类路径。

因此本次适配的本质是：把旧仓中可用的 Ulysses CP 和 DSA CP 能力，迁移到 TorchTitan 0.2.2 的 CP 调用链、输入切分、DTensor layout 和并行化计划中，并作为 NPU patch 自动生效。

## 2. 需求目标

### 2.1 功能目标

1. DeepSeek V3 开启 CP 后可在 NPU 上正常训练。
   - 当 `parallelism.context_parallel_degree > 1` 且 `enable_custom_context_parallel = true` 时，DeepSeek V3 自动走 Ulysses CP 分支。
   - Attention 内部通过 all-to-all 完成 `head` 维和 `sequence` 维重排，使 SDPA 能看到完整上下文。
   - Attention 外部仍保持 TorchTitan 0.2.2 的序列维 CP 切分方式，兼容 TP、EP、FSDP、activation checkpoint 等既有并行能力。

2. DeepSeek V3.2 开启 CP 后可在 NPU 上正常训练。
   - 当 DeepSeek V3.2 配置 `npu_dsa` converter 且 CP degree 大于 1 时，自动走 DSA CP 分支。
   - DSA attention 使用 NPU `npu_lightning_indexer`、`npu_sparse_flash_attention`、`npu_sparse_lightning_indexer_grad_kl_loss` 相关融合路径。
   - CP 域内正确聚合 DSA 所需的 K/V、Indexer K、RoPE K，并按 rank 的因果可见范围裁剪。
   - DSA indexer loss 在 CP、TP、DP、PP 等并行维度下同步统计，日志指标可用于精度归档。

3. 配置方式与 TorchTitan 0.2.2 适配。
   - 保留 `enable_custom_context_parallel` 作为 NPU 自定义 CP 开关。
   - 移除对旧配置项 `custom_context_parallel_path` 的依赖。
   - CP 类型由代码自动选择：DeepSeek V3 选择 Ulysses，DeepSeek V3.2 + `npu_dsa` 选择 DSA。

### 2.2 精度目标

1. CP 开启后，DeepSeek V3 和 DeepSeek V3.2 的 loss 曲线需要与对应基线对齐。
   - 可用 CP=1 的 NPU 训练结果作为直接基线。
   - 如需要横向确认，也可对照 2026_03 旧仓在相同模型、数据、seed、并行配置下的可用结果。

2. 训练过程中不得出现由 CP 适配引入的 NaN/Inf。
   - 主 loss、grad norm、DSA indexer loss 均应保持有限值。
   - DeepSeek V3.2 需要重点观察 DSA indexer loss 是否随训练 step 正常记录和清理。

3. CP 分片不应改变语义。
   - RoPE 必须使用全局 token 位置。
   - DSA 的稀疏索引、KV 可见范围、softmax LSE 和 indexer loss 必须与全序列因果语义一致。
   - TP 与 CP 同开时，头维切分和序列维切分不能导致 DSA loss 输入缺头或重复聚合。

### 2.3 约束目标

1. Ulysses CP 要求 `seq_len` 能被 CP degree 整除，`n_heads` 能被 CP degree 以及 `TP degree * CP degree` 整除。
2. DeepSeek V3.2 DSA CP 当前只支持 SDPA/DSA 路线，不支持 flex attention 或 varlen attention 与 CP 同开。
3. DSA CP 依赖 `npu_dsa` converter；当 attention type 被路由为 `dsa` 时，配置中必须包含 `npu_dsa`。
4. DSA CP 当前按 MLA absorb 路径实现，要求 `num_head_kv == 1`。
5. DSA CP 的因果裁剪依赖 CP rank 的顺序语义，因此 DeepSeek V3.2 + DSA 需要使用顺序序列切分，不能使用 HeadTail load balance。

## 3. 需求实现

### 3.1 总体实现方案

当前仓在 import `torchtitan_npu` 时统一注册 NPU patch。与 CP 相关的主入口包括：

- `torchtitan_npu/__init__.py`：加载 distributed CP patch、DeepSeek V3/V3.2 模型 patch，并将 `deepseek_v32` 注入到 `torchtitan.models`。
- `torchtitan_npu/config/custom_config.py`：扩展 TorchTitan 0.2.2 配置，保留 `enable_custom_context_parallel`，不再提供旧的 `custom_context_parallel_path`。
- `torchtitan_npu/patches/distributed/custom_context_parallel.py`：patch TorchTitan 原生 `apply_cp_to_attention_module`，新增 `ulysses` 和 `dsa` 两条分支。
- `torchtitan_npu/distributed/context_parallel/ulysses_cp.py`：实现 Ulysses all-to-all CP。
- `torchtitan_npu/distributed/context_parallel/dsa_cp.py`：实现 DeepSeek V3.2 DSA CP。
- `torchtitan_npu/patches/distributed/cp_input_sharding.py`：对 DeepSeek V3.2 + DSA 强制使用顺序 CP 切分。
- `torchtitan_npu/train.py`：补充 RoPE 全局 position broadcast、DSA indexer loss scale 和日志统计 patch。

适配后的 CP 调用链为：

```text
Trainer.post_dataloading_process
  -> prepare_context_parallel_input / cp_shard
  -> 模型输入、labels、positions 按 CP 切分

parallelize_deepseekv3 / parallelize_deepseekv32
  -> apply_cp_to_attention_module(...)
  -> torchtitan_npu.patches.distributed.custom_context_parallel.apply_cp_to_attention_module
     -> attention_type == "ulysses": patch_ulysses_for_context_parallel(...)
     -> attention_type == "dsa": patch_dsa_for_context_parallel(...)
     -> 其他类型：回退 TorchTitan 原生 CP
```

### 3.2 DeepSeek V3 Ulysses CP 实现

DeepSeek V3 复用 TorchTitan 0.2.2 原生 `torchtitan.models.deepseek_v3` 模型主体，NPU 侧主要在 `torchtitan_npu/models/deepseek_v3/infra/parallelize.py` 进行包装。

实现要点如下：

1. 包装上游 `parallelize_deepseekv3`。
   - `_parallelize_deepseekv3_wrapper` 在不启用自定义 CP 或 CP degree 为 1 时直接回退上游逻辑。
   - 当 `enable_custom_context_parallel = true` 且 `context_parallel_degree > 1` 时，将上游并行化函数内部的 `apply_cp_to_attention_module` 替换为 NPU Ulysses 路由。
   - 调用时强制传入 `attention_type="ulysses"`，并补充 `job_config` 与 `model_args`，便于后续做合法性校验。

2. 复用 DeepSeek V3.2 的 MoE EP/TP 适配。
   - 当前文件将 `apply_moe_ep_tp` 注入上游 DeepSeek V3 parallelize 的 globals 中。
   - 这样 DeepSeek V3 在 TorchTitan 0.2.2 下仍使用 NPU 适配后的 MoE 并行计划，避免 EP/TP 组合下的 layout 不一致。

3. Ulysses CP 校验。
   - `validate_ulysses_configs` 校验 `n_heads % cp_degree == 0`。
   - 校验 `seq_len % cp_degree == 0`。
   - 当 TP 与 CP 同开时，校验 `n_heads % (tp_degree * cp_degree) == 0`。

4. Ulysses all-to-all 数据流。
   - 输入 q/k/v 布局为 `[B, n_heads, seq_len / CP, head_dim]`。
   - 第一次 all-to-all：沿 head 维 scatter，沿 sequence 维 gather，得到 `[B, n_heads / CP, seq_len, head_dim]`。
   - 调用原始 `ScaledDotProductAttentionWrapper.forward`，此时每个 rank 拥有完整上下文。
   - 第二次 all-to-all：沿 sequence 维 scatter，沿 head 维 gather，恢复为 `[B, n_heads, seq_len / CP, head_dim]`。
   - `AllToAll` 自定义 autograd function 在 backward 中交换 scatter/gather 维度，保证反向梯度通信与 forward 对称。

5. 适配收益。
   - 模型其他部分继续使用 TorchTitan 0.2.2 的 CP 输入切分和 DTensor 语义。
   - 仅 attention 内部临时重排，减少对 DeepSeek V3 模型代码的侵入。
   - 兼容 NPU fused attention、TP、EP、FSDP 和 activation checkpoint。

### 3.3 DeepSeek V3.2 DSA CP 实现

DeepSeek V3.2 在当前仓中提供完整 NPU 模型包，入口为 `torchtitan_npu/models/deepseek_v32/__init__.py`，并行化实现为 `torchtitan_npu/models/deepseek_v32/infra/parallelize.py`。

实现要点如下：

1. 并行化入口适配 TorchTitan 0.2.2。
   - `parallelize_deepseekv32` 基于 TorchTitan 0.2.2 DeepSeek V3 和 Llama4 parallelize 结构适配。
   - 保留 `seq_len % parallel_dims.seq_len_divisor == 0` 的整除约束。
   - 保留 mixed precision 训练要求 FSDP enabled 的约束。
   - 禁止 `npu_gmm` 在仅 TP、无 EP 的组合下使用，避免 grouped matmul 路径不支持。

2. CP 路由。
   - 当 `parallel_dims.cp_enabled` 时，收集所有 transformer block 的 `attention.inner_attention`。
   - 默认使用模型 `attn_type`。
   - 如果 `attn_type == "sdpa"` 且 `model.converters` 包含 `npu_dsa`，则将 CP attention type 覆盖为 `dsa`。
   - 覆盖原因是 DSA 需要绕开 DTensor dispatcher 对 `npu_lightning_indexer` 的拦截，直接进入 NPU DSA CP 实现。
   - 调用 `apply_cp_to_attention_module(..., job_config=job_config, model_args=model.model_args, tp_mesh=...)`，把 CP mesh、模型参数和 TP mesh 一并传入 patch 分支。

3. DSA CP 校验与 patch。
   - `validate_dsa_converters` 要求 `job_config.model.converters` 中必须包含 `npu_dsa`。
   - `patch_dsa_for_context_parallel` 给 `DSASparseAttention` 和 `DSV32_SDPA` 类挂载 `cp_mesh`、`model_args`、`tp_mesh`。
   - 将 `compute_dsa_indexer_loss` 替换为 `SparseLightningIndexerKLLoss`。
   - 将 attention forward 替换为 `dsa_forward_with_cp`。

4. DSA CP forward 数据流。
   - 原始 q/k/v 从 BNSD 转换为 BSND。
   - `k_indexer` 先在 CP 域 all-gather 得到全局 `k_indexer_global`。
   - 根据 CP rank 计算 `slice_end = local_seq_len * (cp_rank + 1)`，只保留当前 rank 因果可见的 key/indexer 范围。
   - 调用 `torch_npu.npu_lightning_indexer` 生成 `topk_indices`。
   - 将 q/k 拆分为 `nope` 与 `rope` 两部分，分别服务 MLA absorb sparse attention 和 RoPE。
   - 在 CP 域 all-gather `k_nope`、`v`、`k_pe`，并按 `slice_end` 裁剪。
   - 调用 `torch_npu.npu_sparse_flash_attention`，配置 `sparse_mode=3`、`attention_mode=2`、`return_softmax_lse=True`。
   - 使用返回的 `softmax_max`、`softmax_sum` 计算 `SparseLightningIndexerKLLoss`。
   - 返回 `(loss, output)`，其中 output 再转回 BNSD 供后续 post attention 使用。

5. CP all-gather 的反向梯度。
   - `allgather_sequence` 默认使用 DTensor 路径。
   - forward 中将 local tensor 标记为 `Shard(1)`，再 redistribute 到 `Replicate()`。
   - `ToLocalWithPartialGrad` 在 forward 返回普通 Tensor，避免后续 NPU 算子被 DTensor dispatcher 干扰。
   - backward 中将梯度包装为 `Partial()` DTensor，使 DTensor 自动完成等价 reduce-scatter 的梯度回传。
   - 保留 `AllgatherOnSequence` 作为非 DTensor 备选路径，其 backward 直接调用 `reduce_scatter_tensor`。

6. TP 与 DSA indexer loss 适配。
   - `apply_non_moe_tp` 为 DSA indexer、inner attention、post attention 配置专门 parallelize plan。
   - CP 开启时，`positions_sharding = Replicate()`，保证 TP 场景下 position ids 不被错误切分。
   - `SparseLightningIndexerKLLoss` 的 query、softmax max/sum 在 TP 维按需要 all-gather，保证 NPU loss kernel 看到满足要求的完整 head 数。
   - `PrepareModuleInputOutputWithBwdAllReduce` 在 MLA absorb 路径下为指定输入注册 backward all-reduce hook，保证反向梯度同步。
   - activation checkpoint 开启时，对非 MoE FFN 使用 `AwaitRowwiseParallel`，等待异步 redistribute 完成，避免 checkpoint 重算场景下异步通信残留。

7. RoPE 全局位置适配。
   - TorchTitan 0.2.2 的 CP 输入会生成 `positions`。
   - 当前仓在 `torchtitan_npu/train.py` 中 patch `reshape_for_broadcast`，支持按 `positions` 从全局 `freqs_cis` 取 RoPE。
   - DeepSeek V3.2 的 `apply_rotary_emb`、`Indexer.forward`、`PreAttention.forward`、`Attention.forward`、`TransformerBlockV32.forward` 和 `DeepSeekV32Model.forward` 均透传 `positions`。
   - 这样 CP rank 1 之后不会错误使用从 0 开始的局部 RoPE，保证 CP 与非 CP 语义一致。

8. CP 输入顺序切分。
   - DSA CP 的 `slice_end` 按 rank 顺序构造因果 key 范围。
   - 如果使用 HeadTail load balance，rank 上的 token 顺序不再等价于连续全局序列，会破坏 DSA 因果裁剪。
   - 因此 `cp_input_sharding.py` 在 DeepSeek V3.2 且开启 DSA/indexer loss 时，将 `context_parallel_load_balancer` 临时置为 `None`，强制顺序切分。

9. DSA indexer loss 统计。
   - `apply_distributed_indexer_loss_tracking` patch `DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics`。
   - 统计时先 clone 每层 indexer loss，随后做一次 global all-reduce AVG。
   - PP 场景下，由于非本 PP stage 的层 loss 为 0，需要乘以 PP degree 修正平均期望。
   - 最终再按层数和 gradient accumulation steps 归一化，记录全局一致的 indexer loss。
   - `train.py` 在训练 step 后按 metrics log 频率记录或清理 tracker，避免非日志 step 的 loss 跨 step 累积。

### 3.4 与旧仓实现的主要差异

1. CP 选择方式变化。
   - 旧仓：通过 `custom_context_parallel_path` 指定 `UlyssesContextParallelContext` 或 `AscendDSAContextParallelContext`。
   - 当前仓：通过 `apply_cp_to_attention_module` 自动分发到 `ulysses` 或 `dsa`，配置只保留 `enable_custom_context_parallel`。

2. CP patch 时机变化。
   - 旧仓：在进入自定义 CP Context 时 patch attention forward，退出时可恢复。
   - 当前仓：在模型并行化阶段 patch attention module/class，使其适配 TorchTitan 0.2.2 的模块级 CP 流程。

3. DeepSeek V3 适配方式变化。
   - 旧仓没有单独的 DeepSeek V3 parallelize wrapper。
   - 当前仓新增 `torchtitan_npu/models/deepseek_v3/infra/parallelize.py`，在复用上游 DeepSeek V3 的同时，注入 NPU MoE EP/TP 计划并把 CP 路由到 Ulysses。

4. DeepSeek V3.2 DSA CP 增强。
   - 当前仓在 DSA CP 中传入 `tp_mesh`，对 TP 切分后的 query、RoPE query、softmax max/sum 做必要 all-gather。
   - `npu_lightning_indexer` 调用使用 `return_value=False`，避免 Torch compile/DTensor meta 路径失败。
   - `actual_seq_lengths_kv` 使用当前 rank 因果可见的 `slice_end`，与裁剪后的 KV 长度保持一致。
   - 模型侧补充 `positions` 透传，解决 CP 下 RoPE 全局位置问题。

5. 配置和训练入口变化。
   - 当前仓版本为 `0.2.2.post1`，对应 TorchTitan 0.2.2。
   - DeepSeek V3.2 32K 训练配置提供 `context_parallel_degree = 8` 的目标配置。
   - DeepSeek V3/V3.2 的 4K 配置默认 CP degree 为 1，可按验收需要打开 CP degree 进行精度对比。

### 3.5 验收建议

1. DeepSeek V3 Ulysses CP。
   - 使用相同 seed、数据、模型 flavor，对比 CP=1 与 CP>1 的 loss/grad norm 曲线。
   - 验证 `n_heads`、`seq_len`、`TP * CP` 整除条件。
   - 覆盖 TP+CP、EP+CP、FSDP+CP 的目标组合。

2. DeepSeek V3.2 DSA CP。
   - 使用包含 `npu_dsa` 的配置进行 CP=1 与 CP>1 对比。
   - 重点观察主 loss、grad norm、indexer loss 是否连续、有限且趋势一致。
   - 覆盖 4K CP 回归和 32K CP 目标场景。
   - 验证 HeadTail load balance 被关闭后，position ids、causal slice、KV all-gather 语义正确。

3. 回归范围。
   - CP 关闭时，DeepSeek V3 和 DeepSeek V3.2 应回退原有路径。
   - 未配置 `npu_dsa` 时，不应误入 DSA CP。
   - 配置非法时应尽早报错，例如 DSA 缺少 `npu_dsa`、Ulysses 头数不可整除、DSA 非 MLA absorb 形态等。












# DeepSeek V3/V3.2 CP NPU 适配需求文档

  

一、需求背景

  

TorchTitan 升级到 0.2.2 版本后，框架侧的 Context Parallel（以下简称 CP）接入方式、训练输入切分方式、模型并行化入口以及 DeepSeek 系列模型的模块组织均发生了变化。当前 `torchtitan-npu` 代码仓定位为 TorchTitan 在 NPU 环境下执行训练任务时加载的 patch 仓，因此需要跟随 TorchTitan 0.2.2 的接口和执行链路进行同步适配，保证 DeepSeek V3 和 DeepSeek V3.2 两个模型在 NPU 上开启 CP 后仍然能够正常运行，并且训练精度与基线保持一致。

  

本需求主要面向 DeepSeek V3 和 DeepSeek V3.2 的 CP 训练适配。DeepSeek V3 使用标准 SDPA/MLA 注意力路径，适合采用 Ulysses CP 方案，即在注意力计算前后插入 all-to-all 通信，把原本按序列维切分的数据临时转换为按注意力头维切分的数据，使每个 CP rank 在注意力算子内部能够看到完整序列上下文。DeepSeek V3.2 使用 DeepSeek Sparse Attention（以下简称 DSA）以及 NPU sparse attention/indexer 融合算子，不能直接套用标准 SDPA CP 逻辑，需要在 CP 域内聚合 KV、Indexer KV 和 RoPE KV，并结合当前 rank 的因果可见范围进行裁剪，以保证稀疏注意力、indexer 选择和 indexer loss 的计算语义正确。

  

从 2026_03 旧仓实现来看，旧版本基于 TorchTitan 0.2.1，主要通过 `custom_context_parallel_path` 配置项指定自定义 CP Context。DeepSeek V3 通过 Ulysses CP Context 进入 Ulysses CP，DeepSeek V3.2 通过 DSA CP Context 进入 DSA CP。旧方案的核心是在创建 CP context 时替换 attention forward，并在 context 生命周期内完成输入切分和 attention patch。

  

TorchTitan 0.2.2 后，CP 的接入方式更偏向模型并行化阶段的模块级处理。当前仓因此将实现调整为在 attention CP 应用入口进行统一分发，由 DeepSeek V3 和 DeepSeek V3.2 的并行化逻辑根据模型类型、CP 开关和 converter 配置自动路由到 Ulysses 或 DSA 分支。这样配置侧只需要保留 `enable_custom_context_parallel` 作为 NPU 自定义 CP 开关，不再依赖旧的 `custom_context_parallel_path`。本次适配的本质，是把旧仓已验证可用的 Ulysses CP 和 DSA CP 能力迁移到 TorchTitan 0.2.2 的 CP 调用链、输入切分、DTensor layout 和并行化计划中，并作为 NPU patch 自动生效。

  

二、需求目标

  

本需求的总体目标是支持 DeepSeek V3 和 DeepSeek V3.2 在 NPU 设备上开启 CP 训练，并保证运行稳定、计算语义正确、精度结果与基线对齐。对于 DeepSeek V3，当 `parallelism.context_parallel_degree` 大于 1 且 `enable_custom_context_parallel` 为 true 时，模型应自动走 Ulysses CP 分支。Attention 内部通过 all-to-all 完成 head 维和 sequence 维的数据重排，使 SDPA 能够看到完整上下文；attention 外部仍保持 TorchTitan 0.2.2 原有的序列维 CP 切分方式，从而兼容 TP、EP、FSDP 和 activation checkpoint 等既有并行能力。

  

对于 DeepSeek V3.2，当模型配置中包含 `npu_dsa` converter 且 CP degree 大于 1 时，模型应自动走 DSA CP 分支。DSA attention 需要继续使用 NPU `npu_lightning_indexer`、`npu_sparse_flash_attention` 以及 `SparseLightningIndexerKLLoss` 对应的融合路径。CP 域内需要正确聚合 DSA 计算所需的 K/V、Indexer K、RoPE K，并按照当前 CP rank 的因果可见范围进行裁剪，避免未来 token 被错误访问。DSA indexer loss 还需要在 CP、TP、DP、PP 等并行维度下进行同步统计，使日志中的 indexer loss 可以作为精度归档和问题定位依据。

  

精度目标方面，DeepSeek V3 和 DeepSeek V3.2 在开启 CP 后，主 loss 曲线需要与对应基线对齐。直接基线可以采用同一代码仓、同一模型、同一数据和同一 seed 下 CP degree 为 1 的 NPU 训练结果；如果需要做版本迁移验证，也可以对照 2026_03 旧仓中相同配置下的可用结果。训练过程中不应出现由 CP 适配引入的 NaN 或 Inf，主 loss、grad norm 以及 DeepSeek V3.2 的 DSA indexer loss 均应保持有限值，并呈现合理的训练趋势。

  

语义一致性是本需求的重点。CP 切分后，RoPE 必须使用全局 token 位置，不能让每个 CP rank 都从位置 0 开始编码。DeepSeek V3.2 的 DSA 稀疏索引、KV 可见范围、softmax LSE 以及 indexer loss 必须符合全序列因果语义。TP 与 CP 同时开启时，头维切分和序列维切分不能导致 DSA loss 输入缺少必要的 head，也不能因重复聚合造成数值偏差。

  

本需求也包含必要的约束。Ulysses CP 要求 `seq_len` 可以被 CP degree 整除，`n_heads` 可以被 CP degree 整除，并且在 TP 与 CP 同开时，`n_heads` 还需要可以被 `TP degree * CP degree` 整除。DeepSeek V3.2 的 DSA CP 当前只支持 SDPA/DSA 路线，不支持 flex attention 或 varlen attention 与 CP 同开。DSA CP 依赖 `npu_dsa` converter，当 attention type 被路由为 `dsa` 时，配置中必须包含 `npu_dsa`。当前 DSA CP 按 MLA absorb 路径实现，要求 `num_head_kv == 1`。同时，DSA CP 的因果裁剪依赖 CP rank 对应连续的全局序列片段，因此 DeepSeek V3.2 + DSA 场景需要使用顺序序列切分，不能使用 HeadTail load balance。

  

三、需求实现

  

本次需求实现总体采用“保持 TorchTitan 0.2.2 主流程不变、在 NPU patch 层补充模型相关 CP 能力”的方式完成。训练数据切分、模型并行化调度、FSDP/TP/EP 等主体流程仍沿用 TorchTitan 0.2.2 的框架能力；NPU patch 负责在模型并行化阶段识别 DeepSeek V3 和 DeepSeek V3.2，并将 attention 模块自动路由到 Ulysses CP 或 DSA CP。这样可以降低对上游框架的侵入，同时保留旧仓中已经验证过的自定义 CP 能力。

在配置和分发逻辑上，当前实现保留 `enable_custom_context_parallel` 作为统一开关，不再要求用户配置 `custom_context_parallel_path`。模型开启 CP 后，由并行化流程根据模型类型和 converter 配置自动选择 CP 类型。DeepSeek V3 自动进入 Ulysses CP，DeepSeek V3.2 在启用 `npu_dsa` converter 时进入 DSA CP，其他 attention 类型仍回退 TorchTitan 原生 CP。同时，分发阶段会提前完成必要校验，例如 Ulysses CP 的序列长度和注意力头数整除关系，以及 DSA CP 对 `npu_dsa` converter 的依赖。

DeepSeek V3 的实现以复用 TorchTitan 0.2.2 原生模型主体为主，仅在并行化入口外层增加 NPU 适配逻辑。未开启 CP 或 CP degree 为 1 时，训练流程走原生路径；开启自定义 CP 后，DeepSeek V3 的 attention 会切换到 Ulysses CP。Ulysses CP 的核心是在 SDPA 前后插入 all-to-all 通信，计算前将序列维切分的数据转换为“完整序列 + 部分 attention head”的布局，计算后再恢复为序列维切分布局，从而保证 attention 看到完整上下文，并保持后续模块的 CP 输入格式不变。

DeepSeek V3.2 的实现重点是适配 DSA 场景。开启 CP 且启用 `npu_dsa` 后，attention 会进入 DSA CP 路径。该路径会在 CP 域内聚合 DSA 所需的 KV、Indexer KV 和 RoPE KV，并按照当前 CP rank 的因果可见范围进行裁剪，再调用 NPU lightning indexer、sparse flash attention 和 indexer loss 融合算子。这样可以保证稀疏索引、注意力计算和 indexer loss 都符合全序列因果语义。

为保证反向梯度和 TP/CP 组合正确，DSA CP 的 all-gather 通信采用带反向语义的分布式实现，前向向 NPU 算子提供普通 Tensor，反向保持等价 reduce-scatter 的梯度传播。TP 与 CP 同时开启时，会对 DSA indexer loss 所需的 head 相关输入做必要聚合，避免 TP 切分导致 loss kernel 输入不完整；position ids 使用复制语义，避免被 TP 错误切分。

为保证精度一致性，当前实现补充了 RoPE 全局位置适配，使 CP 场景下每个 rank 都按照真实全局 token 位置取 RoPE，而不是从局部位置 0 重新开始。DeepSeek V3.2 的 DSA CP 还强制使用顺序序列切分，避免 HeadTail load balance 打乱 token 顺序后破坏因果裁剪。DSA indexer loss 的日志统计也做了分布式同步，能够在 TP、CP、DP、PP 等并行维度下输出全局一致的指标，并在非日志 step 及时清理临时统计值。

  

四、与旧仓实现的主要差异

  

与 2026_03 旧仓相比，当前仓最明显的变化是 CP 选择方式从“配置类路径”改为“并行化阶段自动路由”。旧仓需要在 TOML 中配置 `custom_context_parallel_path`，由训练时创建的自定义 CP Context 替换 attention forward；当前仓在模型并行化阶段通过统一的 CP 分发入口决定走 Ulysses、DSA 或原生 CP，配置侧只保留 `enable_custom_context_parallel`。

  

DeepSeek V3 的适配方式也发生了变化。旧仓主要依赖自定义 Ulysses CP Context，没有单独维护 DeepSeek V3 的 parallelize wrapper。当前仓在复用 TorchTitan 0.2.2 上游 DeepSeek V3 的基础上，增加 NPU wrapper，注入 NPU MoE EP/TP 计划，并在 CP 开启时把 attention CP 路由到 Ulysses 分支。

  

DeepSeek V3.2 的 DSA CP 相比旧仓也做了增强。当前实现向 DSA CP 传入 `tp_mesh`，可以在 TP 切分后对 query、RoPE query、softmax max 和 softmax sum 做必要 all-gather。`npu_lightning_indexer` 调用改为使用 `return_value=False`，用于避免 Torch compile 或 DTensor meta 路径失败。`actual_seq_lengths_kv` 使用当前 rank 因果可见的 `slice_end`，与裁剪后的 KV 长度保持一致。模型侧新增 `positions` 透传和全局 RoPE broadcast 适配，解决 CP 下 RoPE 位置错误导致的精度问题。

  

配置和训练入口也进行了同步调整。当前仓版本为 `0.2.2.post1`，对应 TorchTitan 0.2.2 patch。DeepSeek V3.2 32K 训练配置已经提供 `context_parallel_degree = 8` 的目标场景，DeepSeek V3 和 DeepSeek V3.2 的 4K 配置默认 CP degree 为 1，可在验收时按需要打开 CP degree 做精度对比。

  

五、验收建议

  

DeepSeek V3 建议使用相同 seed、相同数据集、相同模型 flavor 和相同并行配置，对比 CP degree 为 1 与 CP degree 大于 1 时的 loss 和 grad norm 曲线。验收时需要覆盖 Ulysses CP 的整除条件，包括 `seq_len`、`n_heads` 以及 `TP degree * CP degree`，并尽量覆盖 TP+CP、EP+CP、FSDP+CP 等目标组合。

  

DeepSeek V3.2 建议使用包含 `npu_dsa` converter 的配置进行 CP=1 与 CP>1 对比，重点观察主 loss、grad norm 和 indexer loss 是否连续、有限且趋势一致。验收范围建议包括 4K CP 回归场景和 32K CP 目标场景。由于 DSA CP 依赖顺序序列切分，还需要确认 HeadTail load balance 被关闭后，positions、causal slice 和 KV all-gather 的语义符合预期。

  

回归验证时还需要关注 CP 关闭场景。DeepSeek V3 和 DeepSeek V3.2 在 CP degree 为 1 时应回退原有路径，不能因为 CP patch 引入额外行为变化。未配置 `npu_dsa` 时，DeepSeek V3.2 不应误入 DSA CP。配置非法时应尽早报错，例如 DSA 分支缺少 `npu_dsa`、Ulysses 头数不可整除、DSA 非 MLA absorb 形态等。