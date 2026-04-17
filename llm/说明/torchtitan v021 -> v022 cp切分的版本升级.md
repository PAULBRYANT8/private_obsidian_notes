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
