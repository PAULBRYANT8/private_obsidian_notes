# 一、PyTorch context_parallel CP切分策略
## 1、核心策略：Ring Attention

基本策略：
- 序列切分：将输入序列沿 seq_dim 维度切分，每个 rank 只保留一个分片；
- KV 环形传递：在 size（CP 并行度）次迭代中，KV 在各个 rank 之间旋转传递；
- 