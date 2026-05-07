# 1、C128A 中执行 SFA 的详细流程

- 场景设定：
	- 总序列长度：4096
	- cp = 2，rank 1 负责 global 位置 2048...4095
	- window_size = 128，n_boundary = 127， cmp_ratio = 128
- rank 1 拿到的输入张量：
	- query_states： `shape = [B, 2048, N, D]`  ->  local index 0...2047 对应 global 位置 2048...4095
	- kv_states:   `shape = [B, 2175, D]  -> [boundary_127  |  local_2048]
                 kv_states[0...126]  ->  global 位置 1921...2047（boundary tokens）
                 kv_states[127, 2175]  -> global 位置 2048...4095（local tokens）
## （1）sfa 核的 mask 设计逻辑

> [!NOTE] 代码源于：ops-transformer/experimental/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_kernel.h 

```cpp
// 689 - 695
            tempLoopInfo.oriMaskRight = tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size +
                                        static_cast<int32_t>(tempLoopInfo.s1EndIdx) + constInfo.oriWinRight;
            tempLoopInfo.oriMaskLeft = Max(tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size +
                                               static_cast<int32_t>(tempLoopInfo.s1EndIdx) - constInfo.oriWinLeft,
                                           0);
            if (constInfo.templateMode == CFA_TEMPLATE) {
                tempLoopInfo.cmpMaskRight = tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size;
            }
```

代码解释：
```bash

// S1 = query 的序列长度， S2 = ori_kv 的序列长度
// s1EndIdx = 当前 query 的 local index（0-based）

  oriMaskRight = (S2 - S1) + s1EndIdx + oriWinRight   // SWA 右边界
  oriMaskLeft  = (S2 - S1) + s1EndIdx - oriWinLeft    // SWA 左边界
  
  cmpS2IdLimit = ((S2 - S1) + s1EndIdx + 1) / cmpRatio  // cmp_kv 因果截止 
```

> [!IMPORTANT] 核不知道 query 的核心位置，它只有 local index s1EndIdx。但是通过 （S2 - S1）这个偏移量，可以推算出 query 在 kv 空间里的有效位置。

## （2）具体情况说明

- 当 rank 1 上的 query 没有 padding：
	- query_states:  shape = [B,  2048, N, D]  ->  直接使用
	- kv_padded = cat([zeros(B,1921,D), kv_states], dim=1)   ->  shape [B, 4096, D]
	- # S1 =  2048,  S2= 4096

- 以 query_states[0]（= real query[0]，global 位置 2048）为例：
	- s1EndIdx = 0           ← local index 就是 0
	- S2 - S1  = 4096 - 2048 = 2048   ← 这个差值承担了"偏移"的作用
	- oriMaskRight = 2048 + 0 + 0   = 2048  → kv_padded[2048]  ✓（同上）
	- oriMaskLeft  = 2048 + 0 - 127 = 1921  → kv_padded[1921]  ✓（同上）
		- 窗口 = kv_padded[1921..2048]  ← 完全一样！
	- cmpS2IdLimit = (2048 + 0 + 1) / 128 = 16  → 能看到 16 个压缩 token ✓（同上）
	- 
	- 再验证 query_states[2047]（= real query[2047]，global 位置 4095）：                                                                                                                                               
	                  


  oriMaskRight = 2048 + 0 + 0   = 2048  → kv_padded[2048]  ✓（同上）
  oriMaskLeft  = 2048 + 0 - 127 = 1921  → kv_padded[1921]  ✓（同上）
    窗口 = kv_padded[1921..2048]  ← 完全一样！

  cmpS2IdLimit = (2048 + 0 + 1) / 128 = 16  → 能看到 16 个压缩 token ✓（同上）

  再验证 query_states[2047]（= real query[2047]，global 位置 4095）：
  以 query_states[0]（= real query[0]，global 位置 2048）为例：

  s1EndIdx = 0           ← local index 就是 0
  S2 - S1  = 4096 - 2048 = 2048   ← 这个差值承担了"偏移"的作用

  oriMaskRight = 2048 + 0 + 0   = 2048  → kv_padded[2048]  ✓（同上）
  oriMaskLeft  = 2048 + 0 - 127 = 1921  → kv_padded[1921]  ✓（同上）
    窗口 = kv_padded[1921..2048]  ← 完全一样！

  cmpS2IdLimit = (2048 + 0 + 1) / 128 = 16  → 能看到 16 个压缩 token ✓（同上）

  再验证 query_states[2047]（= real query[2047]，global 位置 4095）：

  s1EndIdx = 2047
  S2 - S1  = 2048

  oriMaskRight = 2048 + 2047 + 0   = 4095  → kv_padded[4095]  ✓
  oriMaskLeft  = 2048 + 2047 - 127 = 3968  → kv_padded[3968]  ✓
    窗口 = kv_padded[3968..4095]
    = kv_states[3968-1921..4095-1921] = kv_states[2047..2174]  ✓

  cmpS2IdLimit = (2048 + 2047 + 1) / 128 = 32  → 能看到全部 32 个压缩 token ✓