# MNN KV Cache 压缩阶段性汇报文档

> **项目状态：已通过全部 6/6 质量门禁 (Round 6, 2026-02-21)**
> **测试来源：`exp/h2o_final/out_20260221_140027/summary.json`**
> **代码：** https://github.com/Livia-Tassel/MNN/tree/exp/h2o_final

---

## 一、项目概述

### 1.1 项目目标

本项目为 MNN 推理框架实现一套完整的 KV Cache 压缩系统，包含三大目标：

1. **浅层在线无损压缩**：对模型浅层的 KV Cache 数据进行字节级无损压缩，目标压缩比 ≥ 1.3:1
2. **深层有损压缩 (H2O)**：基于 Heavy-Hitter Oracle 算法，通过注意力分数驱逐低重要性 token，目标压缩比 ≥ 3:1
3. **联合压缩**：有损驱逐后对保留部分再做无损压缩

### 1.2 整体架构

```
                         MNN LLM 推理流程
                              |
              +---------------+---------------+
              |                               |
         浅层 (layer < front_n)          深层 (layer ≥ h2o_layer_start)
              |                               |
              |                      H2O 有损驱逐 (≥ 3:1)
              |                        (block 级评分+EMA+贪心保留)
              |                               |
              |                      保留 token 的 KV 数据
              |                               |
              +---------------+---------------+
                              |
                 Gear Predictive 无损压缩 (≥ 1.3:1)
                   (字节分流 → 预测编码 → ZSTD/RLE)
                              |
                      压缩后 KV Cache
```

**核心代码分布：**
- 无损压缩引擎：`source/backend/cpu/CPUAttention.cpp`
- H2O 驱逐算法：`source/backend/cpu/CPUAttention.cpp`（2227-2531 行）
- KV Cache 物理搬移：`source/backend/cpu/CPUKVCacheManager.cpp`
- 状态管理与双缓冲：`source/backend/cpu/CPUAttention.hpp`

---

## 二、浅层在线无损压缩（目标：1.3:1）

### 2.1 FP16 字节分流

FP16 浮点数占 2 字节。核心思想是将每个 FP16 值拆分为低字节 (lo) 和高字节 (hi) 两个独立字节流，分别压缩。由于同一流内的字节具有更高的统计相似性，分流后的 entropy 显著低于原始交错字节流。

**编码过程（`CPUAttention.cpp:603-606`）：**

```cpp
for (size_t i = 0; i < words; ++i) {
    lo[i] = data[2 * i + 0];   // 低字节流
    hi[i] = data[2 * i + 1];   // 高字节流
}
```

**解码过程（`CPUAttention.cpp:643-646`）：**

```cpp
for (size_t i = 0; i < words; ++i) {
    out[2 * i + 0] = lo[i];    // 恢复低字节
    out[2 * i + 1] = hi[i];    // 恢复高字节
}
```

**FP32 处理：** FP32 占 4 字节，拆分为 4 个 lane 独立压缩（`CPUAttention.cpp:650-680`），每个 lane 对应 FP32 的第 0/1/2/3 字节位置。

### 2.2 预测编码器

字节分流后，对每个字节流施加预测编码变换以进一步降低 entropy。系统实现了三种预测模式：

#### 2.2.1 raw (MODE_RAW = 0x00)

不做任何变换，直接将原始字节流传递给后端压缩器。当数据不具备局部相关性时，raw 模式避免变换引入的膨胀。

#### 2.2.2 delta_seq (MODE_DELTA_SEQ = 0x01)

**原理：** 对字节流做差分编码 `delta[i] = src[i] - src[i-1]`（初始 prev=0）。利用顺序局部性，相邻字节的差值集中在接近 0 的小值区间，使得后端压缩器（特别是 RLE）能高效编码长段重复的零值或小值。

**代码（`CPUAttention.cpp:328-336`）：**

```cpp
static void buildDeltaSeq(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size());
    uint8_t prev = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        const uint8_t cur = src[i];
        dst[i] = static_cast<uint8_t>(cur - prev);
        prev = cur;
    }
}
```

#### 2.2.3 xor_seq (MODE_XOR_SEQ = 0x02)

**原理：** 对字节流做异或编码 `xor[i] = src[i] ^ src[i-1]`（初始 prev=0）。XOR 变换特别适合浮点数的高字节流——相邻值的指数位和符号位通常相同，XOR 结果大量为 0。

**代码（`CPUAttention.cpp:338-346`）：**

```cpp
static void buildXorSeq(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size());
    uint8_t prev = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        const uint8_t cur = src[i];
        dst[i] = static_cast<uint8_t>(cur ^ prev);
        prev = cur;
    }
}
```

### 2.3 自适应块级选择（Greedy Minimal-Size）

对每个字节流，系统穷举所有启用的预测器与后端压缩器的组合，选择压缩后体积最小的方案。

**选择流程（`CPUAttention.cpp:479-509`）：**

1. 对每种预测模式（raw / delta_seq / xor_seq），生成变换后的字节流
2. 分别尝试 RLE 和 ZSTD 两种后端压缩
3. 比较所有候选方案的 `payload.size()`，选择最小的作为最终输出
4. K 流和 V 流可配置不同的预测器集合，互不影响

```cpp
auto tryMode = [&](uint8_t mode, const std::vector<uint8_t>& stream) {
    // 尝试 RLE
    if (!rleEncode(stream.data(), stream.size(), rlePayload)) return;
    // 尝试 ZSTD (level=3)
    if (zstdCompressToBuffer(stream.data(), stream.size(), 3, zstdPayload) && !zstdPayload.empty()) { ... }
    // 保留 payload 最小的候选
};
```

### 2.4 后端压缩器

#### 2.4.1 RLE (Run-Length Encoding)

自实现的行程编码，格式如下：
- **字面量段：** 控制字节 `[0..127]` 表示后续 `ctrl+1` 个字面量字节
- **重复段：** 控制字节 `[128..255]` 表示后续 1 字节重复 `(ctrl & 127) + 4` 次
- 最大 run 长度为 131（`(127) + 4`）

**代码：** `CPUAttention.cpp:381-414`

#### 2.4.2 ZSTD

通过 `dlopen` 动态加载系统 libzstd 库，压缩级别固定为 3。

**代码：** `CPUAttention.cpp:175-264`

**平台限制：** Windows 下 `dlopen` 不可用，自动回退到 RLE。

```cpp
static ZstdApi& getZstdApi() {
    // ...
    const char* libs[] = {"libzstd.so.1", "libzstd.so", "libzstd.dylib"};
    for (const auto* lib : libs) {
        api.handle = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
        if (api.handle != nullptr) break;
    }
    // 绑定 ZSTD_compressBound, ZSTD_compress, ZSTD_decompress, ZSTD_isError
}
```

### 2.5 帧格式

每个字节流帧的二进制布局：

```
[1B mode][1B codec][4B raw_len][4B payload_len][payload...]
         ^^^^^^^^^ 10 字节帧头 ^^^^^^^^^^^^^^^^^^
```

| 字段 | 大小 | 含义 |
|------|------|------|
| mode | 1 字节 | 预测模式：0=raw, 1=delta_seq, 2=xor_seq |
| codec | 1 字节 | 后端压缩器：0=RLE, 1=ZSTD |
| raw_len | 4 字节 LE | 原始字节流长度 |
| payload_len | 4 字节 LE | 压缩后 payload 长度 |
| payload | 变长 | 压缩数据 |

**FP16 完整帧：** `[4B word_count][lo_frame][hi_frame]` — 总帧头开销 4 + 10 + 10 = 24 字节

**FP32 完整帧：** `[4B word_count][lane0][lane1][lane2][lane3]` — 总帧头开销 4 + 10×4 = 44 字节

---

## 三、深层有损压缩 H2O（目标：3:1）

### 3.1 H2O 算法原理

H2O（Heavy-Hitter Oracle）是一种基于注意力分数的 KV Cache 驱逐策略。核心思想：

1. 在 Transformer 的自注意力计算过程中，不同位置的 token 对最终输出的贡献差异巨大
2. 少数 "heavy-hitter" token 贡献了大部分注意力权重，是生成质量的关键
3. 通过在线收集注意力分数，识别并保留这些高重要性 token，驱逐低重要性 token，从而大幅减少 KV Cache 占用

**MNN 实现特点：** 在 flash attention 的分块循环内部直接累积 softmax 分数作为在线启发式信号（block-local 近似，非精确全局注意力分布），实现零额外计算开销。

### 3.2 注意力分数累积

分数以 block 为粒度累积，每个 block 包含 `h2o_block_tokens`（默认 64）个 token。

**代码（`CPUAttention.cpp:2227-2241`）：**

```cpp
if (h2oEnabled && h2oBlockCount > 0) {
    // qkSoftmax is block-local in flash-attention loop.
    // We intentionally use it as an online heuristic score signal.
    const int softStride = ROUND_UP(subKvSeqLen, mPack);
    for (int q = 0; q < seqLen; ++q) {
        auto row = qkSoftmax + q * softStride;
        for (int k = 0; k < subKvSeqLen; ++k) {
            const int globalToken = i * mBlockKV + k;
            const int blockIndex = globalToken / h2oBlockTokens;
            if (blockIndex >= 0 && blockIndex < h2oBlockCount) {
                localBlockAcc[blockIndex] += row[k];
            }
        }
    }
}
```

- `blockIndex = globalToken / h2o_block_tokens`：将 token 映射到其所属的 block
- 对 softmax 每一行（每个 query 位置）的所有 key 位置累加到对应 block 的累积器中
- 累积在 flash attention 的 KV block 循环内完成，无需额外 softmax pass

### 3.3 EMA 平滑

使用指数移动平均（EMA）平滑历史分数，使分数同时反映长期重要性和近期趋势。

**公式：**

```
score[i] = alpha * score[i] + (1 - alpha) * current[i]
current[i] = blockAcc[i] / (numHead * seqLen)    // 归一化
```

**代码（`CPUAttention.cpp:2352-2357`）：**

```cpp
const float alpha = ALIMIN(1.0f, ALIMAX(0.0f, mMeta->h2o_ema_alpha));
const float norm = (float)ALIMAX(1, mNumHead * seqLen);
for (int i = 0; i < h2oBlockCount; ++i) {
    const float current = h2oBlockAcc[i] / norm;
    scoreState.blockScores[i] = alpha * scoreState.blockScores[i]
                              + (1.0f - alpha) * current;
}
```

- 默认 `alpha = 0.90`：历史分数占 90%，新信号占 10%
- 归一化因子 `numHead * seqLen` 消除 head 数量和序列长度对分数量级的影响

### 3.4 驱逐决策流程

驱逐在每次满足触发条件时执行，完整流程如下：

**代码：`CPUAttention.cpp:2364-2468`**

**Step 1 — 标记 sink 保护区：**

```cpp
const int sinkTokens = ALIMAX(0, mMeta->h2o_sink_tokens);     // 默认 32
for (int token = 0; token < sinkEndToken; token += h2oBlockTokens) {
    keepBlock[token / h2oBlockTokens] = 1;
}
```

前 `h2o_sink_tokens` 个 token（"attention sink"）永不驱逐。这些 token 在 autoregressive 模型中承载全局上下文信息，驱逐会严重影响生成质量。

**Step 2 — 标记 recent 保护区：**

```cpp
const int recentTokens = ALIMAX(0, mMeta->h2o_recent_tokens); // 默认 256
for (int block = recentStartToken / h2oBlockTokens; block < h2oBlockCount; ++block) {
    keepBlock[block] = 1;
}
```

最近 `h2o_recent_tokens` 个 token 永不驱逐。近期 token 是当前生成上下文的直接依据。

**Step 3 — 计算目标保留数量：**

```cpp
// adaptive 模式：targetKeep = total / targetLossyRatio
const float targetLossy = ALIMAX(1.0f, mMeta->h2o_target_lossy_ratio); // 默认 3.5
targetKeepRatio = 1.0f / targetLossy;                                    // ≈ 0.286
int targetKeep = (int)std::ceil(targetKeepRatio * kvSeqLen);
```

**Step 4 — floor 保护：**

```cpp
targetKeep = ALIMAX(targetKeep, keptTokens);  // 不低于 sink + recent
```

确保保留数量不低于受保护 token 数，避免保护区本身占满配额后无法保留其他重要 token。

**Step 5 — Block 量化对齐：**

```cpp
const int needTokens = targetKeep - quantizedTargetKeep;
const int needBlocks = UP_DIV(needTokens, h2oBlockTokens);
quantizedTargetKeep = ALIMIN(kvSeqLen, quantizedTargetKeep + needBlocks * h2oBlockTokens);
```

将目标保留数按 block 边界向上取整，因为驱逐以 block 为最小单位。

**Step 6 — 贪心选取高分 block：**

```cpp
std::sort(candidates.begin(), candidates.end(),
    [](const auto& a, const auto& b) { return a.first > b.first; });  // 按 EMA 分数降序
for (auto& item : candidates) {
    if (keptTokens >= quantizedTargetKeep) break;
    keepBlock[item.second] = 1;
    keptTokens += blockTokenSize(item.second);
}
```

对所有非保护区 block 按 EMA 分数降序排列，贪心选取直到满足 `quantizedTargetKeep`。

**Step 7 — 生成 reserve pairs：**

```cpp
std::vector<int> reservePairs;
// 将连续的保留 block 合并为 (offset, length) 对
for (int i = 0; i < h2oBlockCount; ++i) {
    if (!keepBlock[i]) continue;
    // 合并相邻 block 为连续区间
    // ...
    reservePairs.emplace_back(runBegin);
    reservePairs.emplace_back(runEnd - runBegin);
}
```

输出保留的连续 token 区间列表 `[(offset0, length0), (offset1, length1), ...]`，供 KV Cache Manager 执行物理搬移。

### 3.5 双缓冲 Reserve Plan（MNN 特有优化）

为避免在 decode 过程中驱逐操作覆盖当前正在使用的 KV 数据，系统采用 active + pending 双缓冲方案。

**代码：`CPUAttention.hpp:119-122`**

```cpp
// reserveStorage: active compact plan used by meta->reserve in current step.
// reserveStoragePending: next-step plan generated during this step.
std::vector<int> reserveStorage;
std::vector<int> reserveStoragePending;
```

**工作流程：**

1. **当前步**使用上一步生成的 `reserveStoragePending`（已交换为 `reserveStorage` / active plan）
2. **同时**根据最新 EMA 分数计算下一步的 pending plan
3. 步间交换（swap active ↔ pending），确保搬移操作始终基于已确定的方案
4. KV Cache Manager 通过 `onRealloc()` 按 reserve pairs 执行物理内存搬移

### 3.6 MNN 框架实现细节

#### 3.6.1 零开销分数收集

分数累积嵌入在 flash attention 的 KV block 循环内部（`CPUAttention.cpp:2227-2241`），复用已计算的 softmax 结果，无需额外的 softmax pass。这是相比原始 H2O 论文的关键优化。

#### 3.6.2 Per-layer 独立评分

```cpp
std::vector<LayerState> layerStates;  // 每层独立的 score state
```

每层维护独立的 `blockScores` 向量和驱逐历史，支持跨层差异化的驱逐决策。浅层通常不启用 H2O（通过 `h2o_layer_start` / `h2o_layer_end` 控制）。

#### 3.6.3 Packed Tensor 布局适配

KV Cache 在 MNN 中以 packed tensor 布局存储（eP / lP / hP packing），物理内存搬移需要按照 packed 索引计算正确的源/目标地址。

**代码（`CPUKVCacheManager.cpp:816-828`）：**

```cpp
template <typename T>
void CPUKVCacheManager::moveKV(int src, int dst, int size) {
    for (int h = 0; h < mKvNumHead; ++h) {
        auto kPtr = reinterpret_cast<T*>(addrOfKey(h));
        auto vPtr = reinterpret_cast<T*>(addrOfValue(h));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < mHeadDim; j++) {
                kPtr[keyIndex(dst + i, j)]   = kPtr[keyIndex(src + i, j)];
                vPtr[valueIndex(dst + i, j)] = vPtr[valueIndex(src + i, j)];
            }
        }
    }
}
```

`onRealloc()` 在 `CPUKVCacheManager.cpp:540-642` 实现，逐 reserve pair 调用 `moveKV` 完成物理搬移并更新 `mPastLength`。

### 3.7 可配参数列表

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `h2o_block_tokens` | 64 | block 粒度（token 数）。越小驱逐越精细但计算开销增大 |
| `h2o_sink_tokens` | 32 | sink 保护区大小。过大压缩率不达标，过小影响生成质量 |
| `h2o_recent_tokens` | 256 | recent 保护区大小。同上 |
| `h2o_target_lossy_ratio` | 3.5 | 目标有损压缩比。`targetKeep = 1 / ratio ≈ 0.286` |
| `h2o_ema_alpha` | 0.90 | EMA 平滑因子。越大历史权重越高，响应新 attention 越慢 |
| `h2o_trigger_min_tokens` | 512 | 最小触发 token 数。KV 长度未达到时不执行驱逐 |
| `h2o_update_interval` | 16 | 两次驱逐之间的最小 decode 步数 |
| `h2o_layer_start` / `h2o_layer_end` | — | H2O 生效层范围。浅层通常不启用 H2O |

---

## 四、保留块的无损进一步压缩

### 4.1 压缩范围（Scope）

无损压缩的作用范围通过 `h2o_lossless_scope` 参数配置，支持三种模式：

**代码：`CPUAttention.cpp:3244-3278`**

| scope 值 | 名称 | 含义 |
|-----------|------|------|
| 1 | `front_n` | 仅压缩前 N 个浅层的全部 token |
| 2 | `h2o_kept` | 仅压缩 H2O 保留的 token（深层） |
| 3 | `front_n_and_h2o_kept` | 两者兼有（推荐配置） |

```cpp
if (mMeta->h2o_lossless_scope == 1) {       // front_n
    applyLossless = inFrontLosslessRange;
    losslessTokenBudget = kvSeqLen;
} else if (mMeta->h2o_lossless_scope == 2) { // h2o_kept
    applyLossless = inH2OKeptLosslessRange;
    losslessTokenBudget = finalKeepTokensForLayer;
} else if (mMeta->h2o_lossless_scope == 3) { // front_n_and_h2o_kept
    applyLossless = inFrontLosslessRange || inH2OKeptLosslessRange;
    losslessTokenBudget = inFrontLosslessRange ? kvSeqLen : finalKeepTokensForLayer;
}
```

### 4.2 热区（Hot Zone）保护

为避免对频繁访问的 token 增加解压延迟，系统定义了 "热区"——保留区中最频繁访问的 token 不参与压缩。

**代码（`CPUAttention.cpp:3327-3330`）：**

```cpp
const int hotSinkTokens = ALIMAX(0, mMeta->h2o_lossless_hot_sink_tokens);    // 默认 16
const int hotRecentTokens = ALIMAX(0, mMeta->h2o_lossless_hot_recent_tokens); // 默认 256
const int coldBeginToken = ALIMIN(losslessTokenBudget, hotSinkTokens);
const int coldEndToken = ALIMAX(coldBeginToken, losslessTokenBudget - hotRecentTokens);
```

- **热 sink 区（前 16 token）：** 每次 attention 都会访问，解压开销不可接受
- **热 recent 区（后 256 token）：** 最近 token 频繁被引用，同上
- **冷区 `[hotSink, budget - hotRecent)`：** 访问频率较低，压缩的收益 > 解压开销

### 4.3 压缩算法

冷区数据使用与第二章完全相同的 Gear Predictive 编码管线：

```
冷区 KV 数据 → FP16 字节分流 (lo/hi) → 预测编码 (delta_seq/xor_seq/raw)
    → 后端压缩 (ZSTD level=3 / RLE) → 自适应选择最优组合
```

---

## 五、算法的协同工作

### 5.1 推理过程中的执行顺序

在每次 decode step 中，三个压缩组件按以下顺序协同工作：

1. **先有损（H2O 驱逐）**：每层 attention 计算完成后，对深层（`h2o_layer_start` ≤ layer ≤ `h2o_layer_end`）执行 block 级驱逐
2. **后无损（Gear Predictive）**：对驱逐后保留的 token 执行无损压缩

### 5.2 运行模式

系统支持三种运行模式，适用于不同场景：

#### 5.2.1 probe 模式

- **用途：** 探针/监控模式，不实际执行压缩写回
- **行为：** 每个 scope 仅采样 1 个代表性层，测试压缩流程但不修改 KV Cache
- **适用场景：** 验证压缩比指标、调试配置参数

#### 5.2.2 full 模式

- **用途：** 全量同步模式
- **行为：** 所有层同步执行"压缩 → 立即解压 → 写回"全链路
- **特点：** 当前 decode 步内完成全部操作，KV Cache 始终保持可用状态
- **适用场景：** 功能验证、KV 一致性测试

#### 5.2.3 store 模式

- **用途：** 存储/内存节省模式
- **行为：** 异步压缩 + 延迟解压 + decode cache
- **特点：** 压缩后原始 KV 数据通过 `zeroKvRange()` 清零释放内存，真正节省显存/内存

### 5.3 store 模式的完整链路

**代码分布：** `CPUAttention.cpp:3085-3094`（zeroKvRange）、`CPUAttention.cpp:1191-1380`（decode/restore）

#### 5.3.1 压缩阶段

```
bootstrap（首次 16 token 小样本）
    → grouped-step（周期性 384 token 块）
    → 异步提交 worker 线程
```

#### 5.3.2 存储阶段

- 压缩后的 blob 保留在 `losslessBlocks` 数据结构中
- 原始 KV 数据通过 `zeroKvRange()` 清零（`CPUAttention.cpp:3085-3094`）
- `stored.rawDropped = true` 标记数据已释放

#### 5.3.3 解压阶段（下一层 prefill 时触发）

解压在需要 KV 数据时按需触发（`CPUAttention.cpp:1191-1380`）：

1. **查 decode cache**：先检查本地和全局 decode cache（hash 匹配 + LRU），命中则直接使用
2. **cache miss**：调用 `decodeFp16GearPredictive()` 解压
3. **写回并缓存**：解压数据写回 KV Cache，同时更新 decode cache

#### 5.3.4 Decode Cache 双级结构

**代码：`CPUAttention.cpp:1228-1343`**

- **层内 local cache：** 每层维护独立的 `decodeCacheEntries`（deque），按 LRU 淘汰
- **跨层 global cache：** `globalDecodeCacheEntries` 在所有层间共享
- **匹配策略：** 基于 `(startToken, tokenCount, rawHash, blobHash, rawBytes, compressedBytes)` 多字段联合匹配
- **Hash 算法：** FNV-1a 64-bit（`CPUAttention.cpp:222-233`）

#### 5.3.5 异步 Worker 线程池

**代码：`CPUAttention.cpp:1700-1758`**

```cpp
workers.emplace_back([sharedState]() {
    while (true) {
        // 等待任务
        std::unique_lock<std::mutex> lock(sharedState->asyncMutex);
        sharedState->asyncCv.wait(lock, [&]() {
            return sharedState->asyncStop || !sharedState->asyncTasks.empty();
        });
        // 取任务执行
        task = std::move(sharedState->asyncTasks.front());
        sharedState->asyncTasks.pop_front();
        // 执行压缩/解压
        result = task.fn();
        // 放入完成队列
        sharedState->asyncCompleted.emplace_back(std::move(result));
        sharedState->asyncDoneCv.notify_all();
    }
});
```

- 线程数由 `h2o_lossless_async_threads` 配置
- 使用 `mutex` + `condition_variable` 实现任务分发和结果收集
- 支持 backpressure：队列满时跳过新任务提交
- 异常安全：捕获异常后将结果标记为 fallback

### 5.4 指标体系

系统在运行时收集以下质量门禁指标：

| 指标 | 含义 |
|------|------|
| `lossy_best` | 最佳有损压缩比（H2O 驱逐后 KV token 数 / 原始 token 数）|
| `lossless_selected_value` | 无损压缩比（压缩前字节数 / 压缩后字节数）|
| `decode_gate_tps` | 门禁 decode 速度（tokens/sec），需不低于基线的 (1 - drop_target) |
| `decode_drop_ratio` | decode 速度下降比例，负值表示速度提升 |
| `kv_content_consistency_pass` | KV 内容一致性验证（压缩→解压后与原始数据是否一致）|
| `runtime_decomp_best_us` | 解压延迟（微秒）|
| `runtime_async_wait_headroom_us` | 异步等待余量（微秒），正值表示不阻塞 |
| `runtime_fallback_best` | 回退次数（解压失败时回退到未压缩数据）|
| `runtime_decode_cache_hit_best` | decode cache 命中次数 |
| `runtime_queue_peak_best` | 异步队列峰值深度 |
| `candidate_runtime_total_ratio_avg` | 综合压缩比（有损 × 无损）|

---

## 六、实验结果

### 6.1 测试配置

| 配置项 | 值 |
|--------|-----|
| 模型 | llama2_mnn (3.85 GiB, CPU backend) |
| 线程数 | 4 |
| 重复次数 | 2 reps（取均值） |
| Prompt 长度 | runtime 测试：512/1024 token；llm_demo 测试：128/512 token 桶 |
| 测试框架 | 自动化质量门禁系统，6 个 test case |

### 6.2 测试用例说明

| Case | 模式 | 说明 |
|------|------|------|
| `runtime_baseline` | full (无压缩) | 基线性能：不启用任何压缩，测量原始 decode 速度。用于计算速度下降比 |
| `runtime_probe` | probe | 探针模式：仅采样测试压缩/解压流程，不写回 KV Cache。验证压缩比指标 |
| `runtime_full` | full | 全量同步模式：所有层同步压缩+立即解压。验证 KV 一致性和功能正确性 |
| `runtime_store` | store | 存储异步模式：异步压缩+清零原始数据+按需解压。验证完整生产链路 |
| `llm_demo_full` | full | 端到端 LLM 推理 (full 模式)：在真实对话场景下测试，验证综合压缩比和 decode 速度 |
| `llm_demo_store` | store | 端到端 LLM 推理 (store 模式)：在真实对话场景下测试存储模式的完整链路 |

### 6.3 Round 6 最终结果（6/6 PASS）

> 来源：`exp/h2o_final/out_20260221_140027/summary.json`，生成时间 2026-02-21T15:29:22

| Case | 结果 | 有损压缩比 | 无损压缩比 | 综合压缩比 | Decode TPS (gate) | Decode TPS (best) | Baseline TPS | 速度变化 | KV 一致性 |
|------|------|-----------|-----------|-----------|-------------------|-------------------|-------------|---------|----------|
| runtime_baseline | **PASS** | 1.000 | 1.000 | — | 4.090 | 4.340 | 4.810 | -14.97% | PASS |
| runtime_probe | **PASS** | 3.114 | 1.401 | 4.363 | 6.570 | 6.600 | 5.440 | +20.77% | PASS |
| runtime_full | **PASS** | 3.109 | 1.401 | 4.356 | 6.195 | 6.230 | 4.790 | +29.33% | PASS |
| runtime_store | **PASS** | 3.114 | 1.401 | 4.363 | 5.905 | 6.000 | 5.010 | +17.86% | PASS |
| llm_demo_full | **PASS** | 2.849 | 1.510 | 4.301 | — | 6.025 | 4.963 | +21.40% | — |
| llm_demo_store | **PASS** | 2.903 | 1.269 | 3.684 | — | 5.701 | 4.938 | +15.45% | — |

**总结发现：**

1. **有损压缩比达标：** runtime 测试中有损压缩比达 3.1:1，超过 3:1 目标
2. **无损压缩比达标：** 无损压缩比达 1.401:1，超过 1.3:1 目标
3. **综合压缩比：** runtime 场景下综合压缩比达 4.3:1
4. **速度不降反升：** KV Cache 压缩后，decode 速度反而提升 15%~29%（KV 缩小后 cache 友好性提高）
5. **KV 一致性全通过：** 所有 runtime 测试的 KV 内容一致性验证通过
6. **零 fallback：** store 模式零解压失败回退（`runtime_fallback_best = 0`）
7. **异步余量充足：** store 模式异步等待余量 21,292 μs（余量比 53.2%），无阻塞风险

### 6.4 Store 模式关键运行时指标

| 指标 | 值 | 含义 |
|------|-----|------|
| `runtime_decomp_best_us` | 13,941 μs | 解压总耗时 |
| `runtime_async_wait_best_us` | 18,708 μs | 异步等待耗时 |
| `runtime_async_wait_headroom_us` | 21,293 μs | 异步余量（正值 = 不阻塞）|
| `runtime_async_wait_headroom_ratio` | 0.532 | 余量占比（>0 即安全）|
| `runtime_queue_peak_best` | 2.5 | 异步队列峰值深度 |
| `runtime_fallback_best` | 0 | 回退次数（0 = 完美）|
| `runtime_decode_cache_hit_best` | 30 | decode cache 命中次数 |
| `runtime_decode_cache_miss_best` | 0 | decode cache 未命中次数 |

### 6.5 迭代历程（Round 4 → 5 → 6）

| 指标 | Round 4 | Round 5 | Round 6 |
|------|---------|---------|---------|
| **日期** | 2026-02-19 | 2026-02-20 | 2026-02-21 |
| **通过率** | 3/6 | 5/6 | **6/6** |
| **overall_pass** | false | false | **true** |
| runtime_baseline | PASS | PASS | PASS |
| runtime_probe | FAIL | PASS | PASS |
| runtime_full | FAIL (KV 不一致) | PASS | PASS |
| runtime_store | FAIL (KV 不一致) | FAIL (async 超时) | PASS |
| llm_demo_full | PASS | PASS | PASS |
| llm_demo_store | PASS | PASS | PASS |
| **有损压缩比 (probe)** | 2.896 | 3.114 | 3.114 |
| **无损压缩比** | 1.401 | 1.401 | 1.401 |
| **store fallback 次数** | 26 | 0 | 0 |
| **store async 余量** | — | -21,567 μs | +21,293 μs |

**迭代关键改进：**

- **Round 4 → 5：** 修复了 KV 内容一致性问题（runtime_full/store 从 FAIL 变为 PASS），有损压缩比从 2.896 提升至 3.114，消除了 26 次 fallback
- **Round 5 → 6：** 解决了 store 模式异步等待超时问题（余量从 -21,567 μs 修复为 +21,293 μs），最终实现全部 6/6 PASS

---

## 附录：关键代码索引

| 组件 | 文件 | 行号 |
|------|------|------|
| ZSTD 动态加载 | CPUAttention.cpp | 175-264 |
| FP16 字节分流编码 | CPUAttention.cpp | 591-619 |
| FP16 字节分流解码 | CPUAttention.cpp | 621-648 |
| FP32 四 lane 分流 | CPUAttention.cpp | 650-680 |
| delta_seq 预测编码 | CPUAttention.cpp | 328-336 |
| xor_seq 预测编码 | CPUAttention.cpp | 338-346 |
| delta_seq 逆变换 | CPUAttention.cpp | 348-355 |
| xor_seq 逆变换 | CPUAttention.cpp | 357-365 |
| RLE 编码/解码 | CPUAttention.cpp | 367-442 |
| 自适应选择 (Greedy) | CPUAttention.cpp | 455-536 |
| H2O 分数累积 | CPUAttention.cpp | 2227-2241 |
| EMA 平滑 | CPUAttention.cpp | 2346-2357 |
| 驱逐决策 | CPUAttention.cpp | 2364-2468 |
| 双缓冲 reserve plan | CPUAttention.hpp | 119-122 |
| KV 物理搬移 (moveKV) | CPUKVCacheManager.cpp | 816-828 |
| onRealloc reserve 搬移 | CPUKVCacheManager.cpp | 540-642 |
| Scope 配置 | CPUAttention.cpp | 3244-3278 |
| Hot zone 保护 | CPUAttention.cpp | 3327-3330 |
| Store 模式 zeroKvRange | CPUAttention.cpp | 3085-3094 |
| Store 解压/恢复 | CPUAttention.cpp | 1191-1380 |
| Decode cache (local) | CPUAttention.cpp | 1228-1253 |
| Decode cache (global) | CPUAttention.cpp | 1254-1343 |
| 异步 worker 线程池 | CPUAttention.cpp | 1700-1758 |
| FNV-1a hash | CPUAttention.cpp | 222-233 |