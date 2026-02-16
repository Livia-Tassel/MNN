# MNN H2O v5 维护手册（算法 + 源码映射）

本文是 `exp/h2o_v5` 阶段的实现总结，目标是让后续维护者可以快速回答三件事：

1. 三大目标当前完成到什么程度  
2. 每个算法在 MNN 源码里具体落在哪些位置  
3. 下一阶段（v6）该怎么继续推进与验收

---

## 1. 三大目标状态（结论）

### 目标 1：前两层高低位分离无损压缩，目标 `>= 1.3`
- 状态：已完成并通过门禁。
- 当前主验收口径：`offline_online_sim`（`chunked_grouped`）`lossless = 1.4010`。
- 运行时已具备真实压缩与解压统计（`decomp_us > 0`）。

### 目标 2：深层 H2O 有损压缩，目标 `>= 3.0`
- 状态：已完成并通过门禁。
- 当前稳定结果：`lossy_best = 3.1860`（来自多轮 `runtime/m3`）。

### 目标 3：`keep` 下来的块继续无损（可选）
- 状态：功能已上线并可控（`full/store`、联合 scope、全覆盖采样均可跑通）。
- 现状：已可用于工程化门禁；性能仍有优化空间，尤其 `store` 模式的解压开销。
- `v5 final` 验收已通过：`exp/h2o_v5/out_final_v5_20260215_230654`（`overall_pass=true`）。

---

## 2. 代码总览（从配置到报表）

### 配置与参数入口
- `exp/h2o_v5/run_h2o_v5_bench.py`
- `exp/h2o_v5/sweep_h2o_v5.py`
- `exp/h2o_v5/configs/*.json`

### 引擎配置读取与下发（JSON -> KVMeta）
- `transformers/llm/engine/src/llmconfig.hpp`
- `transformers/llm/engine/src/llm.cpp`（`updateH2OMetaFromConfig`）
- `source/core/OpCommonUtils.hpp`（`KVMeta` 字段定义）
- `transformers/llm/engine/src/kvmeta.hpp`（镜像定义）

### 核心算法执行（CPU Attention）
- `source/backend/cpu/CPUAttention.cpp`
- `source/backend/cpu/CPUAttention.hpp`

### 指标透出与 bench 汇总
- `transformers/llm/engine/src/llm.cpp`（`mMeta -> mContext` 复制）
- `transformers/llm/engine/include/llm/llm.hpp`（`LlmContext` 字段）
- `transformers/llm/engine/tools/llm_bench.cpp`（Markdown/JSON 字段输出）

### 离线评估 + 门禁
- `exp/h2o_v5/offline_lossless_fp16.py`
- `exp/h2o_v5/parse_h2o_v5_log.py`
- `exp/h2o_v5/analyze_h2o_v5.py`
- `exp/h2o_v5/test_v5_runtime.sh`
- `exp/h2o_v5/test_v5_m3.sh`
- `exp/h2o_v5/test_v5_final.sh`

---

## 3. 目标 2：H2O 有损压缩算法细节（深层）

核心在 `source/backend/cpu/CPUAttention.cpp`，流程可拆为 6 步。

### Step A：按 block 聚合注意力分数
- 在 softmax 后，将每个 token 的注意力质量累积到对应 block（`h2oBlockAcc`）。
- block 索引由 `globalToken / h2o_block_tokens` 得到。
- 作用：为“保留哪些 block”提供可排序分数。

### Step B：EMA 平滑
- 每个 block 分数做 EMA：`score = alpha * old + (1-alpha) * current`。
- 参数：`kv_h2o_ema_alpha`。
- 作用：抑制单步噪声，避免 eviction 抖动。

### Step C：硬保留 floor（sink + recent）
- sink 区（前 `h2o_sink_tokens`）和 recent 区（后 `h2o_recent_tokens`）强制保留。
- 这给出结构下限 `h2o_floor_keep_by_recent_sink`。

### Step D：目标 keep 计算（静态或自适应）
- `static`：使用 `kv_h2o_target_keep_ratio`
- `adaptive`：由 `target_lossy_ratio` 反推 `target_keep = 1/lossy_target`
- 再结合 floor 得到有效目标 `h2o_target_keep_effective`。

### Step E：block 量化与 Top-K 补齐
- 因为 eviction 粒度是 block，真实可达 keep 会量化（`h2o_block_quantized_keep`）。
- 在 floor 之外，按 EMA 分数从高到低补 block，直到达到量化目标。

### Step F：生成 reserve 计划并延迟应用
- 将保留 block 合并成 `(start, len)` 的 reserve 段。
- 将 `h2o_pending_remove / reserve / n_reserve / plan_ready` 写入 `KVMeta`，下一步统一应用。
- 好处：避免本步读写冲突。

### 指标输出
- `h2o_keep_ratio`, `h2o_lossy_ratio`
- `h2o_target_keep_effective`, `h2o_floor_keep_by_recent_sink`, `h2o_block_quantized_keep`
- `h2o_evict_us`, `h2o_last_evict_tokens`, `h2o_total_evict_tokens`

---

## 4. 目标 1：前层无损压缩算法细节（高低位分离）

核心路径同样在 `source/backend/cpu/CPUAttention.cpp`（runtime），离线路径在 `exp/h2o_v5/offline_lossless_fp16.py`。

### 4.1 Runtime 编码器原理（CPUAttention）

#### (1) 取 KV cold 区
- cold 区定义：`[hot_sink, token_budget - hot_recent)`。
- 热区不压缩，减少反复 encode/decode 开销。

#### (2) 从 KV cache 打平抽取
- 用 `collectKvMergedRange(...)` 将 packed KV 按逻辑顺序提取为连续字节流。
- `K` 与 `V` 分别提取。

#### (3) 预测 + 压缩
- FP16：按字拆成 `lo/hi` 两路，分别做预测后压缩（`encodeFp16GearPredictive`）。
- FP32：按 4-byte lane 分离压缩（`encodeFp32LanePredictive`）。
- 预测候选（runtime 当前有效）：`raw / delta_seq / xor_seq`。
- payload 编码候选：`RLE` 与 `ZSTD`，取更小者。

#### (4) 回退策略（防收益倒挂）
- 若压缩失败、空结果或 `compressed >= raw`，强制 fallback，按 `ratio=1.0` 记账。
- 可选严格校验：`strict_roundtrip_check` 时做 hash 验证。

#### (5) 解压与统计
- `full/probe`：入队后立即解一个 block，保证 `decomp_us` 有真实数据。
- `store`：先存压缩块并将 KV 原数据置零（`zeroKvRange`），下次访问前恢复。

### 4.2 Runtime 模式差异

#### `probe`（mode=0）
- 统计采样模式，扰动最小。
- 仅抽样代表层，bootstrap 样本更小。

#### `full`（mode=1）
- 实际在线压缩 + 即时解压统计。
- 支持 `kept_sample_layers` / `kept_sample_token_interval` 控制开销。

#### `store`（mode=2）
- 真正“压缩后存储 + 按需恢复”路径。
- 新增调参：
  - `store_disable_front`
  - `store_bootstrap_tokens`
  - `store_grouped_step_tokens`
  - `front_sample_token_interval`

### 4.3 离线无损口径（offline_lossless_fp16.py）

#### 三类入口模式
- `aggregate`：上界口径（upper-bound）
- `chunked`：在线分块模拟
- `chunked_grouped`：在线分块 + 分组模拟（当前主门禁）

#### `adaptive_v22` 核心
- 预测器候选集合逐 chunk 比较，选择最小压缩体积 predictor。
- `K`/`V`、`hi`/`lo` 各自可配置 predictor 列表。
- 输出包含 mode usage 统计，便于后续裁剪候选集合。

#### Fix2（已修复）
- `compression_failed` 的 entry 不再计入 `raw_bytes`，避免 ratio 虚高。

---

## 5. 目标 3：联合 scope（front_n + h2o_kept）与 store 细节

### 5.1 联合 scope 选择逻辑
- `kv_lossless_scope = front_n_and_h2o_kept` 时：
  - 前层（`layer < front_n`）优先用完整 token budget。
  - 深层（H2O 范围）使用 lossy 后 keep budget。

### 5.2 采样策略（降开销关键）
- 深层采样：
  - `kept_sample_layers <= 0` => 全层
  - `kept_sample_token_interval <= 1` => 每 decode step
- 前层采样：
  - `front_sample_token_interval` 控制全/店模式前层触发频率

### 5.3 队列与背压
- 统计 pending block 数，记录 `queue_peak`。
- `pending >= max_queue` 时触发 backpressure skip（按 fallback 记账）。

### 5.4 store 模式恢复
- 在 `onExecute` 前阶段尝试恢复 `rawDropped` block（`restoreLosslessBlockToKv`）。
- 恢复完成或失败后移除 block，防止长跑内存膨胀。

### 5.5 当前“保存”语义（浅层无损 vs H2O 有损）
- H2O 有损驱逐部分：不保存原始 KV 内容，只保留下一步生效的 `reserve` 段计划（不可逆）。
- 无损压缩部分：保存为进程内存中的 `losslessBlocks`（`keyBlob/valueBlob`），默认不落盘。
- 因此两者语义不同：
  - H2O 有损是“裁剪保留集”
  - 无损是“可恢复压缩块”

### 5.6 推理时“解压复用”现状
- `probe/full`：
  - 会执行真实解压并统计 `decomp_us`，用于 runtime 证据与门禁。
  - 但没有长期的“解压结果复用缓存池”；主要是即时校验/计量路径。
- `store`：
  - 在后续 step 前按需恢复压缩块到 KV cache（真正 restore 参与计算）。
  - 恢复完成后对应块会被移除，不是长驻的解压复用缓存。
- 结论：当前已有“按需恢复”，但还没有完整的“长期解压复用 + 异步解压”体系。

---

## 6. 指标链路（避免“日志有值但表格是 0”的问题复发）

### 6.1 Runtime 写入点
- `CPUAttention.cpp` 汇总写入 `mMeta->h2o_*`。
- 关键保护：仅在 `raw_bytes/comp_bytes` 有效时覆盖 lossless 指标，避免被后续默认值冲掉。

### 6.2 LLM Context 拷贝
- `llm.cpp` 在 `forwardRaw` 后把 `mMeta` 全量复制到 `mContext`。

### 6.3 bench 采样输出
- `llm_bench.cpp` 每轮 `response` 后收集 `context->h2o_*` 并输出 Markdown/JSON。

### 6.4 CSV 与门禁
- `parse_h2o_v5_log.py`：解析表格并保留稳定列顺序
- `analyze_h2o_v5.py`：计算 `lossy/lossless/decode/runtime_decomp/queue/fallback` 总门禁

---

## 7. v5 Final 验收结果（归档）

### 7.1 总体结论
- 报告：`exp/h2o_v5/out_final_v5_20260215_230654/final_report.md`
- `overall_pass: true`
- `total_cases: 4`（`runtime_cases: 3`, `m3_cases: 1`）

### 7.2 Runtime 三组用例（全部通过）
- `runtime_full_strict`
  - `decode=6.7600`, `drop=-0.024242`
  - `lossy=3.1860`, `lossless_online=1.4010`
  - `decomp_us=10390.3300`, `queue=1.0000`, `fallback=0.0000`
- `runtime_store_tuned`
  - `decode=6.4600`, `drop=0.021212`
  - `lossy=3.1860`, `lossless_online=1.4010`
  - `decomp_us=48175.0000`, `queue=0.0000`, `fallback=0.0000`
- `runtime_full_coverage`
  - `decode=6.4600`, `drop=0.021212`
  - `lossy=3.1860`, `lossless_online=1.4010`
  - `decomp_us=57551.0000`, `queue=1.0000`, `fallback=0.0000`

### 7.3 M3 稳定性（通过）
- `m3_pack`: `overall_pass=true`, `pass_runs=4/4`
- `joint_full_decode=[6.7500, 6.7900]`
- `joint_store_decode=[6.4300, 6.4300]`
- `joint_store_decomp_us_max=51013.0000`

### 7.4 结果解读
- 目标 1（前两层无损）：`1.4010 >= 1.3`，稳定通过。
- 目标 2（深层有损）：`3.1860 >= 3.0`，稳定通过。
- 目标 3（keep 再无损）：
  - `full/store/full_coverage` 均可过门禁。
  - `store/full_coverage` 的 `decomp_us` 明显高于 `full_strict`，仍是下一阶段重点优化点。

---

## 8. 已知边界与维护注意事项

### 8.1 `kv_lossless_codec_runtime` 当前仅配置透传
- 在 `llm.cpp` 会写入 `meta`，但 runtime 编码实现当前固定走 FP16/FP32 predictive 路径。
- 若要支持多 runtime codec，需要在 `CPUAttention.cpp` 增加分派。

### 8.2 `kv_lossless_async_threads` 尚未实装真正异步线程池
- 目前是同步执行 + 队列计数模型，不是独立后台压缩线程。

### 8.3 predictor 能力线上/离线不完全一致
- 离线 `adaptive_v22` 支持 `pair_delta`、`pair_delta_delta_seq`。
- runtime `parsePredictorFlags` 目前只识别 `raw/delta_seq/xor_seq`。

### 8.4 llm_bench 默认是“合成 token”压力，不是真实语义 prompt
- `llm_bench.cpp` 当前使用固定 token（如 `16`）构造输入。
- 适合稳定性与吞吐对比，不等价真实业务 prompt 分布。

### 8.5 store 模式仍是性能敏感路径
- 尤其长上下文时，恢复与内存写回成本明显高于 full。

---

## 9. FP32 与真实 Prompt 适配现状

### 9.1 FP32 支持
- Runtime：支持 `mBytes==4` 分支（`encode/decodeFp32LanePredictive`）。
- Offline：默认允许 `fp32 -> fp16` 归一化后评估；`--strict-fp16` 可强制跳过 FP32。
- 建议：v6 做“纯 FP32 runtime 与离线一致性 + 质量回归”专测。

### 9.2 真实 Prompt 测试
- 已有能力做端到端测试（引擎支持真实输入）。
- 当前 v5 主门禁偏 benchmark 形态，下一步应增加“真实对话/长文档/检索拼接”集合。

---

## 10. v6 建议目标（收官阶段）

### P0（必须）
1. 冻结 v5 默认参数并固化到 `target_core_gate_v5.json`/运行脚本默认值。  
2. 用 `test_v5_final.sh` 做一次最终验收基线归档。  
3. 补齐真实 prompt 集回归（短/中/长上下文，含多轮对话）。

### P1（强烈建议）
1. 为 `store` 模式单独设定更明确的 `decomp_us` 预算（按 p95/p99）。  
2. 打通 runtime codec 分派（让 `kv_lossless_codec_runtime` 真正生效）。  
3. 对齐 predictor 集（runtime 也支持 `pair_delta*`，或离线降配以保持一致）。

### P2（优化项）
1. 实装真正异步压缩线程池（使用 `kv_lossless_async_threads`）。  
2. 增加恢复命中缓存策略，进一步降低 store 恢复抖动。  
3. 引入质量维度评估（PPL/任务准确率）作为压缩收益边界。

---

## 11. 维护者快速排障清单

1. 日志有 `[H2O-LOSSLESS]` 但 CSV 为 0：先查 `CPUAttention.cpp` 聚合覆盖条件与 `llm.cpp` context 复制。  
2. `lossy` 突然低于 3.0：先看 `h2o_floor_keep / h2o_quantized_keep` 是否结构性限制上限。  
3. `decomp_us` 为 0：确认不是 `probe` 模式且 runtime_decomp gate 开启。  
4. `store` 退化：先调 `store_disable_front`, `store_bootstrap_tokens`, `store_grouped_step_tokens`。  
5. 离线/线上结论不一致：优先以 `offline_online_sim(chunked_grouped)` 与 runtime 同 scope 参数对齐后再比较。
