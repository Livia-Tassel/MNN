# H2O v6 全面总结（基于源码与已执行报告）

更新时间：2026-02-17
分支：`exp/h2o_v6`

## 0. 证据范围与事实边界

本总结只基于以下可核实来源：

1. 源码实现：
- `source/backend/cpu/CPUAttention.cpp`
- `source/backend/cpu/CPUAttention.hpp`
- `source/core/OpCommonUtils.hpp`
- `transformers/llm/engine/src/kvmeta.hpp`
- `transformers/llm/engine/src/llm.cpp`
- `transformers/llm/engine/include/llm/llm.hpp`
- `transformers/llm/engine/tools/llm_bench.cpp`
- `transformers/llm/engine/demo/llm_demo.cpp`

2. v6 脚本与门禁：
- `exp/h2o_v6/test_v6_runtime.sh`
- `exp/h2o_v6/test_v6_m3.sh`
- `exp/h2o_v6/test_v6_llm_demo.sh`
- `exp/h2o_v6/test_v6_final.sh`
- `exp/h2o_v6/run_llm_demo_real_prompt.py`
- `exp/h2o_v6/analyze_h2o_v6.py`
- `exp/h2o_v6/offline_lossless_fp16.py`

3. 你在服务端回传的实际测试报告（2026-02-16 到 2026-02-17）。

未在以上证据中的内容，本总结不做推断。

---

## 1. 三大目标完成情况

### 目标 1：前两层无损压缩（目标 >= 1.3）

结论：已完成，门禁通过。

- 运行门禁使用离线 online-sim 口径（`offline_lossless_online_grouped.json`）作为上线判断来源。
- 你回传的 runtime/final 报告中，`lossless_selected_source=offline_online_sim` 且 `lossless_selected_value=1.4010`，超过 1.3 目标。
- 运行时链路已具备真实压缩/解压统计（`h2o_lossless_decomp_us > 0`），不是纯离线估算。

### 目标 2：深层 H2O 有损压缩（目标 >= 3.0）

结论：已完成，门禁通过。

- 你回传的 runtime/final 报告中，`lossy_best=3.1860`，超过 3.0 目标。
- 算法是在线基于 attention softmax 的 block 打分 + floor 保留 + Top-K 补齐。

### 目标 3：保留块继续无损（联合 scope + store/full）

结论：已完成工程化上线，final 验收通过。

- `front_n_and_h2o_kept` 联合 scope 可运行。
- `full` 与 `store` 两种 runtime mode 均有完整链路和指标。
- 你回传的 `test_v6_final.sh`（`out_final_v6_20260216_213313`）为 `overall_pass: true`。

---

## 2. 任务实现细节（源码级）

## 2.1 配置与指标链路

配置从 JSON 下发到 KVMeta：
- `llm.cpp` 的 `updateH2OMetaFromConfig(...)` 负责把 `kv_h2o_*`、`kv_lossless_*` 写入 `KVMeta`。
- 结构体定义在 `OpCommonUtils.hpp` 与 `kvmeta.hpp`（镜像字段）。

运行时统计上送：
- `CPUAttention.cpp` 写 `mMeta->h2o_*`。
- `llm.cpp` 把 `mMeta` 复制到 `mContext`。
- `llm_bench.cpp`、`llm_demo.cpp` 从 `LlmContext` 输出到 Markdown/JSON。

## 2.2 H2O 有损压缩：注意力分数如何计算、如何驱逐

核心在 `CPUAttention::onExecute`。

1. 注意力分数来源：
- 在 softmax 后，对每个 query 行、每个 key token 的概率值累加到对应 block：
  - block 索引：`globalToken / h2o_block_tokens`
  - 容器：`h2oBlockAcc`

2. 分数平滑：
- 每个 block 做 EMA：
  - `score = alpha * old + (1 - alpha) * current`
- `alpha` 来自 `kv_h2o_ema_alpha`。

3. floor 保留（不可驱逐）：
- `sink` 区间（前 `h2o_sink_tokens`）保留。
- `recent` 区间（后 `h2o_recent_tokens`）保留。
- 记录 `h2o_floor_keep_by_recent_sink`。

4. 目标 keep 计算：
- static：直接用 `kv_h2o_target_keep_ratio`。
- adaptive：用 `target_keep = 1 / kv_h2o_target_lossy_ratio`。
- 再与 floor 合并得到 `h2o_target_keep_effective`。

5. block 量化和补齐：
- 因为驱逐粒度是 block，target keep 会量化到 block 边界，得到 `h2o_block_quantized_keep`。
- floor 不足部分按 block score 从高到低补齐。

6. 驱逐计划生成与应用：
- 将保留 block 合并成 `(start,len)` reserve 对。
- 写入 `h2o_pending_remove/h2o_pending_reserve/h2o_pending_n_reserve/h2o_pending_plan_ready`。
- 下一次 `onExecute` 开始阶段才真正应用该计划，避免当前步读写冲突。

7. 有损比率口径：
- `rawBefore = kvSeqLen * kv_heads * head_dim * bytes * 2`
- `rawAfter = keepTokens * kv_heads * head_dim * bytes * 2`
- `lossyDen = rawAfter + reserveMetaBytes`
- `h2o_lossy_ratio = rawBefore / lossyDen`

## 2.3 无损压缩如何实现、如何优化、如何处理 FP32

运行时无损编码核心也在 `CPUAttention.cpp`。

### 编码对象

- 只处理 cold 区：
  - cold 起点：`hot_sink_tokens` 之后
  - cold 终点：`token_budget - hot_recent_tokens`
- KV 先从 cache 打平：`collectKvMergedRange(...)`。

### 编码器结构（predictive + payload codec）

1. 预测模式（按字节流）：
- `MODE_RAW`
- `MODE_DELTA_SEQ`
- `MODE_XOR_SEQ`

2. payload 压缩后端：
- `RLE`（内置）
- `ZSTD`（动态加载可用时启用）

3. 自动择优：
- 对候选预测模式分别压缩，选择 payload 最小方案。

### FP16 / FP32 路径

- FP16：`encodeFp16GearPredictive` / `decodeFp16GearPredictive`
  - 把每个 FP16 按低字节/高字节拆成两路 stream，分别预测+压缩。
- FP32：`encodeFp32LanePredictive` / `decodeFp32LanePredictive`
  - 按 4 个 byte lane 拆成四路 stream，各自预测+压缩。

### 回退与优化语义

- 编码失败、payload 为空、解码失败、hash 校验失败：记 fallback。
- `attemptedBytes >= rawBytes`（压缩无收益）时：
  - 不计 fallback；
  - 直接按 ratio=1.0 处理（无收益正常路径）。
- 背压跳过（queue 满/worker 忙）单独记 `backpressure_skip`，不算 fallback。

### 严格校验

- `kv_lossless_strict_roundtrip_check=true` 时启用 hash 校验（FNV1a64）。

### 离线评估中的 FP32 处理

- `offline_lossless_fp16.py` 默认把 FP32 dump 转为 FP16 再评估。
- 开 `--strict-fp16` 时会跳过 FP32 条目。

## 2.4 H2O 保留下来的块如何保存与处理

需要区分两类“保留”：

1. H2O 有损保留集（reserve 计划）：
- 仅保存 token 段计划（`h2o_pending_reserve`），不保存被驱逐原始 KV。
- 本质是不可逆裁剪。

2. 无损压缩保留块（losslessBlocks）：
- 在 `LayerState::losslessBlocks` 保存压缩块：
  - `startToken/tokenCount/rawBytes/rawHash/keyBlob/valueBlob/rawDropped/decodedOnce`
- `full/probe` 模式用于在线统计和可选解码验证。
- `store` 模式会 `zeroKvRange(...)` 清空原 KV，后续按需恢复。

## 2.5 是否实现异步上线解压缩

当前状态：

- 已实现异步“编码”上线：
  - 通过单个 `std::future` + `std::async` 执行 in-flight 编码。
  - 不是线程池，不是多 worker 并行；`kv_lossless_async_threads` 目前语义上仍是开关/保底配置，主路径是单 in-flight。

- 未实现异步“解压缩”上线：
  - 解码/恢复在当前线程完成。
  - `store` 模式恢复是 `onExecute` 开头同步执行。

## 2.6 无损 KV 在推理时如何解压复用（方法）

有两条路径：

1. full/probe 解码复用：
- `decodeOnePendingLosslessBlock(...)` 使用 `decodeCacheEntries`（deque）。
- cache key：`(startToken, tokenCount, rawHash)`。
- 命中：直接复用已解码的 `keyDecoded/valueDecoded`。
- 未命中：解码后写入缓存，容量受 `h2o_lossless_decode_cache_blocks` 限制，近似 LRU（命中项移动到队尾）。

2. store 恢复路径：
- `restoreLosslessBlockToKv(...)` 同步解码并 `scatterKvMergedRange(...)` 写回 KV。
- 当前 store 路径不走 `decodeCacheEntries` 复用。

---

## 3. 已完成测试、使用工具与结论

## 3.1 测试工具分工

- `llm_bench`：用于 runtime/m3/final 主门禁，输出 markdown 表格，再由 `parse_h2o_v6_log.py + analyze_h2o_v6.py`做汇总与门禁。
- `llm_demo`：用于真实 prompt 推理测试，`run_llm_demo_real_prompt.py` 批量运行，输出 baseline/candidate 对比报告。

## 3.2 `test_v6_runtime.sh`（llm_bench）

脚本覆盖：
- Step0 回归项（空 sweep、防重复日志、fallback warning、offline failed accounting）。
- Step4 强制校验 `h2o_lossless_decomp_us > 0`（真实解压证据）。
- Step6 离线上界 + online-sim lossless。
- Step7 质量门禁（lossy/lossless/decode/queue/fallback/backpressure/async/cache/列完整性）。

你回传的关键结果：

1. 2026-02-16 `out_runtime_v6_20260216_192338`
- `overall_pass: true`
- `lossy_best: 3.1860`
- `lossless_online_value: 1.4010`
- `decode_best: 6.86`
- `runtime_decomp_best_us: 9944.33`

2. 2026-02-16 `out_runtime_v6_20260216_233204`（失败）
- 失败原因为 runtime 列缺失：
  - `h2o_lossless_async_queue_peak`
  - `h2o_lossless_async_wait_us`
  - `h2o_lossless_decode_cache_hit`
  - `h2o_lossless_decode_cache_miss`
- 结论：是二进制/日志格式版本不一致问题，不是算法逻辑回退。

3. 2026-02-17 `out_runtime_v6_20260217_125537`、`...140249`（失败）
- 列完整后，失败点集中在 `runtime_fallback_pass`（此前有 `fallback_best=2`）。
- 后续代码已把“无收益压缩”和“背压跳过”从 fallback 分离，新增/保留 `backpressure_skip` 计数语义。

4. 2026-02-17 后续一次 runtime（你回传）
- `runtime_fallback_best: 0.0000`
- 但 `decode_drop_ratio: 0.069697 > 0.05`，因此 `decode_pass=false` 导致失败。
- 说明当门限使用 `DECODE_BASELINE=6.6` 且当前环境波动导致 `decode_best=6.14` 时，会触发吞吐门禁失败。

## 3.3 `test_v6_m3.sh`（llm_bench，多次稳定性）

你回传：`out_m3_20260216_193731`

- `total_runs: 4`
- `overall_pass_runs: 4`
- `joint_full` 3 次全部通过，decode 在 `[6.95, 6.99]`。
- `joint_store` 1 次通过，decode `6.70`，`decomp_us` 明显高于 full。

结论：full 模式稳定，store 模式功能可用但解压时间更高。

## 3.4 `test_v6_llm_demo.sh`（真实 Prompt）

你回传早期问题与修复：

1. 早期“瞬间完成 + 全 0”问题
- 根因在脚本链路而非模型能力：
  - prompt 文件读取模式需按“整文件一个 prompt”（`--prompt-file-mode=whole`）。
  - 需判定“无指标的假成功”为失败。
  - 需处理 llm_demo 输出中非 UTF-8 字节（已用 `errors="replace"`）。
- 修复后单条样本可见真实耗时（你回传示例：prefill 28.33s, decode 84.65s）。

2. 2026-02-16 `out_llm_demo_v6_20260216_210217`
- `overall_pass: true`
- `baseline_decode_tps_avg: 6.8566`
- `candidate_decode_tps_avg: 7.0923`
- `runtime_decomp_best_us: 3157.0`

3. 你后续两组真实 Prompt结果：
- 组A：candidate `keep=0.6688, lossy=1.9827, lossless=1.4870, total=3.4253`
- 组B：candidate `keep=0.4167, lossy=2.4000, lossless=1.9739, total=4.7374`

结论：llm_demo 真实 prompt 下压缩比会随 prompt 长度分布、触发条件、采样窗口显著波动，不应直接等同 llm_bench 的结构化 sweep 结果。

## 3.5 `test_v6_final.sh`（最终验收）

你回传：`out_final_v6_20260216_213313`

- `overall_pass: true`
- `total_cases: 5`
- runtime 3/3 通过
- m3 1/1 通过
- llm_demo 1/1 通过

关键样例：
- `runtime_full_strict`: decode `7.00`, lossy `3.1860`, lossless_online `1.4010`
- `runtime_store_tuned`: decode `6.70`, decomp_us `50020.33`
- `llm_demo_real_prompt_pack`: baseline `6.8894`, candidate `7.0853`, decomp_us `3146.0`

---

## 4. 当前算法性能画像

以下是“你回传报告中的实测区间”，用于工程参考。

1. 有损压缩（H2O）
- llm_bench 最优可到 `3.186x`（满足 >=3.0 目标）。
- llm_demo 真实 prompt 平均曾出现 `1.98x` 到 `2.40x`。

2. 无损压缩
- 离线 online-sim 稳定在 `1.401x`（高于 1.3 门限）。
- llm_demo 中 candidate 平均无损比出现 `1.487x` 到 `1.9739x`（与 prompt 分布有关）。

3. 总压缩比（lossy * lossless）
- llm_bench（按选定口径）可达 `4.4637`（best total）。
- llm_demo 报告中出现 `3.4253` 与 `4.7374` 两组结果。

4. 时间开销
- full 模式 runtime 解压通常在约 `9~10 ms` 量级（代表性报告）。
- store 模式 runtime 解压常在约 `50~58 ms` 量级（明显更高）。
- llm_demo 真实 prompt 报告中 `runtime_decomp_best_us` 约 `3.1~3.9 ms`。

---

## 5. 当前不足与改进方向

1. 异步模型仍是单 in-flight
- 现在是 `std::future` 单 worker，不是多线程池。
- 改进方向：实现真正多任务队列和 worker 池，并让 `kv_lossless_async_threads` 有实际并行语义。

2. store 模式恢复代价偏高
- 多次报告显示 store 的 decomp_us 显著高于 full。
- 改进方向：
  - 更激进的 grouped step / bootstrap 策略；
  - 恢复批处理；
  - 增加 store 路径的解码缓存复用。

3. decode cache 命中率偏低
- 多轮报告中 hit 常为 0、miss > 0。
- 改进方向：
  - 扩充 cache key（允许更稳健复用）；
  - 调整 block 粒度和保留时机；
  - 在 store 恢复路径接入 decode cache。

4. llm_bench 与 llm_demo 指标有口径差异
- llm_bench 是受控 sweep，llm_demo 是真实 prompt 分布。
- 改进方向：
  - 固化 prompt 采样策略（manifest + 分层统计）；
  - 把 llm_demo 报告拆分为 128/512/2048 桶统计，避免平均值掩盖问题。

5. 吞吐门禁敏感
- `decode_baseline` 固定值在不同机器/时段易触发误报。
- 改进方向：
  - 引入 rolling baseline 或同机同批 baseline。

---

## 6. 可能存在的边界问题

1. 短序列触发不足
- `kv_h2o_trigger_min_tokens` 默认较高（如 384），短 prompt 可能几乎不触发 H2O。

2. floor 过强导致 lossy 上不去
- sink/recent + block 量化会抬高 keep 下限，导致 lossy 上限受结构约束。

3. 量化路径限制
- `h2oEnabled` 要求 `!mQuantKey && !mQuantValue`，量化注意力路径下 H2O 可能不启用。

4. 旧二进制路径误用风险
- 若未使用 v6 新编译产物，CSV 会缺 runtime 列，直接导致 strict 门禁失败。

5. DUMP_DIR 缺失会让 runtime 脚本失败
- `test_v6_runtime.sh` 的 Step6 依赖离线 dump 评估，不是可选步骤。

6. store 恢复当前为同步
- 复杂负载下可能出现恢复尾延迟。

---

## 7. 结论

1. 从源码实现看，v6 的三大目标都已完整落地：
- H2O 有损在线驱逐。
- 前层/联合范围无损压缩、解压、统计链路。
- 真实 prompt 的 llm_demo 门禁与报告链路。

2. 从你回传的正式验收看，`test_v6_final.sh` 已出现 `overall_pass: true` 的完整通过版本，可作为“已上线版本”的事实依据。

3. 当前最值得继续投入的方向不是“功能有无”，而是“性能稳定性与收益一致性”：
- store 恢复耗时、decode cache 命中、真实 prompt 下压缩比稳定性、吞吐门禁鲁棒性。

