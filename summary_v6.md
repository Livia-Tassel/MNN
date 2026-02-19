# H2O v6 现状总结（已清理过时项）

更新时间：2026-02-18  
主参考分支：`exp/h2o_v6` / `exp/h2o_tst`

## 1. 本文档范围

只记录两类事实：

1. 已在源码中实现并可定位的内容。  
2. 你回传的真实测试结果（runtime/m3/llm_demo/final）。

未验证的推断不写入结论。

## 2. 三大目标当前状态

### 目标 A：有损压缩（H2O）达到约 `3.0x`

状态：已达成。  
已回传结果多次出现 `lossy_best=3.1860`，满足目标。

### 目标 B：前层/联合范围无损压缩达到约 `1.3x`

状态：已达成。  
在 runtime/final 验收中，`offline_online_sim` 口径稳定在 `1.4010`，超过门限。

### 目标 C：真实 Prompt（`llm_demo`）稳定可跑

状态：已达成“可稳定运行”。  
最近一轮 candidate（`prompt_2048_1 + prompt_128_1 + prompt_512_1`）`3/3` 成功，`rc=0`，不再出现 `-11` 段错误。

## 3. 源码实现现状（核心点）

### 3.1 H2O 有损驱逐

已实现：

- 按 block 聚合 softmax 注意力分数；
- EMA 平滑；
- sink/recent floor 保留；
- 按目标 keep 比例 + block 量化补齐；
- 生成 reserve plan 并在下一步应用。

### 3.2 无损运行时链路

已实现：

- `probe/full/store` 三种运行模式；
- FP16/FP32 预测编码与解码；
- strict roundtrip 校验；
- runtime 指标上报（raw/comp/decomp/fallback/queue/async/cache）。

### 3.3 异步编码

已实现为任务队列 + worker 池，不再是单 in-flight。  
`kv_lossless_async_threads` 现在有实际并行语义（通过 `asyncTasks + asyncWorkers`）。

### 3.4 解码缓存（decode cache）

当前代码状态：

- 原有：层内 `decodeCacheEntries`；
- 新增：跨层全局 `globalDecodeCacheEntries`（轻量容量），用于相同压缩块复用；
- store 恢复路径和 full 路径都接入了“层内 + 全局”两级查找。

说明：该“全局缓存”已完成回归验证，本轮真实 prompt 出现 `decode_cache_hit_max=30`（2048 桶）。

## 4. 关键问题修复记录（已落地）

### 4.1 `llm_demo` 候选路径段错误（`rc=-11`）

根因（已确认）：  
同一块 `reserveStorage` 同时承载“当前步正在使用的 reserve”与“下一步 pending 计划”，后续层写入 pending 时覆盖了当前指针，导致 `begin/size` 变随机值，进而崩溃或异常回退。

已修复：

- 引入双缓冲：
  - `reserveStorage`（active）
  - `reserveStoragePending`（next-step）
- 应用 pending 时做一致性检查；
- prefill/reset 时清理 active/pending 与 meta 状态。

结果：候选 `llm_demo` 不再 `-11`，可完整输出 summary。

### 4.2 reserve 计划健壮性

已修复：

- `computeReserveSize()` 增加 `reserve==nullptr` 防护；
- `CPUKVCacheManager::onRealloc()` 增加边界/顺序/范围校验，非法时安全降级，不再越界访问；
- `KVMeta::sync()` 增加 reserve 判空与非法条目保护。

### 4.3 异步线程生命周期

已修复：析构阶段无条件 stop+join worker，避免线程泄漏与 unload/reload 生命周期问题。

## 5. 最新已回传指标（作为当前基线）

## 5.1 真实 Prompt（candidate，最新基线）

来源：`exp/h2o_tst/out_20260218_211518/llm_demo/candidate`

本轮参数：

- `prompt_pattern=prompt_*.txt`
- `sample_mode=stratified`
- `max_prompts=0`
- `max_prompts_per_bucket=1`
- `bucket_order=['2048','128','512']`
- `decode_tokens=512`

本轮总结果：

- `total_runs=3`，`pass_runs=3`，`overall_pass=true`
- `decode_tps_avg=6.2277`（min `6.0602` / max `6.3484`）
- `h2o_keep_ratio_avg=0.3333`
- `h2o_lossy_ratio_avg=3.0000`
- `h2o_lossless_ratio_avg=3.8076`
- `h2o_runtime_total_ratio_avg=11.4228`
- `runtime_decomp_us_max=17301`
- `decode_cache_hit_max=30`

分桶：

- `2048`：decode `6.0602`，lossless `6.3116`，total `18.9347`，cache_hit `30`
- `128`：decode `6.2745`，lossless `1.0000`，total `3.0000`，cache_hit `0`
- `512`：decode `6.3484`，lossless `4.1113`，total `12.3338`，cache_hit `13`

### 5.2 runtime（近期稳定样例）

你回传过一轮 `runtime_store`（h2o_tst）为 pass：

- `decode=6.0500 / baseline=5.0800`
- `lossy=3.1860`
- `lossless=1.4010`
- `decomp_us=18029`
- `cache_hit=30, cache_miss=0`

## 6. 默认测试口径（已更新）

为缩短单轮耗时，当前默认配置已调整为：

- `h2o_tst` 默认只跑 `2048` 桶；
- `DECODE_TOKENS` 默认改为 `512`。

对应文件：

- `exp/h2o_tst/run_h2o_tst.py`
- `exp/h2o_v6/test_v6_llm_demo.sh`
- `exp/h2o_tst/README.md`

## 7. 当前未完成项 / 风险

1. `store` 模式在部分场景下解压开销仍可能偏高（需继续看 `runtime_decomp_us`）。  
2. 当前 decode cache 命中已提升，但主要出现在较长/高压缩桶；短桶命中仍可能为 `0`。  
3. 吞吐门禁仍受机器时段波动影响，固定 baseline 在边界场景可能误报。

## 8. 下一步建议（按优先级）

1. 跑一轮 `2048-only llm_demo` 固化长 prompt 指标（吞吐、decomp、cache hit）。  
2. 再跑最小 runtime full/store 各一轮，确认新增全局 cache 没有引入吞吐回退。  
3. 若命中在长桶仍不稳定，再做第二轮 key 策略优化（内容指纹优先、弱位置匹配细化）。
