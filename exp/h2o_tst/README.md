# H2O tst (Minimal)

目标：在现有代码上做一套更轻量、针对性更强的回归测试，并输出精简汇总。

## 一键运行

```bash
cd /home10T/ljq/MNN
bash exp/h2o_tst/test_h2o_tst.sh
```

默认会顺序执行：
1. `runtime_full`（`test_v6_runtime.sh`）
2. `runtime_store`（`test_v6_runtime.sh`，store参数）
3. `llm_demo`（`test_v6_llm_demo.sh`，默认仅 2048 分桶、每桶 1 条 prompt）
   - `llm_demo` 默认使用 full 模式（`KV_LOSSLESS_RUNTIME_MODE=full`）

## 输出目录

- `exp/h2o_tst/out_YYYYmmdd_HHMMSS/summary.md`
- `exp/h2o_tst/out_YYYYmmdd_HHMMSS/summary.json`
- `exp/h2o_tst/out_YYYYmmdd_HHMMSS/cases.jsonl`
- 各 case 子目录与控制台日志保留在同一输出目录下。

## 最近验证结果（2026-02-18）

来源：`exp/h2o_tst/out_20260218_211518/llm_demo/candidate`

本轮参数：

- `prompt_pattern=prompt_*.txt`
- `sample_mode=stratified`
- `max_prompts=0`
- `max_prompts_per_bucket=1`
- `bucket_order=['2048','128','512']`
- `decode_tokens=512`

本轮结果：

- `overall_pass=true`，`pass_runs=3/3`
- `decode_tps_avg=6.2277`（min `6.0602` / max `6.3484`）
- `h2o_keep_ratio_avg=0.3333`
- `h2o_lossy_ratio_avg=3.0000`
- `h2o_lossless_ratio_avg=3.8076`
- `h2o_runtime_total_ratio_avg=11.4228`
- `runtime_decomp_us_max=17301`
- `decode_cache_hit_max=30`

分桶：

- `2048`：decode `6.0602`，lossless `6.3116`，cache_hit `30`
- `128`：decode `6.2745`，lossless `1.0000`，cache_hit `0`
- `512`：decode `6.3484`，lossless `4.1113`，cache_hit `13`

## 常用环境变量

- 运行开关：
  - `RUN_RUNTIME_FULL=0/1`
  - `RUN_RUNTIME_STORE=0/1`
  - `RUN_LLM_DEMO=0/1`
- 共享性能参数：
  - `KV_LOSSLESS_ASYNC_THREADS`（默认 `2`）
  - `KV_LOSSLESS_DECODE_CACHE_BLOCKS`（默认 `64`）
  - `PROMPTS`（默认 `512,1024`）
  - `REPEAT`（默认 `2`）
- runtime 门限：
  - `FULL_MAX_LOSSLESS_ASYNC_WAIT_US`（默认 `20000`）
  - `STORE_MAX_LOSSLESS_ASYNC_WAIT_US`（默认 `40000`）
  - `FULL_MAX_LOSSLESS_DECOMP_US`（默认 `30000`）
  - `STORE_MAX_LOSSLESS_DECOMP_US`（默认 `70000`）
- llm_demo 采样：
  - `PROMPT_DIR`（默认 `/home10T/ljq/MNN/exp/gear_fp16/prompts`）
  - `PROMPT_BUCKET_LIST`（默认 `2048`）
  - `MAX_PROMPTS_PER_BUCKET`（默认 `1`）
  - `DECODE_TOKENS`（默认 `512`）
- llm_demo 稳定性参数：
  - `LLM_DEMO_KV_LOSSLESS_RUNTIME_MODE`（默认 `full`）
  - `LLM_DEMO_KV_LOSSLESS_MAX_QUEUE`（默认 `64`）
  - `LLM_DEMO_KV_LOSSLESS_STORE_DISABLE_FRONT`（默认 `1`）
  - `LLM_DEMO_KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS`（默认 `16`）
  - `LLM_DEMO_KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS`（默认 `384`）
  - 上述 `STORE_*` 参数仅在 `LLM_DEMO_KV_LOSSLESS_RUNTIME_MODE=store` 时生效
