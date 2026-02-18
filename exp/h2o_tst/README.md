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
3. `llm_demo`（`test_v6_llm_demo.sh`，默认 128/512 分桶、每桶 1 条 prompt）
   - `llm_demo` 默认使用更稳的 store 运行时参数（可用环境变量覆盖）

## 输出目录

- `exp/h2o_tst/out_YYYYmmdd_HHMMSS/summary.md`
- `exp/h2o_tst/out_YYYYmmdd_HHMMSS/summary.json`
- `exp/h2o_tst/out_YYYYmmdd_HHMMSS/cases.jsonl`
- 各 case 子目录与控制台日志保留在同一输出目录下。

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
  - `PROMPT_BUCKET_LIST`（默认 `128,512`）
  - `MAX_PROMPTS_PER_BUCKET`（默认 `1`）
- llm_demo 稳定性参数：
  - `LLM_DEMO_KV_LOSSLESS_RUNTIME_MODE`（默认 `store`）
  - `LLM_DEMO_KV_LOSSLESS_MAX_QUEUE`（默认 `64`）
  - `LLM_DEMO_KV_LOSSLESS_STORE_DISABLE_FRONT`（默认 `1`）
  - `LLM_DEMO_KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS`（默认 `16`）
  - `LLM_DEMO_KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS`（默认 `384`）
