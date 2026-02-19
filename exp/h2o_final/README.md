# H2O final

`h2o_final` 是收官测试套件，脚本在本目录独立维护，不依赖 `exp/h2o_v6` / `exp/h2o_tst` 的脚本调用。

## 一键运行（默认全测）

```bash
cd /home10T/ljq/MNN
bash exp/h2o_final/test_h2o_final.sh
```

默认全测矩阵：

- runtime: `baseline` / `probe` / `full` / `store`
- llm_demo: `full` / `store`（每次都会跑 baseline vs candidate）

## 仅测部分模式（参数选择）

```bash
# 只测 runtime
bash exp/h2o_final/test_h2o_final.sh --suite runtime

# 只测 llm_demo
bash exp/h2o_final/test_h2o_final.sh --suite llm_demo

# 自定义开关
bash exp/h2o_final/test_h2o_final.sh \
  --run-runtime-baseline 0 \
  --run-runtime-probe 1 \
  --run-runtime-full 1 \
  --run-runtime-store 1 \
  --run-llm-demo-full 1 \
  --run-llm-demo-store 0
```

## Prompt 目录规范

默认目录：`/home10T/ljq/MNN/exp/h2o_final/prompts`

命名必须匹配：

- `prompt_<bucket>_<index>.txt`
- 例如：`prompt_128_1.txt`、`prompt_512_2.txt`、`prompt_2048_1.txt`

每个文件内容：

- 纯文本，一个文件一个完整 prompt
- 可多行，`llm_demo` 以 whole-file 方式读取
- `bucket` 仅用于分桶统计与门禁（128/512/2048）

可选：

- `PROMPT_MANIFEST`（jsonl/txt）用于固定样本顺序与子集

## 二进制约束

仅使用：

- `./build/llm_bench`
- `./build/llm_demo`

不存在或不可执行会直接失败（不回退到其他路径）。

## 关键默认参数

- `PROMPT_BUCKET_LIST=128,512,2048`
- `MAX_PROMPTS_PER_BUCKET=2`
- `DECODE_TOKENS=512`
- `KV_LOSSLESS_ASYNC_THREADS=2`
- `KV_LOSSLESS_DECODE_CACHE_BLOCKS=64`
- runtime decode 基线模式：`DECODE_BASELINE_MODE=same_batch`

## 输出目录

每轮输出：`exp/h2o_final/out_YYYYmmdd_HHMMSS/`

- `summary.md`: 人类可读总览
- `summary.json`: 结构化总览
- `cases.jsonl`: 每个 case 一行
- `runtime_*` / `llm_demo_*`: 各自完整原始结果
- `*.console.log`: 控制台日志

## 汇总指标（重点）

- 有损压缩：`h2o_lossy_ratio`、`h2o_keep_ratio`
- 无损压缩：`h2o_lossless_ratio`
- 解压耗时：`h2o_lossless_decompress_us`
- cache 复用：`h2o_lossless_decode_cache_hit/miss`
- 吞吐：`decode_tps`（baseline vs candidate）
- 综合压缩：`runtime_total_ratio`（llm_demo）

## 常用环境变量

- 目录与采样：
  - `PROMPT_DIR`
  - `PROMPT_PATTERN`
  - `PROMPT_BUCKET_LIST`
  - `MAX_PROMPTS`
  - `MAX_PROMPTS_PER_BUCKET`
  - `DECODE_TOKENS`
- 运行开关（也可用 CLI）：
  - `RUN_RUNTIME_BASELINE`
  - `RUN_RUNTIME_PROBE`
  - `RUN_RUNTIME_FULL`
  - `RUN_RUNTIME_STORE`
  - `RUN_LLM_DEMO_FULL`
  - `RUN_LLM_DEMO_STORE`
- 门禁：
  - `DECODE_DROP_TARGET`
  - `LLM_DEMO_DECODE_DROP_TARGET`
  - `REQUIRE_DECODE_CACHE_HIT`
  - `LLM_DEMO_REQUIRE_DECODE_CACHE_HIT`
