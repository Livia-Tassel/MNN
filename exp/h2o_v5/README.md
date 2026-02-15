# H2O v5 Runtime + Gate Kit

v5 focuses on finishing the optional target:
- Target 1 (done): front-2 lossless (`online_sim >= 1.3`)
- Target 2 (done): deep H2O lossy (`runtime >= 3.0`)
- Target 3 (v5 focus): `front_n + h2o_kept` joint runtime lossless in a stable, gate-ready form

## v5 Default Policy
- Lossless scope: `front_n_and_h2o_kept`
- Runtime mode: `full`
- Kept sampling: `layers=1`, `token_interval=2`
- Front-layer sampling: `front_token_interval=1` (`store` auto defaults to `2`)
- Store-mode front handling: `store_disable_front=1` by default in `store` runtime script path
- Store-mode throttle: `store_bootstrap_tokens=32`, `store_grouped_step_tokens=256`
- Strict runtime gates enabled by default in runtime/M3 scripts:
  - `MAX_LOSSLESS_QUEUE_PEAK=8`
  - `MAX_LOSSLESS_FALLBACK=0`
  - `MAX_LOSSLESS_DECOMP_US=-1` (disabled unless set)

## Scripts
- `run_h2o_v5_bench.py`: generate configs + run `llm_bench`
- `sweep_h2o_v5.py`: run preset sweeps
- `parse_h2o_v5_log.py`: parse markdown logs into CSV
- `offline_lossless_fp16.py`: offline lossless eval (`aggregate` / `chunked` / `chunked_grouped`)
- `analyze_h2o_v5.py`: runtime + offline merge and gate summary
- `test_v5_runtime.sh`: one-command strict runtime validation
- `test_v5_m3.sh`: one-command core gate pack (`joint_full x3` + `joint_store x1`)
- `test_v5_final.sh`: one-command final acceptance (`full + store + full_coverage + M3`)
- `run_full_eval.sh`: offline + gate pipeline with optional strict runtime gates

## Recommended Commands

### 1) Single strict runtime validation
```bash
bash exp/h2o_v5/test_v5_runtime.sh
```

### 2) Core gate pack (v6-ready acceptance scope)
```bash
bash exp/h2o_v5/test_v5_m3.sh
```

### 3) Store-only smoke
```bash
KV_LOSSLESS_RUNTIME_MODE=store bash exp/h2o_v5/test_v5_runtime.sh
```

### 4) Optional decomp budget gate
```bash
MAX_LOSSLESS_DECOMP_US=30000 bash exp/h2o_v5/test_v5_runtime.sh
```

### 5) Target-3 full coverage run (all kept layers, every decode step)
```bash
python3 exp/h2o_v5/sweep_h2o_v5.py \
  --llm-bench ./build/llm_bench \
  --base-config /path/to/model/config.json \
  --preset exp/h2o_v5/configs/target_joint_scope_v5_full_coverage.json \
  --out-dir exp/h2o_v5/out_full_coverage_$(date +%Y%m%d_%H%M%S)
```

### 6) One-command final acceptance (recommended before closing v5)
```bash
bash exp/h2o_v5/test_v5_final.sh
```

Optional strictness knobs:
```bash
FULL_DECOMP_BUDGET_US=30000 \
STORE_DECOMP_BUDGET_US=70000 \
COVERAGE_DECOMP_BUDGET_US=70000 \
M3_DECOMP_BUDGET_US=70000 \
bash exp/h2o_v5/test_v5_final.sh
```

## Presets
- `configs/target_core_gate_v5.json`: recommended baseline for core gate
- `configs/target_joint_scope_v5.json`: balanced joint-scope full mode
- `configs/target_joint_scope_v5_aggressive.json`: denser sampling (higher overhead)
- `configs/target_joint_scope_v5_full_coverage.json`: full kept coverage (`kept_sample_layers=0`, `kept_sample_token_interval=0`)
- `configs/target_store_v5.json`: store-mode joint scope
- `configs/target_ratio_v5.json`: ratio-oriented sweep
- `configs/target_speed_v5.json`: speed-oriented sweep

## New Runtime Knobs
- `kv_lossless_kept_sample_layers`:
  - `>0`: sample first N kept layers
  - `0`: sample all kept layers (full coverage)
- `kv_lossless_kept_sample_token_interval`:
  - `>1`: sample every N decode-token steps
  - `0` or `1`: sample every decode step
- `kv_lossless_front_sample_token_interval`:
  - control front-layer sampling cadence in `full/store` modes
- `kv_lossless_store_disable_front`:
  - when true and runtime mode is `store`, front_n lossless is skipped to reduce restore overhead
- `kv_lossless_store_bootstrap_tokens`:
  - controls first-store update token span (smaller helps short decode latency)
- `kv_lossless_store_grouped_step_tokens`:
  - controls follow-up store grouped update step (larger reduces update frequency)

## Gate Definition
- `lossy_pass`: best runtime `h2o_lossy >= 3.0`
- `lossless_online_pass`: offline online-sim lossless `>= 1.3`
- `decode_pass`: decode drop ratio `<= 0.05`
- `runtime_decomp_pass`: runtime decode evidence (`decomp_us > 0`) and optional decomp budget
- `runtime_queue_peak_pass`: when enabled, `queue_peak <= target`
- `runtime_fallback_pass`: when enabled, `fallback <= target`
- `overall_pass`: all enabled gates pass

## v6 Exit Criteria (Core Gate Scope)
- `test_v5_m3.sh` completes with all runs `overall_pass=true`
- `joint_full` is stable across 3 runs
- `joint_store` smoke passes with strict queue/fallback gates
