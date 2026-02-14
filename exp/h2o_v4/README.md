# H2O v4 Runtime Lossless (M1 Complete, M2 In Progress)

This folder is the v4 experiment kit for the M1 milestone:
- runtime lossy H2O target: `>= 3.0`
- runtime lossless path upgraded from size-estimation to real encode/decode accounting
- offline lossless deployment gate remains: `online_sim >= 1.3`

## Latest Validated Run
- Probe output: `exp/h2o_v4/out_runtime_v4_20260214_182357`
  - `quality_status: PASS`
  - `runtime_decomp_best_us: 589.6700`
  - `decode_best: 6.9700` vs baseline `6.6000`
- Full output: `exp/h2o_v4/out_runtime_v4_20260214_200411`
  - `quality_status: PASS`
  - `runtime_decomp_best_us: 6246.3300`
  - `decode_best: 6.7000` vs baseline `6.6000`

## Scripts
- `run_h2o_v4_bench.py`: generate configs + run `llm_bench`
- `sweep_h2o_v4.py`: run preset sweeps
- `parse_h2o_v4_log.py`: parse markdown logs into CSV
- `offline_lossless_fp16.py`: offline lossless eval (`upper_bound` / `online_sim`)
- `analyze_h2o_v4.py`: merge runtime + offline and produce quality gate summary
- `test_v4_runtime.sh`: one-command validation for v4 runtime path
- `run_full_eval.sh`: one-command offline+gate pipeline

## One-Command v4 Validation
```bash
bash exp/h2o_v4/test_v4_runtime.sh
```

Validation coverage:
- Python-side regression checks (empty sweep, dedup parse, gate warn, offline failure accounting).
- Runtime sweep execution + log parse.
- Runtime lossless stats check (`raw_mb > 0`, `lossless > 1.0`).
- v4 strict runtime decode check:
  - `h2o_lossless_raw_mb > 0`
  - `h2o_lossless_comp_mb > 0`
  - `h2o_lossless_comp_us > 0`
  - `h2o_lossless_decomp_us > 0`
- Grouped-step check (`tokens != 192`, expect block-step behavior).
- Offline upper/online-sim + quality gate.

## Runtime Sweep Example
```bash
python3 exp/h2o_v4/sweep_h2o_v4.py \
  --llm-bench ./build/llm_bench \
  --base-config /path/to/model/config.json \
  --preset exp/h2o_v4/configs/target_ratio_v4.json \
  --out-dir exp/h2o_v4/out_tune_v4_$(date +%Y%m%d_%H%M%S)
```
For store-mode experiments, use `--preset exp/h2o_v4/configs/target_store_v4.json`.

## Parse + Analyze
```bash
python3 exp/h2o_v4/parse_h2o_v4_log.py \
  --log-dir exp/h2o_v4/out_tune_v4_xxx/logs \
  --out-csv exp/h2o_v4/out_tune_v4_xxx/h2o_metrics.csv

python3 exp/h2o_v4/analyze_h2o_v4.py \
  --csv exp/h2o_v4/out_tune_v4_xxx/h2o_metrics.csv \
  --offline-lossless-json exp/h2o_v4/out_tune_v4_xxx/offline_lossless_upper.json \
  --offline-lossless-online-json exp/h2o_v4/out_tune_v4_xxx/offline_lossless_online_grouped.json \
  --lossy-target 3.0 \
  --lossless-target 1.3 \
  --decode-baseline 6.60 \
  --decode-drop-target 0.05 \
  --out exp/h2o_v4/out_tune_v4_xxx/summary.md
```

## One-Command Full Eval
```bash
bash exp/h2o_v4/run_full_eval.sh \
  exp/h2o_v4/out_tune_v4_xxx \
  /path/to/kv_dumps \
  1.3 \
  6.60 \
  0.05 \
  192 \
  8 \
  64 \
  chunked_grouped \
  true
```
`run_full_eval.sh` last arg `require_runtime_decomp` defaults to `true`.

## Quality Gate Rule
- `lossy_pass`: best runtime `h2o_lossy >= 3.0`
- `lossless_online_pass`: offline online-sim lossless `>= 1.3`
- `decode_pass`: best decode tps drop ratio `<= 0.05`
- `runtime_decomp_pass`: runtime `h2o_lossless_decomp_us > 0` (when `--require-runtime-decomp` is enabled)
- `overall_pass = lossy_pass && lossless_online_pass && decode_pass && runtime_decomp_pass`

## Runtime Mode Knob (New)
- Config key: `kv_lossless_runtime_mode`
- Values:
  - `probe` (default): low-overhead representative sampling for runtime stats and gate stability.
  - `full`: full-scope runtime sampling path for M2 tuning/diagnosis (higher overhead).
  - `store` (experimental): compress then drop selected raw KV slices and restore on next decode step before use.
- `test_v4_runtime.sh` and v4 presets default to `probe`.

## Scope Boundary
- M1 done:
  - Real runtime encode blob generation + runtime decode timing/accounting path.
  - Backpressure default policy (`skip new compression when queue is full`).
  - Runtime decode evidence integrated into quality gate.
- M2 in progress:
  - Configurable runtime mode (`probe` / `full`) to separate stable gate runs from deeper runtime experiments.
- Still not implemented as production storage path:
  - Replacing KV resident layout with compressed blocks for direct attention reads.
  - Therefore offline online-sim remains the deployment acceptance metric for final compression gain.

## Notes
- Use fresh `--out-dir` for each run.
- `test_v4_runtime.sh` writes temp files under `${OUT}/.tmp` to avoid `/tmp` exhaustion.
- You can switch runtime sampling mode without editing the script:
```bash
KV_LOSSLESS_RUNTIME_MODE=full bash exp/h2o_v4/test_v4_runtime.sh
```
Store mode:
```bash
KV_LOSSLESS_RUNTIME_MODE=store bash exp/h2o_v4/test_v4_runtime.sh
```
- Dependencies:
```bash
pip install numpy zstandard
```
