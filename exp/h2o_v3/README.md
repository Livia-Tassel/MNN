# H2O v3 Experiment Kit

This folder is the end-to-end experiment loop for:
- Runtime lossy H2O tuning (`>= 3.0`)
- Offline lossless compression validation (`>= 1.3`)
- Two lossless views:
  - `upper_bound`: aggregate stream, optimistic ceiling
  - `online_sim`: chunked stream, deployment-oriented estimate

## Scripts
- `run_h2o_v3_bench.py`: generate configs + run `llm_bench`
- `sweep_h2o_v3.py`: run a preset sweep
- `parse_h2o_v3_log.py`: parse markdown logs to CSV
- `offline_lossless_fp16.py`: offline lossless ratio eval
- `analyze_h2o_v3.py`: merge runtime and offline metrics, quality gate
- `test_v3_fix.sh`: one-command fix validation (Fix 1~6 + quality gate)

## One-Command Fix Validation
```bash
bash exp/h2o_v3/test_v3_fix.sh
```

What it validates:
- Step 0: Python-side regressions (Fix 2/3/4/5).
- Step 1: `--dry-run` config generation sanity.
- Step 2: runtime benchmark with H2O + runtime lossless enabled.
- Step 3: log -> CSV parse.
- Step 4: runtime lossless fields are non-zero (`h2o_lossless_raw_mb > 0`, `h2o_lossless > 1.0`) (Fix 1).
- Step 4.5: lossy feasibility diagnosis (`theoretical_ceiling ~= 1/max(floor_keep, quantized_keep)`).
- Step 5: groupedStep uses block step (no hardcoded `tokens=192`) (Fix 6).
- Step 6: offline upper-bound + online-sim; verifies compression-failure accounting (Fix 2).
- Step 7: quality gate summary and `overall_pass=true`.

Runtime notes for this script:
- It redirects shell temp files to `${OUT}/.tmp` to avoid `/tmp` exhaustion on shared servers.
- It requires offline codec dependency:
```bash
pip install numpy zstandard
```

## Runtime Sweep
```bash
python3 exp/h2o_v3/sweep_h2o_v3.py \
  --llm-bench ./build/llm_bench \
  --base-config /path/to/model/config.json \
  --preset exp/h2o_v3/configs/target_ratio_v3.json \
  --out-dir exp/h2o_v3/out_tune_v3_$(date +%Y%m%d_%H%M%S)
```

## Parse Logs
```bash
python3 exp/h2o_v3/parse_h2o_v3_log.py \
  --log-dir exp/h2o_v3/out_tune_v3_xxx/logs \
  --out-csv exp/h2o_v3/out_tune_v3_xxx/h2o_metrics.csv
```

## Offline Lossless (Upper Bound)
```bash
python3 exp/h2o_v3/offline_lossless_fp16.py \
  --dump-dir /path/to/kv_dumps \
  --scope front_n \
  --front-n 2 \
  --stage both \
  --entry-mode aggregate \
  --aggregate-by-stage \
  --codec-mode adaptive_v22 \
  --adaptive-block-seq 64 \
  --zstd-level 3 \
  --out-json exp/h2o_v3/out_tune_v3_xxx/offline_lossless_upper.json \
  --out-md exp/h2o_v3/out_tune_v3_xxx/offline_lossless_upper.md
```

## Offline Lossless (Online Sim)
```bash
python3 exp/h2o_v3/offline_lossless_fp16.py \
  --dump-dir /path/to/kv_dumps \
  --scope front_n \
  --front-n 2 \
  --stage both \
  --entry-mode chunked \
  --online-chunk-seq 128 \
  --online-framing-bytes 16 \
  --codec-mode adaptive_v22 \
  --adaptive-block-seq 64 \
  --zstd-level 3 \
  --out-json exp/h2o_v3/out_tune_v3_xxx/offline_lossless_online.json \
  --out-md exp/h2o_v3/out_tune_v3_xxx/offline_lossless_online.md
```

## Offline Lossless (Online Sim, Request-Layer Stream)
Use when `chunked` is stuck below target. This keeps online chunk framing but
compresses along each request-layer stream instead of resetting per meta entry.
```bash
python3 exp/h2o_v3/offline_lossless_fp16.py \
  --dump-dir /path/to/kv_dumps \
  --scope front_n \
  --front-n 2 \
  --stage both \
  --entry-mode chunked_grouped \
  --online-chunk-seq 192 \
  --online-framing-bytes 8 \
  --codec-mode adaptive_v22 \
  --adaptive-block-seq 64 \
  --out-json exp/h2o_v3/out_tune_v3_xxx/offline_lossless_online_grouped.json \
  --out-md exp/h2o_v3/out_tune_v3_xxx/offline_lossless_online_grouped.md
```

## Merge + Quality Gate
```bash
python3 exp/h2o_v3/analyze_h2o_v3.py \
  --csv exp/h2o_v3/out_tune_v3_xxx/h2o_metrics.csv \
  --offline-lossless-json exp/h2o_v3/out_tune_v3_xxx/offline_lossless_upper.json \
  --offline-lossless-online-json exp/h2o_v3/out_tune_v3_xxx/offline_lossless_online.json \
  --lossy-target 3.0 \
  --lossless-target 1.3 \
  --decode-baseline 6.72 \
  --decode-drop-target 0.05 \
  --out exp/h2o_v3/out_tune_v3_xxx/summary.md
```

## One-Command Full Eval
`run_full_eval.sh` runs: parse logs -> offline upper -> offline online-sim -> quality gate.
By default it uses `chunked_grouped` as online-sim source.
```bash
bash exp/h2o_v3/run_full_eval.sh \
  exp/h2o_v3/out_tune_v3_xxx \
  /path/to/kv_dumps \
  1.3 \
  6.72 \
  0.05 \
  192 \
  8 \
  64 \
  chunked_grouped
```

To compare both online modes in one run:
```bash
bash exp/h2o_v3/run_full_eval.sh \
  exp/h2o_v3/out_tune_v3_xxx \
  /path/to/kv_dumps \
  1.3 \
  6.72 \
  0.05 \
  192 \
  8 \
  64 \
  both
```
This writes:
- `summary_chunked.md`
- `summary_grouped.md`
- `summary.md` (alias to grouped)

## Quality Gate Rule
- `lossy_pass`: best runtime `h2o_lossy >= 3.0`
- `lossless_online_pass`: online-sim lossless `>= 1.3`
- `decode_pass`: best decode tps drop ratio `<= 0.05` (when `--decode-baseline` is provided)
- `overall_pass = lossy_pass && lossless_online_pass && decode_pass`

`upper_bound` is still reported to measure algorithm headroom, but deployment gate uses `online_sim`.

## Latest Verified Result (2026-02-13)
Source run:
- Command: `bash exp/h2o_v3/test_v3_fix.sh`
- Output: `exp/h2o_v3/out_runtime_probe_fix_20260213_221312`
- Gate: `overall_pass=true`

Summary metrics:
- rows: `2`
- avg keep: `0.3236`
- avg runtime lossy: `3.0930`
- avg runtime lossless: `2.6210`
- offline upper-bound lossless: `1.4448`
- offline online-sim lossless: `1.4010` (selected for gate)
- avg selected total ratio: `4.3334`
- decode baseline: `6.60`
- best decode tps: `6.65`
- decode gate: pass (`decode_drop_ratio=-0.007576 <= 0.05`)

Best row snapshot:
- `prompt=512`, `decode=128`, `h2o_keep=0.3138`, `h2o_lossy=3.1860`
- selected lossless (online-sim): `1.4010`
- selected total ratio: `4.4637`
- decode tps: `6.65`
- log: `exp/h2o_v3/out_runtime_probe_fix_20260213_221312/logs/run_0001_p512_g128_k0.3_b32_lr3.2.log`

Interpretation:
- Current v3 pipeline (runtime lossy + runtime lossless stats + offline lossless gate) is stable and can pass all targets.
- The previous `lossy_best=1.6` case was parameter-limited (high floor keep), not a confirmed algorithm regression.

Scope boundary:
- `h2o_lossless_decomp_us` is still near zero in runtime bench, so current runtime path should be treated as lossless-stat estimation path, not true online compressed-KV storage/readback.
- Final acceptance for lossless remains `offline online-sim` (`chunked_grouped`) until true runtime decode path is implemented.

## Next Improvements
- Implement true runtime compressed KV storage + on-demand decompression path (replace estimation-only behavior).
- Implement async compression queue, decode cache, and backpressure downgrade policy.
- Expose and validate non-zero runtime decompression metrics (`h2o_lossless_decomp_us`) in gate.
- Reduce runtime codec overhead (`h2o_lossless_comp_us`) by controlling trigger cadence and evaluation token span.
- Add optional strict grouped-step assertion mode: require at least one non-bootstrap sample with `tokens=kv_lossless_block_tokens`.
- Add runtime/offline consistency checks in CI to catch aggregation or overwrite regressions early.

## Notes
- Use a new `--out-dir` each run to avoid overriding old results.
- `offline_lossless_fp16.py` has overwrite protection; add `--overwrite` only when you intentionally replace outputs.
- `run_full_eval.sh` supports optional tuning args:
  - `online_chunk_seq` (default `128`)
  - `online_framing_bytes` (default `16`)
  - `adaptive_block_seq` (default `64`)
  - `online_entry_mode` (default `chunked_grouped`, choices: `chunked`, `chunked_grouped`, `both`)
- Runtime benchmark now exposes additional lossless fields:
  - `h2o_lossless_raw_mb`
  - `h2o_lossless_comp_mb`
  - `h2o_lossless_comp_us`
  - `h2o_lossless_decomp_us`
  - `h2o_lossless_fallback`
- Dependencies:
```bash
pip install numpy zstandard
```
