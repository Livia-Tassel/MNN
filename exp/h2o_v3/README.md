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

## Quality Gate Rule
- `lossy_pass`: best runtime `h2o_lossy >= 3.0`
- `lossless_online_pass`: online-sim lossless `>= 1.3`
- `decode_pass`: best decode tps drop ratio `<= 0.05` (when `--decode-baseline` is provided)
- `overall_pass = lossy_pass && lossless_online_pass && decode_pass`

`upper_bound` is still reported to measure algorithm headroom, but deployment gate uses `online_sim`.

## Notes
- Use a new `--out-dir` each run to avoid overriding old results.
- `offline_lossless_fp16.py` has overwrite protection; add `--overwrite` only when you intentionally replace outputs.
- `run_full_eval.sh` supports optional tuning args:
  - `online_chunk_seq` (default `128`)
  - `online_framing_bytes` (default `16`)
  - `adaptive_block_seq` (default `64`)
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
