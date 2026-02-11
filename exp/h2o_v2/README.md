# H2O v2 Experiment Kit

This folder provides the CPU-side H2O v2 experiment loop for MNN.

## Scope
- Runtime: layer-range H2O, adaptive/static keep target, block-quantized keep policy.
- Runtime stats: `h2o_keep/h2o_lossy/h2o_lossless` plus v2 fields
  (`h2o_target_keep_effective`, `h2o_floor_keep`, `h2o_quantized_keep`, `h2o_evict_us`, `h2o_codec_us`).
- Experiment scripts: sweep, parse logs, summarize best ratio/speed/Pareto.

## Key v2 Configs
```json
{
  "kv_h2o_enable": true,
  "kv_h2o_layer_start": 2,
  "kv_h2o_layer_end": -1,
  "kv_h2o_block_tokens": 32,
  "kv_h2o_sink_tokens": 16,
  "kv_h2o_recent_tokens": 64,
  "kv_h2o_target_keep_ratio": 0.30,
  "kv_h2o_target_mode": "adaptive",
  "kv_h2o_target_lossy_ratio": 3.0,
  "kv_h2o_ema_alpha": 0.9,
  "kv_h2o_update_interval": 16,
  "kv_h2o_trigger_min_tokens": 384,
  "kv_h2o_log_stats": false,
  "kv_lossless_enable": true,
  "kv_lossless_scope": "front_n",
  "kv_lossless_front_n": 2,
  "kv_lossless_codec": "gear_delta"
}
```

## Notes
- `run_h2o_v2_bench.py` auto-normalizes `base_dir` and rewrites model files to be relative to `base_dir`.
- `h2o_lossless_ratio` in v2 is a runtime codec ratio estimate for `gear_delta` over packed KV samples.
  It is intended for fast tuning feedback and should still be validated by offline/bitstream checks.

## Quick Start
1. Sweep:
```bash
python3 exp/h2o_v2/run_h2o_v2_bench.py \
  --llm-bench ./build/llm_bench \
  --base-config /path/to/config.json \
  --out-dir exp/h2o_v2/out \
  --prompts 512 \
  --gens 128 \
  --repeat 3 \
  --h2o-keep-ratios 0.28,0.30,0.33 \
  --h2o-block-tokens 32 \
  --h2o-layer-start 2 \
  --h2o-layer-end -1 \
  --h2o-target-mode adaptive \
  --h2o-target-lossy-ratios 3.0 \
  --kv-lossless-enable \
  --kv-lossless-scope front_n \
  --kv-lossless-front-n 2 \
  --kv-lossless-codec gear_delta
```

2. Parse logs:
```bash
python3 exp/h2o_v2/parse_h2o_v2_log.py \
  --log-dir exp/h2o_v2/out/logs \
  --out-csv exp/h2o_v2/out/h2o_metrics.csv
```

3. Analyze:
```bash
python3 exp/h2o_v2/analyze_h2o_v2.py \
  --csv exp/h2o_v2/out/h2o_metrics.csv \
  --out exp/h2o_v2/out/summary.md
```

## Preset Sweeps
- `exp/h2o_v2/configs/target_ratio.json`
- `exp/h2o_v2/configs/target_speed.json`

Run preset:
```bash
python3 exp/h2o_v2/sweep_h2o_v2.py \
  --llm-bench ./build/llm_bench \
  --base-config /path/to/config.json \
  --preset exp/h2o_v2/configs/target_ratio.json \
  --out-dir exp/h2o_v2/out_target_ratio
```
