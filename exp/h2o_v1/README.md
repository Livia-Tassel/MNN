# H2O v1 Experiment Kit

This folder provides an experiment loop for CPU-side H2O v1 in MNN.

## Scope
- Online H2O policy in runtime (block-level keep/evict) is implemented in CPU attention path.
- This folder focuses on running `llm_bench`, collecting logs, parsing metrics, and generating summary reports.

## Required config keys
Put these keys in your model config (or let scripts generate merged configs):

```json
{
  "kv_h2o_enable": true,
  "kv_h2o_block_tokens": 64,
  "kv_h2o_sink_tokens": 32,
  "kv_h2o_recent_tokens": 256,
  "kv_h2o_target_keep_ratio": 0.5,
  "kv_h2o_ema_alpha": 0.9,
  "kv_h2o_update_interval": 16,
  "kv_h2o_trigger_min_tokens": 512,
  "kv_h2o_log_stats": false,
  "kv_lossless_enable": false,
  "kv_lossless_codec": "none"
}
```

## Quick start
`run_h2o_bench.py` will auto-set `base_dir` to the directory of `--base-config`,
so model-relative files (`llm.mnn`, `llm.mnn.weight`, `tokenizer.txt`, etc.) still resolve correctly.

1. Run a sweep:

```bash
python3 exp/h2o_v1/run_h2o_bench.py \
  --llm-bench ./build/llm_bench \
  --base-config /path/to/config.json \
  --out-dir exp/h2o_v1/out \
  --prompts 512 \
  --gens 128 \
  --repeat 3 \
  --h2o-keep-ratios 0.4,0.5,0.6 \
  --h2o-block-tokens 64,128
```

2. Parse logs:

```bash
python3 exp/h2o_v1/parse_h2o_log.py \
  --log-dir exp/h2o_v1/out/logs \
  --out-csv exp/h2o_v1/out/h2o_metrics.csv
```

3. Analyze:

```bash
python3 exp/h2o_v1/analyze_h2o.py \
  --csv exp/h2o_v1/out/h2o_metrics.csv \
  --out exp/h2o_v1/out/summary.md
```

## Preset sweep configs
- `exp/h2o_v1/configs/small.json`
- `exp/h2o_v1/configs/medium.json`
- `exp/h2o_v1/configs/large.json`

Run preset:

```bash
python3 exp/h2o_v1/sweep_h2o.py \
  --llm-bench ./build/llm_bench \
  --base-config /path/to/config.json \
  --preset exp/h2o_v1/configs/small.json \
  --out-dir exp/h2o_v1/out_small
```
