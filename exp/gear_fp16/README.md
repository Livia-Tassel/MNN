# Gear FP16 Offline Experiment

This directory hosts a self-contained FP16 compression experiment that **does not** modify `exp/distribution`. It reuses dumps from `exp/distribution/dumps` and writes all outputs under `exp/gear_fp16`.

## Goals
Evaluate FP32→FP16 offline conversion and measure compression ratios for:
- Raw zstd baseline
- Gear split (hi/lo)
- Gear split with seq-delta on hi stream

## Dependencies
- Python 3
- `numpy`
- `zstandard`

Install example:
```bash
pip install numpy zstandard
```

## Step 1: Convert FP32 dumps to FP16
```bash
python3 exp/gear_fp16/convert_fp16.py \
  --src-dump-dir exp/distribution/dumps \
  --dst-dump-dir exp/gear_fp16/dumps_fp16 \
  --stage both \
  --min-seq-len 0 \
  --max-seq-len 0
```

Optional flags:
- `--allow-fp16` to copy existing FP16 dumps as-is
- `--overwrite` to replace existing outputs

## Step 2: Analyze FP16 dumps
```bash
python3 exp/gear_fp16/analyze_fp16.py \
  --dump-dir exp/gear_fp16/dumps_fp16 \
  --out-dir exp/gear_fp16/out \
  --stage both \
  --zstd-level 3 \
  --min-seq-len 0 \
  --max-seq-len 0
```

Outputs:
- `exp/gear_fp16/out/layer_metrics.csv`
- `exp/gear_fp16/out/head_metrics.csv`
- `exp/gear_fp16/out/summary.md`

## Notes
- This experiment keeps RoPE as-is. The compression target must match real KV cache storage.
- If you need speed, add `--skip-head-metrics` when running the analyzer.
