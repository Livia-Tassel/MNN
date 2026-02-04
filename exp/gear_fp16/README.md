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
- (Optional) `transformers` for HF tokenizer

Install example:
```bash
pip install numpy zstandard
```
Optional:
```bash
pip install transformers
```

## Step 0a (Recommended): Generate prompts from corpus
```bash
python3 exp/gear_fp16/gen_prompts.py \
  --config /path/to/config.json \
  --corpus /path/to/corpus.txt \
  --out-dir exp/gear_fp16/prompts \
  --targets 128,512,2048 \
  --per-target 3 \
  --tokenizer auto
```
Notes:
- Tokenizer order in `auto`: `MNN -> HF -> approx`
- Use `--strict` to forbid fallback
- JSONL (e.g. MMLU) is supported; add `--jsonl-format mmlu` if needed

## Step 0 (Recommended): Run llm_demo with real prompts
Prepare prompt files (real text) under a directory, then:
```bash
python3 exp/gear_fp16/run_llm_demo_batch.py \
  --llm-demo ./build/llm_demo \
  --config /path/to/config.json \
  --prompt-dir exp/gear_fp16/prompts \
  --decode-tokens 128 \
  --dump-dir exp/gear_fp16/dumps_raw \
  --run-prefix demo \
  --stage both
```
This writes dumps to `exp/gear_fp16/dumps_raw` and logs token counts to:
`exp/gear_fp16/out/llm_demo_runs.jsonl`

## Step 1: Convert FP32 dumps to FP16
```bash
python3 exp/gear_fp16/convert_fp16.py \
  --src-dump-dir exp/gear_fp16/dumps_raw \
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
  --max-seq-len 0 \
  --run-log exp/gear_fp16/out/llm_demo_runs.jsonl
```

Outputs:
- `exp/gear_fp16/out/layer_metrics.csv`
- `exp/gear_fp16/out/head_metrics.csv`
- `exp/gear_fp16/out/summary.md`
- `exp/gear_fp16/out/grouped_summary.md` (by prompt length)

## One-shot pipeline
```bash
python3 exp/gear_fp16/run_pipeline.py \
  --llm-demo ./build/llm_demo \
  --config /path/to/config.json \
  --prompt-dir exp/gear_fp16/prompts \
  --corpus /path/to/corpus.txt \
  --gen-prompts \
  --decode-tokens 128 \
  --dump-dir exp/gear_fp16/dumps_raw \
  --fp16-dump-dir exp/gear_fp16/dumps_fp16 \
  --out-dir exp/gear_fp16/out \
  --stage both \
  --zstd-level 3
```
Each run now writes results under `exp/gear_fp16/out/<run_id>/` and prompts under `exp/gear_fp16/prompts/<run_id>/`.
You can override the run id via `--run-id`.

## Step 3: Verify round-trip correctness (gear+delta)
```bash
python3 exp/gear_fp16/verify_roundtrip.py \
  --dump-dir exp/gear_fp16/dumps_fp16 \
  --stage both \
  --min-seq-len 0 \
  --max-seq-len 0
```

## Step 4: Benchmark encode/decode latency (offline)
```bash
python3 exp/gear_fp16/bench_latency.py \
  --dump-dir exp/gear_fp16/dumps_fp16 \
  --stage both \
  --zstd-level 3 \
  --mode gear-delta \
  --iters 5
```

## Notes
- This experiment keeps RoPE as-is. The compression target must match real KV cache storage.
- If you need speed, add `--skip-head-metrics` when running the analyzer.
- `summary.md` includes both average ratios and weighted ratios (by raw bytes).
