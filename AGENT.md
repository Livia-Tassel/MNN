# KV Cache Compression - Work Summary

This file summarizes the current experimental progress for KV cache compression in this repo.

## Scope
- Target: Llama2/3 KV cache, W4A16, lossless target ratio >= 1.3x.
- Platform: Snapdragon 8 series (offline experiments here).
- Focus: KV dump analysis, FP32->FP16 conversion, lossless and mixed lossy strategies.

## Experiment Directories
- `exp/distribution`: KV dump extraction, layout verification (packed vs normal), RoPE inverse checks, zstd baselines.
- `exp/gear_fp16`: Offline FP32->FP16 conversion and compression analysis (zstd, gear split, gear+delta).
- `exp/n_lossless`: Mixed compression (lossless first N layers + lossy remainder).
- `exp/bitstream`: Lossless bitstream encoders (Huffman, Arithmetic, rANS, CABAC).

## Key Findings
- **KV dump type**: Real dumps were FP32 (`bytes_per_elem=4`) even when expecting FP16; FP16 is created offline.
- **RoPE inverse**: Inverse-RoPE does not improve compressibility. Observed K inverse-RoPE zstd ~1.074 vs baseline ~1.075, and mean abs diff ~0.709. Keeping native KV layout is necessary.
- **K vs V**: V is generally more compressible than K; K remains the bottleneck.
- **Pure lossless (bitstream)**: Huffman/Arithmetic/rANS/CABAC are too slow and do not reach 1.3x once metadata or codebooks are counted.
- **Mixed strategy** (lossless first N layers + light lossy): This is the only path that crosses 1.3x while keeping errors small.

## Baseline Compression (FP16 dumps)
From `exp/gear_fp16` on different corpora/run_ids:
- MMLU corpus (large): Weighted K zstd 1.080, V zstd 1.171. Gear split: K 1.262, V 1.357. Gear+delta: K 1.352, V 1.478.
- Prompt_128 run: Weighted K zstd 1.071, V zstd 1.093. Gear split: K 1.155, V 1.183. Gear+delta: K 1.148, V 1.154.
Conclusion: compression ratios are highly corpus- and prompt-length-dependent.

## Mixed Lossy + Lossless (n=2)
From `exp/n_lossless/analyze_mixed.py`:
- Lossless first 2 layers, lossy quant bits=10, gear-delta:
  - Weighted KV ratio 1.412 (K 1.457, V 1.369)
  - Errors: K MAE 0.007374, V MAE 0.001484, cosine K 0.999983, V 0.999990

## Lossy Compare (Quant / Clip / Clip+Quant)
From `exp/n_lossless/analyze_lossy_compare.py`:
- Quant only:
  - 8 bits: K 1.749, V 1.593, MAE K 0.029653, V 0.005978
  - 10 bits: K 1.480, V 1.370, MAE K 0.007374, V 0.001484
- Clip+Quant (more stable but lower ratios):
  - 8 bits: K 1.521, V 1.447, MAE K 0.072604, V 0.006410
  - 10 bits: K 1.306, V 1.258, MAE K 0.066578, V 0.004840

## Block-Norm (Partial De-Normalization)
From `exp/n_lossless/sweep_blocknorm.py`:
- Strict precision thresholds: K cosine >= 0.9999, V cosine >= 0.9999, K MAE <= 0.01, V MAE <= 0.002.
- Best passing configs (example):
  - `bs128_topk0.02_b10`: KV 1.345, K 1.439, V 1.262, MAE K 0.005050, V 0.000634, cosine K 0.999992, V 0.999998.
- 8-bit configs can reach KV ~1.53 but fail strict cosine thresholds.

## Bitstream Lossless (FP16 hi6/lo10)
From `exp/bitstream`:
- Huffman global codebook:
  - KV ratio ~1.151 without meta, ~1.123 with meta.
- Arithmetic coding: similar ratios (1.12–1.15).
- rANS with context: ~1.135 (no meta).
- CABAC:
  - Total ratio ~1.09 when codebook bytes are included (codebook ~1.35 MB).
  - Too slow in Python and not suitable for realtime.

## Current Recommendation
- **Primary path**: Mixed compression.
  - Lossless first N=2 layers (gear+delta).
  - Lossy remainder with either:
    - Quant 10 bits (best precision, KV ~1.41).
    - Block-norm topk + 10 bits (KV ~1.34 under strict precision).
- **Avoid** pure bitstream lossless methods for realtime inference.

## Repro Entry Points
- Distribution analysis: `exp/distribution/README.md`
- FP16 gear pipeline: `exp/gear_fp16/README.md`
- Mixed lossy/lossless: `exp/n_lossless/README.md`
- Bitstream encoders: `exp/bitstream/README.md`
