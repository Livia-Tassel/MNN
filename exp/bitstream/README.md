# Bitstream Lossless (Huffman + Arithmetic)

This folder implements **bit‑level lossless encoding** for FP16 KV dumps.
It is **pure lossless** and reuses existing FP16 dumps.

## Dependencies
- Python 3
- `numpy`

```bash
pip install numpy
```

## Inputs
Reuse existing FP16 dumps from `exp/gear_fp16`:
```
exp/gear_fp16/dumps_fp16/<run_id>/
```

## Huffman Encode
```bash
python3 exp/bitstream/encode_huffman.py \
  --dump-dir exp/gear_fp16/dumps_fp16/demo_20260205_102843_prompt_128_1 \
  --out-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1 \
  --stage both \
  --codebook global
```

Outputs (per dump):
- `k_hi6_<meta>.huff`, `k_lo10_<meta>.huff`
- `v_hi6_<meta>.huff`, `v_lo10_<meta>.huff`
- `bitstream_meta_<meta>.json`
Global outputs:
- `global_codebook.json`

## Huffman Decode + Verify
```bash
python3 exp/bitstream/decode_huffman.py \
  --bitstream-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1 \
  --orig-dump-dir exp/gear_fp16/dumps_fp16/demo_20260205_102843_prompt_128_1 \
  --verify
```

## Huffman Analyze Ratios
```bash
python3 exp/bitstream/analyze_huffman.py \
  --bitstream-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1 \
  --orig-dump-dir exp/gear_fp16/dumps_fp16/demo_20260205_102843_prompt_128_1 \
  --out-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1 \
  --include-meta
```

## Arithmetic Encode
```bash
python3 exp/bitstream/encode_arith.py \
  --dump-dir exp/gear_fp16/dumps_fp16/demo_20260205_102843_prompt_128_1 \
  --out-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1_arith \
  --stage both \
  --codebook global
```

Outputs (per dump):
- `k_hi6_<meta>.ac`, `k_lo10_<meta>.ac`
- `v_hi6_<meta>.ac`, `v_lo10_<meta>.ac`
- `arith_meta_<meta>.json`
Global outputs:
- `global_codebook_arith.json`

## Arithmetic Decode + Verify
```bash
python3 exp/bitstream/decode_arith.py \
  --bitstream-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1_arith \
  --orig-dump-dir exp/gear_fp16/dumps_fp16/demo_20260205_102843_prompt_128_1 \
  --verify
```

## Arithmetic Analyze Ratios
```bash
python3 exp/bitstream/analyze_arith.py \
  --bitstream-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1_arith \
  --orig-dump-dir exp/gear_fp16/dumps_fp16/demo_20260205_102843_prompt_128_1 \
  --out-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1_arith \
  --include-meta
```

## Notes
- Encoding splits FP16 into `hi6` (sign+exp) and `lo10` (mantissa).
- `global` codebook mode is recommended to keep metadata overhead low.
