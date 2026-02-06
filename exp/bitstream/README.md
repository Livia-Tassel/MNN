# Bitstream Huffman (Lossless)

This folder implements **bit‑level Huffman encoding** for FP16 KV dumps.
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

## Encode
```bash
python3 exp/bitstream/encode_huffman.py \
  --dump-dir exp/gear_fp16/dumps_fp16/demo_20260205_102843_prompt_128_1 \
  --out-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1 \
  --stage both
```

Outputs (per dump):
- `k_hi6_<meta>.huff`, `k_lo10_<meta>.huff`
- `v_hi6_<meta>.huff`, `v_lo10_<meta>.huff`
- `bitstream_meta_<meta>.json` (includes Huffman code lengths)

## Decode + Verify
```bash
python3 exp/bitstream/decode_huffman.py \
  --bitstream-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1 \
  --orig-dump-dir exp/gear_fp16/dumps_fp16/demo_20260205_102843_prompt_128_1 \
  --verify
```

## Analyze Ratios
```bash
python3 exp/bitstream/analyze_huffman.py \
  --bitstream-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1 \
  --orig-dump-dir exp/gear_fp16/dumps_fp16/demo_20260205_102843_prompt_128_1 \
  --out-dir exp/bitstream/out/demo_20260205_102843_prompt_128_1 \
  --include-meta
```

## Notes
- Encoding splits FP16 into `hi6` (sign+exp) and `lo10` (mantissa).
- `bitstream_meta.json` stores Huffman code lengths to enable exact decode.
