#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterator, List, Tuple

import numpy as np

from cabac_utils import AdaptiveBinaryModel, BinaryArithmeticEncoder
from context_utils import ctx_id


def iter_meta_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("meta_") and name.endswith(".json"):
                yield os.path.join(dirpath, name)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def split_fp16(u16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hi6 = (u16 >> 10) & 0x3F
    lo10 = u16 & 0x3FF
    return hi6.astype(np.uint16), lo10.astype(np.uint16)


def get_group(key_is_k: bool, stage: str) -> str:
    return f"{'k' if key_is_k else 'v'}_{stage}"


def iter_bits(value: int, bits: int) -> Iterator[Tuple[int, int]]:
    # yields (bit_idx, bit) from MSB to LSB
    for bit_idx in range(bits - 1, -1, -1):
        bit = (value >> bit_idx) & 1
        yield bit_idx, bit


def main() -> int:
    parser = argparse.ArgumentParser(description=\"CABAC-like binary arithmetic coding with context (lossless).\" )
    parser.add_argument(\"--dump-dir\", required=True)
    parser.add_argument(\"--out-dir\", default=\"exp/bitstream/out\")
    parser.add_argument(\"--stage\", choices=[\"prefill\", \"decode\", \"both\"], default=\"both\")
    parser.add_argument(\"--min-seq-len\", type=int, default=0)
    parser.add_argument(\"--max-seq-len\", type=int, default=0)
    parser.add_argument(\"--hi6-buckets\", type=int, default=512)
    parser.add_argument(\"--lo10-buckets\", type=int, default=16384)
    parser.add_argument(\"--hash-a\", type=int, default=131)
    parser.add_argument(\"--hash-b\", type=int, default=17)
    parser.add_argument(\"--hash-c\", type=int, default=257)
    parser.add_argument(\"--hash-d\", type=int, default=31)
    parser.add_argument(\"--prefix-bits\", type=int, default=3)
    parser.add_argument(\"--rescale-threshold\", type=int, default=1 << 15)
    parser.add_argument(\"--overwrite\", action=\"store_true\")
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.dump_dir))
    if not meta_files:
        print(f\"No meta_*.json found under {args.dump_dir}\", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    prefix_mask = (1 << args.prefix_bits) - 1 if args.prefix_bits > 0 else 0

    # Build prior counts per group using one full pass
    group_hi6_counts = {}
    group_lo10_counts = {}
    for key_is_k in (True, False):
        for stage in (\"prefill\", \"decode\"):
            group = get_group(key_is_k, stage)
            group_hi6_counts[group] = ([1] * args.hi6_buckets, [1] * args.hi6_buckets)
            group_lo10_counts[group] = ([1] * args.lo10_buckets, [1] * args.lo10_buckets)

    for meta_path in meta_files:
        try:
            meta = load_json(meta_path)
        except Exception:
            continue
        stage = meta.get(\"stage\", \"unknown\")
        if args.stage != \"both\" and stage != args.stage:
            continue
        if stage not in (\"prefill\", \"decode\"):
            continue
        seq_len = int(meta.get(\"seq_len\", 0))
        if args.min_seq_len and seq_len < args.min_seq_len:
            continue
        if args.max_seq_len and seq_len > args.max_seq_len:
            continue
        if int(meta.get(\"bytes_per_elem\", 0)) != 2:
            continue

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta.get(\"k_file\", \"\"))
        v_path = os.path.join(dirpath, meta.get(\"v_file\", \"\"))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            continue

        k = np.frombuffer(open(k_path, \"rb\").read(), dtype=np.uint16)
        v = np.frombuffer(open(v_path, \"rb\").read(), dtype=np.uint16)
        k_hi6, k_lo10 = split_fp16(k)
        v_hi6, v_lo10 = split_fp16(v)

        for is_k, hi6, lo10 in ((True, k_hi6, k_lo10), (False, v_hi6, v_lo10)):
            group = get_group(is_k, stage)
            hi0, hi1 = group_hi6_counts[group]
            lo0, lo1 = group_lo10_counts[group]
            prev_hi6 = 0
            prev_lo10 = 0
            for i in range(len(hi6)):
                cur_hi6 = int(hi6[i])
                cur_lo10 = int(lo10[i])
                d_hi6 = (cur_hi6 - prev_hi6) & 0x3F
                # hi6 bits
                cur_val = 0
                for bit_idx, bit in iter_bits(d_hi6, 6):
                    prefix = (cur_val >> (bit_idx + 1)) & prefix_mask if args.prefix_bits else 0
                    ctx = (prev_hi6 * args.hash_a + prefix * args.hash_b + bit_idx) % args.hi6_buckets
                    if bit == 0:
                        hi0[ctx] += 1
                    else:
                        hi1[ctx] += 1
                    cur_val |= (bit << bit_idx)
                # lo10 bits
                cur_val = 0
                for bit_idx, bit in iter_bits(cur_lo10, 10):
                    prefix = (cur_val >> (bit_idx + 1)) & prefix_mask if args.prefix_bits else 0
                    ctx = (cur_hi6 * args.hash_a + prev_hi6 * args.hash_b + prev_lo10 * args.hash_c +
                           prefix * args.hash_d + bit_idx) % args.lo10_buckets
                    if bit == 0:
                        lo0[ctx] += 1
                    else:
                        lo1[ctx] += 1
                    cur_val |= (bit << bit_idx)
                prev_hi6 = cur_hi6
                prev_lo10 = cur_lo10

    # Save prior codebooks
    for group in group_hi6_counts:
        hi0, hi1 = group_hi6_counts[group]
        lo0, lo1 = group_lo10_counts[group]
        write_json(os.path.join(args.out_dir, f\"cabac_codebook_{group}.json\"), {
            \"group\": group,
            \"hi6_buckets\": args.hi6_buckets,
            \"lo10_buckets\": args.lo10_buckets,
            \"hash_a\": args.hash_a,
            \"hash_b\": args.hash_b,
            \"hash_c\": args.hash_c,
            \"hash_d\": args.hash_d,
            \"prefix_bits\": args.prefix_bits,
            \"rescale_threshold\": args.rescale_threshold,
            \"hi6_count0\": hi0,
            \"hi6_count1\": hi1,
            \"lo10_count0\": lo0,
            \"lo10_count1\": lo1,
        })

    # Encode with adaptive models seeded by priors
    encoded = 0
    skipped = 0
    errors = 0
    for meta_path in meta_files:
        try:
            meta = load_json(meta_path)
        except Exception as exc:
            print(f\"Skip {meta_path}: {exc}\", file=sys.stderr)
            errors += 1
            continue
        stage = meta.get(\"stage\", \"unknown\")
        if args.stage != \"both\" and stage != args.stage:
            skipped += 1
            continue
        if stage not in (\"prefill\", \"decode\"):
            skipped += 1
            continue
        seq_len = int(meta.get(\"seq_len\", 0))
        if args.min_seq_len and seq_len < args.min_seq_len:
            skipped += 1
            continue
        if args.max_seq_len and seq_len > args.max_seq_len:
            skipped += 1
            continue
        if int(meta.get(\"bytes_per_elem\", 0)) != 2:
            skipped += 1
            continue

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta.get(\"k_file\", \"\"))
        v_path = os.path.join(dirpath, meta.get(\"v_file\", \"\"))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            skipped += 1
            continue

        rel_dir = os.path.relpath(dirpath, args.dump_dir)
        dst_dir = os.path.join(args.out_dir, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)
        meta_name = os.path.splitext(os.path.basename(meta_path))[0]
        dst_meta_path = os.path.join(dst_dir, f\"cabac_{meta_name}.json\")
        if not args.overwrite and os.path.exists(dst_meta_path):
            skipped += 1
            continue

        try:
            k = np.frombuffer(open(k_path, \"rb\").read(), dtype=np.uint16)
            v = np.frombuffer(open(v_path, \"rb\").read(), dtype=np.uint16)
            k_hi6, k_lo10 = split_fp16(k)
            v_hi6, v_lo10 = split_fp16(v)

            k_group = get_group(True, stage)
            v_group = get_group(False, stage)

            k_hi0, k_hi1 = group_hi6_counts[k_group]
            k_lo0, k_lo1 = group_lo10_counts[k_group]
            v_hi0, v_hi1 = group_hi6_counts[v_group]
            v_lo0, v_lo1 = group_lo10_counts[v_group]

            # K stream
            hi6_model = AdaptiveBinaryModel.from_counts(k_hi0, k_hi1, args.rescale_threshold)
            lo10_model = AdaptiveBinaryModel.from_counts(k_lo0, k_lo1, args.rescale_threshold)
            hi6_enc = BinaryArithmeticEncoder()
            lo10_enc = BinaryArithmeticEncoder()
            prev_hi6 = 0
            prev_lo10 = 0
            for i in range(len(k_hi6)):
                cur_hi6 = int(k_hi6[i])
                cur_lo10 = int(k_lo10[i])
                d_hi6 = (cur_hi6 - prev_hi6) & 0x3F
                cur_val = 0
                for bit_idx, bit in iter_bits(d_hi6, 6):
                    prefix = (cur_val >> (bit_idx + 1)) & prefix_mask if args.prefix_bits else 0
                    ctx = (prev_hi6 * args.hash_a + prefix * args.hash_b + bit_idx) % args.hi6_buckets
                    c0, c1 = hi6_model.get(ctx)
                    hi6_enc.encode_bit(bit, c0, c1)
                    hi6_model.update(ctx, bit)
                    cur_val |= (bit << bit_idx)
                cur_val = 0
                for bit_idx, bit in iter_bits(cur_lo10, 10):
                    prefix = (cur_val >> (bit_idx + 1)) & prefix_mask if args.prefix_bits else 0
                    ctx = (cur_hi6 * args.hash_a + prev_hi6 * args.hash_b + prev_lo10 * args.hash_c +
                           prefix * args.hash_d + bit_idx) % args.lo10_buckets
                    c0, c1 = lo10_model.get(ctx)
                    lo10_enc.encode_bit(bit, c0, c1)
                    lo10_model.update(ctx, bit)
                    cur_val |= (bit << bit_idx)
                prev_hi6 = cur_hi6
                prev_lo10 = cur_lo10

            k_hi6_path = os.path.join(dst_dir, f\"k_hi6_{meta_name}.cabac\")
            k_lo10_path = os.path.join(dst_dir, f\"k_lo10_{meta_name}.cabac\")
            with open(k_hi6_path, \"wb\") as f:
                f.write(hi6_enc.finish())
            with open(k_lo10_path, \"wb\") as f:
                f.write(lo10_enc.finish())

            # V stream
            hi6_model = AdaptiveBinaryModel.from_counts(v_hi0, v_hi1, args.rescale_threshold)
            lo10_model = AdaptiveBinaryModel.from_counts(v_lo0, v_lo1, args.rescale_threshold)
            hi6_enc = BinaryArithmeticEncoder()
            lo10_enc = BinaryArithmeticEncoder()
            prev_hi6 = 0
            prev_lo10 = 0
            for i in range(len(v_hi6)):
                cur_hi6 = int(v_hi6[i])
                cur_lo10 = int(v_lo10[i])
                d_hi6 = (cur_hi6 - prev_hi6) & 0x3F
                cur_val = 0
                for bit_idx, bit in iter_bits(d_hi6, 6):
                    prefix = (cur_val >> (bit_idx + 1)) & prefix_mask if args.prefix_bits else 0
                    ctx = (prev_hi6 * args.hash_a + prefix * args.hash_b + bit_idx) % args.hi6_buckets
                    c0, c1 = hi6_model.get(ctx)
                    hi6_enc.encode_bit(bit, c0, c1)
                    hi6_model.update(ctx, bit)
                    cur_val |= (bit << bit_idx)
                cur_val = 0
                for bit_idx, bit in iter_bits(cur_lo10, 10):
                    prefix = (cur_val >> (bit_idx + 1)) & prefix_mask if args.prefix_bits else 0
                    ctx = (cur_hi6 * args.hash_a + prev_hi6 * args.hash_b + prev_lo10 * args.hash_c +
                           prefix * args.hash_d + bit_idx) % args.lo10_buckets
                    c0, c1 = lo10_model.get(ctx)
                    lo10_enc.encode_bit(bit, c0, c1)
                    lo10_model.update(ctx, bit)
                    cur_val |= (bit << bit_idx)
                prev_hi6 = cur_hi6
                prev_lo10 = cur_lo10

            v_hi6_path = os.path.join(dst_dir, f\"v_hi6_{meta_name}.cabac\")
            v_lo10_path = os.path.join(dst_dir, f\"v_lo10_{meta_name}.cabac\")
            with open(v_hi6_path, \"wb\") as f:
                f.write(hi6_enc.finish())
            with open(v_lo10_path, \"wb\") as f:
                f.write(lo10_enc.finish())

            out_meta = dict(meta)
            out_meta.update({
                \"layout\": \"bitstream_cabac_ctx\",
                \"bytes_per_elem\": 2,
                \"k_group\": k_group,
                \"v_group\": v_group,
                \"hi6_buckets\": args.hi6_buckets,
                \"lo10_buckets\": args.lo10_buckets,
                \"hash_a\": args.hash_a,
                \"hash_b\": args.hash_b,
                \"hash_c\": args.hash_c,
                \"hash_d\": args.hash_d,
                \"prefix_bits\": args.prefix_bits,
                \"rescale_threshold\": args.rescale_threshold,
                \"k_hi6_file\": os.path.basename(k_hi6_path),
                \"k_lo10_file\": os.path.basename(k_lo10_path),
                \"v_hi6_file\": os.path.basename(v_hi6_path),
                \"v_lo10_file\": os.path.basename(v_lo10_path),
                \"num_elements\": int(k.size),
            })
            write_json(dst_meta_path, out_meta)
            encoded += 1
        except Exception as exc:
            print(f\"Error {meta_path}: {exc}\", file=sys.stderr)
            errors += 1

    print(f\"Encoded: {encoded}\")
    print(f\"Skipped: {skipped}\")
    if errors:
        print(f\"Errors: {errors}\", file=sys.stderr)
        return 2
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
