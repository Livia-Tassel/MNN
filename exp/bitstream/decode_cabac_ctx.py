#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterator, Tuple

import numpy as np

from cabac_utils import AdaptiveBinaryModel, BinaryArithmeticDecoder
from context_utils import ctx_id


def iter_meta_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("cabac_") and name.endswith(".json"):
                yield os.path.join(dirpath, name)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_fp16(u16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hi6 = (u16 >> 10) & 0x3F
    lo10 = u16 & 0x3FF
    return hi6.astype(np.uint16), lo10.astype(np.uint16)


def iter_bits(bits: int) -> Iterator[int]:
    for bit_idx in range(bits - 1, -1, -1):
        yield bit_idx


def load_codebook(path: str) -> Dict:
    data = load_json(path)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode CABAC context bitstreams and verify exact match.")
    parser.add_argument("--bitstream-dir", required=True)
    parser.add_argument("--orig-dump-dir", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.bitstream_dir))
    if not meta_files:
        print(f"No cabac_*.json found under {args.bitstream_dir}", file=sys.stderr)
        return 1

    codebooks = {}
    for name in ("k_prefill", "k_decode", "v_prefill", "v_decode"):
        path = os.path.join(args.bitstream_dir, f"cabac_codebook_{name}.json")
        if os.path.exists(path):
            codebooks[name] = load_codebook(path)

    checked = 0
    failed = 0
    for meta_path in meta_files:
        meta = load_json(meta_path)
        rel_dir = os.path.relpath(os.path.dirname(meta_path), args.bitstream_dir)
        orig_dir = os.path.join(args.orig_dump_dir, rel_dir)

        k_group = meta.get("k_group")
        v_group = meta.get("v_group")
        if k_group not in codebooks or v_group not in codebooks:
            print(f"Missing codebook for {meta_path}", file=sys.stderr)
            failed += 1
            continue

        num = int(meta["num_elements"])
        prefix_bits = int(meta["prefix_bits"])
        prefix_mask = (1 << prefix_bits) - 1 if prefix_bits > 0 else 0

        # K decode
        cb = codebooks[k_group]
        hi6_model = AdaptiveBinaryModel.from_counts(cb["hi6_count0"], cb["hi6_count1"], cb["rescale_threshold"])
        lo10_model = AdaptiveBinaryModel.from_counts(cb["lo10_count0"], cb["lo10_count1"], cb["rescale_threshold"])
        hi6_dec = BinaryArithmeticDecoder(open(os.path.join(os.path.dirname(meta_path), meta["k_hi6_file"]), "rb").read())
        lo10_dec = BinaryArithmeticDecoder(open(os.path.join(os.path.dirname(meta_path), meta["k_lo10_file"]), "rb").read())

        k_hi6 = [0] * num
        prev_hi6 = 0
        for i in range(num):
            d_val = 0
            for bit_idx in iter_bits(6):
                prefix = (d_val >> (bit_idx + 1)) & prefix_mask if prefix_bits else 0
                ctx = (prev_hi6 * cb["hash_a"] + prefix * cb["hash_b"] + bit_idx) % cb["hi6_buckets"]
                c0, c1 = hi6_model.get(ctx)
                bit = hi6_dec.decode_bit(c0, c1)
                hi6_model.update(ctx, bit)
                d_val |= (bit << bit_idx)
            cur = (prev_hi6 + d_val) & 0x3F
            k_hi6[i] = cur
            prev_hi6 = cur

        k_lo10 = [0] * num
        prev_hi6 = 0
        prev_lo10 = 0
        for i in range(num):
            cur_hi6 = k_hi6[i]
            val = 0
            for bit_idx in iter_bits(10):
                prefix = (val >> (bit_idx + 1)) & prefix_mask if prefix_bits else 0
                ctx = (cur_hi6 * cb["hash_a"] + prev_hi6 * cb["hash_b"] + prev_lo10 * cb["hash_c"] +
                       prefix * cb["hash_d"] + bit_idx) % cb["lo10_buckets"]
                c0, c1 = lo10_model.get(ctx)
                bit = lo10_dec.decode_bit(c0, c1)
                lo10_model.update(ctx, bit)
                val |= (bit << bit_idx)
            k_lo10[i] = val
            prev_hi6 = cur_hi6
            prev_lo10 = val

        # V decode
        cb = codebooks[v_group]
        hi6_model = AdaptiveBinaryModel.from_counts(cb["hi6_count0"], cb["hi6_count1"], cb["rescale_threshold"])
        lo10_model = AdaptiveBinaryModel.from_counts(cb["lo10_count0"], cb["lo10_count1"], cb["rescale_threshold"])
        hi6_dec = BinaryArithmeticDecoder(open(os.path.join(os.path.dirname(meta_path), meta["v_hi6_file"]), "rb").read())
        lo10_dec = BinaryArithmeticDecoder(open(os.path.join(os.path.dirname(meta_path), meta["v_lo10_file"]), "rb").read())

        v_hi6 = [0] * num
        prev_hi6 = 0
        for i in range(num):
            d_val = 0
            for bit_idx in iter_bits(6):
                prefix = (d_val >> (bit_idx + 1)) & prefix_mask if prefix_bits else 0
                ctx = (prev_hi6 * cb["hash_a"] + prefix * cb["hash_b"] + bit_idx) % cb["hi6_buckets"]
                c0, c1 = hi6_model.get(ctx)
                bit = hi6_dec.decode_bit(c0, c1)
                hi6_model.update(ctx, bit)
                d_val |= (bit << bit_idx)
            cur = (prev_hi6 + d_val) & 0x3F
            v_hi6[i] = cur
            prev_hi6 = cur

        v_lo10 = [0] * num
        prev_hi6 = 0
        prev_lo10 = 0
        for i in range(num):
            cur_hi6 = v_hi6[i]
            val = 0
            for bit_idx in iter_bits(10):
                prefix = (val >> (bit_idx + 1)) & prefix_mask if prefix_bits else 0
                ctx = (cur_hi6 * cb["hash_a"] + prev_hi6 * cb["hash_b"] + prev_lo10 * cb["hash_c"] +
                       prefix * cb["hash_d"] + bit_idx) % cb["lo10_buckets"]
                c0, c1 = lo10_model.get(ctx)
                bit = lo10_dec.decode_bit(c0, c1)
                lo10_model.update(ctx, bit)
                val |= (bit << bit_idx)
            v_lo10[i] = val
            prev_hi6 = cur_hi6
            prev_lo10 = val

        k_u16 = (np.array(k_hi6, dtype=np.uint16) << 10) | np.array(k_lo10, dtype=np.uint16)
        v_u16 = (np.array(v_hi6, dtype=np.uint16) << 10) | np.array(v_lo10, dtype=np.uint16)

        if args.verify:
            k_path = os.path.join(orig_dir, meta.get("k_file", ""))
            v_path = os.path.join(orig_dir, meta.get("v_file", ""))
            if not os.path.exists(k_path) or not os.path.exists(v_path):
                print(f"Missing original K/V for {meta_path}", file=sys.stderr)
                failed += 1
                continue
            k_orig = np.frombuffer(open(k_path, "rb").read(), dtype=np.uint16)
            v_orig = np.frombuffer(open(v_path, "rb").read(), dtype=np.uint16)
            if not (np.array_equal(k_u16, k_orig) and np.array_equal(v_u16, v_orig)):
                print(f"Mismatch: {meta_path}", file=sys.stderr)
                failed += 1
            else:
                checked += 1
        else:
            checked += 1

    print(f"Checked: {checked}, Failed: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
