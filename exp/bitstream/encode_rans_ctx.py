#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterator, List, Tuple

import numpy as np

from context_utils import ctx_id
from rans_utils import build_symbol_table, rans_encode, rans_encode_ctx


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Lossless rANS with context (lo10 | hi6, prev_hi6, prev_lo10).")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--out-dir", default="exp/bitstream/out")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--min-seq-len", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--buckets", type=int, default=4096)
    parser.add_argument("--hash-a", type=int, default=131)
    parser.add_argument("--hash-b", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.dump_dir))
    if not meta_files:
        print(f"No meta_*.json found under {args.dump_dir}", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    # Build codebooks per group (k/v + stage)
    hi6_freqs: Dict[str, List[int]] = {}
    lo10_freqs: Dict[str, List[List[int]]] = {}
    for key_is_k in (True, False):
        for stage in ("prefill", "decode"):
            group = get_group(key_is_k, stage)
            hi6_freqs[group] = [0] * 64
            lo10_freqs[group] = [[0] * 1024 for _ in range(args.buckets)]

    for meta_path in meta_files:
        try:
            meta = load_json(meta_path)
        except Exception:
            continue
        stage = meta.get("stage", "unknown")
        if args.stage != "both" and stage != args.stage:
            continue
        if stage not in ("prefill", "decode"):
            continue
        seq_len = int(meta.get("seq_len", 0))
        if args.min_seq_len and seq_len < args.min_seq_len:
            continue
        if args.max_seq_len and seq_len > args.max_seq_len:
            continue
        if int(meta.get("bytes_per_elem", 0)) != 2:
            continue

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta.get("k_file", ""))
        v_path = os.path.join(dirpath, meta.get("v_file", ""))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            continue

        k = np.frombuffer(open(k_path, "rb").read(), dtype=np.uint16)
        v = np.frombuffer(open(v_path, "rb").read(), dtype=np.uint16)
        k_hi6, k_lo10 = split_fp16(k)
        v_hi6, v_lo10 = split_fp16(v)

        for is_k, hi6, lo10 in ((True, k_hi6, k_lo10), (False, v_hi6, v_lo10)):
            group = get_group(is_k, stage)
            prev_hi6 = 0
            prev_lo10 = 0
            for i in range(len(hi6)):
                cur_hi6 = int(hi6[i])
                cur_lo10 = int(lo10[i])
                # hi6 delta (mod 64)
                d_hi6 = (cur_hi6 - prev_hi6) & 0x3F
                hi6_freqs[group][d_hi6] += 1
                # lo10 context
                bucket = ctx_id(cur_hi6, prev_hi6, prev_lo10, args.buckets, args.hash_a, args.hash_b)
                lo10_freqs[group][bucket][cur_lo10] += 1
                prev_hi6 = cur_hi6
                prev_lo10 = cur_lo10

    # Build codebook tables (normalized freqs + cums + symbol tables)
    codebooks = {}
    for group in hi6_freqs:
        hi6_norm, hi6_cum, hi6_table = build_symbol_table(hi6_freqs[group])
        lo10_tables = []
        lo10_cums = []
        lo10_norms = []
        for bucket in range(args.buckets):
            bucket_freqs = lo10_freqs[group][bucket]
            if sum(bucket_freqs) == 0:
                bucket_freqs = [1] + [0] * 1023
                lo10_freqs[group][bucket] = bucket_freqs
            norm, cum, table = build_symbol_table(bucket_freqs)
            lo10_norms.append(norm)
            lo10_cums.append(cum)
            lo10_tables.append(table)
        codebooks[group] = {
            "hi6_norm": hi6_norm,
            "hi6_cum": hi6_cum,
            "hi6_table": hi6_table,
            "lo10_norms": lo10_norms,
            "lo10_cums": lo10_cums,
            "lo10_tables": lo10_tables,
        }

    # Save codebooks (frequency tables only)
    for group in hi6_freqs:
        write_json(os.path.join(args.out_dir, f"codebook_{group}.json"), {
            "group": group,
            "buckets": args.buckets,
            "hash_a": args.hash_a,
            "hash_b": args.hash_b,
            "hi6_freq": hi6_freqs[group],
            "lo10_freqs": lo10_freqs[group],
        })

    # Encode
    encoded = 0
    skipped = 0
    errors = 0
    for meta_path in meta_files:
        try:
            meta = load_json(meta_path)
        except Exception as exc:
            print(f"Skip {meta_path}: {exc}", file=sys.stderr)
            errors += 1
            continue
        stage = meta.get("stage", "unknown")
        if args.stage != "both" and stage != args.stage:
            skipped += 1
            continue
        if stage not in ("prefill", "decode"):
            skipped += 1
            continue
        seq_len = int(meta.get("seq_len", 0))
        if args.min_seq_len and seq_len < args.min_seq_len:
            skipped += 1
            continue
        if args.max_seq_len and seq_len > args.max_seq_len:
            skipped += 1
            continue
        if int(meta.get("bytes_per_elem", 0)) != 2:
            skipped += 1
            continue

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta.get("k_file", ""))
        v_path = os.path.join(dirpath, meta.get("v_file", ""))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            skipped += 1
            continue

        rel_dir = os.path.relpath(dirpath, args.dump_dir)
        dst_dir = os.path.join(args.out_dir, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)
        meta_name = os.path.splitext(os.path.basename(meta_path))[0]
        dst_meta_path = os.path.join(dst_dir, f"rans_{meta_name}.json")
        if not args.overwrite and os.path.exists(dst_meta_path):
            skipped += 1
            continue

        try:
            k = np.frombuffer(open(k_path, "rb").read(), dtype=np.uint16)
            v = np.frombuffer(open(v_path, "rb").read(), dtype=np.uint16)
            k_hi6, k_lo10 = split_fp16(k)
            v_hi6, v_lo10 = split_fp16(v)

            k_group = ""
            v_group = ""
            for is_k, hi6, lo10 in ((True, k_hi6, k_lo10), (False, v_hi6, v_lo10)):
                group = get_group(is_k, stage)
                book = codebooks[group]
                prev_hi6 = 0
                prev_lo10 = 0

                d_hi6 = []
                lo10_syms = []
                lo10_ctx = []
                for i in range(len(hi6)):
                    cur_hi6 = int(hi6[i])
                    cur_lo10 = int(lo10[i])
                    d_hi6.append((cur_hi6 - prev_hi6) & 0x3F)
                    ctx = ctx_id(cur_hi6, prev_hi6, prev_lo10, args.buckets, args.hash_a, args.hash_b)
                    lo10_syms.append(cur_lo10)
                    lo10_ctx.append(ctx)
                    prev_hi6 = cur_hi6
                    prev_lo10 = cur_lo10

                hi6_bs = rans_encode(d_hi6, book["hi6_norm"], book["hi6_cum"])
                # Encode lo10 with per-symbol context using single rANS stream.
                lo10_bs = rans_encode_ctx(lo10_syms, lo10_ctx, book["lo10_norms"], book["lo10_cums"])

                prefix = "k" if is_k else "v"
                hi6_path = os.path.join(dst_dir, f"{prefix}_hi6_{meta_name}.rans")
                lo10_path = os.path.join(dst_dir, f"{prefix}_lo10_{meta_name}.rans")
                with open(hi6_path, "wb") as f:
                    f.write(hi6_bs)
                with open(lo10_path, "wb") as f:
                    f.write(lo10_bs)

                if is_k:
                    k_hi6_file = os.path.basename(hi6_path)
                    k_lo10_file = os.path.basename(lo10_path)
                    k_group = group
                else:
                    v_hi6_file = os.path.basename(hi6_path)
                    v_lo10_file = os.path.basename(lo10_path)
                    v_group = group

            out_meta = dict(meta)
            out_meta.update({
                "layout": "bitstream_rans_ctx",
                "bytes_per_elem": 2,
                "codebook": "grouped",
                "k_group": k_group,
                "v_group": v_group,
                "buckets": args.buckets,
                "hash_a": args.hash_a,
                "hash_b": args.hash_b,
                "k_hi6_file": k_hi6_file,
                "k_lo10_file": k_lo10_file,
                "v_hi6_file": v_hi6_file,
                "v_lo10_file": v_lo10_file,
                "num_elements": int(k.size),
            })
            write_json(dst_meta_path, out_meta)
            encoded += 1
        except Exception as exc:
            print(f"Error {meta_path}: {exc}", file=sys.stderr)
            errors += 1

    print(f"Encoded: {encoded}")
    print(f"Skipped: {skipped}")
    if errors:
        print(f"Errors: {errors}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
