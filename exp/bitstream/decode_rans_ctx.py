#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterator, Tuple

import numpy as np

from context_utils import ctx_id
from rans_utils import build_symbol_table, rans_decode, RansDecoder


def iter_meta_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("rans_") and name.endswith(".json"):
                yield os.path.join(dirpath, name)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_fp16(u16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hi6 = (u16 >> 10) & 0x3F
    lo10 = u16 & 0x3FF
    return hi6.astype(np.uint16), lo10.astype(np.uint16)


def load_codebook(path: str) -> Dict:
    data = load_json(path)
    hi6_norm, hi6_cum, hi6_table = build_symbol_table(data["hi6_freq"])
    lo10_norms = []
    lo10_cums = []
    lo10_tables = []
    for bucket_freqs in data["lo10_freqs"]:
        norm, cum, table = build_symbol_table(bucket_freqs)
        lo10_norms.append(norm)
        lo10_cums.append(cum)
        lo10_tables.append(table)
    return {
        "buckets": data["buckets"],
        "hash_a": data["hash_a"],
        "hash_b": data["hash_b"],
        "hi6_norm": hi6_norm,
        "hi6_cum": hi6_cum,
        "hi6_table": hi6_table,
        "lo10_norms": lo10_norms,
        "lo10_cums": lo10_cums,
        "lo10_tables": lo10_tables,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode rANS context-coded bitstreams and verify exact match.")
    parser.add_argument("--bitstream-dir", required=True)
    parser.add_argument("--orig-dump-dir", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.bitstream_dir))
    if not meta_files:
        print(f"No rans_*.json found under {args.bitstream_dir}", file=sys.stderr)
        return 1

    # Load codebooks
    codebooks = {}
    for name in ("k_prefill", "k_decode", "v_prefill", "v_decode"):
        path = os.path.join(args.bitstream_dir, f"codebook_{name}.json")
        if os.path.exists(path):
            codebooks[name] = load_codebook(path)

    checked = 0
    failed = 0
    for meta_path in meta_files:
        meta = load_json(meta_path)
        rel_dir = os.path.relpath(os.path.dirname(meta_path), args.bitstream_dir)
        orig_dir = os.path.join(args.orig_dump_dir, rel_dir)

        stage = meta.get("stage", "unknown")
        k_group = meta.get("k_group", f"k_{stage}")
        v_group = meta.get("v_group", f"v_{stage}")
        if k_group not in codebooks or v_group not in codebooks:
            print(f"Missing codebook for {meta_path}", file=sys.stderr)
            failed += 1
            continue

        num = int(meta["num_elements"])
        k_hi6_path = os.path.join(os.path.dirname(meta_path), meta["k_hi6_file"])
        k_lo10_path = os.path.join(os.path.dirname(meta_path), meta["k_lo10_file"])
        v_hi6_path = os.path.join(os.path.dirname(meta_path), meta["v_hi6_file"])
        v_lo10_path = os.path.join(os.path.dirname(meta_path), meta["v_lo10_file"])

        # Decode K
        k_book = codebooks[k_group]
        d_hi6_k = rans_decode(
            open(k_hi6_path, "rb").read(),
            num,
            k_book["hi6_norm"],
            k_book["hi6_cum"],
            k_book["hi6_table"],
        )
        k_hi6 = [0] * num
        prev_hi6 = 0
        for i, d in enumerate(d_hi6_k):
            cur = (prev_hi6 + d) & 0x3F
            k_hi6[i] = cur
            prev_hi6 = cur

        k_lo10 = [0] * num
        decoder = RansDecoder(open(k_lo10_path, "rb").read())
        prev_hi6 = 0
        prev_lo10 = 0
        for i in range(num):
            cur_hi6 = k_hi6[i]
            bucket = ctx_id(cur_hi6, prev_hi6, prev_lo10, k_book["buckets"], k_book["hash_a"], k_book["hash_b"])
            sym = decoder.decode_symbol(
                k_book["lo10_tables"][bucket],
                k_book["lo10_norms"][bucket],
                k_book["lo10_cums"][bucket],
            )
            k_lo10[i] = sym
            prev_hi6 = cur_hi6
            prev_lo10 = sym

        # Decode V
        v_book = codebooks[v_group]
        d_hi6_v = rans_decode(
            open(v_hi6_path, "rb").read(),
            num,
            v_book["hi6_norm"],
            v_book["hi6_cum"],
            v_book["hi6_table"],
        )
        v_hi6 = [0] * num
        prev_hi6 = 0
        for i, d in enumerate(d_hi6_v):
            cur = (prev_hi6 + d) & 0x3F
            v_hi6[i] = cur
            prev_hi6 = cur

        v_lo10 = [0] * num
        decoder = RansDecoder(open(v_lo10_path, "rb").read())
        prev_hi6 = 0
        prev_lo10 = 0
        for i in range(num):
            cur_hi6 = v_hi6[i]
            bucket = ctx_id(cur_hi6, prev_hi6, prev_lo10, v_book["buckets"], v_book["hash_a"], v_book["hash_b"])
            sym = decoder.decode_symbol(
                v_book["lo10_tables"][bucket],
                v_book["lo10_norms"][bucket],
                v_book["lo10_cums"][bucket],
            )
            v_lo10[i] = sym
            prev_hi6 = cur_hi6
            prev_lo10 = sym

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
