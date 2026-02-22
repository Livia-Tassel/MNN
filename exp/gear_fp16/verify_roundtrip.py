#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterator, Tuple

import numpy as np


def iter_meta_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("meta_") and name.endswith(".json"):
                yield os.path.join(dirpath, name)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def should_keep_stage(meta_stage: str, target_stage: str) -> bool:
    if target_stage == "both":
        return True
    return meta_stage == target_stage


def delta_encode_u8(arr_u8: np.ndarray, seq_len: int, heads: int, head_dim: int) -> np.ndarray:
    if seq_len <= 1:
        return arr_u8.copy()
    view = arr_u8.reshape((seq_len, heads, head_dim))
    first = view[0:1]
    delta = (view[1:].astype(np.int16) - view[:-1].astype(np.int16)) & 0xFF
    delta = delta.astype(np.uint8)
    return np.concatenate([first, delta], axis=0)


def delta_decode_u8(encoded: np.ndarray, seq_len: int, heads: int, head_dim: int) -> np.ndarray:
    if seq_len <= 1:
        return encoded.copy()
    view = encoded.reshape((seq_len, heads, head_dim))
    out = np.empty_like(view)
    out[0] = view[0]
    for t in range(1, seq_len):
        out[t] = (out[t - 1].astype(np.uint16) + view[t].astype(np.uint16)) & 0xFF
    return out


def roundtrip_fp16(data: bytes, seq_len: int, heads: int, head_dim: int) -> Tuple[bool, int]:
    u16 = np.frombuffer(data, dtype=np.uint16)
    hi = (u16 >> 8).astype(np.uint8)
    lo = (u16 & 0xFF).astype(np.uint8)

    hi_enc = delta_encode_u8(hi, seq_len, heads, head_dim)
    hi_dec = delta_decode_u8(hi_enc, seq_len, heads, head_dim)

    if not np.array_equal(hi, hi_dec.reshape(-1)):
        diff = np.not_equal(hi, hi_dec.reshape(-1))
        return False, int(diff.sum())

    u16_rec = ((hi_dec.reshape(-1).astype(np.uint16) << 8) | lo.astype(np.uint16)).astype(np.uint16)
    if not np.array_equal(u16, u16_rec):
        diff = np.not_equal(u16, u16_rec)
        return False, int(diff.sum())
    return True, 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify gear+delta roundtrip correctness on FP16 KV dumps.")
    parser.add_argument("--dump-dir", required=True, help="Root dump directory containing run_* subdirs.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--min-seq-len", type=int, default=0, help="Skip entries with seq_len smaller than this.")
    parser.add_argument("--max-seq-len", type=int, default=0, help="Skip entries with seq_len larger than this (0 = no limit).")
    parser.add_argument("--limit", type=int, default=0, help="Max number of dumps to check (0 = all).")
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.dump_dir))
    if not meta_files:
        print(f"No meta_*.json found under {args.dump_dir}", file=sys.stderr)
        return 1

    checked = 0
    failed = 0

    for meta_path in meta_files:
        meta = load_json(meta_path)
        stage = meta.get("stage", "unknown")
        if not should_keep_stage(stage, args.stage):
            continue
        seq_len = int(meta.get("seq_len", 0))
        if args.min_seq_len and seq_len < args.min_seq_len:
            continue
        if args.max_seq_len and seq_len > args.max_seq_len:
            continue
        bytes_per_elem = int(meta.get("bytes_per_elem", 0))
        if bytes_per_elem != 2:
            continue

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta.get("k_file", ""))
        v_path = os.path.join(dirpath, meta.get("v_file", ""))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            continue

        kv_heads = int(meta.get("kv_heads", 0))
        head_dim = int(meta.get("head_dim", 0))

        with open(k_path, "rb") as f:
            k_bytes = f.read()
        with open(v_path, "rb") as f:
            v_bytes = f.read()

        ok_k, diff_k = roundtrip_fp16(k_bytes, seq_len, kv_heads, head_dim)
        ok_v, diff_v = roundtrip_fp16(v_bytes, seq_len, kv_heads, head_dim)
        checked += 1

        if not ok_k or not ok_v:
            failed += 1
            print(f"FAIL {meta_path}: K diff={diff_k}, V diff={diff_v}")

        if args.limit and checked >= args.limit:
            break

    print(f"Checked: {checked}, Failed: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
