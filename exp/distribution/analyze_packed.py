#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np

try:
    import zstandard as zstd
except Exception:
    zstd = None


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def entropy_bytes(data: bytes) -> float:
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / arr.size
    return float(-(probs * np.log2(probs)).sum())


def zstd_ratio(data: bytes, level: int) -> float:
    if not data or zstd is None:
        return 0.0
    compressor = zstd.ZstdCompressor(level=level)
    compressed = compressor.compress(data)
    if not compressed:
        return 0.0
    return len(data) / len(compressed)


def gear_ratios_fp16(data: bytes, level: int) -> Tuple[float, float, float, float, float]:
    u16 = np.frombuffer(data, dtype=np.uint16)
    hi = (u16 >> 8).astype(np.uint8).tobytes()
    lo = (u16 & 0xFF).astype(np.uint8).tobytes()
    hi_ratio = zstd_ratio(hi, level)
    lo_ratio = zstd_ratio(lo, level)
    gear_ratio = 0.0
    if zstd is not None and (hi_ratio > 0.0 or lo_ratio > 0.0):
        compressor = zstd.ZstdCompressor(level=level)
        hi_c = compressor.compress(hi)
        lo_c = compressor.compress(lo)
        if hi_c and lo_c:
            gear_ratio = len(data) / (len(hi_c) + len(lo_c))
    return entropy_bytes(hi), entropy_bytes(lo), hi_ratio, lo_ratio, gear_ratio


def split16_ratios_fp32(data: bytes, level: int) -> Tuple[float, float, float, float, float]:
    u32 = np.frombuffer(data, dtype=np.uint32)
    hi = (u32 >> 16).astype(np.uint16).tobytes()
    lo = (u32 & 0xFFFF).astype(np.uint16).tobytes()
    hi_entropy = entropy_bytes(hi)
    lo_entropy = entropy_bytes(lo)
    hi_ratio = zstd_ratio(hi, level)
    lo_ratio = zstd_ratio(lo, level)
    split_ratio = 0.0
    if zstd is not None and (hi_ratio > 0.0 or lo_ratio > 0.0):
        compressor = zstd.ZstdCompressor(level=level)
        hi_c = compressor.compress(hi)
        lo_c = compressor.compress(lo)
        if hi_c and lo_c:
            split_ratio = len(data) / (len(hi_c) + len(lo_c))
    return hi_entropy, lo_entropy, hi_ratio, lo_ratio, split_ratio


def iter_packed_meta(root: str):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("packed_meta_") and name.endswith(".json"):
                yield os.path.join(dirpath, name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze packed KV dumps.")
    parser.add_argument("--dump-dir", required=True, help="Root dump directory containing run_* subdirs.")
    parser.add_argument("--out-dir", default="exp/distribution/out", help="Output directory for csv/summary.")
    parser.add_argument("--zstd-level", type=int, default=3, help="Zstd compression level.")
    parser.add_argument("--min-seq-len", type=int, default=0, help="Skip entries with used_seq_len smaller than this.")
    parser.add_argument("--max-seq-len", type=int, default=0, help="Skip entries with used_seq_len larger than this (0 = no limit).")
    args = parser.parse_args()

    if zstd is None:
        print("Missing dependency: zstandard. Install with `pip install zstandard`.", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "packed_layer_metrics.csv")
    summary_path = os.path.join(args.out_dir, "packed_summary.md")

    rows = []
    meta_files = list(iter_packed_meta(args.dump_dir))
    if not meta_files:
        print(f"No packed_meta_*.json found under {args.dump_dir}", file=sys.stderr)
        return 1

    for meta_path in meta_files:
        meta = load_json(meta_path)
        used_seq_len = int(meta.get("used_seq_len", 0))
        if args.min_seq_len and used_seq_len < args.min_seq_len:
            continue
        if args.max_seq_len and used_seq_len > args.max_seq_len:
            continue

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta["k_file"])
        v_path = os.path.join(dirpath, meta["v_file"])
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            continue

        with open(k_path, "rb") as f:
            k_bytes = f.read()
        with open(v_path, "rb") as f:
            v_bytes = f.read()

        bytes_per_elem = int(meta["bytes_per_elem"])

        k_entropy = entropy_bytes(k_bytes)
        v_entropy = entropy_bytes(v_bytes)
        k_zstd = zstd_ratio(k_bytes, args.zstd_level)
        v_zstd = zstd_ratio(v_bytes, args.zstd_level)

        k_hi16_entropy = k_lo16_entropy = k_hi16_ratio = k_lo16_ratio = k_split16_ratio = 0.0
        v_hi16_entropy = v_lo16_entropy = v_hi16_ratio = v_lo16_ratio = v_split16_ratio = 0.0
        k_hi8_entropy = k_lo8_entropy = k_hi8_ratio = k_lo8_ratio = k_gear_ratio = 0.0
        v_hi8_entropy = v_lo8_entropy = v_hi8_ratio = v_lo8_ratio = v_gear_ratio = 0.0

        if bytes_per_elem == 4:
            k_hi16_entropy, k_lo16_entropy, k_hi16_ratio, k_lo16_ratio, k_split16_ratio = split16_ratios_fp32(k_bytes, args.zstd_level)
            v_hi16_entropy, v_lo16_entropy, v_hi16_ratio, v_lo16_ratio, v_split16_ratio = split16_ratios_fp32(v_bytes, args.zstd_level)
        elif bytes_per_elem == 2:
            k_hi8_entropy, k_lo8_entropy, k_hi8_ratio, k_lo8_ratio, k_gear_ratio = gear_ratios_fp16(k_bytes, args.zstd_level)
            v_hi8_entropy, v_lo8_entropy, v_hi8_ratio, v_lo8_ratio, v_gear_ratio = gear_ratios_fp16(v_bytes, args.zstd_level)

        row = {
            "layer_id": meta.get("layer_id"),
            "stage": meta.get("stage"),
            "used_seq_len": meta.get("used_seq_len"),
            "max_seq_len": meta.get("max_seq_len"),
            "kv_heads": meta.get("kv_heads"),
            "head_dim": meta.get("head_dim"),
            "bytes_per_elem": bytes_per_elem,
            "pack_h": meta.get("pack_h"),
            "pack_l": meta.get("pack_l"),
            "flash_upper_kv": meta.get("flash_upper_kv"),
            "key_used_bytes": meta.get("key_used_bytes"),
            "value_used_bytes": meta.get("value_used_bytes"),
            "k_entropy": k_entropy,
            "v_entropy": v_entropy,
            "k_zstd_ratio": k_zstd,
            "v_zstd_ratio": v_zstd,
            "k_hi16_entropy": k_hi16_entropy,
            "k_lo16_entropy": k_lo16_entropy,
            "k_hi16_ratio": k_hi16_ratio,
            "k_lo16_ratio": k_lo16_ratio,
            "k_split16_ratio": k_split16_ratio,
            "v_hi16_entropy": v_hi16_entropy,
            "v_lo16_entropy": v_lo16_entropy,
            "v_hi16_ratio": v_hi16_ratio,
            "v_lo16_ratio": v_lo16_ratio,
            "v_split16_ratio": v_split16_ratio,
            "k_hi8_entropy": k_hi8_entropy,
            "k_lo8_entropy": k_lo8_entropy,
            "k_hi8_ratio": k_hi8_ratio,
            "k_lo8_ratio": k_lo8_ratio,
            "k_gear_ratio": k_gear_ratio,
            "v_hi8_entropy": v_hi8_entropy,
            "v_lo8_entropy": v_lo8_entropy,
            "v_hi8_ratio": v_hi8_ratio,
            "v_lo8_ratio": v_lo8_ratio,
            "v_gear_ratio": v_gear_ratio,
        }
        rows.append(row)

    if not rows:
        print("No rows to write after filtering.", file=sys.stderr)
        return 1

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def avg(values):
        vals = [v for v in values if v is not None and v > 0.0]
        return sum(vals) / len(vals) if vals else 0.0

    k_zstd_avg = avg([r["k_zstd_ratio"] for r in rows])
    v_zstd_avg = avg([r["v_zstd_ratio"] for r in rows])
    k_split16_avg = avg([r["k_split16_ratio"] for r in rows])
    v_split16_avg = avg([r["v_split16_ratio"] for r in rows])
    k_gear_avg = avg([r["k_gear_ratio"] for r in rows])
    v_gear_avg = avg([r["v_gear_ratio"] for r in rows])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Packed KV Dump Summary\n\n")
        f.write(f"- Dumps analyzed: {len(rows)}\n")
        f.write(f"- Avg K zstd ratio: {k_zstd_avg:.3f}\n")
        f.write(f"- Avg V zstd ratio: {v_zstd_avg:.3f}\n")
        if k_split16_avg > 0.0:
            f.write(f"- Avg K split16 zstd ratio: {k_split16_avg:.3f}\n")
        if v_split16_avg > 0.0:
            f.write(f"- Avg V split16 zstd ratio: {v_split16_avg:.3f}\n")
        if k_gear_avg > 0.0:
            f.write(f"- Avg K gear ratio: {k_gear_avg:.3f}\n")
        if v_gear_avg > 0.0:
            f.write(f"- Avg V gear ratio: {v_gear_avg:.3f}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
