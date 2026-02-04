#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterator, Tuple

import numpy as np

try:
    import zstandard as zstd
except Exception:
    zstd = None


def iter_meta_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("meta_") and name.endswith(".json"):
                yield os.path.join(dirpath, name)


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


def zstd_compress(data: bytes, level: int) -> bytes:
    if zstd is None:
        return b""
    if not data:
        return b""
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(data)


def zstd_ratio(data: bytes, level: int) -> float:
    if not data or zstd is None:
        return 0.0
    compressed = zstd_compress(data, level)
    if not compressed:
        return 0.0
    return len(data) / len(compressed)


def delta_stream_u8(arr_u8: np.ndarray, seq_len: int, heads: int, head_dim: int) -> bytes:
    if seq_len <= 1:
        return arr_u8.tobytes()
    view = arr_u8.reshape((seq_len, heads, head_dim))
    first = view[0:1]
    delta = (view[1:].astype(np.int16) - view[:-1].astype(np.int16)) & 0xFF
    delta = delta.astype(np.uint8)
    return np.concatenate([first, delta], axis=0).tobytes()


def delta_stream_u16(arr_u16: np.ndarray, seq_len: int, heads: int, head_dim: int) -> bytes:
    if seq_len <= 1:
        return arr_u16.tobytes()
    view = arr_u16.reshape((seq_len, heads, head_dim))
    first = view[0:1]
    delta = (view[1:].astype(np.int32) - view[:-1].astype(np.int32)) & 0xFFFF
    delta = delta.astype(np.uint16)
    return np.concatenate([first, delta], axis=0).tobytes()


def fp16_metrics(data: bytes, seq_len: int, heads: int, head_dim: int, level: int) -> Dict[str, float]:
    u16 = np.frombuffer(data, dtype=np.uint16)
    hi = (u16 >> 8).astype(np.uint8)
    lo = (u16 & 0xFF).astype(np.uint8)
    hi_bytes = hi.tobytes()
    lo_bytes = lo.tobytes()
    hi_entropy = entropy_bytes(hi_bytes)
    lo_entropy = entropy_bytes(lo_bytes)
    hi_c = zstd_compress(hi_bytes, level)
    lo_c = zstd_compress(lo_bytes, level)
    hi_ratio = len(hi_bytes) / len(hi_c) if hi_c else 0.0
    lo_ratio = len(lo_bytes) / len(lo_c) if lo_c else 0.0
    gear_ratio = len(data) / (len(hi_c) + len(lo_c)) if hi_c and lo_c else 0.0

    hi_delta_bytes = delta_stream_u8(hi, seq_len, heads, head_dim)
    hi_delta_c = zstd_compress(hi_delta_bytes, level)
    hi_delta_ratio = len(hi_bytes) / len(hi_delta_c) if hi_delta_c else 0.0
    gear_delta_ratio = len(data) / (len(hi_delta_c) + len(lo_c)) if hi_delta_c and lo_c else 0.0

    return {
        "hi_entropy": hi_entropy,
        "lo_entropy": lo_entropy,
        "hi_ratio": hi_ratio,
        "lo_ratio": lo_ratio,
        "gear_ratio": gear_ratio,
        "hi_delta_ratio": hi_delta_ratio,
        "gear_delta_ratio": gear_delta_ratio,
    }


def fp32_split16_metrics(data: bytes, seq_len: int, heads: int, head_dim: int, level: int) -> Dict[str, float]:
    u32 = np.frombuffer(data, dtype=np.uint32)
    hi16 = (u32 >> 16).astype(np.uint16)
    lo16 = (u32 & 0xFFFF).astype(np.uint16)
    hi_bytes = hi16.tobytes()
    lo_bytes = lo16.tobytes()
    hi_entropy = entropy_bytes(hi_bytes)
    lo_entropy = entropy_bytes(lo_bytes)
    hi_c = zstd_compress(hi_bytes, level)
    lo_c = zstd_compress(lo_bytes, level)
    hi_ratio = len(hi_bytes) / len(hi_c) if hi_c else 0.0
    lo_ratio = len(lo_bytes) / len(lo_c) if lo_c else 0.0
    split_ratio = len(data) / (len(hi_c) + len(lo_c)) if hi_c and lo_c else 0.0

    hi_delta_bytes = delta_stream_u16(hi16, seq_len, heads, head_dim)
    hi_delta_c = zstd_compress(hi_delta_bytes, level)
    hi_delta_ratio = len(hi_bytes) / len(hi_delta_c) if hi_delta_c else 0.0
    split_delta_ratio = len(data) / (len(hi_delta_c) + len(lo_c)) if hi_delta_c and lo_c else 0.0

    return {
        "hi16_entropy": hi_entropy,
        "lo16_entropy": lo_entropy,
        "hi16_ratio": hi_ratio,
        "lo16_ratio": lo_ratio,
        "split16_ratio": split_ratio,
        "hi16_delta_ratio": hi_delta_ratio,
        "split16_delta_ratio": split_delta_ratio,
    }


def avg(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def should_keep_stage(meta_stage: str, target_stage: str) -> bool:
    if target_stage == "both":
        return True
    return meta_stage == target_stage


def build_layer_rows(args) -> Tuple[list, list]:
    rows = []
    head_rows = []
    meta_files = list(iter_meta_files(args.dump_dir))
    if not meta_files:
        print(f"No meta_*.json found under {args.dump_dir}", file=sys.stderr)
        return rows, head_rows

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

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta.get("k_file", ""))
        v_path = os.path.join(dirpath, meta.get("v_file", ""))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            continue

        with open(k_path, "rb") as f:
            k_bytes = f.read()
        with open(v_path, "rb") as f:
            v_bytes = f.read()

        bytes_per_elem = int(meta.get("bytes_per_elem", 0))
        kv_heads = int(meta.get("kv_heads", 0))
        head_dim = int(meta.get("head_dim", 0))

        k_entropy = entropy_bytes(k_bytes)
        v_entropy = entropy_bytes(v_bytes)
        k_zstd = zstd_ratio(k_bytes, args.zstd_level)
        v_zstd = zstd_ratio(v_bytes, args.zstd_level)

        k_fp16 = {
            "hi_entropy": 0.0,
            "lo_entropy": 0.0,
            "hi_ratio": 0.0,
            "lo_ratio": 0.0,
            "gear_ratio": 0.0,
            "hi_delta_ratio": 0.0,
            "gear_delta_ratio": 0.0,
        }
        v_fp16 = dict(k_fp16)
        k_fp32 = {
            "hi16_entropy": 0.0,
            "lo16_entropy": 0.0,
            "hi16_ratio": 0.0,
            "lo16_ratio": 0.0,
            "split16_ratio": 0.0,
            "hi16_delta_ratio": 0.0,
            "split16_delta_ratio": 0.0,
        }
        v_fp32 = dict(k_fp32)

        if bytes_per_elem == 2:
            k_fp16 = fp16_metrics(k_bytes, seq_len, kv_heads, head_dim, args.zstd_level)
            v_fp16 = fp16_metrics(v_bytes, seq_len, kv_heads, head_dim, args.zstd_level)
        elif bytes_per_elem == 4:
            k_fp32 = fp32_split16_metrics(k_bytes, seq_len, kv_heads, head_dim, args.zstd_level)
            v_fp32 = fp32_split16_metrics(v_bytes, seq_len, kv_heads, head_dim, args.zstd_level)

        row = {
            "layer_id": meta.get("layer_id"),
            "stage": stage,
            "seq_start": meta.get("seq_start"),
            "seq_len": seq_len,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "bytes_per_elem": bytes_per_elem,
            "k_entropy": k_entropy,
            "v_entropy": v_entropy,
            "k_zstd_ratio": k_zstd,
            "v_zstd_ratio": v_zstd,
            "k_hi_entropy": k_fp16["hi_entropy"],
            "k_lo_entropy": k_fp16["lo_entropy"],
            "k_hi_ratio": k_fp16["hi_ratio"],
            "k_lo_ratio": k_fp16["lo_ratio"],
            "k_gear_ratio": k_fp16["gear_ratio"],
            "k_hi_delta_ratio": k_fp16["hi_delta_ratio"],
            "k_gear_delta_ratio": k_fp16["gear_delta_ratio"],
            "v_hi_entropy": v_fp16["hi_entropy"],
            "v_lo_entropy": v_fp16["lo_entropy"],
            "v_hi_ratio": v_fp16["hi_ratio"],
            "v_lo_ratio": v_fp16["lo_ratio"],
            "v_gear_ratio": v_fp16["gear_ratio"],
            "v_hi_delta_ratio": v_fp16["hi_delta_ratio"],
            "v_gear_delta_ratio": v_fp16["gear_delta_ratio"],
            "k_hi16_entropy": k_fp32["hi16_entropy"],
            "k_lo16_entropy": k_fp32["lo16_entropy"],
            "k_hi16_ratio": k_fp32["hi16_ratio"],
            "k_lo16_ratio": k_fp32["lo16_ratio"],
            "k_split16_ratio": k_fp32["split16_ratio"],
            "k_hi16_delta_ratio": k_fp32["hi16_delta_ratio"],
            "k_split16_delta_ratio": k_fp32["split16_delta_ratio"],
            "v_hi16_entropy": v_fp32["hi16_entropy"],
            "v_lo16_entropy": v_fp32["lo16_entropy"],
            "v_hi16_ratio": v_fp32["hi16_ratio"],
            "v_lo16_ratio": v_fp32["lo16_ratio"],
            "v_split16_ratio": v_fp32["split16_ratio"],
            "v_hi16_delta_ratio": v_fp32["hi16_delta_ratio"],
            "v_split16_delta_ratio": v_fp32["split16_delta_ratio"],
        }
        rows.append(row)

        if not args.skip_head_metrics and kv_heads > 0 and head_dim > 0:
            if bytes_per_elem == 2:
                k_view = np.frombuffer(k_bytes, dtype=np.uint16).reshape((seq_len, kv_heads, head_dim))
                v_view = np.frombuffer(v_bytes, dtype=np.uint16).reshape((seq_len, kv_heads, head_dim))
                for h in range(kv_heads):
                    k_head = k_view[:, h, :].tobytes()
                    v_head = v_view[:, h, :].tobytes()
                    k_fp16_h = fp16_metrics(k_head, seq_len, 1, head_dim, args.zstd_level)
                    v_fp16_h = fp16_metrics(v_head, seq_len, 1, head_dim, args.zstd_level)
                    head_rows.append({
                        "layer_id": meta.get("layer_id"),
                        "head_id": h,
                        "stage": stage,
                        "seq_start": meta.get("seq_start"),
                        "seq_len": seq_len,
                        "head_dim": head_dim,
                        "bytes_per_elem": bytes_per_elem,
                        "k_entropy": entropy_bytes(k_head),
                        "v_entropy": entropy_bytes(v_head),
                        "k_zstd_ratio": zstd_ratio(k_head, args.zstd_level),
                        "v_zstd_ratio": zstd_ratio(v_head, args.zstd_level),
                        "k_hi_entropy": k_fp16_h["hi_entropy"],
                        "k_lo_entropy": k_fp16_h["lo_entropy"],
                        "k_hi_ratio": k_fp16_h["hi_ratio"],
                        "k_lo_ratio": k_fp16_h["lo_ratio"],
                        "k_gear_ratio": k_fp16_h["gear_ratio"],
                        "k_hi_delta_ratio": k_fp16_h["hi_delta_ratio"],
                        "k_gear_delta_ratio": k_fp16_h["gear_delta_ratio"],
                        "v_hi_entropy": v_fp16_h["hi_entropy"],
                        "v_lo_entropy": v_fp16_h["lo_entropy"],
                        "v_hi_ratio": v_fp16_h["hi_ratio"],
                        "v_lo_ratio": v_fp16_h["lo_ratio"],
                        "v_gear_ratio": v_fp16_h["gear_ratio"],
                        "v_hi_delta_ratio": v_fp16_h["hi_delta_ratio"],
                        "v_gear_delta_ratio": v_fp16_h["gear_delta_ratio"],
                    })
            elif bytes_per_elem == 4:
                k_view = np.frombuffer(k_bytes, dtype=np.uint32).reshape((seq_len, kv_heads, head_dim))
                v_view = np.frombuffer(v_bytes, dtype=np.uint32).reshape((seq_len, kv_heads, head_dim))
                for h in range(kv_heads):
                    k_head = k_view[:, h, :].tobytes()
                    v_head = v_view[:, h, :].tobytes()
                    k_fp32_h = fp32_split16_metrics(k_head, seq_len, 1, head_dim, args.zstd_level)
                    v_fp32_h = fp32_split16_metrics(v_head, seq_len, 1, head_dim, args.zstd_level)
                    head_rows.append({
                        "layer_id": meta.get("layer_id"),
                        "head_id": h,
                        "stage": stage,
                        "seq_start": meta.get("seq_start"),
                        "seq_len": seq_len,
                        "head_dim": head_dim,
                        "bytes_per_elem": bytes_per_elem,
                        "k_entropy": entropy_bytes(k_head),
                        "v_entropy": entropy_bytes(v_head),
                        "k_zstd_ratio": zstd_ratio(k_head, args.zstd_level),
                        "v_zstd_ratio": zstd_ratio(v_head, args.zstd_level),
                        "k_hi16_entropy": k_fp32_h["hi16_entropy"],
                        "k_lo16_entropy": k_fp32_h["lo16_entropy"],
                        "k_hi16_ratio": k_fp32_h["hi16_ratio"],
                        "k_lo16_ratio": k_fp32_h["lo16_ratio"],
                        "k_split16_ratio": k_fp32_h["split16_ratio"],
                        "k_hi16_delta_ratio": k_fp32_h["hi16_delta_ratio"],
                        "k_split16_delta_ratio": k_fp32_h["split16_delta_ratio"],
                        "v_hi16_entropy": v_fp32_h["hi16_entropy"],
                        "v_lo16_entropy": v_fp32_h["lo16_entropy"],
                        "v_hi16_ratio": v_fp32_h["hi16_ratio"],
                        "v_lo16_ratio": v_fp32_h["lo16_ratio"],
                        "v_split16_ratio": v_fp32_h["split16_ratio"],
                        "v_hi16_delta_ratio": v_fp32_h["hi16_delta_ratio"],
                        "v_split16_delta_ratio": v_fp32_h["split16_delta_ratio"],
                    })

    return rows, head_rows


def write_csv(path: str, rows: list) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: str, rows: list) -> None:
    if not rows:
        return

    k_zstd = avg([r["k_zstd_ratio"] for r in rows])
    v_zstd = avg([r["v_zstd_ratio"] for r in rows])
    k_gear = avg([r["k_gear_ratio"] for r in rows if r["k_gear_ratio"] > 0.0])
    v_gear = avg([r["v_gear_ratio"] for r in rows if r["v_gear_ratio"] > 0.0])
    k_gear_delta = avg([r["k_gear_delta_ratio"] for r in rows if r["k_gear_delta_ratio"] > 0.0])
    v_gear_delta = avg([r["v_gear_delta_ratio"] for r in rows if r["v_gear_delta_ratio"] > 0.0])
    k_split16 = avg([r["k_split16_ratio"] for r in rows if r["k_split16_ratio"] > 0.0])
    v_split16 = avg([r["v_split16_ratio"] for r in rows if r["v_split16_ratio"] > 0.0])
    k_split16_delta = avg([r["k_split16_delta_ratio"] for r in rows if r["k_split16_delta_ratio"] > 0.0])
    v_split16_delta = avg([r["v_split16_delta_ratio"] for r in rows if r["v_split16_delta_ratio"] > 0.0])

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Gear FP16 Summary\n\n")
        f.write(f"- Dumps analyzed: {len(rows)}\n")
        f.write(f"- Avg K zstd ratio: {k_zstd:.3f}\n")
        f.write(f"- Avg V zstd ratio: {v_zstd:.3f}\n")
        if k_gear > 0.0:
            f.write(f"- Avg K gear ratio: {k_gear:.3f}\n")
        if v_gear > 0.0:
            f.write(f"- Avg V gear ratio: {v_gear:.3f}\n")
        if k_gear_delta > 0.0:
            f.write(f"- Avg K gear+delta ratio: {k_gear_delta:.3f}\n")
        if v_gear_delta > 0.0:
            f.write(f"- Avg V gear+delta ratio: {v_gear_delta:.3f}\n")
        if k_split16 > 0.0:
            f.write(f"- Avg K split16 ratio: {k_split16:.3f}\n")
        if v_split16 > 0.0:
            f.write(f"- Avg V split16 ratio: {v_split16:.3f}\n")
        if k_split16_delta > 0.0:
            f.write(f"- Avg K split16+delta ratio: {k_split16_delta:.3f}\n")
        if v_split16_delta > 0.0:
            f.write(f"- Avg V split16+delta ratio: {v_split16_delta:.3f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze FP16 KV dumps with gear and delta compression metrics.")
    parser.add_argument("--dump-dir", required=True, help="Root dump directory containing run_* subdirs.")
    parser.add_argument("--out-dir", default="exp/gear_fp16/out", help="Output directory for csv/summary.")
    parser.add_argument("--zstd-level", type=int, default=3, help="Zstd compression level.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--min-seq-len", type=int, default=0, help="Skip entries with seq_len smaller than this.")
    parser.add_argument("--max-seq-len", type=int, default=0, help="Skip entries with seq_len larger than this (0 = no limit).")
    parser.add_argument("--skip-head-metrics", action="store_true", help="Skip per-head metrics for speed.")
    args = parser.parse_args()

    if zstd is None:
        print("Missing dependency: zstandard. Install with `pip install zstandard`.", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    layer_csv = os.path.join(args.out_dir, "layer_metrics.csv")
    head_csv = os.path.join(args.out_dir, "head_metrics.csv")
    summary_path = os.path.join(args.out_dir, "summary.md")

    rows, head_rows = build_layer_rows(args)
    if not rows:
        print("No rows to write after filtering.", file=sys.stderr)
        return 1

    write_csv(layer_csv, rows)
    if head_rows:
        write_csv(head_csv, head_rows)
    write_summary(summary_path, rows)

    print(f"Wrote {layer_csv}")
    if head_rows:
        print(f"Wrote {head_csv}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
