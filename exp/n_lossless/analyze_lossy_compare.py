#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterator, List, Tuple

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


def zstd_compress(data: bytes, level: int) -> bytes:
    if zstd is None or not data:
        return b""
    return zstd.ZstdCompressor(level=level).compress(data)


def delta_stream_u8(arr_u8: np.ndarray, seq_len: int, heads: int, head_dim: int) -> bytes:
    if seq_len <= 1:
        return arr_u8.tobytes()
    view = arr_u8.reshape((seq_len, heads, head_dim))
    first = view[0:1]
    delta = (view[1:].astype(np.int16) - view[:-1].astype(np.int16)) & 0xFF
    return np.concatenate([first, delta.astype(np.uint8)], axis=0).tobytes()


def delta_stream_u16(arr_u16: np.ndarray, seq_len: int, heads: int, head_dim: int) -> bytes:
    if seq_len <= 1:
        return arr_u16.tobytes()
    view = arr_u16.reshape((seq_len, heads, head_dim))
    first = view[0:1]
    delta = (view[1:].astype(np.int32) - view[:-1].astype(np.int32)) & 0xFFFF
    return np.concatenate([first, delta.astype(np.uint16)], axis=0).tobytes()


def compress_fp16_gear(data: bytes, seq_len: int, heads: int, head_dim: int, level: int, delta: bool) -> int:
    u16 = np.frombuffer(data, dtype=np.uint16)
    hi = (u16 >> 8).astype(np.uint8)
    lo = (u16 & 0xFF).astype(np.uint8)
    hi_bytes = delta_stream_u8(hi, seq_len, heads, head_dim) if delta else hi.tobytes()
    lo_bytes = lo.tobytes()
    hi_c = zstd_compress(hi_bytes, level)
    lo_c = zstd_compress(lo_bytes, level)
    if not hi_c or not lo_c:
        return 0
    return len(hi_c) + len(lo_c)


def compress_fp32_split16(data: bytes, seq_len: int, heads: int, head_dim: int, level: int, delta: bool) -> int:
    u32 = np.frombuffer(data, dtype=np.uint32)
    hi16 = (u32 >> 16).astype(np.uint16)
    lo16 = (u32 & 0xFFFF).astype(np.uint16)
    hi_bytes = delta_stream_u16(hi16, seq_len, heads, head_dim) if delta else hi16.tobytes()
    lo_bytes = lo16.tobytes()
    hi_c = zstd_compress(hi_bytes, level)
    lo_c = zstd_compress(lo_bytes, level)
    if not hi_c or not lo_c:
        return 0
    return len(hi_c) + len(lo_c)


def compute_error_metrics(orig: np.ndarray, lossy: np.ndarray) -> Dict[str, float]:
    a = orig.astype(np.float32)
    b = lossy.astype(np.float32)
    if a.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0, "cosine": 0.0}
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "max_abs": max_abs, "cosine": cosine}


def clip_by_percentile(x: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    xf = x.astype(np.float32, copy=False)
    finite = np.isfinite(xf)
    if not finite.any():
        return x
    xf = xf[finite]
    low = float(np.percentile(xf, p_low))
    high = float(np.percentile(xf, p_high))
    return np.clip(x, low, high)


def clip_by_sigma(x: np.ndarray, k: float) -> np.ndarray:
    xf = x.astype(np.float32, copy=False)
    finite = np.isfinite(xf)
    if not finite.any():
        return x
    xf = xf[finite]
    mean = float(np.mean(xf))
    std = float(np.std(xf))
    return np.clip(x, mean - k * std, mean + k * std)


def quantize_uniform(x: np.ndarray, bits: int, per_axis: int = None) -> np.ndarray:
    # symmetric quantization
    if per_axis is None:
        xf = x.astype(np.float32, copy=False)
        max_abs = float(np.max(np.abs(xf))) if x.size else 0.0
        if max_abs == 0:
            return x.copy()
        qmax = (1 << (bits - 1)) - 1
        scale = max_abs / qmax
        q = np.round(xf / scale).clip(-qmax, qmax)
        return (q * scale).astype(x.dtype)

    # per-axis: quantize along given axis (e.g., head_dim or head)
    x_q = np.empty_like(x)
    qmax = (1 << (bits - 1)) - 1
    for idx in range(x.shape[per_axis]):
        slicer = [slice(None)] * x.ndim
        slicer[per_axis] = idx
        sub = x[tuple(slicer)]
        sub_f = sub.astype(np.float32, copy=False)
        max_abs = float(np.max(np.abs(sub_f))) if sub.size else 0.0
        if max_abs == 0:
            x_q[tuple(slicer)] = sub
            continue
        scale = max_abs / qmax
        q = np.round(sub_f / scale).clip(-qmax, qmax)
        x_q[tuple(slicer)] = (q * scale).astype(x.dtype)
    return x_q


def should_keep_stage(meta_stage: str, target_stage: str) -> bool:
    if target_stage == "both":
        return True
    return meta_stage == target_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare lossy strategies for KV compression.")
    parser.add_argument("--dump-dir", required=True, help="Root dump directory.")
    parser.add_argument("--out-dir", default="exp/n_lossless/out", help="Output directory.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--min-seq-len", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--lossless-first-n", type=int, default=2)
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--compress-mode", choices=["gear", "gear-delta", "zstd"], default="gear-delta")
    parser.add_argument("--bits", type=int, default=8, help="Quantization bits for lossy layers.")
    parser.add_argument("--clip-percentiles", default="1,99", help="Percentile clip range, e.g., 1,99")
    parser.add_argument("--clip-sigma", type=float, default=3.0, help="Sigma clip (mean ± k*std)")
    args = parser.parse_args()

    if zstd is None:
        print("Missing dependency: zstandard. Install with `pip install zstandard`.", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "lossy_compare.csv")
    summary_path = os.path.join(args.out_dir, "lossy_compare_summary.md")

    p_low, p_high = [float(x.strip()) for x in args.clip_percentiles.split(",")]

    rows = []
    totals = {}

    meta_files = list(iter_meta_files(args.dump_dir))
    if not meta_files:
        print(f"No meta_*.json found under {args.dump_dir}", file=sys.stderr)
        return 1

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

        layer_id = int(meta.get("layer_id", -1))
        kv_heads = int(meta.get("kv_heads", 0))
        head_dim = int(meta.get("head_dim", 0))
        bytes_per_elem = int(meta.get("bytes_per_elem", 0))

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta.get("k_file", ""))
        v_path = os.path.join(dirpath, meta.get("v_file", ""))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            continue

        with open(k_path, "rb") as f:
            k_bytes = f.read()
        with open(v_path, "rb") as f:
            v_bytes = f.read()

        dtype = np.float16 if bytes_per_elem == 2 else np.float32
        k = np.frombuffer(k_bytes, dtype=dtype)
        v = np.frombuffer(v_bytes, dtype=dtype)

        is_lossless = layer_id < args.lossless_first_n
        if is_lossless:
            continue

        # Lossy methods
        methods = [
            ("clip_percentile", clip_by_percentile(k, p_low, p_high)),
            ("clip_sigma", clip_by_sigma(k, args.clip_sigma)),
        ]

        for name, k_clip in methods:
            k_q = quantize_uniform(k_clip, args.bits)
            v_q = quantize_uniform(v, args.bits)
            k_err = compute_error_metrics(k, k_q)
            v_err = compute_error_metrics(v, v_q)

            if bytes_per_elem == 2:
                k_comp = compress_fp16_gear(k_q.tobytes(), seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                v_comp = compress_fp16_gear(v_q.tobytes(), seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
            else:
                k_comp = compress_fp32_split16(k_q.tobytes(), seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                v_comp = compress_fp32_split16(v_q.tobytes(), seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")

            k_raw = k.nbytes
            v_raw = v.nbytes
            k_ratio = k_raw / k_comp if k_comp else 0.0
            v_ratio = v_raw / v_comp if v_comp else 0.0

            rows.append({
                "layer_id": layer_id,
                "stage": stage,
                "seq_len": seq_len,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "bytes_per_elem": bytes_per_elem,
                "method": name,
                "bits": args.bits,
                "k_ratio": k_ratio,
                "v_ratio": v_ratio,
                "k_mae": k_err["mae"],
                "k_cosine": k_err["cosine"],
                "v_mae": v_err["mae"],
                "v_cosine": v_err["cosine"],
            })

            key = (name, args.bits)
            if key not in totals:
                totals[key] = {"k_raw": 0.0, "k_comp": 0.0, "v_raw": 0.0, "v_comp": 0.0,
                               "k_mae_sum": 0.0, "k_cos_sum": 0.0, "v_mae_sum": 0.0, "v_cos_sum": 0.0, "elems": 0}
            t = totals[key]
            t["k_raw"] += k_raw
            t["k_comp"] += k_comp
            t["v_raw"] += v_raw
            t["v_comp"] += v_comp
            t["k_mae_sum"] += k_err["mae"] * k.size
            t["k_cos_sum"] += k_err["cosine"] * k.size
            t["v_mae_sum"] += v_err["mae"] * v.size
            t["v_cos_sum"] += v_err["cosine"] * v.size
            t["elems"] += k.size

    if not rows:
        print("No lossy rows after filtering.", file=sys.stderr)
        return 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Lossy Compare Summary\n\n")
        for (name, bits), t in totals.items():
            k_ratio = t["k_raw"] / t["k_comp"] if t["k_comp"] > 0 else 0.0
            v_ratio = t["v_raw"] / t["v_comp"] if t["v_comp"] > 0 else 0.0
            k_mae = t["k_mae_sum"] / t["elems"] if t["elems"] else 0.0
            v_mae = t["v_mae_sum"] / t["elems"] if t["elems"] else 0.0
            k_cos = t["k_cos_sum"] / t["elems"] if t["elems"] else 0.0
            v_cos = t["v_cos_sum"] / t["elems"] if t["elems"] else 0.0
            f.write(f"## {name} (bits={bits})\n")
            f.write(f"- Weighted K ratio: {k_ratio:.3f}\n")
            f.write(f"- Weighted V ratio: {v_ratio:.3f}\n")
            f.write(f"- K MAE: {k_mae:.6f}, K Cosine: {k_cos:.6f}\n")
            f.write(f"- V MAE: {v_mae:.6f}, V Cosine: {v_cos:.6f}\n\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
