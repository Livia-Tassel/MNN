#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterator

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


def truncate_fp16(data: bytes, keep_mantissa_bits: int) -> bytes:
    keep = max(0, min(10, keep_mantissa_bits))
    u16 = np.frombuffer(data, dtype=np.uint16)
    mask = 0xFC00
    if keep > 0:
        mask |= ((1 << keep) - 1) << (10 - keep)
    return (u16 & mask).astype(np.uint16).tobytes()


def truncate_fp32(data: bytes, keep_mantissa_bits: int) -> bytes:
    keep = max(0, min(23, keep_mantissa_bits))
    u32 = np.frombuffer(data, dtype=np.uint32)
    mask = 0xFF800000
    if keep > 0:
        mask |= ((1 << keep) - 1) << (23 - keep)
    return (u32 & mask).astype(np.uint32).tobytes()


def quantize_uniform_array(x: np.ndarray, bits: int) -> np.ndarray:
    xf = x.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(xf))) if xf.size else 0.0
    if max_abs == 0.0:
        return x.copy()
    qmax = (1 << (bits - 1)) - 1
    scale = max_abs / qmax
    q = np.round(xf / scale).clip(-qmax, qmax)
    return (q * scale).astype(x.dtype)


def quantize_uniform_bytes(data: bytes, dtype: np.dtype, bits: int) -> bytes:
    arr = np.frombuffer(data, dtype=dtype)
    q = quantize_uniform_array(arr, bits)
    return q.tobytes()


def compute_error_metrics(orig: bytes, lossy: bytes, dtype: np.dtype) -> Dict[str, float]:
    a = np.frombuffer(orig, dtype=dtype).astype(np.float32)
    b = np.frombuffer(lossy, dtype=dtype).astype(np.float32)
    if a.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0, "cosine": 0.0}
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "max_abs": max_abs, "cosine": cosine}


def should_keep_stage(meta_stage: str, target_stage: str) -> bool:
    if target_stage == "both":
        return True
    return meta_stage == target_stage


def ratio(raw: float, comp: float) -> float:
    return raw / comp if comp > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Mixed lossless/lossy KV compression analysis.")
    parser.add_argument("--dump-dir", required=True, help="Root dump directory.")
    parser.add_argument("--out-dir", default="exp/n_lossless/out", help="Output directory.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--min-seq-len", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--lossless-first-n", type=int, default=2, help="First N layers to use lossless compression.")
    parser.add_argument("--lossy-method", choices=["truncate", "quant"], default="truncate", help="Lossy method for tail layers.")
    parser.add_argument("--lossy-mantissa-bits", type=int, default=6, help="Mantissa bits to keep (truncate method).")
    parser.add_argument("--lossy-bits", type=int, default=8, help="Quantization bits (quant method).")
    parser.add_argument("--compress-mode", choices=["gear", "gear-delta", "zstd"], default="gear-delta")
    parser.add_argument("--zstd-level", type=int, default=3)
    args = parser.parse_args()

    if zstd is None:
        print("Missing dependency: zstandard. Install with `pip install zstandard`.", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    layer_csv = os.path.join(args.out_dir, "mixed_layer_metrics.csv")
    lossy_csv = os.path.join(args.out_dir, "lossy_error_metrics.csv")
    summary_path = os.path.join(args.out_dir, "mixed_summary.md")

    rows = []
    lossy_rows = []

    totals = {
        "k_raw": 0.0,
        "v_raw": 0.0,
        "k_comp": 0.0,
        "v_comp": 0.0,
        "k_lossless_raw": 0.0,
        "k_lossless_comp": 0.0,
        "v_lossless_raw": 0.0,
        "v_lossless_comp": 0.0,
        "k_lossy_raw": 0.0,
        "k_lossy_comp": 0.0,
        "v_lossy_raw": 0.0,
        "v_lossy_comp": 0.0,
    }
    error_acc = {
        "k": {"sum_mae": 0.0, "sum_rmse": 0.0, "sum_max": 0.0, "sum_cos": 0.0, "count": 0, "elems": 0},
        "v": {"sum_mae": 0.0, "sum_rmse": 0.0, "sum_max": 0.0, "sum_cos": 0.0, "count": 0, "elems": 0},
    }

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

        is_lossless = layer_id < args.lossless_first_n
        dtype = np.float16 if bytes_per_elem == 2 else np.float32

        if is_lossless:
            if args.compress_mode == "zstd":
                k_comp = len(zstd_compress(k_bytes, args.zstd_level))
                v_comp = len(zstd_compress(v_bytes, args.zstd_level))
            elif bytes_per_elem == 2:
                k_comp = compress_fp16_gear(k_bytes, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                v_comp = compress_fp16_gear(v_bytes, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
            else:
                k_comp = compress_fp32_split16(k_bytes, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                v_comp = compress_fp32_split16(v_bytes, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
        else:
            if args.lossy_method == "truncate":
                if bytes_per_elem == 2:
                    k_lossy = truncate_fp16(k_bytes, args.lossy_mantissa_bits)
                    v_lossy = truncate_fp16(v_bytes, args.lossy_mantissa_bits)
                    k_comp = compress_fp16_gear(k_lossy, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                    v_comp = compress_fp16_gear(v_lossy, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                else:
                    k_lossy = truncate_fp32(k_bytes, args.lossy_mantissa_bits)
                    v_lossy = truncate_fp32(v_bytes, args.lossy_mantissa_bits)
                    k_comp = compress_fp32_split16(k_lossy, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                    v_comp = compress_fp32_split16(v_lossy, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
            else:
                k_lossy = quantize_uniform_bytes(k_bytes, dtype, args.lossy_bits)
                v_lossy = quantize_uniform_bytes(v_bytes, dtype, args.lossy_bits)
                if bytes_per_elem == 2:
                    k_comp = compress_fp16_gear(k_lossy, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                    v_comp = compress_fp16_gear(v_lossy, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                else:
                    k_comp = compress_fp32_split16(k_lossy, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")
                    v_comp = compress_fp32_split16(v_lossy, seq_len, kv_heads, head_dim, args.zstd_level, args.compress_mode == "gear-delta")

            k_err = compute_error_metrics(k_bytes, k_lossy, dtype)
            v_err = compute_error_metrics(v_bytes, v_lossy, dtype)
            lossy_rows.append({
                "layer_id": layer_id,
                "stage": stage,
                "seq_len": seq_len,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "bytes_per_elem": bytes_per_elem,
                "k_mae": k_err["mae"],
                "k_rmse": k_err["rmse"],
                "k_max_abs": k_err["max_abs"],
                "k_cosine": k_err["cosine"],
                "v_mae": v_err["mae"],
                "v_rmse": v_err["rmse"],
                "v_max_abs": v_err["max_abs"],
                "v_cosine": v_err["cosine"],
            })
            elems = seq_len * kv_heads * head_dim
            error_acc["k"]["sum_mae"] += k_err["mae"] * elems
            error_acc["k"]["sum_rmse"] += k_err["rmse"] * elems
            error_acc["k"]["sum_max"] += k_err["max_abs"]
            error_acc["k"]["sum_cos"] += k_err["cosine"] * elems
            error_acc["k"]["count"] += 1
            error_acc["k"]["elems"] += elems

            error_acc["v"]["sum_mae"] += v_err["mae"] * elems
            error_acc["v"]["sum_rmse"] += v_err["rmse"] * elems
            error_acc["v"]["sum_max"] += v_err["max_abs"]
            error_acc["v"]["sum_cos"] += v_err["cosine"] * elems
            error_acc["v"]["count"] += 1
            error_acc["v"]["elems"] += elems

        k_raw = len(k_bytes)
        v_raw = len(v_bytes)
        k_ratio = ratio(k_raw, k_comp)
        v_ratio = ratio(v_raw, v_comp)

        rows.append({
            "layer_id": layer_id,
            "stage": stage,
            "seq_len": seq_len,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "bytes_per_elem": bytes_per_elem,
            "lossless": is_lossless,
            "k_ratio": k_ratio,
            "v_ratio": v_ratio,
        })

        totals["k_raw"] += k_raw
        totals["v_raw"] += v_raw
        totals["k_comp"] += k_comp
        totals["v_comp"] += v_comp

        if is_lossless:
            totals["k_lossless_raw"] += k_raw
            totals["k_lossless_comp"] += k_comp
            totals["v_lossless_raw"] += v_raw
            totals["v_lossless_comp"] += v_comp
        else:
            totals["k_lossy_raw"] += k_raw
            totals["k_lossy_comp"] += k_comp
            totals["v_lossy_raw"] += v_raw
            totals["v_lossy_comp"] += v_comp

    if not rows:
        print("No rows after filtering.", file=sys.stderr)
        return 1

    with open(layer_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if lossy_rows:
        with open(lossy_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(lossy_rows[0].keys()))
            writer.writeheader()
            writer.writerows(lossy_rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Mixed Compression Summary\n\n")
        f.write(f"- Lossless first N layers: {args.lossless_first_n}\n")
        f.write(f"- Lossy method: {args.lossy_method}\n")
        if args.lossy_method == "truncate":
            f.write(f"- Lossy mantissa bits: {args.lossy_mantissa_bits}\n")
        else:
            f.write(f"- Lossy quant bits: {args.lossy_bits}\n")
        f.write(f"- Compress mode: {args.compress_mode}\n")
        f.write(f"- Dumps analyzed: {len(rows)}\n\n")
        f.write("## Weighted Ratios (overall)\n")
        f.write(f"- Weighted K ratio: {ratio(totals['k_raw'], totals['k_comp']):.3f}\n")
        f.write(f"- Weighted V ratio: {ratio(totals['v_raw'], totals['v_comp']):.3f}\n")
        f.write(f"- Weighted KV ratio: {ratio(totals['k_raw'] + totals['v_raw'], totals['k_comp'] + totals['v_comp']):.3f}\n\n")
        f.write("## Weighted Ratios (lossless layers)\n")
        f.write(f"- K: {ratio(totals['k_lossless_raw'], totals['k_lossless_comp']):.3f}\n")
        f.write(f"- V: {ratio(totals['v_lossless_raw'], totals['v_lossless_comp']):.3f}\n\n")
        f.write("## Weighted Ratios (lossy layers)\n")
        f.write(f"- K: {ratio(totals['k_lossy_raw'], totals['k_lossy_comp']):.3f}\n")
        f.write(f"- V: {ratio(totals['v_lossy_raw'], totals['v_lossy_comp']):.3f}\n\n")
        if error_acc["k"]["count"] > 0:
            f.write("## Lossy Error Metrics (weighted by elements)\n")
            k_elems = max(1, error_acc["k"]["elems"])
            v_elems = max(1, error_acc["v"]["elems"])
            f.write(f"- K MAE: {error_acc['k']['sum_mae'] / k_elems:.6f}\n")
            f.write(f"- K RMSE: {error_acc['k']['sum_rmse'] / k_elems:.6f}\n")
            f.write(f"- K max_abs (avg): {error_acc['k']['sum_max'] / max(1, error_acc['k']['count']):.6f}\n")
            f.write(f"- K cosine: {error_acc['k']['sum_cos'] / k_elems:.6f}\n")
            f.write(f"- V MAE: {error_acc['v']['sum_mae'] / v_elems:.6f}\n")
            f.write(f"- V RMSE: {error_acc['v']['sum_rmse'] / v_elems:.6f}\n")
            f.write(f"- V max_abs (avg): {error_acc['v']['sum_max'] / max(1, error_acc['v']['count']):.6f}\n")
            f.write(f"- V cosine: {error_acc['v']['sum_cos'] / v_elems:.6f}\n")

    print(f"Wrote {layer_csv}")
    if lossy_rows:
        print(f"Wrote {lossy_csv}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
