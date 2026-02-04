#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import zstandard as zstd
except Exception as exc:
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
    if not data:
        return 0.0
    if zstd is None:
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


def load_rope_config(path: Optional[str]) -> Dict[str, Optional[float]]:
    if not path:
        return {}
    cfg = load_json(path)
    rope_theta = (
        cfg.get("rope_theta")
        or cfg.get("rope_base")
        or cfg.get("rope_freq_base")
        or cfg.get("rope_freq_base")
    )
    rope_dim = cfg.get("rotary_dim") or cfg.get("rope_dim")
    return {"rope_theta": rope_theta, "rope_dim": rope_dim}


def inverse_rope_fp16(data: bytes, seq_len: int, kv_heads: int, head_dim: int, rope_theta: float, rope_dim: int) -> bytes:
    k = np.frombuffer(data, dtype=np.float16).reshape((seq_len, kv_heads, head_dim))
    k_f = k.astype(np.float32, copy=True)
    rotary_dim = min(rope_dim, head_dim)
    if rotary_dim % 2 != 0:
        rotary_dim -= 1
    if rotary_dim <= 0:
        return k.tobytes()
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    pos = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(pos, inv_freq)
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    x1 = k_f[..., 0:rotary_dim:2]
    x2 = k_f[..., 1:rotary_dim:2]
    k_f[..., 0:rotary_dim:2] = x1 * cos + x2 * sin
    k_f[..., 1:rotary_dim:2] = -x1 * sin + x2 * cos
    return k_f.astype(np.float16).tobytes()


def iter_meta_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("meta_") and name.endswith(".json"):
                yield os.path.join(dirpath, name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze KV dump distributions and compression ratios.")
    parser.add_argument("--dump-dir", required=True, help="Root dump directory containing run_* subdirs.")
    parser.add_argument("--out-dir", default="exp/distribution/out", help="Output directory for csv/summary.")
    parser.add_argument("--zstd-level", type=int, default=3, help="Zstd compression level.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--rope-config", default=None, help="Path to config.json or llm_config.json for RoPE params.")
    parser.add_argument("--rope-theta", type=float, default=None, help="Override rope_theta.")
    parser.add_argument("--rope-dim", type=int, default=None, help="Override rotary_dim.")
    args = parser.parse_args()

    if zstd is None:
        print("Missing dependency: zstandard. Install with `pip install zstandard`.", file=sys.stderr)
        return 2

    rope_cfg = load_rope_config(args.rope_config)
    rope_theta = args.rope_theta if args.rope_theta is not None else rope_cfg.get("rope_theta")
    rope_dim = args.rope_dim if args.rope_dim is not None else rope_cfg.get("rope_dim")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "layer_metrics.csv")
    summary_path = os.path.join(args.out_dir, "summary.md")

    rows = []
    meta_files = list(iter_meta_files(args.dump_dir))
    if not meta_files:
        print(f"No meta_*.json found under {args.dump_dir}", file=sys.stderr)
        return 1

    for meta_path in meta_files:
        meta = load_json(meta_path)
        stage = meta.get("stage", "unknown")
        if args.stage != "both" and stage != args.stage:
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
        seq_len = int(meta["seq_len"])
        kv_heads = int(meta["kv_heads"])
        head_dim = int(meta["head_dim"])

        k_entropy = entropy_bytes(k_bytes)
        v_entropy = entropy_bytes(v_bytes)
        k_zstd = zstd_ratio(k_bytes, args.zstd_level)
        v_zstd = zstd_ratio(v_bytes, args.zstd_level)

        k_hi_ent = k_lo_ent = k_hi_ratio = k_lo_ratio = k_gear = 0.0
        v_hi_ent = v_lo_ent = v_hi_ratio = v_lo_ratio = v_gear = 0.0
        k_rope_inv_gear = 0.0
        k_rope_inv_zstd = 0.0

        if bytes_per_elem == 2:
            k_hi_ent, k_lo_ent, k_hi_ratio, k_lo_ratio, k_gear = gear_ratios_fp16(k_bytes, args.zstd_level)
            v_hi_ent, v_lo_ent, v_hi_ratio, v_lo_ratio, v_gear = gear_ratios_fp16(v_bytes, args.zstd_level)

            if rope_theta is not None:
                rope_dim_final = int(rope_dim) if rope_dim is not None else head_dim
                k_inv = inverse_rope_fp16(k_bytes, seq_len, kv_heads, head_dim, float(rope_theta), rope_dim_final)
                k_rope_inv_zstd = zstd_ratio(k_inv, args.zstd_level)
                _, _, _, _, k_rope_inv_gear = gear_ratios_fp16(k_inv, args.zstd_level)

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
            "k_hi_entropy": k_hi_ent,
            "k_lo_entropy": k_lo_ent,
            "k_hi_ratio": k_hi_ratio,
            "k_lo_ratio": k_lo_ratio,
            "k_gear_ratio": k_gear,
            "v_hi_entropy": v_hi_ent,
            "v_lo_entropy": v_lo_ent,
            "v_hi_ratio": v_hi_ratio,
            "v_lo_ratio": v_lo_ratio,
            "v_gear_ratio": v_gear,
            "k_rope_inv_zstd_ratio": k_rope_inv_zstd,
            "k_rope_inv_gear_ratio": k_rope_inv_gear,
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

    # Summary
    def avg(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    k_gear_avg = avg([r["k_gear_ratio"] for r in rows])
    v_gear_avg = avg([r["v_gear_ratio"] for r in rows])
    k_zstd_avg = avg([r["k_zstd_ratio"] for r in rows])
    v_zstd_avg = avg([r["v_zstd_ratio"] for r in rows])
    k_rope_inv_avg = avg([r["k_rope_inv_gear_ratio"] for r in rows])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# KV Dump Summary\n")
        f.write("\n")
        f.write(f"- Dumps analyzed: {len(rows)}\n")
        f.write(f"- Avg K gear ratio: {k_gear_avg:.3f}\n")
        f.write(f"- Avg V gear ratio: {v_gear_avg:.3f}\n")
        f.write(f"- Avg K zstd ratio: {k_zstd_avg:.3f}\n")
        f.write(f"- Avg V zstd ratio: {v_zstd_avg:.3f}\n")
        if rope_theta is not None:
            f.write(f"- Avg K inverse-RoPE gear ratio: {k_rope_inv_avg:.3f}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
