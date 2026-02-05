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


def should_keep_stage(meta_stage: str, target_stage: str) -> bool:
    if target_stage == "both":
        return True
    return meta_stage == target_stage


def ratio(raw: float, comp: float) -> float:
    return raw / comp if comp > 0 else 0.0


def block_normalize_quantize(
    x: np.ndarray,
    block_size: int,
    bits: int,
    topk_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    # x shape: (seq_len, heads, head_dim)
    seq_len, heads, head_dim = x.shape
    blocks_per_vec = int(np.ceil(head_dim / block_size))
    padded_dim = blocks_per_vec * block_size

    x_2d = x.reshape(-1, head_dim)
    if padded_dim > head_dim:
        pad = np.zeros((x_2d.shape[0], padded_dim - head_dim), dtype=x.dtype)
        x_2d = np.concatenate([x_2d, pad], axis=1)

    blocks = x_2d.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1)
    scales[scales == 0] = 1.0
    norm = blocks / scales[:, None]

    qmax = (1 << (bits - 1)) - 1
    q = np.round(norm * qmax).clip(-qmax, qmax)
    norm_q = (q / qmax).astype(x.dtype)

    # top-k blocks by energy
    energy = np.sum(blocks.astype(np.float32) ** 2, axis=1)
    total_blocks = energy.size
    topk = max(1, int(total_blocks * topk_ratio))
    topk_idx = np.argpartition(-energy, topk - 1)[:topk]

    # restore only top-k blocks
    restored = norm_q.copy()
    restored[topk_idx] = restored[topk_idx] * scales[topk_idx, None]

    # reshape back and trim padding
    restored_2d = restored.reshape(-1, padded_dim)[:, :head_dim]
    stored_2d = norm_q.reshape(-1, padded_dim)[:, :head_dim]

    restored_x = restored_2d.reshape(seq_len, heads, head_dim)
    stored_x = stored_2d.reshape(seq_len, heads, head_dim)

    return stored_x, restored_x, scales, total_blocks, topk


def main() -> int:
    parser = argparse.ArgumentParser(description="Block normalization with partial de-normalization (top-k).")
    parser.add_argument("--dump-dir", required=True, help="Root dump directory.")
    parser.add_argument("--out-dir", default="exp/n_lossless/out", help="Output directory.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--min-seq-len", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--lossless-first-n", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--topk-ratio", type=float, default=0.8)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--scale-bytes", type=int, default=2, help="Bytes per stored scale.")
    parser.add_argument("--index-bytes", type=int, default=4, help="Bytes per stored block index.")
    parser.add_argument("--compress-mode", choices=["gear", "gear-delta", "zstd"], default="gear-delta")
    parser.add_argument("--zstd-level", type=int, default=3)
    args = parser.parse_args()

    rows, summary = evaluate_blocknorm(
        dump_dir=args.dump_dir,
        stage=args.stage,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        lossless_first_n=args.lossless_first_n,
        block_size=args.block_size,
        topk_ratio=args.topk_ratio,
        bits=args.bits,
        scale_bytes=args.scale_bytes,
        index_bytes=args.index_bytes,
        compress_mode=args.compress_mode,
        zstd_level=args.zstd_level,
        out_dir=args.out_dir,
    )

    if rows is None:
        return 1

    csv_path = summary["csv_path"]
    summary_path = summary["summary_path"]
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


def evaluate_blocknorm(
    *,
    dump_dir: str,
    stage: str,
    min_seq_len: int,
    max_seq_len: int,
    lossless_first_n: int,
    block_size: int,
    topk_ratio: float,
    bits: int,
    scale_bytes: int,
    index_bytes: int,
    compress_mode: str,
    zstd_level: int,
    out_dir: str,
) -> Tuple[Iterator[Dict], Dict]:
    if zstd is None:
        print("Missing dependency: zstandard. Install with `pip install zstandard`.", file=sys.stderr)
        return None, {}

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "blocknorm_metrics.csv")
    summary_path = os.path.join(out_dir, "blocknorm_summary.md")

    rows = []
    totals = {"k_raw": 0.0, "v_raw": 0.0, "k_comp": 0.0, "v_comp": 0.0, "k_mae_sum": 0.0, "v_mae_sum": 0.0,
              "k_cos_sum": 0.0, "v_cos_sum": 0.0, "elems": 0}

    meta_files = list(iter_meta_files(dump_dir))
    if not meta_files:
        print(f"No meta_*.json found under {dump_dir}", file=sys.stderr)
        return None, {}

    for meta_path in meta_files:
        meta = load_json(meta_path)
        meta_stage = meta.get("stage", "unknown")
        if not should_keep_stage(meta_stage, stage):
            continue
        seq_len = int(meta.get("seq_len", 0))
        if min_seq_len and seq_len < min_seq_len:
            continue
        if max_seq_len and seq_len > max_seq_len:
            continue

        layer_id = int(meta.get("layer_id", -1))
        if layer_id < lossless_first_n:
            continue

        kv_heads = int(meta.get("kv_heads", 0))
        head_dim = int(meta.get("head_dim", 0))
        bytes_per_elem = int(meta.get("bytes_per_elem", 0))
        dtype = np.float16 if bytes_per_elem == 2 else np.float32

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta.get("k_file", ""))
        v_path = os.path.join(dirpath, meta.get("v_file", ""))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            continue

        k = np.frombuffer(open(k_path, "rb").read(), dtype=dtype).reshape(seq_len, kv_heads, head_dim)
        v = np.frombuffer(open(v_path, "rb").read(), dtype=dtype).reshape(seq_len, kv_heads, head_dim)

        k_stored, k_restored, k_scales, k_total_blocks, k_topk = block_normalize_quantize(
            k, block_size, bits, topk_ratio
        )
        v_stored, v_restored, v_scales, v_total_blocks, v_topk = block_normalize_quantize(
            v, block_size, bits, topk_ratio
        )

        # compression size = compressed stored data + topk metadata
        k_bytes = k_stored.tobytes()
        v_bytes = v_stored.tobytes()
        if bytes_per_elem == 2:
            k_comp = compress_fp16_gear(k_bytes, seq_len, kv_heads, head_dim, zstd_level, compress_mode == "gear-delta")
            v_comp = compress_fp16_gear(v_bytes, seq_len, kv_heads, head_dim, zstd_level, compress_mode == "gear-delta")
        else:
            k_comp = compress_fp32_split16(k_bytes, seq_len, kv_heads, head_dim, zstd_level, compress_mode == "gear-delta")
            v_comp = compress_fp32_split16(v_bytes, seq_len, kv_heads, head_dim, zstd_level, compress_mode == "gear-delta")

        # metadata bytes: store scales + indices for top-k blocks only
        k_meta = k_topk * (scale_bytes + index_bytes)
        v_meta = v_topk * (scale_bytes + index_bytes)
        k_comp_total = k_comp + k_meta
        v_comp_total = v_comp + v_meta

        k_err = compute_error_metrics(k, k_restored)
        v_err = compute_error_metrics(v, v_restored)

        k_raw = k.nbytes
        v_raw = v.nbytes

        rows.append({
            "layer_id": layer_id,
            "stage": meta_stage,
            "seq_len": seq_len,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "bytes_per_elem": bytes_per_elem,
            "block_size": block_size,
            "topk_ratio": topk_ratio,
            "bits": bits,
            "k_ratio": ratio(k_raw, k_comp_total),
            "v_ratio": ratio(v_raw, v_comp_total),
            "k_mae": k_err["mae"],
            "k_cosine": k_err["cosine"],
            "v_mae": v_err["mae"],
            "v_cosine": v_err["cosine"],
        })

        totals["k_raw"] += k_raw
        totals["v_raw"] += v_raw
        totals["k_comp"] += k_comp_total
        totals["v_comp"] += v_comp_total
        totals["k_mae_sum"] += k_err["mae"] * k.size
        totals["v_mae_sum"] += v_err["mae"] * v.size
        totals["k_cos_sum"] += k_err["cosine"] * k.size
        totals["v_cos_sum"] += v_err["cosine"] * v.size
        totals["elems"] += k.size

    if not rows:
        print("No rows after filtering.", file=sys.stderr)
        return None, {}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Block-Norm Summary\n\n")
        f.write(f"- Lossless first N layers: {lossless_first_n}\n")
        f.write(f"- Block size: {block_size}\n")
        f.write(f"- Top-k ratio: {topk_ratio}\n")
        f.write(f"- Quant bits: {bits}\n")
        f.write(f"- Compress mode: {compress_mode}\n\n")
        k_ratio = ratio(totals["k_raw"], totals["k_comp"])
        v_ratio = ratio(totals["v_raw"], totals["v_comp"])
        kv_ratio = ratio(totals["k_raw"] + totals["v_raw"], totals["k_comp"] + totals["v_comp"])
        f.write(f"- Weighted K ratio: {k_ratio:.3f}\n")
        f.write(f"- Weighted V ratio: {v_ratio:.3f}\n")
        f.write(f"- Weighted KV ratio: {kv_ratio:.3f}\n\n")
        f.write("## Error (weighted by elements)\n")
        elems = max(1, totals["elems"])
        k_mae = totals["k_mae_sum"] / elems
        v_mae = totals["v_mae_sum"] / elems
        k_cos = totals["k_cos_sum"] / elems
        v_cos = totals["v_cos_sum"] / elems
        f.write(f"- K MAE: {k_mae:.6f}\n")
        f.write(f"- K Cosine: {k_cos:.6f}\n")
        f.write(f"- V MAE: {v_mae:.6f}\n")
        f.write(f"- V Cosine: {v_cos:.6f}\n")

    summary = {
        "csv_path": csv_path,
        "summary_path": summary_path,
        "weighted_k_ratio": k_ratio,
        "weighted_v_ratio": v_ratio,
        "weighted_kv_ratio": kv_ratio,
        "k_mae": k_mae,
        "v_mae": v_mae,
        "k_cosine": k_cos,
        "v_cosine": v_cos,
        "rows": len(rows),
    }
    return rows, summary


if __name__ == "__main__":
    raise SystemExit(main())
