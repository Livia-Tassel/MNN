#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
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


def should_keep_stage(meta_stage: str, target_stage: str) -> bool:
    if target_stage == "both":
        return True
    return meta_stage == target_stage


def delta_stream_u8(arr_u8: np.ndarray, seq_len: int, heads: int, head_dim: int) -> bytes:
    if seq_len <= 1:
        return arr_u8.tobytes()
    view = arr_u8.reshape((seq_len, heads, head_dim))
    first = view[0:1]
    delta = (view[1:].astype(np.int16) - view[:-1].astype(np.int16)) & 0xFF
    delta = delta.astype(np.uint8)
    return np.concatenate([first, delta], axis=0).tobytes()


def delta_decode_u8(encoded: np.ndarray, seq_len: int, heads: int, head_dim: int) -> np.ndarray:
    if seq_len <= 1:
        return encoded.copy()
    view = encoded.reshape((seq_len, heads, head_dim))
    out = np.empty_like(view)
    out[0] = view[0]
    for t in range(1, seq_len):
        out[t] = (out[t - 1].astype(np.uint16) + view[t].astype(np.uint16)) & 0xFF
    return out


def encode_gear(data: bytes, level: int) -> Tuple[bytes, bytes]:
    u16 = np.frombuffer(data, dtype=np.uint16)
    hi = (u16 >> 8).astype(np.uint8).tobytes()
    lo = (u16 & 0xFF).astype(np.uint8).tobytes()
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(hi), compressor.compress(lo)


def encode_gear_delta(data: bytes, seq_len: int, heads: int, head_dim: int, level: int) -> Tuple[bytes, bytes]:
    u16 = np.frombuffer(data, dtype=np.uint16)
    hi = (u16 >> 8).astype(np.uint8)
    lo = (u16 & 0xFF).astype(np.uint8).tobytes()
    hi_delta = delta_stream_u8(hi, seq_len, heads, head_dim)
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(hi_delta), compressor.compress(lo)


def decode_gear(hi_c: bytes, lo_c: bytes) -> bytes:
    decompressor = zstd.ZstdDecompressor()
    hi = decompressor.decompress(hi_c)
    lo = decompressor.decompress(lo_c)
    u16 = ((np.frombuffer(hi, dtype=np.uint8).astype(np.uint16) << 8) |
           np.frombuffer(lo, dtype=np.uint8).astype(np.uint16))
    return u16.astype(np.uint16).tobytes()


def decode_gear_delta(hi_c: bytes, lo_c: bytes, seq_len: int, heads: int, head_dim: int) -> bytes:
    decompressor = zstd.ZstdDecompressor()
    hi_delta = decompressor.decompress(hi_c)
    lo = decompressor.decompress(lo_c)
    hi_delta_arr = np.frombuffer(hi_delta, dtype=np.uint8)
    hi_dec = delta_decode_u8(hi_delta_arr, seq_len, heads, head_dim).reshape(-1)
    u16 = ((hi_dec.astype(np.uint16) << 8) |
           np.frombuffer(lo, dtype=np.uint8).astype(np.uint16))
    return u16.astype(np.uint16).tobytes()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark gear/gear+delta compression latency on FP16 dumps.")
    parser.add_argument("--dump-dir", required=True, help="Root dump directory containing run_* subdirs.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--min-seq-len", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--mode", choices=["gear", "gear-delta"], default="gear-delta")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if zstd is None:
        print("Missing dependency: zstandard. Install with `pip install zstandard`.", file=sys.stderr)
        return 2

    meta_files = list(iter_meta_files(args.dump_dir))
    if not meta_files:
        print(f"No meta_*.json found under {args.dump_dir}", file=sys.stderr)
        return 1

    total_encode = 0.0
    total_decode = 0.0
    total_ratio = 0.0
    total_bytes = 0
    counted = 0

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
        if not os.path.exists(k_path):
            continue

        with open(k_path, "rb") as f:
            data = f.read()

        kv_heads = int(meta.get("kv_heads", 0))
        head_dim = int(meta.get("head_dim", 0))
        data_len = len(data)

        enc_time = 0.0
        dec_time = 0.0
        ratio = 0.0

        for _ in range(args.iters):
            t0 = time.perf_counter()
            if args.mode == "gear":
                hi_c, lo_c = encode_gear(data, args.zstd_level)
            else:
                hi_c, lo_c = encode_gear_delta(data, seq_len, kv_heads, head_dim, args.zstd_level)
            t1 = time.perf_counter()
            if args.mode == "gear":
                _ = decode_gear(hi_c, lo_c)
            else:
                _ = decode_gear_delta(hi_c, lo_c, seq_len, kv_heads, head_dim)
            t2 = time.perf_counter()

            enc_time += (t1 - t0)
            dec_time += (t2 - t1)
            ratio = data_len / (len(hi_c) + len(lo_c)) if (hi_c and lo_c) else 0.0

        enc_time /= args.iters
        dec_time /= args.iters

        total_encode += enc_time
        total_decode += dec_time
        total_ratio += ratio
        total_bytes += data_len
        counted += 1

        if args.verbose:
            print(f"{meta_path}: ratio={ratio:.3f} enc_ms={enc_time*1000:.3f} dec_ms={dec_time*1000:.3f}")

        if args.limit and counted >= args.limit:
            break

    if counted == 0:
        print("No FP16 dumps matched filters.", file=sys.stderr)
        return 1

    avg_ratio = total_ratio / counted
    avg_enc = total_encode / counted
    avg_dec = total_decode / counted
    avg_total = avg_enc + avg_dec
    mb = total_bytes / (1024 * 1024)
    throughput = mb / (total_encode + total_decode) if (total_encode + total_decode) > 0 else 0.0

    print(f"Mode: {args.mode}, zstd level: {args.zstd_level}")
    print(f"Dumps: {counted}")
    print(f"Avg ratio: {avg_ratio:.3f}")
    print(f"Avg encode: {avg_enc*1000:.3f} ms")
    print(f"Avg decode: {avg_dec*1000:.3f} ms")
    print(f"Avg total: {avg_total*1000:.3f} ms")
    print(f"Aggregate throughput: {throughput:.3f} MB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
