#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterator, Tuple

import numpy as np

from huffman_utils import build_huffman_code_lengths, encode_symbols


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Lossless Huffman encoding for FP16 KV dumps (hi6/lo10).")
    parser.add_argument("--dump-dir", required=True, help="Source FP16 dump directory.")
    parser.add_argument("--out-dir", default="exp/bitstream/out", help="Output directory.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--codebook", choices=["per-dump", "global"], default="global")
    parser.add_argument("--min-seq-len", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.dump_dir))
    if not meta_files:
        print(f"No meta_*.json found under {args.dump_dir}", file=sys.stderr)
        return 1

    encoded = 0
    skipped = 0
    errors = 0

    os.makedirs(args.out_dir, exist_ok=True)
    global_lengths = None
    if args.codebook == "global":
        k_hi6_freq = np.zeros(64, dtype=np.int64)
        k_lo10_freq = np.zeros(1024, dtype=np.int64)
        v_hi6_freq = np.zeros(64, dtype=np.int64)
        v_lo10_freq = np.zeros(1024, dtype=np.int64)
        for meta_path in meta_files:
            try:
                meta = load_json(meta_path)
            except Exception:
                continue
            stage = meta.get("stage", "unknown")
            if args.stage != "both" and stage != args.stage:
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
            k_hi6_freq += np.bincount(k_hi6, minlength=64)
            k_lo10_freq += np.bincount(k_lo10, minlength=1024)
            v_hi6_freq += np.bincount(v_hi6, minlength=64)
            v_lo10_freq += np.bincount(v_lo10, minlength=1024)
        global_lengths = {
            "k_hi6": build_huffman_code_lengths(k_hi6_freq.tolist()),
            "k_lo10": build_huffman_code_lengths(k_lo10_freq.tolist()),
            "v_hi6": build_huffman_code_lengths(v_hi6_freq.tolist()),
            "v_lo10": build_huffman_code_lengths(v_lo10_freq.tolist()),
        }
        write_json(os.path.join(args.out_dir, "global_codebook.json"), {
            "codebook": "global",
            "k_hi6_code_lengths": global_lengths["k_hi6"],
            "k_lo10_code_lengths": global_lengths["k_lo10"],
            "v_hi6_code_lengths": global_lengths["v_hi6"],
            "v_lo10_code_lengths": global_lengths["v_lo10"],
        })

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

        seq_len = int(meta.get("seq_len", 0))
        if args.min_seq_len and seq_len < args.min_seq_len:
            skipped += 1
            continue
        if args.max_seq_len and seq_len > args.max_seq_len:
            skipped += 1
            continue

        bytes_per_elem = int(meta.get("bytes_per_elem", 0))
        if bytes_per_elem != 2:
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
        dst_meta_path = os.path.join(dst_dir, f"bitstream_{meta_name}.json")

        if not args.overwrite and os.path.exists(dst_meta_path):
            skipped += 1
            continue

        try:
            k = np.frombuffer(open(k_path, "rb").read(), dtype=np.uint16)
            v = np.frombuffer(open(v_path, "rb").read(), dtype=np.uint16)

            k_hi6, k_lo10 = split_fp16(k)
            v_hi6, v_lo10 = split_fp16(v)

            if args.codebook == "global":
                k_hi6_len = global_lengths["k_hi6"]
                k_lo10_len = global_lengths["k_lo10"]
                v_hi6_len = global_lengths["v_hi6"]
                v_lo10_len = global_lengths["v_lo10"]
            else:
                k_hi6_freq = np.bincount(k_hi6, minlength=64).tolist()
                k_lo10_freq = np.bincount(k_lo10, minlength=1024).tolist()
                v_hi6_freq = np.bincount(v_hi6, minlength=64).tolist()
                v_lo10_freq = np.bincount(v_lo10, minlength=1024).tolist()
                k_hi6_len = build_huffman_code_lengths(k_hi6_freq)
                k_lo10_len = build_huffman_code_lengths(k_lo10_freq)
                v_hi6_len = build_huffman_code_lengths(v_hi6_freq)
                v_lo10_len = build_huffman_code_lengths(v_lo10_freq)

            k_hi6_bs = encode_symbols(k_hi6.tolist(), k_hi6_len)
            k_lo10_bs = encode_symbols(k_lo10.tolist(), k_lo10_len)
            v_hi6_bs = encode_symbols(v_hi6.tolist(), v_hi6_len)
            v_lo10_bs = encode_symbols(v_lo10.tolist(), v_lo10_len)

            k_hi6_path = os.path.join(dst_dir, f"k_hi6_{meta_name}.huff")
            k_lo10_path = os.path.join(dst_dir, f"k_lo10_{meta_name}.huff")
            v_hi6_path = os.path.join(dst_dir, f"v_hi6_{meta_name}.huff")
            v_lo10_path = os.path.join(dst_dir, f"v_lo10_{meta_name}.huff")

            with open(k_hi6_path, "wb") as f:
                f.write(k_hi6_bs)
            with open(k_lo10_path, "wb") as f:
                f.write(k_lo10_bs)
            with open(v_hi6_path, "wb") as f:
                f.write(v_hi6_bs)
            with open(v_lo10_path, "wb") as f:
                f.write(v_lo10_bs)

            out_meta = dict(meta)
            out_meta.update({
                "layout": "bitstream_huffman",
                "bytes_per_elem": 2,
                "k_hi6_file": os.path.basename(k_hi6_path),
                "k_lo10_file": os.path.basename(k_lo10_path),
                "v_hi6_file": os.path.basename(v_hi6_path),
                "v_lo10_file": os.path.basename(v_lo10_path),
                "codebook": args.codebook,
                "num_elements": int(k.size),
            })
            if args.codebook == "per-dump":
                out_meta["k_hi6_code_lengths"] = k_hi6_len
                out_meta["k_lo10_code_lengths"] = k_lo10_len
                out_meta["v_hi6_code_lengths"] = v_hi6_len
                out_meta["v_lo10_code_lengths"] = v_lo10_len
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
