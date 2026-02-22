#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterator

import numpy as np


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


def should_keep_stage(meta_stage: str, target_stage: str) -> bool:
    if target_stage == "both":
        return True
    return meta_stage == target_stage


def convert_fp32_to_fp16(data: bytes) -> bytes:
    if len(data) % 4 != 0:
        raise ValueError(f"FP32 buffer size {len(data)} not divisible by 4")
    arr = np.frombuffer(data, dtype=np.float32)
    return arr.astype(np.float16).tobytes()


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert FP32 KV dumps to FP16 dumps for offline compression analysis.")
    parser.add_argument("--src-dump-dir", required=True, help="Source dump directory, e.g. exp/distribution/dumps")
    parser.add_argument("--dst-dump-dir", default="exp/gear_fp16/dumps_fp16", help="Destination dump directory")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--min-seq-len", type=int, default=0, help="Skip entries with seq_len smaller than this.")
    parser.add_argument("--max-seq-len", type=int, default=0, help="Skip entries with seq_len larger than this (0 = no limit).")
    parser.add_argument("--allow-fp16", action="store_true", help="If set, copy existing FP16 dumps as-is.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.src_dump_dir))
    if not meta_files:
        print(f"No meta_*.json found under {args.src_dump_dir}", file=sys.stderr)
        return 1

    converted = 0
    skipped = 0
    errors = 0

    for meta_path in meta_files:
        try:
            meta = load_json(meta_path)
        except json.JSONDecodeError:
            print(f"Skip {meta_path}: invalid JSON (likely partial write).", file=sys.stderr)
            errors += 1
            continue
        except Exception as exc:
            print(f"Skip {meta_path}: {exc}", file=sys.stderr)
            errors += 1
            continue
        stage = meta.get("stage", "unknown")
        if not should_keep_stage(stage, args.stage):
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
        if bytes_per_elem not in (2, 4):
            print(f"Skip {meta_path}: unsupported bytes_per_elem={bytes_per_elem}", file=sys.stderr)
            skipped += 1
            continue

        if bytes_per_elem == 2 and not args.allow_fp16:
            print(f"Skip {meta_path}: already FP16 (use --allow-fp16 to copy).", file=sys.stderr)
            skipped += 1
            continue

        dirpath = os.path.dirname(meta_path)
        k_path = os.path.join(dirpath, meta.get("k_file", ""))
        v_path = os.path.join(dirpath, meta.get("v_file", ""))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            print(f"Skip {meta_path}: missing k/v files.", file=sys.stderr)
            skipped += 1
            continue

        rel_dir = os.path.relpath(dirpath, args.src_dump_dir)
        dst_dir = os.path.join(args.dst_dump_dir, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)
        dst_meta_path = os.path.join(dst_dir, os.path.basename(meta_path))
        dst_k_path = os.path.join(dst_dir, meta.get("k_file", "k.bin"))
        dst_v_path = os.path.join(dst_dir, meta.get("v_file", "v.bin"))

        if not args.overwrite and (os.path.exists(dst_meta_path) or os.path.exists(dst_k_path) or os.path.exists(dst_v_path)):
            print(f"Skip {meta_path}: output exists (use --overwrite to replace).", file=sys.stderr)
            skipped += 1
            continue

        try:
            with open(k_path, "rb") as f:
                k_bytes = f.read()
            with open(v_path, "rb") as f:
                v_bytes = f.read()

            if bytes_per_elem == 4:
                k_out = convert_fp32_to_fp16(k_bytes)
                v_out = convert_fp32_to_fp16(v_bytes)
                new_meta = dict(meta)
                new_meta["source_bytes_per_elem"] = bytes_per_elem
                new_meta["bytes_per_elem"] = 2
                new_meta["dtype"] = "fp16"
                new_meta["converted_from"] = "fp32"
            else:
                k_out = k_bytes
                v_out = v_bytes
                new_meta = dict(meta)
                new_meta["source_bytes_per_elem"] = bytes_per_elem
                new_meta["bytes_per_elem"] = 2
                new_meta["dtype"] = "fp16"

            with open(dst_k_path, "wb") as f:
                f.write(k_out)
            with open(dst_v_path, "wb") as f:
                f.write(v_out)
            write_json(dst_meta_path, new_meta)
            converted += 1
        except OSError as exc:
            if exc.errno == 28:
                print(f"Error converting {meta_path}: No space left on device.", file=sys.stderr)
                return 3
            print(f"Error converting {meta_path}: {exc}", file=sys.stderr)
            errors += 1
        except Exception as exc:
            print(f"Error converting {meta_path}: {exc}", file=sys.stderr)
            errors += 1

    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    if errors:
        print(f"Errors: {errors}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
