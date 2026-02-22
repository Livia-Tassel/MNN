#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterator

import numpy as np

from arith_utils import arithmetic_decode


def iter_meta_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("arith_meta_") and name.endswith(".json"):
                yield os.path.join(dirpath, name)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode Arithmetic-coded bitstreams and verify exact match.")
    parser.add_argument("--bitstream-dir", required=True)
    parser.add_argument("--orig-dump-dir", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.bitstream_dir))
    if not meta_files:
        print(f"No arith_meta_*.json found under {args.bitstream_dir}", file=sys.stderr)
        return 1

    global_codebook = None
    global_path = os.path.join(args.bitstream_dir, "global_codebook_arith.json")
    if os.path.exists(global_path):
        global_codebook = load_json(global_path)

    checked = 0
    failed = 0
    for meta_path in meta_files:
        meta = load_json(meta_path)
        rel_dir = os.path.relpath(os.path.dirname(meta_path), args.bitstream_dir)
        orig_dir = os.path.join(args.orig_dump_dir, rel_dir)

        if meta.get("codebook") == "global":
            if global_codebook is None:
                raise ValueError("Missing global_codebook_arith.json for global codebook decode.")
            k_hi6_freq = global_codebook["k_hi6_freq"]
            k_lo10_freq = global_codebook["k_lo10_freq"]
            v_hi6_freq = global_codebook["v_hi6_freq"]
            v_lo10_freq = global_codebook["v_lo10_freq"]
        else:
            k_hi6_freq = meta["k_hi6_freq"]
            k_lo10_freq = meta["k_lo10_freq"]
            v_hi6_freq = meta["v_hi6_freq"]
            v_lo10_freq = meta["v_lo10_freq"]

        num = int(meta["num_elements"])
        k_hi6 = arithmetic_decode(
            open(os.path.join(os.path.dirname(meta_path), meta["k_hi6_file"]), "rb").read(),
            k_hi6_freq,
            num,
        )
        k_lo10 = arithmetic_decode(
            open(os.path.join(os.path.dirname(meta_path), meta["k_lo10_file"]), "rb").read(),
            k_lo10_freq,
            num,
        )
        v_hi6 = arithmetic_decode(
            open(os.path.join(os.path.dirname(meta_path), meta["v_hi6_file"]), "rb").read(),
            v_hi6_freq,
            num,
        )
        v_lo10 = arithmetic_decode(
            open(os.path.join(os.path.dirname(meta_path), meta["v_lo10_file"]), "rb").read(),
            v_lo10_freq,
            num,
        )

        k_u16 = (np.array(k_hi6, dtype=np.uint16) << 10) | np.array(k_lo10, dtype=np.uint16)
        v_u16 = (np.array(v_hi6, dtype=np.uint16) << 10) | np.array(v_lo10, dtype=np.uint16)

        if args.verify:
            k_path = os.path.join(orig_dir, meta.get("k_file", ""))
            v_path = os.path.join(orig_dir, meta.get("v_file", ""))
            if not os.path.exists(k_path) or not os.path.exists(v_path):
                print(f"Missing original K/V for {meta_path}", file=sys.stderr)
                failed += 1
                continue
            k_orig = np.frombuffer(open(k_path, "rb").read(), dtype=np.uint16)
            v_orig = np.frombuffer(open(v_path, "rb").read(), dtype=np.uint16)
            if not (np.array_equal(k_u16, k_orig) and np.array_equal(v_u16, v_orig)):
                print(f"Mismatch: {meta_path}", file=sys.stderr)
                failed += 1
            else:
                checked += 1
        else:
            checked += 1

    print(f"Checked: {checked}, Failed: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
