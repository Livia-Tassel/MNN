#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterator


def iter_meta_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("bitstream_meta_") and name.endswith(".json"):
                yield os.path.join(dirpath, name)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def size(path: str) -> int:
    return os.path.getsize(path)


def ratio(raw: float, comp: float) -> float:
    return raw / comp if comp > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Huffman bitstream compression ratios.")
    parser.add_argument("--bitstream-dir", required=True)
    parser.add_argument("--orig-dump-dir", required=True)
    parser.add_argument("--out-dir", default="exp/bitstream/out")
    parser.add_argument("--include-meta", action="store_true")
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.bitstream_dir))
    if not meta_files:
        print(f"No bitstream_meta.json found under {args.bitstream_dir}", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    totals = {"k_raw": 0.0, "v_raw": 0.0, "k_comp": 0.0, "v_comp": 0.0}
    global_meta_bytes = 0
    global_path = os.path.join(args.bitstream_dir, "global_codebook.json")
    if os.path.exists(global_path):
        global_meta_bytes = size(global_path)

    for meta_path in meta_files:
        meta = load_json(meta_path)
        rel_dir = os.path.relpath(os.path.dirname(meta_path), args.bitstream_dir)
        orig_dir = os.path.join(args.orig_dump_dir, rel_dir)

        k_path = os.path.join(orig_dir, meta.get("k_file", ""))
        v_path = os.path.join(orig_dir, meta.get("v_file", ""))
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            continue

        k_raw = size(k_path)
        v_raw = size(v_path)

        k_hi6 = size(os.path.join(os.path.dirname(meta_path), meta["k_hi6_file"]))
        k_lo10 = size(os.path.join(os.path.dirname(meta_path), meta["k_lo10_file"]))
        v_hi6 = size(os.path.join(os.path.dirname(meta_path), meta["v_hi6_file"]))
        v_lo10 = size(os.path.join(os.path.dirname(meta_path), meta["v_lo10_file"]))

        k_comp = k_hi6 + k_lo10
        v_comp = v_hi6 + v_lo10

        if args.include_meta:
            k_comp += size(meta_path) / 2.0
            v_comp += size(meta_path) / 2.0

        totals["k_raw"] += k_raw
        totals["v_raw"] += v_raw
        totals["k_comp"] += k_comp
        totals["v_comp"] += v_comp

        rows.append({
            "run_id": meta.get("run_id", ""),
            "layer_id": meta.get("layer_id", ""),
            "stage": meta.get("stage", ""),
            "seq_len": meta.get("seq_len", ""),
            "k_raw": k_raw,
            "v_raw": v_raw,
            "k_comp": k_comp,
            "v_comp": v_comp,
            "k_ratio": ratio(k_raw, k_comp),
            "v_ratio": ratio(v_raw, v_comp),
        })

    csv_path = os.path.join(args.out_dir, "huffman_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = list(rows[0].keys()) if rows else []
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    summary_path = os.path.join(args.out_dir, "huffman_summary.md")
    if args.include_meta and global_meta_bytes:
        totals["k_comp"] += global_meta_bytes / 2.0
        totals["v_comp"] += global_meta_bytes / 2.0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Huffman Bitstream Summary\n\n")
        f.write(f"- Dumps analyzed: {len(rows)}\n")
        f.write(f"- Include meta bytes: {bool(args.include_meta)}\n\n")
        if args.include_meta and global_meta_bytes:
            f.write(f"- Global codebook bytes: {global_meta_bytes}\n\n")
        f.write(f"- Weighted K ratio: {ratio(totals['k_raw'], totals['k_comp']):.3f}\n")
        f.write(f"- Weighted V ratio: {ratio(totals['v_raw'], totals['v_comp']):.3f}\n")
        f.write(f"- Weighted KV ratio: {ratio(totals['k_raw'] + totals['v_raw'], totals['k_comp'] + totals['v_comp']):.3f}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
