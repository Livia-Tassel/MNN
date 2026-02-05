#!/usr/bin/env python3
import argparse
import os
import sys
from itertools import product
from typing import List

sys.path.append(os.path.dirname(__file__))
from analyze_blocknorm import evaluate_blocknorm  # noqa: E402


def parse_list(value: str, cast_type):
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast_type(part))
    return items


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep block-norm parameters and find best trade-offs.")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--out-dir", default="exp/n_lossless/out/blocknorm_sweep")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--min-seq-len", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--lossless-first-n", type=int, default=2)
    parser.add_argument("--block-sizes", default="64,128,256")
    parser.add_argument("--topk-ratios", default="0.02,0.05,0.1")
    parser.add_argument("--bits-list", default="8,10")
    parser.add_argument("--scale-bytes", type=int, default=2)
    parser.add_argument("--index-bytes", type=int, default=4)
    parser.add_argument("--compress-mode", choices=["gear", "gear-delta", "zstd"], default="gear-delta")
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--min-k-cosine", type=float, default=0.9999)
    parser.add_argument("--min-v-cosine", type=float, default=0.9999)
    parser.add_argument("--max-k-mae", type=float, default=0.01)
    parser.add_argument("--max-v-mae", type=float, default=0.002)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--save-combo", action="store_true", help="Keep per-combo outputs on disk.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    block_sizes = parse_list(args.block_sizes, int)
    topk_ratios = parse_list(args.topk_ratios, float)
    bits_list = parse_list(args.bits_list, int)

    if not block_sizes or not topk_ratios or not bits_list:
        print("block_sizes/topk_ratios/bits_list cannot be empty.", file=sys.stderr)
        return 2

    results = []
    combo_id = 0
    for block_size, topk_ratio, bits in product(block_sizes, topk_ratios, bits_list):
        combo_id += 1
        combo_name = f"bs{block_size}_topk{topk_ratio}_b{bits}"
        combo_out = os.path.join(args.out_dir, combo_name)
        ensure_dir(combo_out)

        rows, summary = evaluate_blocknorm(
            dump_dir=args.dump_dir,
            stage=args.stage,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            lossless_first_n=args.lossless_first_n,
            block_size=block_size,
            topk_ratio=topk_ratio,
            bits=bits,
            scale_bytes=args.scale_bytes,
            index_bytes=args.index_bytes,
            compress_mode=args.compress_mode,
            zstd_level=args.zstd_level,
            out_dir=combo_out,
        )
        if rows is None:
            continue

        passes = (
            summary["k_cosine"] >= args.min_k_cosine
            and summary["v_cosine"] >= args.min_v_cosine
            and summary["k_mae"] <= args.max_k_mae
            and summary["v_mae"] <= args.max_v_mae
        )

        results.append({
            "combo": combo_name,
            "block_size": block_size,
            "topk_ratio": topk_ratio,
            "bits": bits,
            "weighted_k_ratio": summary["weighted_k_ratio"],
            "weighted_v_ratio": summary["weighted_v_ratio"],
            "weighted_kv_ratio": summary["weighted_kv_ratio"],
            "k_mae": summary["k_mae"],
            "v_mae": summary["v_mae"],
            "k_cosine": summary["k_cosine"],
            "v_cosine": summary["v_cosine"],
            "rows": summary["rows"],
            "passes_precision": passes,
            "out_dir": combo_out if args.save_combo else "",
        })

        if not args.save_combo:
            try:
                os.remove(summary["csv_path"])
                os.remove(summary["summary_path"])
                os.rmdir(combo_out)
            except Exception:
                pass

    if not results:
        print("No results generated.", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r["weighted_kv_ratio"], reverse=True)

    csv_path = os.path.join(args.out_dir, "blocknorm_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = list(results[0].keys())
        f.write(",".join(headers) + "\n")
        for r in results:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    best_path = os.path.join(args.out_dir, "blocknorm_best.md")
    passing = [r for r in results if r["passes_precision"]]
    with open(best_path, "w", encoding="utf-8") as f:
        f.write("# Block-Norm Sweep Summary\n\n")
        f.write(f"- Total combos: {len(results)}\n")
        f.write(f"- Passing precision: {len(passing)}\n")
        f.write(f"- Precision thresholds: K cosine>={args.min_k_cosine}, V cosine>={args.min_v_cosine}, "
                f"K MAE<={args.max_k_mae}, V MAE<={args.max_v_mae}\n\n")

        f.write("## Top Overall (by KV ratio)\n")
        for r in results[: args.top_n]:
            f.write(
                f"- {r['combo']}: KV {r['weighted_kv_ratio']:.3f}, "
                f"K {r['weighted_k_ratio']:.3f}, V {r['weighted_v_ratio']:.3f}, "
                f"K MAE {r['k_mae']:.6f}, V MAE {r['v_mae']:.6f}, "
                f"K cos {r['k_cosine']:.6f}, V cos {r['v_cosine']:.6f}\n"
            )

        f.write("\n## Top Passing Precision\n")
        for r in passing[: args.top_n]:
            f.write(
                f"- {r['combo']}: KV {r['weighted_kv_ratio']:.3f}, "
                f"K {r['weighted_k_ratio']:.3f}, V {r['weighted_v_ratio']:.3f}, "
                f"K MAE {r['k_mae']:.6f}, V MAE {r['v_mae']:.6f}, "
                f"K cos {r['k_cosine']:.6f}, V cos {r['v_cosine']:.6f}\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
