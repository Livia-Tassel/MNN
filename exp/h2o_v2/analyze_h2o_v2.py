#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def parse_float(value, default=0.0):
    if value is None:
        return default
    if not isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return default
    # Markdown cells are often formatted as "x ± y"; take the first numeric token.
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", value)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return default
    try:
        return float(value)
    except Exception:
        return default


def parse_decode_speed(speed_cell):
    # Example: "100.00 +/- 1.00<br>30.00 +/- 0.20"
    if not speed_cell:
        return 0.0
    m = re.search(r"<br>\s*([0-9]+(?:\.[0-9]+)?)", speed_cell)
    if not m:
        return 0.0
    return parse_float(m.group(1), 0.0)


def main():
    parser = argparse.ArgumentParser(description="Analyze parsed H2O CSV metrics.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="exp/h2o_v2/out/summary.md")
    args = parser.parse_args()

    rows = []
    with Path(args.csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["_h2o_keep"] = parse_float(row.get("h2o_keep", "0"))
            row["_h2o_lossy"] = parse_float(row.get("h2o_lossy", "0"))
            row["_h2o_lossless"] = parse_float(row.get("h2o_lossless", "0"))
            row["_h2o_target_keep_effective"] = parse_float(row.get("h2o_target_keep_effective", "0"))
            row["_h2o_floor_keep"] = parse_float(row.get("h2o_floor_keep", "0"))
            row["_h2o_quantized_keep"] = parse_float(row.get("h2o_quantized_keep", "0"))
            row["_h2o_evict_us"] = parse_float(row.get("h2o_evict_us", "0"))
            row["_h2o_codec_us"] = parse_float(row.get("h2o_codec_us", "0"))
            row["_decode_tps"] = parse_decode_speed(row.get("speed(tok/s)", ""))
            row["_h2o_total"] = row["_h2o_lossy"] * row["_h2o_lossless"]
            rows.append(row)

    if not rows:
        raise SystemExit("No rows found in CSV.")

    best_lossy = max(rows, key=lambda r: r["_h2o_lossy"])
    best_decode = max(rows, key=lambda r: r["_decode_tps"])
    best_total = max(rows, key=lambda r: r["_h2o_total"])
    avg_keep = sum(r["_h2o_keep"] for r in rows) / len(rows)
    avg_lossy = sum(r["_h2o_lossy"] for r in rows) / len(rows)
    avg_lossless = sum(r["_h2o_lossless"] for r in rows) / len(rows)
    avg_total = sum(r["_h2o_total"] for r in rows) / len(rows)
    avg_target_keep_effective = sum(r["_h2o_target_keep_effective"] for r in rows) / len(rows)
    avg_floor_keep = sum(r["_h2o_floor_keep"] for r in rows) / len(rows)
    avg_quantized_keep = sum(r["_h2o_quantized_keep"] for r in rows) / len(rows)
    avg_evict_us = sum(r["_h2o_evict_us"] for r in rows) / len(rows)
    avg_codec_us = sum(r["_h2o_codec_us"] for r in rows) / len(rows)
    pareto_rows = sorted(rows, key=lambda r: (r["_h2o_total"], r["_decode_tps"]), reverse=True)[:5]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# H2O v2 Summary\n\n")
        f.write(f"- Rows: {len(rows)}\n")
        f.write(f"- Avg keep ratio: {avg_keep:.4f}\n")
        f.write(f"- Avg lossy ratio: {avg_lossy:.4f}\n")
        f.write(f"- Avg lossless ratio: {avg_lossless:.4f}\n")
        f.write(f"- Avg total ratio (lossy*lossless): {avg_total:.4f}\n")
        f.write(f"- Avg target keep effective: {avg_target_keep_effective:.4f}\n")
        f.write(f"- Avg floor keep (sink+recent): {avg_floor_keep:.4f}\n")
        f.write(f"- Avg quantized keep: {avg_quantized_keep:.4f}\n")
        f.write(f"- Avg evict us: {avg_evict_us:.2f}\n")
        f.write(f"- Avg codec us: {avg_codec_us:.2f}\n\n")

        f.write("## Best Lossy Ratio\n\n")
        f.write(f"- lossy ratio: {best_lossy['_h2o_lossy']:.4f}\n")
        f.write(f"- keep ratio: {best_lossy['_h2o_keep']:.4f}\n")
        f.write(f"- lossless ratio: {best_lossy['_h2o_lossless']:.4f}\n")
        f.write(f"- total ratio: {best_lossy['_h2o_total']:.4f}\n")
        f.write(f"- decode tps: {best_lossy['_decode_tps']:.2f}\n")
        f.write(f"- log: `{best_lossy.get('log_file', '')}`\n\n")

        f.write("## Best Decode TPS\n\n")
        f.write(f"- decode tps: {best_decode['_decode_tps']:.2f}\n")
        f.write(f"- keep ratio: {best_decode['_h2o_keep']:.4f}\n")
        f.write(f"- lossy ratio: {best_decode['_h2o_lossy']:.4f}\n")
        f.write(f"- lossless ratio: {best_decode['_h2o_lossless']:.4f}\n")
        f.write(f"- total ratio: {best_decode['_h2o_total']:.4f}\n")
        f.write(f"- log: `{best_decode.get('log_file', '')}`\n\n")

        f.write("## Best Total Ratio\n\n")
        f.write(f"- total ratio: {best_total['_h2o_total']:.4f}\n")
        f.write(f"- lossy ratio: {best_total['_h2o_lossy']:.4f}\n")
        f.write(f"- lossless ratio: {best_total['_h2o_lossless']:.4f}\n")
        f.write(f"- decode tps: {best_total['_decode_tps']:.2f}\n")
        f.write(f"- log: `{best_total.get('log_file', '')}`\n\n")

        f.write("## Pareto Top-5 (by total ratio then decode)\n\n")
        for i, row in enumerate(pareto_rows, 1):
            f.write(f"{i}. total={row['_h2o_total']:.4f}, decode={row['_decode_tps']:.2f}, ")
            f.write(f"lossy={row['_h2o_lossy']:.4f}, lossless={row['_h2o_lossless']:.4f}, ")
            f.write(f"log=`{row.get('log_file', '')}`\n")
        f.write("\n## Quality Gate\n\n")
        f.write("- quality_status: N/A (pending server-side quality metrics integration)\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
