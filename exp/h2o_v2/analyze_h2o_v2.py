#!/usr/bin/env python3
import argparse
import csv
import json
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


def load_offline_lossless(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"offline lossless json not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to parse offline lossless json: {p} ({exc})")
    ratio = parse_float(obj.get("weighted_lossless_ratio", obj.get("lossless_ratio", 0.0)), 0.0)
    if ratio <= 0.0:
        raise SystemExit(f"invalid offline lossless ratio in {p}")
    return ratio, obj


def main():
    parser = argparse.ArgumentParser(description="Analyze parsed H2O CSV metrics.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--offline-lossless-json", default="", help="Optional offline lossless json.")
    parser.add_argument("--lossy-target", type=float, default=3.0, help="Quality gate for lossy ratio.")
    parser.add_argument("--lossless-target", type=float, default=1.3, help="Quality gate for lossless ratio.")
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
            rows.append(row)

    if not rows:
        raise SystemExit("No rows found in CSV.")

    best_lossy = max(rows, key=lambda r: r["_h2o_lossy"])
    best_decode = max(rows, key=lambda r: r["_decode_tps"])
    avg_keep = sum(r["_h2o_keep"] for r in rows) / len(rows)
    avg_lossy = sum(r["_h2o_lossy"] for r in rows) / len(rows)
    avg_runtime_lossless = sum(r["_h2o_lossless"] for r in rows) / len(rows)
    avg_target_keep_effective = sum(r["_h2o_target_keep_effective"] for r in rows) / len(rows)
    avg_floor_keep = sum(r["_h2o_floor_keep"] for r in rows) / len(rows)
    avg_quantized_keep = sum(r["_h2o_quantized_keep"] for r in rows) / len(rows)
    avg_evict_us = sum(r["_h2o_evict_us"] for r in rows) / len(rows)
    avg_codec_us = sum(r["_h2o_codec_us"] for r in rows) / len(rows)

    offline_lossless_ratio = 0.0
    offline_obj = {}
    if args.offline_lossless_json:
        offline_lossless_ratio, offline_obj = load_offline_lossless(args.offline_lossless_json)
    effective_lossless_ratio = offline_lossless_ratio if offline_lossless_ratio > 0.0 else avg_runtime_lossless
    lossless_source = "offline" if offline_lossless_ratio > 0.0 else "runtime"

    for row in rows:
        row["_h2o_total_effective"] = row["_h2o_lossy"] * effective_lossless_ratio

    best_total = max(rows, key=lambda r: r["_h2o_total_effective"])
    avg_total_effective = sum(r["_h2o_total_effective"] for r in rows) / len(rows)
    pareto_rows = sorted(rows, key=lambda r: (r["_h2o_total_effective"], r["_decode_tps"]), reverse=True)[:5]

    lossy_best = best_lossy["_h2o_lossy"]
    lossy_pass = lossy_best >= args.lossy_target
    lossless_pass = effective_lossless_ratio >= args.lossless_target
    overall_pass = lossy_pass and lossless_pass
    quality_status = "PASS" if overall_pass else "FAIL"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# H2O v2 Summary\n\n")
        f.write(f"- Rows: {len(rows)}\n")
        f.write(f"- Avg keep ratio: {avg_keep:.4f}\n")
        f.write(f"- Avg lossy ratio: {avg_lossy:.4f}\n")
        f.write(f"- Avg runtime lossless ratio: {avg_runtime_lossless:.4f}\n")
        f.write(f"- Effective lossless ratio ({lossless_source}): {effective_lossless_ratio:.4f}\n")
        f.write(f"- Avg total ratio (lossy*lossless): {avg_total_effective:.4f}\n")
        f.write(f"- Avg target keep effective: {avg_target_keep_effective:.4f}\n")
        f.write(f"- Avg floor keep (sink+recent): {avg_floor_keep:.4f}\n")
        f.write(f"- Avg quantized keep: {avg_quantized_keep:.4f}\n")
        f.write(f"- Avg evict us: {avg_evict_us:.2f}\n")
        f.write(f"- Avg codec us: {avg_codec_us:.2f}\n\n")

        f.write("## Best Lossy Ratio\n\n")
        f.write(f"- lossy ratio: {best_lossy['_h2o_lossy']:.4f}\n")
        f.write(f"- keep ratio: {best_lossy['_h2o_keep']:.4f}\n")
        f.write(f"- lossless ratio ({lossless_source}): {effective_lossless_ratio:.4f}\n")
        f.write(f"- runtime lossless ratio: {best_lossy['_h2o_lossless']:.4f}\n")
        f.write(f"- total ratio: {best_lossy['_h2o_total_effective']:.4f}\n")
        f.write(f"- decode tps: {best_lossy['_decode_tps']:.2f}\n")
        f.write(f"- log: `{best_lossy.get('log_file', '')}`\n\n")

        f.write("## Best Decode TPS\n\n")
        f.write(f"- decode tps: {best_decode['_decode_tps']:.2f}\n")
        f.write(f"- keep ratio: {best_decode['_h2o_keep']:.4f}\n")
        f.write(f"- lossy ratio: {best_decode['_h2o_lossy']:.4f}\n")
        f.write(f"- lossless ratio ({lossless_source}): {effective_lossless_ratio:.4f}\n")
        f.write(f"- runtime lossless ratio: {best_decode['_h2o_lossless']:.4f}\n")
        f.write(f"- total ratio: {best_decode['_h2o_total_effective']:.4f}\n")
        f.write(f"- log: `{best_decode.get('log_file', '')}`\n\n")

        f.write("## Best Total Ratio\n\n")
        f.write(f"- total ratio: {best_total['_h2o_total_effective']:.4f}\n")
        f.write(f"- lossy ratio: {best_total['_h2o_lossy']:.4f}\n")
        f.write(f"- lossless ratio ({lossless_source}): {effective_lossless_ratio:.4f}\n")
        f.write(f"- runtime lossless ratio: {best_total['_h2o_lossless']:.4f}\n")
        f.write(f"- decode tps: {best_total['_decode_tps']:.2f}\n")
        f.write(f"- log: `{best_total.get('log_file', '')}`\n\n")

        f.write("## Pareto Top-5 (by total ratio then decode)\n\n")
        for i, row in enumerate(pareto_rows, 1):
            f.write(f"{i}. total={row['_h2o_total_effective']:.4f}, decode={row['_decode_tps']:.2f}, ")
            f.write(f"lossy={row['_h2o_lossy']:.4f}, lossless={effective_lossless_ratio:.4f}, ")
            f.write(f"log=`{row.get('log_file', '')}`\n")
        f.write("\n## Quality Gate\n\n")
        f.write(f"- quality_status: {quality_status}\n")
        f.write(f"- lossy_target: {args.lossy_target:.4f}\n")
        f.write(f"- lossy_best: {lossy_best:.4f}\n")
        f.write(f"- lossy_pass: {'true' if lossy_pass else 'false'}\n")
        f.write(f"- lossless_target: {args.lossless_target:.4f}\n")
        f.write(f"- lossless_source: {lossless_source}\n")
        f.write(f"- lossless_value: {effective_lossless_ratio:.4f}\n")
        f.write(f"- lossless_pass: {'true' if lossless_pass else 'false'}\n")
        f.write(f"- overall_pass: {'true' if overall_pass else 'false'}\n")
        if offline_obj:
            f.write(f"- offline_selected_entries: {int(parse_float(offline_obj.get('selected_entries', 0), 0))}\n")
            f.write(f"- offline_selected_layers: {offline_obj.get('selected_layers', [])}\n")
            f.write(f"- offline_dump_json: `{args.offline_lossless_json}`\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
