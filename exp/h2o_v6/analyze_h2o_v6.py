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
    if not speed_cell:
        return 0.0
    m = re.search(r"<br>\s*([0-9]+(?:\.[0-9]+)?)", speed_cell)
    if not m:
        return 0.0
    return parse_float(m.group(1), 0.0)


def load_lossless_json(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"offline lossless json not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to parse json: {p} ({exc})")
    ratio = parse_float(obj.get("weighted_lossless_ratio", obj.get("lossless_ratio", 0.0)), 0.0)
    if ratio <= 0.0:
        raise SystemExit(f"invalid lossless ratio in {p}")
    return ratio, obj


def format_gate_bool(v):
    return "true" if v else "false"


def main():
    parser = argparse.ArgumentParser(description="Analyze parsed H2O v6 CSV metrics.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--offline-lossless-json", default="", help="Upper-bound offline lossless JSON.")
    parser.add_argument(
        "--offline-lossless-online-json",
        default="",
        help="Online-sim offline lossless JSON (preferred for deployment gate).",
    )
    parser.add_argument("--lossy-target", type=float, default=3.0)
    parser.add_argument("--lossless-target", type=float, default=1.3)
    parser.add_argument("--decode-baseline", type=float, default=0.0, help="Baseline decode TPS for regression gate.")
    parser.add_argument(
        "--decode-baseline-source",
        default="",
        help="Optional decode baseline source label (e.g. fixed, rolling_median, same_batch).",
    )
    parser.add_argument(
        "--decode-baseline-samples",
        type=int,
        default=0,
        help="Optional sample count used to derive decode baseline.",
    )
    parser.add_argument("--decode-drop-target", type=float, default=0.05, help="Max allowed decode TPS drop ratio.")
    parser.add_argument(
        "--max-lossless-queue-peak",
        type=float,
        default=-1.0,
        help="Optional gate: require runtime queue peak <= target (negative disables).",
    )
    parser.add_argument(
        "--max-lossless-fallback",
        type=float,
        default=-1.0,
        help="Optional gate: require runtime fallback count <= target (negative disables).",
    )
    parser.add_argument(
        "--max-lossless-backpressure-skip",
        type=float,
        default=-1.0,
        help="Optional gate: require runtime backpressure skip count <= target (negative disables).",
    )
    parser.add_argument(
        "--max-lossless-decomp-us",
        type=float,
        default=-1.0,
        help="Optional gate: require runtime decomp us <= target (negative disables).",
    )
    parser.add_argument(
        "--max-lossless-async-wait-us",
        type=float,
        default=-1.0,
        help="Optional gate: require runtime async wait us <= target (negative disables).",
    )
    parser.add_argument(
        "--require-runtime-decomp",
        action="store_true",
        help="Require runtime h2o_lossless_decomp_us > 0 for at least one row.",
    )
    parser.add_argument(
        "--require-decode-cache-hit",
        action="store_true",
        help="Require runtime h2o_lossless_decode_cache_hit > 0 for at least one row.",
    )
    parser.add_argument(
        "--require-async-queue-activity",
        action="store_true",
        help="Require runtime h2o_lossless_async_queue_peak > 0 for at least one row.",
    )
    parser.add_argument(
        "--require-decode-cache-activity",
        action="store_true",
        help="Require runtime (decode_cache_hit + decode_cache_miss) > 0 for at least one row.",
    )
    parser.add_argument(
        "--strict-runtime-metric-columns",
        action="store_true",
        help="Fail quality gate when required runtime metric columns are missing in CSV.",
    )
    parser.add_argument("--out", default="exp/h2o_v6/out/summary.md")
    args = parser.parse_args()

    runtime_metric_columns = [
        "h2o_lossless_decomp_us",
        "h2o_lossless_queue_peak",
        "h2o_lossless_fallback",
        "h2o_lossless_backpressure_skip",
        "h2o_lossless_async_queue_peak",
        "h2o_lossless_async_wait_us",
        "h2o_lossless_decode_cache_hit",
        "h2o_lossless_decode_cache_miss",
    ]
    rows = []
    with Path(args.csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_fieldnames = list(reader.fieldnames or [])
        missing_runtime_metric_columns = [c for c in runtime_metric_columns if c not in csv_fieldnames]
        runtime_metric_columns_ok = len(missing_runtime_metric_columns) == 0
        runtime_metric_columns_pass = (
            runtime_metric_columns_ok if args.strict_runtime_metric_columns else True
        )
        if not runtime_metric_columns_ok:
            print(
                "[WARN] runtime metric columns missing in CSV: "
                + ", ".join(missing_runtime_metric_columns)
            )
            print(
                "[WARN] likely using an older llm_bench binary/log format; "
                "rebuild and use the expected v6 llm_bench."
            )
        for row in reader:
            row["_h2o_keep"] = parse_float(row.get("h2o_keep", "0"))
            row["_h2o_lossy"] = parse_float(row.get("h2o_lossy", "0"))
            row["_h2o_lossless_runtime"] = parse_float(row.get("h2o_lossless", "0"))
            row["_h2o_target_keep_effective"] = parse_float(row.get("h2o_target_keep_effective", "0"))
            row["_h2o_floor_keep"] = parse_float(row.get("h2o_floor_keep", "0"))
            row["_h2o_quantized_keep"] = parse_float(row.get("h2o_quantized_keep", "0"))
            row["_h2o_evict_us"] = parse_float(row.get("h2o_evict_us", "0"))
            row["_h2o_codec_us"] = parse_float(row.get("h2o_codec_us", "0"))
            row["_h2o_lossless_raw_mb"] = parse_float(row.get("h2o_lossless_raw_mb", "0"))
            row["_h2o_lossless_comp_mb"] = parse_float(row.get("h2o_lossless_comp_mb", "0"))
            row["_h2o_lossless_comp_us"] = parse_float(row.get("h2o_lossless_comp_us", "0"))
            row["_h2o_lossless_decomp_us"] = parse_float(row.get("h2o_lossless_decomp_us", "0"))
            row["_h2o_lossless_queue_peak"] = parse_float(row.get("h2o_lossless_queue_peak", "0"))
            row["_h2o_lossless_fallback"] = parse_float(row.get("h2o_lossless_fallback", "0"))
            row["_h2o_lossless_backpressure_skip"] = parse_float(row.get("h2o_lossless_backpressure_skip", "0"))
            row["_h2o_lossless_async_queue_peak"] = parse_float(row.get("h2o_lossless_async_queue_peak", "0"))
            row["_h2o_lossless_async_wait_us"] = parse_float(row.get("h2o_lossless_async_wait_us", "0"))
            row["_h2o_lossless_decode_cache_hit"] = parse_float(row.get("h2o_lossless_decode_cache_hit", "0"))
            row["_h2o_lossless_decode_cache_miss"] = parse_float(row.get("h2o_lossless_decode_cache_miss", "0"))
            row["_decode_tps"] = parse_decode_speed(row.get("speed(tok/s)", ""))
            rows.append(row)

    if not rows:
        raise SystemExit("No rows found in CSV.")

    avg_keep = sum(r["_h2o_keep"] for r in rows) / len(rows)
    avg_lossy = sum(r["_h2o_lossy"] for r in rows) / len(rows)
    avg_runtime_lossless = sum(r["_h2o_lossless_runtime"] for r in rows) / len(rows)
    avg_target_keep_effective = sum(r["_h2o_target_keep_effective"] for r in rows) / len(rows)
    avg_floor_keep = sum(r["_h2o_floor_keep"] for r in rows) / len(rows)
    avg_quantized_keep = sum(r["_h2o_quantized_keep"] for r in rows) / len(rows)
    avg_evict_us = sum(r["_h2o_evict_us"] for r in rows) / len(rows)
    avg_codec_us = sum(r["_h2o_codec_us"] for r in rows) / len(rows)
    avg_lossless_raw_mb = sum(r["_h2o_lossless_raw_mb"] for r in rows) / len(rows)
    avg_lossless_comp_mb = sum(r["_h2o_lossless_comp_mb"] for r in rows) / len(rows)
    avg_lossless_comp_us = sum(r["_h2o_lossless_comp_us"] for r in rows) / len(rows)
    avg_lossless_decomp_us = sum(r["_h2o_lossless_decomp_us"] for r in rows) / len(rows)
    avg_lossless_queue_peak = sum(r["_h2o_lossless_queue_peak"] for r in rows) / len(rows)
    avg_lossless_fallback = sum(r["_h2o_lossless_fallback"] for r in rows) / len(rows)
    avg_lossless_backpressure_skip = sum(r["_h2o_lossless_backpressure_skip"] for r in rows) / len(rows)
    avg_lossless_async_queue_peak = sum(r["_h2o_lossless_async_queue_peak"] for r in rows) / len(rows)
    avg_lossless_async_wait_us = sum(r["_h2o_lossless_async_wait_us"] for r in rows) / len(rows)
    avg_lossless_decode_cache_hit = sum(r["_h2o_lossless_decode_cache_hit"] for r in rows) / len(rows)
    avg_lossless_decode_cache_miss = sum(r["_h2o_lossless_decode_cache_miss"] for r in rows) / len(rows)

    upper_lossless = 0.0
    upper_obj = {}
    if args.offline_lossless_json:
        upper_lossless, upper_obj = load_lossless_json(args.offline_lossless_json)

    online_lossless = 0.0
    online_obj = {}
    if args.offline_lossless_online_json:
        online_lossless, online_obj = load_lossless_json(args.offline_lossless_online_json)

    if online_lossless > 0.0:
        selected_lossless = online_lossless
        selected_source = "offline_online_sim"
    elif upper_lossless > 0.0:
        selected_lossless = upper_lossless
        selected_source = "offline_upper_bound"
        print("[WARN] online-sim lossless JSON not provided or invalid; "
              "falling back to upper-bound. Gate result may be unreliable.")
    else:
        selected_lossless = avg_runtime_lossless
        selected_source = "runtime"
        print("[WARN] no offline lossless JSON provided; "
              "falling back to runtime value. Gate result may be unreliable.")

    for row in rows:
        row["_total_selected"] = row["_h2o_lossy"] * selected_lossless
        row["_total_upper"] = row["_h2o_lossy"] * (upper_lossless if upper_lossless > 0.0 else selected_lossless)
        row["_total_online"] = row["_h2o_lossy"] * (online_lossless if online_lossless > 0.0 else selected_lossless)

    best_lossy = max(rows, key=lambda r: r["_h2o_lossy"])
    best_decode = max(rows, key=lambda r: r["_decode_tps"])
    best_total_selected = max(rows, key=lambda r: r["_total_selected"])
    pareto_rows = sorted(rows, key=lambda r: (r["_total_selected"], r["_decode_tps"]), reverse=True)[:5]

    lossy_best = best_lossy["_h2o_lossy"]
    lossy_pass = lossy_best >= args.lossy_target

    selected_lossless_pass = selected_lossless >= args.lossless_target
    online_gate_lossless = online_lossless if online_lossless > 0.0 else selected_lossless
    online_lossless_pass = online_gate_lossless >= args.lossless_target
    runtime_decomp_best = max(r["_h2o_lossless_decomp_us"] for r in rows)
    runtime_queue_peak_best = max(r["_h2o_lossless_queue_peak"] for r in rows)
    runtime_fallback_best = max(r["_h2o_lossless_fallback"] for r in rows)
    runtime_backpressure_skip_best = max(r["_h2o_lossless_backpressure_skip"] for r in rows)
    runtime_async_queue_peak_best = max(r["_h2o_lossless_async_queue_peak"] for r in rows)
    runtime_async_wait_best = max(r["_h2o_lossless_async_wait_us"] for r in rows)
    runtime_decode_cache_hit_best = max(r["_h2o_lossless_decode_cache_hit"] for r in rows)
    runtime_decode_cache_miss_best = max(r["_h2o_lossless_decode_cache_miss"] for r in rows)
    runtime_decode_cache_activity_best = max(
        (r["_h2o_lossless_decode_cache_hit"] + r["_h2o_lossless_decode_cache_miss"])
        for r in rows
    )
    runtime_decomp_required_pass = (runtime_decomp_best > 0.0) if args.require_runtime_decomp else True
    runtime_decode_cache_required_pass = (
        runtime_decode_cache_hit_best > 0.0
        if args.require_decode_cache_hit
        else True
    )
    runtime_async_queue_activity_pass = (
        runtime_async_queue_peak_best > 0.0
        if args.require_async_queue_activity
        else True
    )
    runtime_decode_cache_activity_pass = (
        runtime_decode_cache_activity_best > 0.0
        if args.require_decode_cache_activity
        else True
    )
    decomp_gate_enabled = args.max_lossless_decomp_us >= 0.0
    async_wait_gate_enabled = args.max_lossless_async_wait_us >= 0.0
    runtime_decomp_budget_pass = (
        runtime_decomp_best <= args.max_lossless_decomp_us
        if decomp_gate_enabled
        else True
    )
    runtime_decomp_pass = runtime_decomp_required_pass and runtime_decomp_budget_pass
    runtime_async_wait_pass = (
        runtime_async_wait_best <= args.max_lossless_async_wait_us
        if async_wait_gate_enabled
        else True
    )
    queue_peak_gate_enabled = args.max_lossless_queue_peak >= 0.0
    fallback_gate_enabled = args.max_lossless_fallback >= 0.0
    backpressure_skip_gate_enabled = args.max_lossless_backpressure_skip >= 0.0
    runtime_queue_peak_pass = (
        runtime_queue_peak_best <= args.max_lossless_queue_peak
        if queue_peak_gate_enabled
        else True
    )
    runtime_fallback_pass = (
        runtime_fallback_best <= args.max_lossless_fallback
        if fallback_gate_enabled
        else True
    )
    runtime_backpressure_skip_pass = (
        runtime_backpressure_skip_best <= args.max_lossless_backpressure_skip
        if backpressure_skip_gate_enabled
        else True
    )
    decode_drop_ratio = 0.0
    decode_pass = True
    if args.decode_baseline > 0.0:
        decode_drop_ratio = (args.decode_baseline - best_decode["_decode_tps"]) / args.decode_baseline
        decode_pass = decode_drop_ratio <= args.decode_drop_target
    overall_pass = (
        lossy_pass
        and online_lossless_pass
        and decode_pass
        and runtime_metric_columns_pass
        and runtime_decomp_pass
        and runtime_async_wait_pass
        and runtime_async_queue_activity_pass
        and runtime_decode_cache_required_pass
        and runtime_decode_cache_activity_pass
        and runtime_queue_peak_pass
        and runtime_fallback_pass
        and runtime_backpressure_skip_pass
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# H2O v6 Summary\n\n")
        f.write(f"- Rows: {len(rows)}\n")
        f.write(f"- Avg keep ratio: {avg_keep:.4f}\n")
        f.write(f"- Avg lossy ratio: {avg_lossy:.4f}\n")
        f.write(f"- Avg runtime lossless ratio: {avg_runtime_lossless:.4f}\n")
        f.write(f"- Lossless selected source: {selected_source}\n")
        f.write(f"- Lossless selected value: {selected_lossless:.4f}\n")
        if upper_lossless > 0.0:
            f.write(f"- Offline upper-bound lossless: {upper_lossless:.4f}\n")
        if online_lossless > 0.0:
            f.write(f"- Offline online-sim lossless: {online_lossless:.4f}\n")
        f.write(f"- Avg total ratio (selected): {sum(r['_total_selected'] for r in rows)/len(rows):.4f}\n")
        f.write(f"- Avg target keep effective: {avg_target_keep_effective:.4f}\n")
        f.write(f"- Avg floor keep (sink+recent): {avg_floor_keep:.4f}\n")
        f.write(f"- Avg quantized keep: {avg_quantized_keep:.4f}\n")
        f.write(f"- Avg evict us: {avg_evict_us:.2f}\n")
        f.write(f"- Avg codec us: {avg_codec_us:.2f}\n\n")
        f.write(f"- Avg lossless raw MB: {avg_lossless_raw_mb:.4f}\n")
        f.write(f"- Avg lossless comp MB: {avg_lossless_comp_mb:.4f}\n")
        f.write(f"- Avg lossless comp us: {avg_lossless_comp_us:.2f}\n")
        f.write(f"- Avg lossless decomp us: {avg_lossless_decomp_us:.2f}\n")
        f.write(f"- Avg lossless queue peak: {avg_lossless_queue_peak:.2f}\n")
        f.write(f"- Avg lossless fallback: {avg_lossless_fallback:.2f}\n\n")
        f.write(f"- Avg lossless backpressure skip: {avg_lossless_backpressure_skip:.2f}\n\n")
        f.write(f"- Avg lossless async queue peak: {avg_lossless_async_queue_peak:.2f}\n")
        f.write(f"- Avg lossless async wait us: {avg_lossless_async_wait_us:.2f}\n")
        f.write(f"- Avg lossless decode cache hit: {avg_lossless_decode_cache_hit:.2f}\n")
        f.write(f"- Avg lossless decode cache miss: {avg_lossless_decode_cache_miss:.2f}\n\n")

        f.write("## Best Lossy Ratio\n\n")
        f.write(f"- lossy ratio: {best_lossy['_h2o_lossy']:.4f}\n")
        f.write(f"- keep ratio: {best_lossy['_h2o_keep']:.4f}\n")
        f.write(f"- lossless ratio ({selected_source}): {selected_lossless:.4f}\n")
        f.write(f"- runtime lossless ratio: {best_lossy['_h2o_lossless_runtime']:.4f}\n")
        f.write(f"- total ratio: {best_lossy['_total_selected']:.4f}\n")
        f.write(f"- decode tps: {best_lossy['_decode_tps']:.2f}\n")
        f.write(f"- log: `{best_lossy.get('log_file', '')}`\n\n")

        f.write("## Best Decode TPS\n\n")
        f.write(f"- decode tps: {best_decode['_decode_tps']:.2f}\n")
        f.write(f"- keep ratio: {best_decode['_h2o_keep']:.4f}\n")
        f.write(f"- lossy ratio: {best_decode['_h2o_lossy']:.4f}\n")
        f.write(f"- lossless ratio ({selected_source}): {selected_lossless:.4f}\n")
        f.write(f"- runtime lossless ratio: {best_decode['_h2o_lossless_runtime']:.4f}\n")
        f.write(f"- total ratio: {best_decode['_total_selected']:.4f}\n")
        f.write(f"- log: `{best_decode.get('log_file', '')}`\n\n")

        f.write("## Best Total Ratio\n\n")
        f.write(f"- total ratio: {best_total_selected['_total_selected']:.4f}\n")
        f.write(f"- lossy ratio: {best_total_selected['_h2o_lossy']:.4f}\n")
        f.write(f"- lossless ratio ({selected_source}): {selected_lossless:.4f}\n")
        f.write(f"- runtime lossless ratio: {best_total_selected['_h2o_lossless_runtime']:.4f}\n")
        f.write(f"- decode tps: {best_total_selected['_decode_tps']:.2f}\n")
        f.write(f"- log: `{best_total_selected.get('log_file', '')}`\n\n")

        f.write("## Pareto Top-5 (by selected total then decode)\n\n")
        for idx, row in enumerate(pareto_rows, 1):
            f.write(
                f"{idx}. total={row['_total_selected']:.4f}, decode={row['_decode_tps']:.2f}, "
                f"lossy={row['_h2o_lossy']:.4f}, lossless={selected_lossless:.4f}, "
                f"log=`{row.get('log_file', '')}`\n"
            )

        f.write("\n## Quality Gate\n\n")
        f.write(f"- quality_status: {'PASS' if overall_pass else 'FAIL'}\n")
        f.write(f"- lossy_target: {args.lossy_target:.4f}\n")
        f.write(f"- lossy_best: {lossy_best:.4f}\n")
        f.write(f"- lossy_pass: {format_gate_bool(lossy_pass)}\n")
        f.write(f"- lossless_target: {args.lossless_target:.4f}\n")
        f.write(f"- lossless_selected_source: {selected_source}\n")
        f.write(f"- lossless_selected_value: {selected_lossless:.4f}\n")
        f.write(f"- lossless_selected_pass: {format_gate_bool(selected_lossless_pass)}\n")
        f.write(f"- lossless_online_value: {online_gate_lossless:.4f}\n")
        f.write(f"- lossless_online_pass: {format_gate_bool(online_lossless_pass)}\n")
        f.write(f"- runtime_decomp_required: {format_gate_bool(args.require_runtime_decomp)}\n")
        f.write(f"- runtime_decomp_best_us: {runtime_decomp_best:.4f}\n")
        f.write(
            f"- runtime_metric_columns_strict: {format_gate_bool(args.strict_runtime_metric_columns)}\n"
        )
        f.write(f"- runtime_metric_columns_ok: {format_gate_bool(runtime_metric_columns_ok)}\n")
        f.write(
            f"- runtime_metric_columns_pass: {format_gate_bool(runtime_metric_columns_pass)}\n"
        )
        f.write(f"- runtime_metric_missing_columns: {missing_runtime_metric_columns}\n")
        f.write(f"- runtime_decomp_gate_enabled: {format_gate_bool(decomp_gate_enabled)}\n")
        if decomp_gate_enabled:
            f.write(f"- runtime_decomp_target_us: {args.max_lossless_decomp_us:.4f}\n")
        f.write(f"- runtime_async_queue_peak_best: {runtime_async_queue_peak_best:.4f}\n")
        f.write(
            f"- runtime_async_queue_activity_required: {format_gate_bool(args.require_async_queue_activity)}\n"
        )
        f.write(f"- runtime_async_wait_best_us: {runtime_async_wait_best:.4f}\n")
        f.write(f"- runtime_async_wait_gate_enabled: {format_gate_bool(async_wait_gate_enabled)}\n")
        if async_wait_gate_enabled:
            f.write(f"- runtime_async_wait_target_us: {args.max_lossless_async_wait_us:.4f}\n")
        f.write(f"- runtime_decode_cache_hit_best: {runtime_decode_cache_hit_best:.4f}\n")
        f.write(f"- runtime_decode_cache_miss_best: {runtime_decode_cache_miss_best:.4f}\n")
        f.write(f"- runtime_decode_cache_activity_best: {runtime_decode_cache_activity_best:.4f}\n")
        f.write(f"- runtime_decode_cache_required: {format_gate_bool(args.require_decode_cache_hit)}\n")
        f.write(
            f"- runtime_decode_cache_activity_required: "
            f"{format_gate_bool(args.require_decode_cache_activity)}\n"
        )
        f.write(f"- runtime_decomp_required_pass: {format_gate_bool(runtime_decomp_required_pass)}\n")
        f.write(f"- runtime_decomp_budget_pass: {format_gate_bool(runtime_decomp_budget_pass)}\n")
        f.write(
            f"- runtime_async_queue_activity_pass: "
            f"{format_gate_bool(runtime_async_queue_activity_pass)}\n"
        )
        f.write(f"- runtime_async_wait_pass: {format_gate_bool(runtime_async_wait_pass)}\n")
        f.write(f"- runtime_decode_cache_required_pass: {format_gate_bool(runtime_decode_cache_required_pass)}\n")
        f.write(
            f"- runtime_decode_cache_activity_pass: "
            f"{format_gate_bool(runtime_decode_cache_activity_pass)}\n"
        )
        f.write(f"- runtime_queue_peak_best: {runtime_queue_peak_best:.4f}\n")
        f.write(f"- runtime_fallback_best: {runtime_fallback_best:.4f}\n")
        f.write(f"- runtime_backpressure_skip_best: {runtime_backpressure_skip_best:.4f}\n")
        f.write(f"- runtime_queue_peak_gate_enabled: {format_gate_bool(queue_peak_gate_enabled)}\n")
        if queue_peak_gate_enabled:
            f.write(f"- runtime_queue_peak_target: {args.max_lossless_queue_peak:.4f}\n")
        f.write(f"- runtime_queue_peak_pass: {format_gate_bool(runtime_queue_peak_pass)}\n")
        f.write(f"- runtime_fallback_gate_enabled: {format_gate_bool(fallback_gate_enabled)}\n")
        if fallback_gate_enabled:
            f.write(f"- runtime_fallback_target: {args.max_lossless_fallback:.4f}\n")
        f.write(f"- runtime_fallback_pass: {format_gate_bool(runtime_fallback_pass)}\n")
        f.write(
            f"- runtime_backpressure_skip_gate_enabled: "
            f"{format_gate_bool(backpressure_skip_gate_enabled)}\n"
        )
        if backpressure_skip_gate_enabled:
            f.write(
                f"- runtime_backpressure_skip_target: "
                f"{args.max_lossless_backpressure_skip:.4f}\n"
            )
        f.write(
            f"- runtime_backpressure_skip_pass: "
            f"{format_gate_bool(runtime_backpressure_skip_pass)}\n"
        )
        f.write(f"- runtime_decomp_pass: {format_gate_bool(runtime_decomp_pass)}\n")
        f.write(f"- overall_pass: {format_gate_bool(overall_pass)}\n")
        if args.decode_baseline > 0.0:
            f.write(f"- decode_baseline: {args.decode_baseline:.4f}\n")
            if args.decode_baseline_source:
                f.write(f"- decode_baseline_source: {args.decode_baseline_source}\n")
            if args.decode_baseline_samples > 0:
                f.write(f"- decode_baseline_samples: {int(args.decode_baseline_samples)}\n")
            f.write(f"- decode_best: {best_decode['_decode_tps']:.4f}\n")
            f.write(f"- decode_drop_ratio: {decode_drop_ratio:.6f}\n")
            f.write(f"- decode_drop_target: {args.decode_drop_target:.6f}\n")
            f.write(f"- decode_pass: {format_gate_bool(decode_pass)}\n")
        if upper_obj:
            f.write(f"- upper_selected_entries: {int(parse_float(upper_obj.get('selected_entries', 0), 0))}\n")
            f.write(f"- upper_selected_layers: {upper_obj.get('selected_layers', [])}\n")
            f.write(f"- upper_json: `{args.offline_lossless_json}`\n")
        if online_obj:
            f.write(f"- online_selected_entries: {int(parse_float(online_obj.get('selected_entries', 0), 0))}\n")
            f.write(f"- online_selected_layers: {online_obj.get('selected_layers', [])}\n")
            f.write(f"- online_json: `{args.offline_lossless_online_json}`\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
