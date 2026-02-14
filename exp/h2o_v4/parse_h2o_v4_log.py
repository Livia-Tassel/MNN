#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def parse_markdown_table(text):
    rows = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return rows

    header = [x.strip() for x in lines[0].strip("|").split("|")]
    for ln in lines[2:]:
        parts = [x.strip() for x in ln.strip("|").split("|")]
        if len(parts) != len(header):
            continue
        if all(re.fullmatch(r"[-:]+", p) for p in parts):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def parse_logs(log_files):
    all_rows = []
    for path in log_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rows = parse_markdown_table(text)
        for row in rows:
            row["log_file"] = str(path)
            all_rows.append(row)
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Parse llm_bench markdown logs into CSV for H2O v4.")
    parser.add_argument("--log-dir", default="", help="Directory containing *.log")
    parser.add_argument("--log-files", default="", help="Comma list of log files")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    files = []
    if args.log_dir:
        files.extend(sorted(Path(args.log_dir).glob("*.log")))
    if args.log_files:
        files.extend(Path(x.strip()) for x in args.log_files.split(",") if x.strip())
    files = [f for f in files if f.exists()]
    # Deduplicate: --log-dir and --log-files may reference the same file.
    seen = set()
    deduped = []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    files = deduped
    if not files:
        raise SystemExit("No log files found.")

    rows = parse_logs(files)
    if not rows:
        raise SystemExit("No markdown table rows found in logs.")

    # Keep stable field order: known fields first, then extras.
    known = [
        "model",
        "modelSize",
        "backend",
        "threads",
        "llm_demo",
        "speed(tok/s)",
        "h2o_keep",
        "h2o_lossy",
        "h2o_lossless",
        "h2o_target_keep_effective",
        "h2o_floor_keep",
        "h2o_quantized_keep",
        "h2o_evict_us",
        "h2o_codec_us",
        "h2o_lossless_raw_mb",
        "h2o_lossless_comp_mb",
        "h2o_lossless_comp_us",
        "h2o_lossless_decomp_us",
        "h2o_lossless_fallback",
        "log_file",
    ]
    extra = sorted({k for r in rows for k in r.keys()} - set(known))
    fieldnames = [k for k in known if any(k in r for r in rows)] + extra

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
