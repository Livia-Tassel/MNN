#!/usr/bin/env python3
import argparse
import collections
import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path


PROMPT_TOKENS_RE = re.compile(r"prompt tokens num = (\d+)")
DECODE_TOKENS_RE = re.compile(r"decode tokens num = (\d+)")
PREFILL_TIME_RE = re.compile(r"prefill time = ([\d\.]+) s")
DECODE_TIME_RE = re.compile(r"decode time = ([\d\.]+) s")
PROMPT_BUCKET_RE = re.compile(r"prompt_(\d+)_\d+\.txt$", re.IGNORECASE)


def parse_cli_metrics(output: str):
    metrics = {}
    m = PROMPT_TOKENS_RE.search(output)
    if m:
        metrics["prompt_tokens"] = int(m.group(1))
    m = DECODE_TOKENS_RE.search(output)
    if m:
        metrics["decode_tokens"] = int(m.group(1))
    m = PREFILL_TIME_RE.search(output)
    if m:
        metrics["prefill_s"] = float(m.group(1))
    m = DECODE_TIME_RE.search(output)
    if m:
        metrics["decode_s"] = float(m.group(1))
    return metrics


def load_metrics_jsonl(path: Path):
    if not path.exists():
        return {}
    try:
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            return {}
        return json.loads(lines[-1])
    except Exception:
        return {}

def summarize_returncode(rc: int) -> str:
    if rc == 0:
        return "ok"
    if rc < 0:
        return f"signal_{-rc}"
    return f"rc_{rc}"


def safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return a / b


def avg_or_zero(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def coerce_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def coerce_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def decode_output(data) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    # llm_demo may emit non-UTF8 bytes in generated text; keep run alive.
    return data.decode("utf-8", errors="replace")


def has_cli_summary_metrics(cli_metrics: dict) -> bool:
    return (
        int(cli_metrics.get("prompt_tokens", 0)) > 0
        and float(cli_metrics.get("prefill_s", 0.0)) > 0.0
    )


def has_jsonl_summary_metrics(h2o_metrics: dict) -> bool:
    if not isinstance(h2o_metrics, dict) or not h2o_metrics:
        return False
    return (
        coerce_int(h2o_metrics.get("prompt_tokens", 0)) > 0
        and coerce_float(h2o_metrics.get("prefill_us", 0.0)) > 0.0
    )


def collect_h2o_values(rows, key):
    values = []
    for row in rows:
        h2o = row.get("h2o_metrics", {})
        if not isinstance(h2o, dict):
            continue
        values.append(coerce_float(h2o.get(key, 0.0)))
    return values


def resolve_manifest_prompt_path(raw_path: str, prompt_dir: Path) -> Path:
    p = Path(raw_path)
    if not p.is_absolute():
        p = (prompt_dir / p).resolve()
    else:
        p = p.resolve()
    return p


def extract_manifest_path(obj: dict):
    keys = (
        "prompt_file",
        "prompt_path",
        "file",
        "path",
        "filename",
        "name",
    )
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def load_prompt_files_from_manifest(manifest_path: Path, prompt_dir: Path):
    if not manifest_path.exists():
        raise SystemExit(f"prompt manifest not found: {manifest_path}")
    files = []
    seen = set()
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        candidate = ""
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    candidate = extract_manifest_path(obj)
            except Exception:
                candidate = ""
        if not candidate:
            candidate = line
        p = resolve_manifest_prompt_path(candidate, prompt_dir)
        if p.exists() and p.is_file():
            key = os.fspath(p)
            if key not in seen:
                seen.add(key)
                files.append(p)
    return files


def parse_bucket_list(text: str):
    if not text:
        return []
    buckets = []
    seen = set()
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            buckets.append(item)
    return buckets


def infer_prompt_bucket(prompt_file: Path):
    m = PROMPT_BUCKET_RE.search(prompt_file.name)
    if m:
        return m.group(1)
    return "other"


def bucket_sort_key(bucket: str):
    try:
        return (0, int(bucket))
    except Exception:
        return (1, bucket)


def build_bucket_map(files):
    bucket_map = collections.defaultdict(list)
    for path in files:
        bucket_map[infer_prompt_bucket(path)].append(path)
    return bucket_map


def build_bucket_order(bucket_map, requested_buckets):
    order = []
    for bucket in requested_buckets:
        if bucket in bucket_map and bucket not in order:
            order.append(bucket)
    remaining = [b for b in bucket_map.keys() if b not in order]
    remaining = sorted(remaining, key=bucket_sort_key)
    order.extend(remaining)
    return order


def select_prompt_files(bucket_map, bucket_order, sample_mode, max_prompts, max_prompts_per_bucket):
    capped = {}
    for bucket in bucket_order:
        files = list(bucket_map.get(bucket, []))
        if max_prompts_per_bucket > 0:
            files = files[:max_prompts_per_bucket]
        capped[bucket] = files

    selected = []
    if sample_mode == "sequential":
        for bucket in bucket_order:
            selected.extend(capped[bucket])
            if max_prompts > 0 and len(selected) >= max_prompts:
                return selected[:max_prompts]
        return selected

    # default: stratified round-robin to avoid one bucket dominating the run.
    cursor = {bucket: 0 for bucket in bucket_order}
    while True:
        progressed = False
        for bucket in bucket_order:
            idx = cursor[bucket]
            files = capped[bucket]
            if idx >= len(files):
                continue
            selected.append(files[idx])
            cursor[bucket] = idx + 1
            progressed = True
            if max_prompts > 0 and len(selected) >= max_prompts:
                return selected
        if not progressed:
            break
    return selected


def summarize_rows(rows):
    decode_tps_values = [float(r.get("decode_tps", 0.0)) for r in rows if float(r.get("decode_tps", 0.0)) > 0.0]
    keep_values = []
    lossy_values = []
    lossless_values = []
    runtime_total_values = []
    raw_mb_values = []
    comp_mb_values = []
    decomp_values = []
    cache_hits = []
    for row in rows:
        h2o = row.get("h2o_metrics", {})
        if not isinstance(h2o, dict):
            continue
        keep = coerce_float(h2o.get("h2o_keep_ratio", 0.0))
        lossy = coerce_float(h2o.get("h2o_lossy_ratio", 0.0))
        lossless = coerce_float(h2o.get("h2o_lossless_ratio", 0.0))
        keep_values.append(keep)
        lossy_values.append(lossy)
        lossless_values.append(lossless)
        runtime_total_values.append(lossy * lossless)
        raw_mb_values.append(coerce_float(h2o.get("h2o_lossless_raw_bytes", 0.0)) / 1024.0 / 1024.0)
        comp_mb_values.append(coerce_float(h2o.get("h2o_lossless_compressed_bytes", 0.0)) / 1024.0 / 1024.0)
        decomp_values.append(coerce_float(h2o.get("h2o_lossless_decompress_us", 0.0)))
        cache_hits.append(coerce_float(h2o.get("h2o_lossless_decode_cache_hit", 0.0)))

    return {
        "total_runs": len(rows),
        "pass_runs": sum(1 for r in rows if int(r.get("effective_returncode", 1)) == 0),
        "decode_tps_avg": avg_or_zero(decode_tps_values),
        "decode_tps_min": min(decode_tps_values) if decode_tps_values else 0.0,
        "decode_tps_max": max(decode_tps_values) if decode_tps_values else 0.0,
        "h2o_keep_ratio_avg": avg_or_zero(keep_values),
        "h2o_lossy_ratio_avg": avg_or_zero(lossy_values),
        "h2o_lossless_ratio_avg": avg_or_zero(lossless_values),
        "h2o_runtime_total_ratio_avg": avg_or_zero(runtime_total_values),
        "h2o_lossless_raw_mb_avg": avg_or_zero(raw_mb_values),
        "h2o_lossless_comp_mb_avg": avg_or_zero(comp_mb_values),
        "runtime_decomp_us_max": max(decomp_values) if decomp_values else 0.0,
        "decode_cache_hit_max": max(cache_hits) if cache_hits else 0.0,
        "overall_pass": len(rows) > 0 and all(int(r.get("effective_returncode", 1)) == 0 for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Run llm_demo with real prompt files and collect structured metrics.")
    parser.add_argument("--llm-demo", default="./build/llm_demo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--prompt-pattern", default="prompt_*.txt")
    parser.add_argument("--prompt-manifest", default="", help="Optional manifest (.jsonl/.txt) listing prompt files.")
    parser.add_argument("--max-prompts", type=int, default=0, help="Optional cap on number of prompts to run.")
    parser.add_argument("--max-prompts-per-bucket", type=int, default=0, help="Optional per-bucket cap.")
    parser.add_argument("--bucket-list", default="", help="Optional preferred bucket order, e.g. 128,512,2048.")
    parser.add_argument("--sample-mode", choices=["stratified", "sequential"], default="stratified")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--out-dir", default="exp/h2o_final/out_llm_demo")
    parser.add_argument("--run-tag", default="candidate")
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    if args.prompt_manifest:
        files = load_prompt_files_from_manifest(Path(args.prompt_manifest), prompt_dir)
        if args.prompt_pattern:
            files = [f for f in files if fnmatch.fnmatch(f.name, args.prompt_pattern)]
    else:
        files = sorted(prompt_dir.glob(args.prompt_pattern))

    bucket_map = build_bucket_map(files)
    bucket_order = build_bucket_order(bucket_map, parse_bucket_list(args.bucket_list))
    files = select_prompt_files(
        bucket_map=bucket_map,
        bucket_order=bucket_order,
        sample_mode=args.sample_mode,
        max_prompts=int(args.max_prompts),
        max_prompts_per_bucket=int(args.max_prompts_per_bucket),
    )
    if not files:
        raise SystemExit(
            f"No prompt files found under {prompt_dir} with pattern {args.prompt_pattern}"
            + (f" and manifest {args.prompt_manifest}" if args.prompt_manifest else "")
        )

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs_jsonl = out_dir / f"{args.run_tag}_runs.jsonl"
    summary_json = out_dir / f"{args.run_tag}_summary.json"
    summary_md = out_dir / f"{args.run_tag}_summary.md"
    if runs_jsonl.exists():
        runs_jsonl.unlink()

    rows = []
    for idx, prompt_file in enumerate(files, 1):
        run_id = f"{args.run_tag}_{prompt_file.stem}"
        metrics_path = metrics_dir / f"{run_id}.jsonl"
        stdout_path = logs_dir / f"{run_id}.stdout.log"
        stderr_path = logs_dir / f"{run_id}.stderr.log"
        tmp_run_dir = out_dir / "tmp" / run_id
        tmp_run_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["LLM_DEMO_METRICS_JSONL"] = str(metrics_path)
        # Keep per-prompt process artifacts isolated (tmp/tuning/cache files).
        env["LLM_DEMO_TMP_PATH"] = str(tmp_run_dir)
        cmd = [
            args.llm_demo,
            args.config,
            str(prompt_file),
            str(args.decode_tokens),
            "--metrics-jsonl",
            str(metrics_path),
            "--prompt-file-mode=whole",
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True)
        stdout = decode_output(proc.stdout)
        stderr = decode_output(proc.stderr)
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        cli_metrics = parse_cli_metrics(stdout + "\n" + stderr)
        h2o_metrics = load_metrics_jsonl(metrics_path)

        effective_rc = int(proc.returncode)
        reason = summarize_returncode(proc.returncode)
        if "LLM init error" in (stdout + "\n" + stderr):
            effective_rc = 1
            reason = "llm_init_error"
        has_cli_summary = has_cli_summary_metrics(cli_metrics)
        has_jsonl_summary = has_jsonl_summary_metrics(h2o_metrics)
        if effective_rc == 0 and not (has_cli_summary or has_jsonl_summary):
            # Treat "instant success with no metrics" as invalid run.
            effective_rc = 2
            reason = "missing_summary_metrics"

        cli_prompt_tokens = int(cli_metrics.get("prompt_tokens", 0))
        cli_decode_tokens = int(cli_metrics.get("decode_tokens", 0))
        cli_prefill_s = float(cli_metrics.get("prefill_s", 0.0))
        cli_decode_s = float(cli_metrics.get("decode_s", 0.0))

        jsonl_prompt_tokens = coerce_int(h2o_metrics.get("prompt_tokens", 0))
        jsonl_decode_tokens = coerce_int(h2o_metrics.get("decode_tokens", 0))
        jsonl_prefill_s = coerce_float(h2o_metrics.get("prefill_us", 0.0)) / 1e6
        jsonl_decode_s = coerce_float(h2o_metrics.get("decode_us", 0.0)) / 1e6

        # Prefer structured microsecond metrics from llm_demo JSONL.
        # CLI seconds are only 2-decimal formatted and may become 0.00 for short runs.
        prompt_tokens = jsonl_prompt_tokens if jsonl_prompt_tokens > 0 else cli_prompt_tokens
        decode_tokens = jsonl_decode_tokens if jsonl_decode_tokens > 0 else cli_decode_tokens
        prefill_s = jsonl_prefill_s if jsonl_prefill_s > 0 else cli_prefill_s
        decode_s = jsonl_decode_s if jsonl_decode_s > 0 else cli_decode_s
        row = {
            "run_index": idx,
            "run_tag": args.run_tag,
            "prompt_file": str(prompt_file),
            "prompt_bucket": infer_prompt_bucket(prompt_file),
            "returncode": proc.returncode,
            "effective_returncode": effective_rc,
            "status_reason": reason,
            "has_cli_summary_metrics": has_cli_summary,
            "has_jsonl_summary_metrics": has_jsonl_summary,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "prefill_tps": safe_div(prompt_tokens, prefill_s),
            "decode_tps": safe_div(decode_tokens, decode_s),
            "h2o_metrics": h2o_metrics,
            "cli_prompt_tokens": cli_prompt_tokens,
            "cli_decode_tokens": cli_decode_tokens,
            "cli_prefill_s": cli_prefill_s,
            "cli_decode_s": cli_decode_s,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "metrics_jsonl": str(metrics_path),
        }
        rows.append(row)
        with runs_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    bucket_stats = {}
    ordered_buckets = []
    for bucket in build_bucket_order(build_bucket_map(files), parse_bucket_list(args.bucket_list)):
        subset = [r for r in rows if r.get("prompt_bucket") == bucket]
        if not subset:
            continue
        bucket_stats[bucket] = summarize_rows(subset)
        ordered_buckets.append(bucket)

    summary_metrics = summarize_rows(rows)

    summary = {
        "run_tag": args.run_tag,
        "prompt_pattern": args.prompt_pattern,
        "prompt_manifest": args.prompt_manifest,
        "max_prompts": int(args.max_prompts),
        "max_prompts_per_bucket": int(args.max_prompts_per_bucket),
        "sample_mode": args.sample_mode,
        "bucket_list": parse_bucket_list(args.bucket_list),
        "bucket_order": ordered_buckets,
        "total_runs": summary_metrics["total_runs"],
        "pass_runs": summary_metrics["pass_runs"],
        "decode_tps_avg": summary_metrics["decode_tps_avg"],
        "decode_tps_min": summary_metrics["decode_tps_min"],
        "decode_tps_max": summary_metrics["decode_tps_max"],
        "h2o_keep_ratio_avg": summary_metrics["h2o_keep_ratio_avg"],
        "h2o_lossy_ratio_avg": summary_metrics["h2o_lossy_ratio_avg"],
        "h2o_lossless_ratio_avg": summary_metrics["h2o_lossless_ratio_avg"],
        "h2o_runtime_total_ratio_avg": summary_metrics["h2o_runtime_total_ratio_avg"],
        "h2o_lossless_raw_mb_avg": summary_metrics["h2o_lossless_raw_mb_avg"],
        "h2o_lossless_comp_mb_avg": summary_metrics["h2o_lossless_comp_mb_avg"],
        "runtime_decomp_us_max": summary_metrics["runtime_decomp_us_max"],
        "decode_cache_hit_max": summary_metrics["decode_cache_hit_max"],
        "overall_pass": summary_metrics["overall_pass"],
        "bucket_stats": bucket_stats,
        "runs_jsonl": str(runs_jsonl),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# LLM Demo Real Prompt Summary ({args.run_tag})")
    lines.append("")
    lines.append(f"- total_runs: {summary['total_runs']}")
    lines.append(f"- pass_runs: {summary['pass_runs']}")
    lines.append(f"- overall_pass: {str(summary['overall_pass']).lower()}")
    lines.append(f"- prompt_pattern: {summary['prompt_pattern']}")
    lines.append(f"- prompt_manifest: {summary['prompt_manifest']}")
    lines.append(f"- max_prompts: {summary['max_prompts']}")
    lines.append(f"- max_prompts_per_bucket: {summary['max_prompts_per_bucket']}")
    lines.append(f"- sample_mode: {summary['sample_mode']}")
    lines.append(f"- bucket_order: {summary['bucket_order']}")
    lines.append(f"- decode_tps_avg: {summary['decode_tps_avg']:.4f}")
    lines.append(f"- decode_tps_min: {summary['decode_tps_min']:.4f}")
    lines.append(f"- decode_tps_max: {summary['decode_tps_max']:.4f}")
    lines.append(f"- h2o_keep_ratio_avg: {summary['h2o_keep_ratio_avg']:.4f}")
    lines.append(f"- h2o_lossy_ratio_avg: {summary['h2o_lossy_ratio_avg']:.4f}")
    lines.append(f"- h2o_lossless_ratio_avg: {summary['h2o_lossless_ratio_avg']:.4f}")
    lines.append(f"- h2o_runtime_total_ratio_avg: {summary['h2o_runtime_total_ratio_avg']:.4f}")
    lines.append(f"- h2o_lossless_raw_mb_avg: {summary['h2o_lossless_raw_mb_avg']:.4f}")
    lines.append(f"- h2o_lossless_comp_mb_avg: {summary['h2o_lossless_comp_mb_avg']:.4f}")
    lines.append(f"- runtime_decomp_us_max: {summary['runtime_decomp_us_max']:.4f}")
    lines.append(f"- decode_cache_hit_max: {summary['decode_cache_hit_max']:.4f}")
    lines.append("")
    lines.append("## Bucket Stats")
    lines.append("")
    for bucket in summary["bucket_order"]:
        stats = summary["bucket_stats"].get(bucket)
        if not stats:
            continue
        lines.append(f"### bucket_{bucket}")
        lines.append("")
        lines.append(f"- total_runs: {int(stats.get('total_runs', 0))}")
        lines.append(f"- pass_runs: {int(stats.get('pass_runs', 0))}")
        lines.append(f"- overall_pass: {str(bool(stats.get('overall_pass', False))).lower()}")
        lines.append(f"- decode_tps_avg: {float(stats.get('decode_tps_avg', 0.0)):.4f}")
        lines.append(f"- decode_tps_min: {float(stats.get('decode_tps_min', 0.0)):.4f}")
        lines.append(f"- decode_tps_max: {float(stats.get('decode_tps_max', 0.0)):.4f}")
        lines.append(f"- h2o_keep_ratio_avg: {float(stats.get('h2o_keep_ratio_avg', 0.0)):.4f}")
        lines.append(f"- h2o_lossy_ratio_avg: {float(stats.get('h2o_lossy_ratio_avg', 0.0)):.4f}")
        lines.append(f"- h2o_lossless_ratio_avg: {float(stats.get('h2o_lossless_ratio_avg', 0.0)):.4f}")
        lines.append(f"- h2o_runtime_total_ratio_avg: {float(stats.get('h2o_runtime_total_ratio_avg', 0.0)):.4f}")
        lines.append(f"- runtime_decomp_us_max: {float(stats.get('runtime_decomp_us_max', 0.0)):.4f}")
        lines.append(f"- decode_cache_hit_max: {float(stats.get('decode_cache_hit_max', 0.0)):.4f}")
        lines.append("")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for r in rows:
        h2o = r.get("h2o_metrics", {}) if isinstance(r.get("h2o_metrics"), dict) else {}
        lines.append(
            f"- {Path(r['prompt_file']).name} [bucket={r.get('prompt_bucket', 'other')}]: "
            f"rc={r['returncode']}, effective_rc={r['effective_returncode']}, "
            f"reason={r.get('status_reason', '')}, "
            f"cli_summary={str(bool(r.get('has_cli_summary_metrics', False))).lower()}, "
            f"jsonl_summary={str(bool(r.get('has_jsonl_summary_metrics', False))).lower()}, "
            f"decode_tps={float(r.get('decode_tps', 0.0)):.4f}, "
            f"keep={float(h2o.get('h2o_keep_ratio', 0.0)):.4f}, "
            f"lossy={float(h2o.get('h2o_lossy_ratio', 0.0)):.4f}, "
            f"lossless={float(h2o.get('h2o_lossless_ratio', 0.0)):.4f}, "
            f"decomp_us={float(h2o.get('h2o_lossless_decompress_us', 0.0)):.4f}, "
            f"cache_hit={float(h2o.get('h2o_lossless_decode_cache_hit', 0.0)):.4f}"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {runs_jsonl}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
