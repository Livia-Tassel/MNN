#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path


PROMPT_TOKENS_RE = re.compile(r"prompt tokens num = (\d+)")
DECODE_TOKENS_RE = re.compile(r"decode tokens num = (\d+)")
PREFILL_TIME_RE = re.compile(r"prefill time = ([\d\.]+) s")
DECODE_TIME_RE = re.compile(r"decode time = ([\d\.]+) s")


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


def safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return a / b


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


def main():
    parser = argparse.ArgumentParser(description="Run llm_demo with real prompt files and collect structured metrics.")
    parser.add_argument("--llm-demo", default="./build/llm_demo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--prompt-pattern", default="prompt_*.txt")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--out-dir", default="exp/h2o_v6/out_llm_demo")
    parser.add_argument("--run-tag", default="candidate")
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    files = sorted(prompt_dir.glob(args.prompt_pattern))
    if not files:
        raise SystemExit(f"No prompt files found under {prompt_dir} with pattern {args.prompt_pattern}")

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

        env = os.environ.copy()
        env["LLM_DEMO_METRICS_JSONL"] = str(metrics_path)
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
        if "LLM init error" in (stdout + "\n" + stderr):
            effective_rc = 1
        if effective_rc == 0 and not (
            has_cli_summary_metrics(cli_metrics) or has_jsonl_summary_metrics(h2o_metrics)
        ):
            # Treat "instant success with no metrics" as invalid run.
            effective_rc = 2

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
            "returncode": proc.returncode,
            "effective_returncode": effective_rc,
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

    decode_tps_values = [float(r.get("decode_tps", 0.0)) for r in rows if r.get("decode_tps", 0.0) > 0.0]
    decomp_values = [
        float(r.get("h2o_metrics", {}).get("h2o_lossless_decompress_us", 0.0))
        for r in rows
        if isinstance(r.get("h2o_metrics"), dict)
    ]
    cache_hits = [
        float(r.get("h2o_metrics", {}).get("h2o_lossless_decode_cache_hit", 0.0))
        for r in rows
        if isinstance(r.get("h2o_metrics"), dict)
    ]

    summary = {
        "run_tag": args.run_tag,
        "total_runs": len(rows),
        "pass_runs": sum(1 for r in rows if int(r.get("effective_returncode", 1)) == 0),
        "decode_tps_avg": (sum(decode_tps_values) / len(decode_tps_values)) if decode_tps_values else 0.0,
        "decode_tps_min": min(decode_tps_values) if decode_tps_values else 0.0,
        "decode_tps_max": max(decode_tps_values) if decode_tps_values else 0.0,
        "runtime_decomp_us_max": max(decomp_values) if decomp_values else 0.0,
        "decode_cache_hit_max": max(cache_hits) if cache_hits else 0.0,
        "overall_pass": len(rows) > 0 and all(int(r.get("effective_returncode", 1)) == 0 for r in rows),
        "runs_jsonl": str(runs_jsonl),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# LLM Demo Real Prompt Summary ({args.run_tag})")
    lines.append("")
    lines.append(f"- total_runs: {summary['total_runs']}")
    lines.append(f"- pass_runs: {summary['pass_runs']}")
    lines.append(f"- overall_pass: {str(summary['overall_pass']).lower()}")
    lines.append(f"- decode_tps_avg: {summary['decode_tps_avg']:.4f}")
    lines.append(f"- decode_tps_min: {summary['decode_tps_min']:.4f}")
    lines.append(f"- decode_tps_max: {summary['decode_tps_max']:.4f}")
    lines.append(f"- runtime_decomp_us_max: {summary['runtime_decomp_us_max']:.4f}")
    lines.append(f"- decode_cache_hit_max: {summary['decode_cache_hit_max']:.4f}")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for r in rows:
        h2o = r.get("h2o_metrics", {}) if isinstance(r.get("h2o_metrics"), dict) else {}
        lines.append(
            f"- {Path(r['prompt_file']).name}: rc={r['returncode']}, effective_rc={r['effective_returncode']}, "
            f"decode_tps={float(r.get('decode_tps', 0.0)):.4f}, "
            f"decomp_us={float(h2o.get('h2o_lossless_decompress_us', 0.0)):.4f}, "
            f"cache_hit={float(h2o.get('h2o_lossless_decode_cache_hit', 0.0)):.4f}"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {runs_jsonl}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
