#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "")


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def parse_float(v, default: float = 0.0) -> float:
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(v))
    return float(m.group(0)) if m else default


def parse_quality_gate(summary_md: Path) -> Dict[str, str]:
    gate: Dict[str, str] = {}
    if not summary_md.exists():
        return gate
    in_gate = False
    text = summary_md.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if line == "## Quality Gate":
            in_gate = True
            continue
        if in_gate and line.startswith("## "):
            break
        if in_gate and line.startswith("- ") and ":" in line:
            k, v = line[2:].split(":", 1)
            gate[k.strip()] = v.strip().strip("`")
    return gate


def extract_failure_reason(console_log: Path) -> str:
    if not console_log.exists():
        return "console log missing"
    lines = console_log.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[-240:] if len(lines) > 240 else lines
    patterns = (
        "FAIL:",
        "error:",
        "Error:",
        "Traceback",
        "No such file",
        "not executable",
        "not found",
        "Segmentation fault",
        "Aborted",
    )
    for line in reversed(tail):
        s = line.strip()
        for p in patterns:
            if p in s:
                return s
    for line in reversed(tail):
        s = line.strip()
        if s:
            return s
    return "unknown failure"


def run_logged(cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def runtime_row(
    root: Path,
    base_out: Path,
    case_name: str,
    runtime_mode: str,
    shared_env: Dict[str, str],
    extra_env: Dict[str, str],
) -> Dict:
    out_dir = base_out / case_name
    console_log = base_out / f"{case_name}.console.log"
    summary_md = out_dir / "summary.md"

    run_env = dict(os.environ)
    run_env.update(shared_env)
    run_env.update(extra_env)
    run_env["OUT"] = str(out_dir)
    run_env["KV_LOSSLESS_RUNTIME_MODE"] = runtime_mode

    rc = run_logged(["bash", "exp/h2o_final/test_h2o_final_runtime.sh"], root, run_env, console_log)
    gate = parse_quality_gate(summary_md)
    summary_exists = summary_md.exists() and summary_md.stat().st_size > 0
    gate_found = bool(gate)
    overall_pass = (rc == 0) and gate.get("overall_pass", "").lower() == "true"

    fail_reason = ""
    if not overall_pass:
        if rc != 0:
            fail_reason = extract_failure_reason(console_log)
        elif not summary_exists:
            fail_reason = "summary missing"
        elif not gate_found:
            fail_reason = "quality gate section missing"
        else:
            fail_reason = f"quality gate failed: status={gate.get('quality_status', 'unknown')}"

    return {
        "suite": "runtime",
        "case": case_name,
        "mode": runtime_mode,
        "return_code": rc,
        "summary_exists": summary_exists,
        "quality_gate_found": gate_found,
        "failure_reason": fail_reason,
        "overall_pass": overall_pass,
        "summary": str(summary_md),
        "console_log": str(console_log),
        "quality_status": gate.get("quality_status", ""),
        "lossy_best": parse_float(gate.get("lossy_best", 0.0)),
        "lossless_selected_value": parse_float(gate.get("lossless_selected_value", 0.0)),
        "decode_aggregation": gate.get("decode_aggregation", "best"),
        "decode_gate_tps": parse_float(gate.get("decode_gate_tps", gate.get("decode_best", 0.0))),
        "decode_best": parse_float(gate.get("decode_best", 0.0)),
        "decode_baseline": parse_float(gate.get("decode_baseline", 0.0)),
        "decode_drop_ratio": parse_float(gate.get("decode_drop_ratio", 0.0)),
        "decode_drop_target": parse_float(gate.get("decode_drop_target", 0.0)),
        "runtime_decomp_best_us": parse_float(gate.get("runtime_decomp_best_us", 0.0)),
        "runtime_async_wait_best_us": parse_float(gate.get("runtime_async_wait_best_us", 0.0)),
        "runtime_async_wait_headroom_us": parse_float(gate.get("runtime_async_wait_headroom_us", 0.0)),
        "runtime_async_wait_headroom_ratio": parse_float(gate.get("runtime_async_wait_headroom_ratio", 0.0)),
        "runtime_async_wait_headroom_warn": gate.get("runtime_async_wait_headroom_warn", "").lower() == "true",
        "runtime_queue_peak_best": parse_float(gate.get("runtime_queue_peak_best", 0.0)),
        "runtime_fallback_best": parse_float(gate.get("runtime_fallback_best", 0.0)),
        "runtime_backpressure_skip_best": parse_float(gate.get("runtime_backpressure_skip_best", 0.0)),
        "runtime_decode_cache_hit_best": parse_float(gate.get("runtime_decode_cache_hit_best", 0.0)),
        "runtime_decode_cache_miss_best": parse_float(gate.get("runtime_decode_cache_miss_best", 0.0)),
        "kv_content_consistency_pass": gate.get("kv_content_consistency_pass", "").lower() == "true",
        "kv_invalid_log_count": parse_float(gate.get("kv_invalid_log_count", 0.0)),
    }


def llm_demo_row(
    root: Path,
    base_out: Path,
    case_name: str,
    runtime_mode: str,
    shared_env: Dict[str, str],
    extra_env: Dict[str, str],
) -> Dict:
    out_dir = base_out / case_name
    console_log = base_out / f"{case_name}.console.log"
    report_json = out_dir / "llm_demo_report.json"
    report_md = out_dir / "llm_demo_report.md"

    run_env = dict(os.environ)
    run_env.update(shared_env)
    run_env.update(extra_env)
    run_env["OUT"] = str(out_dir)
    run_env["KV_LOSSLESS_RUNTIME_MODE"] = runtime_mode

    rc = run_logged(["bash", "exp/h2o_final/test_h2o_final_llm_demo.sh"], root, run_env, console_log)

    obj: Dict = {}
    if report_json.exists():
        try:
            obj = json.loads(report_json.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            obj = {}

    overall_pass = (rc == 0) and bool(obj.get("overall_pass", False))

    fail_reason = ""
    if not overall_pass:
        if rc != 0:
            fail_reason = extract_failure_reason(console_log)
        elif not report_json.exists():
            fail_reason = "report json missing"
        else:
            fail_reason = "llm_demo gate failed"

    return {
        "suite": "llm_demo",
        "case": case_name,
        "mode": runtime_mode,
        "return_code": rc,
        "failure_reason": fail_reason,
        "overall_pass": overall_pass,
        "summary": str(report_md),
        "report_json": str(report_json),
        "console_log": str(console_log),
        "baseline_decode_tps_avg": float(obj.get("baseline_decode_tps_avg", 0.0)),
        "candidate_decode_tps_avg": float(obj.get("candidate_decode_tps_avg", 0.0)),
        "decode_drop_ratio": float(obj.get("decode_drop_ratio", 0.0)),
        "decode_drop_target": float(obj.get("decode_drop_target", 0.0)),
        "candidate_lossy_ratio_avg": float(obj.get("candidate_lossy_ratio_avg", 0.0)),
        "candidate_lossless_ratio_avg": float(obj.get("candidate_lossless_ratio_avg", 0.0)),
        "candidate_runtime_total_ratio_avg": float(obj.get("candidate_runtime_total_ratio_avg", 0.0)),
        "runtime_decomp_best_us": float(obj.get("runtime_decomp_best_us", 0.0)),
        "decode_cache_hit_best": float(obj.get("decode_cache_hit_best", 0.0)),
        "decode_pass": bool(obj.get("decode_pass", False)),
        "run_order": str(obj.get("run_order", "")),
    }


def write_report(base_out: Path, rows: List[Dict]) -> Dict:
    report_json = base_out / "summary.json"
    report_md = base_out / "summary.md"
    results_jsonl = base_out / "cases.jsonl"

    base_out.mkdir(parents=True, exist_ok=True)
    with results_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    overall_pass = all(bool(r.get("overall_pass", False)) for r in rows) if rows else False
    runtime_rows = [r for r in rows if r.get("suite") == "runtime"]
    llm_rows = [r for r in rows if r.get("suite") == "llm_demo"]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_pass": overall_pass,
        "total_cases": len(rows),
        "runtime_cases": len(runtime_rows),
        "llm_demo_cases": len(llm_rows),
        "rows": rows,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# H2O final Summary")
    lines.append("")
    lines.append(f"- overall_pass: {str(overall_pass).lower()}")
    lines.append(f"- total_cases: {len(rows)}")
    lines.append(f"- runtime_cases: {len(runtime_rows)}")
    lines.append(f"- llm_demo_cases: {len(llm_rows)}")
    lines.append("")

    if runtime_rows:
        lines.append("## Runtime")
        lines.append("")
        for i, r in enumerate(runtime_rows, 1):
            reason = r.get("failure_reason", "")
            reason_seg = f", reason={reason}" if reason else ""
            lines.append(
                f"{i}. case={r['case']}, mode={r.get('mode','')}, pass={str(r['overall_pass']).lower()}, rc={r['return_code']}, "
                f"decode={r.get('decode_gate_tps', r['decode_best']):.4f}/{r['decode_baseline']:.4f}"
                f"({r.get('decode_aggregation', 'best')}), "
                f"drop={r['decode_drop_ratio']:.6f}/{r['decode_drop_target']:.6f}, "
                f"lossy={r['lossy_best']:.4f}, lossless={r['lossless_selected_value']:.4f}, "
                f"decomp_us={r['runtime_decomp_best_us']:.4f}, async_wait_us={r['runtime_async_wait_best_us']:.4f}, "
                f"async_wait_headroom={r.get('runtime_async_wait_headroom_ratio', 0.0):.4f}, "
                f"queue={r['runtime_queue_peak_best']:.4f}, fallback={r['runtime_fallback_best']:.4f}, "
                f"cache_hit={r['runtime_decode_cache_hit_best']:.4f}, cache_miss={r['runtime_decode_cache_miss_best']:.4f}, "
                f"kv_consistency={str(r.get('kv_content_consistency_pass', True)).lower()}, "
                f"kv_invalid_logs={r.get('kv_invalid_log_count', 0.0):.0f}, "
                f"summary=`{r['summary']}`{reason_seg}"
            )
        lines.append("")

    if llm_rows:
        lines.append("## llm_demo")
        lines.append("")
        for i, r in enumerate(llm_rows, 1):
            reason = r.get("failure_reason", "")
            reason_seg = f", reason={reason}" if reason else ""
            lines.append(
                f"{i}. case={r['case']}, mode={r.get('mode','')}, pass={str(r['overall_pass']).lower()}, rc={r['return_code']}, "
                f"decode={r['baseline_decode_tps_avg']:.4f}->{r['candidate_decode_tps_avg']:.4f}, "
                f"drop={r['decode_drop_ratio']:.6f}/{r['decode_drop_target']:.6f}, "
                f"lossy={r['candidate_lossy_ratio_avg']:.4f}, lossless={r['candidate_lossless_ratio_avg']:.4f}, "
                f"total={r['candidate_runtime_total_ratio_avg']:.4f}, decomp_us={r['runtime_decomp_best_us']:.4f}, "
                f"cache_hit={r['decode_cache_hit_best']:.4f}, run_order={r.get('run_order', '')}, "
                f"summary=`{r['summary']}`{reason_seg}"
            )
        lines.append("")

    lines.append(f"- summary_json: `{report_json}`")
    lines.append(f"- cases_jsonl: `{results_jsonl}`")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def add_toggle_arg(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    parser.add_argument(f"--{name}", type=int, choices=[0, 1], default=None, help=help_text)


def resolve_toggle(value: Optional[int], env_name: str, default: bool) -> bool:
    if value is not None:
        return bool(value)
    return env_bool(env_name, default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run H2O final comprehensive suite.")
    parser.add_argument("--suite", choices=["all", "runtime", "llm_demo"], default=env_str("SUITE", "all"))
    parser.add_argument("--base-out", default=env_str("BASE_OUT", f"exp/h2o_final/out_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    add_toggle_arg(parser, "run-runtime-baseline", "Run runtime baseline case (0/1).")
    add_toggle_arg(parser, "run-runtime-probe", "Run runtime probe mode case (0/1).")
    add_toggle_arg(parser, "run-runtime-full", "Run runtime full mode case (0/1).")
    add_toggle_arg(parser, "run-runtime-store", "Run runtime store mode case (0/1).")
    add_toggle_arg(parser, "run-llm-demo-full", "Run llm_demo full mode case (0/1).")
    add_toggle_arg(parser, "run-llm-demo-store", "Run llm_demo store mode case (0/1).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    os.chdir(root)

    base_out = Path(args.base_out)

    runtime_default = args.suite in ("all", "runtime")
    llm_default = args.suite in ("all", "llm_demo")

    run_runtime_baseline = resolve_toggle(args.run_runtime_baseline, "RUN_RUNTIME_BASELINE", runtime_default)
    run_runtime_probe = resolve_toggle(args.run_runtime_probe, "RUN_RUNTIME_PROBE", runtime_default)
    run_runtime_full = resolve_toggle(args.run_runtime_full, "RUN_RUNTIME_FULL", runtime_default)
    run_runtime_store = resolve_toggle(args.run_runtime_store, "RUN_RUNTIME_STORE", runtime_default)
    run_llm_demo_full = resolve_toggle(args.run_llm_demo_full, "RUN_LLM_DEMO_FULL", llm_default)
    run_llm_demo_store = resolve_toggle(args.run_llm_demo_store, "RUN_LLM_DEMO_STORE", llm_default)

    runtime_shared = {
        "PROMPTS": env_str("PROMPTS", "512,1024"),
        "GENS": env_str("GENS", "128"),
        "REPEAT": str(env_int("REPEAT", 2)),
        "KV_LOSSLESS_ASYNC_THREADS": str(env_int("KV_LOSSLESS_ASYNC_THREADS", 2)),
        "KV_LOSSLESS_DECODE_CACHE_BLOCKS": str(env_int("KV_LOSSLESS_DECODE_CACHE_BLOCKS", 64)),
        "KV_LOSSLESS_STRICT_ROUNDTRIP_CHECK": str(env_int("KV_LOSSLESS_STRICT_ROUNDTRIP_CHECK", 1)),
        "DECODE_BASELINE_MODE": env_str("DECODE_BASELINE_MODE", "same_batch"),
        "DECODE_BASELINE": f"{env_float('DECODE_BASELINE', 6.60):.6f}",
        "DECODE_DROP_TARGET": f"{env_float('DECODE_DROP_TARGET', 0.05):.6f}",
        "STRICT_RUNTIME_METRIC_COLUMNS": str(env_int("STRICT_RUNTIME_METRIC_COLUMNS", 1)),
        "MAX_LOSSLESS_QUEUE_PEAK": str(env_int("MAX_LOSSLESS_QUEUE_PEAK", 8)),
        "MAX_LOSSLESS_FALLBACK": str(env_int("MAX_LOSSLESS_FALLBACK", 0)),
        "MAX_LOSSLESS_BACKPRESSURE_SKIP": str(env_int("MAX_LOSSLESS_BACKPRESSURE_SKIP", -1)),
        "ASYNC_WAIT_HEADROOM_WARN_RATIO": f"{env_float('ASYNC_WAIT_HEADROOM_WARN_RATIO', 0.15):.6f}",
        "REQUIRE_DECODE_CACHE_HIT": str(env_int("REQUIRE_DECODE_CACHE_HIT", 0)),
        "REQUIRE_ASYNC_QUEUE_ACTIVITY": str(env_int("REQUIRE_ASYNC_QUEUE_ACTIVITY", 0)),
        "REQUIRE_DECODE_CACHE_ACTIVITY": str(env_int("REQUIRE_DECODE_CACHE_ACTIVITY", 0)),
        "REQUIRE_KV_CONTENT_CONSISTENCY": str(env_int("REQUIRE_KV_CONTENT_CONSISTENCY", -1)),
        "MAX_KV_INVALID_LOG_LINES": str(env_int("MAX_KV_INVALID_LOG_LINES", 0)),
    }

    baseline_extra = {
        "DISABLE_H2O": "1",
        "DISABLE_LOSSLESS": "1",
        "LOSSY_TARGET": f"{env_float('BASELINE_LOSSY_TARGET', 1.0):.6f}",
        "LOSSLESS_TARGET": f"{env_float('BASELINE_LOSSLESS_TARGET', 1.0):.6f}",
        "REQUIRE_RUNTIME_DECOMP": "0",
        "REQUIRE_KV_CONTENT_CONSISTENCY": "0",
        "DECODE_DROP_TARGET": f"{env_float('BASELINE_DECODE_DROP_TARGET', 1.0):.6f}",
        "STRICT_RUNTIME_METRIC_COLUMNS": str(env_int("BASELINE_STRICT_RUNTIME_METRIC_COLUMNS", 0)),
    }
    probe_extra = {
        "H2O_TARGET_LOSSY_RATIO": f"{env_float('PROBE_H2O_TARGET_LOSSY_RATIO', 3.5):.6f}",
        "REQUIRE_RUNTIME_DECOMP": "0",
        "REQUIRE_KV_CONTENT_CONSISTENCY": "0",
        "MAX_LOSSLESS_DECOMP_US": str(env_int("PROBE_MAX_LOSSLESS_DECOMP_US", -1)),
        "MAX_LOSSLESS_ASYNC_WAIT_US": str(env_int("PROBE_MAX_LOSSLESS_ASYNC_WAIT_US", 20000)),
    }
    full_extra = {
        "H2O_TARGET_LOSSY_RATIO": f"{env_float('FULL_H2O_TARGET_LOSSY_RATIO', 3.5):.6f}",
        "REQUIRE_RUNTIME_DECOMP": "1",
        "REQUIRE_KV_CONTENT_CONSISTENCY": "1",
        "MAX_LOSSLESS_DECOMP_US": str(env_int("FULL_MAX_LOSSLESS_DECOMP_US", 30000)),
        "MAX_LOSSLESS_ASYNC_WAIT_US": str(env_int("FULL_MAX_LOSSLESS_ASYNC_WAIT_US", 20000)),
    }
    store_extra = {
        "REQUIRE_RUNTIME_DECOMP": "1",
        "REQUIRE_KV_CONTENT_CONSISTENCY": "1",
        "KV_LOSSLESS_STORE_DISABLE_FRONT": str(env_int("KV_LOSSLESS_STORE_DISABLE_FRONT", 1)),
        # Use conservative defaults to reduce async wait spikes in store mode.
        "KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS": str(env_int("KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS", 32)),
        "KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS": str(env_int("KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS", 256)),
        "MAX_LOSSLESS_DECOMP_US": str(env_int("STORE_MAX_LOSSLESS_DECOMP_US", 70000)),
        "MAX_LOSSLESS_ASYNC_WAIT_US": str(env_int("STORE_MAX_LOSSLESS_ASYNC_WAIT_US", 40000)),
    }

    llm_shared = {
        "PROMPT_DIR": env_str("PROMPT_DIR", "/home10T/ljq/MNN/exp/h2o_final/prompts"),
        "PROMPT_PATTERN": env_str("PROMPT_PATTERN", "prompt_*.txt"),
        "PROMPT_MANIFEST": env_str("PROMPT_MANIFEST", ""),
        "PROMPT_BUCKET_LIST": env_str("PROMPT_BUCKET_LIST", "128,512,2048"),
        "PROMPT_SAMPLE_MODE": env_str("PROMPT_SAMPLE_MODE", "stratified"),
        "MAX_PROMPTS": str(env_int("MAX_PROMPTS", 0)),
        "MAX_PROMPTS_PER_BUCKET": str(env_int("MAX_PROMPTS_PER_BUCKET", 2)),
        "DECODE_TOKENS": str(env_int("DECODE_TOKENS", 512)),
        "DECODE_DROP_TARGET": f"{env_float('LLM_DEMO_DECODE_DROP_TARGET', 0.08):.6f}",
        "REQUIRE_RUNTIME_DECOMP": str(env_int("LLM_DEMO_REQUIRE_RUNTIME_DECOMP", 1)),
        "REQUIRE_DECODE_CACHE_HIT": str(env_int("LLM_DEMO_REQUIRE_DECODE_CACHE_HIT", 0)),
        "REQUIRE_BASELINE_DECODE": str(env_int("LLM_DEMO_REQUIRE_BASELINE_DECODE", 1)),
        "REQUIRE_BUCKET_DECODE_PASS": str(env_int("LLM_DEMO_REQUIRE_BUCKET_DECODE_PASS", 1)),
        "LLM_DEMO_RUN_ORDER": env_str("LLM_DEMO_RUN_ORDER", "baseline_first"),
        "KV_LOSSLESS_ASYNC_THREADS": str(env_int("KV_LOSSLESS_ASYNC_THREADS", 2)),
        "KV_LOSSLESS_DECODE_CACHE_BLOCKS": str(env_int("KV_LOSSLESS_DECODE_CACHE_BLOCKS", 64)),
    }
    llm_store_extra = {
        "KV_LOSSLESS_STORE_DISABLE_FRONT": str(env_int("LLM_DEMO_KV_LOSSLESS_STORE_DISABLE_FRONT", 1)),
        "KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS": str(env_int("LLM_DEMO_KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS", 32)),
        "KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS": str(env_int("LLM_DEMO_KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS", 256)),
        "KV_LOSSLESS_MAX_QUEUE": str(env_int("LLM_DEMO_KV_LOSSLESS_MAX_QUEUE", 64)),
    }

    print("============================================================")
    print("H2O final comprehensive suite")
    print(f"BASE_OUT={base_out}")
    print(f"SUITE={args.suite}")
    print(
        "RUNTIME: baseline={} probe={} full={} store={}".format(
            int(run_runtime_baseline), int(run_runtime_probe), int(run_runtime_full), int(run_runtime_store)
        )
    )
    print(
        "LLM_DEMO: full={} store={}".format(int(run_llm_demo_full), int(run_llm_demo_store))
    )
    print("============================================================")

    rows: List[Dict] = []
    if run_runtime_baseline:
        print("[runtime] baseline")
        rows.append(runtime_row(root, base_out, "runtime_baseline", "full", runtime_shared, baseline_extra))
    if run_runtime_probe:
        print("[runtime] probe")
        rows.append(runtime_row(root, base_out, "runtime_probe", "probe", runtime_shared, probe_extra))
    if run_runtime_full:
        print("[runtime] full")
        rows.append(runtime_row(root, base_out, "runtime_full", "full", runtime_shared, full_extra))
    if run_runtime_store:
        print("[runtime] store")
        rows.append(runtime_row(root, base_out, "runtime_store", "store", runtime_shared, store_extra))

    if run_llm_demo_full:
        print("[llm_demo] full")
        rows.append(llm_demo_row(root, base_out, "llm_demo_full", "full", llm_shared, {}))
    if run_llm_demo_store:
        print("[llm_demo] store")
        rows.append(llm_demo_row(root, base_out, "llm_demo_store", "store", llm_shared, llm_store_extra))

    if not rows:
        print("No case selected. Use --suite and/or --run-* flags to enable at least one case.")
        return 2

    report = write_report(base_out, rows)
    print(f"Summary: {base_out / 'summary.md'}")
    print(f"Overall pass: {str(report['overall_pass']).lower()}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
