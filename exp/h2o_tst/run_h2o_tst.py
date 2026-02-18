#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


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

    rc = run_logged(["bash", "exp/h2o_v6/test_v6_runtime.sh"], root, run_env, console_log)
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

    row = {
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
        "decode_best": parse_float(gate.get("decode_best", 0.0)),
        "decode_baseline": parse_float(gate.get("decode_baseline", 0.0)),
        "decode_drop_ratio": parse_float(gate.get("decode_drop_ratio", 0.0)),
        "decode_drop_target": parse_float(gate.get("decode_drop_target", 0.0)),
        "runtime_decomp_best_us": parse_float(gate.get("runtime_decomp_best_us", 0.0)),
        "runtime_async_wait_best_us": parse_float(gate.get("runtime_async_wait_best_us", 0.0)),
        "runtime_queue_peak_best": parse_float(gate.get("runtime_queue_peak_best", 0.0)),
        "runtime_fallback_best": parse_float(gate.get("runtime_fallback_best", 0.0)),
        "runtime_backpressure_skip_best": parse_float(gate.get("runtime_backpressure_skip_best", 0.0)),
        "runtime_decode_cache_hit_best": parse_float(gate.get("runtime_decode_cache_hit_best", 0.0)),
        "runtime_decode_cache_miss_best": parse_float(gate.get("runtime_decode_cache_miss_best", 0.0)),
        "runtime_async_wait_pass": gate.get("runtime_async_wait_pass", "").lower() == "true",
        "runtime_decomp_pass": gate.get("runtime_decomp_pass", "").lower() == "true",
        "decode_pass": gate.get("decode_pass", "").lower() == "true",
    }
    return row


def llm_demo_row(root: Path, base_out: Path, shared_env: Dict[str, str]) -> Dict:
    case_name = "llm_demo"
    out_dir = base_out / case_name
    console_log = base_out / f"{case_name}.console.log"
    report_json = out_dir / "llm_demo_report.json"
    report_md = out_dir / "llm_demo_report.md"

    run_env = dict(os.environ)
    run_env.update(shared_env)
    run_env["OUT"] = str(out_dir)

    rc = run_logged(["bash", "exp/h2o_v6/test_v6_llm_demo.sh"], root, run_env, console_log)

    obj = {}
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

    row = {
        "suite": "llm_demo",
        "case": case_name,
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
    }
    return row


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
    lines.append("# H2O tst Summary")
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
                f"{i}. case={r['case']}, pass={str(r['overall_pass']).lower()}, rc={r['return_code']}, "
                f"decode={r['decode_best']:.4f}/{r['decode_baseline']:.4f}, "
                f"drop={r['decode_drop_ratio']:.6f}/{r['decode_drop_target']:.6f}, "
                f"lossy={r['lossy_best']:.4f}, lossless={r['lossless_selected_value']:.4f}, "
                f"decomp_us={r['runtime_decomp_best_us']:.4f}, async_wait_us={r['runtime_async_wait_best_us']:.4f}, "
                f"queue={r['runtime_queue_peak_best']:.4f}, fallback={r['runtime_fallback_best']:.4f}, "
                f"cache_hit={r['runtime_decode_cache_hit_best']:.4f}, cache_miss={r['runtime_decode_cache_miss_best']:.4f}, "
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
                f"{i}. case={r['case']}, pass={str(r['overall_pass']).lower()}, rc={r['return_code']}, "
                f"decode={r['baseline_decode_tps_avg']:.4f}->{r['candidate_decode_tps_avg']:.4f}, "
                f"drop={r['decode_drop_ratio']:.6f}/{r['decode_drop_target']:.6f}, "
                f"lossy={r['candidate_lossy_ratio_avg']:.4f}, lossless={r['candidate_lossless_ratio_avg']:.4f}, "
                f"total={r['candidate_runtime_total_ratio_avg']:.4f}, decomp_us={r['runtime_decomp_best_us']:.4f}, "
                f"summary=`{r['summary']}`{reason_seg}"
            )
        lines.append("")

    lines.append(f"- summary_json: `{report_json}`")
    lines.append(f"- cases_jsonl: `{results_jsonl}`")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)

    base_out = Path(env_str("BASE_OUT", f"exp/h2o_tst/out_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    run_runtime_full = env_bool("RUN_RUNTIME_FULL", True)
    run_runtime_store = env_bool("RUN_RUNTIME_STORE", True)
    run_llm_demo = env_bool("RUN_LLM_DEMO", True)

    runtime_shared = {
        "PROMPTS": env_str("PROMPTS", "512,1024"),
        "GENS": env_str("GENS", "128"),
        "REPEAT": str(env_int("REPEAT", 2)),
        "KV_LOSSLESS_ASYNC_THREADS": str(env_int("KV_LOSSLESS_ASYNC_THREADS", 2)),
        "KV_LOSSLESS_DECODE_CACHE_BLOCKS": str(env_int("KV_LOSSLESS_DECODE_CACHE_BLOCKS", 64)),
        "DECODE_BASELINE_MODE": env_str("DECODE_BASELINE_MODE", "same_batch"),
        "DECODE_BASELINE": f"{env_float('DECODE_BASELINE', 6.60):.6f}",
        "DECODE_DROP_TARGET": f"{env_float('DECODE_DROP_TARGET', 0.05):.6f}",
        "STRICT_RUNTIME_METRIC_COLUMNS": str(env_int("STRICT_RUNTIME_METRIC_COLUMNS", 1)),
        "MAX_LOSSLESS_QUEUE_PEAK": str(env_int("MAX_LOSSLESS_QUEUE_PEAK", 8)),
        "MAX_LOSSLESS_FALLBACK": str(env_int("MAX_LOSSLESS_FALLBACK", 0)),
        "MAX_LOSSLESS_BACKPRESSURE_SKIP": str(env_int("MAX_LOSSLESS_BACKPRESSURE_SKIP", -1)),
        "REQUIRE_DECODE_CACHE_HIT": str(env_int("REQUIRE_DECODE_CACHE_HIT", 0)),
        "REQUIRE_ASYNC_QUEUE_ACTIVITY": str(env_int("REQUIRE_ASYNC_QUEUE_ACTIVITY", 0)),
        "REQUIRE_DECODE_CACHE_ACTIVITY": str(env_int("REQUIRE_DECODE_CACHE_ACTIVITY", 0)),
    }

    full_extra = {
        "MAX_LOSSLESS_DECOMP_US": str(env_int("FULL_MAX_LOSSLESS_DECOMP_US", 30000)),
        "MAX_LOSSLESS_ASYNC_WAIT_US": str(env_int("FULL_MAX_LOSSLESS_ASYNC_WAIT_US", 20000)),
    }

    store_extra = {
        "KV_LOSSLESS_STORE_DISABLE_FRONT": str(env_int("KV_LOSSLESS_STORE_DISABLE_FRONT", 1)),
        "KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS": str(env_int("KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS", 16)),
        "KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS": str(env_int("KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS", 384)),
        "MAX_LOSSLESS_DECOMP_US": str(env_int("STORE_MAX_LOSSLESS_DECOMP_US", 70000)),
        "MAX_LOSSLESS_ASYNC_WAIT_US": str(env_int("STORE_MAX_LOSSLESS_ASYNC_WAIT_US", 40000)),
    }

    llm_env = {
        "PROMPT_DIR": env_str("PROMPT_DIR", "/home10T/ljq/MNN/exp/gear_fp16/prompts"),
        "PROMPT_PATTERN": env_str("PROMPT_PATTERN", "prompt_*.txt"),
        "PROMPT_MANIFEST": env_str("PROMPT_MANIFEST", ""),
        "PROMPT_BUCKET_LIST": env_str("PROMPT_BUCKET_LIST", "128,512"),
        "PROMPT_SAMPLE_MODE": env_str("PROMPT_SAMPLE_MODE", "stratified"),
        "MAX_PROMPTS": str(env_int("MAX_PROMPTS", 0)),
        "MAX_PROMPTS_PER_BUCKET": str(env_int("MAX_PROMPTS_PER_BUCKET", 1)),
        "DECODE_TOKENS": str(env_int("DECODE_TOKENS", 128)),
        "DECODE_DROP_TARGET": f"{env_float('LLM_DEMO_DECODE_DROP_TARGET', 0.08):.6f}",
        "REQUIRE_RUNTIME_DECOMP": str(env_int("LLM_DEMO_REQUIRE_RUNTIME_DECOMP", 1)),
        "REQUIRE_DECODE_CACHE_HIT": str(env_int("LLM_DEMO_REQUIRE_DECODE_CACHE_HIT", 0)),
        "REQUIRE_BUCKET_DECODE_PASS": str(env_int("LLM_DEMO_REQUIRE_BUCKET_DECODE_PASS", 1)),
        "KV_LOSSLESS_ASYNC_THREADS": str(env_int("KV_LOSSLESS_ASYNC_THREADS", 2)),
        "KV_LOSSLESS_DECODE_CACHE_BLOCKS": str(env_int("KV_LOSSLESS_DECODE_CACHE_BLOCKS", 64)),
    }

    print("============================================================")
    print("H2O tst minimal suite")
    print(f"BASE_OUT={base_out}")
    print(f"RUN_RUNTIME_FULL={int(run_runtime_full)} RUN_RUNTIME_STORE={int(run_runtime_store)} RUN_LLM_DEMO={int(run_llm_demo)}")
    print("============================================================")

    rows: List[Dict] = []
    if run_runtime_full:
        print("[1/3] runtime_full")
        rows.append(runtime_row(root, base_out, "runtime_full", "full", runtime_shared, full_extra))
    if run_runtime_store:
        print("[2/3] runtime_store")
        rows.append(runtime_row(root, base_out, "runtime_store", "store", runtime_shared, store_extra))
    if run_llm_demo:
        print("[3/3] llm_demo")
        rows.append(llm_demo_row(root, base_out, llm_env))

    if not rows:
        print("No case selected. Enable at least one of RUN_RUNTIME_FULL/RUN_RUNTIME_STORE/RUN_LLM_DEMO.")
        return 2

    report = write_report(base_out, rows)
    print(f"Summary: {base_out / 'summary.md'}")
    print(f"Overall pass: {str(report['overall_pass']).lower()}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
