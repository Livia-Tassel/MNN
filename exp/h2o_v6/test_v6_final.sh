#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# H2O v6 Final Acceptance Runner
# One-command full validation for:
#   1) front-2 lossless (offline online-sim >= 1.3)
#   2) deep H2O lossy (runtime >= 3.0)
#   3) optional kept-scope lossless (joint scope full/store + full coverage)
# ============================================================================

BASE_OUT="${BASE_OUT:-exp/h2o_v6/out_final_v6_$(date +%Y%m%d_%H%M%S)}"
RESULTS_JSONL="${BASE_OUT}/results.jsonl"
REPORT_MD="${BASE_OUT}/final_report.md"
REPORT_JSON="${BASE_OUT}/final_report.json"

MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK:-8}"
MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK:-0}"
FULL_DECOMP_BUDGET_US="${FULL_DECOMP_BUDGET_US:-30000}"
STORE_DECOMP_BUDGET_US="${STORE_DECOMP_BUDGET_US:-70000}"
COVERAGE_DECOMP_BUDGET_US="${COVERAGE_DECOMP_BUDGET_US:-70000}"
M3_DECOMP_BUDGET_US="${M3_DECOMP_BUDGET_US:-70000}"
MAX_LOSSLESS_ASYNC_WAIT_US="${MAX_LOSSLESS_ASYNC_WAIT_US:--1}"
REQUIRE_DECODE_CACHE_HIT="${REQUIRE_DECODE_CACHE_HIT:-0}"
REQUIRE_ASYNC_QUEUE_ACTIVITY="${REQUIRE_ASYNC_QUEUE_ACTIVITY:-0}"
REQUIRE_DECODE_CACHE_ACTIVITY="${REQUIRE_DECODE_CACHE_ACTIVITY:-0}"
STRICT_RUNTIME_METRIC_COLUMNS="${STRICT_RUNTIME_METRIC_COLUMNS:-1}"
KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS:-1}"
KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS:-64}"
M3_FULL_RUNS="${M3_FULL_RUNS:-3}"
M3_STORE_RUNS="${M3_STORE_RUNS:-1}"
LLM_DEMO_DECODE_DROP_TARGET="${LLM_DEMO_DECODE_DROP_TARGET:-0.08}"
LLM_DEMO_REQUIRE_RUNTIME_DECOMP="${LLM_DEMO_REQUIRE_RUNTIME_DECOMP:-1}"
LLM_DEMO_REQUIRE_DECODE_CACHE_HIT="${LLM_DEMO_REQUIRE_DECODE_CACHE_HIT:-0}"

mkdir -p "${BASE_OUT}"
: > "${RESULTS_JSONL}"

echo "============================================================"
echo " H2O v6 Final Acceptance"
echo " BASE_OUT = ${BASE_OUT}"
echo "============================================================"
echo "Gate policy:"
echo "  queue_peak <= ${MAX_LOSSLESS_QUEUE_PEAK}"
echo "  fallback   <= ${MAX_LOSSLESS_FALLBACK}"
echo "  full decomp budget us      = ${FULL_DECOMP_BUDGET_US}"
echo "  store decomp budget us     = ${STORE_DECOMP_BUDGET_US}"
echo "  coverage decomp budget us  = ${COVERAGE_DECOMP_BUDGET_US}"
echo "  m3 decomp budget us        = ${M3_DECOMP_BUDGET_US}"
echo "  async wait budget us       = ${MAX_LOSSLESS_ASYNC_WAIT_US}"
echo "  require decode cache hit   = ${REQUIRE_DECODE_CACHE_HIT}"
echo "  require async queue active = ${REQUIRE_ASYNC_QUEUE_ACTIVITY}"
echo "  require cache activity     = ${REQUIRE_DECODE_CACHE_ACTIVITY}"
echo "  strict metric columns      = ${STRICT_RUNTIME_METRIC_COLUMNS}"
echo "  llm_demo decode drop target = ${LLM_DEMO_DECODE_DROP_TARGET}"
echo ""

append_runtime_row() {
  local summary_path="$1"
  local case_name="$2"
  python3 - "${summary_path}" "${case_name}" >> "${RESULTS_JSONL}" <<'PY'
import json
import re
import sys
from pathlib import Path

summary = Path(sys.argv[1])
case_name = sys.argv[2]
if not summary.exists():
    raise SystemExit(f"missing summary: {summary}")

gate = {}
in_gate = False
for raw in summary.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line == "## Quality Gate":
        in_gate = True
        continue
    if in_gate and line.startswith("## "):
        break
    if in_gate and line.startswith("- ") and ":" in line:
        k, v = line[2:].split(":", 1)
        gate[k.strip()] = v.strip().strip("`")

def parse_float(s, default=0.0):
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(s))
    return float(m.group(0)) if m else default

row = {
    "suite": "runtime",
    "case": case_name,
    "summary": str(summary),
    "quality_status": gate.get("quality_status", ""),
    "overall_pass": gate.get("overall_pass", "") == "true",
    "decode_pass": gate.get("decode_pass", "") == "true",
    "lossy_pass": gate.get("lossy_pass", "") == "true",
    "lossless_online_pass": gate.get("lossless_online_pass", "") == "true",
    "runtime_decomp_pass": gate.get("runtime_decomp_pass", "") == "true",
    "runtime_queue_peak_pass": gate.get("runtime_queue_peak_pass", "") == "true",
    "runtime_fallback_pass": gate.get("runtime_fallback_pass", "") == "true",
    "decode_best": parse_float(gate.get("decode_best", 0.0)),
    "decode_drop_ratio": parse_float(gate.get("decode_drop_ratio", 0.0)),
    "lossy_best": parse_float(gate.get("lossy_best", 0.0)),
    "lossless_online_value": parse_float(gate.get("lossless_online_value", 0.0)),
    "runtime_decomp_best_us": parse_float(gate.get("runtime_decomp_best_us", 0.0)),
    "runtime_queue_peak_best": parse_float(gate.get("runtime_queue_peak_best", 0.0)),
    "runtime_fallback_best": parse_float(gate.get("runtime_fallback_best", 0.0)),
}
print(json.dumps(row, ensure_ascii=True))
PY
}

run_runtime_case() {
  local case_name="$1"
  local runtime_mode="$2"
  shift 2
  local out_dir="${BASE_OUT}/${case_name}"
  local console_log="${BASE_OUT}/${case_name}.console.log"
  local summary_path="${out_dir}/summary.md"
  local -a extra_env=("$@")

  echo "---- Runtime case: ${case_name} (mode=${runtime_mode}) ----"
  if ! env \
    OUT="${out_dir}" \
    KV_LOSSLESS_RUNTIME_MODE="${runtime_mode}" \
    KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS}" \
    KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS}" \
    REQUIRE_DECODE_CACHE_HIT="${REQUIRE_DECODE_CACHE_HIT}" \
    REQUIRE_ASYNC_QUEUE_ACTIVITY="${REQUIRE_ASYNC_QUEUE_ACTIVITY}" \
    REQUIRE_DECODE_CACHE_ACTIVITY="${REQUIRE_DECODE_CACHE_ACTIVITY}" \
    STRICT_RUNTIME_METRIC_COLUMNS="${STRICT_RUNTIME_METRIC_COLUMNS}" \
    MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK}" \
    MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK}" \
    MAX_LOSSLESS_ASYNC_WAIT_US="${MAX_LOSSLESS_ASYNC_WAIT_US}" \
    "${extra_env[@]}" \
    bash exp/h2o_v6/test_v6_runtime.sh > "${console_log}" 2>&1; then
    echo "FAIL: ${case_name}"
    echo "See: ${console_log}"
    tail -n 80 "${console_log}" || true
    exit 1
  fi
  if [[ ! -s "${summary_path}" ]]; then
    echo "FAIL: missing summary: ${summary_path}"
    exit 1
  fi
  append_runtime_row "${summary_path}" "${case_name}"
  echo "PASS: ${summary_path}"
}

run_m3_case() {
  local case_name="$1"
  local out_dir="${BASE_OUT}/${case_name}"
  local console_log="${BASE_OUT}/${case_name}.console.log"
  local report_json="${out_dir}/m3_report.json"
  local report_md="${out_dir}/m3_report.md"

  echo "---- M3 pack: ${case_name} ----"
  if ! env \
    BASE_OUT="${out_dir}" \
    FULL_RUNS="${M3_FULL_RUNS}" \
    STORE_RUNS="${M3_STORE_RUNS}" \
    KV_LOSSLESS_STORE_DISABLE_FRONT="1" \
    KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS="16" \
    KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS="384" \
    KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS}" \
    KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS}" \
    REQUIRE_DECODE_CACHE_HIT="${REQUIRE_DECODE_CACHE_HIT}" \
    REQUIRE_ASYNC_QUEUE_ACTIVITY="${REQUIRE_ASYNC_QUEUE_ACTIVITY}" \
    REQUIRE_DECODE_CACHE_ACTIVITY="${REQUIRE_DECODE_CACHE_ACTIVITY}" \
    STRICT_RUNTIME_METRIC_COLUMNS="${STRICT_RUNTIME_METRIC_COLUMNS}" \
    MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK}" \
    MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK}" \
    MAX_LOSSLESS_DECOMP_US="${M3_DECOMP_BUDGET_US}" \
    MAX_LOSSLESS_ASYNC_WAIT_US="${MAX_LOSSLESS_ASYNC_WAIT_US}" \
    bash exp/h2o_v6/test_v6_m3.sh > "${console_log}" 2>&1; then
    echo "FAIL: ${case_name}"
    echo "See: ${console_log}"
    tail -n 80 "${console_log}" || true
    exit 1
  fi
  if [[ ! -s "${report_json}" ]]; then
    echo "FAIL: missing m3 report json: ${report_json}"
    exit 1
  fi

  python3 - "${report_json}" "${report_md}" "${case_name}" >> "${RESULTS_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report_md = sys.argv[2]
case_name = sys.argv[3]

obj = json.loads(report_path.read_text(encoding="utf-8"))
total_runs = int(obj.get("total_runs", 0))
overall_pass_runs = int(obj.get("overall_pass_runs", 0))
cases = obj.get("cases", {})
joint_full = cases.get("joint_full", {})
joint_store = cases.get("joint_store", {})

row = {
    "suite": "m3",
    "case": case_name,
    "summary": report_md,
    "overall_pass": total_runs > 0 and overall_pass_runs == total_runs,
    "total_runs": total_runs,
    "overall_pass_runs": overall_pass_runs,
    "joint_full_decode_best_min": float(joint_full.get("decode_best_min", 0.0)),
    "joint_full_decode_best_max": float(joint_full.get("decode_best_max", 0.0)),
    "joint_full_runtime_decomp_best_us_max": float(joint_full.get("runtime_decomp_best_us_max", 0.0)),
    "joint_store_decode_best_min": float(joint_store.get("decode_best_min", 0.0)),
    "joint_store_decode_best_max": float(joint_store.get("decode_best_max", 0.0)),
    "joint_store_runtime_decomp_best_us_max": float(joint_store.get("runtime_decomp_best_us_max", 0.0)),
}
print(json.dumps(row, ensure_ascii=True))
PY
  echo "PASS: ${report_md}"
}

run_llm_demo_case() {
  local case_name="$1"
  local out_dir="${BASE_OUT}/${case_name}"
  local console_log="${BASE_OUT}/${case_name}.console.log"
  local report_json="${out_dir}/llm_demo_report.json"
  local report_md="${out_dir}/llm_demo_report.md"

  echo "---- llm_demo pack: ${case_name} ----"
  if ! env \
    OUT="${out_dir}" \
    DECODE_DROP_TARGET="${LLM_DEMO_DECODE_DROP_TARGET}" \
    REQUIRE_RUNTIME_DECOMP="${LLM_DEMO_REQUIRE_RUNTIME_DECOMP}" \
    REQUIRE_DECODE_CACHE_HIT="${LLM_DEMO_REQUIRE_DECODE_CACHE_HIT}" \
    KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS}" \
    KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS}" \
    bash exp/h2o_v6/test_v6_llm_demo.sh > "${console_log}" 2>&1; then
    echo "FAIL: ${case_name}"
    echo "See: ${console_log}"
    tail -n 80 "${console_log}" || true
    exit 1
  fi
  if [[ ! -s "${report_json}" ]]; then
    echo "FAIL: missing llm_demo report json: ${report_json}"
    exit 1
  fi

  python3 - "${report_json}" "${report_md}" "${case_name}" >> "${RESULTS_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report_md = sys.argv[2]
case_name = sys.argv[3]
obj = json.loads(report_path.read_text(encoding="utf-8"))

row = {
    "suite": "llm_demo",
    "case": case_name,
    "summary": report_md,
    "overall_pass": bool(obj.get("overall_pass", False)),
    "baseline_decode_tps_avg": float(obj.get("baseline_decode_tps_avg", 0.0)),
    "candidate_decode_tps_avg": float(obj.get("candidate_decode_tps_avg", 0.0)),
    "decode_drop_ratio": float(obj.get("decode_drop_ratio", 0.0)),
    "decode_drop_target": float(obj.get("decode_drop_target", 0.0)),
    "runtime_decomp_best_us": float(obj.get("runtime_decomp_best_us", 0.0)),
    "decode_cache_hit_best": float(obj.get("decode_cache_hit_best", 0.0)),
}
print(json.dumps(row, ensure_ascii=True))
PY
  echo "PASS: ${report_md}"
}

# 1) Full-mode strict gate
run_runtime_case \
  "runtime_full_strict" \
  "full" \
  "MAX_LOSSLESS_DECOMP_US=${FULL_DECOMP_BUDGET_US}"

# 2) Store-mode tuned gate (target 3 critical path)
run_runtime_case \
  "runtime_store_tuned" \
  "store" \
  "KV_LOSSLESS_STORE_DISABLE_FRONT=1" \
  "KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS=16" \
  "KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS=384" \
  "MAX_LOSSLESS_DECOMP_US=${STORE_DECOMP_BUDGET_US}"

# 3) Full-coverage kept-scope gate
run_runtime_case \
  "runtime_full_coverage" \
  "full" \
  "KV_LOSSLESS_KEPT_SAMPLE_LAYERS=0" \
  "KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL=0" \
  "MAX_LOSSLESS_DECOMP_US=${COVERAGE_DECOMP_BUDGET_US}"

# 4) Stability pack
run_m3_case "m3_pack"

# 5) Real prompt llm_demo pack
run_llm_demo_case "llm_demo_real_prompt_pack"

python3 - "${RESULTS_JSONL}" "${REPORT_MD}" "${REPORT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
out_md = Path(sys.argv[2])
out_json = Path(sys.argv[3])
rows = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
if not rows:
    raise SystemExit("no results collected")

runtime_rows = [r for r in rows if r.get("suite") == "runtime"]
m3_rows = [r for r in rows if r.get("suite") == "m3"]
llm_demo_rows = [r for r in rows if r.get("suite") == "llm_demo"]
overall_pass = all(bool(r.get("overall_pass", False)) for r in rows)

report = {
    "overall_pass": overall_pass,
    "total_cases": len(rows),
    "runtime_cases": len(runtime_rows),
    "m3_cases": len(m3_rows),
    "llm_demo_cases": len(llm_demo_rows),
    "rows": rows,
}
out_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

lines = []
lines.append("# H2O v6 Final Acceptance Report")
lines.append("")
lines.append(f"- overall_pass: {str(overall_pass).lower()}")
lines.append(f"- total_cases: {len(rows)}")
lines.append(f"- runtime_cases: {len(runtime_rows)}")
lines.append(f"- m3_cases: {len(m3_rows)}")
lines.append(f"- llm_demo_cases: {len(llm_demo_rows)}")
lines.append("")

if runtime_rows:
    lines.append("## Runtime Cases")
    lines.append("")
    for idx, r in enumerate(runtime_rows, 1):
        lines.append(
            f"{idx}. case={r['case']}, overall_pass={str(r['overall_pass']).lower()}, "
            f"decode={r['decode_best']:.4f}, drop={r['decode_drop_ratio']:.6f}, "
            f"lossy={r['lossy_best']:.4f}, lossless_online={r['lossless_online_value']:.4f}, "
            f"decomp_us={r['runtime_decomp_best_us']:.4f}, "
            f"queue={r['runtime_queue_peak_best']:.4f}, fallback={r['runtime_fallback_best']:.4f}, "
            f"summary=`{r['summary']}`"
        )
    lines.append("")

if m3_rows:
    lines.append("## M3 Cases")
    lines.append("")
    for idx, r in enumerate(m3_rows, 1):
        lines.append(
            f"{idx}. case={r['case']}, overall_pass={str(r['overall_pass']).lower()}, "
            f"pass_runs={int(r['overall_pass_runs'])}/{int(r['total_runs'])}, "
            f"joint_full_decode=[{r['joint_full_decode_best_min']:.4f},{r['joint_full_decode_best_max']:.4f}], "
            f"joint_store_decode=[{r['joint_store_decode_best_min']:.4f},{r['joint_store_decode_best_max']:.4f}], "
            f"joint_store_decomp_us_max={r['joint_store_runtime_decomp_best_us_max']:.4f}, "
            f"summary=`{r['summary']}`"
        )
    lines.append("")

if llm_demo_rows:
    lines.append("## llm_demo Cases")
    lines.append("")
    for idx, r in enumerate(llm_demo_rows, 1):
        lines.append(
            f"{idx}. case={r['case']}, overall_pass={str(r['overall_pass']).lower()}, "
            f"baseline_decode={r['baseline_decode_tps_avg']:.4f}, "
            f"candidate_decode={r['candidate_decode_tps_avg']:.4f}, "
            f"drop={r['decode_drop_ratio']:.6f}/{r['decode_drop_target']:.6f}, "
            f"decomp_us={r['runtime_decomp_best_us']:.4f}, "
            f"cache_hit={r['decode_cache_hit_best']:.4f}, "
            f"summary=`{r['summary']}`"
        )
    lines.append("")

out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {out_md}")
print(f"Wrote {out_json}")
if not overall_pass:
    raise SystemExit(1)
PY

echo ""
echo "============================================================"
echo " Final Result"
echo "============================================================"
cat "${REPORT_MD}"
echo ""
echo "Output directory: ${BASE_OUT}"
echo "Done."
