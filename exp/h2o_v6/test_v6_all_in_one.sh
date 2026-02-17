#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# H2O v6 All-In-One Validation
# - Runs runtime (fixed/same_batch/rolling), m3, llm_demo, final acceptance
# - Keeps each suite's raw outputs under dedicated folders
# - Produces one top-level aggregate report
# ============================================================================

BASE_OUT="${BASE_OUT:-exp/h2o_v6/out_all_in_one_v6_$(date +%Y%m%d_%H%M%S)}"
RESULTS_JSONL="${BASE_OUT}/all_in_one_results.jsonl"
REPORT_MD="${BASE_OUT}/all_in_one_report.md"
REPORT_JSON="${BASE_OUT}/all_in_one_report.json"

# Run matrix toggles
RUN_RUNTIME_FIXED="${RUN_RUNTIME_FIXED:-1}"
RUN_RUNTIME_SAME_BATCH="${RUN_RUNTIME_SAME_BATCH:-1}"
RUN_RUNTIME_ROLLING="${RUN_RUNTIME_ROLLING:-1}"
RUN_M3="${RUN_M3:-1}"
RUN_LLM_DEMO="${RUN_LLM_DEMO:-1}"
RUN_FINAL="${RUN_FINAL:-1}"

# Shared knobs
DECODE_BASELINE="${DECODE_BASELINE:-6.60}"
DECODE_DROP_TARGET="${DECODE_DROP_TARGET:-0.05}"
DECODE_BASELINE_HISTORY="${DECODE_BASELINE_HISTORY:-${BASE_OUT}/decode_baseline_history.jsonl}"
DECODE_BASELINE_ROLLING_WINDOW="${DECODE_BASELINE_ROLLING_WINDOW:-8}"
DECODE_BASELINE_ROLLING_MIN_SAMPLES="${DECODE_BASELINE_ROLLING_MIN_SAMPLES:-3}"
DECODE_BASELINE_KEY="${DECODE_BASELINE_KEY:-$(hostname 2>/dev/null)_all_in_one}"
UPDATE_DECODE_BASELINE_HISTORY="${UPDATE_DECODE_BASELINE_HISTORY:-1}"

MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK:-8}"
MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK:-0}"
MAX_LOSSLESS_BACKPRESSURE_SKIP="${MAX_LOSSLESS_BACKPRESSURE_SKIP:--1}"
MAX_LOSSLESS_DECOMP_US="${MAX_LOSSLESS_DECOMP_US:--1}"
MAX_LOSSLESS_ASYNC_WAIT_US="${MAX_LOSSLESS_ASYNC_WAIT_US:--1}"
STRICT_RUNTIME_METRIC_COLUMNS="${STRICT_RUNTIME_METRIC_COLUMNS:-1}"
REQUIRE_DECODE_CACHE_HIT="${REQUIRE_DECODE_CACHE_HIT:-0}"
REQUIRE_ASYNC_QUEUE_ACTIVITY="${REQUIRE_ASYNC_QUEUE_ACTIVITY:-0}"
REQUIRE_DECODE_CACHE_ACTIVITY="${REQUIRE_DECODE_CACHE_ACTIVITY:-0}"
KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS:-1}"
KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS:-64}"

# M3 knobs
M3_FULL_RUNS="${M3_FULL_RUNS:-3}"
M3_STORE_RUNS="${M3_STORE_RUNS:-1}"

# llm_demo knobs
PROMPT_DIR="${PROMPT_DIR:-/home10T/ljq/MNN/exp/gear_fp16/prompts}"
PROMPT_PATTERN="${PROMPT_PATTERN:-prompt_*.txt}"
PROMPT_MANIFEST="${PROMPT_MANIFEST:-}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
MAX_PROMPTS_PER_BUCKET="${MAX_PROMPTS_PER_BUCKET:-0}"
PROMPT_BUCKET_LIST="${PROMPT_BUCKET_LIST:-128,512,2048}"
PROMPT_SAMPLE_MODE="${PROMPT_SAMPLE_MODE:-stratified}"
LLM_DEMO_DECODE_DROP_TARGET="${LLM_DEMO_DECODE_DROP_TARGET:-0.08}"
LLM_DEMO_REQUIRE_RUNTIME_DECOMP="${LLM_DEMO_REQUIRE_RUNTIME_DECOMP:-1}"
LLM_DEMO_REQUIRE_DECODE_CACHE_HIT="${LLM_DEMO_REQUIRE_DECODE_CACHE_HIT:-0}"
LLM_DEMO_REQUIRE_BUCKET_DECODE_PASS="${LLM_DEMO_REQUIRE_BUCKET_DECODE_PASS:-1}"

mkdir -p "${BASE_OUT}"
: > "${RESULTS_JSONL}"

echo "============================================================"
echo " H2O v6 All-In-One Validation"
echo " BASE_OUT = ${BASE_OUT}"
echo "============================================================"
echo "Run matrix:"
echo "  runtime_fixed      = ${RUN_RUNTIME_FIXED}"
echo "  runtime_same_batch = ${RUN_RUNTIME_SAME_BATCH}"
echo "  runtime_rolling    = ${RUN_RUNTIME_ROLLING}"
echo "  m3                 = ${RUN_M3}"
echo "  llm_demo           = ${RUN_LLM_DEMO}"
echo "  final              = ${RUN_FINAL}"
echo ""
echo "Shared knobs:"
echo "  DECODE_BASELINE=${DECODE_BASELINE}"
echo "  DECODE_DROP_TARGET=${DECODE_DROP_TARGET}"
echo "  DECODE_BASELINE_HISTORY=${DECODE_BASELINE_HISTORY}"
echo "  MAX_LOSSLESS_QUEUE_PEAK=${MAX_LOSSLESS_QUEUE_PEAK}"
echo "  MAX_LOSSLESS_FALLBACK=${MAX_LOSSLESS_FALLBACK}"
echo "  MAX_LOSSLESS_BACKPRESSURE_SKIP=${MAX_LOSSLESS_BACKPRESSURE_SKIP}"
echo "  MAX_LOSSLESS_DECOMP_US=${MAX_LOSSLESS_DECOMP_US}"
echo "  MAX_LOSSLESS_ASYNC_WAIT_US=${MAX_LOSSLESS_ASYNC_WAIT_US}"
echo "  KV_LOSSLESS_ASYNC_THREADS=${KV_LOSSLESS_ASYNC_THREADS}"
echo "  KV_LOSSLESS_DECODE_CACHE_BLOCKS=${KV_LOSSLESS_DECODE_CACHE_BLOCKS}"
echo ""

run_runtime_case() {
  local case_name="$1"
  local baseline_mode="$2"
  shift 2
  local -a extra_env=("$@")
  local out_dir="${BASE_OUT}/${case_name}"
  local console_log="${BASE_OUT}/${case_name}.console.log"
  local summary_path="${out_dir}/summary.md"

  echo "---- runtime: ${case_name} (baseline_mode=${baseline_mode}) ----"
  set +e
  env \
    OUT="${out_dir}" \
    DECODE_BASELINE_MODE="${baseline_mode}" \
    DECODE_BASELINE="${DECODE_BASELINE}" \
    DECODE_DROP_TARGET="${DECODE_DROP_TARGET}" \
    DECODE_BASELINE_HISTORY="${DECODE_BASELINE_HISTORY}" \
    DECODE_BASELINE_ROLLING_WINDOW="${DECODE_BASELINE_ROLLING_WINDOW}" \
    DECODE_BASELINE_ROLLING_MIN_SAMPLES="${DECODE_BASELINE_ROLLING_MIN_SAMPLES}" \
    DECODE_BASELINE_KEY="${DECODE_BASELINE_KEY}_${case_name}" \
    UPDATE_DECODE_BASELINE_HISTORY="${UPDATE_DECODE_BASELINE_HISTORY}" \
    MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK}" \
    MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK}" \
    MAX_LOSSLESS_BACKPRESSURE_SKIP="${MAX_LOSSLESS_BACKPRESSURE_SKIP}" \
    MAX_LOSSLESS_DECOMP_US="${MAX_LOSSLESS_DECOMP_US}" \
    MAX_LOSSLESS_ASYNC_WAIT_US="${MAX_LOSSLESS_ASYNC_WAIT_US}" \
    STRICT_RUNTIME_METRIC_COLUMNS="${STRICT_RUNTIME_METRIC_COLUMNS}" \
    REQUIRE_DECODE_CACHE_HIT="${REQUIRE_DECODE_CACHE_HIT}" \
    REQUIRE_ASYNC_QUEUE_ACTIVITY="${REQUIRE_ASYNC_QUEUE_ACTIVITY}" \
    REQUIRE_DECODE_CACHE_ACTIVITY="${REQUIRE_DECODE_CACHE_ACTIVITY}" \
    KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS}" \
    KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS}" \
    "${extra_env[@]}" \
    bash exp/h2o_v6/test_v6_runtime.sh > "${console_log}" 2>&1
  local rc=$?
  set -e

  python3 - "${RESULTS_JSONL}" "${case_name}" "${baseline_mode}" "${rc}" "${summary_path}" "${out_dir}" "${console_log}" <<'PY'
import json
import re
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
case_name = sys.argv[2]
baseline_mode = sys.argv[3]
return_code = int(sys.argv[4])
summary_path = Path(sys.argv[5])
out_dir = sys.argv[6]
console_log = sys.argv[7]

gate = {}
if summary_path.exists():
    in_gate = False
    for raw in summary_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line == "## Quality Gate":
            in_gate = True
            continue
        if in_gate and line.startswith("## "):
            break
        if in_gate and line.startswith("- ") and ":" in line:
            k, v = line[2:].split(":", 1)
            gate[k.strip()] = v.strip().strip("`")

def parse_float(v, default=0.0):
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(v))
    return float(m.group(0)) if m else default

overall_pass = (return_code == 0) and gate.get("overall_pass", "").lower() == "true"
row = {
    "suite": "runtime",
    "case": case_name,
    "baseline_mode": baseline_mode,
    "return_code": return_code,
    "overall_pass": overall_pass,
    "summary": str(summary_path),
    "out_dir": out_dir,
    "console_log": console_log,
    "decode_baseline": parse_float(gate.get("decode_baseline", 0.0)),
    "decode_baseline_source": gate.get("decode_baseline_source", ""),
    "decode_baseline_samples": int(parse_float(gate.get("decode_baseline_samples", 0), 0)),
    "decode_best": parse_float(gate.get("decode_best", 0.0)),
    "decode_drop_ratio": parse_float(gate.get("decode_drop_ratio", 0.0)),
    "lossy_best": parse_float(gate.get("lossy_best", 0.0)),
    "lossless_online_value": parse_float(gate.get("lossless_online_value", 0.0)),
    "runtime_decomp_best_us": parse_float(gate.get("runtime_decomp_best_us", 0.0)),
    "runtime_queue_peak_best": parse_float(gate.get("runtime_queue_peak_best", 0.0)),
    "runtime_fallback_best": parse_float(gate.get("runtime_fallback_best", 0.0)),
    "runtime_backpressure_skip_best": parse_float(gate.get("runtime_backpressure_skip_best", 0.0)),
}
with results_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=True) + "\n")
PY

  if [[ ${rc} -eq 0 ]]; then
    echo "PASS: ${case_name}"
  else
    echo "FAIL: ${case_name} (rc=${rc}), see ${console_log}"
  fi
}

run_m3_case() {
  local case_name="$1"
  local out_dir="${BASE_OUT}/${case_name}"
  local console_log="${BASE_OUT}/${case_name}.console.log"
  local report_json="${out_dir}/m3_report.json"
  local report_md="${out_dir}/m3_report.md"

  echo "---- m3: ${case_name} ----"
  set +e
  env \
    BASE_OUT="${out_dir}" \
    FULL_RUNS="${M3_FULL_RUNS}" \
    STORE_RUNS="${M3_STORE_RUNS}" \
    MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK}" \
    MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK}" \
    MAX_LOSSLESS_BACKPRESSURE_SKIP="${MAX_LOSSLESS_BACKPRESSURE_SKIP}" \
    MAX_LOSSLESS_DECOMP_US="${MAX_LOSSLESS_DECOMP_US}" \
    MAX_LOSSLESS_ASYNC_WAIT_US="${MAX_LOSSLESS_ASYNC_WAIT_US}" \
    STRICT_RUNTIME_METRIC_COLUMNS="${STRICT_RUNTIME_METRIC_COLUMNS}" \
    REQUIRE_DECODE_CACHE_HIT="${REQUIRE_DECODE_CACHE_HIT}" \
    REQUIRE_ASYNC_QUEUE_ACTIVITY="${REQUIRE_ASYNC_QUEUE_ACTIVITY}" \
    REQUIRE_DECODE_CACHE_ACTIVITY="${REQUIRE_DECODE_CACHE_ACTIVITY}" \
    KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS}" \
    KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS}" \
    bash exp/h2o_v6/test_v6_m3.sh > "${console_log}" 2>&1
  local rc=$?
  set -e

  python3 - "${RESULTS_JSONL}" "${case_name}" "${rc}" "${report_json}" "${report_md}" "${out_dir}" "${console_log}" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
case_name = sys.argv[2]
return_code = int(sys.argv[3])
report_json_path = Path(sys.argv[4])
report_md_path = Path(sys.argv[5])
out_dir = sys.argv[6]
console_log = sys.argv[7]

obj = {}
if report_json_path.exists():
    obj = json.loads(report_json_path.read_text(encoding="utf-8", errors="ignore"))

total_runs = int(obj.get("total_runs", 0))
overall_pass_runs = int(obj.get("overall_pass_runs", 0))
overall_pass = (return_code == 0) and (total_runs > 0 and overall_pass_runs == total_runs)

row = {
    "suite": "m3",
    "case": case_name,
    "return_code": return_code,
    "overall_pass": overall_pass,
    "summary": str(report_md_path),
    "report_json": str(report_json_path),
    "out_dir": out_dir,
    "console_log": console_log,
    "total_runs": total_runs,
    "overall_pass_runs": overall_pass_runs,
}
with results_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=True) + "\n")
PY

  if [[ ${rc} -eq 0 ]]; then
    echo "PASS: ${case_name}"
  else
    echo "FAIL: ${case_name} (rc=${rc}), see ${console_log}"
  fi
}

run_llm_demo_case() {
  local case_name="$1"
  local out_dir="${BASE_OUT}/${case_name}"
  local console_log="${BASE_OUT}/${case_name}.console.log"
  local report_json="${out_dir}/llm_demo_report.json"
  local report_md="${out_dir}/llm_demo_report.md"

  echo "---- llm_demo: ${case_name} ----"
  set +e
  env \
    OUT="${out_dir}" \
    PROMPT_DIR="${PROMPT_DIR}" \
    PROMPT_PATTERN="${PROMPT_PATTERN}" \
    PROMPT_MANIFEST="${PROMPT_MANIFEST}" \
    MAX_PROMPTS="${MAX_PROMPTS}" \
    MAX_PROMPTS_PER_BUCKET="${MAX_PROMPTS_PER_BUCKET}" \
    PROMPT_BUCKET_LIST="${PROMPT_BUCKET_LIST}" \
    PROMPT_SAMPLE_MODE="${PROMPT_SAMPLE_MODE}" \
    DECODE_DROP_TARGET="${LLM_DEMO_DECODE_DROP_TARGET}" \
    REQUIRE_RUNTIME_DECOMP="${LLM_DEMO_REQUIRE_RUNTIME_DECOMP}" \
    REQUIRE_DECODE_CACHE_HIT="${LLM_DEMO_REQUIRE_DECODE_CACHE_HIT}" \
    REQUIRE_BUCKET_DECODE_PASS="${LLM_DEMO_REQUIRE_BUCKET_DECODE_PASS}" \
    KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS}" \
    KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS}" \
    bash exp/h2o_v6/test_v6_llm_demo.sh > "${console_log}" 2>&1
  local rc=$?
  set -e

  python3 - "${RESULTS_JSONL}" "${case_name}" "${rc}" "${report_json}" "${report_md}" "${out_dir}" "${console_log}" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
case_name = sys.argv[2]
return_code = int(sys.argv[3])
report_json_path = Path(sys.argv[4])
report_md_path = Path(sys.argv[5])
out_dir = sys.argv[6]
console_log = sys.argv[7]

obj = {}
if report_json_path.exists():
    obj = json.loads(report_json_path.read_text(encoding="utf-8", errors="ignore"))

overall_pass = (return_code == 0) and bool(obj.get("overall_pass", False))
row = {
    "suite": "llm_demo",
    "case": case_name,
    "return_code": return_code,
    "overall_pass": overall_pass,
    "summary": str(report_md_path),
    "report_json": str(report_json_path),
    "out_dir": out_dir,
    "console_log": console_log,
    "baseline_decode_tps_avg": float(obj.get("baseline_decode_tps_avg", 0.0)),
    "candidate_decode_tps_avg": float(obj.get("candidate_decode_tps_avg", 0.0)),
    "decode_drop_ratio": float(obj.get("decode_drop_ratio", 0.0)),
    "candidate_runtime_total_ratio_avg": float(obj.get("candidate_runtime_total_ratio_avg", 0.0)),
    "runtime_decomp_best_us": float(obj.get("runtime_decomp_best_us", 0.0)),
    "bucket_decode_pass": bool(obj.get("bucket_decode_pass", True)),
}
with results_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=True) + "\n")
PY

  if [[ ${rc} -eq 0 ]]; then
    echo "PASS: ${case_name}"
  else
    echo "FAIL: ${case_name} (rc=${rc}), see ${console_log}"
  fi
}

run_final_case() {
  local case_name="$1"
  local out_dir="${BASE_OUT}/${case_name}"
  local console_log="${BASE_OUT}/${case_name}.console.log"
  local report_json="${out_dir}/final_report.json"
  local report_md="${out_dir}/final_report.md"

  echo "---- final: ${case_name} ----"
  set +e
  env \
    BASE_OUT="${out_dir}" \
    DECODE_BASELINE="${DECODE_BASELINE}" \
    DECODE_DROP_TARGET="${DECODE_DROP_TARGET}" \
    MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK}" \
    MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK}" \
    MAX_LOSSLESS_BACKPRESSURE_SKIP="${MAX_LOSSLESS_BACKPRESSURE_SKIP}" \
    MAX_LOSSLESS_ASYNC_WAIT_US="${MAX_LOSSLESS_ASYNC_WAIT_US}" \
    STRICT_RUNTIME_METRIC_COLUMNS="${STRICT_RUNTIME_METRIC_COLUMNS}" \
    REQUIRE_DECODE_CACHE_HIT="${REQUIRE_DECODE_CACHE_HIT}" \
    REQUIRE_ASYNC_QUEUE_ACTIVITY="${REQUIRE_ASYNC_QUEUE_ACTIVITY}" \
    REQUIRE_DECODE_CACHE_ACTIVITY="${REQUIRE_DECODE_CACHE_ACTIVITY}" \
    KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS}" \
    KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS}" \
    PROMPT_DIR="${PROMPT_DIR}" \
    PROMPT_PATTERN="${PROMPT_PATTERN}" \
    PROMPT_MANIFEST="${PROMPT_MANIFEST}" \
    MAX_PROMPTS="${MAX_PROMPTS}" \
    MAX_PROMPTS_PER_BUCKET="${MAX_PROMPTS_PER_BUCKET}" \
    PROMPT_BUCKET_LIST="${PROMPT_BUCKET_LIST}" \
    PROMPT_SAMPLE_MODE="${PROMPT_SAMPLE_MODE}" \
    LLM_DEMO_DECODE_DROP_TARGET="${LLM_DEMO_DECODE_DROP_TARGET}" \
    LLM_DEMO_REQUIRE_RUNTIME_DECOMP="${LLM_DEMO_REQUIRE_RUNTIME_DECOMP}" \
    LLM_DEMO_REQUIRE_DECODE_CACHE_HIT="${LLM_DEMO_REQUIRE_DECODE_CACHE_HIT}" \
    bash exp/h2o_v6/test_v6_final.sh > "${console_log}" 2>&1
  local rc=$?
  set -e

  python3 - "${RESULTS_JSONL}" "${case_name}" "${rc}" "${report_json}" "${report_md}" "${out_dir}" "${console_log}" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
case_name = sys.argv[2]
return_code = int(sys.argv[3])
report_json_path = Path(sys.argv[4])
report_md_path = Path(sys.argv[5])
out_dir = sys.argv[6]
console_log = sys.argv[7]

obj = {}
if report_json_path.exists():
    obj = json.loads(report_json_path.read_text(encoding="utf-8", errors="ignore"))

overall_pass = (return_code == 0) and bool(obj.get("overall_pass", False))
row = {
    "suite": "final",
    "case": case_name,
    "return_code": return_code,
    "overall_pass": overall_pass,
    "summary": str(report_md_path),
    "report_json": str(report_json_path),
    "out_dir": out_dir,
    "console_log": console_log,
    "total_cases": int(obj.get("total_cases", 0)),
    "runtime_cases": int(obj.get("runtime_cases", 0)),
    "m3_cases": int(obj.get("m3_cases", 0)),
    "llm_demo_cases": int(obj.get("llm_demo_cases", 0)),
}
with results_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=True) + "\n")
PY

  if [[ ${rc} -eq 0 ]]; then
    echo "PASS: ${case_name}"
  else
    echo "FAIL: ${case_name} (rc=${rc}), see ${console_log}"
  fi
}

if [[ "${RUN_RUNTIME_FIXED}" -ne 0 ]]; then
  run_runtime_case "runtime_fixed" "fixed"
fi
if [[ "${RUN_RUNTIME_SAME_BATCH}" -ne 0 ]]; then
  run_runtime_case "runtime_same_batch" "same_batch"
fi
if [[ "${RUN_RUNTIME_ROLLING}" -ne 0 ]]; then
  run_runtime_case "runtime_rolling" "rolling"
fi
if [[ "${RUN_M3}" -ne 0 ]]; then
  run_m3_case "m3_pack"
fi
if [[ "${RUN_LLM_DEMO}" -ne 0 ]]; then
  run_llm_demo_case "llm_demo_real_prompt_pack"
fi
if [[ "${RUN_FINAL}" -ne 0 ]]; then
  run_final_case "final_acceptance_pack"
fi

python3 - "${RESULTS_JSONL}" "${REPORT_MD}" "${REPORT_JSON}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

results_path = Path(sys.argv[1])
report_md = Path(sys.argv[2])
report_json = Path(sys.argv[3])

rows = [json.loads(x) for x in results_path.read_text(encoding="utf-8").splitlines() if x.strip()]
if not rows:
    raise SystemExit("no cases executed")

by_suite = defaultdict(list)
for r in rows:
    by_suite[r.get("suite", "unknown")].append(r)

overall_pass = all(bool(r.get("overall_pass", False)) for r in rows)
report = {
    "overall_pass": overall_pass,
    "total_cases": len(rows),
    "suite_counts": {k: len(v) for k, v in sorted(by_suite.items())},
    "rows": rows,
}
report_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

lines = []
lines.append("# H2O v6 All-In-One Report")
lines.append("")
lines.append(f"- overall_pass: {str(overall_pass).lower()}")
lines.append(f"- total_cases: {len(rows)}")
for suite, items in sorted(by_suite.items()):
    suite_pass = sum(1 for x in items if bool(x.get("overall_pass", False)))
    lines.append(f"- {suite}_cases: {len(items)} ({suite_pass} pass)")
lines.append("")

for suite, items in sorted(by_suite.items()):
    lines.append(f"## {suite}")
    lines.append("")
    for idx, r in enumerate(items, 1):
        lines.append(
            f"{idx}. case={r.get('case','')}, pass={str(bool(r.get('overall_pass', False))).lower()}, "
            f"rc={int(r.get('return_code', 1))}, summary=`{r.get('summary','')}`, "
            f"out_dir=`{r.get('out_dir','')}`, console=`{r.get('console_log','')}`"
        )
    lines.append("")

report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {report_md}")
print(f"Wrote {report_json}")
if not overall_pass:
    raise SystemExit(1)
PY

echo ""
echo "============================================================"
echo " All-In-One Result"
echo "============================================================"
cat "${REPORT_MD}"
echo ""
echo "Output directory: ${BASE_OUT}"
