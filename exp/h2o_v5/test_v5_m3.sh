#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# H2O v5 M3 Validation Runner
# - Runs multiple strict-gate scenarios
# - Collects per-run quality snapshots
# - Produces consolidated M3 report (markdown + json)
# ============================================================================

BASE_OUT="${BASE_OUT:-exp/h2o_v5/out_m3_$(date +%Y%m%d_%H%M%S)}"
FULL_RUNS="${FULL_RUNS:-3}"
STORE_RUNS="${STORE_RUNS:-1}"

KV_LOSSLESS_SCOPE="${KV_LOSSLESS_SCOPE:-front_n_and_h2o_kept}"
KV_LOSSLESS_KEPT_SAMPLE_LAYERS="${KV_LOSSLESS_KEPT_SAMPLE_LAYERS:-1}"
KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL="${KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL:-2}"
KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL="${KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL:--1}"
KV_LOSSLESS_STORE_DISABLE_FRONT="${KV_LOSSLESS_STORE_DISABLE_FRONT:--1}"
KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS="${KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS:--1}"
KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS="${KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS:--1}"
MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK:-8}"
MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK:-0}"
MAX_LOSSLESS_DECOMP_US="${MAX_LOSSLESS_DECOMP_US:--1}"

RESULTS_JSONL="${BASE_OUT}/results.jsonl"
REPORT_MD="${BASE_OUT}/m3_report.md"
REPORT_JSON="${BASE_OUT}/m3_report.json"

mkdir -p "${BASE_OUT}"
: > "${RESULTS_JSONL}"

echo "============================================================"
echo " H2O v5 M3 Validation"
echo " BASE_OUT = ${BASE_OUT}"
echo " FULL_RUNS = ${FULL_RUNS}, STORE_RUNS = ${STORE_RUNS}"
echo " LOSSLESS: front_interval=${KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL}, store_disable_front=${KV_LOSSLESS_STORE_DISABLE_FRONT}"
echo " STORE_THROTTLE: bootstrap=${KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS}, grouped_step=${KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS}"
echo " GATES: queue<=${MAX_LOSSLESS_QUEUE_PEAK}, fallback<=${MAX_LOSSLESS_FALLBACK}, decomp<=${MAX_LOSSLESS_DECOMP_US}"
echo "============================================================"

run_case() {
  local case_name="$1"
  local runtime_mode="$2"
  local run_idx="$3"
  local out_dir="${BASE_OUT}/${case_name}_run$(printf '%02d' "${run_idx}")"
  local console_log="${BASE_OUT}/${case_name}_run$(printf '%02d' "${run_idx}").console.log"
  local summary_path="${out_dir}/summary.md"

  echo ""
  echo "---- ${case_name} run ${run_idx} (mode=${runtime_mode}) ----"
  if ! env \
    OUT="${out_dir}" \
    KV_LOSSLESS_SCOPE="${KV_LOSSLESS_SCOPE}" \
    KV_LOSSLESS_RUNTIME_MODE="${runtime_mode}" \
    KV_LOSSLESS_KEPT_SAMPLE_LAYERS="${KV_LOSSLESS_KEPT_SAMPLE_LAYERS}" \
    KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL="${KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL}" \
    KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL="${KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL}" \
    KV_LOSSLESS_STORE_DISABLE_FRONT="${KV_LOSSLESS_STORE_DISABLE_FRONT}" \
    KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS="${KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS}" \
    KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS="${KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS}" \
    MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK}" \
    MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK}" \
    MAX_LOSSLESS_DECOMP_US="${MAX_LOSSLESS_DECOMP_US}" \
    bash exp/h2o_v5/test_v5_runtime.sh > "${console_log}" 2>&1; then
    echo "FAIL: ${case_name} run ${run_idx} failed"
    echo "See: ${console_log}"
    tail -n 60 "${console_log}" || true
    return 1
  fi

  if [[ ! -s "${summary_path}" ]]; then
    echo "FAIL: missing summary: ${summary_path}"
    return 1
  fi

  python3 - "${summary_path}" "${case_name}" "${runtime_mode}" "${run_idx}" >> "${RESULTS_JSONL}" <<'PY'
import json
import re
import sys
from pathlib import Path

summary = Path(sys.argv[1])
case_name = sys.argv[2]
runtime_mode = sys.argv[3]
run_idx = int(sys.argv[4])

if not summary.exists():
    raise SystemExit(f"summary not found: {summary}")

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

obj = {
    "case": case_name,
    "runtime_mode": runtime_mode,
    "run": run_idx,
    "summary": str(summary),
    "quality_status": gate.get("quality_status", ""),
    "overall_pass": gate.get("overall_pass", "") == "true",
    "decode_pass": gate.get("decode_pass", "") == "true",
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
print(json.dumps(obj, ensure_ascii=True))
PY

  echo "PASS: ${summary_path}"
}

for i in $(seq 1 "${FULL_RUNS}"); do
  run_case "joint_full" "full" "${i}"
done

for i in $(seq 1 "${STORE_RUNS}"); do
  run_case "joint_store" "store" "${i}"
done

python3 - "${RESULTS_JSONL}" "${REPORT_MD}" "${REPORT_JSON}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

jsonl = Path(sys.argv[1])
out_md = Path(sys.argv[2])
out_json = Path(sys.argv[3])

rows = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
if not rows:
    raise SystemExit("no rows collected")

by_case = defaultdict(list)
for r in rows:
    by_case[r["case"]].append(r)

report = {
    "total_runs": len(rows),
    "overall_pass_runs": sum(1 for r in rows if r["overall_pass"]),
    "cases": {},
}

for case, items in sorted(by_case.items()):
    report["cases"][case] = {
        "runs": len(items),
        "overall_pass_runs": sum(1 for r in items if r["overall_pass"]),
        "decode_best_min": min(r["decode_best"] for r in items),
        "decode_best_max": max(r["decode_best"] for r in items),
        "decode_drop_ratio_max": max(r["decode_drop_ratio"] for r in items),
        "runtime_decomp_best_us_max": max(r["runtime_decomp_best_us"] for r in items),
        "runtime_queue_peak_best_max": max(r["runtime_queue_peak_best"] for r in items),
        "runtime_fallback_best_max": max(r["runtime_fallback_best"] for r in items),
    }

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

lines = []
lines.append("# H2O v5 M3 Report")
lines.append("")
lines.append(f"- total_runs: {report['total_runs']}")
lines.append(f"- overall_pass_runs: {report['overall_pass_runs']}")
lines.append("")
for case, info in sorted(report["cases"].items()):
    lines.append(f"## {case}")
    lines.append("")
    lines.append(f"- runs: {info['runs']}")
    lines.append(f"- overall_pass_runs: {info['overall_pass_runs']}")
    lines.append(f"- decode_best_min: {info['decode_best_min']:.4f}")
    lines.append(f"- decode_best_max: {info['decode_best_max']:.4f}")
    lines.append(f"- decode_drop_ratio_max: {info['decode_drop_ratio_max']:.6f}")
    lines.append(f"- runtime_decomp_best_us_max: {info['runtime_decomp_best_us_max']:.4f}")
    lines.append(f"- runtime_queue_peak_best_max: {info['runtime_queue_peak_best_max']:.4f}")
    lines.append(f"- runtime_fallback_best_max: {info['runtime_fallback_best_max']:.4f}")
    lines.append("")

lines.append("## Runs")
lines.append("")
for idx, r in enumerate(rows, 1):
    lines.append(
        f"{idx}. case={r['case']}, mode={r['runtime_mode']}, run={r['run']}, "
        f"overall_pass={str(r['overall_pass']).lower()}, decode={r['decode_best']:.4f}, "
        f"drop={r['decode_drop_ratio']:.6f}, decomp_us={r['runtime_decomp_best_us']:.4f}, "
        f"queue={r['runtime_queue_peak_best']:.4f}, fallback={r['runtime_fallback_best']:.4f}, "
        f"summary=`{r['summary']}`"
    )

out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {out_md}")
print(f"Wrote {out_json}")
PY

echo ""
echo "============================================================"
echo " M3 Result"
echo "============================================================"
cat "${REPORT_MD}"
echo ""
echo "Output directory: ${BASE_OUT}"
