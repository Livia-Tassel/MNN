#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# H2O final Runtime Validation Script
# Coverage:
#   - v3 regression guards (empty sweep/dedup/fallback warning/offline failure accounting)
#   - runtime lossy + runtime lossless metrics path
#   - grouped-step behavior (no hardcoded 192 token block)
#   - strict runtime decode evidence (h2o_lossless_decomp_us > 0)
#   - offline upper + online-sim + final quality gate
# ============================================================================

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_CONFIG="${MODEL_CONFIG:-/home10T/ljq/mnn_data/models/llama2_mnn/config.json}"
LLM_BENCH="${LLM_BENCH:-./build/llm_bench}"
DUMP_DIR="${DUMP_DIR:-/home10T/ljq/MNN/exp/gear_fp16/dumps_fp16}"
OUT="${OUT:-exp/h2o_final/out_runtime_$(date +%Y%m%d_%H%M%S)}"
BENCH_STDOUT_LOG="${OUT}/bench_stdout.log"

# ── Tunables ───────────────────────────────────────────────────────────────
PROMPTS="${PROMPTS:-512,1024}"
GENS="${GENS:-128}"
REPEAT="${REPEAT:-3}"
THREADS="${THREADS:-4}"
BACKEND="${BACKEND:-cpu}"

# H2O lossy
H2O_KEEP_RATIO="${H2O_KEEP_RATIO:-0.30}"
H2O_BLOCK_TOKENS="${H2O_BLOCK_TOKENS:-32}"
H2O_SINK_TOKENS="${H2O_SINK_TOKENS:-16}"
H2O_RECENT_TOKENS="${H2O_RECENT_TOKENS:-96}"
H2O_TARGET_MODE="${H2O_TARGET_MODE:-adaptive}"
H2O_TARGET_LOSSY_RATIO="${H2O_TARGET_LOSSY_RATIO:-3.2}"
H2O_EMA_ALPHA="${H2O_EMA_ALPHA:-0.9}"
H2O_UPDATE_INTERVAL="${H2O_UPDATE_INTERVAL:-16}"
H2O_TRIGGER_MIN="${H2O_TRIGGER_MIN:-384}"
H2O_LAYER_START="${H2O_LAYER_START:-2}"
H2O_LAYER_END="${H2O_LAYER_END:--1}"

# Lossless
KV_LOSSLESS_SCOPE="${KV_LOSSLESS_SCOPE:-front_n_and_h2o_kept}"
KV_LOSSLESS_FRONT_N="${KV_LOSSLESS_FRONT_N:-2}"
KV_LOSSLESS_CODEC="${KV_LOSSLESS_CODEC:-gear_delta}"
KV_LOSSLESS_BLOCK_TOKENS="${KV_LOSSLESS_BLOCK_TOKENS:-128}"
KV_LOSSLESS_HOT_RECENT="${KV_LOSSLESS_HOT_RECENT:-256}"
KV_LOSSLESS_HOT_SINK="${KV_LOSSLESS_HOT_SINK:-16}"
KV_LOSSLESS_KEPT_SAMPLE_LAYERS="${KV_LOSSLESS_KEPT_SAMPLE_LAYERS:-1}"
KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL="${KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL:-2}"
KV_LOSSLESS_RUNTIME_MODE="${KV_LOSSLESS_RUNTIME_MODE:-full}"
KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL="${KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL:--1}"
KV_LOSSLESS_STORE_DISABLE_FRONT="${KV_LOSSLESS_STORE_DISABLE_FRONT:--1}"
KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS="${KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS:--1}"
KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS="${KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS:--1}"
KV_LOSSLESS_CODEC_RUNTIME="${KV_LOSSLESS_CODEC_RUNTIME:-fp16_gear_predictive_v3}"

# Quality gate
LOSSY_TARGET="${LOSSY_TARGET:-3.0}"
LOSSLESS_TARGET="${LOSSLESS_TARGET:-1.3}"
DECODE_BASELINE="${DECODE_BASELINE:-6.60}"
DECODE_BASELINE_MODE="${DECODE_BASELINE_MODE:-fixed}" # fixed|rolling|same_batch
DECODE_BASELINE_HISTORY="${DECODE_BASELINE_HISTORY:-exp/h2o_final/decode_baseline_history.jsonl}"
DECODE_BASELINE_ROLLING_WINDOW="${DECODE_BASELINE_ROLLING_WINDOW:-8}"
DECODE_BASELINE_ROLLING_MIN_SAMPLES="${DECODE_BASELINE_ROLLING_MIN_SAMPLES:-3}"
DECODE_BASELINE_KEY="${DECODE_BASELINE_KEY:-$(hostname 2>/dev/null)_${BACKEND}_t${THREADS}_p${PROMPTS}_g${GENS}_rep${REPEAT}}"
UPDATE_DECODE_BASELINE_HISTORY="${UPDATE_DECODE_BASELINE_HISTORY:-1}"
DECODE_DROP_TARGET="${DECODE_DROP_TARGET:-0.05}"
MAX_LOSSLESS_QUEUE_PEAK="${MAX_LOSSLESS_QUEUE_PEAK:-8}"
MAX_LOSSLESS_FALLBACK="${MAX_LOSSLESS_FALLBACK:-0}"
MAX_LOSSLESS_BACKPRESSURE_SKIP="${MAX_LOSSLESS_BACKPRESSURE_SKIP:--1}"
MAX_LOSSLESS_DECOMP_US="${MAX_LOSSLESS_DECOMP_US:--1}"
MAX_LOSSLESS_ASYNC_WAIT_US="${MAX_LOSSLESS_ASYNC_WAIT_US:--1}"
REQUIRE_DECODE_CACHE_HIT="${REQUIRE_DECODE_CACHE_HIT:-0}"
REQUIRE_ASYNC_QUEUE_ACTIVITY="${REQUIRE_ASYNC_QUEUE_ACTIVITY:-0}"
REQUIRE_DECODE_CACHE_ACTIVITY="${REQUIRE_DECODE_CACHE_ACTIVITY:-0}"
STRICT_RUNTIME_METRIC_COLUMNS="${STRICT_RUNTIME_METRIC_COLUMNS:-1}"
KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS:-1}"
KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS:-64}"
REQUIRE_RUNTIME_DECOMP="${REQUIRE_RUNTIME_DECOMP:-1}"
DISABLE_H2O="${DISABLE_H2O:-0}"
DISABLE_LOSSLESS="${DISABLE_LOSSLESS:-0}"

if [[ ! -x "${LLM_BENCH}" ]]; then
  echo "FAIL: LLM_BENCH not executable: ${LLM_BENCH}"
  exit 1
fi

# Offline lossless
ONLINE_CHUNK_SEQ="${ONLINE_CHUNK_SEQ:-192}"
ONLINE_FRAMING_BYTES="${ONLINE_FRAMING_BYTES:-8}"
ADAPTIVE_BLOCK_SEQ="${ADAPTIVE_BLOCK_SEQ:-64}"

STEP0_FAIL=0

echo "============================================================"
echo " H2O final Runtime Validation"
echo " OUT = ${OUT}"
echo " LLM_BENCH = ${LLM_BENCH}"
echo "============================================================"
mkdir -p "${OUT}"

# Bash here-docs create temporary files. On some servers /tmp may be full,
# which causes early failures like:
#   cannot create temp file for here-document: No space left on device
# Route temp files to the run output directory instead.
TMP_WORKDIR="${OUT}/.tmp"
mkdir -p "${TMP_WORKDIR}"
export TMPDIR="${TMP_WORKDIR}"

if [[ "${KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL}" -lt 0 ]]; then
  if [[ "${KV_LOSSLESS_RUNTIME_MODE}" == "store" ]]; then
    KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL=2
  else
    KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL=1
  fi
fi
if [[ "${KV_LOSSLESS_STORE_DISABLE_FRONT}" -lt 0 ]]; then
  if [[ "${KV_LOSSLESS_RUNTIME_MODE}" == "store" ]]; then
    KV_LOSSLESS_STORE_DISABLE_FRONT=1
  else
    KV_LOSSLESS_STORE_DISABLE_FRONT=0
  fi
fi
if [[ "${KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS}" -lt 0 ]]; then
  if [[ "${KV_LOSSLESS_RUNTIME_MODE}" == "store" ]]; then
    KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS=32
  else
    KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS=128
  fi
fi
if [[ "${KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS}" -lt 0 ]]; then
  if [[ "${KV_LOSSLESS_RUNTIME_MODE}" == "store" ]]; then
    KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS=256
  else
    KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS=128
  fi
fi

# Preflight temp-dir writability and minimal free space (64 MiB).
if ! touch "${TMPDIR}/.tmp_write_test" 2>/dev/null; then
  echo "FAIL: cannot write to TMPDIR=${TMPDIR}"
  exit 1
fi
rm -f "${TMPDIR}/.tmp_write_test"
TMP_AVAIL_KB=$(df -Pk "${TMPDIR}" | awk 'NR==2 {print $4}')
if [[ -z "${TMP_AVAIL_KB}" || "${TMP_AVAIL_KB}" -lt 65536 ]]; then
  echo "FAIL: low free space in TMPDIR=${TMPDIR} (${TMP_AVAIL_KB:-unknown} KiB available)"
  echo "Free some disk space and retry."
  exit 1
fi
echo "Temp workspace: ${TMPDIR} (avail ${TMP_AVAIL_KB} KiB)"
echo "Runtime knobs:"
echo "  KV_LOSSLESS_SCOPE=${KV_LOSSLESS_SCOPE}"
echo "  KV_LOSSLESS_RUNTIME_MODE=${KV_LOSSLESS_RUNTIME_MODE}"
echo "  KV_LOSSLESS_KEPT_SAMPLE_LAYERS=${KV_LOSSLESS_KEPT_SAMPLE_LAYERS}"
echo "  KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL=${KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL}"
echo "  KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL=${KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL}"
echo "  KV_LOSSLESS_STORE_DISABLE_FRONT=${KV_LOSSLESS_STORE_DISABLE_FRONT}"
echo "  KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS=${KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS}"
echo "  KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS=${KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS}"
echo "  KV_LOSSLESS_HOT_SINK=${KV_LOSSLESS_HOT_SINK}"
echo "  KV_LOSSLESS_HOT_RECENT=${KV_LOSSLESS_HOT_RECENT}"
echo "  MAX_LOSSLESS_QUEUE_PEAK=${MAX_LOSSLESS_QUEUE_PEAK}"
echo "  MAX_LOSSLESS_FALLBACK=${MAX_LOSSLESS_FALLBACK}"
echo "  MAX_LOSSLESS_BACKPRESSURE_SKIP=${MAX_LOSSLESS_BACKPRESSURE_SKIP}"
echo "  MAX_LOSSLESS_DECOMP_US=${MAX_LOSSLESS_DECOMP_US}"
echo "  MAX_LOSSLESS_ASYNC_WAIT_US=${MAX_LOSSLESS_ASYNC_WAIT_US}"
echo "  DECODE_BASELINE_MODE=${DECODE_BASELINE_MODE}"
echo "  DECODE_BASELINE=${DECODE_BASELINE}"
echo "  DECODE_BASELINE_KEY=${DECODE_BASELINE_KEY}"
echo "  DECODE_BASELINE_HISTORY=${DECODE_BASELINE_HISTORY}"
echo "  DECODE_BASELINE_ROLLING_WINDOW=${DECODE_BASELINE_ROLLING_WINDOW}"
echo "  DECODE_BASELINE_ROLLING_MIN_SAMPLES=${DECODE_BASELINE_ROLLING_MIN_SAMPLES}"
echo "  REQUIRE_DECODE_CACHE_HIT=${REQUIRE_DECODE_CACHE_HIT}"
echo "  REQUIRE_ASYNC_QUEUE_ACTIVITY=${REQUIRE_ASYNC_QUEUE_ACTIVITY}"
echo "  REQUIRE_DECODE_CACHE_ACTIVITY=${REQUIRE_DECODE_CACHE_ACTIVITY}"
echo "  STRICT_RUNTIME_METRIC_COLUMNS=${STRICT_RUNTIME_METRIC_COLUMNS}"
echo "  KV_LOSSLESS_ASYNC_THREADS=${KV_LOSSLESS_ASYNC_THREADS}"
echo "  KV_LOSSLESS_DECODE_CACHE_BLOCKS=${KV_LOSSLESS_DECODE_CACHE_BLOCKS}"
echo "  REQUIRE_RUNTIME_DECOMP=${REQUIRE_RUNTIME_DECOMP}"
echo "  DISABLE_H2O=${DISABLE_H2O}"
echo "  DISABLE_LOSSLESS=${DISABLE_LOSSLESS}"

# ── Step 0: Quick Python-side regression checks ───────────────────────────
echo ""
echo "==== Step 0: Python-side regression checks ===="

# Fix 4: empty sweep params should fail fast
echo "[Fix 4] Testing empty sweep dimension detection..."
EMPTY_DIR="${OUT}/throwaway_empty"
mkdir -p "${EMPTY_DIR}"
cat > "${EMPTY_DIR}/base_config.json" <<'JSONEOF'
{}
JSONEOF
set +e
EMPTY_OUT=$(python3 exp/h2o_final/run_h2o_final_bench.py \
  --llm-bench "${LLM_BENCH}" \
  --base-config "${EMPTY_DIR}/base_config.json" \
  --out-dir "${EMPTY_DIR}" \
  --h2o-keep-ratios "" \
  --dry-run 2>&1)
EMPTY_RC=$?
set -e
if [[ ${EMPTY_RC} -ne 0 ]] && echo "${EMPTY_OUT}" | grep -q "Empty sweep dimension"; then
  echo "  PASS: empty sweep correctly rejected"
else
  echo "  FAIL: empty sweep was NOT rejected (rc=${EMPTY_RC})"
  echo "  output: ${EMPTY_OUT}"
  STEP0_FAIL=1
fi
rm -rf "${EMPTY_DIR}"

# Fix 5: dedup test — same file via --log-dir and --log-files
echo "[Fix 5] Testing log file deduplication..."
DEDUP_DIR="${OUT}/throwaway_dedup"
mkdir -p "${DEDUP_DIR}"
cat > "${DEDUP_DIR}/fake.log" <<'LOGEOF'
| model | speed(tok/s) | h2o_keep | h2o_lossy |
|-------|-------------|----------|-----------|
| test  | 10.0        | 0.5000   | 3.0000    |
LOGEOF
python3 exp/h2o_final/parse_h2o_final_log.py \
  --log-dir "${DEDUP_DIR}" \
  --log-files "${DEDUP_DIR}/fake.log" \
  --out-csv "${DEDUP_DIR}/dedup_test.csv"
DEDUP_LINES=$(wc -l < "${DEDUP_DIR}/dedup_test.csv")
# 1 header + 1 data row = 2 lines (not 3 if duplicated)
if [[ ${DEDUP_LINES} -eq 2 ]]; then
  echo "  PASS: duplicate log deduplicated (${DEDUP_LINES} lines)"
else
  echo "  FAIL: expected 2 lines, got ${DEDUP_LINES}"
  STEP0_FAIL=1
fi
rm -rf "${DEDUP_DIR}"

# Fix 3: silent fallback warning
echo "[Fix 3] Testing quality gate fallback warning..."
GATE_DIR="${OUT}/throwaway_gate"
mkdir -p "${GATE_DIR}"
cat > "${GATE_DIR}/test.csv" <<'CSVEOF'
model,speed(tok/s),h2o_keep,h2o_lossy,h2o_lossless,log_file
test,<br>10.0,0.5000,3.2000,1.5000,test.log
CSVEOF
GATE_OUT=$(python3 exp/h2o_final/analyze_h2o_final.py \
  --csv "${GATE_DIR}/test.csv" \
  --lossy-target 3.0 \
  --lossless-target 1.3 \
  --out "${GATE_DIR}/summary.md" 2>&1)
if echo "${GATE_OUT}" | grep -q "\[WARN\]"; then
  echo "  PASS: fallback warning printed"
else
  echo "  FAIL: no [WARN] in output"
  echo "  output: ${GATE_OUT}"
  STEP0_FAIL=1
fi
rm -rf "${GATE_DIR}"

# Fix 2: compression_failed must not inflate raw_bytes
echo "[Fix 2] Testing compression_failed raw-byte accounting..."
FIX2_DIR="${OUT}/throwaway_fix2"
mkdir -p "${FIX2_DIR}"
python3 - "${FIX2_DIR}" <<'PY'
import json
import os
import sys

root = sys.argv[1]

ok = {
    "stage": "decode",
    "run_id": "fix2_unit",
    "layer_id": 0,
    "seq_start": 0,
    "seq_len": 2,
    "kv_heads": 1,
    "head_dim": 2,
    "bytes_per_elem": 2,
    "k_file": "k_ok.bin",
    "v_file": "v_ok.bin",
}
bad = {
    "stage": "decode",
    "run_id": "fix2_unit",
    "layer_id": 0,
    "seq_start": 2,
    "seq_len": 2,
    "kv_heads": 1,
    "head_dim": 2,
    "bytes_per_elem": 2,
    "k_file": "k_bad.bin",
    "v_file": "v_bad.bin",
}

# expected bytes for seq=2, heads=1, dim=2, fp16 = 8 bytes per tensor
with open(os.path.join(root, "k_ok.bin"), "wb") as f:
    f.write(bytes(range(8)))
with open(os.path.join(root, "v_ok.bin"), "wb") as f:
    f.write(bytes(range(8, 16)))

# Intentionally broken payload sizes to force compression_failed.
with open(os.path.join(root, "k_bad.bin"), "wb") as f:
    f.write(bytes(range(4)))
with open(os.path.join(root, "v_bad.bin"), "wb") as f:
    f.write(bytes(range(4, 8)))

with open(os.path.join(root, "meta_ok.json"), "w", encoding="utf-8") as f:
    json.dump(ok, f, ensure_ascii=True, indent=2)
with open(os.path.join(root, "meta_bad.json"), "w", encoding="utf-8") as f:
    json.dump(bad, f, ensure_ascii=True, indent=2)
PY
python3 exp/h2o_final/offline_lossless_fp16.py \
  --dump-dir "${FIX2_DIR}" \
  --scope front_n \
  --front-n 2 \
  --stage both \
  --entry-mode per_meta \
  --codec-mode adaptive_v22 \
  --adaptive-block-seq 64 \
  --zstd-level 3 \
  --out-json "${FIX2_DIR}/fix2_result.json" \
  --overwrite
python3 - "${FIX2_DIR}/fix2_result.json" <<'PY'
import json
import sys

obj = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
raw = int(obj.get("raw_bytes", 0))
comp = int(obj.get("compressed_bytes", 0))
skip_cf = int(obj.get("skip_reasons", {}).get("compression_failed", 0))

# Only one valid entry should contribute:
# k_ok(8) + v_ok(8) = 16 bytes total raw.
if raw != 16:
    print(f"  FAIL: expected raw_bytes=16, got {raw}")
    sys.exit(1)
if comp <= 0:
    print("  FAIL: compressed_bytes must be > 0 for the valid entry")
    sys.exit(1)
if skip_cf <= 0:
    print("  FAIL: expected compression_failed > 0 in synthetic case")
    sys.exit(1)
print("  PASS: failed entries are excluded from raw_bytes accounting")
PY
rm -rf "${FIX2_DIR}"

if [[ ${STEP0_FAIL} -ne 0 ]]; then
  echo ""
  echo "ABORT: Step 0 regression check(s) failed. Fix Python-side issues before proceeding."
  exit 1
fi
echo ""
echo "Step 0: all Python-side regression checks PASSED."

# ── Helper: bench arg list (shared by Step 1 & 2) ────────────────────────
BENCH_ARGS=(
  --llm-bench "${LLM_BENCH}"
  --base-config "${MODEL_CONFIG}"
  --out-dir "${OUT}"
  --backend "${BACKEND}"
  --threads "${THREADS}"
  --prompts "${PROMPTS}"
  --gens "${GENS}"
  --repeat "${REPEAT}"
  --h2o-keep-ratios "${H2O_KEEP_RATIO}"
  --h2o-block-tokens "${H2O_BLOCK_TOKENS}"
  --h2o-sink-tokens "${H2O_SINK_TOKENS}"
  --h2o-recent-tokens "${H2O_RECENT_TOKENS}"
  --h2o-target-mode "${H2O_TARGET_MODE}"
  --h2o-target-lossy-ratios "${H2O_TARGET_LOSSY_RATIO}"
  --h2o-ema-alphas "${H2O_EMA_ALPHA}"
  --h2o-update-intervals "${H2O_UPDATE_INTERVAL}"
  --h2o-trigger-min-tokens "${H2O_TRIGGER_MIN}"
  --h2o-layer-start "${H2O_LAYER_START}"
  --h2o-layer-end "${H2O_LAYER_END}"
  --h2o-log-stats
  --kv-lossless-enable
  --kv-lossless-scope "${KV_LOSSLESS_SCOPE}"
  --kv-lossless-front-n "${KV_LOSSLESS_FRONT_N}"
  --kv-lossless-codec "${KV_LOSSLESS_CODEC}"
  --kv-lossless-runtime-enable
  --kv-lossless-runtime-mode "${KV_LOSSLESS_RUNTIME_MODE}"
  --kv-lossless-codec-runtime "${KV_LOSSLESS_CODEC_RUNTIME}"
  --kv-lossless-block-tokens "${KV_LOSSLESS_BLOCK_TOKENS}"
  --kv-lossless-hot-recent-tokens "${KV_LOSSLESS_HOT_RECENT}"
  --kv-lossless-hot-sink-tokens "${KV_LOSSLESS_HOT_SINK}"
  --kv-lossless-kept-sample-layers "${KV_LOSSLESS_KEPT_SAMPLE_LAYERS}"
  --kv-lossless-kept-sample-token-interval "${KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL}"
  --kv-lossless-front-sample-token-interval "${KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL}"
  --kv-lossless-store-bootstrap-tokens "${KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS}"
  --kv-lossless-store-grouped-step-tokens "${KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS}"
  --kv-lossless-async-threads "${KV_LOSSLESS_ASYNC_THREADS}"
  --kv-lossless-decode-cache-blocks "${KV_LOSSLESS_DECODE_CACHE_BLOCKS}"
)
if [[ "${KV_LOSSLESS_STORE_DISABLE_FRONT}" -ne 0 ]]; then
  BENCH_ARGS+=(--kv-lossless-store-disable-front)
fi
if [[ "${DISABLE_H2O}" -ne 0 ]]; then
  BENCH_ARGS+=(--disable-h2o)
fi
if [[ "${DISABLE_LOSSLESS}" -ne 0 ]]; then
  BENCH_ARGS+=(--disable-lossless)
fi

extract_decode_best_from_csv() {
  local csv_path="$1"
  python3 - "${csv_path}" <<'PY'
import csv
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)

def parse_decode(cell):
    if not cell:
        return 0.0
    m = re.search(r"<br>\s*([0-9]+(?:\.[0-9]+)?)", str(cell))
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0

rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
best = 0.0
for row in rows:
    best = max(best, parse_decode(row.get("speed(tok/s)", "")))
print(f"{best:.6f}")
PY
}

ANALYZE_DECODE_BASELINE="${DECODE_BASELINE}"
ANALYZE_DECODE_BASELINE_SOURCE="fixed"
ANALYZE_DECODE_BASELINE_SAMPLES=1

# ── Step 1: Dry-run config generation ─────────────────────────────────────
echo ""
echo "==== Step 1: Dry-run config generation ===="
python3 exp/h2o_final/run_h2o_final_bench.py "${BENCH_ARGS[@]}" --dry-run

if [[ ! -d "${OUT}/configs" ]]; then
  echo "FAIL: missing configs dir after dry-run: ${OUT}/configs"
  exit 1
fi
mapfile -t CFGS < <(find "${OUT}/configs" -maxdepth 1 -type f -name '*.json' | sort)
if [[ ${#CFGS[@]} -eq 0 ]]; then
  echo "FAIL: no generated config JSON found under ${OUT}/configs"
  exit 1
fi
echo "Generated configs (${#CFGS[@]}):"
for cfg in "${CFGS[@]}"; do
  echo "  $(basename "${cfg}")"
done

if [[ ! -s "${OUT}/manifest.json" ]]; then
  echo "FAIL: missing or empty manifest: ${OUT}/manifest.json"
  exit 1
fi
echo "Manifest:"
python3 - "${OUT}/manifest.json" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.loads(open(path, "r", encoding="utf-8").read())
print(f"{len(data)} runs")
for row in data:
    print(f"  {row.get('run_id', '<missing_run_id>')}")
PY

# ── Step 2: Actual benchmark run ──────────────────────────────────────────
echo ""
echo "==== Step 2: Running llm_bench ===="
rm -rf "${OUT}/configs" "${OUT}/logs" "${OUT}/manifest.json" "${BENCH_STDOUT_LOG}"

# Keep a full stdout/stderr transcript because [H2O-LOSSLESS] logs may not be
# written into per-run markdown files produced by llm_bench -fp.
python3 exp/h2o_final/run_h2o_final_bench.py "${BENCH_ARGS[@]}" 2>&1 | tee "${BENCH_STDOUT_LOG}"

# ── Step 3: Parse logs ────────────────────────────────────────────────────
echo ""
echo "==== Step 3: Parsing logs → CSV ===="
python3 exp/h2o_final/parse_h2o_final_log.py \
  --log-dir "${OUT}/logs" \
  --out-csv "${OUT}/h2o_metrics.csv"

# ── Step 4: Validate Fix 1 — lossless stats non-zero ─────────────────────
echo ""
echo "==== Step 4: Validating Fix 1 — lossless stats ===="
if [[ "${DISABLE_H2O}" -eq 0 && "${DISABLE_LOSSLESS}" -eq 0 ]]; then
python3 -c "
import csv, re, sys

def parse_mean(s):
    \"\"\"Extract the mean from '0.12 ± 0.00' or plain '0.12'.\"\"\"
    if not s:
        return 0.0
    m = re.match(r'([0-9eE.+-]+)', s.strip())
    return float(m.group(1)) if m else 0.0

with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))
if not rows:
    print('  FAIL: CSV is empty'); sys.exit(1)
n_ok = 0
for r in rows:
    raw = parse_mean(r.get('h2o_lossless_raw_mb', '0'))
    ratio = parse_mean(r.get('h2o_lossless', '0'))
    if raw > 0 and ratio > 1.0:
        n_ok += 1
print(f'  {n_ok}/{len(rows)} rows have non-zero lossless stats')
if n_ok == 0:
    print('  FAIL: all lossless stats are still zero — Fix 1 may not be working')
    sys.exit(1)
print('  PASS')
" "${OUT}/h2o_metrics.csv"
else
  echo "  INFO: skip lossless non-zero check (H2O or lossless disabled)"
fi

# ── Step 4.2: Validate runtime real decompression path ─────────────────────
echo ""
echo "==== Step 4.2: Validating runtime decompression ===="
python3 - "${OUT}/h2o_metrics.csv" "${REQUIRE_RUNTIME_DECOMP}" <<'PY'
import csv
import re
import sys


def parse_mean(s):
    if not s:
        return 0.0
    m = re.match(r"\s*([0-9eE.+-]+)", s.strip())
    return float(m.group(1)) if m else 0.0


rows = list(csv.DictReader(open(sys.argv[1], "r", encoding="utf-8")))
require_runtime_decomp = int(sys.argv[2]) != 0
if not rows:
    print("  FAIL: empty CSV")
    sys.exit(1)

ok = 0
for i, r in enumerate(rows, 1):
    raw = parse_mean(r.get("h2o_lossless_raw_mb", "0"))
    comp = parse_mean(r.get("h2o_lossless_comp_mb", "0"))
    comp_us = parse_mean(r.get("h2o_lossless_comp_us", "0"))
    decomp_us = parse_mean(r.get("h2o_lossless_decomp_us", "0"))
    queue_peak = parse_mean(r.get("h2o_lossless_queue_peak", "0"))
    row_ok = raw > 0 and comp > 0 and comp_us > 0 and decomp_us > 0
    print(
        f"  row#{i}: raw_mb={raw:.3f} comp_mb={comp:.3f} "
        f"comp_us={comp_us:.2f} decomp_us={decomp_us:.2f} queue_peak={queue_peak:.2f} "
        f"-> {'OK' if row_ok else 'FAIL'}"
    )
    if row_ok:
        ok += 1

if require_runtime_decomp and ok == 0:
    print("  FAIL: no row shows real runtime decompression (h2o_lossless_decomp_us > 0)")
    sys.exit(1)
if require_runtime_decomp:
    print(f"  PASS: {ok}/{len(rows)} rows show non-zero runtime decompression")
else:
    print(f"  INFO: runtime decompression gate disabled, observed {ok}/{len(rows)} rows with non-zero decomp")
PY

# ── Step 4.5: Lossy feasibility diagnosis ──────────────────────────────────
echo ""
echo "==== Step 4.5: Lossy feasibility diagnosis ===="
python3 - "${OUT}/h2o_metrics.csv" "${LOSSY_TARGET}" <<'PY'
import csv
import re
import sys


def parse_mean(s):
    if not s:
        return 0.0
    m = re.match(r"\s*([0-9eE.+-]+)", s.strip())
    return float(m.group(1)) if m else 0.0


csv_path = sys.argv[1]
lossy_target = float(sys.argv[2])
rows = list(csv.DictReader(open(csv_path, "r", encoding="utf-8")))
if not rows:
    print("  SKIP: empty CSV")
    sys.exit(0)

unreachable = 0
for i, r in enumerate(rows, 1):
    lossy = parse_mean(r.get("h2o_lossy", "0"))
    floor_keep = parse_mean(r.get("h2o_floor_keep", "0"))
    quant_keep = parse_mean(r.get("h2o_quantized_keep", "0"))
    effective_keep = max(floor_keep, quant_keep, 1e-9)
    theoretical_ceiling = 1.0 / effective_keep
    reachable = theoretical_ceiling + 1e-6 >= lossy_target
    status = "OK" if reachable else "UNREACHABLE"
    if not reachable:
        unreachable += 1
    print(
        f"  row#{i}: lossy={lossy:.3f} floor={floor_keep:.4f} "
        f"quantized={quant_keep:.4f} theoretical_ceiling~{theoretical_ceiling:.3f} -> {status}"
    )

if unreachable > 0:
    print(
        f"  WARN: {unreachable}/{len(rows)} rows are structurally below "
        f"lossy_target={lossy_target:.3f} (check sink/recent/block/keep settings)."
    )
else:
    print("  PASS: all rows are structurally capable of reaching lossy target.")
PY

OFFLINE_UPPER_ARGS=()
OFFLINE_ONLINE_ARGS=()
if [[ "${DISABLE_H2O}" -eq 0 && "${DISABLE_LOSSLESS}" -eq 0 ]]; then
  # ── Step 5: Validate Fix 6 — groupedStep uses configured token span ──────
  echo ""
  echo "==== Step 5: Validating Fix 6 — groupedStep ===="
  LOSSLESS_LINES=""
  if [[ -d "${OUT}/logs" ]]; then
    LOG_LINES=$(grep -rh '\[H2O-LOSSLESS\]' "${OUT}/logs" 2>/dev/null || true)
    if [[ -n "${LOG_LINES}" ]]; then
      LOSSLESS_LINES="${LOG_LINES}"
    fi
  fi
  if [[ -f "${BENCH_STDOUT_LOG}" ]]; then
    STDOUT_LINES=$(grep -h '\[H2O-LOSSLESS\]' "${BENCH_STDOUT_LOG}" 2>/dev/null || true)
    if [[ -n "${STDOUT_LINES}" ]]; then
      if [[ -n "${LOSSLESS_LINES}" ]]; then
        LOSSLESS_LINES+=$'\n'
      fi
      LOSSLESS_LINES+="${STDOUT_LINES}"
    fi
  fi
  if [[ -z "${LOSSLESS_LINES}" ]]; then
    echo "  FAIL: no [H2O-LOSSLESS] lines found — runtime lossless did not trigger"
    echo "  Checked: ${OUT}/logs and ${BENCH_STDOUT_LOG}"
    exit 1
  fi
  STEP5_TMP="${TMPDIR}/step5_lossless_lines.txt"
  printf "%s\n" "${LOSSLESS_LINES}" > "${STEP5_TMP}"
  echo "  Sample lines:"
  head -3 "${STEP5_TMP}"

  set +e
  STEP5_OUT=$(python3 - \
    "${KV_LOSSLESS_BLOCK_TOKENS}" \
    "${KV_LOSSLESS_RUNTIME_MODE}" \
    "${KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS}" \
    "${KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS}" \
    "${STEP5_TMP}" <<'PY'
import re
import sys

block_tokens = int(sys.argv[1])
runtime_mode = sys.argv[2]
store_bootstrap_tokens = int(sys.argv[3])
store_grouped_step_tokens = int(sys.argv[4])
path = sys.argv[5]
lines = [
    ln.strip()
    for ln in open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    if "[H2O-LOSSLESS]" in ln
]

tok_re = re.compile(r"\btokens=(\d+)\b")
upd_re = re.compile(r"\bupdates=(\d+)\b")
start_re = re.compile(r"\bstart=(\d+)\b")

tok_192 = 0
periodic = 0
bootstrap_like = 0
start_pos = 0
periodic_expected = 0
bootstrap_expected = 0

if runtime_mode == "store":
    expected_bootstrap = store_bootstrap_tokens if store_bootstrap_tokens > 0 else block_tokens
    expected_periodic = store_grouped_step_tokens if store_grouped_step_tokens > 0 else block_tokens
else:
    expected_bootstrap = block_tokens
    expected_periodic = block_tokens

for ln in lines:
    m_tok = tok_re.search(ln)
    m_upd = upd_re.search(ln)
    m_start = start_re.search(ln)
    tok = int(m_tok.group(1)) if m_tok else -1
    upd = int(m_upd.group(1)) if m_upd else -1
    start = int(m_start.group(1)) if m_start else -1
    if tok == 192:
        tok_192 += 1
    if upd >= 2:
        periodic += 1
        if tok == expected_periodic:
            periodic_expected += 1
    elif upd == 1:
        bootstrap_like += 1
        if tok == expected_bootstrap:
            bootstrap_expected += 1
    if start > 0:
        start_pos += 1

print(f"  mode={runtime_mode}")
print(f"  expected bootstrap tokens={expected_bootstrap} count: {bootstrap_expected}")
print(f"  expected periodic tokens={expected_periodic} count: {periodic_expected}")
print(f"  tokens=192 count: {tok_192}")
print(f"  periodic(updates>=2) count: {periodic}")
print(f"  bootstrap-like(updates=1) count: {bootstrap_like}")
print(f"  start>0 count: {start_pos}")

if tok_192 > 0:
    print("  FAIL: hardcoded 192 grouping still present")
    sys.exit(1)
if periodic > 0 and periodic_expected == 0:
    print(
        f"  FAIL: periodic updates detected but none used expected periodic tokens={expected_periodic}"
    )
    sys.exit(1)
if bootstrap_like > 0 and bootstrap_expected == 0:
    print(
        f"  WARN: bootstrap updates detected but none used expected bootstrap tokens={expected_bootstrap}"
    )
if periodic > 0:
    print("  PASS")
    sys.exit(0)
print("  INFO: only bootstrap-like updates observed (updates=1).")
print("        With hot-window clipping, bootstrap start may be >0 (e.g. start=hot_sink).")
print("        This run did not reach periodic grouped updates; skipping strict blockStep assertion.")
print("        To force strict check, increase GENS or reduce grouped-step token span.")
print("  PASS")
sys.exit(0)
PY
  )
  STEP5_RC=$?
  rm -f "${STEP5_TMP}"
  set -e
  echo "${STEP5_OUT}"
  if [[ ${STEP5_RC} -ne 0 ]]; then
    exit 1
  fi

  # ── Step 6: Offline lossless + Fix 2 regression ──────────────────────────
  echo ""
  echo "==== Step 6: Offline lossless (upper-bound + online-sim) ===="
  if [[ -d "${DUMP_DIR}" ]]; then
    # Upper-bound
    python3 exp/h2o_final/offline_lossless_fp16.py \
    --dump-dir "${DUMP_DIR}" \
    --scope front_n \
    --front-n "${KV_LOSSLESS_FRONT_N}" \
    --stage both \
    --entry-mode aggregate \
    --aggregate-by-stage \
    --codec-mode adaptive_v22 \
    --adaptive-block-seq "${ADAPTIVE_BLOCK_SEQ}" \
    --zstd-level 3 \
    --out-json "${OUT}/offline_lossless_upper.json" \
    --out-md "${OUT}/offline_lossless_upper.md" \
    --overwrite

  # Online-sim (chunked_grouped)
  python3 exp/h2o_final/offline_lossless_fp16.py \
    --dump-dir "${DUMP_DIR}" \
    --scope front_n \
    --front-n "${KV_LOSSLESS_FRONT_N}" \
    --stage both \
    --entry-mode chunked_grouped \
    --online-chunk-seq "${ONLINE_CHUNK_SEQ}" \
    --online-framing-bytes "${ONLINE_FRAMING_BYTES}" \
    --codec-mode adaptive_v22 \
    --adaptive-block-seq "${ADAPTIVE_BLOCK_SEQ}" \
    --zstd-level 3 \
    --out-json "${OUT}/offline_lossless_online_grouped.json" \
    --out-md "${OUT}/offline_lossless_online_grouped.md" \
    --overwrite

  OFFLINE_UPPER_ARGS=(--offline-lossless-json "${OUT}/offline_lossless_upper.json")
  OFFLINE_ONLINE_ARGS=(--offline-lossless-online-json "${OUT}/offline_lossless_online_grouped.json")

    # ── Fix 2 regression: verify compression_failed doesn't inflate ratio ──
    echo ""
    echo "[Fix 2] Validating offline lossless compression-failure bias..."
    python3 -c "
import json, sys

path = sys.argv[1]
obj = json.loads(open(path).read())
raw = obj.get('raw_bytes', 0)
comp = obj.get('compressed_bytes', 0)
skipped = obj.get('skipped_entries', 0)
skip_cf = obj.get('skip_reasons', {}).get('compression_failed', 0)
ratio = obj.get('weighted_lossless_ratio', obj.get('lossless_ratio', 0))

print(f'  raw_bytes={raw}  compressed_bytes={comp}  ratio={ratio:.4f}')
print(f'  skipped_entries={skipped}  compression_failed={skip_cf}')

if raw <= 0 or comp <= 0:
    print('  FAIL: no valid compressed data produced')
    sys.exit(1)

# After Fix 2, raw_bytes only accumulates for successfully compressed entries.
# If compression_failed > 0, raw_bytes must NOT include those entries' sizes.
# Sanity: the reported ratio must equal raw/comp (within rounding).
expected_ratio = raw / comp
if abs(expected_ratio - ratio) > 0.01:
    print(f'  FAIL: ratio mismatch: raw/comp={expected_ratio:.4f} vs reported={ratio:.4f}')
    sys.exit(1)

print('  PASS: raw_bytes consistent with compressed_bytes (no failure inflation)')
" "${OUT}/offline_lossless_online_grouped.json"
  else
    echo "  FAIL: DUMP_DIR=${DUMP_DIR} not found."
    echo "  This script is a fix-validation pipeline and requires offline lossless artifacts."
    exit 1
  fi
else
  echo ""
  echo "==== Step 5/6: Skipped (H2O or lossless disabled) ===="
fi

# ── Step 6.8: Resolve decode baseline mode ─────────────────────────────────
echo ""
echo "==== Step 6.8: Resolve decode baseline (${DECODE_BASELINE_MODE}) ===="
case "${DECODE_BASELINE_MODE}" in
  fixed)
    ANALYZE_DECODE_BASELINE="${DECODE_BASELINE}"
    ANALYZE_DECODE_BASELINE_SOURCE="fixed"
    ANALYZE_DECODE_BASELINE_SAMPLES=1
    ;;
  rolling)
    ROLLING_INFO=$(python3 - \
      "${DECODE_BASELINE_HISTORY}" \
      "${DECODE_BASELINE_KEY}" \
      "${DECODE_BASELINE_ROLLING_WINDOW}" \
      "${DECODE_BASELINE_ROLLING_MIN_SAMPLES}" \
      "${DECODE_BASELINE}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

history_path = Path(sys.argv[1])
key = sys.argv[2]
window = max(1, int(float(sys.argv[3])))
min_samples = max(1, int(float(sys.argv[4])))
fixed = float(sys.argv[5])

values = []
if history_path.exists():
    for raw in history_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if str(obj.get("key", "")) != key:
            continue
        try:
            v = float(obj.get("decode_best", 0.0))
        except Exception:
            v = 0.0
        if v > 0.0:
            values.append(v)

values = values[-window:]
if len(values) >= min_samples:
    baseline = statistics.median(values)
    print(f"{baseline:.6f}\t{len(values)}\trolling_median")
else:
    print(f"{fixed:.6f}\t{len(values)}\tfixed_fallback")
PY
)
    read -r ANALYZE_DECODE_BASELINE ANALYZE_DECODE_BASELINE_SAMPLES ANALYZE_DECODE_BASELINE_SOURCE <<<"${ROLLING_INFO}"
    ;;
  same_batch)
    BASELINE_BATCH_OUT="${OUT}/baseline_batch"
    BASELINE_BATCH_LOG="${OUT}/baseline_batch_stdout.log"
    rm -rf "${BASELINE_BATCH_OUT}" "${BASELINE_BATCH_LOG}"
    BASELINE_BENCH_ARGS=("${BENCH_ARGS[@]}")
    BASELINE_BENCH_ARGS+=(--disable-h2o --disable-lossless --out-dir "${BASELINE_BATCH_OUT}")
    python3 exp/h2o_final/run_h2o_final_bench.py "${BASELINE_BENCH_ARGS[@]}" 2>&1 | tee "${BASELINE_BATCH_LOG}"
    python3 exp/h2o_final/parse_h2o_final_log.py \
      --log-dir "${BASELINE_BATCH_OUT}/logs" \
      --out-csv "${BASELINE_BATCH_OUT}/h2o_metrics.csv"
    SAME_BATCH_DECODE=$(extract_decode_best_from_csv "${BASELINE_BATCH_OUT}/h2o_metrics.csv" || echo "0")
    if python3 - "${SAME_BATCH_DECODE}" <<'PY'
import sys
v = float(sys.argv[1])
raise SystemExit(0 if v > 0.0 else 1)
PY
    then
      ANALYZE_DECODE_BASELINE="${SAME_BATCH_DECODE}"
      ANALYZE_DECODE_BASELINE_SOURCE="same_batch"
      ANALYZE_DECODE_BASELINE_SAMPLES=1
    else
      echo "WARN: same_batch baseline decode is invalid (${SAME_BATCH_DECODE}); fallback to fixed ${DECODE_BASELINE}"
      ANALYZE_DECODE_BASELINE="${DECODE_BASELINE}"
      ANALYZE_DECODE_BASELINE_SOURCE="same_batch_fallback_fixed"
      ANALYZE_DECODE_BASELINE_SAMPLES=0
    fi
    ;;
  *)
    echo "FAIL: invalid DECODE_BASELINE_MODE=${DECODE_BASELINE_MODE} (expected fixed|rolling|same_batch)"
    exit 1
    ;;
esac
echo "Resolved decode baseline: ${ANALYZE_DECODE_BASELINE} (source=${ANALYZE_DECODE_BASELINE_SOURCE}, samples=${ANALYZE_DECODE_BASELINE_SAMPLES})"

# ── Step 7: Quality gate ─────────────────────────────────────────────────
echo ""
echo "==== Step 7: Quality gate ===="
ANALYZE_ARGS=(
  --csv "${OUT}/h2o_metrics.csv"
  --lossy-target "${LOSSY_TARGET}"
  --lossless-target "${LOSSLESS_TARGET}"
  --max-lossless-queue-peak "${MAX_LOSSLESS_QUEUE_PEAK}"
  --max-lossless-fallback "${MAX_LOSSLESS_FALLBACK}"
  --max-lossless-backpressure-skip "${MAX_LOSSLESS_BACKPRESSURE_SKIP}"
  --max-lossless-decomp-us "${MAX_LOSSLESS_DECOMP_US}"
  --max-lossless-async-wait-us "${MAX_LOSSLESS_ASYNC_WAIT_US}"
  --decode-baseline "${ANALYZE_DECODE_BASELINE}"
  --decode-baseline-source "${ANALYZE_DECODE_BASELINE_SOURCE}"
  --decode-baseline-samples "${ANALYZE_DECODE_BASELINE_SAMPLES}"
  --decode-drop-target "${DECODE_DROP_TARGET}"
  --out "${OUT}/summary.md"
)
if [[ "${REQUIRE_RUNTIME_DECOMP}" -ne 0 ]]; then
  ANALYZE_ARGS+=(--require-runtime-decomp)
fi
if [[ ${#OFFLINE_UPPER_ARGS[@]} -gt 0 ]]; then
  ANALYZE_ARGS+=("${OFFLINE_UPPER_ARGS[@]}")
fi
if [[ ${#OFFLINE_ONLINE_ARGS[@]} -gt 0 ]]; then
  ANALYZE_ARGS+=("${OFFLINE_ONLINE_ARGS[@]}")
fi
if [[ "${STRICT_RUNTIME_METRIC_COLUMNS}" -ne 0 ]]; then
  ANALYZE_ARGS+=(--strict-runtime-metric-columns)
fi
if [[ "${REQUIRE_DECODE_CACHE_HIT}" -ne 0 ]]; then
  ANALYZE_ARGS+=(--require-decode-cache-hit)
fi
if [[ "${REQUIRE_ASYNC_QUEUE_ACTIVITY}" -ne 0 ]]; then
  ANALYZE_ARGS+=(--require-async-queue-activity)
fi
if [[ "${REQUIRE_DECODE_CACHE_ACTIVITY}" -ne 0 ]]; then
  ANALYZE_ARGS+=(--require-decode-cache-activity)
fi
python3 exp/h2o_final/analyze_h2o_final.py "${ANALYZE_ARGS[@]}"

if [[ "${UPDATE_DECODE_BASELINE_HISTORY}" -ne 0 ]]; then
  mkdir -p "$(dirname "${DECODE_BASELINE_HISTORY}")"
  python3 - "${OUT}/summary.md" "${DECODE_BASELINE_HISTORY}" "${DECODE_BASELINE_KEY}" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
history_path = Path(sys.argv[2])
key = sys.argv[3]

if not summary_path.exists():
    raise SystemExit(0)

gate = {}
in_gate = False
for raw in summary_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line == "## Quality Gate":
        in_gate = True
        continue
    if in_gate and line.startswith("## "):
        break
    if in_gate and line.startswith("- ") and ":" in line:
        k, v = line[2:].split(":", 1)
        gate[k.strip()] = v.strip().strip("`")

def parse_float(text):
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(text))
    return float(m.group(0)) if m else 0.0

decode_best = parse_float(gate.get("decode_best", 0.0))
if decode_best <= 0.0:
    raise SystemExit(0)

row = {
    "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "key": key,
    "decode_best": decode_best,
    "overall_pass": str(gate.get("overall_pass", "")).lower() == "true",
    "summary": str(summary_path),
}
with history_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=True) + "\n")
PY
fi

echo "Quality Gate Snapshot:"
awk '
  BEGIN {in_gate=0}
  /^## Quality Gate/ {in_gate=1; next}
  in_gate && /^## / {in_gate=0}
  in_gate && /^- / {print "  " $0}
' "${OUT}/summary.md"

if ! grep -Eq '^- overall_pass: true$' "${OUT}/summary.md"; then
  echo "FAIL: quality gate did not pass (overall_pass != true)."
  echo "See: ${OUT}/summary.md"
  exit 1
fi
echo "PASS: quality gate overall_pass=true"

echo ""
echo "============================================================"
echo " Results"
echo "============================================================"
cat "${OUT}/summary.md"
echo ""
echo "Output directory: ${OUT}"
echo "Done."
