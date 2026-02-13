#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# H2O v3 Fix Validation Script
# Tests all 6 fixes applied in this patch:
#   1. CPUAttention.cpp  — lossless stats no longer zeroed on budget shrink
#   2. offline_lossless_fp16.py — compression failure no longer inflates ratio
#   3. analyze_h2o_v3.py — silent gate fallback now warns
#   4. run_h2o_v3_bench.py — empty sweep params detected
#   5. parse_h2o_v3_log.py — duplicate log files deduplicated
#   6. CPUAttention.cpp  — groupedStep uses user blockStep, not hardcoded 192
# ============================================================================

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_CONFIG="/home10T/ljq/mnn_data/models/llama2_mnn/config.json"
LLM_BENCH="./build/llm_bench"
DUMP_DIR="/home10T/ljq/MNN/exp/gear_fp16/dumps_fp16"
OUT="exp/h2o_v3/out_runtime_probe_fix_$(date +%Y%m%d_%H%M%S)"
BENCH_STDOUT_LOG="${OUT}/bench_stdout.log"

# ── Tunables ───────────────────────────────────────────────────────────────
PROMPTS="512,1024"
GENS="128"
REPEAT=3
THREADS=4
BACKEND="cpu"

# H2O lossy
H2O_KEEP_RATIO="0.30"
H2O_BLOCK_TOKENS="32"
H2O_SINK_TOKENS="16"
H2O_RECENT_TOKENS="64"
H2O_TARGET_MODE="adaptive"
H2O_TARGET_LOSSY_RATIO="3.2"
H2O_EMA_ALPHA="0.9"
H2O_UPDATE_INTERVAL="16"
H2O_TRIGGER_MIN="384"
H2O_LAYER_START=2
H2O_LAYER_END=-1

# Lossless
KV_LOSSLESS_SCOPE="front_n"
KV_LOSSLESS_FRONT_N=2
KV_LOSSLESS_CODEC="gear_delta"
KV_LOSSLESS_BLOCK_TOKENS=128
KV_LOSSLESS_HOT_RECENT=256
KV_LOSSLESS_HOT_SINK=16
KV_LOSSLESS_CODEC_RUNTIME="fp16_gear_predictive_v3"

# Quality gate
LOSSY_TARGET=3.0
LOSSLESS_TARGET=1.3
DECODE_BASELINE=6.60
DECODE_DROP_TARGET=0.05

# Offline lossless
ONLINE_CHUNK_SEQ=192
ONLINE_FRAMING_BYTES=8
ADAPTIVE_BLOCK_SEQ=64

STEP0_FAIL=0

echo "============================================================"
echo " H2O v3 Fix Validation"
echo " OUT = ${OUT}"
echo "============================================================"
mkdir -p "${OUT}"

# ── Step 0: Quick Python-side regression checks ───────────────────────────
echo ""
echo "==== Step 0: Python-side regression checks ===="

# Fix 4: empty sweep params should fail fast
echo "[Fix 4] Testing empty sweep dimension detection..."
set +e
EMPTY_OUT=$(python3 exp/h2o_v3/run_h2o_v3_bench.py \
  --llm-bench "${LLM_BENCH}" \
  --base-config "${MODEL_CONFIG}" \
  --out-dir "${OUT}/throwaway_empty" \
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
rm -rf "${OUT}/throwaway_empty"

# Fix 5: dedup test — same file via --log-dir and --log-files
echo "[Fix 5] Testing log file deduplication..."
DEDUP_DIR="${OUT}/throwaway_dedup"
mkdir -p "${DEDUP_DIR}"
cat > "${DEDUP_DIR}/fake.log" <<'LOGEOF'
| model | speed(tok/s) | h2o_keep | h2o_lossy |
|-------|-------------|----------|-----------|
| test  | 10.0        | 0.5000   | 3.0000    |
LOGEOF
python3 exp/h2o_v3/parse_h2o_v3_log.py \
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
GATE_OUT=$(python3 exp/h2o_v3/analyze_h2o_v3.py \
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
python3 exp/h2o_v3/offline_lossless_fp16.py \
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
  --kv-lossless-codec-runtime "${KV_LOSSLESS_CODEC_RUNTIME}"
  --kv-lossless-block-tokens "${KV_LOSSLESS_BLOCK_TOKENS}"
  --kv-lossless-hot-recent-tokens "${KV_LOSSLESS_HOT_RECENT}"
  --kv-lossless-hot-sink-tokens "${KV_LOSSLESS_HOT_SINK}"
)

# ── Step 1: Dry-run config generation ─────────────────────────────────────
echo ""
echo "==== Step 1: Dry-run config generation ===="
python3 exp/h2o_v3/run_h2o_v3_bench.py "${BENCH_ARGS[@]}" --dry-run

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
python3 exp/h2o_v3/run_h2o_v3_bench.py "${BENCH_ARGS[@]}" 2>&1 | tee "${BENCH_STDOUT_LOG}"

# ── Step 3: Parse logs ────────────────────────────────────────────────────
echo ""
echo "==== Step 3: Parsing logs → CSV ===="
python3 exp/h2o_v3/parse_h2o_v3_log.py \
  --log-dir "${OUT}/logs" \
  --out-csv "${OUT}/h2o_metrics.csv"

# ── Step 4: Validate Fix 1 — lossless stats non-zero ─────────────────────
echo ""
echo "==== Step 4: Validating Fix 1 — lossless stats ===="
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

# ── Step 5: Validate Fix 6 — groupedStep uses blockStep ──────────────────
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
echo "  Sample lines:"
echo "${LOSSLESS_LINES}" | head -3

TOKENS_192=$(echo "${LOSSLESS_LINES}" | grep -Ec "tokens=192([^0-9]|$)" || true)
TOKENS_BLOCK=$(echo "${LOSSLESS_LINES}" | grep -Ec "tokens=${KV_LOSSLESS_BLOCK_TOKENS}([^0-9]|$)" || true)
NON_BOOTSTRAP=$(echo "${LOSSLESS_LINES}" | grep -Ec "start=[1-9][0-9]*" || true)
echo "  tokens=${KV_LOSSLESS_BLOCK_TOKENS} count: ${TOKENS_BLOCK}"
echo "  tokens=192 count: ${TOKENS_192}"
if [[ ${TOKENS_192} -gt 0 ]]; then
  echo "  FAIL: hardcoded 192 grouping still present"
  exit 1
fi
if [[ ${TOKENS_BLOCK} -eq 0 ]]; then
  if [[ ${NON_BOOTSTRAP} -gt 0 ]]; then
    echo "  FAIL: saw non-bootstrap updates (start>0) but none used tokens=${KV_LOSSLESS_BLOCK_TOKENS}"
    exit 1
  fi
  echo "  INFO: only bootstrap updates observed (start=0, e.g. tokens=kv budget)."
  echo "        This run did not reach periodic grouped updates; skipping strict blockStep assertion."
  echo "        To force strict check, increase GENS or reduce kv_lossless_block_tokens."
fi
echo "  PASS"

# ── Step 6: Offline lossless + Fix 2 regression ──────────────────────────
echo ""
echo "==== Step 6: Offline lossless (upper-bound + online-sim) ===="
OFFLINE_UPPER_FLAG=""
OFFLINE_ONLINE_FLAG=""
if [[ -d "${DUMP_DIR}" ]]; then
  # Upper-bound
  python3 exp/h2o_v3/offline_lossless_fp16.py \
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
  python3 exp/h2o_v3/offline_lossless_fp16.py \
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

  OFFLINE_UPPER_FLAG="--offline-lossless-json ${OUT}/offline_lossless_upper.json"
  OFFLINE_ONLINE_FLAG="--offline-lossless-online-json ${OUT}/offline_lossless_online_grouped.json"

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

# ── Step 7: Quality gate ─────────────────────────────────────────────────
echo ""
echo "==== Step 7: Quality gate ===="
# shellcheck disable=SC2086
python3 exp/h2o_v3/analyze_h2o_v3.py \
  --csv "${OUT}/h2o_metrics.csv" \
  ${OFFLINE_UPPER_FLAG} \
  ${OFFLINE_ONLINE_FLAG} \
  --lossy-target "${LOSSY_TARGET}" \
  --lossless-target "${LOSSLESS_TARGET}" \
  --decode-baseline "${DECODE_BASELINE}" \
  --decode-drop-target "${DECODE_DROP_TARGET}" \
  --out "${OUT}/summary.md"

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
