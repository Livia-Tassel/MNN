#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# H2O v6 llm_demo Real Prompt Validation
# - Reuse existing real prompts from gear_fp16
# - Run baseline(H2O off) and candidate(H2O on) in one script
# - Compare decode TPS regression + runtime H2O evidence
# ============================================================================

MODEL_CONFIG="${MODEL_CONFIG:-/home10T/ljq/mnn_data/models/llama2_mnn/config.json}"
if [[ -z "${LLM_DEMO:-}" ]]; then
  if [[ -x "./build_h2o_v6/llm_demo" ]]; then
    LLM_DEMO="./build_h2o_v6/llm_demo"
  else
    LLM_DEMO="./build/llm_demo"
  fi
fi
PROMPT_DIR="${PROMPT_DIR:-/home10T/ljq/MNN/exp/gear_fp16/prompts}"
PROMPT_PATTERN="${PROMPT_PATTERN:-prompt_*.txt}"
OUT="${OUT:-exp/h2o_v6/out_llm_demo_v6_$(date +%Y%m%d_%H%M%S)}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
DECODE_DROP_TARGET="${DECODE_DROP_TARGET:-0.08}"
REQUIRE_RUNTIME_DECOMP="${REQUIRE_RUNTIME_DECOMP:-1}"
REQUIRE_DECODE_CACHE_HIT="${REQUIRE_DECODE_CACHE_HIT:-0}"

# Candidate knobs (aligned with runtime script defaults)
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
KV_LOSSLESS_SCOPE="${KV_LOSSLESS_SCOPE:-front_n_and_h2o_kept}"
KV_LOSSLESS_FRONT_N="${KV_LOSSLESS_FRONT_N:-2}"
KV_LOSSLESS_CODEC="${KV_LOSSLESS_CODEC:-gear_delta}"
KV_LOSSLESS_RUNTIME_MODE="${KV_LOSSLESS_RUNTIME_MODE:-full}"
KV_LOSSLESS_BLOCK_TOKENS="${KV_LOSSLESS_BLOCK_TOKENS:-128}"
KV_LOSSLESS_HOT_RECENT="${KV_LOSSLESS_HOT_RECENT:-256}"
KV_LOSSLESS_HOT_SINK="${KV_LOSSLESS_HOT_SINK:-16}"
KV_LOSSLESS_KEPT_SAMPLE_LAYERS="${KV_LOSSLESS_KEPT_SAMPLE_LAYERS:-1}"
KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL="${KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL:-2}"
KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL="${KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL:-1}"
KV_LOSSLESS_STORE_DISABLE_FRONT="${KV_LOSSLESS_STORE_DISABLE_FRONT:-0}"
KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS="${KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS:-32}"
KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS="${KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS:-256}"
KV_LOSSLESS_CODEC_RUNTIME="${KV_LOSSLESS_CODEC_RUNTIME:-fp16_gear_predictive_v3}"
KV_LOSSLESS_PREDICTORS_K="${KV_LOSSLESS_PREDICTORS_K:-raw,delta_seq,xor_seq,pair_delta}"
KV_LOSSLESS_PREDICTORS_V="${KV_LOSSLESS_PREDICTORS_V:-raw,delta_seq,xor_seq}"
KV_LOSSLESS_ASYNC_THREADS="${KV_LOSSLESS_ASYNC_THREADS:-1}"
KV_LOSSLESS_MAX_QUEUE="${KV_LOSSLESS_MAX_QUEUE:-256}"
KV_LOSSLESS_DECODE_CACHE_BLOCKS="${KV_LOSSLESS_DECODE_CACHE_BLOCKS:-64}"
KV_LOSSLESS_STRICT_ROUNDTRIP_CHECK="${KV_LOSSLESS_STRICT_ROUNDTRIP_CHECK:-0}"

BASELINE_CFG="${OUT}/configs/baseline_config.json"
CANDIDATE_CFG="${OUT}/configs/candidate_config.json"
BASELINE_OUT="${OUT}/baseline"
CANDIDATE_OUT="${OUT}/candidate"
REPORT_MD="${OUT}/llm_demo_report.md"
REPORT_JSON="${OUT}/llm_demo_report.json"

mkdir -p "${OUT}/configs"

if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "FAIL: MODEL_CONFIG not found: ${MODEL_CONFIG}"
  exit 1
fi
if [[ ! -d "${PROMPT_DIR}" ]]; then
  echo "FAIL: PROMPT_DIR not found: ${PROMPT_DIR}"
  exit 1
fi
if [[ ! -x "${LLM_DEMO}" ]]; then
  echo "FAIL: LLM_DEMO not executable: ${LLM_DEMO}"
  exit 1
fi

echo "============================================================"
echo " H2O v6 llm_demo Real Prompt Validation"
echo " OUT = ${OUT}"
echo " LLM_DEMO = ${LLM_DEMO}"
echo " PROMPT_DIR = ${PROMPT_DIR}"
echo " PROMPT_PATTERN = ${PROMPT_PATTERN}"
echo "============================================================"

python3 - "${MODEL_CONFIG}" "${BASELINE_CFG}" "${CANDIDATE_CFG}" <<'PY'
import json
import os
import sys
from pathlib import Path

base_cfg = Path(sys.argv[1])
baseline_cfg = Path(sys.argv[2])
candidate_cfg = Path(sys.argv[3])

obj = json.loads(base_cfg.read_text(encoding="utf-8"))

baseline = dict(obj)
baseline.update({
    "kv_h2o_enable": False,
    "kv_lossless_enable": False,
    "kv_lossless_runtime_enable": False,
})
baseline_cfg.write_text(json.dumps(baseline, ensure_ascii=True, indent=2), encoding="utf-8")

def geti(name, default):
    return int(os.environ.get(name, str(default)))

def getf(name, default):
    return float(os.environ.get(name, str(default)))

def gets(name, default):
    return os.environ.get(name, default)

def getb(name, default):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return v not in ("0", "false", "False", "")

candidate = dict(obj)
candidate.update({
    "kv_h2o_enable": True,
    "kv_h2o_layer_start": geti("H2O_LAYER_START", 2),
    "kv_h2o_layer_end": geti("H2O_LAYER_END", -1),
    "kv_h2o_block_tokens": geti("H2O_BLOCK_TOKENS", 32),
    "kv_h2o_sink_tokens": geti("H2O_SINK_TOKENS", 16),
    "kv_h2o_recent_tokens": geti("H2O_RECENT_TOKENS", 96),
    "kv_h2o_target_keep_ratio": getf("H2O_KEEP_RATIO", 0.30),
    "kv_h2o_target_mode": gets("H2O_TARGET_MODE", "adaptive"),
    "kv_h2o_target_lossy_ratio": getf("H2O_TARGET_LOSSY_RATIO", 3.2),
    "kv_h2o_ema_alpha": getf("H2O_EMA_ALPHA", 0.9),
    "kv_h2o_update_interval": geti("H2O_UPDATE_INTERVAL", 16),
    "kv_h2o_trigger_min_tokens": geti("H2O_TRIGGER_MIN", 384),
    "kv_h2o_log_stats": True,
    "kv_lossless_enable": True,
    "kv_lossless_scope": gets("KV_LOSSLESS_SCOPE", "front_n_and_h2o_kept"),
    "kv_lossless_front_n": geti("KV_LOSSLESS_FRONT_N", 2),
    "kv_lossless_codec": gets("KV_LOSSLESS_CODEC", "gear_delta"),
    "kv_lossless_runtime_enable": True,
    "kv_lossless_runtime_mode": gets("KV_LOSSLESS_RUNTIME_MODE", "full"),
    "kv_lossless_block_tokens": geti("KV_LOSSLESS_BLOCK_TOKENS", 128),
    "kv_lossless_hot_recent_tokens": geti("KV_LOSSLESS_HOT_RECENT", 256),
    "kv_lossless_hot_sink_tokens": geti("KV_LOSSLESS_HOT_SINK", 16),
    "kv_lossless_kept_sample_layers": geti("KV_LOSSLESS_KEPT_SAMPLE_LAYERS", 1),
    "kv_lossless_kept_sample_token_interval": geti("KV_LOSSLESS_KEPT_SAMPLE_TOKEN_INTERVAL", 2),
    "kv_lossless_front_sample_token_interval": geti("KV_LOSSLESS_FRONT_SAMPLE_TOKEN_INTERVAL", 1),
    "kv_lossless_store_disable_front": getb("KV_LOSSLESS_STORE_DISABLE_FRONT", False),
    "kv_lossless_store_bootstrap_tokens": geti("KV_LOSSLESS_STORE_BOOTSTRAP_TOKENS", 32),
    "kv_lossless_store_grouped_step_tokens": geti("KV_LOSSLESS_STORE_GROUPED_STEP_TOKENS", 256),
    "kv_lossless_codec_runtime": gets("KV_LOSSLESS_CODEC_RUNTIME", "fp16_gear_predictive_v3"),
    "kv_lossless_predictors_k": gets("KV_LOSSLESS_PREDICTORS_K", "raw,delta_seq,xor_seq,pair_delta"),
    "kv_lossless_predictors_v": gets("KV_LOSSLESS_PREDICTORS_V", "raw,delta_seq,xor_seq"),
    "kv_lossless_async_threads": geti("KV_LOSSLESS_ASYNC_THREADS", 1),
    "kv_lossless_max_queue": geti("KV_LOSSLESS_MAX_QUEUE", 256),
    "kv_lossless_decode_cache_blocks": geti("KV_LOSSLESS_DECODE_CACHE_BLOCKS", 64),
    "kv_lossless_strict_roundtrip_check": getb("KV_LOSSLESS_STRICT_ROUNDTRIP_CHECK", False),
})
candidate_cfg.write_text(json.dumps(candidate, ensure_ascii=True, indent=2), encoding="utf-8")
PY

python3 exp/h2o_v6/run_llm_demo_real_prompt.py \
  --llm-demo "${LLM_DEMO}" \
  --config "${BASELINE_CFG}" \
  --prompt-dir "${PROMPT_DIR}" \
  --prompt-pattern "${PROMPT_PATTERN}" \
  --decode-tokens "${DECODE_TOKENS}" \
  --run-tag baseline \
  --out-dir "${BASELINE_OUT}"

python3 exp/h2o_v6/run_llm_demo_real_prompt.py \
  --llm-demo "${LLM_DEMO}" \
  --config "${CANDIDATE_CFG}" \
  --prompt-dir "${PROMPT_DIR}" \
  --prompt-pattern "${PROMPT_PATTERN}" \
  --decode-tokens "${DECODE_TOKENS}" \
  --run-tag candidate \
  --out-dir "${CANDIDATE_OUT}"

python3 - \
  "${BASELINE_OUT}/baseline_summary.json" \
  "${CANDIDATE_OUT}/candidate_summary.json" \
  "${DECODE_DROP_TARGET}" \
  "${REQUIRE_RUNTIME_DECOMP}" \
  "${REQUIRE_DECODE_CACHE_HIT}" \
  "${REPORT_MD}" \
  "${REPORT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
decode_drop_target = float(sys.argv[3])
require_runtime_decomp = int(sys.argv[4]) != 0
require_decode_cache_hit = int(sys.argv[5]) != 0
report_md = Path(sys.argv[6])
report_json = Path(sys.argv[7])

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

baseline_decode = float(baseline.get("decode_tps_avg", 0.0))
candidate_decode = float(candidate.get("decode_tps_avg", 0.0))
decode_drop_ratio = 0.0
if baseline_decode > 0:
    decode_drop_ratio = (baseline_decode - candidate_decode) / baseline_decode

decode_pass = decode_drop_ratio <= decode_drop_target
runtime_decomp_best = float(candidate.get("runtime_decomp_us_max", 0.0))
runtime_decomp_pass = (runtime_decomp_best > 0.0) if require_runtime_decomp else True
decode_cache_hit_best = float(candidate.get("decode_cache_hit_max", 0.0))
decode_cache_hit_pass = (decode_cache_hit_best > 0.0) if require_decode_cache_hit else True

overall_pass = (
    bool(baseline.get("overall_pass", False))
    and bool(candidate.get("overall_pass", False))
    and decode_pass
    and runtime_decomp_pass
    and decode_cache_hit_pass
)

report = {
    "overall_pass": overall_pass,
    "baseline_summary": str(baseline_path),
    "candidate_summary": str(candidate_path),
    "baseline_decode_tps_avg": baseline_decode,
    "candidate_decode_tps_avg": candidate_decode,
    "decode_drop_ratio": decode_drop_ratio,
    "decode_drop_target": decode_drop_target,
    "decode_pass": decode_pass,
    "require_runtime_decomp": require_runtime_decomp,
    "runtime_decomp_best_us": runtime_decomp_best,
    "runtime_decomp_pass": runtime_decomp_pass,
    "require_decode_cache_hit": require_decode_cache_hit,
    "decode_cache_hit_best": decode_cache_hit_best,
    "decode_cache_hit_pass": decode_cache_hit_pass,
    "baseline_overall_pass": bool(baseline.get("overall_pass", False)),
    "candidate_overall_pass": bool(candidate.get("overall_pass", False)),
}
report_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

lines = []
lines.append("# H2O v6 llm_demo Real Prompt Report")
lines.append("")
lines.append(f"- overall_pass: {str(overall_pass).lower()}")
lines.append(f"- baseline_overall_pass: {str(report['baseline_overall_pass']).lower()}")
lines.append(f"- candidate_overall_pass: {str(report['candidate_overall_pass']).lower()}")
lines.append(f"- baseline_decode_tps_avg: {baseline_decode:.4f}")
lines.append(f"- candidate_decode_tps_avg: {candidate_decode:.4f}")
lines.append(f"- decode_drop_ratio: {decode_drop_ratio:.6f}")
lines.append(f"- decode_drop_target: {decode_drop_target:.6f}")
lines.append(f"- decode_pass: {str(decode_pass).lower()}")
lines.append(f"- require_runtime_decomp: {str(require_runtime_decomp).lower()}")
lines.append(f"- runtime_decomp_best_us: {runtime_decomp_best:.4f}")
lines.append(f"- runtime_decomp_pass: {str(runtime_decomp_pass).lower()}")
lines.append(f"- require_decode_cache_hit: {str(require_decode_cache_hit).lower()}")
lines.append(f"- decode_cache_hit_best: {decode_cache_hit_best:.4f}")
lines.append(f"- decode_cache_hit_pass: {str(decode_cache_hit_pass).lower()}")
lines.append(f"- baseline_summary: `{baseline_path}`")
lines.append(f"- candidate_summary: `{candidate_path}`")
report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {report_md}")
print(f"Wrote {report_json}")
if not overall_pass:
    raise SystemExit(1)
PY

echo ""
echo "============================================================"
echo " llm_demo Result"
echo "============================================================"
cat "${REPORT_MD}"
echo ""
echo "Output directory: ${OUT}"
