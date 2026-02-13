#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <out_dir> <dump_dir> <lossless_target> [decode_baseline] [decode_drop_target] [online_chunk_seq] [online_framing_bytes] [adaptive_block_seq] [online_entry_mode]"
  echo "Example: $0 exp/h2o_v3/out_tune_v3_20260212 /home10T/ljq/MNN/exp/gear_fp16/dumps_fp16 1.3 6.72 0.05 192 12 64 chunked_grouped"
  echo "online_entry_mode: chunked | chunked_grouped | both (default: chunked_grouped)"
  exit 1
fi

OUT_DIR="$1"
DUMP_DIR="$2"
LOSSLESS_TARGET="$3"
DECODE_BASELINE="${4:-0}"
DECODE_DROP_TARGET="${5:-0.05}"
ONLINE_CHUNK_SEQ="${6:-128}"
ONLINE_FRAMING_BYTES="${7:-16}"
ADAPTIVE_BLOCK_SEQ="${8:-64}"
ONLINE_ENTRY_MODE="${9:-chunked_grouped}"

if [[ "${ONLINE_ENTRY_MODE}" != "chunked" && "${ONLINE_ENTRY_MODE}" != "chunked_grouped" && "${ONLINE_ENTRY_MODE}" != "both" ]]; then
  echo "Invalid online_entry_mode: ${ONLINE_ENTRY_MODE}"
  echo "Expected: chunked | chunked_grouped | both"
  exit 2
fi

if [[ -d "${OUT_DIR}/logs" ]] && ls "${OUT_DIR}/logs"/*.log >/dev/null 2>&1; then
  python3 exp/h2o_v3/parse_h2o_v3_log.py \
    --log-dir "${OUT_DIR}/logs" \
    --out-csv "${OUT_DIR}/h2o_metrics.csv"
elif [[ -f "${OUT_DIR}/h2o_metrics.csv" ]]; then
  echo "Logs not found, reuse existing CSV: ${OUT_DIR}/h2o_metrics.csv"
else
  echo "Neither logs nor CSV found under ${OUT_DIR}"
  exit 3
fi

python3 exp/h2o_v3/offline_lossless_fp16.py \
  --dump-dir "${DUMP_DIR}" \
  --scope front_n \
  --front-n 2 \
  --stage both \
  --entry-mode aggregate \
  --aggregate-by-stage \
  --codec-mode adaptive_v22 \
  --adaptive-block-seq "${ADAPTIVE_BLOCK_SEQ}" \
  --zstd-level 3 \
  --out-json "${OUT_DIR}/offline_lossless_upper.json" \
  --out-md "${OUT_DIR}/offline_lossless_upper.md"

run_online_lossless() {
  local entry_mode="$1"
  local out_json="$2"
  local out_md="$3"
  python3 exp/h2o_v3/offline_lossless_fp16.py \
    --dump-dir "${DUMP_DIR}" \
    --scope front_n \
    --front-n 2 \
    --stage both \
    --entry-mode "${entry_mode}" \
    --online-chunk-seq "${ONLINE_CHUNK_SEQ}" \
    --online-framing-bytes "${ONLINE_FRAMING_BYTES}" \
    --codec-mode adaptive_v22 \
    --adaptive-block-seq "${ADAPTIVE_BLOCK_SEQ}" \
    --zstd-level 3 \
    --out-json "${out_json}" \
    --out-md "${out_md}"
}

run_summary() {
  local online_json="$1"
  local summary_md="$2"
  python3 exp/h2o_v3/analyze_h2o_v3.py \
    --csv "${OUT_DIR}/h2o_metrics.csv" \
    --offline-lossless-json "${OUT_DIR}/offline_lossless_upper.json" \
    --offline-lossless-online-json "${online_json}" \
    --lossy-target 3.0 \
    --lossless-target "${LOSSLESS_TARGET}" \
    --decode-baseline "${DECODE_BASELINE}" \
    --decode-drop-target "${DECODE_DROP_TARGET}" \
    --out "${summary_md}"
}

if [[ "${ONLINE_ENTRY_MODE}" == "chunked" ]]; then
  run_online_lossless "chunked" \
    "${OUT_DIR}/offline_lossless_online.json" \
    "${OUT_DIR}/offline_lossless_online.md"
  run_summary "${OUT_DIR}/offline_lossless_online.json" "${OUT_DIR}/summary.md"
  echo "Done: ${OUT_DIR}/summary.md (online mode: chunked)"
elif [[ "${ONLINE_ENTRY_MODE}" == "chunked_grouped" ]]; then
  run_online_lossless "chunked_grouped" \
    "${OUT_DIR}/offline_lossless_online_grouped.json" \
    "${OUT_DIR}/offline_lossless_online_grouped.md"
  run_summary "${OUT_DIR}/offline_lossless_online_grouped.json" "${OUT_DIR}/summary.md"
  echo "Done: ${OUT_DIR}/summary.md (online mode: chunked_grouped)"
else
  run_online_lossless "chunked" \
    "${OUT_DIR}/offline_lossless_online.json" \
    "${OUT_DIR}/offline_lossless_online.md"
  run_online_lossless "chunked_grouped" \
    "${OUT_DIR}/offline_lossless_online_grouped.json" \
    "${OUT_DIR}/offline_lossless_online_grouped.md"

  run_summary "${OUT_DIR}/offline_lossless_online.json" "${OUT_DIR}/summary_chunked.md"
  run_summary "${OUT_DIR}/offline_lossless_online_grouped.json" "${OUT_DIR}/summary_grouped.md"
  cp "${OUT_DIR}/summary_grouped.md" "${OUT_DIR}/summary.md"
  echo "Done:"
  echo "  - ${OUT_DIR}/summary_chunked.md"
  echo "  - ${OUT_DIR}/summary_grouped.md"
  echo "  - ${OUT_DIR}/summary.md (alias to grouped)"
fi
