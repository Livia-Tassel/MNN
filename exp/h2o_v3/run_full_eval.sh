#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <out_dir> <dump_dir> <lossless_target> [decode_baseline] [decode_drop_target]"
  echo "Example: $0 exp/h2o_v3/out_tune_v3_20260212 /home10T/ljq/MNN/exp/gear_fp16/dumps_fp16 1.3 6.72 0.05"
  exit 1
fi

OUT_DIR="$1"
DUMP_DIR="$2"
LOSSLESS_TARGET="$3"
DECODE_BASELINE="${4:-0}"
DECODE_DROP_TARGET="${5:-0.05}"

python3 exp/h2o_v3/parse_h2o_v3_log.py \
  --log-dir "${OUT_DIR}/logs" \
  --out-csv "${OUT_DIR}/h2o_metrics.csv"

python3 exp/h2o_v3/offline_lossless_fp16.py \
  --dump-dir "${DUMP_DIR}" \
  --scope front_n \
  --front-n 2 \
  --stage both \
  --entry-mode aggregate \
  --aggregate-by-stage \
  --codec-mode adaptive_v22 \
  --adaptive-block-seq 64 \
  --zstd-level 3 \
  --out-json "${OUT_DIR}/offline_lossless_upper.json" \
  --out-md "${OUT_DIR}/offline_lossless_upper.md"

python3 exp/h2o_v3/offline_lossless_fp16.py \
  --dump-dir "${DUMP_DIR}" \
  --scope front_n \
  --front-n 2 \
  --stage both \
  --entry-mode chunked \
  --online-chunk-seq 128 \
  --online-framing-bytes 16 \
  --codec-mode adaptive_v22 \
  --adaptive-block-seq 64 \
  --zstd-level 3 \
  --out-json "${OUT_DIR}/offline_lossless_online.json" \
  --out-md "${OUT_DIR}/offline_lossless_online.md"

python3 exp/h2o_v3/analyze_h2o_v3.py \
  --csv "${OUT_DIR}/h2o_metrics.csv" \
  --offline-lossless-json "${OUT_DIR}/offline_lossless_upper.json" \
  --offline-lossless-online-json "${OUT_DIR}/offline_lossless_online.json" \
  --lossy-target 3.0 \
  --lossless-target "${LOSSLESS_TARGET}" \
  --decode-baseline "${DECODE_BASELINE}" \
  --decode-drop-target "${DECODE_DROP_TARGET}" \
  --out "${OUT_DIR}/summary.md"

echo "Done: ${OUT_DIR}/summary.md"
