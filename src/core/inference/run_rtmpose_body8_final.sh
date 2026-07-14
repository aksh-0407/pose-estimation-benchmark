#!/usr/bin/env bash
set -euo pipefail

# Final RTMPose-L Body8 inference using the best large-sweep throughput setting:
# median 50.13 FPS, det_batch=32, pose_batch=96, failed=0.
#
# By default this also renders five overlay frames per camera/delivery:
# selected-frame row positions 1, 150, 300, 450, and 600.
#
# Extra arguments are passed through to run_phase1_rtmpose_inference.py, e.g.
#   bash scripts/run_rtmpose_body8_final.sh --groups bt_01 --frame-limit 100

DRIVE_ROOT="${DRIVE_ROOT:-drive}"
MODEL_ID="${MODEL_ID:-rtmpose_l_body8}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-rtmpose-l-body8-full-db32-pb96}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-32}"
POSE_BATCH_SIZE="${POSE_BATCH_SIZE:-96}"
IO_WORKERS="${IO_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" src/core/inference/run_phase1_rtmpose_inference.py \
  --drive-root "${DRIVE_ROOT}" \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --det-batch-size "${DET_BATCH_SIZE}" \
  --pose-batch-size "${POSE_BATCH_SIZE}" \
  --io-workers "${IO_WORKERS}" \
  --run-id "${RUN_ID}" \
  --overlay \
  --overlay-row-indices 1 150 300 450 600 \
  --overlay-limit 5 \
  "$@"
