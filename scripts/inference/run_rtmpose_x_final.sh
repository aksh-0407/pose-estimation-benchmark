#!/usr/bin/env bash
set -euo pipefail

# Full RTMPose-x (Body8/Halpe-26) Phase-1 2D-pose inference over every delivery.
#
# RTMPose-x is the largest RTMPose body model. It emits Halpe-26 keypoints; each
# player record carries the full 26-keypoint `pose_2d_native` (incl. feet) PLUS the
# COCO-17 `pose_2d` (native[0:17]) that the rest of the pipeline consumes.
#
# Defaults below are a good starting point on a laptop-class GPU. On a new machine,
# first find the fastest det/pose/io/prefetch with:
#   python scripts/tuning/tune_rtmpose_batches.py --model-id rtmpose_x_body8 \
#          --io-workers-list 8 16 24 --prefetch-list 2 4
# then export DET_BATCH_SIZE / POSE_BATCH_SIZE / IO_WORKERS / PREFETCH_BATCHES.
#
# Runs are resumable: re-running continues from the last finished frame. Extra args
# pass through to run_phase1_rtmpose_inference.py, e.g.
#   bash scripts/inference/run_rtmpose_x_final.sh --groups bt_02 --frame-limit 100

DRIVE_ROOT="${DRIVE_ROOT:-drive}"
MODEL_ID="${MODEL_ID:-rtmpose_x_body8}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-rtmpose-x}"
RUN_DIR="${RUN_DIR:-benchmarks/runs/rtmpose-x}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-32}"
POSE_BATCH_SIZE="${POSE_BATCH_SIZE:-96}"
IO_WORKERS="${IO_WORKERS:-16}"
CV2_THREADS="${CV2_THREADS:-2}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# Keep BLAS/OMP libraries from each spawning a full thread pool per io-worker; the
# GPU carries the compute, the CPU only decodes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

"${PYTHON_BIN}" scripts/inference/run_phase1_rtmpose_inference.py \
  --drive-root "${DRIVE_ROOT}" \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --det-batch-size "${DET_BATCH_SIZE}" \
  --pose-batch-size "${POSE_BATCH_SIZE}" \
  --io-workers "${IO_WORKERS}" \
  --cv2-threads "${CV2_THREADS}" \
  --prefetch-batches "${PREFETCH_BATCHES}" \
  --run-id "${RUN_ID}" \
  --run-dir "${RUN_DIR}" \
  "$@"
