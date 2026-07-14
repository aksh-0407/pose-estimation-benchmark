#!/usr/bin/env bash
set -euo pipefail

# RTMPose-x (Body8/Halpe-26) Phase-1 2D-pose over the L40S capture machine's data.
#
# Wraps src/core/inference/run_phase1_l40s.py: native bt1/bt2/bt3 layout under
# /home/ubuntu/pose_data, output to /home/ubuntu/pose-rtm-x/. RTMPose-x emits 26
# Halpe keypoints; each player record carries the full `pose_2d_native` (26, incl.
# feet) plus COCO-17 `pose_2d`. Resumable: re-run to continue from the last frame.
#
# Tune batch sizes for the L40S first (in-process, writes only best.json):
#   python src/core/inference/run_phase1_l40s.py --model-id rtmpose_x_body8 \
#          --output-dir /home/ubuntu/pose-rtm-x --sweep --grid
# then export DET_BATCH_SIZE / POSE_BATCH_SIZE / IO_WORKERS from best.json.
#
# Extra args pass through, e.g.:
#   bash src/core/inference/run_rtmpose_x_l40s.sh --groups bt1 --frame-limit 100

POSE_DATA="${POSE_DATA:-/home/ubuntu/pose_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/pose-rtm-x}"
MODEL_ID="${MODEL_ID:-rtmpose_x_body8}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-rtmpose-x-l40s}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-24}"
POSE_BATCH_SIZE="${POSE_BATCH_SIZE:-256}"
IO_WORKERS="${IO_WORKERS:-16}"
CV2_THREADS="${CV2_THREADS:-2}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# Keep BLAS/OMP from spawning a full pool per io-worker; the GPU does the compute.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

"${PYTHON_BIN}" src/core/inference/run_phase1_l40s.py \
  --pose-data "${POSE_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --run-id "${RUN_ID}" \
  --det-batch-size "${DET_BATCH_SIZE}" \
  --pose-batch-size "${POSE_BATCH_SIZE}" \
  --io-workers "${IO_WORKERS}" \
  --cv2-threads "${CV2_THREADS}" \
  --prefetch-batches "${PREFETCH_BATCHES}" \
  "$@"
