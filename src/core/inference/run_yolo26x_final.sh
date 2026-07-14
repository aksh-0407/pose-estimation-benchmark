#!/usr/bin/env bash
set -euo pipefail

# Final YOLO26x-pose Phase 1 inference wrapper.
#
# Defaults come from local YOLO tuning:
#   batch_size=8, resize_long_side=640, decode_workers=4.
#
# By default this discovers deliveries from drive/dataset/bt_01 and writes all
# outputs into one RTMPose-style run directory. Each delivery run covers all
# camera groups because run_phase1_yolo_inference.py resolves cameras across
# bt_01/bt_02/bt_03.
#
# Examples:
#   bash scripts/run_yolo26x_final.sh
#   bash scripts/run_yolo26x_final.sh --deliveries CCPL080626M1_1_14_1
#   bash scripts/run_yolo26x_final.sh --groups bt_01 --list
#   bash scripts/run_yolo26x_final.sh --frame-limit 100 --no-resume
#
# Visualization controls:
#   YOLO_VISUALIZE=0 bash scripts/run_yolo26x_final.sh
#   YOLO_OVERLAY_ROW_INDICES="1 150 300 450 600" YOLO_OVERLAY_LIMIT=5 bash scripts/run_yolo26x_final.sh

DRIVE_ROOT="${DRIVE_ROOT:-drive}"
CAPTURE_GROUPS="${CAPTURE_GROUPS:-bt_01}"
MODEL_ID="${MODEL_ID:-yolo26x_pose}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-yolo26x-pose-full-db8}"
RUN_DIR="${RUN_DIR:-data/derived/runs/${RUN_ID}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.7}"
NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.6}"
RESIZE_LONG_SIDE="${RESIZE_LONG_SIDE:-640}"
DECODE_WORKERS="${DECODE_WORKERS:-4}"
YOLO_VISUALIZE="${YOLO_VISUALIZE:-1}"
YOLO_OVERLAY_ROW_INDICES="${YOLO_OVERLAY_ROW_INDICES:-${YOLO_OVERLAY_FRAME_IDS:-1 150 300 450 600}}"
YOLO_OVERLAY_LIMIT="${YOLO_OVERLAY_LIMIT:-5}"
YOLO_OVERLAY_KEYPOINT_THRESHOLD="${YOLO_OVERLAY_KEYPOINT_THRESHOLD:-0.2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

delivery_ids=()
groups=()
passthrough=()
dry_run=0
list_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deliveries)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        delivery_ids+=("$1")
        shift
      done
      ;;
    --groups)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        groups+=("$1")
        shift
      done
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --list)
      list_only=1
      shift
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

if [[ ${#groups[@]} -eq 0 ]]; then
  # shellcheck disable=SC2206
  groups=(${CAPTURE_GROUPS})
fi

discover_deliveries() {
  local group delivery_dir
  for group in "${groups[@]}"; do
    local group_dir="${DRIVE_ROOT}/dataset/${group}"
    if [[ ! -d "${group_dir}" ]]; then
      echo "missing dataset group: ${group_dir}" >&2
      return 1
    fi
    for delivery_dir in "${group_dir}"/*; do
      [[ -d "${delivery_dir}" ]] || continue
      basename "${delivery_dir}"
    done
  done | sort -u
}

if [[ ${#delivery_ids[@]} -eq 0 ]]; then
  mapfile -t delivery_ids < <(discover_deliveries)
fi

if [[ ${#delivery_ids[@]} -eq 0 ]]; then
  echo "No deliveries selected." >&2
  exit 1
fi

echo "YOLO26x final settings:"
echo "  model=${MODEL_ID} device=${DEVICE} batch=${BATCH_SIZE} imgsz=${IMGSZ}"
echo "  resize_long_side=${RESIZE_LONG_SIDE} decode_workers=${DECODE_WORKERS}"
echo "  visualizations=${YOLO_VISUALIZE} overlay_row_indices=${YOLO_OVERLAY_ROW_INDICES} overlay_limit=${YOLO_OVERLAY_LIMIT}"
echo "  run_id=${RUN_ID}"
echo "  run_dir=${RUN_DIR}"
echo "Selected ${#delivery_ids[@]} delivery run(s):"
printf '  %s\n' "${delivery_ids[@]}"

if [[ "${list_only}" -eq 1 ]]; then
  exit 0
fi

for delivery_id in "${delivery_ids[@]}"; do
  command=(
    "${PYTHON_BIN}" src/core/inference/run_phase1_yolo.py
    --drive-root "${DRIVE_ROOT}"
    --delivery-id "${delivery_id}"
    --model-id "${MODEL_ID}"
    --run-id "${RUN_ID}"
    --run-dir "${RUN_DIR}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --imgsz "${IMGSZ}"
    --conf "${CONF}"
    --iou "${IOU}"
    --nms-iou-threshold "${NMS_IOU_THRESHOLD}"
    --resize-long-side "${RESIZE_LONG_SIDE}"
    --decode-workers "${DECODE_WORKERS}"
  )

  if [[ "${ALLOW_CPU:-0}" == "1" ]]; then
    command+=(--allow-cpu)
  fi
  if [[ "${NO_HALF:-0}" == "1" ]]; then
    command+=(--no-half)
  fi
  if [[ "${NO_PROGRESS:-0}" == "1" ]]; then
    command+=(--no-progress)
  fi

  command+=("${passthrough[@]}")

  printf '+ '
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "${dry_run}" -eq 0 ]]; then
    "${command[@]}"
  fi
done

if [[ "${YOLO_VISUALIZE}" == "1" ]]; then
  overlay_command=(
    "${PYTHON_BIN}" src/identity/visualization/render_phase1_overlays.py
    --drive-root "${DRIVE_ROOT}"
    --run-dir "${RUN_DIR}"
    --artifact-dir "${RUN_DIR}/visualizations"
    --row-indices
  )
  # shellcheck disable=SC2206
  overlay_row_indices=(${YOLO_OVERLAY_ROW_INDICES})
  overlay_command+=("${overlay_row_indices[@]}")
  overlay_command+=(
    --max-per-camera "${YOLO_OVERLAY_LIMIT}"
    --keypoint-threshold "${YOLO_OVERLAY_KEYPOINT_THRESHOLD}"
  )
  printf '+ '
  printf '%q ' "${overlay_command[@]}"
  printf '\n'
  if [[ "${dry_run}" -eq 0 ]]; then
    "${overlay_command[@]}"
  fi
fi
