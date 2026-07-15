#!/bin/bash
# v8 production chain on the L40S box: all deliveries discovered from the tiled P1
# tree -> /home/ubuntu/pipetrack_v8/. Run from the repo root on the box, inside the
# pose-lab env (mm-stack + numpy/scipy/cv2/yaml).
#
#   cd <repo> && bash tools/run_v8_l40s.sh
#
# Prereqs (already in place): ~/render_drive/dataset/{bt_0X symlinks -> pose_data,
# calibration-data, events-data}; P1 predictions at
# /home/ubuntu/pipetrack_v8/p1_rtmpose-x-tiled (core/inference/run_phase1_l40s.py --tiled-det).
set -euo pipefail

PY=${PY:-$HOME/miniconda3/envs/pose-lab/bin/python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-data/derived/40_full/pipetrack_v8}
P1_TREE=${P1_TREE:-$OUT/p1_rtmpose-x-tiled}
DRIVE=${DRIVE:-$HOME/render_drive}

cd "$REPO"
exec $PY -m main \
  --deliveries all \
  --input-tree "$P1_TREE" \
  --output-tree "$OUT" \
  --drive-root "$DRIVE" \
  --p1b-config configs/01_stabilization.yaml \
  --p2-config configs/02_tracking.yaml \
  --p3-config configs/03_association.yaml \
  --p4-config configs/05_global_id.yaml \
  --p5-config configs/06_roles.yaml \
  --skip-render --jobs 7 --p2-max-workers 2 \
  "$@"
