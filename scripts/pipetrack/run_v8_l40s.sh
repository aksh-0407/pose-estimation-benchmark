#!/bin/bash
# v8 production chain on the L40S box: all deliveries discovered from the tiled P1
# tree -> /home/ubuntu/pipetrack_v8/. Run from the repo root on the box, inside the
# cricket-rtmpose-l env (numpy/scipy/cv2/yaml verified present).
#
#   cd ~/pose-estimation-benchmark && bash scripts/pipetrack/run_v8_l40s.sh
#
# Prereqs (already in place): ~/render_drive/dataset/{bt_0X symlinks -> pose_data,
# calibration-data, events-data}; P1 predictions at
# /home/ubuntu/pipetrack_v8/p1_rtmpose-x-tiled (run_phase1_l40s.py --tiled-det).
set -euo pipefail

PY=${PY:-$HOME/miniconda3/envs/cricket-rtmpose-l/bin/python}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-/home/ubuntu/pipetrack_v8}
P1_TREE=${P1_TREE:-$OUT/p1_rtmpose-x-tiled}
DRIVE=${DRIVE:-$HOME/render_drive}

cd "$REPO"
exec $PY -m scripts.pipetrack.run_full_pipeline \
  --deliveries all \
  --input-tree "$P1_TREE" \
  --output-tree "$OUT" \
  --drive-root "$DRIVE" \
  --p1b-config configs/v8/p1b_stabilization.yaml \
  --p2-config configs/v8/p2_tracking.yaml \
  --p3-config configs/v8/p3_association.yaml \
  --p4-config configs/v8/p4_global_id.yaml \
  --p5-config configs/v8/p5_roles.yaml \
  --skip-render --jobs 7 --p2-max-workers 2 \
  "$@"
