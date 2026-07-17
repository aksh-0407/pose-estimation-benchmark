#!/bin/bash
# Full pipeline chain on the L40S box: all deliveries discovered from the run
# tree's per-delivery P1 output, stages 01..07 written next to it. Run from the
# repo root on the box, inside the pose-lab env.
#
#   cd <repo> && bash tools/run_v8_l40s.sh
#
# Prereqs: P1 predictions at $OUT/<DELIVERY>/00_inference/predictions/ (written by
# src/core/inference/run_phase1_l40s.py with the default per-delivery layout), and
# $DRIVE holding the dataset (bt_0X frame dirs, calibration-data, events-data).
set -euo pipefail

PY=${PY:-$HOME/miniconda3/envs/pose-lab/bin/python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-data/derived/40_full/pipetrack}
DRIVE=${DRIVE:-$HOME/render_drive}

cd "$REPO"
exec $PY src/main.py \
  --deliveries all \
  --output-tree "$OUT" \
  --drive-root "$DRIVE" \
  --p1b-config configs/01_stabilization.yaml \
  --p2-config configs/02_tracking.yaml \
  --p3-config configs/03_association.yaml \
  --p4-config configs/05_global_id.yaml \
  --p5-config configs/06_roles.yaml \
  --skip-render --jobs 6 --p2-max-workers 2 \
  "$@"
