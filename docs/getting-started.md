# Getting started

This guide takes you from a fresh checkout to a **rendered mosaic** on one cricket
delivery. Every step ends with a **✓ Check** so you know it worked before moving on.

The reference delivery throughout is `CCPL080626M1_1_14_1` (7 cameras × 600 frames).

## Prerequisites

- **Linux with an NVIDIA GPU** (P1 pose inference is the throughput bottleneck).
- **Conda** on your `PATH`.
- **Python 3** plus the small dependency set in [requirements.txt](../requirements.txt):
  `pip install -r requirements.txt`.
- The frame dataset laid out as
  `drive/dataset/bt_0{1,2,3}/<delivery>/camera<NN>/frame_*.jpg` and calibration under
  `drive/dataset/calibration-data/` (not committed — provided out of band).

Quick environment probe:

```bash
python3 tools/check_environment.py
```

**✓ Check:** it prints your Python version and, if a GPU is visible, your CUDA device.

## Step 0 — Confirm the checkout is healthy

```bash
python3 -m pytest -q
python3 tools/audit_repo.py --fail
```

**✓ Check:** tests pass and the audit prints `Repository hygiene audit passed`.

## Step 1 — Set up the P1 model (RTMPose-X + RTMDet detector)

P1 is top-down: an RTMDet person detector feeds the RTMPose-X pose model. Both are set up
together in the `pose-lab` conda env.

```bash
python3 tools/setup_model_envs.py --models rtmpose_x_body8 --download-assets
python3 tools/sync_model_store.py
python3 tools/check_assets.py --models rtmpose_x_body8 --fail-missing
```

**✓ Check:** `check_assets` exits 0 with the RTMPose-X pose weights and the RTMDet
detector weights both present under `models/`.

For a full install/run/tune walkthrough (incl. remote GPU boxes), see
[rtmpose-x-runbook.md](rtmpose-x-runbook.md).

## Step 2 — P1: 2D pose over the delivery

```bash
conda run -n pose-lab python src/core/inference/run_phase1_rtmpose_inference.py \
  --model-id rtmpose_x_body8 --deliveries CCPL080626M1_1_14_1 \
  --run-id p1_demo --run-dir data/derived/runs/p1_demo
```

**✓ Check:** `data/derived/runs/p1_demo/predictions/` has 7 JSONL files (one per camera),
each ~600 lines. Every player record carries `pose_2d` (COCO-17) and `pose_2d_native`
(Halpe-26, 26 kpts incl. feet).

## Step 3 — P2 → P4: tracking, association, global identity

Run the identity stages in an env with NumPy ≥ 1.23.5 / SciPy ≥ 1.10:

```bash
PY=/home/aksh/miniconda3/envs/pose-lab/bin/python
D=CCPL080626M1_1_14_1 ; ROOT=data/derived/runs/demo

$PY -m identity.p2_tracking.run_per_camera_tracking \
  --input-run-dir data/derived/runs/p1_demo --output-run-dir $ROOT/p2 \
  --drive-root drive --delivery-id $D --config configs/02_tracking.yaml

$PY -m identity.p3_association.run_cross_camera_association \
  --input-run-dir $ROOT/p2 --output-run-dir $ROOT/p3 \
  --drive-root drive --delivery-id $D --config configs/03_association.yaml

$PY -m identity.p5_global_id.run_global_id \
  --input-run-dir $ROOT/p3 --output-run-dir $ROOT/p4 \
  --drive-root drive --delivery-id $D --config configs/05_global_id.yaml
```

**✓ Check:** `$ROOT/p4/` contains `predictions/*.jsonl` with `global_player_id`,
`diagnostics/ground_tracks.jsonl`, and `global_id_metrics.json`. Open the metrics: distinct
IDs should be near the ~13–15 roster and same-camera collisions should be 0.

## Step 4 — Render the mosaic

```bash
$PY -m identity.visualization.render_videos \
  --drive-root drive --run-dir $ROOT/p4 --delivery-id $D --mode mosaic --show p4
```

**✓ Check:** `$ROOT/p4/visualizations/videos/<delivery>__all_cameras.mp4` plays: 7 camera
tiles + a bird's-eye ground monitor + a roster panel, with skeletons coloured by stable
global ID. A top-down-only view is `--mode ground`.

## You're done — now what?

- Understand *why* each stage does what it does, and where it's weak → the
  **[critical analysis](critical-analysis/README.md)**.
- Run all identity stages over every delivery at once →
  [`identity.id_pipeline`](scripts.md#the-batch-identity-driver).
- Improve jitter / identity / 3D location → [improving-models.md](improving-models.md).
