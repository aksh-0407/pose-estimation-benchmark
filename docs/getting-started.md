# Getting started

From a fresh checkout to a **rendered mosaic** on one cricket delivery. Every step ends with a
**✓ Check**. The reference delivery is `CCPL080626M1_1_14_1` (7 cameras × 600 frames).

## Prerequisites

- **Linux with an NVIDIA GPU** (P1 pose inference is the throughput bottleneck).
- **Conda** on your `PATH`; a single env named **`pose-lab`** runs everything.
- The frame dataset laid out as `drive/dataset/bt_0{1,2,3}/<delivery>/camera<NN>/frame_*.jpg`
  and calibration under `drive/dataset/calibration-data/` (not committed — provided out of band).

```bash
pip install -e .            # puts src/ on the path (core.* / identity.* / tools.*)
python tools/check_environment.py
```

**✓ Check:** it prints your Python version and, if a GPU is visible, your CUDA device.

## Step 0 — Confirm the checkout is healthy

```bash
python -m pytest -q
python tools/audit_repo.py --fail
```

**✓ Check:** tests pass and the audit prints `Repository hygiene audit passed`.

## Step 1 — Set up the P1 model (RTMPose-X + RTMDet detector)

P1 is top-down: an RTMDet person detector feeds the RTMPose-X pose model, both in `pose-lab`.

```bash
python tools/setup_model_envs.py --models rtmpose_x_body8 --download-assets
python tools/sync_model_store.py
python tools/check_assets.py --models rtmpose_x_body8 --fail-missing
```

**✓ Check:** `check_assets` exits 0 with the RTMPose-X pose weights and the RTMDet detector
weights present under `models/`. Full runbook: [rtmpose-x-runbook.md](rtmpose-x-runbook.md).

## Step 2 — P1: 2D pose over the delivery

```bash
python -m core.inference.run_phase1_rtmpose_inference \
  --model-id rtmpose_x_body8 --deliveries CCPL080626M1_1_14_1 \
  --run-id p1_demo --run-dir data/derived/runs/p1_demo
```

**✓ Check:** `data/derived/runs/p1_demo/predictions/` has 7 JSONL files (one per camera), each
~600 lines. Every player record carries `pose_2d` (COCO-17) and `pose_2d_native` (Halpe-26).

## Step 3 — Run the identity chain

The whole chain (stabilization → tracking → association → lift → global-id → roles) is one command:

```bash
python -m main --deliveries CCPL080626M1_1_14_1 \
  --input-tree data/derived/runs/p1_demo \
  --output-tree data/derived/runs/demo \
  --artifacts-root data/derived/mosaics/demo
```

Or run stages individually (useful when debugging one stage):

```bash
D=CCPL080626M1_1_14_1 ; ROOT=data/derived/runs/demo/deliveries/$D

python -m identity.p2_tracking.run_per_camera_tracking \
  --input-run-dir data/derived/runs/p1_demo --output-run-dir $ROOT/02_tracking \
  --drive-root drive --delivery-id $D --config configs/02_tracking.yaml

python -m identity.p3_association.run_cross_camera_association \
  --input-run-dir $ROOT/02_tracking --output-run-dir $ROOT/03_association \
  --drive-root drive --delivery-id $D --config configs/03_association.yaml

python -m identity.p5_global_id.run_global_id \
  --input-run-dir $ROOT/03_association --output-run-dir $ROOT/05_global_id \
  --drive-root drive --delivery-id $D --config configs/05_global_id.yaml
```

**✓ Check:** `$ROOT/05_global_id/` contains `predictions/*.jsonl` with `global_player_id`,
`diagnostics/ground_tracks.jsonl`, and `global_id_metrics.json`. In the metrics, distinct IDs
should be near the ~13–15 roster and same-camera collisions should be 0.

## Step 4 — Render the mosaic

```bash
python -m identity.visualization.render_videos \
  --drive-root drive --run-dir $ROOT/05_global_id --delivery-id $D --mode mosaic --show p4
```

**✓ Check:** the mosaic `.mp4` plays: 7 camera tiles + a bird's-eye ground monitor + a roster
panel, skeletons coloured by stable global ID. A top-down-only view is `--mode ground`.

## You're done — now what?

- Understand *why* each stage does what it does, and where it's weak → the
  [pipeline reference](pipeline/README.md) and [architecture](architecture.md).
- The current measured issues → [diagnosis/](diagnosis/README.md); the fix backlog →
  [changes_tbd.md](changes_tbd.md).
