# Scripts reference

Every script in the cricket pipeline, what it's for, and how to call it. The stages are
separate canonical runs — each reads an `--input-run-dir` and writes an `--output-run-dir`
(holding `predictions/*.jsonl`, `diagnostics/`, `run_manifest.json`, and a `*_metrics.json`).

Quick map (in pipeline order):

| Script | Stage | Purpose |
| ------ | ----- | ------- |
| [`run_phase1_rtmpose_inference.py`](#p1-run_phase1_rtmpose_inferencepy) | P1 | RTMDet detect + RTMPose-X pose per camera → 2D keypoints (COCO-17 + Halpe-26). |
| [`run_phase1_l40s.py`](#p1-remote-run_phase1_l40spy) | P1 | Same P1, for the remote L40S capture box's `bt1/bt2/bt3` layout. |
| [`run_per_camera_tracking.py`](#p2-run_per_camera_trackingpy) | P2 | Link detections into per-camera tracklets. |
| [`run_cross_camera_association.py`](#p3-run_cross_camera_associationpy) | P3 | Cluster the same player across cameras (tracklet graph). |
| [`run_global_id.py`](#p4-run_global_idpy) | P4 | Persistent global IDs + tracklet stitching + ground tracks. |
| [`run_id_pipeline.py`](#the-batch-identity-driver) | P3→P4 | Batch driver over all deliveries + metric panel/diff. |
| [`run_role_assignment.py`](#p5-run_role_assignmentpy) | P5 | Batter/bowler/fielder roles from ground geometry. |
| [`triangulate_predictions.py`](#p6-triangulate_predictionspy) | P6 | Multi-view 2D → smoothed 3D world skeleton. |
| [`export_ue_packets.py`](#export_ue_packetspy) | export | Triangulated 3D → Unreal Engine pose packets. |
| [`render_phase1_videos.py`](#render_phase1_videospy) | render | Mosaic / per-camera / bird's-eye videos. |

Setup helpers (shared): [`setup_model_envs.py`](#setup--hygiene), `check_assets.py`,
`check_environment.py`, `sync_model_store.py`, `audit_repo.py`.

---

## P1 — `run_phase1_rtmpose_inference.py`

Top-down 2D pose over a delivery's frames. An **RTMDet** person detector produces boxes,
then **RTMPose-X** estimates keypoints per box. Output is harmonised to **COCO-17**
(`pose_2d`, the downstream contract) with the full **Halpe-26** kept alongside
(`pose_2d_native`, 26 kpts incl. 6 foot points); `pose_2d.keypoints_px == pose_2d_native[0:17]`.

```bash
conda run -n pose-lab python src/core/inference/run_phase1_rtmpose_inference.py \
  --model-id rtmpose_x_body8 --deliveries CCPL080626M1_1_14_1 \
  --det-batch-size 32 --pose-batch-size 96 --io-workers 16 --prefetch-batches 3 \
  --run-id rtmpose-x --run-dir data/derived/runs/rtmpose-x
```

Key flags: `--groups/--deliveries/--cameras/--frame-limit` (scope), `--list` (preview),
`--no-resume` (recompute), `--overlay` (sample overlays), `--det-config/--det-checkpoint`
(detector selection), `--bbox-thr` (detector score threshold). **Reads:** frames +
`configs/model_envs.yaml`. **Writes:** `predictions/*.jsonl`, `run_manifest.json`,
`p1_metrics.json`. Batch/prefetch changes speed only, never the keypoints. Full runbook:
[rtmpose-x-runbook.md](rtmpose-x-runbook.md).

### P1 remote — `run_phase1_l40s.py`

Self-contained P1 for the remote capture box, whose frames are laid out as
`/home/ubuntu/pose_data/{bt1,bt2,bt3}/<delivery>/camera<NN>/frame_*.jpg`. Reuses the exact
mmdet/mmpose building blocks and the P1 schema (incl. `pose_2d_native`). Supports an
in-process `--sweep --grid` batch tuner.

---

## P2 — `run_per_camera_tracking.py`

Links per-frame detections into per-camera **tracklets** (a `local_track_id` per camera):
a constant-velocity Kalman filter + ByteTrack-style two-stage association, with a masked
weighted-cosine **pose distance** cue and a calibrated ground-reachability gate.

```bash
$PY -m identity.p2_tracking.run_per_camera_tracking \
  --input-run-dir <p1-run> --output-run-dir <p2-run> \
  --drive-root drive --delivery-id <delivery> --config configs/02_tracking.yaml \
  --expected-frames 600
```

**Reads:** P1 `predictions/*.jsonl`, calibration, `configs/02_tracking.yaml`.
**Writes:** re-emitted `predictions/*.jsonl` (+ `local_track_id`), tracking metrics.

## P3 — `run_cross_camera_association.py`

Decides **who-is-who across cameras**. The default `tracklet_graph` mode fuses per-cue
log-likelihood ratios (ground proximity, epipolar/Sampson, appearance, pose-shape,
motion) over whole tracklet-pairs and merges with a constrained agglomerative clustering
that respects the one-detection-per-camera and same-camera cannot-link constraints. The
emitted per-player ground position uses the z=0 reprojection solver (`z0_reproj`).

```bash
$PY -m identity.p3_association.run_cross_camera_association \
  --input-run-dir <p2-run> --output-run-dir <p3-run> \
  --drive-root drive --delivery-id <delivery> --config configs/03_association.yaml \
  --expected-frames 600
```

**Writes:** `predictions/*.jsonl`, `association_metrics.json`, and
`diagnostics/correspondences.jsonl` (per-frame cross-camera cluster membership). A ground
-accuracy evaluator is `tools/diagnosis/eval_ground_accuracy.py`.

## P4 — `run_global_id.py`

Assigns persistent `global_player_id`s (online single-hypothesis MOT on the ground plane
with a role-aware Singer-acceleration Kalman), stitches fragmented tracklets (post-hoc
min-cost-flow path cover), and emits fused per-player ground tracks.

```bash
$PY -m identity.p5_global_id.run_global_id \
  --input-run-dir <p3-run> --output-run-dir <p4-run> \
  --drive-root drive --delivery-id <delivery> --config configs/05_global_id.yaml \
  [--ground-truth labels.jsonl]
```

**Writes:** `predictions/*.jsonl` (global IDs), `diagnostics/ground_tracks.jsonl`,
`id_switch_report.json`, `global_id_metrics.json`. With `--ground-truth`, also
MOTA/IDF1-style metrics; without labels, all identity figures are explicitly proxies.

### The batch identity driver

`src/identity/id_pipeline.py` re-runs P3→P4 across all 8 deliveries in parallel
(BLAS threads capped), then prints a joint metric panel (agreement, distinct IDs,
teleports, collisions, single-camera rate, churn) and optionally diffs it against a frozen
baseline snapshot.

```bash
$PY -m identity.id_pipeline \
  --input-tree data/derived/runs/pipetrack_v3 --output-tree data/derived/runs/pipetrack_v5 \
  --baseline data/derived/runs/pipetrack_v3/_baseline_snapshot --jobs 8
```

## P5 — `run_role_assignment.py`

Classifies each global player's role (batter / bowler / fielder / keeper / umpire) from
ground geometry relative to the pitch axis and bowling direction. **Writes:**
`p5/roles.json` — the mosaic roster reads roles only from this artifact.

## P6 — `triangulate_predictions.py`

Lifts the multi-view 2D keypoints to a **3D world skeleton**: per-joint pairwise-RANSAC
DLT triangulation (`--reprojection-threshold-px 10`, `--min-views 2`), temporal
occlusion fill, skeletal-prior fill for never-seen joints, and confidence-aware temporal
EMA (`--ema-alpha 0.65`).

```bash
$PY -m scripts.export.triangulate_predictions \
  --input-run-dir <p4-run> --output-run-dir <p6-run> \
  --drive-root drive --delivery-id <delivery>
```

Groups observations by `global_player_id`, writes `pose_3d.keypoints_world_m` back into
each camera stream. A legacy flat-JSONL mode (`--predictions --calibration --output`) also
exists.

### `export_ue_packets.py`

Converts triangulated 3D JSONL into Unreal Engine-ready pose packets, tagging each with a
`--model-version`.

## `render_phase1_videos.py`

Renders the diagnostic videos from a P4 run, colouring skeletons by stable global ID.

```bash
# mosaic: 7 calibration-ordered camera tiles + bird's-eye monitor + roster
$PY -m identity.visualization.render_videos \
  --drive-root drive --run-dir <p4-run> --delivery-id <delivery> --mode mosaic --show p4

# bird's-eye ground view only
$PY -m identity.visualization.render_videos \
  --drive-root drive --run-dir <p4-run> --delivery-id <delivery> --mode ground --show p4
```

`--mode {all,per-camera,mosaic,ground}`; `--show {p2,p3,p4}` selects which stage's IDs to
overlay. The tile layout is derived from calibration (no hardcoded camera IDs) by
`src/identity/visualization/mosaic_layout.py`; stable ID colours come from
`src/identity/visualization/identity_colors.py`. A standalone top-down renderer is
`src/identity/visualization/render_bird_eye_view.py`.

---

## Setup & hygiene

| Script | When | Purpose |
| ------ | ---- | ------- |
| `tools/setup_model_envs.py` | first | Create the model conda env + download P1 weights (RTMPose-X + RTMDet). |
| `tools/check_assets.py` | before runs | Report which checkpoints are present (`--fail-missing`). |
| `tools/check_environment.py` | debugging | Print Python, binaries (`nvidia-smi`, `ffmpeg`, …), packages, GPU. |
| `tools/sync_model_store.py` | after asset changes | Regenerate `models/<id>/model.yaml`/`README`/checksums. |
| `tools/audit_repo.py --fail` | before commit | Fail if weights/frames/raw artifacts got tracked by mistake. |

`configs/*.yaml` are documented in [configuration.md](configuration.md). Metrics and
proxies in [metrics.md](metrics.md).
