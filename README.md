# Cricket Multi-Camera 3D Pose & Identity Pipeline (Group 1)

This repository turns synchronised **7-camera cricket broadcast footage** into per-player
**3D pose** and **stable global identities**, and renders the result as a diagnostic
**mosaic video** (all seven camera tiles + a bird's-eye ground monitor + a team roster).
The end target is production use (feeding an Unreal Engine graphics layer); the code here
is the perception pipeline that produces the poses, identities, and 3D locations.

> The COCO model-benchmarking / model-selection framework that this repo grew out of now
> lives on the **`benchmark`** branch. `main` is the cricket delivery pipeline only.

## The pipeline at a glance

Each stage consumes and produces a **canonical run directory** (`predictions/*.jsonl` +
`diagnostics/` + a `*_metrics.json`), so any stage can be inspected or re-run on its own.

| Stage | What it does | Entry point |
|---|---|---|
| **P1** 2D inference | Detect people (RTMDet) + estimate 2D keypoints (RTMPose-X, top-down) per camera; emits COCO-17 **and** Halpe-26 | [`scripts/inference/run_phase1_rtmpose_inference.py`](scripts/inference/run_phase1_rtmpose_inference.py) |
| **P2** per-camera tracking | Link detections into per-camera tracklets (ByteTrack-style Kalman + pose-cosine) | [`scripts/tracking/run_per_camera_tracking.py`](scripts/tracking/run_per_camera_tracking.py) |
| **P3** cross-camera association | Cluster the same physical player across cameras (tracklet-graph LLR on the ground plane) | [`scripts/association/run_cross_camera_association.py`](scripts/association/run_cross_camera_association.py) |
| **P4** global identity | Assign persistent global IDs + stitch fragmented tracks (online Singer-KF MOT + min-cost-flow) | [`scripts/global_id/run_global_id.py`](scripts/global_id/run_global_id.py) |
| **P5** roles | Batter / bowler / fielder from ground geometry | [`scripts/roles/run_role_assignment.py`](scripts/roles/run_role_assignment.py) |
| **P6** 3D lift | Multi-view triangulation of the full skeleton (weighted-DLT + RANSAC + occlusion fill) | [`scripts/export/triangulate_predictions.py`](scripts/export/triangulate_predictions.py) |
| **Export / render** | Unreal Engine pose packets; the mosaic + bird's-eye videos | [`scripts/export/export_ue_packets.py`](scripts/export/export_ue_packets.py), [`scripts/visualization/render_phase1_videos.py`](scripts/visualization/render_phase1_videos.py) |

A batch driver for the identity stages (P3→P4 over all deliveries) lives in
[`scripts/pipetrack/run_id_pipeline.py`](scripts/pipetrack/run_id_pipeline.py).

## Start here

- New to the repo? Read **[docs/getting-started.md](docs/getting-started.md)** — from a
  fresh checkout to a rendered mosaic on one delivery.
- Want the honest engineering picture — every phase's methods, math, weaknesses, and a
  prioritised fix roadmap grounded in the code and the measured results? Read the
  **[critical analysis](docs/critical-analysis/README.md)**.

## Documentation

- [docs/index.md](docs/index.md) — the documentation map.
- [docs/critical-analysis/](docs/critical-analysis/README.md) — **the deep dive**: phases,
  per-phase method analysis, issues, and fixes.
- [docs/rtmpose-x-runbook.md](docs/rtmpose-x-runbook.md) — install & run P1 (RTMPose-X) on a
  new/remote machine.
- [docs/scripts.md](docs/scripts.md) — every pipeline script and its I/O.
- [docs/configuration.md](docs/configuration.md) — the `configs/*.yaml` files.
- [docs/metrics.md](docs/metrics.md) — the quality metrics and proxies the pipeline reports.
- [docs/improving-models.md](docs/improving-models.md) — the ongoing quality work.
- [docs/troubleshooting.md](docs/troubleshooting.md) — when setup breaks.

## Data & calibration

Frames live under `drive/dataset/bt_0{1,2,3}/<delivery>/camera<NN>/frame_*.jpg` (7 cameras,
~600 frames/delivery). Bundle-adjusted calibration (per-camera 3×4 projection matrices) and
pitch geometry come from `drive/dataset/calibration-data/`. The calibration is
**centimetre-accurate** (ball reprojection p95 ≤ 4.5 px), which is why the pipeline solves
identity and location directly on the calibrated ground plane. Heavy inputs (frames,
weights) are never committed.
