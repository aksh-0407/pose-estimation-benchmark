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

The identity stages are numbered in execution order (`01` to `07`); 2D pose inference (P1) is the shared
upstream producer and lives in `src/core/`. Note the logical order Associate, then Triangulate, then
Track: the 3D lift (`04_lift`) runs before global identity so ID can build on 3D positions. Stage 07
refine (new in pipetrack_v9) makes the final 3D physically valid after identity is frozen.

| Stage | What it does | Entry point |
|---|---|---|
| **P1** 2D inference (foundation) | Detect people (RTMDet) then estimate the 2D Halpe-26 skeleton (RTMPose-X, top-down) per camera | [`src/core/inference/run_phase1_rtmpose_inference.py`](src/core/inference/run_phase1_rtmpose_inference.py) |
| **01** stabilization | Temporal One-Euro smoothing of 2D keypoints before tracking | [`src/identity/p1_stabilization/run_stabilization.py`](src/identity/p1_stabilization/run_stabilization.py) |
| **02** per-camera tracking | Link detections into per-camera tracklets (ByteTrack default, plus a constant-velocity Kalman and pose-cosine; OC-SORT optional via `tracker: ocsort`) | [`src/identity/p2_tracking/run_per_camera_tracking.py`](src/identity/p2_tracking/run_per_camera_tracking.py) |
| **03** cross-camera association | Cluster the same physical player across cameras (tracklet-graph LLR on the ground plane) | [`src/identity/p3_association/run_cross_camera_association.py`](src/identity/p3_association/run_cross_camera_association.py) |
| **04** 3D lift | Multi-view triangulation of the full skeleton (weighted-DLT plus RANSAC plus occlusion fill); binding-keyed, feeds global identity | [`src/identity/p4_lift/run_triangulation.py`](src/identity/p4_lift/run_triangulation.py) |
| **05** global identity | Assign persistent global IDs plus stitch fragmented tracks (online Singer-acceleration Kalman plus min-cost-flow) | [`src/identity/p5_global_id/run_global_id.py`](src/identity/p5_global_id/run_global_id.py) |
| **06** roles | Batter, bowler, fielder from ground geometry (plus peripheral suppression) | [`src/identity/p6_roles/run_role_assignment.py`](src/identity/p6_roles/run_role_assignment.py) |
| **07** refine | Physics-constrained 3D skeleton rebuild, hip de-wobble, low-confidence refill; runs after identity, rewrites only `pose_3d` | [`src/identity/p7_refine/run_refinement.py`](src/identity/p7_refine/run_refinement.py) |
| **08** export and render | Unreal Engine pose packets; the mosaic and bird's-eye videos | [`src/identity/export/export_ue_packets.py`](src/identity/export/export_ue_packets.py), [`src/identity/visualization/render_videos.py`](src/identity/visualization/render_videos.py) |

The whole chain is driven by [`src/main.py`](src/main.py) (`python -m main`, phase-select via
`--from-stage`/`--until-stage`); an identity-only batch driver (association to global_id over all
deliveries) lives in [`src/identity/id_pipeline.py`](src/identity/id_pipeline.py).

## Start here

- New to the repo? Read **[docs/getting-started.md](docs/getting-started.md)**, from a
  fresh checkout to a rendered mosaic on one delivery.
- Want the honest engineering picture, every phase's methods, math, weaknesses, and a
  prioritised fix roadmap grounded in the code and the measured results? Read the
  **[critical analysis](docs/pipeline/README.md)**.

## Documentation

- [docs/index.md](docs/index.md): the documentation map.
- [docs/methods_log.md](docs/methods_log.md): the combined method ledger. Every method tried, its
  before/after A/B, pros and cons, status, and whether it is on or off by default.
- [docs/roadmap.md](docs/roadmap.md): the consolidated backlog and guiding principles.
- [docs/pipeline/](docs/pipeline/README.md): the deep dive, phases, per-phase method analysis, issues,
  and fixes.
- [docs/rtmpose-x-runbook.md](docs/rtmpose-x-runbook.md), install & run P1 (RTMPose-X) on a
  new/remote machine.
- [docs/reference/cli.md](docs/reference/cli.md), every pipeline script and its I/O.
- [docs/reference/configuration.md](docs/reference/configuration.md), the `configs/*.yaml` files.
- [docs/reference/metrics.md](docs/reference/metrics.md), the quality metrics and proxies the pipeline reports.
- [docs/improving-models.md](docs/improving-models.md), the ongoing quality work.
- [docs/troubleshooting.md](docs/troubleshooting.md), when setup breaks.

## Data & calibration

Frames live under `data/raw/<dataset>/bt_0{1,2,3}/<delivery>/camera<NN>/frame_*.jpg` (7 cameras,
~600 frames/delivery). Bundle-adjusted calibration (per-camera 3×4 projection matrices) and pitch
geometry come from `data/raw/8_init/calibration-data/` (one session, shared by both matches). The calibration is
**centimetre-accurate** (ball reprojection p95 ≤ 4.5 px), which is why the pipeline solves
identity and location directly on the calibrated ground plane. Heavy inputs (frames,
weights) are never committed.
