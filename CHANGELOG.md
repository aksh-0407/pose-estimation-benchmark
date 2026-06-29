# Changelog & Engineering Guide — Cricket Multi-Camera Pose, Tracking & Global ID (Group 1)

This is the authoritative onboarding + handover document for the Group-1 pipeline. It is both a
**changelog** (what changed and why) and an **engineering reference** (repo layout, every file and how
it works, the methods/tools used, how to run everything, and the status of every task). Read §1 for the
story, §2–§4 for structure/files, §5–§6 for how it works, §7 to run it, §8–§9 for status.

Maintainers: Group 1. Match dataset: `CCPL080626` (8 deliveries, 7 cameras, 600 frames each).

---

## 0. What this project does

Seven calibrated, frame-synchronized telephoto broadcast cameras observe a cricket ground. The Group-1
pipeline produces, for every visible person and every frame, a **stable global player ID** that is the
same across every camera that can see them, together with their 2D pose (and optionally 3D). The central
difficulty is **cross-camera identity**: if `cam_01` labels someone `P001`, every other camera that sees
that same physical person must also label them `P001`, and a single camera frame must never show two
different people sharing one ID.

End-to-end stages:

```
P1  per-camera 2D detection + 17-keypoint pose   (RTMPose-L or YOLO)  -> predictions/*.jsonl
P2  per-camera tracking                          (Kalman + Hungarian) -> local_track_id
P3  cross-camera association                     (ground-plane + cycle-consistent clustering)
P4a online global-ID tracking                    (world-space MOT on the pitch plane)
P4b post-delivery stitching                      (min-cost-flow gap-bridge)
viz mosaic / per-camera / ground videos          (identity-stable colours)
(P6 optional 3D lift; P5 role classification — not in the main path yet)
```

---

## 1. Changelog

### 2026-06-29 — Phase 2–4 overhaul: global-ID correctness + coverage

**Problem.** The earlier P3/P4 implementation produced physically impossible output: the same camera
frame showed two different poses with one global ID; many detections received no ID; facing cameras
often failed to share an ID. Root causes, confirmed by reading every module:
- a convoluted, unverifiable P4a "track manager" with five interacting assignment passes;
- a P4b stitcher that was **dead code** (`edges=[]`, `links={}` hardcoded) plus a *backwards* heuristic
  that merged IDs whenever P3 happened to co-associate them;
- a **mislabeled camera-pair config** that confused antipodal *positions* with *facing* (co-observing)
  cameras.

**1.1 Camera geometry — corrected (`configs/facing.jpeg`).**
The cameras that actually **co-observe** the same players are the **facing pairs** — anti-parallel
optical axes aimed at the same ground strip: **C1↔C4 (centre), C2↔C6 (near end), C3↔C5 (far end)**;
**C7** is independent. This is *not* the same as diametrically-opposite *positions*: C2 and C5 sit
antipodally but frame different strips and never co-observe.
- Fixed `configs/p3_association.yaml` `opposite_camera_pairs` from the wrong `(C2,C5),(C3,C6)` to
  `(C2,C6),(C3,C5)`; fixed the default in `scripts/association/config.py`.
- Added `pose_estimation/cricket/geometry.py::derive_facing_pairs()` + `camera_axis_lookat()` which
  **auto-derive** facing pairs from the calibration optical axes (principal-axis → z=0 look-at point,
  anti-parallel forward + nearest look-at). The P3 runner now **overrides** any hand-edited list with
  the calibration-derived one and records it as `facing_camera_pairs` in `association_metrics.json`.
  Calibration is the single source of truth.
- Verified three independent ways: the `facing.jpeg` diagram; the look-at computation from
  `Bundle_Adjusted_extrinsics.json`; and landmark visibility (C2/C6 see the near stumps `nsmb`, C3/C5
  the far stumps `fsmb`). Surveyed-landmark back-projection round-trips to **4–7 cm** on the z=0 pitch
  plane, confirming the world frame and the foot→ground homography.

**1.2 P4a online tracker — rebuilt (`scripts/global_id/track_manager.py`).**
Replaced the five-pass manager with a single, auditable **world-space multi-object tracker** on the
ground plane. P3 already solves the *spatial* cross-camera association (each correspondence = one world
detection at a foot point, ≤1 member per camera), so P4a only solves *temporal* association. Per frame:
predict (Singer-acceleration Kalman) → **Stage 1** exact P2 `local_track_id` continuity (revives even a
deleted track; identity beats a bad foot projection) → **Stage 2** χ²-gated Mahalanobis Hungarian on
ground distance → **Stage 3** re-entry from a deleted pool or birth → confirm after `confirm_hits` /
delete after a lost window.
- **The "two poses, one ID, one camera-frame" bug is now impossible by construction:** each
  correspondence maps to exactly one track (Hungarian is 1:1, re-entry claims distinct deleted tracks,
  births are fresh), so two detections in one camera-frame are in different correspondences → different
  tracks → different IDs.
- Single-camera people are tracked too, so they receive IDs.

**1.3 P4b stitching — wired correctly (`scripts/global_id/stitching.py`, `runner.py`).**
The min-cost-flow path cover (`extract_segments → build_link_costs → solve_flow → remap_ids`) is now
actually called; it bridges fragmented confirmed tracklets using temporal + spatial + role +
velocity-continuity costs. `remap_ids` forbids any merge whose two histories ever share a
`(camera, frame)` cell, so stitching can never create a collision. Removed the backwards
`cross_camera_identity_evidence` / `remap_ids_from_cross_camera_evidence` functions. Contract validation
now runs on the **final** records (after stitching).

**1.4 ID-coverage tuning.** `p4a.confidence_discard` lowered 0.30 → 0.15 so the world tracker also
follows **untracked single-camera detections** (P3 confidence ~0.20 — people P2 never managed to
tracklet). Persistent ones confirm into a geometric track and get an ID; transient noise never reaches
`confirm_hits`.

**1.5 Tests.** Full suite **107 passed**. Added geometry tests for `derive_facing_pairs`/
`camera_axis_lookat`; a realistic "two distinct same-camera detections get distinct IDs" invariant test;
a facing-pair ID-sharing test; an "every confident grounded detection gets an ID" test. Removed the dead
evidence-merge test.

**1.6 Operational note — BLAS thread pinning (important).** P2 spawns 7 worker processes; with default
multi-threaded NumPy/SciPy this oversubscribes the CPU and *appears to hang*. Always export
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before running the pipeline.

**1.7 Verified end-to-end** on delivery `CCPL080626M1_1_14_1` (RTMPose-L, 7 cams × 600 frames):
`status: pass`, **0** same-(camera,frame) collisions across all 4,200 records / 13,493 detections,
**100 % of detections carry an ID** (5 end-of-clip stragglers), **21 distinct IDs** (12 cross-camera;
e.g. P001 spans cam_01/03/04/05/07). Before the discard tuning: 17 IDs, 84.8 % coverage.

**Files touched this release:** `configs/p3_association.yaml`, `configs/p4_global_id.yaml`,
`scripts/association/config.py`, `scripts/association/runner.py`,
`pose_estimation/cricket/geometry.py`, `scripts/global_id/track_manager.py`,
`scripts/global_id/runner.py`, `scripts/global_id/stitching.py`,
`tests/test_cricket_track_manager.py`, `tests/test_cricket_stitching.py`,
`tests/test_cricket_geometry.py`, and this `CHANGELOG.md`.

---

## 2. Repository structure (annotated)

```
Pose_Estimation/
├── CHANGELOG.md  README.md  CONTRIBUTING.md  requirements.txt  pytest.ini
├── configs/                         # all runtime configuration + reference diagrams
├── pose_estimation/                 # LIBRARY: importable, unit-tested algorithms + data contract
│   ├── cricket/                     # cricket-domain geometry, tracking, contract, calibration
│   ├── adapters/ datasets/ evaluation/ hardware/ visualization/   # model-IO, COCO, metrics, reports
│   └── *.py                         # keypoints, metrics, triangulation, schemas, ue_export, …
├── scripts/                         # CLI ENTRY POINTS (thin orchestration over the library)
│   ├── inference/                   # P1 pose inference (RTMPose / YOLO)
│   ├── tracking/                    # P2 per-camera tracking
│   ├── association/                 # P3 cross-camera association
│   ├── global_id/                   # P4 global-ID tracking + stitching
│   ├── export/                      # P6 3D triangulation + Unreal Engine packets
│   ├── visualization/               # mosaic / per-camera / ground videos + still overlays
│   ├── benchmark/ setup/ tuning/    # model benchmarking, env setup, batch-size tuning
│   └── README.md
├── tests/                           # 107 pytest tests (unit + CLI integration)
├── docs/                            # longer-form docs (getting-started, workflow, scripts, …)
├── drive/dataset/                   # INPUT DATA: frames + calibration (see §7)
├── benchmarks/runs/                 # P1 outputs + downstream P2/P3/P4 run folders
├── models/ external/ wheels/        # model weights, vendored model repos, prebuilt wheels
├── data/ results/ report/ archive/ presentables/ Papers/ vedant/   # datasets, results, docs, scratch
```

**Design principle — library vs scripts.** `pose_estimation/` is reusable, importable, unit-tested
*algorithms and the data contract*; `scripts/` is *thin CLI runners* that compose them per pipeline
stage. Tests target the library. (A larger reorganization is deliberately deferred — see §9.)

**Canonical prediction file naming:** `<capture_group>__<delivery_id>__<cam_id>.jsonl`, e.g.
`bt_01__CCPL080626M1_1_14_1__cam_01.jsonl`.

---

## 3. File-by-file reference

### 3.1 `pose_estimation/cricket/` — the domain core (used by P2–P6)

| File | Responsibility |
| --- | --- |
| `geometry.py` | Multi-view primitives: `camera_center_from_P`, `compute_fundamental_matrix`, `sampson_distance`, `ground_homography_from_projection`, **`pixel_to_ground_xy`** (image→world z=0), `parallax_angle_deg`, `ground_contact_pixel` (robust foot keypoint), and **new** `camera_axis_lookat` + `derive_facing_pairs` (auto-derive co-observing pairs from optical axes). |
| `ground_kalman.py` | `SingerGroundKalman` — Singer-acceleration motion model on the 2D ground plane (state `[x,y,vx,vy,ax,ay]`), with role-aware process noise (`RoleParams`), Mahalanobis gating, Joseph-form update, covariance capping, and `switch_role`. |
| `triangulation.py` (repo-root `pose_estimation/triangulation.py`) | Canonical weighted-DLT + RANSAC triangulation (`triangulate_point_dlt`, `ransac_triangulate_point`, `reprojection_errors_for_point`). Single home for the SVD solve. |
| `calibration.py` | Loads `Bundle_Adjusted_*` matrices, projects/validates surveyed landmarks, ball-reprojection audit, Phase-0 calibration readiness report. |
| `contract.py` | The Group-1 output **schema + validator** (`validate_group1_frame`). Enforces COCO-17 shape, role/track-state enums, and the **uniqueness invariant**: a `global_player_id` may not repeat within one camera frame. |
| `tracking_metrics.py` | Geometry/fragmentation **proxy metrics**: `identity_collision_metrics` (same-camera-frame collisions), `track_completeness`, `identity_fragmentation_proxy`, `association_proxy_metrics`, and `evaluate_ground_truth` (when labels are supplied). |
| `dataset.py` | Delivery/camera directory resolution and canonical filename parsing. |
| `phase1_outputs.py`, `phase1_runner.py`, `phase1_yolo_adapter.py` | P1 schema (COCO-17 edges, schema version) and inference orchestration for RTMPose/YOLO. |
| `events.py`, `blockers.py` | Event-artifact structures (ball data) and masking/blocking helpers. |

### 3.2 `pose_estimation/` — general (model-agnostic) library

| File | Responsibility |
| --- | --- |
| `keypoints.py` | Keypoint-set mappings (COCO-17 ↔ MediaPipe-33 ↔ OpenPose-25 ↔ WholeBody-133). |
| `metrics.py`, `evaluation/metrics2d.py`, `evaluation/metrics3d.py`, `coco_keypoint_eval.py` | PCK, OKS, MPJPE/pMPJPE, PCK3D, acceleration error, COCO OKS AP/AR. |
| `triangulation.py`, `ue_export.py` | 3D solve and Unreal Engine pose-packet export (coordinate mapping + serialization). |
| `adapters/` (`base.py`, `registry.py`) | Normalize different model outputs into the prediction record; adapter registry. |
| `predictions.py`, `schemas.py`, `results_io.py`, `registry.py`, `constants.py` | Prediction schema/versioning, Pydantic payloads, result (de)serialization, model registry, constants. |
| `datasets/coco.py`, `hardware/reporting.py`, `visualization/reports.py` | COCO utilities, hardware stats, HTML/Jupyter report generation. |

### 3.3 `scripts/inference/` — P1 (pose detection)

| File | Responsibility |
| --- | --- |
| `run_phase1_rtmpose_inference.py` | Top-down RTMPose-L inference over drive frames → canonical P1 JSONL. **Preferred source.** |
| `run_phase1_yolo_inference.py` | Single-stage YOLO-pose inference → canonical P1 JSONL. |

### 3.4 `scripts/tracking/` — P2 (per-camera tracking)

| File | Responsibility |
| --- | --- |
| `run_per_camera_tracking.py` | CLI entry point. |
| `runner.py` | Orchestrates per-camera tracking across cameras with a `ProcessPoolExecutor` (one worker per camera); writes run manifest + metrics. **Run with BLAS threads pinned.** |
| `tracker.py` | `CameraTracker` — two-stage (high-/low-confidence) association using IoU + pose similarity, Mahalanobis gating, and a pose gallery. |
| `track.py` | Per-camera `Track` state machine; deterministic `local_track_id` (`{cam}_trk_{n:04d}`). |
| `kalman.py` | Constant-velocity Kalman for per-camera bbox/position. |
| `pose_vector.py` | Pose-appearance feature vectors for association. |
| `calibration.py` | Builds per-camera ground calibrators; loads projection matrices from the drive. |
| `config.py`, `jsonl_io.py` | P2 config dataclasses; JSONL read/write. |

### 3.5 `scripts/association/` — P3 (cross-camera association)

| File | Responsibility |
| --- | --- |
| `run_cross_camera_association.py` | CLI entry point. |
| `runner.py` | Per-delivery orchestration: load calibration, **auto-derive facing pairs**, build geometry cache, associate each frame, write `diagnostics/correspondences.jsonl` + metrics. |
| `associator.py` | The association engine. `multiway_cycle` mode: pairwise ground/epipolar costs → **constrained single-linkage clustering** that merges clusters only if (a) ≤1 detection per camera, (b) ground consensus holds, (c) RANSAC triangulation reprojects within tolerance into every member view (cycle-consistency). Emits `Correspondence` objects. |
| `geometry_cache.py` | Precomputes per-pair fundamental matrices, degeneracy flags (epipole-in-image / near-collinear baseline), and epipolar/triangulation weights. |
| `appearance.py` | Appearance-distance cue (low weight — identical kit). |
| `config.py`, `jsonl_io.py` | P3 config (incl. `opposite_camera_pairs`, gates, weights); JSONL helpers + `record_to_detections`. |

### 3.6 `scripts/global_id/` — P4 (global ID) — **rebuilt this release**

| File | Responsibility |
| --- | --- |
| `run_global_id.py` | CLI entry point. |
| `runner.py` | Drives P4a frame-by-frame, writes back `global_player_id`/`track_state`/`track_confidence` to each record, runs P4b stitching, builds `ground_tracks.jsonl` + `id_switch_report.json`, validates final records, emits `global_id_metrics.json`. |
| `track_manager.py` | **`TrackManager`** — the rebuilt world-space tracker (predict → identity → geometry-Hungarian → re-entry/birth → confirm/delete). |
| `global_track.py` | `GlobalTrack` dataclass + lifecycle (`apply_hit`, `apply_identity_only_hit`, `mark_missed`, `maybe_confirm`, `should_delete`, local-id history). |
| `stitching.py` | P4b min-cost-flow path cover: `Segment`/`Edge`, `extract_segments`, `build_link_costs`, `solve_flow` (`scipy.optimize.linear_sum_assignment`), `remap_ids` (occupancy-guarded). |
| `config.py`, `jsonl_io.py` | P4 config (`P4AConfig`/`P4BConfig`, role params, gates); `row_to_correspondences`, `write_prediction_streams`. |

### 3.7 `scripts/visualization/` — see §6 for full details

| File | Responsibility |
| --- | --- |
| `render_phase1_videos.py` | Mosaic / per-camera / ground MP4 renderer (`--mode`, `--show p2|p3|p4`). |
| `render_phase1_overlays.py` | Still-frame JPEG overlays for QA of selected frames. |
| `identity_colors.py` | Deterministic identity→colour palette (stable across cameras/time). |
| `check_visualization_quota.py` | Disk-quota guard for large renders. |

### 3.8 `scripts/export/`, `benchmark/`, `setup/`, `tuning/`

| File | Responsibility |
| --- | --- |
| `export/triangulate_predictions.py` | P6: group P4 output by `global_player_id`, triangulate per-joint 3D (`pose_3d.keypoints_world_m`), confidence-aware EMA smoothing. |
| `export/export_ue_packets.py` | Convert triangulated 3D → Unreal Engine pose packets. |
| `benchmark/benchmark.py` (+ `*_coco_benchmark.py`, `score_models.py`, `validate_results.py`, `write_phase1_shortlist.py`, `download_coco_keypoints.py`, `benchmark_models.py`) | Master benchmark CLI and COCO model-selection tooling used to choose P1 models. |
| `setup/` (`setup_model_envs.py`, `smoke_model_envs.py`, `setup_openpose.py`, `sync_model_store.py`, `phase0_audit.py`, `audit_repo.py`, `check_assets.py`, `check_environment.py`, `run_model_smoke.py`, `relativize_run_paths.py`) | Per-model conda env creation, smoke checks, repo/asset audits, Phase-0 calibration audit. |
| `tuning/` (`tune_rtmpose_batches.py`, `tune_yolo_batches.py`) | Inference batch-size tuning. |

### 3.9 `configs/`

| File | Responsibility |
| --- | --- |
| `conventions.jpeg` | World-frame convention (origin = pitch centre, +Y = Far End, z=0 = ground). |
| `facing.jpeg` | **Authoritative** co-observing camera pairs: C1↔C4, C2↔C6, C3↔C5; C7 independent. |
| `p2_tracking.yaml` | P2 tracking parameters (confidence gates, χ² gate, gallery, etc.). |
| `p3_association.yaml` | P3 parameters (matching mode, ground/epipolar weights, gates, **facing `opposite_camera_pairs`**, anchor priority). |
| `p4_global_id.yaml` | P4a/P4b parameters (role Kalman params, χ² gate, `confidence_discard=0.15`, lost windows, stitching weights). |
| `benchmark_protocol.yaml`, `datasets.yaml`, `keypoint_mappings.yaml`, `model_envs.yaml`, `model_registry.yaml` | Benchmark scoring weights, dataset paths, keypoint maps, model env install profiles, model registry. |

### 3.10 `tests/` (107 tests)

P4: `test_cricket_track_manager.py`, `test_cricket_stitching.py`. P3:
`test_cricket_association.py`, `test_cricket_association_multiway.py`, `test_cricket_geometry.py`,
`test_cricket_geometry_cache.py`. P2: `test_cricket_tracking_metrics.py`, `test_tracking_jsonl_io.py`.
Kalman/contract/3D: `test_cricket_ground_kalman.py`, `test_cricket_contract_tracking_fields.py`,
`test_triangulation.py`, `test_prediction_schema.py`. CLI integration: `test_pipetrack_clis.py`.
Plus P1/Phase-0, metrics, keypoint maps, identity colours, UE export, benchmark CLI, repo/asset audits.

---

## 4. Data layout, conventions & calibration

- **World frame:** origin = pitch centre, **+Y** toward the Far End, **+X** lateral, **z = 0** = the
  pitch surface (verified: crease/stump-base landmarks sit at z≈0). Units = metres. The pitch is
  ~20.16 m long (stumps at Y = ±10.08, X = 0).
- **Cameras (positions from `Bundle_Adjusted_extrinsics.json`):** C1 behind the near end (Y≈−97), C4
  behind the far end (Y≈+108), C2/C3 on +X, C5/C6 on −X, C7 diagonal. All ~7× telephoto, image
  2560×1440, distortion ≈ 0.
- **Facing (co-observing) pairs — the key fact:** **C1↔C4, C2↔C6, C3↔C5** (and C7 independent). These
  are anti-parallel optical axes looking at the same ground strip; they are the high-overlap
  association backbone *and* the epipolar-degenerate pairs (so they rely on the ground plane, not
  triangulation). Auto-derived at runtime from calibration.
- **Capture groups:** `bt_01` = cams 1,4; `bt_02` = cams 2,5,7; `bt_03` = cams 3,6.
- **Deliveries (8):** `CCPL080626M1_1_14_1` … `CCPL080626M1_1_14_7`, `CCPL080626M2_1_12_1`.
- **Calibration files:** `drive/dataset/calibration-data/CCPL080626/calibration_data/` →
  `Bundle_Adjusted_extrinsics.json` (3×4 projection matrices + camera locations),
  `Bundle_Adjusted_intrinsics.json` (K), `pitch_calibration_config.json` (surveyed landmarks).
- **P1 outputs (ready):** `benchmarks/runs/rtmpose-l-body8-full-db32-pb96/` (preferred) and
  `benchmarks/runs/yolo26x-pose-full-db8/`.

---

## 5. How each stage works (methods)

**P2 — per-camera tracking.** Per camera, a constant-velocity Kalman + two-stage Hungarian (high-conf
with pose-appearance, then low-conf IoU) links detections over time and emits `local_track_id`. A pose
gallery handles brief occlusion. Output mirrors P1 with tracking fields filled.

**P3 — cross-camera association (per frame).** Each detection's foot is back-projected to the pitch
plane (`pixel_to_ground_xy`). Pairwise costs combine ground distance (primary), epipolar Sampson
distance (skipped for degenerate/facing pairs), and a small appearance term. A **constrained
single-linkage clustering** then merges detections into per-frame **player clusters**, accepting a merge
only when (a) no camera repeats, (b) the members' ground points agree, and (c) RANSAC triangulation of
all members reprojects within tolerance into every view (cycle-consistency). Each cluster becomes a
`Correspondence` with a consensus `ground_xy`, a `track_confidence`, and a `single_camera` flag.

**P4a — world-space online tracking.** P3 already did spatial association, so P4a is a clean temporal
multi-object tracker on the ground plane (Chen et al. CVPR'20 cross-view tracking; AI-City'24
geometric-consistency MTMC; EarlyBird BEV fusion). Per frame: predict every track (Singer Kalman) →
**Stage 1** attach a correspondence to the unique track that already owns one of its P2 tracklets
(revives lost/deleted tracks; if the foot projection is an outlier, keep identity but skip the position
update) → **Stage 2** χ²-gated Mahalanobis Hungarian for the rest → **Stage 3** re-entry from a deleted
pool (kinematically gated) or spawn a tentative track. Tracks confirm after `confirm_hits` (minting a
deterministic `P###`) and are deleted after a lost window (longer for bowlers). The correspondence→track
map is injective each frame ⇒ **no same-camera-frame ID collision is representable.**

**P4b — post-delivery stitching.** Confirmed tracklets are split into contiguous **segments**; a
min-cost-flow path cover (`linear_sum_assignment`) links segment ends across short gaps using temporal +
spatial + role + velocity-continuity costs, then `remap_ids` merges linked identities — but never when
the two histories share a `(camera, frame)` cell. This reduces fragmentation without risking collisions.

**Tools/libraries:** NumPy, SciPy (`linear_sum_assignment`, `expm`, `solve_discrete_lyapunov`), OpenCV
(image I/O + drawing), FFmpeg (libx264) for video, PyYAML, pytest, tqdm.

---

## 6. Visualization scripts & their outputs

All read a run folder's `predictions/*.jsonl` plus source frames under
`drive/dataset/<capture_group>/<delivery>/camera0N/`. Identity colours are **stable across cameras and
time** (a person keeps one colour everywhere).

**`render_phase1_videos.py`** — main video renderer (FFmpeg libx264, `--crf 22`, `--preset veryfast`,
yuv420p; OpenCV `mp4v` fallback via `--no-ffmpeg`). Key flags: `--run-dir`, `--delivery-id`,
`--show {p2|p3|p4}` (print local_track_id / cluster_id / **global_player_id**), `--fps` (25),
`--sample-every`, `--max-frames`, `--cameras`, `--keypoint-threshold` (0.2), `--mosaic-size`. Each person
is drawn with its COCO-17 skeleton, bbox, and an ID label + confidence in its identity colour;
un-IDed detections print `unassigned`. `--mode`:
- **`mosaic`** *(headline deliverable)* → `<run-dir>/visualizations/videos/<delivery>__all_cameras.mp4`
  (1920×1080): a 3×3 grid of the **7 camera tiles** + a **Delivery Monitor** panel (delivery, run,
  frame, camera/detection counts, threshold) + an **Active Global Roster** panel (current global IDs in
  their colours). Only all-camera-present frames are rendered (time-synced tiles).
- **`per-camera`** → `per_camera/<delivery>__<cam>.mp4` (1280×720) each, with optional ball trail from
  `drive/dataset/events-data/`.
- **`ground`** → `<delivery>__ground_tracks.mp4`: top-down pitch-plane view with per-identity track
  trails (reads `diagnostics/ground_tracks.jsonl` + pitch bounds).
- **`all`** → all of the above. A `video_manifest.json` records what was produced.

**`render_phase1_overlays.py`** — still-frame JPEG overlays for QA of selected frames
(`--frame-ids`/`--row-indices`, `--max-per-camera`, `--show`), written under
`<artifact_dir>/<group>/<delivery>/<cam>/frame_*.jpg` with a `visual_qa_manifest.json`.

**`identity_colors.py`** — `color_for_player(global_player_id, local_track_id)`: global IDs `P###` index
a fixed 12-colour BGR palette by `(N-1) % 12`; local/unknown IDs use a Blake2b hash. This is what keeps a
person's colour identical across all seven mosaic tiles and over time.

---

## 7. How to use the repo / run the scripts

**Environment + thread pinning (required):**
```bash
PY=/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/python   # NumPy>=1.23.5, SciPy>=1.10
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/aksh/quidich/Pose_Estimation
cd /home/aksh/quidich/Pose_Estimation
```

**Pipeline for one delivery** (P1 is already computed):
```bash
P1=benchmarks/runs/rtmpose-l-body8-full-db32-pb96
DELIV=CCPL080626M1_1_14_1

$PY -m scripts.tracking.run_per_camera_tracking \
  --input-run-dir $P1 --output-run-dir runs/p2 \
  --drive-root drive --delivery-id $DELIV --config configs/p2_tracking.yaml

$PY -m scripts.association.run_cross_camera_association \
  --input-run-dir runs/p2 --output-run-dir runs/p3 \
  --drive-root drive --delivery-id $DELIV --config configs/p3_association.yaml

$PY -m scripts.global_id.run_global_id \
  --input-run-dir runs/p3 --output-run-dir runs/p4 \
  --drive-root drive --delivery-id $DELIV --config configs/p4_global_id.yaml
```
Final per-camera records (`runs/p4/predictions/*.jsonl`) carry `global_player_id`. Check
`runs/p4/global_id_metrics.json`: `status: pass`, `same_camera_identity_collision_frames: 0`,
`distinct_global_id_count`.

**Render the 7-camera mosaic:**
```bash
$PY -m scripts.visualization.render_phase1_videos \
  --drive-root drive --run-dir runs/p4 --delivery-id $DELIV --mode mosaic --show p4 --fps 25
```

**Tests / optional 3D:**
```bash
OMP_NUM_THREADS=1 $PY -m pytest tests/ -q                     # expect: all passed
$PY -m scripts.export.triangulate_predictions \
  --input-run-dir runs/p4 --output-run-dir runs/p6 --drive-root drive --delivery-id $DELIV
```
To process all 8 deliveries, loop the three pipeline commands (and the mosaic render) over each
`delivery_id`; predictions accumulate by filename in the same output run folder.

---

## 8. Tasks completed

- [x] Diagnosed the full pipeline; root-caused the impossible-output bug, the dead P4b stitcher, and the
      camera-pair config error.
- [x] Corrected the facing-pair geometry and added calibration-driven **auto-derivation**
      (`derive_facing_pairs`) with a runtime override; verified vs `facing.jpeg`, optical-axis look-at,
      and landmark visibility (4–7 cm round-trip).
- [x] **Rebuilt P4a** as a single, auditable world-space ground-plane tracker; the same-camera-frame
      collision is now impossible by construction.
- [x] **Wired the real P4b** min-cost-flow stitcher; removed the backwards evidence-merge; moved contract
      validation to the final records.
- [x] **Fixed ID coverage** (untracked single-camera people now get IDs) → 100 % coverage on the sample.
- [x] Updated/added tests; **full suite 107 passed**.
- [x] Root-caused the P2 "hang" (BLAS oversubscription) → documented thread pinning.
- [x] Verified end-to-end on `CCPL080626M1_1_14_1`: 0 collisions, 100 % ID, 21 IDs (12 cross-camera).
- [x] Rendered a **sample mosaic** for human review.
- [x] Wrote this CHANGELOG (changelog + structure + files + methods + run guide + viz docs + status).

## 9. Tasks remaining / open issues

Correctness is solved and test-locked; the remainder is **quality/completeness** and **deliverables**:

1. **Human review of the sample mosaic** — the authoritative acceptance test (pending).
2. **Render all 8 deliveries** (Phase E) once the sample is signed off.
3. **ID fragmentation / over-segmentation** — ~21 IDs vs ~13–15 real people. Caused by (a) admitting
   untracked single-camera detections, (b) re-IDs after long losses, (c) limited P4b bridging. Levers:
   P4b gates/weights, `lost_window_frames`, fewer spurious single-camera spawns.
4. **Incomplete cross-camera linkage** — P3's per-frame clustering is somewhat unstable
   (`local_track_reassignment_conflicts_prevented` ≈ 2,700), so a person may carry different IDs in
   different cameras for stretches. Lever: P3 temporal stickiness / carry the 3D state into association.
5. **P2 under-tracking** — P2 fails to tracklet some persistent people (recovered geometrically by P4,
   but not root-caused).
6. **Single-camera ground accuracy** — single-view foot→z=0 assumes the foot is on the pitch plane;
   distant outfielders can have metres of error (IDs still produced, treat as lower-confidence).
7. **Appearance unused** — identical kit makes appearance re-ID near-useless; association is geometry-only
   (correct for cricket, but no cue for close crossings).
8. **No roles / no 3D in the main path** — `role` is always `unknown` (P5 dormant); `pose_3d` null unless
   the optional P6 export is run.
9. **No ground-truth validation** — metrics are geometry/fragmentation proxies, not MOTA/IDF1; a small
   labelled set would enable objective tuning (the runner accepts `--ground-truth`).
10. **Untuned parameters** — `confidence_discard`, χ² gates, lost windows, stitching weights are reasoned
    defaults, not swept against data.
11. **5 residual no-ID detections** in the sample — end-of-clip tentative tracks that never reach
    `confirm_hits`; expected, harmless.

### Deferred by request
- Aggressive `pose_estimation/` vs `scripts/` restructuring (revisit after pipeline validation).
- Role classification (P5).
