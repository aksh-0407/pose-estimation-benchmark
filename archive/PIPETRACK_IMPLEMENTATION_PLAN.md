# PipeTrack — P2–P4 Re-ID + Tracking: Implementation Plan & Handoff

> **Implementation status (2026-06-29): COMPLETE.** P2--P4, proxy/optional-GT
> metrics, identity-stable visualization, ground-track rendering, and the light P6
> lift are implemented in the canonical layout below. The full repository suite
> passes in `cricket-yolo26x-pose`; a 600-frame, seven-camera delivery was also run
> through P2→P3→P4→P6. Accuracy remains a proxy claim until owned GT labels exist,
> as documented in §10.

> **Handoff doc for a fresh agent.** It is self-contained: architecture, decisions,
> what is already built (with verification), what remains (file-level), gotchas, and
> the exact commands/runtime to use. Read top-to-bottom once before coding.

---

## 0. RUNTIME — READ FIRST (hard blocker)

The default `python3` (`/usr/bin/python3`) has **numpy 1.21.5** + user-site **scipy 1.15.3**.
`scipy.linalg` imports, but **`scipy.optimize` (Hungarian / `linear_sum_assignment`) FAILS to
import** (`np.ndarray[...]` subscripting needs numpy ≥1.22). P2/P3/P4 all use
`linear_sum_assignment`, so they cannot run under `python3`.

**Use a conda env with numpy ≥1.23 (and <2 ideally):**

| Env | python | numpy | scipy.optimize | has | recommendation |
|---|---|---|---|---|---|
| `cricket-yolo26x-pose` | `/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/python` | 1.26.4 | ✅ | cv2, torch, ultralytics (also runs P1 + viz); **no pytest** | **Pipeline runtime** |
| `balltrack` | `/home/aksh/miniconda3/envs/balltrack/bin/python` | 2.2.6 | ✅ | — | numpy 2.x (riskier) |
| system `python3` | `/usr/bin/python3` | 1.21.5 | ❌ | pytest, numpy, scipy.linalg, yaml | tests of non-`scipy.optimize` code only |

**Do this once** so one interpreter runs both pytest and the pipeline:
```bash
/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/pip install pytest
```
Then use `PY=/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/python` for everything
(`"$PY" -m pytest tests/…`, `"$PY" -m scripts.…`). Also add a compatible pin to
`requirements.txt` (e.g. `numpy>=1.23.5,<2` and `scipy>=1.10`) — currently `scipy` is
listed unpinned and `numpy` is unpinned.

All commands below assume `PY` is set to a working interpreter.

---

## 1. Context & goal

Quidich Group-1 cricket CV pipeline: 7 calibrated, synchronized broadcast cameras →
player tracks. Phases (`presentables/grp_1/CORE_TASKS.md`, `plan.md`):

- **P0** foundation (done) · **P1** per-camera detection + 2D pose (done: YOLO26x + RTMPose).
- **P2** per-camera tracking → `local_track_id` (temporal re-ID). *Already exists* in
  `scripts/tracking/` (uncommitted, git-untracked). It is the input to P3 and the **style template**.
- **P3** cross-camera association → correspondences. **Geometry-first** (identical kits ⇒ appearance re-ID is weak).
- **P4** global ID + tracklet stitching → one `global_player_id` per player (P4a online Kalman + P4b min-cost-flow repair).
- **P5** roles · **P6** 3D lift + filter · **P7** validation — out of scope (P6 kept light at the end).

**Goal of this work:** build canonical, professional P2–P4 in the **main repo** delivering
**accurate re-ID + tracking**, structured as the modular "PipeTrack" pipeline, then update
visualization to render it.

**"PipeTrack" = the modular pipeline-based tracking PARADIGM** (not a library):
`2D pose → intra-view tracking → cross-view geometric association → 3D lift → 3D trajectory filter`.
This is exactly P1→P4(→P6) and validates Vedant's geometry-first design. Adopt the paradigm; do not add a dependency.

### Reference verdicts (from user's manager / materials)
| Reference | Verdict | Use |
|---|---|---|
| Modular pipeline paradigm | **Adopt** | The architecture itself. |
| **mvpose** (zju3dv) | **Algorithm ref** | Multi-way matching + **cycle-consistency** to harden P3 (vs pairwise-anchor). Old code → borrow algorithm only. |
| mv3dpose, Anipose, filterpy | Optional refs / building blocks | Temporal smoothing & robust triangulation (P6). Not deps now. |
| VoxelPose / VoxelTrack | **Reject** | Voxel grid scales cubically — impractical for ~150 m field + telephoto + sparse players. |
| DuoMo (Meta '26) | **Reject** here | Monocular world-space mesh diffusion, no multi-cam calibration. |

### Decisions confirmed with user
1. Build P2–P4 in the **MAIN repo** (`scripts/` + `pose_estimation/`); `vedant/` is **read-only reference**.
2. **Reuse Vedant's P3+P4 logic** but **re-home to main-repo structure/conventions/professionalism** — do not copy Vedant's layout/style.
3. Geometry-first stays; **add mvpose-style cycle-consistency** to P3 (done — see §4).
4. **Update visualization at the end** for `local_track_id` (P2), correspondences (P3), and identity-stable `global_player_id` colours across the 7-cam mosaic + ground-track view.

---

## 2. Dataset / calibration facts (verified)

- Match `CCPL080626`; cams `cam_01..cam_07`; 600 frames/cam @ 2560×1440; multiple deliveries (e.g. `CCPL080626M1_1_14_1`).
- Capture-group→camera map (`pose_estimation/cricket/dataset.py`): bt_01={01,04}, bt_02={02,05,07}, bt_03={03,06}.
  P3/P4 are cross-camera: read **all** bt_* files for a delivery, key by `cam_NN`, but **preserve each record's `capture_group`** so canonical filenames round-trip.
- Calibration under `drive/dataset/calibration-data/CCPL080626/calibration_data/`:
  `Bundle_Adjusted_extrinsics.json` (3×4 projection matrices + `camera_locations`), `Bundle_Adjusted_intrinsics.json`, `pitch_calibration_config.json`, `CPL08626_coord_aligned.csv` (survey points).
- Coordinate frame: origin = pitch centre; Y along pitch (toward Far End), X perpendicular, Z up. C1 behind Near End, C4 behind Far End, C2/C3 left, C5/C6 right, C7 diagonal. (So opposing pairs like C1↔C4 are degeneracy candidates.)
- Existing P1 run on disk to use as input: `benchmarks/runs/rtmpose-l-body8-full-db32-pb96/` (predictions/diagnostics).
- Run layout (clone it): `benchmarks/runs/<run_id>/{predictions,diagnostics}/` with canonical filenames
  `<capture_group>__<delivery_id>__<cam_NN>.jsonl`, e.g. `bt_01__CCPL080626M1_1_14_1__cam_01.jsonl`.
- JSONL schema = `g1_player_frame/v0` (`pose_estimation/cricket/contract.py`).

---

## 3. Critical findings / gotchas (do not relearn the hard way)

1. **Contract** (`pose_estimation/cricket/contract.py`): 3D field is `pose_3d.keypoints_world_m`
   (NOT `keypoints_world`). `validate_player` **ignores unknown keys** but `validate_pose_2d`
   was unconditional. **Already extended** (see §5 done-list): `track_state ∈ {confirmed,lost,tentative}`,
   `single_camera` bool, `local_track_id` str|null, `track_confidence ∈ [0,1]`, and `pose_2d`/bbox may be
   null **only** when `track_state ∈ {lost,tentative}`.
2. **Our pipeline never emits null-pose players** on the happy path (P2/P3/P4 only annotate real
   detections; "lost" is an internal lifecycle state, not an emitted record). The null-pose relaxation is
   forward-compat only.
3. **Vedant's "cycle consistency" is fake**: `p3_association.associate_frame` does anchor→partner Hungarian
   then `np.mean(valid_xy)` over partners — no B↔C loop closure. This is the defect the multiway upgrade fixes.
4. **Vedant `_compute_calibration_stats` returns hard-coded `(15.0, 5.0)`** (perfect survey reprojection ⇒ 0
   error). Replaced by config `mu_fine_score`/`sigma_fine_score` → `huber_delta = mu + 1.645·sigma`.
5. **Vedant `p4b_flow.py` imports `networkx` only as a passive edge container**, then solves with
   `scipy.optimize.linear_sum_assignment`. **Drop networkx** when porting (not installed; don't add it).
6. **Vedant `p4a_tracker` uses a process-global `_ID_COUNTER`** → breaks determinism across deliveries and
   under process pools. Use a **per-`TrackManager` instance counter**, reset per delivery (`P001…`).
7. **Frame-index semantics**: P2 drives its tracker on a per-camera ordinal (0..N) but **writes the true
   `frame_index`**. P3/P4 must group across cameras on the true synchronized `frame_index`.
8. **Degenerate pairs**: near-collinear/opposing pairs make epipolar geometry unreliable; cache zeroes
   `w_epi` for them and leans on triangulation+parallax. Config allows an explicit `degenerate_pairs` override.
9. **Ground-truth labels have no owner and no targets** (both briefs). All accuracy numbers are
   geometry-self-consistency **proxies** + visual QA until labels exist. Build a `--ground-truth` hook for later.
10. **`scripts/tracking/` is git-untracked.** It's P3's input and the template. Recommend committing it
    (with `configs/p2_tracking.yaml`) so the pipeline is reproducible. (Don't reach into Vedant's package.)
11. **Reuse, do not re-port**: triangulation (`pose_estimation/triangulation.py`) and calibration
    (`scripts/tracking/calibration.py`, `pose_estimation/cricket/calibration.py`) already exist and are better
    than Vedant's duplicates. Delete Vedant's `calibration.py`/`contract.py` concepts (already-merged).

---

## 4. Target module layout

Repo convention: **stateless math → library `pose_estimation/cricket/`**; **stateful engines + CLI/IO/run-artifacts → `scripts/`** (mirror `scripts/tracking/`). All packages use explicit `__init__.py`.

```
pose_estimation/cricket/
  geometry.py        # DONE — F-matrix, epipole, Sampson, parallax, condition #, huber, bbox_bottom_center,
                     #        camera_center_from_P; triangulate_dlt delegates to triangulation.triangulate_point_dlt
  ground_kalman.py   # DONE — SingerGroundKalman (public KF attrs x,P,F,Q,R,H) + RoleParams + ROLE_PARAMS
  contract.py        # DONE (extended in place)
  tracking_metrics.py# DONE — ID-switch / completeness / association proxies (unit-testable)

scripts/association/                       # P3 — cross-view geometric association
  __init__.py                              # DONE
  config.py                                # DONE — P3AssociationConfig + load_association_config
  geometry_cache.py                        # DONE — build_geometry_cache → GeometryCache (21 pairs, degeneracy, weights)
  associator.py                            # DONE — Detection3, Correspondence, associate_frame (pairwise_anchor + multiway_cycle)
  jsonl_io.py                              # DONE — group P2 jsonl by frame; write P3 jsonl + correspondences.jsonl + diagnostics
  runner.py                                # DONE — discover/run/write artifacts
  run_cross_camera_association.py          # DONE — canonical CLI

scripts/global_id/                         # P4 — global ID + stitching
  __init__.py                              # DONE
  config.py                                # DONE — P4Config (nested p4a:/p4b:)
  global_track.py                          # DONE — GlobalTrack dataclass + lifecycle
  track_manager.py                         # DONE — P4a online manager
  stitching.py                             # DONE — P4b segments + assignment solver, NO networkx
  jsonl_io.py / runner.py / run_global_id.py  # DONE

configs/
  p2_tracking.yaml      # DONE (mirrors TrackingConfig defaults)
  p3_association.yaml   # DONE — P3AssociationConfig values
  p4_global_id.yaml     # DONE — P4Config values

scripts/visualization/
  identity_colors.py    # DONE — color_for_global_id(gid) stable mapping→palette
  render_phase1_overlays.py / render_phase1_videos.py  # DONE — identity colours + correspondences + ground minimap
```

### The multi-way P3 algorithm (already implemented in `associator.py`)
Mode flag `matching_mode: {pairwise_anchor | multiway_cycle}` (default `multiway_cycle`).
**multiway_cycle** = geometry-guided constrained agglomerative clustering:
1. For every present camera pair, `build_cost_matrix` (ground-plane DLT + Sampson epipolar + parallax-weighted
   triangulation, soft Huber, dummy no-match row/col) → Hungarian → accepted edges (cost < dummy).
2. Single-linkage union-find over detections, processing edges cheapest-first. A merge is allowed **only if**:
   (a) the two clusters share **no camera** (≤1 detection per camera per cluster), and
   (b) the foot triangulated (`ransac_triangulate_point`) over **all** merged members reprojects within
   `cycle_reproj_tol_px` into **every** member view (RANSAC = cycle-consistency substitute; a wrong A–C edge
   makes that member an outlier and is rejected), and
   (c) existing cluster centroids agree within `cycle_xy_tol_m`.
3. Clusters = physical players (1..7 members). `ground_xy` = RANSAC foot triangulation;
   `track_confidence = clip(1 - mean_reproj/cycle_reproj_tol_px,0,1) · min(n_views,4)/4`; single-cam → `single_camera_confidence`.
`pairwise_anchor` mode = Vedant's sticky-anchor star matching, kept for A/B + fallback; shares the same
cluster-finalization path. **Note:** I intentionally did NOT port Vedant's `compute_track_confidence` sigmoid
(it capped at 0.5 with no runner-up and would leave clusters perpetually "low-confidence"); the view-count ×
reprojection formula above is monotonic, in [0,1], and configurable. `mu/sigma_fine_score` still drive `huber_delta`.

**Key data contracts in `associator.py`** (the IO + P4 depend on these):
- `Detection3(cam_id, player_index, bbox_xywh_px, keypoints_px(17,2), keypoint_conf(17,), confidence, local_track_id)`.
  `player_index` = index into the frame record's `players[]` (for write-back).
- `Correspondence(cluster_id, members: dict[cam_id→Detection3], ground_xy(2,), track_confidence, single_camera)`.
- `associate_frame(dets_per_cam, proj_matrices, geo, anchor: AnchorState|None, config) → (list[Correspondence], AnchorState)`.
  Maintain `anchor` across frames in the runner (sticky hysteresis stabilizes cluster_id ordering).

---

## 5. STATUS — completed and verified

### ✅ Step 1 — Contract + deps + P2 config  (verified: 29 tests pass under system python3)
- **Modified** `pose_estimation/cricket/contract.py`: added `TRACK_STATE_VALUES`; extended `validate_player`
  (track_state enum, single_camera bool, local_track_id str|null, `track_confidence∈[0,1]`, null pose/bbox iff
  lost/tentative); added `track_state`/`single_camera` to `example_group1_frame`. Backward compatible (P1 records still validate).
- **Created** `configs/p2_tracking.yaml` (mirrors `scripts/tracking/config.TrackingConfig` defaults).
- **Created** `tests/test_cricket_contract_tracking_fields.py`.
- **Modified** `requirements.txt`: added `scipy` (still needs a numpy/scipy pin — see §0).

### ✅ Step 2 — Library primitives  (verified: 23 tests pass under system python3)
- **Created** `pose_estimation/cricket/geometry.py` (ported Vedant `p3_geometry.py`; `triangulate_dlt`
  delegates to `triangulation.triangulate_point_dlt`; added `camera_center_from_P`).
- **Created** `pose_estimation/cricket/ground_kalman.py` (ported Vedant `p4a_kalman.py`; **public** KF attrs
  `x,P,F,Q,R,H` so the track manager never reaches into privates; `role_params`/`dt` injectable).
- **Created** `tests/test_cricket_geometry.py`, `tests/test_cricket_ground_kalman.py`.

### ✅ Step 3 — P3 algorithm and synthetic multi-way tests
- **Created** `scripts/association/__init__.py`, `config.py`, `geometry_cache.py`, `associator.py` (see §4).
- **Verification added for step 3:**
  - `tests/test_cricket_geometry_cache.py` — 21 pairs built; degeneracy flag (epipole-in-image OR small
    baseline OR explicit `degenerate_pairs`); weights; `huber_delta = mu+1.645σ` from config.
  - `tests/test_cricket_association.py` — `select_anchor` hysteresis (no flap unless margin exceeded);
    `build_cost_matrix` shape (M+1,N+1), finite, dummy corner 0; `_foot_pixel` ankle vs bbox-bottom.
  - `tests/test_cricket_association_multiway.py` — **synthetic 3-cam scene** (build P = K[R|t] like
    `tests/test_cricket_geometry.py::_make_cameras`): assert a deliberately wrong A–C edge is dropped by
    cycle-consistency, one cluster per physical player, ≤1 member per camera, correct RANSAC ground XY.
  Run: `"$PY" -m pytest tests/test_cricket_association*.py tests/test_cricket_geometry_cache.py -q`.

### ✅ Step 4 — P3 runner + IO
- `scripts/association/jsonl_io.py`:
  - `read_prediction_frames(path)` (reuse pattern from `scripts/tracking/jsonl_io.py`).
  - Build `dets_per_cam` per synchronized `frame_index`: for each record's `players[]`, make `Detection3`
    with `player_index`, `keypoints_px` from `pose_2d.keypoints_px`, `keypoint_conf` from `pose_2d.confidence`,
    `bbox_xywh_px`, `confidence` = `detection_confidence`, `local_track_id`. Skip players with null/short pose.
  - Write predictions through **unchanged** except set `single_camera` + `track_confidence` on each
    player (by `player_index`); leave `global_player_id` null. Validate each with `validate_group1_frame`.
  - Write `diagnostics/correspondences.jsonl`: one row/frame `{frame_index, clusters:[{cluster_id,
    members:[{cam_id, player_index, local_track_id, bbox_xywh_px}], ground_xy, track_confidence, single_camera}]}`.
- `scripts/association/runner.py`: clone `scripts/tracking/runner.py` — copy `CANONICAL_PREDICTION_RE`,
  `parse_prediction_filename`, `infer_match_id`, `discover_prediction_files` (verbatim — already perfect).
  Load calibration once via `scripts/tracking/calibration.load_projection_matrices_from_drive(drive_root, match_id)`
  (+ optionally `pose_estimation/cricket/calibration.load_survey_points`); `build_geometry_cache`; iterate frames
  in sorted `frame_index`, maintain `AnchorState`; write predictions + diagnostics + `run_manifest.json`
  (schema `association_run/v1`) + `association_metrics.json` (cluster counts, single-camera rate, mean/p-tile
  track_confidence, degenerate-pair usage, anchor-switch frames). **Run single-process per delivery** (cross-camera,
  frame-sequential) — no ProcessPoolExecutor.
- `scripts/association/run_cross_camera_association.py`: CLI clone of `run_per_camera_tracking.py`
  (`--input-run-dir --output-run-dir --drive-root --delivery-id --config --camera --expected-frames`).
- `configs/p3_association.yaml`: all `P3AssociationConfig` fields (see §6).
- **Verify:** run P3 on `CCPL080626M1_1_14_1` from the existing P1/P2 run; inspect `correspondences.jsonl`.

### ✅ Step 5 — P4a online track manager
- `scripts/global_id/global_track.py` ← `p4a_track.py`: `GlobalTrack` dataclass + lifecycle
  (`mark_missed`, `apply_hit`, `maybe_confirm`, `should_delete`, role-latch fields, `velocity_toward_crease`).
  Move magic numbers (`_CONFIRM_HITS`, `_LOST_WINDOW`, `_BOWLER_LOST_WINDOW`) to `P4Config`. Add `single_camera`
  and `local_track_ids_by_cam` fields so the writer can stamp every camera. Keep `velocity_toward_crease` as a
  **dormant P5 hook** (no caller yet).
- `scripts/global_id/track_manager.py` ← `p4a_tracker.py`: `TrackManager` (predict-all → two-stage Mahalanobis
  Hungarian against Confirmed+Lost then unmatched → re-entry pool → new tentative; `finalize`). Uses
  `pose_estimation.cricket.ground_kalman.SingerGroundKalman` (public `.R`/`.P` — **do not** reach into privates).
  **Replace process-global `_ID_COUNTER` with a per-instance counter** (`P001…`, reset per delivery). Gates/
  `frame_rate`/`v_max`/role params from `P4Config`. `propose_role` stays a dormant P5 hook.
- `scripts/global_id/config.py`: `P4Config` (frozen dataclass, nested `p4a`/`p4b`), same loader pattern as
  `scripts/association/config.py`.
- Tests: `tests/test_cricket_ground_kalman.py` already covers the filter; add
  `tests/test_cricket_track_manager.py` (tentative→confirmed after `confirm_hits`; re-entry revives a deleted id
  within gates; kinematic-infeasible re-entry rejected; **deterministic per-delivery id reset** — guards the counter fix).

### ✅ Step 6 — P4b stitching + P4 runner
- `scripts/global_id/stitching.py` ← `p4b_flow.py`: `Segment`, `extract_segments`, role/velocity-continuity/
  spatial/temporal costs, `solve_flow` (`scipy.optimize.linear_sum_assignment`), `remap_ids` + switch report.
  **Remove networkx** (replace the `nx.DiGraph` container with `build_link_costs(segments, cfg) → list[Edge]`).
  **Never mutate the contract record** with a `_ground_xy` key — keep ground XY in a side table keyed by
  `(cam, frame, track)` / `(global_id, frame)`. Simplify the dead `find_root`/`seen` loop in `remap_ids`.
  `_INCOMPATIBLE_ROLE_PAIRS` stays a dormant P5 hook.
- `scripts/global_id/jsonl_io.py`, `runner.py`, `run_global_id.py`: input = the **P3 run dir** (predictions +
  `correspondences.jsonl`). P4a online (one `TrackManager`/delivery) → stamp `global_player_id`/`track_state`/
  `track_confidence` on each member record by `player_index`; `finalize()`; P4b post → `id_switch_report.json`;
  backfill ids; validate each record (`final_handoff=False`). Manifest schema `global_id_run/v1`,
  metrics `global_id_metrics.json`.
- `configs/p4_global_id.yaml`.
- **Verify:** full P2→P3→P4 on one delivery; `cat global_id_metrics.json id_switch_report.json`.
- Tests: `tests/test_cricket_stitching.py` (segments from contiguous confirmed runs; role/velocity costs;
  feasible links solved, infeasible left unlinked; `remap_ids` merges to earliest id) + CLI smoke tests
  (`tests/test_association_cli.py`, `tests/test_global_id_cli.py`) on a tiny fabricated run dir (assert canonical
  filenames, manifest schema, metric keys, `id_switch_report.json` exists).

### ✅ Step 7 — Visualization
- `scripts/visualization/identity_colors.py`: `color_for_global_id(gid)` deterministic hash → fixed palette
  (extend existing `PLAYER_COLORS`), so the **same `global_player_id` is the same colour in all 7 mosaic tiles**.
- `render_phase1_overlays.py`: box colour by `color_for_global_id` (fallback local_track_id hash → grey); drop the
  literal `"(to be implemented)"`; show real `global id / local track id / track state / single_camera / track confidence`;
  add `--show {p2,p3,p4}` (p3 reads `correspondences.jsonl`, draws cluster badge per member).
- `render_phase1_videos.py`: stop colouring by per-frame `player_index`; colour by `global_player_id`; chip label =
  actual id; mosaic → same player same colour across tiles automatically + active-roster legend. Add
  `render_ground_tracks`: top-down pitch minimap (extents from `pitch_calibration_config.json`) drawing each
  global id's foot-trail in its identity colour (standalone `<delivery>__ground_tracks.mp4` or extra panel).
- Resolve which stage to render from the input run's `run_manifest.json` `task` field; bump viz manifest schema versions.

### ✅ Step 8 — Metrics
- `pose_estimation/cricket/tracking_metrics.py` (unit-testable; mirror `pose_estimation/metrics.py`); wire into
  the runners' metrics JSON. Intra-cam ID-switch proxy (P2 diagnostics), cross-cam ID-switch proxy
  (`id_switch_report` merges + distinct-id-vs-roster), track completeness (confirmed-frames/span, cameras/cluster),
  P3 association proxies (cluster reprojection residual via `reprojection_errors_for_point`, cycle-consistency rate,
  single-camera rate, anchor-switch frequency). Add optional `--ground-truth <jsonl>` hook in the P4 runner
  (`frame_index, camera_id, bbox, gt_id`) so IDF1/MOTA (`motmetrics`-style) drops in later.

### ✅ Step 9 — P6 (light)
- Adapt to group detections by `global_player_id` (not generic id) and write `pose_3d.keypoints_world_m` +
  per-joint `mean_reprojection_error_px` via `triangulation.triangulate_skeleton_ransac` + `confidence_ema_smooth`.
  Note Anipose/filterpy as future temporal-filter upgrade (no dep now).

---

## 6. Config strategy

Loader pattern = `scripts/tracking/config.py` exactly (frozen dataclass, `get_type_hints` validation, reject
unknown keys, `to_dict()` for manifest, `__post_init__` range checks). `P3AssociationConfig` is already written
(`scripts/association/config.py`). Still to create:

- `configs/p3_association.yaml` — emit all `P3AssociationConfig` fields. Key ones: `matching_mode`
  (`multiway_cycle`), `image_w/h`, `baseline_angle_degen_deg`, `degenerate_pairs` (verify opposing pairs from
  extrinsics), `w_epi/w_tri`, `parallax_min_deg/full_deg`, `mu_fine_score/sigma_fine_score`, `dummy_cost_scale`,
  `ankle_conf_min`, `keypoint_match_conf_min`, `cycle_xy_tol_m`, `cycle_reproj_tol_px`,
  `triangulation_min_views/_reproj_threshold_px`, `chi2_gate_2dof`, `confidence_high/discard`,
  `single_camera_confidence`, `anchor_hysteresis_margin/frames`, `anchor_priority`.
- `configs/p4_global_id.yaml` → `P4Config(p4a:/p4b:)`: `frame_rate_fps`, `kinematic_v_max_mps`; P4a `confirm_hits`,
  `lost_window_frames`, `bowler_lost_window_frames`, `chi2_gate_2dof`, re-entry temporal/Mahalanobis gates,
  `role_latch_frames`, per-role params (alpha/sigma_a/measurement_noise), `cap_max_pos_var`; P4b `temporal_gate_frames`,
  `w_temporal/w_spatial/w_role`, `new_traj_cost_factor`, `velocity_continuity_weight`, `incompatible_role_pairs`.

---

## 7. Data flow & CLI

```
P1 run (predictions/<grp>__<del>__<cam>.jsonl; ids null)
 → scripts.tracking.run_per_camera_tracking        (EXISTING P2)  + local_track_id
 → scripts.association.run_cross_camera_association (NEW P3)       + single_camera, track_confidence; correspondences.jsonl
 → scripts.global_id.run_global_id                 (NEW P4=4a+4b)  + global_player_id; id_switch_report.json
 → scripts.export.triangulate_predictions          (P6, light)     + pose_3d.keypoints_world_m
```
P3 and P4 are **separate runs with a JSONL handoff**, mirroring P1→P2. Both CLIs clone
`run_per_camera_tracking.py`'s arg shape.

---

## 8. Verification recipe (with a working env)

```bash
PY=/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/python    # numpy 1.26.4; pip install pytest into it first

# unit tests (after writing the remaining test files)
"$PY" -m pytest tests/ -q

# end-to-end on one delivery
"$PY" -m scripts.tracking.run_per_camera_tracking \
  --input-run-dir benchmarks/runs/rtmpose-l-body8-full-db32-pb96 \
  --output-run-dir benchmarks/runs/p2-CCPL080626M1_1_14_1 \
  --drive-root drive --delivery-id CCPL080626M1_1_14_1 --config configs/p2_tracking.yaml
"$PY" -m scripts.association.run_cross_camera_association \
  --input-run-dir benchmarks/runs/p2-CCPL080626M1_1_14_1 \
  --output-run-dir benchmarks/runs/p3-CCPL080626M1_1_14_1 \
  --drive-root drive --delivery-id CCPL080626M1_1_14_1 --config configs/p3_association.yaml
"$PY" -m scripts.global_id.run_global_id \
  --input-run-dir benchmarks/runs/p3-CCPL080626M1_1_14_1 \
  --output-run-dir benchmarks/runs/p4-CCPL080626M1_1_14_1 \
  --drive-root drive --delivery-id CCPL080626M1_1_14_1 --config configs/p4_global_id.yaml
"$PY" -m scripts.visualization.render_phase1_videos \
  --drive-root drive --run-dir benchmarks/runs/p4-CCPL080626M1_1_14_1 \
  --delivery-id CCPL080626M1_1_14_1 --mode mosaic --max-frames 200 --show p4
```
**Pass criterion:** in the mosaic each physical player keeps one colour across all 7 tiles and the whole clip;
the bowler keeps its id through run-up/occlusion; the ID-switch report count is low and its merges look correct
on the ground-track minimap.

---

## 9. Source map (port FROM `vedant/`, reuse from main)

**Port from `vedant/Quidich/pose_estimation/cricket/`** (adapt to main-repo style):
`p4a_tracker.py`, `p4a_track.py` (→ `scripts/global_id/`), `p4b_flow.py` (→ `scripts/global_id/stitching.py`).
P3 `p3_association.py`/`p3_precompute.py`/`p3_geometry.py` and `p4a_kalman.py` are **already ported**.
Blueprint for runners: `vedant/Quidich/scripts/run_cricket_p3_p4.py` (split into P3 + P4 runners).
Tests to port/adapt: `vedant/Quidich/tests/test_cricket_p4a_tracker.py`, `test_cricket_p4b_flow.py`,
`test_cricket_p3_association.py`, `test_cricket_p3_p4_contract.py`.

**Reuse from main repo (do NOT re-port):**
- `pose_estimation/triangulation.py` — `triangulate_point_dlt`, `ransac_triangulate_point`,
  `triangulate_skeleton_ransac`, `reprojection_errors_for_point`, `confidence_ema_smooth`.
- `scripts/tracking/calibration.py` — `load_projection_matrices_from_drive`, `build_ground_calibrators`,
  `current_calibration_dir` (C1..C7→cam_01..07, ground homography).
- `pose_estimation/cricket/calibration.py` — `load_survey_points`, `audit_calibration` (run as a P3 pre-flight).
- `scripts/tracking/runner.py` — discovery + run-artifact pattern (clone).
- `scripts/tracking/jsonl_io.py` — `read_prediction_frames`.
- `scripts/tracking/pose_vector.py` — `masked_weighted_cosine` (optional pose tie-break).
- `scripts/visualization/render_phase1_*.py` — drawing utils (`draw_chip`, `blend_rect`, `draw_players`,
  `render_feed_frame`, `VideoSink`, `resolve_delivery_camera_dirs`, `COCO_17_EDGES`).

---

## 10. Risks
- **R1 GT labels (highest):** none owned, no targets → geometry proxies + visual QA + `--ground-truth` hook; qualify accuracy claims.
- **R2 Calibration drives P3:** one bad projection matrix silently merges players → run `calibration.audit_calibration` as a P3 pre-flight; gate on warnings.
- **R3 Uncommitted P2:** commit `scripts/tracking/` + `configs/p2_tracking.yaml` for reproducibility.
- **R4 Env (ACTIVE BLOCKER):** see §0 — use numpy ≥1.23 env; pin in `requirements.txt`.
- **R5 Frame-index:** group P3/P4 strictly on synchronized `frame_index`.
- **R6 Degenerate pairs:** verify `degenerate_pairs` against extrinsics; don't over-prune when only degenerate pairs connect two views.
- **R7 Vedant cruft:** keep P5 hooks dormant; replace fake stats with config (done for P3).

---

*The original approved plan also lives at*
`/home/aksh/.claude-accounts/tony/plans/go-thoroughly-through-presentables-encapsulated-swing.md`.
*This doc supersedes it with progress + the env discovery; prefer this one.*
