# Architecture & core concepts

The shared concepts every pipeline stage builds on: the camera rig, calibration, the data
contract, the skeletons, the run-directory layout, and the quality metrics. Read this once;
the [per-phase docs](pipeline/README.md) then make sense end-to-end.

## The problem

Turn synchronised **7-camera cricket broadcast footage** into, per player: **2D pose** in each
view, one **stable global identity** across all cameras and the whole delivery, a **3D pose &
ground location**, and a cricket **role**, then render a diagnostic **mosaic**. The end
consumer is an Unreal Engine graphics layer; downstream groups (broadcast biomechanics,
event/officiating) consume our per-player 3D + identity + role via the JSON contract below.

## The camera rig & facing-pair geometry

Seven cameras, captured in three groups (`src/core/frames.py`):

| Group | Cameras | Resolution |
|---|---|---|
| `bt_01` | cam_01, cam_04 | 2560×1440 |
| `bt_02` | cam_02, cam_05, cam_07 | 2560×1440 (cam_07 panoramic ~3775×960) |
| `bt_03` | cam_03, cam_06 | 2560×1440 |

The co-observing pairs are **facing pairs**, cameras viewing the *same* ground strip from
opposite sides: **C1-C4, C2-C6, C3-C5** (cam_07 unpaired). They are derived from calibration,
not hardcoded (`derive_facing_pairs`, `src/identity/common/geometry.py`): a mutual-best match on
anti-parallel optical axes with nearest ground look-at points. Facing pairs have **low-parallax,
near-degenerate epipolar geometry**, the epipolar cue is dropped when the baseline is degenerate
(`03-association`), which is the central hard problem of cross-camera identity on this rig.

## Calibration

Bundle-adjusted per-camera 3×4 projection matrices + intrinsics + pitch geometry load from
`data/raw/8_init/calibration-data/<match>/` (`src/core/calibration.py`; both matches share this one
session, so other datasets borrow it). Calibration is
**centimetre-accurate** (ball reprojection p95 ≤ 4.5 px), and the **pitch centre is the world
origin**. Because of that accuracy, identity and location are solved directly on the calibrated
**z = 0 ground plane** (the ground homography is `P[:, [0,1,3]]` inverted).

## The data contract (`g1_player_frame/v1`)

Every stage reads and writes one canonical JSONL record per camera-frame
(`src/core/contract.py`, `validate_group1_frame`). This is also the **hand-off surface** to the
other groups, they consume these fields, not our code.

- **frame**: `schema_version, match_id, delivery_id, camera_id (cam_NN), frame_index, frame_name, players[]`
- **player**: `global_player_id` (unique within a camera-frame; required at final hand-off),
  `local_track_id`, `role`, `track_state` (confirmed/lost/tentative), `single_camera`,
  `bbox_xywh_px/_norm`, `detection_confidence`, `track_confidence`, `pose_2d`, `pose_3d`
- **pose_2d**: `{skeleton: halpe26, keypoints_px[26][2], keypoints_norm[26][2], confidence[26]}`
- **pose_3d** (null until triangulation; per-joint nullable): `{keypoints_world_m[26][3], confidence[26], mean_reprojection_error_px[26]}`
- **pose_3d_named** (self-describing, added at the lift): `{root_joint, root_world_m[3], joints_root_relative_m{name:[dx,dy,dz]}}`, root in world metres, every joint relative to it.

## Skeletons

The pipeline skeleton is **Halpe-26** (`pose_2d` / `pose_3d`, 26 joints). Indices 0-16 are COCO-17
in COCO order; joints 17-25 add head/neck/hip-mid and big-toe/small-toe/heel L&R. Halpe feet drive
ground-contact estimation. Names + edges are in `src/core/keypoints.py` (`HALPE26_KEYPOINTS`,
`HALPE26_EDGES`); a labelled reference is `docs/reference/skeleton-halpe26.md`.

## Run-directory layout

Each stage consumes an `--input-run-dir` and writes an `--output-run-dir` holding
`predictions/*.jsonl` (+ `diagnostics/`, `*_metrics.json`, `run_manifest.json`), so any stage is
inspectable and re-runnable in isolation. The batch driver `src/main.py` lays a delivery out as:

```
data/derived/<dataset>/pipetrack_v<num>/<DELIVERY>/
  00_inference/  01_stabilization/  02_tracking/  03_association/  04_lift/
  05_global_id/  06_roles/  logs/
data/viz/<dataset>/pipetrack_v<num>/<DELIVERY>/*.mp4
```

## Environment

One conda env, **`pose-lab`** (mm-stack + PyTorch + numpy/scipy/opencv + ultralytics), runs
every stage. Everything is invoked as a module: `python -m main`,
`python -m identity.p2_tracking.run_per_camera_tracking`,
`python -m core.inference.run_phase1_rtmpose_inference` (`pip install -e .` puts `src/` on the path).

## Metrics & proxies (read jointly, never singly)

There is no per-player identity or 3D-location ground truth for this footage (and none is planned , 
the final quality judgement is human review of the rendered mosaics), so identity quality is read as
a **joint panel** of calibration/geometry-anchored proxies (`src/identity/common/metrics.py`):

- **cross_camera_agreement**, for different-camera detection pairs whose *independent*
  calibration ground projections coincide (< 1.5 m), the fraction sharing a `global_player_id`.
  (Uses calibration only, so it does not echo the clustering.)
- **distinct_global_id_count / excess_id_fragment_count**, IDs vs the ~13-15-person roster.
- **teleport_proxy**, IDs jumping faster than ~9 m/s between frames on the mean-over-cameras
  foot projection (noisy: dominated by single-camera grazing-foot error, see `diagnosis/03`).
- **colocated_identity**, distinct IDs co-located < 0.75 m for ≥ 25 frames with disjoint camera
  occupancy = one physical player carrying two IDs (the W9 ghost-swap tripwire).
- **same_camera_identity_collision_frames**, must be **0** (hard invariant, held by construction).
- **confirmed_frame_completeness** (id-persistence), **pair_link_churn**, **cycle_consistency**,
  **single_camera_rate**, association/tracking health.

Conventional labelled metrics (MOTA/IDF1/HOTA, MPJPE/P-MPJPE) are deliberately **not** used: no
ground truth exists for this footage and none is planned, so quality is decided by the proxy panel
above plus human mosaic review.

## The one-line causal chain

The dominant failure mode, end to end (measured, `diagnosis/09`):

> **P1** deep-field recall gap to single-camera players to **03** no binding cue on grazing/facing
> cameras to split identity + **05** ID over-mint to risky stitch + mean-of-fragments emission  to 
> emitted teleports + colour flicker; **06** drops the same frames to 3D coverage gaps.

Identity, not location, is the current quality ceiling. The prioritized fixes are in
[`roadmap.md`](roadmap.md); the measured 40-delivery diagnosis is in
[`diagnosis/`](analysis/README.md).
