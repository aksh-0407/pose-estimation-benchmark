# Configuration reference

The pipeline is driven by YAML in `configs/`. Two families:

- **Model setup** — `model_registry.yaml`, `model_envs.yaml`, `keypoint_mappings.yaml`
  (which model P1 runs, how it's installed, how any skeleton reduces to COCO-17).
- **Pipeline stages** — `p2_tracking.yaml`, `p3_association.yaml`, `p4_global_id.yaml`
  (and their `_v5` variants) tune tracking, cross-camera association, and global identity.

Each stage's config is validated on load (unknown keys are rejected) by the stage's
`config.py` (`src/identity/p2_tracking/config.py`, `src/identity/p3_association/config.py`,
`src/identity/p5_global_id/config.py`).

---

## Model setup

### `model_registry.yaml`
The catalogue of candidate models (identity, skeleton, input size, checkpoint path,
strengths/risks) plus a `default_target` block (production target `nvidia_gpu` / `fp16` /
`coco_17` and the latency/reprojection/MPJPE budgets). P1 uses `rtmpose_x_body8`.

### `model_envs.yaml`
How each model is **installed and run**: a reusable install `profiles:` block (conda +
pip recipes per ecosystem) and a per-model `models:` block (conda `env_name`, mmpose
`config`, `checkpoint`, and downloadable `assets:` with optional `fallback_urls`).
RTMPose-X runs in `pose-lab` (mmpose 1.3.2 / mmcv 2.1.0 / mmdet 3.2.0 /
torch 2.1.0-cu121) with the shared RTMDet-m person detector.

### `keypoint_mappings.yaml`
Defines the canonical **COCO-17** target skeleton (joint names + edges) and, per native
skeleton, the index list reducing it to COCO-17. This is what lets Halpe-26 (RTMPose-X)
expose its first 17 as COCO-17 while keeping feet in `pose_2d_native`.

---

## `p2_tracking.yaml` — per-camera tracking

Governs how detections become per-camera tracklets. Notable keys:

| Key | Meaning |
|---|---|
| `stage1_score` / `stage2_score` | high/low detection-confidence split (ByteTrack two-stage). |
| `chi2_gate` | Mahalanobis gate on the box-centre Kalman (default 9.21, 2-DOF χ²). |
| `iou_alpha` / `pose_beta` | blend of IoU cost and masked pose-cosine cost in matching. |
| `gate_max_distance_px`, `gate_bbox_factor` | spatial gates. |
| `ground_gate_base_m`, `ground_vmax_mps` (9.0), `frame_rate_fps` (50) | calibrated ground-reachability gate. |
| `pose_cosine_reid_threshold`, `reid_ambiguity_margin` | dormant-track re-ID acceptance. |
| `tentative_confirm_hits` (3) / `_window` (5), `dormant_max_frames` (60), `pose_gallery_size` (30) | track lifecycle + pose gallery. |

## `p3_association.yaml` — cross-camera association

The largest config. Highlights:

| Key | Meaning |
|---|---|
| `association_mode` | `tracklet_graph` (default, offline, decides identity per tracklet-pair) or `multiway_cycle` (per-frame). |
| `opposite_camera_pairs` | the three **facing** (co-observing) pairs `cam_01↔cam_04`, `cam_02↔cam_06`, `cam_03↔cam_05`. These are low-parallax / epipolar-degenerate — the code zeroes their epipolar term and widens their gate. |
| `image_w` / `image_h` | global image size (2560×1440). **Note:** camera 07 is ~3775×960; per-camera size is recovered from intrinsics in code, but config paths using the global size mis-handle C07. |
| ground/epipolar/appearance weights, `ground_distance_gate_m` (3.5), `opposite_pair_ground_gate_m` (2.5) | pairwise cost + gates. |
| `graph_*` block | tracklet-graph LLR fusion: `graph_llr_merge_threshold` (2.0), `graph_llr_veto` (−4.5), `graph_hard_dist_gate_m` (2.75), hysteresis/attach gates. |
| `graph_corrob_merge`, `graph_llr_merge_single` (1.2), `graph_facing_gate_scale` (1.3) | **v5** additions that admit strong single-cue facing-pair merges (fix cross-camera under-merge). |
| `ground_fusion_mode` | `z0_reproj` (default emitted position) / `robust_cov` / `median`. |
| posture / synthetic-tracklet / approx-feet flags | pose-shape descriptor and dark-umpire handling. |

`configs/p3_association_v5.yaml` is the tuned generation used for the current identity
results; confirm which file the run loads before comparing metrics.

## `p4_global_id.yaml` — global identity + stitching

Two blocks:

- **`p4a`** (online MOT): `confirm_hits` (3), lost/`ownership_ttl_frames` (50) windows,
  `chi2_gate_2dof` (5.991), re-entry gates, `expected_roster_max` (15) with
  `roster_cap_min_separation_m` (3.0), `shadow_confirm_*`, per-role `role_params`
  (Singer α / σ_a / measurement noise), `emit_kalman_posterior: true`, and **v5** pose
  vetoes (`pose_gate_veto_distance`, `reentry_pose_max_distance` = 0.30) plus
  `adaptive_lost_window` / `lost_window_max_frames` (90).
- **`p4b`** (stitching): min-cost-flow link weights (`w_temporal`, `w_spatial`, `w_pose`,
  velocity continuity), `temporal_gate_frames` (120), `incompatible_role_pairs`,
  `pose_stitch_max_distance` (0.30), `new_traj_cost_factor`, and the `min_emit_frames`
  (30) cardinality prior.

The `_v5` variants differ materially (corroboration merge, pose vetoes, raised stitch
cost) — see [improving-models.md](improving-models.md) and
[critical-analysis/phase-4-global-id.md](critical-analysis/phase-4-global-id.md).

---

Calibration is **not** a YAML config — it is read from
`drive/dataset/calibration-data/<match>/calibration_data/{camera,pitch}_calibration_config.json`
(bundle-adjusted 3×4 projection matrices + pitch geometry).
