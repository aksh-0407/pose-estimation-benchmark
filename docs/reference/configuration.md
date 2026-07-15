# Configuration reference

One numbered YAML per identity stage under `configs/`, matching the `src/identity/pN_<stage>/`
packages and the `0N_<stage>/` run-dir folders. `src/main.py` loads these by default; override
per stage with `--pN-config`. Every loader **rejects unknown keys** and validates ranges, so a
typo fails fast. Stage **04 (3D lift)** has no YAML — its parameters are CLI flags.

See each stage's doc for what a knob *does* and what's been tried: [../pipeline/](../pipeline/README.md).

## `01_stabilization.yaml`

`enabled` (byte-identical pass-through when false), `frame_rate_fps`, `smooth_native`;
`link.iou_min`, `link.max_gap_frames`; One-Euro `smoothing.{min_cutoff, beta, d_cutoff}`
(lower `min_cutoff` = smoother/more lag; higher `beta` = less lag on fast motion);
gating `confidence_min`, `max_jump_bbox_frac`, `max_jump_px`.

## `02_tracking.yaml`

`stage1_confidence_threshold`, `stage2_confidence_min`, `cost_accept_threshold`,
**`lowconf_can_spawn: false`** (v8 default — low-conf specks associate but never birth);
cost mix `iou_alpha` / `pose_beta`, `min_shared_keypoints`; gates `chi2_gate`,
`gate_bbox_factor`, `gate_max_distance_px`, `v_max_px_per_frame`; ground gating
`ground_vmax_mps`, `ground_gate_base_m`, `ground_cost_weight`, `ankle_confidence_min`;
dormant re-ID `pose_cosine_reid_threshold`, `dormant_max_frames`; lifecycle
`tentative_confirm_hits`/`_window`, `pose_gallery_size`, `gallery_repr`.

## `03_association.yaml`

Cue weights `ground_weight`, `epipolar_weight`, `appearance_weight`; gates
`ground_distance_gate_m`, `opposite_pair_ground_gate_m`, `ground_cluster_gate_m`;
tracklet-graph `graph_llr_merge_threshold`, `graph_llr_veto`, `graph_llr_positive_cap`,
`graph_min_covis_frames`, `binding_min_single_frames`, `graph_corrob_merge`,
`graph_facing_gate_scale`; ground fusion `ground_fusion_mode: z0_reproj`, `ground_var_floor_m`
(absorbs ~0.7–1.2 m cross-camera calibration bias); feet recovery `approx_feet_enabled`,
`synthetic_tracklets_enabled`; W9 union-lift `graph_union_lift_merge`, `graph_union_colocate_m`,
`graph_union_min_co_frames`; `foot_contact_mode: v3`, `graph_split_enabled`,
`calibration_mode: auto`. Many opt-ins gate off (`contested_iou`, `airborne_pelvis_emit`).

## `05_global_id.yaml`

**P4a** `confirm_hits`, `lost_window_frames` / `bowler_lost_window_frames`, `chi2_gate_2dof`,
`online_role_proxy`, `use_measurement_covariance` (asymmetric R: `r_floor_m`, `r_ceiling_m`),
`emit_kalman_posterior`, `ownership_ttl_frames`, `shadow_confirm_gate_m`, `expected_roster_max`,
`pose_match_weight`, `adaptive_lost_window`, `min_emit_frames`, per-`role_params`.
**P4b** stitching `enabled`, `occupancy_bridge` + `temporal_gate_frames_occupancy`,
`normalized_costs`, `w_spatial` / `w_temporal` / `w_role`, `new_traj_cost_factor`,
`incompatible_role_pairs`, `colocated_merge` + `colocated_radius_m` + `colocated_min_frames`.

## `06_roles.yaml`

`role_assignment_version: v1`, `min_track_frames`, `epoch_frames`, `role_epoch_latch_count`,
`role_assignment_max_cost`; suppression `suppression_enabled`, `suppress_min_kp_conf`,
`suppress_min_completeness`, `suppress_single_cam_det_conf`, `suppress_protect_umpires`.
v1 requires the 05 run to have `online_role_proxy: true`.

## Shared

- `keypoint_mappings.yaml` — COCO-17 + Halpe-26 skeleton definitions and the source→COCO-17 index
  maps (the pipeline skeleton is Halpe-26; names/edges in `src/core/keypoints.py`).
- `model_envs.yaml` / `model_registry.yaml` — the P1 model catalog (checkpoints, the `pose-lab`
  env, latency/reprojection budgets). P1 uses `rtmpose_x_body8`.
- `reference/{ground_conventions,camera_layout}.jpeg` — the ground-convention and camera-layout
  reference images (the association runner derives facing pairs from calibration, not these).
