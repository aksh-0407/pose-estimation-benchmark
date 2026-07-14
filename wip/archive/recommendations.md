# Recommendations and Next Work

Companion to `wip/methods_log.md`. This file holds the forward-looking items (recommended
configuration and planned work) so the methods log stays a record of what was tested and measured.

## Recommended Configuration

### Location (accepted, recommended as default)

- Enable `ground_fusion_mode: z0_reproj`.
- Enable `p4a.emit_kalman_posterior`.
- Keep `foot_contact_mode: legacy` as the conservative default unless a downstream task specifically
  benefits from the v2 foot stack.

### Location (opt-in, not default)

- `foot_contact_mode: v2`.
- Single-camera ankle-height projection.
- `foot_smooth_window` greater than 1.
- Covariance-based measurement-noise modelling for future Kalman work.

### Identity (validated, held pending mosaic-review sign-off)

These flags reproduce the identity result reported in the methods log. They are kept out of the
committed configs until reviewed.

- P3 (`configs/p3_association_v5.yaml`): `graph_corrob_merge: true`, `graph_llr_merge_single: 1.2`,
  `graph_facing_gate_scale: 1.3`.
- P4 (`configs/p4_global_id_v5.yaml`): `p4a.min_emit_frames: 30`, `p4a.adaptive_lost_window: true`,
  `p4a.pose_gate_veto_distance: 0.30`, `p4a.reentry_pose_max_distance: 0.30`,
  `p4b.new_traj_cost_factor: 3.0`, `p4b.pose_stitch_max_distance: 0.30`, `p4b.w_pose: 2.0`.

### Rejected or no-action

- `robust_cov` as the emitted-position estimator.
- Colour-profile changes to the P1 detector and pose path.
- A trainable person-ReID embedding. There is no identity ground truth to train on, and geometry plus
  pose-shape carry the available signal.

## Next Work

### Identity (highest priority)

1. Promote the validated v5 identity flags into the committed configs after review.
2. Feed the billboard posture descriptor into the P4a teleport veto. It works on the facing pairs,
   unlike the triangulated descriptor, and is the most direct route to the residual M2 teleports.
3. Add cross-delivery prior calibration for anchor-starved hard clips: fit the cue distributions on the
   clean clips and reuse them as a prior when a hard clip has too few anchors.
4. Optional: hand-label a few hundred frames on `_7` and `M2` to report real IDF1 and ID-switch counts
   through `evaluate_ground_truth`.

### Model comparison

- RTMPose-X three-way study: RTMPose-L COCO-17 against RTMPose-X COCO-17 against RTMPose-X Halpe-26
  (with foot keypoints), once the full 8-delivery RTMPose-X P1 detections are available. Write-up in
  `wip/model_comparison.md`.

### Location and 3D pose follow-ups

- Single-view 3D pose reconstruction for single-camera frames.
- Distance-aware Kalman measurement noise if future filtering needs finer modelling.
- Offline trajectory smoothing for export-quality artifacts.
