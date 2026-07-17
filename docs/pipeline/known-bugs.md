# Known bugs and latent issues, pipeline tracker

A living register of defects and latent traps, cross-checked against the code
so nothing here is speculative. Distinct from the per-stage "Known issues"
sections (design limitations) and from [fixes-log.md](fixes-log.md) (the A/B
ledger of changes). The 2026-07-17 full-codebase audit's working register,
with per-finding evidence and dispositions, is
[docs/audit/bugs.md](../audit/bugs.md); its confirmed findings are folded in
here as BUG-9 onward.

Status: `open` | `pinned` (open, root cause identified) | `needs-verify` |
`resolved` | `not-a-bug`.
Severity: `S1` corrupts output or silently disables a feature | `S2` material |
`S3` minor.

| ID | Title | Sev | Status | Stage |
|---|---|---|---|---|
| [BUG-9](#bug-9) | Foot-contact shape guard: all ground points are bbox-bottom | S1 | open | 03 |
| [BUG-4](#bug-4) | Detector-recall bound (dark/distant subjects lost) | S1 | pinned | P1 |
| [BUG-5](#bug-5) | 02 constant-velocity model breaks under manoeuvre | S1 | pinned | 02 |
| [BUG-1](#bug-1) | `emit_kalman_posterior` active but ineffective teleport guard | S2 | pinned | 05 |
| [BUG-3](#bug-3) | Splittable clustering IS on but conservative; sub-threshold chimeras persist | S2 | pinned | 03 |
| [BUG-6](#bug-6) | Stitching silently under-merges | S2 | pinned | 05 |
| [BUG-7](#bug-7) | ~39% single-camera frames get no 3D pose | S2 | pinned | 04 |
| [BUG-10](#bug-10) | Track finalization bypasses the shadow/roster gates | S2 | open | 05 |
| [BUG-11](#bug-11) | Abandoned merge invariant: positive cap above merge threshold | S2 | open (documented) | 03 |
| [BUG-12](#bug-12) | Bowler slot cost uses absolute speed | S3 | open | 06 |
| [BUG-13](#bug-13) | Batch driver misclassifies stages that exit 1 | S2 | open | driver |
| [BUG-14](#bug-14) | Render always reads stage 05, ignoring 06/07 outputs | S2 | open | render |
| [NB-2](#nb-2) | Dataclass config defaults disagree with production YAML | S2 | open | all |
| [BUG-8](#bug-8) | 01 stabilization not wired into the default flow | S3 | open | 01 |
| [BUG-2](#bug-2) | Distance-blind Kalman `R` (RETRACTED; covariance-R is on in prod) | - | not-a-bug | 05 |
| [NB-1](#nb-1) | C07 global image size in config | - | not-a-bug | 03 |

---

## BUG-9, Every production ground point is the bbox bottom-centre {#bug-9}
**S1, open (report-only by owner decision 2026-07-17), stage 03 (also 02)**

- **Symptom:** none visible; that is the problem. All foot-contact refinement
  is silently inactive.
- **Root cause:** `ground_contact_pixel_ex`
  ([geometry.py](../../src/identity/common/geometry.py), shape guard near the
  top of the function) requires exactly 17 keypoints; since the Halpe-26
  migration every caller passes 26, so the function always returns the bbox
  bottom-centre. The `legacy` ankle mode, `v2`, and `v3` heel/toe modes never
  execute; `foot_contact_mode: v3` in `configs/03_association.yaml` is inert.
- **Impact:** the clustering gate positions, emitted feet and heights, and the
  z0-reprojection input all ride on bbox bottoms. Every published baseline
  (the 0.689 verdict, all 40-delivery numbers) was measured with this
  behavior, so fixing it is a behavior change requiring evaluation, not a
  cleanup.
- **Proposed fix (evaluated, not yet applied):** make the guard length-aware
  (17 or more rows; Halpe indices 0-16 are the COCO-17 subset so ankle indices
  15/16 remain valid), gate the emit path, the clustering path, and the
  tracking path behind three separate flags, and A/B each on the 8-delivery
  set before any production enablement. Downstream tuned thresholds
  (clustering gates in metres, the global-id `r_ceiling_m`,
  `confidence_discard`) were tuned on bbox-bottom grounds and may need a
  retune sweep after.
- Full evidence: [docs/audit/bugs.md BUG-A1](../audit/bugs.md).

## BUG-10, `finalize()` can mint ids the shadow gate would have blocked {#bug-10}
**S2, open, stage 05**

End-of-delivery promotion mints a global id for any tentative track with 2 or
more hits without running the shadow-duplicate and roster-cap gates that
normal promotion applies
([track_manager.py](../../src/identity/p5_global_id/track_manager.py),
`finalize`). A late shadow duplicate of a confirmed player can be minted as a
fresh id in the last frames of a delivery. Proposed fix: route finalize
through the same gates as `_promote_and_prune`; evaluate on the 8-delivery set
(id count, colocated-pair diagnostics).

## BUG-11, The "capped ground cue alone can never merge" invariant is abandoned {#bug-11}
**S2, open (documented in code), stage 03**

With the dataclass defaults the per-cue positive cap (1.5) sits below the
merge threshold (2.0), so a single ground cue cannot merge two tracklets. The
production YAML deliberately raises the cap to 3.5 (a measured association
improvement), which abandons that structural bound: a single strongly
supported ground cue can now clear the threshold alone. Merge safety rests on
support weighting and the confident-contradiction veto. The code comments
asserting the old invariant were corrected in the 2026-07-17 audit; recorded
here so the trade-off is not forgotten. Any cap retune is a separate evaluated
change.

## BUG-12, Bowler slot cost uses absolute speed {#bug-12}
**S3, open, stage 06**

`_windowed_axis_speed` was deliberately made signed so a sprint in the wrong
axis direction cannot look like a bowler run-up, but the v1 slot cost uses
`abs(speed)` ([assigner.py](../../src/identity/p6_roles/assigner.py)),
reintroducing the ambiguity at slot level. The two-direction axis trial in the
runner mitigates it. Proposed fix: sign-aware slot cost; evaluate on role
accuracy.

## BUG-13, Batch driver misclassifies stages that exit 1 {#bug-13}
**S2, open, batch drivers**

Exit code 1 is read as a "warn verdict" for stages 03 and 05 (a crash is
distinguished only by the missing metrics artifact); `id_pipeline.py` applies
the same convention to 03 and never gates 05's return code. Any other stage
(or unexpected exception path) exiting 1 is misread. Proposed fix: reserve a
distinct exit code (for example 3) for warn-verdicts across stage CLIs and
both drivers in one change. Until then the convention is documented in
[08-export-and-render.md](08-export-and-render.md).

## BUG-14, Render always reads stage 05 {#bug-14}
**S2, open, render**

`src/main.py run_render` renders from `05_global_id` even when `06_roles`
(role stamps, suppression) and `07_refine` (physically constrained 3D)
produced the terminal predictions. Roles and suppression reach the mosaic via
side files, but refined 3D never reaches the render. Proposed fix: render from
the latest completed stage in the window.

## BUG-1, `emit_kalman_posterior` is an active but ineffective teleport guard {#bug-1}
**S2, pinned, stage 05**

- **Symptom:** `configs/05_global_id.yaml` sets `emit_kalman_posterior: true`
  and the code and docs present it as the fix that stops a single bad frame
  teleporting the reported track. Yet emitted teleports persist at 33
  (8-delivery set) / 367 (40-delivery set) with it on.
- **Verified (isolated off-vs-on A/B, `emit_ground_source=foot`, 8_init):**
  the emitted `ground_tracks.jsonl` differs 8/8 between off and on, so the
  flag is active and does change the emission (the posterior branch in
  [runner.py](../../src/identity/p5_global_id/runner.py), near the
  `emit_kalman_posterior` check around line 352). It is not a no-op. (An
  earlier "byte-identical no-op" claim compared true-vs-true and is
  retracted.)
- **Root cause:** it is a weak guard; the chi-squared-gated Kalman posterior
  still follows the mis-associated measurement, so the emitted position moves
  but the teleport is not prevented.
- **Fix:** either tighten the gate so the posterior can reject the outlier, or
  drop the framing that this flag is the teleport fix. The effective fix
  already shipped is the emitted-track velocity drop-gate
  ([fixes-log.md](fixes-log.md), teleports 367 to 0), which works regardless
  of this flag.

## BUG-2, Distance-blind Kalman `R`: NOT a bug in production (retracted) {#bug-2}
**not-a-bug (retraction kept for the record), stage 05**

- **Correction (2026-07-16):** the distance/uncertainty-dependent `R` is
  enabled in production: `configs/05_global_id.yaml` sets
  `use_measurement_covariance: true` (with `r_floor_m 0.15`, `r_ceiling_m
  0.8`), fed by `emit_ground_cov: true` in `configs/03_association.yaml`. The
  ground Kalman does scale its measurement trust by the per-cluster covariance
  ([track_manager.py](../../src/identity/p5_global_id/track_manager.py),
  `_measurement_R`).
- **The error:** reading the dataclass default (`use_measurement_covariance:
  bool = False`, [config.py:110](../../src/identity/p5_global_id/config.py#L110))
  and concluding the feature was off. The production YAML overrides it. The
  fixed per-role `R` is only the flag-off fallback.
- The residual hazard is [NB-2](#nb-2).

## BUG-3, Chimera split IS on, but conservative {#bug-3}
**S2, pinned (partially addressed), stage 03**

- **Correction (2026-07-16):** the base clustering is merge-only union-find,
  but production runs the undo pass: `configs/03_association.yaml` sets
  `graph_split_enabled: true`, which fires the chimera-veto/eviction pass
  ([tracklet_graph.py](../../src/identity/p3_association/tracklet_graph.py),
  `_chimera_veto_pass`): it lifts each multi-camera cluster, detects the
  torso-residual chimera signature, evicts the intruding camera's chunks, and
  vetoes the pair so no later pass re-welds them.
- **Residual (real):** the split fires only above conservative thresholds
  (`graph_chimera_torso_residual_px: 30`, `graph_chimera_frame_fraction:
  0.6`, chosen after an earlier over-split), so a mild chimera under 30 px
  torso residual, or one present in under 60% of frames, persists. The
  residual chimera rate (historically 10-32% of 3-plus-view clusters) needs
  re-measurement under the current config.
- **Proposed follow-up:** measure residual chimeras with split on; if
  material, consider a reprojection-gated correlation-clustering objective or
  a graduated split threshold. See [03-association.md](03-association.md).

## BUG-4, Detector-recall bound {#bug-4}
**S1, pinned, stage P1**

- **Symptom:** dark / distant / occluded subjects (the "dark umpire") are
  missed by detection; a miss here is unrecoverable downstream.
- **Root cause:** RTMDet-m at `bbox_thr=0.3`, chosen for speed and
  unbenchmarked for this domain.
- **Evidence:** the association layer contains machinery (synthetic tracklets,
  feet approximation) that exists only to paper over players the detector
  never tracked.
- **Proposed fix:** stronger detector (the `--detector rtmdet_l/x/dino`
  presets exist for exactly this bake-off) plus per-camera adaptive
  `bbox_thr`. See [00-inference.md](00-inference.md).

## BUG-5, Constant-velocity model breaks under manoeuvre {#bug-5}
**S1, pinned, stage 02**

- **Symptom:** tracks fragment exactly when players accelerate, turn, or
  dive; stage 05 then has to stitch the pieces back together.
- **Root cause:** the per-camera tracker uses a constant-velocity Kalman
  ([kalman.py:16](../../src/identity/p2_tracking/kalman.py#L16)); non-linear
  cricket motion violates it, so gating drops the track.
- **Proposed fix:** OC-SORT observation-centric modules (built for non-linear
  motion; `tracker: ocsort` is already wired). See
  [02-tracking.md](02-tracking.md).

## BUG-6, Stitching silently under-merges {#bug-6}
**S2, pinned, stage 05**

- **Symptom:** the offline stitcher that should bridge fragments barely
  fires; `stitched_id_switch_proxy = 0` everywhere, so the distinct-id count
  stays inflated (18-25 against a roster of about 13).
- **Root cause:** the min-cost-flow feasibility gates
  (`temporal_gate_frames=120`, kinematic, occupancy) are too conservative for
  real occlusion gaps. Two further inertness notes from the audit: the role
  penalty treats `unknown` as free (and the online proxy labels most players
  unknown), and velocity continuity contributes zero for near-static pairs,
  exactly where co-located ghosts need discrimination.
- **Proposed fix:** loosen bridging where occupancy proves two segments
  cannot be simultaneous; add a pose/appearance descriptor to the stitch
  cost. See [05-global-id.md](05-global-id.md).

## BUG-7, Single-camera frames get no 3D pose {#bug-7}
**S2, pinned, stage 04**

- **Symptom:** about 39% of player-frames are single-camera and receive no
  triangulated 3D pose.
- **Root cause:** triangulation needs 2 or more rays (`--min-views 2`).
- **Proposed fix:** single-view lift fitting the player's learned skeleton to
  the lone view. See [04-lift.md](04-lift.md). (The narrower single-view
  sticky-hip attempt was rejected: it raised teleports; see fixes-log.md.)

## BUG-8, 01 stabilization not wired into the default delivery flow {#bug-8}
**S3, open, stage 01**

- **Note (2026-07-17 audit):** `src/main.py` defaults
  `--enable-stabilization` to on, so the full-chain driver does run 01.
  The remaining gap is narrower than originally filed: single-stage
  invocations and older wrappers can still skip it silently. Kept open until
  the jitter metric is confirmed in the panel for every production run.

---

## NB-1, C07 global image size in config {#nb-1}
**not-a-bug, stage 03**

- **Claim:** `configs/03_association.yaml` hard-codes `image_w/h = 2560x1440`
  while cam_07 is about 3776x960, so C07 handling would be wrong.
- **Finding:** `load_image_sizes_from_drive` returns cam_07's true native
  size and threads it into the epipole test, the feet check, and
  `keypoints_norm`; the `config.image_w/h` default has no consumers (dead
  config). Kept here so it is not re-chased.

## NB-2, Dataclass config defaults disagree with production YAML {#nb-2}
**S2, open, all stages (config hygiene)**

- **Hazard:** many feature flags are `False`/weaker in the Python dataclass
  defaults but enabled/tuned in the shipped YAML. Reading `config.py` alone
  gives a wrong picture of production; it caused two mis-filed bugs above
  (BUG-2, BUG-3) and a wasted A/B.
- **Confirmed divergent, global-id stage (dataclass vs
  `configs/05_global_id.yaml`):** `confidence_discard` 0.3 vs 0.15,
  `r_ceiling_m` 2.0 vs 0.8, `use_measurement_covariance` False vs true,
  `emit_kalman_posterior` False vs true, `online_role_proxy` False vs true,
  `adaptive_lost_window` False vs true, `min_emit_frames` 0 vs 30,
  `pose_gate_veto_distance` 0 vs 0.3, `reentry_pose_max_distance` 0 vs 0.3,
  `posture_gate_veto_z` 0 vs 3.0, `reentry_posture_max_z` 0 vs 3.0, and in
  stitching: `new_traj_cost_factor` 0.5 vs 3.0, `normalized_costs` False vs
  true, `occupancy_bridge` False vs true, `colocated_merge` False vs true,
  `pose_stitch_max_distance`/`w_pose` 0/0 vs 0.3/2.0,
  `posture_stitch_max_z`/`w_posture` 0/0 vs 3.0/0.5.
- **Confirmed divergent, association stage (dataclass vs
  `configs/03_association.yaml`):** `association_mode` per_frame vs
  tracklet_graph, `graph_llr_positive_cap` 1.5 vs 3.5, `ground_fusion_mode`
  median vs z0_reproj, `foot_contact_mode` legacy vs v3 (inert either way,
  BUG-9), `single_camera_confidence` 0.3 vs 0.45, `graph_corrob_merge` /
  `graph_facing_gate_scale` / `graph_shape_enabled` / `graph_split_enabled` /
  `graph_union_lift_merge` / `graph_lift_feedback` / `emit_posture` /
  `emit_ground_cov` / `single_cam_height_emit` / `anchor_relax_enabled` off
  vs on, `graph_chimera_frame_fraction` 0.3 vs 0.6,
  `graph_chimera_torso_residual_px` 20 vs 30, `graph_union_colocate_m` 1.0
  vs 2.6.
- **Deliberate design:** dataclass defaults preserve the historical baseline
  so `load_*_config(None)` reproduces it; the YAML is the tuned production
  state. The rule: production truth is `configs/*.yaml` and, for any given
  run, its `run_manifest.json`; never audit behavior from dataclass defaults.
  Config-module docstrings now say this explicitly.
