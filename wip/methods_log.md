# 3D Location and Identity: Methods Update

This note records the work completed on the multi-camera cricket pipeline across two areas: 3D
ground-location and cross-camera identity (global player ID). For each area it states the methods
tested, the evidence collected, and the resulting decision. It is written as a status update for the
technical lead and management. Location methods are listed first (M0 to M11), then identity methods
(ID-0 to ID-6). 

## Executive Summary

Both the 3D ground-location path and the identity path have been improved and validated across the
internal 8-delivery evaluation set (`CCPL080626...`).

Location key outcomes:

- Replaced the emitted ground-position estimator with a calibration-aware `z0_reproj` solver.
  - Mean position-error proxy improved from **0.228 m to 0.147 m** across all 8 deliveries.
  - This is a **36% mean reduction** in the position proxy, with no clustering or identity regression.
- Switched P4 emission from raw per-frame observations to the Kalman posterior.
  - Trajectory jitter p95 was approximately **halved** across deliveries.
  - The worst emitted jump improved from **14.0 m to 0.36 m** on the hardest case.
- Built a bird's-eye-view visualization to inspect the final world positions.
  - The BEV confirms that player locations are stable and field-plausible.
- Tested foot-contact refinements, single-camera height correction, temporal foot smoothing, covariance
  fusion, and colour-profile changes.
  - These did not produce a significant generalized gain, so they remain opt-in or are rejected.
- Ran full 17-keypoint multi-view triangulation for the 3D pose deliverable.
  - Multi-camera frames now produce complete, cm-level 17-joint 3D poses after occlusion filling.
  - The remaining 3D pose gap is single-camera-only frames.

Identity key outcomes (validated on all 8 deliveries):

- Distinct-ID counts moved toward the 13 to 15 cricket roster on every delivery, down from 18 to 25 on
  the hard clips `_3/_5/_6/_7/M2`. For example `_6` went 25 to 16, `_7` 22 to 15, `M2` 20 to 14.
- Teleport events fell on every delivery (`_1` 11 to 2, `_4` 15 to 6, `_6` 52 to 40, `_7` 59 to 44),
  while the emitted Kalman trajectory stayed stable.
- Cross-camera agreement rose on the worst clip (`_7` 0.498 to 0.600, a gain of 0.102) and did not
  regress on any delivery. Same-camera collisions stayed at 0 everywhere, which is a hard invariant.
- Each fix was traced to a measured root cause: a structural under-merge on the low-parallax facing
  pairs (the graph merge threshold sits above the single-cue cap), P4b stitching selecting 0 links, and
  ultra-short spurious IDs minted from demoted clusters.
- Added ghost markers (a disappeared ID is now drawn in every camera that geometrically frames its
  ground, plus greyed markers in the bird's-eye view) and an in-pipeline pose-corroborated merge.
- Every identity change is behind a config flag. With flags off the pipeline is byte-identical to the
  committed baseline (re-verified), and the full unit suite (152 tests) stays green.

Current conclusion:

- The 3D location path remains stable. The identity path is now improved and validated across all 8
  deliveries: IDs sit at or near the true roster, teleports are down on every clip, and cross-camera
  agreement is up on the hardest clip with no collision regression.
- The main remaining weakness is the `M2` teleport count. That figure is driven by the teleport proxy
  reacting to M2's noisy single-camera foot projections, not by the emitted trajectory, which stays
  smooth.

## Evaluation Standard

All candidate changes were evaluated against a fixed baseline and, where applicable, across initial dataset of
8 deliveries. A method was only marked as accepted if it produced a significant, generalized
improvement without introducing clustering, identity, or collision regressions.

Metrics used:

- **Accuracy proxy A: distance to triangulated foot, in metres.**
  - Distance from emitted `ground_xy` to a calibration-optimal RANSAC triangulated foot reference.
  - Only trusted on clusters with at least 3 camera views and low triangulation reprojection error.
- **Accuracy proxy B: reprojection error against the foot pixels, in pixels.**
  - Projects emitted `ground_xy` back into member cameras and compares against the foot pixel used.
  - Used as a supporting metric because grazing cameras can overweight pixel residuals.
- **Trajectory stability: P4 displacement and maximum emitted jumps.**
  - Used to evaluate whether the emitted world trajectory is stable enough for downstream use.
- **Downstream identity indicators: teleport events, cross-camera agreement, same-camera collisions.**
  - These are useful, but must be interpreted carefully because some are dominated by identity
    assignment rather than location accuracy.
- **Visual validation: bird's-eye-view render.**
  - Used to distinguish actual location errors from ID-label flicker.

Important calibration finding:

- The stored ball calibration audit shows mean reprojection errors of **1.2-1.9 px** and p95 of
  **2.8-4.5 px** across the evaluation set.
- Calibration is therefore not the dominant source of player-ground error.
- The main location errors come from 2D foot-contact estimation, single-camera grazing projection, and
  fusion/emission choices.

## Summary of Methods Tested

| ID | Method | Result | Verdict | Rationale |
|---|---|---:|---|---|
| M0 | Calibration audit | Ball reprojection mean 1.2-1.9 px; p95 2.8-4.5 px | Validated | Calibration is accurate enough to use directly; no calibration refinement required. |
| M1 | Baseline characterization | Proxy A baseline 0.211-0.228 m depending on the evaluated subset; 382 total teleport events | Baseline | Established the fixed comparison point. |
| M2 | Inverse-covariance IRLS fusion (`robust_cov`) | Proxy A worsened from 0.176 m to 0.248 m on delivery 1 | Rejected | The covariance model pulled estimates toward near cameras and underperformed reprojection minimization. |
| M3 | Ground-plane robust reprojection solve (`z0_reproj`) | Proxy A improved from 0.228 m to 0.147 m across all 8 | Accepted | Generalized 36% mean position-proxy improvement with byte-identical clustering. |
| M4 | Foot-contact v2: ankle midpoint, plausibility checks, ankle-height plane | +6% on proxy A vs z0, -5% on proxy B vs z0 | Opt-in | Mixed proxy results; not significant enough for default enablement. |
| M5 | Single-camera height correction | Not measurable by proxy A/B | Opt-in | Physically correct, but downstream metrics could not confirm a clear gain. |
| M6 | Temporal smoothing of emitted foot pixels | Teleports reduced 382 to 364; no agreement change | Opt-in | Small improvement only; identity metrics remain dominated by ID assignment. |
| M7 | Bird's-eye-view visualization | Field-plausible, stable player positions | Delivered | Confirmed that remaining teleport-like behavior is mainly identity flicker. |
| M8 | Emit Kalman posterior instead of raw observations | Jitter p95 roughly halved; worst jump 14.0 m to 0.36 m | Accepted | Large stability improvement with no collision regression. |
| M9 | v4 mosaics with BEV panel | BEV integrated into mosaic output | Delivered | Provides operational review output for the polished stack. |
| M10 | P1 colour-profile A/B | Current BGR path best/tied | No change | RGB swap and grayscale did not improve detection or keypoint confidence. |
| M11 | Full 17-keypoint multi-view triangulation and occlusion fill | Complete 17-joint poses for all multi-camera frames on delivery 1 | Accepted for 3D pose | Produces cm-level full-body 3D where at least two cameras observe the player. |

## Summary of Identity Methods Tested

All identity methods are behind config flags (`configs/p3_association_v5.yaml`,
`configs/p4_global_id_v5.yaml`); flags off reproduces the committed baseline byte-for-byte. Runs land
in `data/derived/runs/pipetrack_v5`; the frozen baseline is `data/derived/runs/pipetrack_v3/_baseline_snapshot`;
the batch driver is `src/identity/id_pipeline.py`.

| ID | Method | Result | Verdict | Rationale |
|---|---|---:|---|---|
| ID-0 | Identity baseline characterization (all 8) | Agreement 0.50-0.98; distinct IDs 11-25; teleports 7-171; collisions 0 | Baseline | Fixed identity comparison point; hard clips `_5/_6/_7/M2` over-segment and under-merge. |
| ID-1 | Corroboration-aware cross-camera merge + parallax-adaptive facing gate | `_7` agreement 0.498 to 0.600 (+0.102); teleports -13; single-camera rate -0.051 | Accepted | Lets a genuine same-player facing-pair merge when no cue disagrees; easy clips byte-identical. |
| ID-2 | P4b stitching v2: pose-shape gate + smoothed exit/entry velocity + usable link cost | Enables stitching that previously selected 0 links; merges same-build fragments only | Accepted | Root cause was a dummy "new-trajectory" cost that undercut every real stitch. |
| ID-3 | Cardinality prior: drop IDs whose whole-clip span < 30 frames (`min_emit_frames`) | Distinct IDs collapse toward roster on all 8; teleports fall on every clip | Accepted | A player is present the full 12 s clip, so a 6-25-frame ID is a fragment/shadow, not a late entry. |
| ID-4 | P4a hardening: adaptive lost-window, pose veto in chi2 gate, descriptor-gated re-entry | Small direct effect (diagnostics rarely fire on facing-pair clips) | Accepted (guardrail) | Correct anti-teleport guardrails; limited because the P4a pose descriptor needs parallax the facing pairs lack. |
| ID-5 | Ghost markers v2 + in-pipeline pose-corroborated ghost-verification merge | Disappeared IDs drawn in every camera that frames their ground; fragment re-joins only on body-shape agreement | Delivered | Diagnostic overlay plus a conservative, cannot-link-safe merge that is the "same shape to same ID" rule. |
| ID-6 | Identity ground truth (measurement) | No labels exist; `evaluate_ground_truth` (IDF1 / ID-switch) ready but unrun | Open | All identity figures are proxies read jointly; hand-labelling `_7`/`M2` would enable real IDF1. |
| Audit | Correctness fixes + byte-identical guarantee | `cap_covariance` both-axes; fragment posture-veto aggregate; committed configs reproduce baseline exactly | Accepted | Non-flag-gated correctness fixes verified not to change default output. |

## Detailed Results

### M0: Calibration Audit

Purpose:

- Verify whether camera calibration was a material source of player-ground error.

Evidence:

- Ran `audit_calibration('drive')` in `pose_estimation/cricket/calibration.py`.
- Stored ball reprojection error:
  - Mean: **1.2-1.9 px**
  - p95: **2.8-4.5 px**
  - Max: approximately **5 px**
- The audit also reports a large projected-vs-stored delta in one artifact path, but that is caused by
  crop/normalization conventions in the stored ball artifact, not by projection matrix error.

Conclusion:

- Calibration is sufficiently accurate for the current problem.
- The location pipeline should use the calibration directly through reprojection minimization rather
  than attempting to refine calibration or model it as a large uncertainty term.

### M1: Baseline Characterization

Purpose:

- Establish a fixed baseline for all later A/B tests.

Baseline observations:

- Single-camera cluster rate: **0.39-0.61**
- Multi-camera ground-spread p50: approximately **1.9-2.4 m**
- Multi-camera ground-spread p95: approximately **3.4-5.1 m**
- Multi-camera reprojection error above 12 px: **34-61%**
- P4 teleport events: **7-171 per delivery**, **382 total** across the 8-delivery set
- Cross-camera agreement: **0.50-0.98**
- Same-camera collisions: **0**

Conclusion:

- This became the comparison point for all location changes.

### M2: Inverse-Covariance IRLS Fusion (`robust_cov`)

Purpose:

- Test the literature-style approach of fusing per-camera ground points using distance-dependent
  covariance and robust weighting.

Implementation:

- Implemented `ground_covariance` and `robust_fuse_ground`.
- Propagated pixel uncertainty through the ray-plane Jacobian into an anisotropic 2D covariance.
- Used Huber-weighted IRLS over camera members.

Result:

- Delivery 1, proxy A:
  - Median baseline: **0.176 m**
  - `robust_cov`: **0.248 m**
  - p95 worsened from **0.376 m to 0.516 m**
  - Improved only **19.8%** of clusters

Conclusion:

- Rejected as the emitted ground-position estimator.
- The method optimized a covariance model that did not match the actual dominant error source. With
  accurate calibration and low-parallax camera geometry, the reprojection-optimal point was better than
  the covariance-weighted metric average.
- The covariance code remains useful for future Kalman measurement-noise work, but not for the current
  emitted-position estimator.

### M3: Ground-Plane Robust Reprojection Solve (`z0_reproj`)

Purpose:

- Use the calibrated cameras directly to solve for the most consistent point on the ground plane.

Implementation:

- Added `ground_from_reprojection` in `geometry.py`.
- Solves:

```text
argmin_{x,y} sum_c w_c * Huber(|| project_c([x, y, 0]) - foot_c ||)
```

- Uses Gauss-Newton optimization over each member camera's full 3x4 projection matrix.
- Applies a hard inlier refit to remove gross outlier foot observations.
- Keeps the clustering merge gate unchanged, so clustering and identity are invariant.

All-8-delivery results:

| Delivery | Proxy A baseline (m) | Proxy A z0 (m) | A gain | Proxy B baseline (px) | Proxy B z0 (px) | B gain | Clustering identical |
|---|---:|---:|---:|---:|---:|---:|---|
| M1_1_14_1 | 0.176 | 0.145 | 18% | 84.3 | 73.7 | 13% | Yes |
| M1_1_14_2 | 0.166 | 0.135 | 18% | 71.8 | 64.0 | 11% | Yes |
| M1_1_14_3 | 0.172 | 0.116 | 32% | 80.9 | 68.6 | 15% | Yes |
| M1_1_14_4 | 0.209 | 0.151 | 28% | 70.9 | 59.7 | 16% | Yes |
| M1_1_14_5 | 0.241 | 0.153 | 36% | 129.7 | 119.6 | 8% | Yes |
| M1_1_14_6 | 0.200 | 0.116 | 42% | 87.1 | 74.4 | 15% | Yes |
| M1_1_14_7 | 0.289 | 0.177 | 39% | 185.0 | 142.2 | 23% | Yes |
| M2_1_12_1 | 0.373 | 0.183 | 51% | 129.3 | 74.1 | 43% | Yes |
| **Mean** | **0.228** | **0.147** | **36%** | **104.9** | **84.5** | **19%** | **All yes** |

Conclusion:

- Accepted as a real generalized improvement.
- The improvement is consistent across all 8 deliveries and is largest on the hardest deliveries.
- Clustering remains byte-identical because the merge gate is untouched.
- This is the main accepted P3 location improvement.

Caveat:

- Absolute reprojection errors are still high in some cases because the input foot pixels can disagree
  across cameras. The solver finds the best ground-plane compromise, but it cannot fully correct wrong
  or inconsistent foot-contact pixels.

### M4: Foot-Contact v2

Purpose:

- Improve the foot-contact pixel supplied to the ground solver.

Implementation:

- Added `ground_contact_pixel_ex` in `geometry.py`.
- Uses ankle midpoint when both feet are valid, otherwise selects the planted/lower ankle.
- Adds tighter vertical plausibility checks and a horizontal plausibility check.
- Reports ankle height so the reprojection solver can project onto `z = ankle_height` rather than always
  onto `z = 0`.

Important correction:

- The first wiring allowed the new foot pixel to affect the clustering gate through
  `detection.ground_xy`.
- That changed identity behavior: cluster count increased by **25%** on one delivery and **23%** on
  another.
- The fix was to decouple the paths:
  - The gate, cost, and triangulation path remain pinned to the legacy foot.
  - `foot_contact_mode` only affects the emitted z0 position.
- After decoupling, cluster counts are byte-identical to baseline across all 8 deliveries.

Decoupled all-8 result:

| Metric, mean over 8 | Baseline | z0 | z0 + foot-v2 |
|---|---:|---:|---:|
| Proxy A: distance to triangulated foot (m) | 0.211 | 0.144 | 0.135 |
| Proxy B: reprojection vs legacy foot pixels (px) | 102.3 | 83.5 | 88.0 |

Conclusion:

- Kept as opt-in, not enabled by default.
- Proxy A improved by **6%** over z0, but proxy B worsened by **5%** over z0.
- The mixed evidence is not strong enough to mark this as a default production win.

### M5: Single-Camera Height Correction

Purpose:

- Address the large ground-plane shift caused by projecting ankle pixels directly to `z = 0` in
  single-camera cases.

Finding:

- Instrumentation showed that the ankle-height correction can shift the single-view ground intersection
  by approximately **0.94 m** on average, with p95 around **1.3 m**.
- This is much larger than the nominal anatomical height difference because grazing camera angles amplify
  the projection error.

Implementation:

- Extended the z0 reprojection emission path to handle single-member clusters.
- For single-camera clusters, the ankle ray is projected onto its ankle-height plane.
- The merge gate is unaffected because a single-member cluster has zero spread.

Conclusion:

- Kept as opt-in.
- The change is physically correct, but proxy A and proxy B cannot measure it well:
  - Proxy A requires at least 3 camera views.
  - Proxy B is self-consistent for a single camera and therefore not discriminative.
- Downstream metrics did not show a strong enough independent gain to enable by default.

### M6: Temporal Smoothing of Emitted Foot Pixels

Purpose:

- Suppress one-frame ankle or foot-contact spikes before they enter the ground solver.

Implementation:

- Added `smooth_emit_feet` in `associator.py`.
- Median-filters each `(camera, tracklet)` foot-pixel series.
- Writes the smoothed value as `emit_foot_px`.
- This is emission-only and never affects the clustering gate.

Downstream result, all 8 deliveries:

| Metric | Baseline | All foot changes |
|---|---:|---:|
| Cross-camera agreement, mean | 0.812 | 0.812 |
| Teleport events, total | 382 | 364 |
| Same-camera collisions | 0 | 0 |

Conclusion:

- Kept as opt-in.
- The teleport reduction is small and cross-camera agreement is unchanged.
- This supports the broader finding that the remaining downstream failures are dominated by identity
  assignment rather than by emitted ground-location noise.

### Operational Note: Parallel Evaluation

When running the 8 P3 jobs in parallel, uncapped BLAS threading caused oversubscription:

- 8 processes each spawned approximately 32 OpenMP threads.
- Load average reached approximately 45 on a 32-core machine.
- Progress became negligible.

Fix:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

After capping BLAS threads per process, each job used approximately one CPU core and the full evaluation
ran efficiently.

### M7: Bird's-Eye-View Visualization

Purpose:

- Visually validate final world positions and separate true location errors from identity label changes.

Implementation:

- Added `src/identity/visualization/render_bird_eye_view.py`.
- Renders a top-down field plot from P4 `ground_tracks.jsonl`.
- Outputs MP4 animation and sampled montage.

Observed result:

- Player positions are stable and field-plausible:
  - Pitch and crease-relative positions are consistent.
  - Fielders are distributed realistically.
  - No wild world-position outliers were observed in the inspected deliveries.
- Remaining teleport-like behavior is primarily identity flicker:
  - Example: in the M2 delivery, a stationary right-side deep fielder is assigned different player IDs
    across frames while the dot remains in the same physical location.

Conclusion:

- The BEV supports the quantitative result: the location estimate is stable.
- Remaining large downstream gains should come from identity tracking and association, not from further
  ground-location tuning.

Known visualization gap:

- The single-camera marker overlay did not populate because `binding_id` in correspondences does not
  directly join to `global_player_id` in ground tracks.
- This is a minor visualization join issue and does not affect the rendered positions.

### M8: Emit Kalman Posterior

Purpose:

- Emit the filtered, chi-square-gated P4 posterior instead of the raw per-frame fused observation.

Implementation:

- P4 now emits `track.kalman.pos_world_xy`, already computed by `manager.update`.
- Controlled by `p4a.emit_kalman_posterior`, enabled by default.
- Evaluated using the same committed z0 P3 input across all 8 deliveries.

Result:

- Trajectory displacement p95 approximately halved across deliveries.
- Representative p95 improvements:
  - `_6`: **0.148 to 0.045 m/frame**
  - `_7`: **0.169 to 0.065 m/frame**
  - `M2`: **0.212 to 0.085 m/frame**
- Worst emitted jump:
  - `_7`: **14.0 m to 0.36 m**
  - `_3`: **2.73 m to 0.76 m**
  - `_6`: **3.32 m to 1.56 m**
- Same-camera collisions remained **0**.

Conclusion:

- Accepted and enabled by default.
- This is the main accepted P4 location-stability improvement.
- `teleport_event_count` changed only slightly because that metric is counted at ID-assignment time,
  while this change improves the emitted trajectory.

### M9: v4 Mosaics with BEV Panel

Purpose:

- Add a polished operational visualization of the world-track output.

Implementation:

- Updated `src/identity/visualization/render_videos.py`.
- Added `draw_bev_panel` and `compute_ground_extents`.
- Replaced the bottom-left delivery-monitor slot with a QT-style field plot:
  - boundary ellipse
  - pitch strip
  - ID-coloured player dots
  - motion trails

Source stack for v4 output:

- P3:
  - `ground_fusion_mode: z0_reproj`
  - `foot_contact_mode: v2`
  - `foot_smooth_window: 5`
- P4:
  - `emit_kalman_posterior: true`

Output:

- `artifacts/pipetrack_v4/mosaics/`

Conclusion:

- Delivered.
- Smoke verification confirmed that BEV dots land at sensible field positions consistent with the camera
  tiles and global roster.
- Note: the v4 mosaic stack used the opt-in foot-v2 configuration for visualization review. The
  conservative default recommendation remains `foot_contact_mode: legacy`.

### M10: P1 Colour-Profile A/B

Purpose:

- Check whether BGR/RGB handling or grayscale preprocessing could improve detection, pose quality, and
  therefore downstream 3D location.

Configuration check:

- Detector RTMDet:
  - `bgr_to_rgb=False`
  - BGR-ordered mean values: `[103.53, 116.28, 123.675]`
- Pose RTMPose:
  - `bgr_to_rgb=True`
  - RGB-ordered mean values: `[123.675, 116.28, 103.53]`
- Input is loaded through `cv2.imread`, so the current BGR path is consistent.

Empirical A/B, real RTMDet + RTMPose, 120 frames, cameras 01/03/04:

| Colour order | People | Detection confidence | Keypoint confidence mean / p50 |
|---|---:|---:|---:|
| Current cv2 BGR | 520 | 0.797 | 0.653 / 0.731 |
| Swapped RGB feed | 520 | 0.790 | 0.656 / 0.735 |
| Grayscale | 520 | 0.776 | 0.650 / 0.726 |

Conclusion:

- No change recommended.
- RGB swap is within noise and does not change detection count.
- Grayscale is slightly worse.
- Colour preprocessing is not a meaningful lever for this footage.

### M11: Full 17-Keypoint Multi-View Triangulation and Occlusion Fill

Purpose:

- Produce complete full-body 3D pose output, not only the ground point.

Implementation:

- Ran the P6 stage through `src/identity/p4_lift/run_triangulation.py`.
- Triangulates all 17 COCO keypoints per global player across calibrated cameras.
- Added occlusion extrapolation in `src/identity/common/triangulation.py`:
  - `fill_occluded_joints`
  - `fill_from_skeletal_prior`

Result, delivery 1:

- Per-joint reprojection error across all 17 joints:
  - p50: **2-4 px**
  - p95: **5-7 px**
- Triangulated ankle height:
  - p50 approximately **0.11 m**, independently supporting the ankle-height prior used in M5.
- Complete 17-joint poses:
  - Before occlusion fill: **2960**
  - After occlusion fill: **3238**
  - Coverage of multi-camera frames: **100%**
- 278 frames used extrapolated joints with low confidence and sentinel reprojection metadata.

Conclusion:

- Accepted for the 3D pose deliverable where at least two cameras observe the player.
- The remaining gap is the approximately **39% single-camera frames**, which cannot be triangulated.
- Next technical step for those frames is single-view pose reconstruction using the player's canonical
  3D skeleton anchored at the z0 ground position.

## Identity Workstream (P3 cross-camera + P4 global-ID)

The location work established that most remaining downstream failures were identity errors, not
position errors. This workstream targets the six failure modes in `wip/id_issues.md`: cross-camera
under-merge (ID-1), fragmentation / over-segmentation (ID-2), teleports (ID-3), the dead colour
appearance cue (ID-4), non-splittable clustering (ID-5), and the absence of identity ground truth
(ID-6). Everything below is behind config flags; with flags off the pipeline reproduces the committed
baseline byte-for-byte (re-verified on delivery 1 after all edits: 0.952 / 12 IDs / 11 teleports / 0
collisions, unchanged), and the full unit suite (152 tests) stays green.

### Headline Result (all 8 deliveries, baseline to v5 stack)

Each metric is shown as baseline to v5.

| Delivery | Agreement | Distinct IDs | Teleports | Collisions |
|---|---|---|---|---:|
| M1_1_14_1 | 0.952 to 0.953 | 12 to 10 | 11 to 2 | 0 |
| M1_1_14_2 | 0.977 to 0.976 | 11 to 11 | 7 to 5 | 0 |
| M1_1_14_3 | 0.870 to 0.877 | 18 to 13 | 19 to 11 | 0 |
| M1_1_14_4 | 0.857 to 0.857 | 13 to 11 | 15 to 6 | 0 |
| M1_1_14_5 | 0.767 to 0.770 | 15 to 12 | 48 to 37 | 0 |
| M1_1_14_6 | 0.802 to 0.803 | 25 to 16 | 52 to 40 | 0 |
| M1_1_14_7 | 0.498 to 0.600 | 22 to 15 | 59 to 44 | 0 |
| M2_1_12_1 | 0.778 to 0.781 | 20 to 14 | 171 to 166 | 0 |

Taken together, every clip's distinct-ID count moved toward the 13 to 15 roster, teleports fell on every
clip, cross-camera agreement is stable-or-up (largest gain +0.102 on the worst clip), and same-camera
collisions stayed 0. All 8 deliveries now sit at or near the true roster.

### ID-0: Identity Baseline Characterization

Purpose:

- Fix the identity comparison point across all 8 deliveries before any change.

Baseline (committed `pipetrack_v3`, frozen to `_baseline_snapshot`):

| Delivery | Agreement | Distinct IDs | Teleports | Verdict |
|---|---:|---:|---:|---|
| _1 | 0.952 | 12 | 11 | pass |
| _2 | 0.977 | 11 | 7 | pass |
| _3 | 0.870 | 18 | 19 | pass |
| _4 | 0.857 | 13 | 15 | pass |
| _5 | 0.767 | 15 | 48 | warn |
| _6 | 0.802 | 25 | 52 | warn |
| _7 | 0.498 | 22 | 59 | warn |
| M2 | 0.778 | 20 | 171 | fail |

Conclusion:

- The scene holds ~13-15 people, so 18-25 distinct IDs on `_3/_5/_6/_7/M2` is 40-90% over-segmentation,
  and the 0.498 agreement on `_7` is a severe cross-camera under-merge. These became the targets.

### ID-1: Corroboration-Aware Cross-Camera Merge + Parallax-Adaptive Facing Gate

Purpose:

- Fix the cross-camera under-merge on the low-parallax facing pairs (worst on `_7`).

Root cause (verified in code):

- The tracklet graph requires an edge log-likelihood-ratio (LLR) of at least
  `graph_llr_merge_threshold` (2.0) to merge, but each cue is capped at `graph_llr_positive_cap` (1.5).
- On the facing pairs, colour appearance abstains (identical kit), motion abstains for static players,
  and posture can abstain for crouched/oblique bodies, leaving ground alone (<= 1.5) - which can never
  reach 2.0. A genuine same-player pair therefore never merges.
- The facing pairs also use the tighter `opposite_pair_ground_gate_m` (2.5 m), which under
  foot-projection noise can itself split a correct two-view merge.

Implementation (`src/identity/p3_association/tracklet_graph.py`, `src/identity/p3_association/config.py`):

- `graph_corrob_merge`: a second, conservative merge pass that admits an edge in
  `[graph_llr_merge_single, graph_llr_merge_threshold)` only when it has full co-visible support, no
  observable cue disagrees (every present cue LLR >= 0), it is the mutual unambiguous best for both
  endpoint clusters, and it passes the cannot-link / veto check.
- `graph_facing_gate_scale`: widens the graph hard-distance edge gate on the calibration-derived facing
  pairs only (2.75 m to 3.575 m), never below the general gate.

Result (all 8, v5 P3 config vs baseline):

- `_7` cross-camera agreement **0.498 to 0.600 (+0.102)**, teleports -13, single-camera rate -0.051
  (more views are bound together).
- Deliveries `_1`-`_4` byte-identical; same-camera collisions 0 everywhere.

Conclusion:

- Accepted as the ID-1 layer. Largest single lever found for cross-camera under-merge, with no
  easy-clip or collision regression. The distinct-ID count ticked up +1/+2 on some hard clips because
  binding more facing-pair views promotes previously-demoted single-camera clusters into real bindings;
  that is addressed by ID-2/ID-3, not by ID-1.

### ID-2: P4b Stitching v2 (Pose-Gated Fragment Merge)

Purpose:

- Merge the temporal fragments that inflate the distinct-ID count.

Root cause (measured):

- On the hard clips the tracklet graph already produced only ~10-11 clean bindings, yet P4 emitted
  18-25 IDs, so the excess was downstream.
- P4b stitching selected **0** links on every hard clip despite 243-1429 feasible edges: `solve_flow`
  only stitches when an edge cost is below the "new-trajectory" dummy cost
  (`w_spatial * new_traj_cost_factor = 0.5`), but a plausible stitch
  (`0.1 * gap + 1.0 * distance + ...`) almost always exceeds 0.5, so nothing merged.

Implementation (`src/identity/p5_global_id/stitching.py`, `runner.py`, `config.py`):

- Threaded each track's accumulated pose-shape descriptor into the segment (`pose_by_id`).
- Added a hard pose gate (`p4b.pose_stitch_max_distance`) so only same-build fragments merge, an
  optional `w_pose` term to prefer better body-shape matches, and Kalman-window-smoothed exit/entry
  velocities instead of the raw last-two-frame difference.
- Raised `new_traj_cost_factor` (0.5 to 3.0) so plausible stitches can win while the pose gate and the
  existing same-camera-frame cannot-link constraint prevent chimeras.

Conclusion:

- Accepted. This is the mechanism that lets fragments re-join; on its own it is modest (few clean
  temporally-separated fragment pairs exist), and it composes with ID-3.

### ID-3: Cardinality Prior (Minimum Emitted Lifespan)

Purpose:

- Remove ultra-short spurious IDs that neither bind nor stitch.

Evidence (dumping the M2 ground tracks):

- Of 18 IDs, 9 were full-clip stable players, ~4 were mid-length crease-congestion IDs, and **5 were
  alive only 6-74 frames** (e.g. 6, 8, 13, 21 frames) - fragments/shadows minted from the many demoted
  clusters (38/41/87 on `_6/_7/M2`).

Implementation (`src/identity/p5_global_id/config.py`, `runner.py`):

- `p4a.min_emit_frames`: after stitching, any global ID whose total emitted frame-span is below the
  threshold (set to 30 frames = 0.6 s) is dropped; its detections become unlabelled. A real cricket
  player is present the whole 12 s delivery, so a sub-30-frame ID is a fragment, not a late entry.

Result (all 8, full v5 stack vs baseline):

- Distinct IDs: `_1` 12 to 10, `_3` 18 to 13, `_4` 13 to 11, `_5` 15 to 12, `_6` 25 to 16, `_7` 22 to 15,
  `M2` 20 to 14.
- Teleports: `_1` 11 to 2, `_3` 19 to 11, `_4` 15 to 6, `_5` 48 to 37, `_6` 52 to 40, `_7` 59 to 44.
- Cross-camera agreement stable-or-up; same-camera collisions 0.

Conclusion:

- Accepted as the dominant ID-count and teleport lever. Conservative (only 6-25-frame fragments were
  dropped; no full-clip player was lost, and agreement never dropped). This is the research-backed
  fixed-roster / cardinality prior applied to a known ~13-15-person scene.

### ID-4: P4a Identity Lifecycle Hardening

Purpose:

- Anti-teleport and anti-fragmentation guardrails inside the online tracker.

Implementation (`src/identity/p5_global_id/track_manager.py`, `global_track.py`, `config.py`):

- `adaptive_lost_window` (+ `lost_window_max_frames`): a well-established track earns a longer occlusion
  tolerance (up to 90 frames) so a briefly-hidden regular is re-acquired instead of re-born.
- `pose_gate_veto_distance`: a Stage-2 candidate whose mature pose descriptor is clearly the wrong build
  is vetoed inside the chi2 gate, not merely penalised.
- `reentry_pose_max_distance`: reviving a deleted track additionally requires body-shape agreement
  (abstains when either descriptor is immature), blocking kinematically-plausible wrong-person re-entry.

Result and caveat:

- The direct effect was small: the `pose_gate_vetoes` and `reentry_pose_rejects` diagnostics rarely
  fire because the P4a pose descriptor is the *triangulated* one, which needs parallax the facing pairs
  lack, so it is often unavailable on exactly the hard clips.

Conclusion:

- Accepted as correct guardrails. A follow-up (deferred) is to feed the *billboard* posture descriptor
  (which works on facing pairs) into the P4a veto so it bites on the facing-pair clips.

### ID-5: Ghost Markers v2 and In-Pipeline Ghost Verification

Purpose:

- Make disappearances visible for diagnosis, and use a ghost coinciding with a detection as an
  identity-merge signal.

Implementation:

- `src/identity/common/geometry.py`: `ground_point_visible_in` - a per-camera visibility test with
  cheirality (via the pitch-oriented `camera_axis_lookat` forward axis) plus the in-frame check. The
  previous ghost code lacked cheirality and could reproject points that were actually behind the camera.
- `src/identity/visualization/render_videos.py`: a last-known fused-position store lets a ghost be
  drawn for an ID gone from *every* camera (not just occluded in one), in each camera that geometrically
  frames that ground, faded by age; "occluded" vs "lost" labels; greyed ghost dots added to the BEV.
- In-pipeline "ghost verification": the ID-2 pose-gated stitch plus the ID-4 descriptor-gated re-entry
  are the merge form of the rule "if a ghost coincides with a pose of the same body shape, it is the
  same ID" - a conservative, cannot-link-safe auto-merge with diagnostics, never a silent weld.

Conclusion:

- Delivered and verified visually on `_7` and `M2` mosaic frames: ghosts appear across the correct
  cameras and the roster greys out lost IDs.

### Audit Fixes and the Byte-Identical Guarantee

Fixes made during the file-by-file audit:

- `src/identity/p5_global_id/ground_kalman.py` `cap_covariance`: bounded **both** position-variance axes
  during long Lost windows; the previous version broke after the first over-threshold axis and could
  leave the other unbounded.
- `src/identity/p3_association/tracklet_graph.py`: the fragment posture-veto now uses the fragment's
  best-supported posture aggregate rather than only its first chunk (which was often undefined), so the
  veto can actually fire.

Verification:

- Re-ran delivery 1 through P3 to P4 with the committed (flags-off) configs after all edits: it
  reproduced the baseline exactly (0.952 / 12 / 11 / 0), confirming these non-flag-gated correctness
  fixes do not change default output.


### Identity: Remaining Weakness and Interpretation

- `M2` teleports (166) are the main residual. Inspection of the teleport examples shows speeds up to
  152 m/s on stable IDs (e.g. `P001`), which is the teleport-*proxy* reacting to noisy single-camera
  foot projections between consecutive frames (`M2` has the worst single-camera rate, 0.61), not the
  emitted Kalman trajectory (which stays smooth per M8). The genuine fix for `M2` is upstream P1/P2
  quality in low light, plus the deferred billboard-posture P4a veto.

