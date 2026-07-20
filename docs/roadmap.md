# Roadmap — what to try next

The forward-looking research and improvement directions for the multi-camera cricket 3D-pose and
identity pipeline: the quality levers, ranked, with the measured reasoning behind each. This is the
"what should be tried next" companion to [`methods_log.md`](methods_log.md) (what has been tried and
tested) and [`analysis/README.md`](analysis/README.md) (the measured diagnosis that motivates these
levers). Concrete pre-hand-over bug-fix and cleanup work is tracked separately in the internal
`wip/` register, not here.

Priority legend: [HIGH] high impact · [MED] medium · [LOW] low or cleanup. Effort: (S)mall, (M)edium,
(L)arge.

---

## 1. Guiding principles

- Accuracy first. Optimization is secondary and only after the highest-quality pipeline exists, though
  implementations must stay efficient enough to generate mosaics and A/Bs quickly.
- Never blindly trust the existing implementation, this document, or internet resources. Every
  modification is experimentally verified with before/after numbers, and human mosaic review is the final
  qualitative check.
- Decisions are driven by measured metrics. Primary metrics: reprojection error, ID switches, ID
  teleports, cross-camera agreement, same-camera collisions (hard invariant 0). Do not optimize for
  metrics that need ground truth (none exists).
- Everything important should ultimately be decided in 3D: the core representation is the full 3D
  skeleton, and identity reasoning should prioritise skeleton shape and location over isolated 2D
  observations. This is the direction behind A0.
- Robust multi-camera fusion: identify and down-weight the bad, noisy, occluded, or outlier camera rather
  than averaging all cameras equally.
- No ground-truth or labelled-data assumptions anywhere. No fine-tuning that requires labels.

---

## 2. Pipeline and algorithm backlog (the quality levers)

### A0 [HIGH] (L) decide-in-3D plus 3D-coverage recovery. The current top item.
The single-triangulation rewire is structural only: stage 04 triangulates once (binding-keyed), 05 and 06
carry the 3D forward but do not consume it. Two coupled pieces:
- Consume the 3D in 05 behind a `--track-in-3d` flag (default off equals today's ground-plane path): feed
  04's `pelvis_ground_xy` as the Kalman measurement and `pelvis_cov_m2` as the R, add a 3D pose-shape
  re-ID for re-entry, and use per-cluster reprojection or cycle-consistency as a chimera-split signal.
- Coverage recovery: `tri_cov` dropped from v8 to v9 (M1_1_14_1 v8 terminal 0.817 to v9 04 0.566) mainly
  because the global-keyed terminal re-triangulation was dropped. Recover by re-triangulating per
  `global_player_id` after 05, or by the 3D-pooling above. A/B on 8 then 40.

### A8 [HIGH] (L) single-view PnP lift. Raise multi-camera coverage.
About 39% of frames are single-camera and get no 3D. Fit the identity's canonical skeleton (bone lengths
from its multi-view frames) to a lone 2D view, PnP-style, with honest covariance, giving 3D on
single-camera frames and a physically constrained ground position. The structural lever behind the
coverage and single-camera gap. A prior sticky-hip attempt (1F) was rejected; PnP against the canonical
skeleton is the principled version.

### A9 [MED] (M) detection recall on the deep field and small subjects. UPDATED this session.
Status: tiling was tested on the 40-set this session (see [`methods_log.md`](methods_log.md) Part A). Result is
two-edged: tiled detection raises cross-camera agreement on all 8 hardest deliveries (mean +0.115, cleanly
attributable to tiling and not NMS) but raises underlying teleport events (+704 total, worst on crowded
clips) at about 3x GPU cost. No stronger detector weights exist on the box (RTMDet-l/x, RTMO-l, YOLO are
empty placeholders). Open next steps:
- Re-measure tiling with the A3 emit-gate on, since A3 masks exactly the teleports tiling inflates. This
  is the combination that would actually ship, and the honest way to read the tradeoff.
- If tiling is adopted, pair it with W6 peripheral suppression (it consumes the noisy peripherals tiling
  adds).
- Decision pending from the human: accept the agreement gain against the teleport and 3x-cost tradeoff,
  or park tiling.

### A1 [HIGH] (S) fix the teleport metric and verdict.
Every `fail` verdict is `teleport > 60` computed on raw bbox-bottom foot projections averaged across
cameras (single-camera grazing noise). Add a velocity-gated teleport metric on the emitted
`ground_tracks.jsonl`, multi-camera segments only; demote the raw proxy to a tripwire. Note: the A3
emit-velocity gate (implemented, 40-confirmed, off by default) already provides the emitted-track velocity
gate; this item is the metric-and-verdict half.

### A4 [HIGH] (M) depth-aware association weighting for grazing and facing cameras.
Split identity persists on the low-parallax facing pairs. Weight the ground-distance LLR by each camera's
calibrated depth-uncertainty and up-weight the triangulation-consistency (union-lift) cue. The cap fix
(accepted) only partially closed this; this is the principled version.

### 05b stitching under-merge [MED] (M).
Fragments are not being bridged (18 to 25 distinct IDs vs the roughly 11 roster on hard clips). This is
the ID-inflation lever independent of detection.

### A5 [MED] (S) depth-aware colocated-merge radius (replace the flat 0.75 m).
Make the colocated radius a function of the projecting camera's depth-uncertainty. Clears the residual
coloc pairs.

### A6 [MED] (S) cap the cross-space stitch budget in absolute metres, not gap-scaled.
A long occlusion currently licenses a long cross-field stitch. Add an absolute metres ceiling.

### A7 [HIGH] (L) lock global id per P2 tracklet, not per frame. WITH A CAVEAT.
About 517 flicker events are intra-tracklet global-id flips. Caveat, learned this session: a naive
per-tracklet lock was rejected because it puts a stable wrong-person id on a player. Any lock must act at
the cross-camera assignment level, not as a post-hoc per-tracklet relabel. Do not repeat the simple lock.

### Flag cleanup [LOW] (S).
The 40-set flag A/B (methods_log Part A) showed `graph_shape_enabled` is fully inert on all 40 and
`graph_split_enabled` is a slight agreement drag. Candidate action, pending the human decision: remove or
disable the inert `graph_shape` and the slightly-negative `graph_split`, keep the teleport-suppressors
(distance-R, facing gate, adaptive lost window).

### OC-SORT ablation [LOW] (S).
OC-SORT is implemented and config-selectable but net-negative as a whole (methods_log Part A). If
revisited, disable the aggressive OCR recovery pass and keep only ORU and OCM, then re-A/B. Otherwise it
stays off.

### A11 pack-handling for in-pack peripherals. The quality floor.
`_6` is weakest. Close catchers enter P3 as low-parallax tracks. Ideas: pack-aware clustering, per-pack
chimera passes, stricter peripheral birth in packs.

### A12 flag-gated off, awaiting A/B (implemented, dormant).
- G1 Hartley conditioning and G3 parallax-ordered RANSAC (`triangulation.py`). A/B reproj p95 without
  coverage loss.
- Airborne pelvis-emit (`airborne_pelvis_emit`), emit-only. A/B teleports and jitter on jump clips.
- `density_lost_window`, cross-delivery prior calibration, `temporal_link_decay`.

### A13 other deferred experiments (decision-gated).
Skeleton-gait embedding cue (needs sign-off, weak cue), G4 inter-cue LLR correlation, per-track standing
height. F15 (3D-informed 05 costs) is subsumed by A0.

---

## 3. Research A/Bs pending

- A3 emit-velocity gate and IMPACT-2 partial-drop: 40-confirmed, off by default, awaiting the human keep
  decision and a before/after mosaic sign-off.
- Tiling plus A3 combined 40-set A/B (see A9).
- decide-in-3D A/B (pending A0), on 8 then 40.
- Identity ground truth: dropped. No ground truth exists or is planned; labelling and fine-tuning are off
  the table. Tuning is proxy-guided and the final judgement is human mosaic review.

---

## 4. Measured reference facts (do not re-derive)

- Reprojection: panel (RANSAC-inlier, pre-smoothing) 3.07 to 3.56 px; post-smoothing vs all confident 2D
  views mean 6.8, p95 24.5 px; hips worst (11 to 12 px, cross-view keypoint-definition difference). The
  "1 px" figure applies to calibration targets, not pose-model keypoints (own 2D noise 2 to 3 px).
- Pixel to ground metres: `reference/localization-error.md` (anisotropic, cross about 3 mm/px, depth
  25 to 60 mm/px at 5 to 9 degree grazing).
- Calibration: one session for both matches (team-confirmed), cm-accurate.
- Box-vs-local panel variance is expected, not a bug: the same code on the two P1 binaries (the L40S fp16
  fast path vs the laptop generic path) gives slightly different panels on the 8 overlap deliveries. Same
  algorithm, different detector and pose numerics upstream.
- Co-observing facing pairs are C1-C4, C2-C6, C3-C5 (they face each other from opposite sides, low
  parallax), not the diametrically-opposite positions. World frame z equals 0 is the ground.
- Per-camera reprojection (all 40): cam_06 8.5px (cleanest) to cam_07 12.9px (worst); only ~1.5x spread,
  no pathological camera. Leave-one-camera-out shifts the triangulated hip only 5.7-7.0 cm (robust to
  losing any one camera). A 2-view estimate differs from full multi-view by ~8 cm mean / 30 cm p95.
