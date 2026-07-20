# Methods log: the complete A/B ledger

Authoritative, structured record of every method tried on the multi-camera cricket 3D-pose and
identity pipeline, its before/after evidence, its pros and cons, its current status, and whether it is
on or off by default. This is the single authoritative method log: it combines and supersedes the former
`wip/methods_log.md` (3D location and identity methods), `docs/pipeline/fixes-log.md` (the v6 to v8.1 fix
campaign), the P1 model/detector study (Part E), and the 2026-07-16 session A/Bs (Part F). Performance
(speed) findings live separately in [`reference/performance.md`](reference/performance.md); the
forward-looking backlog is [`roadmap.md`](roadmap.md). Historical version names and file paths inside
quoted results are accurate as of each dated entry and are not rewritten.

## How to read this file

- Status vocabulary: ACCEPTED (default in the shipped pipeline), ENABLED-INCONCLUSIVE (on in the shipped
  config but effect not established on the test set), PENDING (built and measured, awaiting a human
  keep/enable decision), NEUTRAL (built, metric-neutral, kept as an off-by-default option), REJECTED
  (measured worse or unsafe, not used), OPEN (not yet built).
- Sign convention for deltas: unless stated, a delta is variant minus baseline. For "flag OFF" rows a
  positive agreement delta means turning the flag off helped.
- Every keep, enable, or disable decision is deferred to human review. This file reports measured facts
  and does not auto-verdict marginal results.
- Working standard: flag-gated, flags-off byte-identity proven by execution, A/B on all 8 benchmark
  deliveries (or all 40 for a production claim), accept only a significant generalized improvement with
  no clustering, identity, or same-camera-collision regression (collisions must stay 0 everywhere).

## Datasets and baselines

- 8_init: the 8 benchmark deliveries `CCPL080626M1_1_14_{1..7}` and `CCPL080626M2_1_12_1`.
- 40_full: the 40-delivery production set.
- Current baseline trees (on the L40S box, `~/bits-pose-data/derived/40_full/`): `pipetrack_v90` holds
  P1 (`00_inference`) only; `pipetrack_v91_base` is the 40-delivery all-flags-on chain built as the
  reference for the 2026-07-16/17 A/Bs.

---

## Master status table

| ID | Method | Stage | Best measured A/B | Status | Default |
|---|---|---|---|---|---|
| Cap fix | `graph_llr_positive_cap` 1.5 to 3.5 | 03 | 8 cap curve: agreement 0.880 to 0.916 (peak at 3.5); 40: 0.853 to 0.883; under-merge 16% to 7%; coloc 5 to 0; collisions 0 | ACCEPTED | on |
| A3 velocity gate | `emit_velocity_gate` | 05 | 40: emitted teleports 367 to 0, max 2224 to 11.9 m/s, no IDs lost | PENDING (40-confirmed) | off |
| IMPACT-2 partial drop | `drop_partial_singlecam` | 05 | 40: 13 ghost IDs dropped (462 to 449), agreement held, collisions 0 | PENDING | off |
| Refine stage | 07_refine (physics FK, hip de-wobble, low-conf refill) | 07 | fixes stretched-limb / backward-knee / hip-wobble on 3D | ACCEPTED | on |
| Visibility-aware re-lift | `relift` (per-joint conf-gated views; 1-view bone-length ray lift) | 07 | umpire legs (1-view) 2.6 m sideways to 0.9 m straight down; per-delivery reproj +1-3 px | ACCEPTED | on |
| Group smoothing (face/foot/wrist) | `face_window`/`foot_window`/`mid_window` | 07 | 14_1 head-rel face p95 10.0 to 3.1 mm; heel p95 81 to 3.7 mm; wrist p95 15 to 5.7 mm | ACCEPTED | on |
| Overall limb/root window sweep | `ma_limb_window` 5 to 9, `ma_root_window` 9 to 13 | 07 | 14_1: jitter mean 6.5 to 6.2 mm (flat), p95 18.6 to 20.7 mm (worse), reproj flat | REJECTED (saturated) | off (window 5/9) |
| One-Euro limb smoother | `limb_smoother: one_euro` (c1.5, b0.5) | 07 | 3-deliv avg: jitter mean 11.27 to 10.81 mm (-4%), p95 29.4 to 27.6 mm (-6%), reproj flat 11.65 px, +1.2% runtime | PENDING (subset) | off (moving_average) |
| Roles v1 solver | epoch Hungarian roles | 06 | core-role coverage 24/32 to 29/32 | ACCEPTED | on |
| Restructure | Halpe-26, single binding-keyed triangulation | all | base for v9; single 3D lift before identity | ACCEPTED | on |
| distance-R | `use_measurement_covariance` | 05 | 40 (flag OFF): dAgree -0.0013, +71 teleports | ENABLED-INCONCLUSIVE | on |
| facing gate | `graph_facing_gate_scale` 1.3 | 03 | 40 (flag OFF): dAgree +0.0010, +46 teleports | ENABLED-INCONCLUSIVE | on |
| adaptive lost window | `adaptive_lost_window` | 05 | 40 (flag OFF): dAgree -0.0002, +40 teleports | ENABLED-INCONCLUSIVE | on |
| pose-shape corroboration | `graph_shape_enabled` | 03 | 40 (flag OFF): zero change on all 40 (inert) | ENABLED-INCONCLUSIVE | on |
| chimera split | `graph_split_enabled` | 03 | 40 (flag OFF): dAgree +0.0032, -3 IDs (slight drag) | ENABLED-INCONCLUSIVE | on |
| Tiled detection | `--tiled-det` (RTMDet-m tiles) | P1 | 8-hardest: agreement +0.115 mean (all positive), teleports +704 sum, ~3x GPU cost | PENDING (two-edged) | off |
| OC-SORT tracker | `tracker: ocsort` (OCM/ORU/OCR) | 02 | 40: fragmentation -26, agreement -0.0129, teleports +151 | REJECTED as-is (net-negative) | off (bytetrack) |
| Pre-ID ground smoothing | `presmooth_ground_enabled` | 05 | 8: agreement flat, 14_5 teleports +4; 40: mean dAgree -0.0003, teleports +9, one clip -0.013 | REJECTED (inert) | off |
| 1A hip-to-ground emit | `emit_ground_source: triangulated_hip` | 05 | 8: teleports 33 to 32, agreement unchanged | NEUTRAL | off |
| 1C robust triangulation refit | `--tri-robust-refit` | 04 | 8: reproj p95 6.61 to 6.56 px | NEUTRAL | off |
| Kalman posterior emit | `emit_kalman_posterior` | 05 | active but ineffective as a teleport guard (teleports persist) | ACCEPTED (limited) | on |
| 1F single-view sticky-hip | `single_view_hip_fallback` | 05 | 8: teleports 33 to 35 (worse), p95 1.4 to 2.0 | REJECTED | off |
| Tracklet-id lock | per-tracklet id relabel | 05 | 8: 2D switches 65 to 39 but stable wrong-person id | REJECTED (code removed) | n/a |
| Emitted-foot temporal smoothing | `smooth_emit_feet` | 05 | 8: teleports 382 to 364, agreement unchanged | NEUTRAL | off |
| z0_reproj ground solve | `ground_fusion_mode: z0_reproj` | 05 | 8: position proxy 0.228 to 0.147 m (36% mean) | ACCEPTED | on |
| Inverse-covariance IRLS fuse | `robust_cov` | 05 | delivery 1: proxy 0.176 to 0.248 m (worse) | REJECTED | off |
| Foot-contact v2 | `foot_contact_mode: v2` | 05 | 8: proxy A +6%, proxy B -5% (mixed) | NEUTRAL | off |
| Single-camera height correction | ankle-height plane emit | 05 | not measurable by the proxies | NEUTRAL | off |
| One-Euro 2D stabilization | `--enable-stabilization` | 01 | ~32% less 2D jitter; helps hardest clip, trades `_5` | ACCEPTED | on |
| Native 26-keypoint triangulation | `--tri-native-skeleton` | 04 | complete 26-joint 3D on multi-camera frames | ACCEPTED | on |
| Cheirality gate | `--tri-cheirality` | 04 | fixed a silent behind-camera sign bug on this rig | ACCEPTED | on |
| Butterworth 3D smoothing | `--tri-smoother butterworth` | 04 | zero-phase smoothing, verified no lag | ACCEPTED | on |
| P1 colour-profile swap | RGB or grayscale P1 feed | P1 | within noise; grayscale slightly worse | REJECTED | off (BGR) |
| Tiled P1 (v8 era) | `--tiled-det --nms-thr 0.55` | P1 | +0.76 to +6.02 new boxes/frame, 0 lost | context (see tiling entry) | off in v9 |

Optimization methods (speed only) are recorded separately in
[`reference/performance.md`](reference/performance.md).

---

## Part A0: 2026-07-17 full-codebase audit campaign

A structural and hygiene campaign, not a tuning one: legacy YOLO stack deleted, the P1
runners refactored onto a shared module, the 1982-line renderer split, the global-id
package renamed to stage-05 terms (old YAML keys still load), whole-repo slop and jargon
purge, and a documentation refresh. Every refactor was verified byte-identical against a
golden re-run of stages 02 to 07 on delivery CCPL080626M1_1_14_1 (and a 40-frame GPU
check for P1). The full change ledger with verification evidence, and the open-defect register, are
maintained in the repo's internal `wip/` working notes (not part of this hand-over documentation).

One deliberate metric-definition change shipped (nothing else moved any output): the
global-id diagnostic "feet unusable" guard required exactly 17 confidence values, so
with Halpe-26 input the ankle check never ran and every bottom-of-frame bbox was
anchored on the upper-body plane. Fixed to length-aware. Emitted tracks and
predictions are byte-identical; only the agreement diagnostic's anchoring changed.
Measured on the golden delivery: cross-camera agreement 0.9327 to 0.9297 (agreeing
pairs 14572 to 13634), teleports/ids/persistence/verdict unchanged. Agreement numbers
after this date are on the corrected definition and read about 0.003 lower on this
delivery; do not interpret that step as a regression.

## Part A: 2026-07-16 and 2026-07-17 session

This is the most recent work. It follows the pipetrack_v9 push and runs entirely on the L40S box.

### A/B environment

All identity A/Bs re-run only the stages a change touches, from the frozen `pipetrack_v91_base` upstream,
at `--jobs 8` (the box has 8 CPU cores) with BLAS threads capped to 1 per subprocess. Detector A/Bs
re-run P1 on the L40S and chain 01 to 05 on the new detections. Every metric is compared to the
byte-identical `pipetrack_v91_base` baseline.

### 40-set flag verification (the first hard-set measurement of the shipped flags)

Five flags ship on in the production config but were only ever measured on the easy 8-clip set, where
they were inert or noise-level. This is their first measurement on all 40 deliveries. Each was turned off
one at a time; delta is flag-OFF minus flag-ON (a positive agreement delta means the flag was hurting).

| Flag turned OFF | mean dAgreement | sum dTeleports | sum dIDs | collisions | coloc |
|---|---|---|---|---|---|
| `graph_shape_enabled` | +0.0000 | +0 | +0 | 0 | 0 |
| `graph_split_enabled` | +0.0032 | +0 | -3 | 0 | 0 |
| `graph_facing_gate_scale` | +0.0010 | +46 | +2 | 0 | 0 |
| `use_measurement_covariance` (distance-R) | -0.0013 | +71 | +2 | 0 | 0 |
| `adaptive_lost_window` | -0.0002 | +40 | +0 | 0 | 0 |

Findings, neutral:
- `graph_shape_enabled` is fully inert on all 40 (zero change), not just on the easy 8. It is doing
  nothing on this data. Pro: harmless. Con: dead weight and a maintenance surface.
- `graph_split_enabled` is a slight agreement drag here (turning it off nudged agreement up 0.0032 and
  removed 3 IDs). Pro: conservative, no collision risk. Con: no measured benefit on this set.
- `use_measurement_covariance`, `graph_facing_gate_scale`, and `adaptive_lost_window` each suppress real
  underlying teleport events (71, 46, 40 across 40) at negligible agreement cost. Pro: teleport
  robustness. Con: the teleport counts are underlying events; the A3 emit-gate masks the visible ones
  regardless, so their production value is robustness rather than on-screen marker count.
- Collisions and coloc held at 0 under every toggle. The hard invariant survived.

### Tiled detection (recall lever)

Context: the current detector is RTMDet-m at plain 640 with NMS 0.3 (production v90 ran tiling OFF). No
stronger detector weights exist on the box: the RTMDet-l/x, RTMO-l, and YOLO model directories contain
only placeholders. The only runnable recall lever with zero new assets is the already-implemented tiled
detector.

Recall-gap evidence (from existing P1 outputs, no ground truth): detections are large and high-confidence
(14_7: p50 confidence 0.84, p50 box height 452 px), with almost nothing borderline (3% below 0.4
confidence, 2% below 100 px). Missing distant and dark players score near zero rather than just under
threshold. This is a scale and resolution gap, not a threshold gap, so lowering the confidence threshold
would recover little; tiling, which re-scales distant subjects to the detector's trained size, is the
correctly targeted lever. Per-camera counts show cam_07 at roughly one person per frame and cam_02 and
cam_06 pinned at a constant count.

First pass (14_6, 14_7; tiled plus NMS 0.55 vs plain-640 NMS 0.3 baseline): agreement 14_6 0.906 to
0.919, 14_7 0.831 to 0.961 (+0.130). This changed two variables at once (tiling and NMS), so it was
isolated.

Isolation (plain-640 plus NMS 0.55, NMS changed alone): agreement 14_6 0.906 to 0.873, 14_7 0.831 to
0.790 (both worse), teleports roughly doubled, and it produced the most detections yet performed worst.
Conclusion: NMS 0.55 alone hurts; the win is tiling, not NMS; raw detection count is not the goal.

Broadened (tiled plus NMS 0.3, clean single-variable isolation, on the 8 hardest deliveries by baseline
agreement):

| Delivery | dAgreement | dTeleports | dIDs |
|---|---|---|---|
| M1_1_14_6 | +0.002 | -6 | +3 |
| M1_1_14_7 | +0.138 | -3 | -2 |
| M1_1_17_5 | +0.103 | +13 | +1 |
| M1_1_17_1 | +0.236 | +76 | -1 |
| M2_2_3_4 | +0.229 | +148 | -1 |
| M2_2_3_1 | +0.019 | +222 | +2 |
| M2_2_3_2 | +0.139 | +182 | +1 |
| M1_1_17_4 | +0.056 | +72 | +2 |
| Mean or sum | +0.115 (all positive) | +704 | mixed |

Status PENDING, two-edged. Pros: agreement improves on all 8 hardest deliveries (mean +0.115), the
benefit generalizes, and it is cleanly attributable to tiling. Cons: underlying teleport events regress
badly on crowded clips (+704 total, worst on the M2_2_3 group); detection recovery is only +2 to +4% and
does not land on the starved cameras; it costs about 3x GPU time (8.8 vs 28 fps). Collisions held at 0.
Important context: the teleports here are underlying events, and the A3 emit-gate masks the visible ones,
so the production reading of this tradeoff depends on whether A3 is enabled. Detection totals: 14_6
12964 to 13445 (+481), 14_7 12927 to 13196 (+269). Trees: `pipetrack_tiledAB` (tiled+NMS0.55),
`pipetrack_nms055AB` (plain+NMS0.55), `pipetrack_tiled03` (tiled+NMS0.3).

### OC-SORT tracker (stage 02)

Built OC-SORT as a config-selectable alternative to the ByteTrack-plus-constant-velocity-Kalman tracker,
targeting the documented fragmentation on sharp manoeuvres. Three mechanisms, all guarded so the
`bytetrack` default stays byte-identical: OCM (an observation-centric velocity-direction penalty added to
the association cost), ORU (on recovery after a gap, re-derive the Kalman state along a virtual straight
trajectory between the last real observation and the new one), and OCR (a second association pass matching
still-unmatched detections against each track's last observation rather than the drifted Kalman
prediction). Config keys in `configs/02_tracking.yaml` (`tracker`, `ocm_weight`, `ocm_delta_t`,
`ocr_enabled`, `ocr_cost_threshold`, `oru_enabled`); experiment config
`configs/experiments/02_tracking_ocsort.yaml`.

Byte-identity of the default: re-running stage 02 with `tracker: bytetrack` reproduced `pipetrack_v91_base`
exactly on 14_7 (agreement 0.831, teleports 36, IDs 12), confirming the additive changes are default-safe.

40-set A/B (OC-SORT vs the byte-identical bytetrack baseline):

| Metric | Result |
|---|---|
| p2_tracks (fragmentation proxy) | -26 (fewer per-camera fragments) |
| mean agreement | -0.0129 |
| sum teleports | +151 |
| sum distinct IDs | +2 |
| collisions | 0 |

Status REJECTED as implemented. Pros: it did exactly what it targets, reducing per-camera fragmentation
by 26 across 40. Cons: the fragmentation reduction did not translate downstream: agreement dropped and
teleports rose. The likely mechanism is that the OCR and ORU recovery reconnects fragments across gaps and,
on the low-parallax facing-pair geometry, some reconnections are wrong-player merges that hurt cross-camera
agreement and spawn teleports. Worst clips: 17_2 -0.135, 2_3_1 -0.123, 14_7 -0.107. Possible follow-up:
disable the aggressive OCR pass and keep only ORU and OCM. Tree: `pipetrack_v91_ocsort`.

### Pre-identity 3D-ground smoothing (smoothing before identity instead of after)

Question tested: the physics-and-smoothing stage 07 refine runs after identity and roles. Would running
the 3D-ground smoothing before global identity give 05 a cleaner signal and improve cross-camera
association and agreement?

Verified first, because the answer hinges on it: stage 05 does not associate on `pose_3d`. It tracks each
P3 binding's ground foot position (`ground_xy`, the z0 reprojection solve from 03) with a Kalman filter
(`track_manager.py`). The triangulated pelvis (`pelvis_ground_xy`) that 04 emits is not consumed by 05's
tracking (that is the unbuilt decide-in-3D item A0). So the only way smoothing before identity can affect
association is by smoothing the ground position 05 tracks.

Implementation (flag-gated, off by default, byte-identical off): `presmooth_ground_enabled` in
`configs/05_global_id.yaml` (`presmooth_ground_cutoff_hz` 3.0, `presmooth_ground_min_frames` 16). Before
the tracking loop, each binding's `ground_xy` trajectory is grouped by `binding_id` (the pre-identity
cluster key, so no two players mix) and low-pass filtered with the same zero-phase Butterworth used by 07
refine's root de-wobble (`_presmooth_binding_ground` in `p5_global_id/runner.py`). Experiment config
`configs/experiments/05_global_id_presmooth.yaml`.

Byte-identity of the default verified: the control (flag off) reproduced the v90 05 metrics exactly on the
8_init set.

Result, status REJECTED (inert):
- 8_init (9 deliveries): agreement unchanged on all; teleports unchanged except 14_5 (+4, worse);
  collisions 0; IDs unchanged. The smoothing was confirmed active (about 10 binding trajectories smoothed
  per clip).
- 40_full: mean agreement -0.0003 (noise-level, slightly negative), teleports sum +9, distinct IDs 0,
  collisions 0. Only 3 of 40 deliveries moved agreement beyond noise, the largest being 16_5 at -0.013 (a
  regression).

Mechanism (why it is inert): 05's Kalman filter already low-pass filters the measurement internally, so
pre-smoothing the input to it is largely redundant and does not flip the discrete association decisions
that agreement depends on. The residual teleports are id-level mis-associations (the wrong binding is
grabbed), not position noise, so a smoother position does not fix them. Corollary: smoothing `pose_3d`
before vs after identity makes no difference to agreement either, because 05 does not read `pose_3d` at
all. The lever that would actually let a smoother 3D help association is to make 05 decide on the 3D
(a 3D pose-shape cue or 3D-position tracking), which is the decide-in-3D work (A0 in the backlog), a
larger change than reordering the smoothing. Tree: `pipetrack_v91_presmooth`.

### Script optimization pass

Six fixes applied to the run scripts before any A/B, verified by compile and dry-run. Full detail in
`reference/performance.md`. Headline: the data-parallel P1 launcher
(`run_phase1_parallel.py`) had a broken runner path and failed on every shard; it is fixed and dry-run
validated, restoring the roughly 2x GPU-throughput lever. Plus thread-oversubscription fixes in the
render and P1 shards for the 8-core box.

### Earlier this cycle (carried from the pre-v9 push, verified on 40)

- Cap fix `graph_llr_positive_cap` 1.5 to 3.5: the facing-pair under-merge fix. On 8_init the cap curve
  moves agreement from 0.880 (cap 1.5) to 0.916 (peak at 3.5); on 40_full 0.853 to 0.883; central-player
  under-merge 16% to 7%; coloc 5 to 0; collisions 0. The separate 0.782 to 0.916 figure quoted in the
  presentation is the full V8.0-to-current delta across the whole stack, not the cap fix alone.
  ACCEPTED, on. The single largest agreement lever of the campaign. Pro: one line, reversible, over-merge
  guards untouched. Con: only partially closes the facing-pair under-merge (about 7% residual), and it is
  a blunt global cap rather than per-pair adaptive.
- A3 emitted-track velocity gate `emit_velocity_gate`: 8-set teleports 33 to 0, 40-set 367 to 0, max 2224
  to 11.9 m/s, no IDs lost, agreement unchanged. PENDING (40-confirmed, off by default, awaiting the human
  keep decision). Pro: removes all emitted teleports with zero measured collateral; drop-only so it can
  never move a marker to a wrong place. Con: a symptom fix; the id-level mis-association behind the
  teleport still exists (the deeper fix is decide-in-3D).
- IMPACT-2 `drop_partial_singlecam`: drops head-only or cut-off single-camera ghost IDs at emission. 40:
  13 dropped, agreement held, collisions 0. PENDING. Drop-only, same safety class as A3.
- `emit_kalman_posterior`: active but ineffective as a teleport guard (an earlier no-op claim was
  retracted after an isolated off-vs-on A/B showed it does change the emission, but teleports persist with
  it on). The effective fix is the A3 emission velocity gate.

---

## Part B: 3D ground-location methods (M0 to M11)

Source: the former `wip/methods_log.md`. Evaluated against a fixed baseline across the 8-delivery set.
Accuracy proxies: proxy A is distance from the emitted ground point to a RANSAC-triangulated foot
reference (metres, trusted on 3-plus-camera low-reprojection clusters); proxy B is reprojection error of
the emitted point against the foot pixels (px).

- M0 calibration audit: ball reprojection mean 1.2 to 1.9 px, p95 2.8 to 4.5 px. VALIDATED. Calibration
  is accurate enough to use directly; the location errors come from 2D foot estimation, single-camera
  grazing projection, and fusion choices, not calibration.
- M1 baseline characterization: single-camera rate 0.39 to 0.61, teleports 7 to 171 per delivery (382
  total), agreement 0.50 to 0.98, collisions 0. BASELINE.
- M2 inverse-covariance IRLS fusion (`robust_cov`): delivery 1 proxy A 0.176 to 0.248 m (worse), p95
  0.376 to 0.516 m, improved only 19.8% of clusters. REJECTED. The covariance model pulled estimates
  toward near cameras; with accurate calibration the reprojection-optimal point beat the
  covariance-weighted average.
- M3 ground-plane robust reprojection solve (`z0_reproj`): proxy A 0.228 to 0.147 m across all 8 (36%
  mean), proxy B 104.9 to 84.5 px (19%), clustering byte-identical. ACCEPTED, on. The main location
  improvement. Con: absolute reprojection stays high where the input foot pixels disagree across cameras.
- M4 foot-contact v2: proxy A +6% over z0 but proxy B -5% (mixed). NEUTRAL, opt-in. First wiring leaked
  the new foot into the clustering gate and moved cluster counts 23 to 25%; decoupled so it affects only
  the emitted position.
- M5 single-camera height correction: physically correct (shifts the single-view ground intersection
  about 0.94 m mean), but not measurable by proxy A (needs 3-plus cameras) or proxy B (self-consistent
  for one camera). NEUTRAL, opt-in.
- M6 temporal smoothing of emitted foot pixels (`smooth_emit_feet`): teleports 382 to 364, agreement
  unchanged. NEUTRAL, opt-in. Supports the finding that residual downstream failures are identity
  assignment, not location noise.
- M7 bird's-eye-view visualization: field-plausible, stable player positions. DELIVERED. Confirmed the
  remaining teleport-like behaviour is mostly identity flicker, not location error.
- M8 emit Kalman posterior instead of raw observations: trajectory jitter p95 roughly halved, worst
  emitted jump 14.0 m to 0.36 m on the hardest case, collisions 0. ACCEPTED, on. Note the separate
  finding (Part A) that as a per-frame teleport guard the posterior is ineffective; its accepted value is
  trajectory stability.
- M9 v4 mosaics with a BEV panel: DELIVERED.
- M10 P1 colour-profile A/B: RGB swap within noise, grayscale slightly worse, detection count unchanged.
  NO CHANGE, BGR kept.
- M11 full 26-keypoint (originally 17) multi-view triangulation and occlusion fill: complete multi-joint
  3D on all multi-camera frames, per-joint reprojection p50 2 to 4 px, p95 5 to 7 px. ACCEPTED for the 3D
  pose deliverable. Remaining gap: the roughly 39% single-camera frames, which cannot be triangulated
  (the single-view PnP lift, A8 in the backlog, is the planned fix).

---

## Part C: identity methods (ID-0 to ID-6)

Source: the former `wip/methods_log.md`. All behind config flags; flags off reproduced the committed
baseline byte-for-byte. Headline (baseline to the v5 identity stack, all 8): distinct-ID counts moved
toward the 13 to 15 roster on every delivery, teleports fell on every delivery, agreement rose on the
worst clip (`_7` 0.498 to 0.600), collisions stayed 0.

- ID-0 identity baseline: agreement 0.50 to 0.98, distinct IDs 11 to 25, teleports 7 to 171, collisions
  0. BASELINE. The hard clips over-segment and under-merge.
- ID-1 corroboration-aware cross-camera merge plus parallax-adaptive facing gate: `_7` agreement 0.498 to
  0.600 (+0.102), teleports -13, single-camera rate -0.051. ACCEPTED. The largest lever for cross-camera
  under-merge; easy clips byte-identical.
- ID-2 P4b stitching v2 (pose-gated fragment merge): enabled stitching that previously selected zero
  links; merges same-build fragments only. ACCEPTED. Root cause was a dummy new-trajectory cost that
  undercut every real stitch.
- ID-3 cardinality prior (drop IDs whose whole-clip span is under 30 frames): distinct IDs collapse
  toward roster on all 8, teleports fall on every clip. ACCEPTED. The dominant ID-count and teleport
  lever; conservative (only 6 to 25-frame fragments dropped, no full-clip player lost).
- ID-4 P4a lifecycle hardening (adaptive lost window, pose veto in the chi2 gate, descriptor-gated
  re-entry): small direct effect because the triangulated P4a pose descriptor needs parallax the facing
  pairs lack. ACCEPTED as a guardrail. Follow-up: feed the billboard posture descriptor into the veto.
- ID-5 ghost markers v2 and in-pipeline ghost verification: DELIVERED, verified visually.
- ID-6 identity ground truth: no labels exist and none are planned; the ground-truth evaluator was
  removed. All identity figures are proxies read jointly. OPEN and explicitly deferred.

---

## Part D: fix campaign (F0 to F15 and the waves)

Source: `docs/pipeline/fixes-log.md`. Condensed; the dated file holds the full per-fix detail.

- F0 pipetrack_v6.0 ground baseline: the first full chain from the RTMPose-X (Halpe-26) P1 data. IDs 10
  to 18, agreement 0.60 to 0.92, 3D reprojection 3.2 to 3.6 px, collisions 0. BASELINE.
- F1 wire One-Euro 2D stabilization: helps the hardest clip's identity structure, trades `_5` agreement
  for `_7`. Later accepted as the v7 default. ACCEPTED, on.
- F3 cheirality gate: found and fixed a silent behind-camera sign bug on this rig's world handedness.
  ACCEPTED, on.
- F4 Halpe-26 feet as ground contact (v3 foot mode): implemented, emit-path only.
- F6b billboard posture to P4a teleport veto: no effect as built (teleport sources are unbound
  single-camera clusters that carry no posture). Folded into later waves.
- F7 zero-phase Butterworth 3D smoothing: verified no lag. ACCEPTED, on.
- F9 P3.5 binding-keyed lift plus covariance plus purity: the lift running in-pipeline for the first
  time; measures chimera suspects with per-camera bias attribution.
- F10 per-measurement (uncertainty) Kalman R: symmetric use raised teleports (wide R loosens the
  admission gate); resolved with asymmetric R (gate on the conservative role R, update on the measurement
  R).
- F11 pose-shape (bone-ratio) as the primary P3 cue: self-calibrated then abstained on all 8 (d-prime
  under 0.5); bone ratios do not separate players on this footage. Honest negative, matches the
  literature. The live shape path is the billboard-posture channel.
- F13 splittable clustering (purity-driven eviction): the mechanism works (chimera suspects drop to 1 to
  2 per clip, M2 id-persistence +0.074) but the configuration over-splits (agreement -0.08 to -0.20, IDs
  +2 to +4). Mechanism validated, aggressive config rejected; carried at conservative thresholds.
- v7-rc2 accepted as the v7 default (2026-07-11): the P4b stitcher selected links for the first time in
  project history; `_7` agreement 0.603 to 0.703 with IDs 18 to 13, `M2` id-persistence 0.699 to 0.956.
- W5 detector recall bake-off (2026-07-13): only tiling helps recall at the detector's trained scale
  (native hi-res misses people; tiles keep them at trained scale). t640 added +0.76 to +6.02 boxes/frame
  with zero lost boxes. This is the same tiling lever re-measured downstream in Part A.
- W5B contested-camera down-weighting: no change on current data because P1's detector NMS (IoU 0.3)
  already deletes one of any two overlapping same-camera boxes (max same-camera IoU 0.298 on `_7`).
  Machinery kept, flag-gated, byte-identical off.
- Roles v1 solver (Vedant drop, 2026-07-13): epoch Hungarian role solver, defects fixed, merged.
  Core-role coverage 24/32 to 29/32. ACCEPTED as the v7 default. Later W8 roles v1.2 added a
  bowling-end auto-flip (run-plausibility band 3.0 to 9.5 m/s, pre-shot cost-flip fallback); on the v8.0
  set the full core roster plus both umpires were named on all deliveries.
- v8.0 accepted (2026-07-13, tiled detection era): P1 tiled RTMDet-m plus NMS 0.55, no-spawn P2, v7
  identity stack, roles v1.1, W6 peripheral suppression on. Hard-clip transformations: `_4` agreement
  0.770 to 0.972, `_7` 0.703 to 0.811, `M2` 0.781 to 0.886; mean agreement flat (0.783 to 0.782) because
  tiling makes about 20% more real people visible as genuinely hard tracks; core-role identity equal or
  better on every delivery.
- W6 role-aware peripheral suppression (`suppression_enabled`): per-global-id quality aggregates drop
  low-confidence non-core peripherals at output; core roles never suppressed. ACCEPTED as a v8 default;
  flag-off byte-identical.
- W9 union-lift merge plus colocated-id merge (v8.1, 2026-07-14): the split-identity / ghost-swap fix.
  `graph_union_lift_merge` (03) adjudicates co-located cluster pairs by triangulating the union of both
  clusters' views (one coherent low-residual 3D skeleton in every view means one person);
  `p4b.colocated_merge` (05) merges two emitted IDs co-located 25-plus frames within 0.75 m that never
  share a camera-frame. 8-set A/B: `_7` agreement 0.811 to 0.962, `_6` 0.477 to 0.625, mean 0.782 to
  0.834, coloc pairs to 0 everywhere, collisions 0. ACCEPTED, on. This is the immediate predecessor stack
  to the pipetrack_v9 restructure and is why coloc pairs are near zero today.
- 40-delivery production record (v8.1, 2026-07-14): mean agreement 0.862 (range 0.527 to 0.992),
  reprojection 3.07 to 3.56 px on every delivery, collisions 0 everywhere, coloc 0 on 38 of 40. The
  reference production panel that pipetrack_v9 built on.

### Detector literature check (July 2026, carried forward)

Relevant to the tiling and detector work in Part A. No drop-in detector beats RTMDet decisively inside
mmdet at this resolution; the dominant recall lever is input resolution via tiled (SAHI-style) inference
over the existing RTMDet, before any model swap. RTMO is demoted to a detector-miss fallback because it
is COCO-17 only and would drop the Halpe-26 feet that the ground-contact and stature cues depend on.
Identical-kit appearance ReID remains unsolved in 2026 (jersey number and colour are dead here). This is
consistent with the Part A finding that no stronger detector weights are worth fetching before tiling is
settled.

---

## Part B: 2026-07-18 session — 07 refinement jitter smoothing + reprojection metric

Follows the manager rejection of jittery/stretched 3D output. The re-lift (umpire fix) and the
face/foot/wrist/hip group smoothing were already accepted; this session asked whether the *full-pose*
(major-limb) jitter could be reduced further, and added the reprojection-error metric to quantify the
smoothness-vs-fidelity trade.

### Environment and caveats

Local runs on this laptop, `smoother: moving_average` base (scipy is broken locally, so the box's
zero-phase Butterworth base could not be exercised here — see the L40S box for the production Butterworth
path). Measured on 3 of the 8 benchmark deliveries — `CCPL080626M1_1_14_1`, `CCPL080626M1_1_14_7`,
`CCPL080626M2_1_12_1` — not the full 8/40, so both entries below are subset-measured and PENDING a full
run + human keep decision. Jitter = mean/p95 frame-to-frame 3D joint displacement (metres). Reproj =
pixel gap when the refined 3D joint is projected into every camera that *reliably* saw it (2D conf >=
`vis_conf` 0.5), so a corrected hallucinated keypoint (the umpire's edge legs) is never scored as a
regression. Both metrics are emitted per delivery in `07_refine/refinement_metrics.json`.

### New metric: reprojection error (baseline stage-04 3D vs refined 3D)

| delivery | jitter mean mm (before to after) | jitter p95 mm | reproj px mean | reproj px p90 |
|---|---|---|---|---|
| 14_1 | 9.2 to 6.5 (-29%) | 33.7 to 18.6 (-45%) | 7.9 to 11.1 | 16.7 to 21.2 |
| 14_7 | 11.1 to 9.9 (-11%) | 33.0 to 26.9 (-18%) | 8.0 to 10.3 | 17.1 to 18.9 |
| M2 12_1 | 21.1 to 17.4 (-18%) | 53.8 to 42.8 (-20%) | 12.3 to 13.5 | 23.3 to 24.0 |

The +1-3 px reproj rise is the cost of the physics constraints (anatomically-constant rigid bone lengths
cannot match the per-frame apparent bone lengths of the noisy 2D). It is **independent of the temporal
smoother**: in A/B B1 below the reproj stayed at 11.6 px whether smoothing was light or heavy, so the
temporal smoothing is fidelity-free and the whole +3 px is the rigid-bone trade the manager mandated.

### A/B B1 — overall smoothing window sweep (does cranking the global smoother help?)

Base moving-average window on all limb-bone directions and the root trajectory, swept up. Delivery 14_1:

| setting (limb / root window) | jitter mean mm | jitter p95 mm | reproj px |
|---|---|---|---|
| 5 / 9 (current) | 6.5 | 18.6 | 11.1 |
| 7 / 11 | 6.3 | 18.9 | 11.1 |
| 9 / 13 | 6.2 | 20.7 | 11.1 |

Finding: saturated. Heavier windows move the mean by <0.3 mm and *worsen* p95 (over-smoothing artifacts)
while reproj is flat. A per-joint ranking of the residual jitter showed it is concentrated in the major
limbs (thigh `knee<-hip` 11-12 mm, upper arm `elbow<-shoulder` 10-13 mm, shank/forearm) — segments that
genuinely swing fast during a batting/bowling action. So the residual is largely *real motion*, not noise;
a fixed low-pass cannot remove it without lagging the swing. **REJECTED** (no gain), window stays 5/9.

### A/B B2 — One-Euro adaptive limb smoother vs the fixed moving average

One-Euro (Casiez et al., CHI 2012): a low-pass whose cutoff rises with joint speed, so it smooths hard
when a joint is still and stays responsive when it moves fast — exactly the lever B1 lacked. Implemented
bidirectionally (`one_euro_smooth`) for zero-phase, replacing only the base limb-direction smooth
(`limb_smoother: one_euro`); the face/foot/wrist group overrides are unchanged. Average over the 3
deliveries (all AFTER):

| base limb smoother | jitter mean mm | jitter p95 mm | reproj px | wall time / delivery |
|---|---|---|---|---|
| moving average (current) | 11.27 | 29.43 | 11.65 | 16.8 s |
| One-Euro c2.0 b0.7 | 11.02 | 28.45 | 11.63 | 17.4 s |
| One-Euro c1.5 b0.5 | 10.81 | 27.64 | 11.65 | 16.3 s |
| One-Euro c3.0 b1.0 | 11.33 | 29.49 | 11.63 | 16.3 s |

Best is c1.5/b0.5: jitter mean -4%, p95 -6%, reproj unchanged, no lag (One-Euro preserves fast motion by
design). c3.0 is too responsive (== MA); c2.0 in between.

### Compute cost vs output improvement

Isolated micro-benchmark of the pure smoother on a representative sequence (600 frames x 25 bone-direction
channels x 3 axes, 20 reps):

- moving average: **2.86 ms** / player-sequence (vectorized convolution).
- One-Euro bidirectional: **200.5 ms** / player-sequence — **~70x** slower, because it is a sequential
  per-timestep Python loop (vectorized only across the 3 axes) run twice (forward+backward).

In absolute terms that is +197 ms / player, ~1 s / delivery for ~5 players, which is **~1.2% of the
~16 s/delivery end-to-end refine cost** (the re-lift re-triangulation dominates). So the end-to-end wall
time is unchanged within noise (16.3-17.4 s), but note the raw-smoother cost is 70x and would become the
bottleneck if the re-lift were ever optimized or skipped.

Verdict, neutral: a small but effectively-free (in the current pipeline) jitter reduction with no fidelity
loss. **PENDING** — subset-measured (3/8), and per the working standard the keep/enable decision is
deferred to human review after a full 8 (and ideally 40) confirmation on the box with the production
Butterworth base. Default stays `moving_average`.

**Human verdict (2026-07-18): keep the current default as-is — do NOT enable One-Euro, and the window
sweep stays rejected.** The One-Euro code remains in place as an off-by-default option (`limb_smoother:
one_euro`) to revisit only if a full 8/40 run on the box shows the -4/-6% jitter holds and the team wants
it. Nothing about the shipped 07_refine changes: the accepted core (physics FK bone-length + anatomical
limits, visibility-aware re-lift, face/foot/wrist/hip group smoothing, hip stabilization) stays on.

---

## Rejected, with the reason (do not revisit without a new signal)

- Tracklet-id lock (per-tracklet id relabel): stabilized IDs by putting a stable wrong-person id on a
  player, a regression the baseline never had. Code removed. Any flicker fix must act at the cross-camera
  assignment level, not as a post-hoc per-tracklet relabel.
- 1F single-view sticky-hip lift: raised teleports (33 to 35) and p95 (1.4 to 2.0); a single-view hip on
  a sticky plane swings with torso lean, noisier than the foot.
- M2 inverse-covariance IRLS fusion (`robust_cov`): worse than the reprojection-optimal solve on accurate
  calibration.
- NMS 0.55 alone (without tiling): hurt agreement on both tested clips and roughly doubled teleports
  despite producing more detections.
- OC-SORT as implemented: net-negative downstream (fixes fragmentation but hurts agreement and teleports).
- P1 RGB or grayscale colour swap: within noise or slightly worse.

---

## Open levers not yet built or measured

See the roadmap [`roadmap.md`](roadmap.md). The highest-value items are
decide-in-3D consumption in stage 05, single-view PnP lift for the roughly 39% single-camera frames, the
05b stitching under-merge fix, and identity ground truth (currently dropped, all metrics are proxies).

---

## Part E: P1 model & detector selection (RTMPose-X, tiled RTMDet)

Closes the long-open P1 model study (written 2026-07-14). Sources: archived run docs in `runs/`
(rtmpose-l-body8-full-db32-pb96, rtmpose-x, yolo26x-pose-full-db8, bakeoff_w5), the W5/W5B-LIVE fix
entries, and meeting/production panels.

### Pose model: RTMPose-X (Halpe-26) is the accepted choice

| Aspect | RTMPose-L (body8, COCO-17) | RTMPose-X (body8, Halpe-26) |
|---|---|---|
| Skeleton | 17 joints, no feet | 26 joints: COCO-17 + head/neck/pelvis + 6 foot points |
| Ground contact | ankle-based only | heel/toe landmarks to `foot_contact_mode: v3` (F4) |
| Throughput (L40S, plain 640 det) | ~30+ fps | 27.5 fps (134k frames / 82 min) |
| Identity-era baseline | v5 stack: `_7` agreement 0.498 to 0.600 | v6.0 ground baseline onward: all campaign gains built on X |

Decision drivers (in order):
1. **Feet.** The entire ground-position channel (z=0 reprojection solve, F4 heel/toe contact,
   billboard posture anchoring) improves with real foot landmarks. Only the X body8 model
   ships Halpe-26; there is no COCO-17-only reason to stay on L.
2. **Accuracy-first mandate**: X is the accuracy flagship; the throughput delta (~10%) is
   irrelevant next to identity quality on this project.
3. The per-metric L-vs-X ablation on identical downstream configs was **never run in
   isolation**, the v5(L) to v6.0(X) jump changed model and campaign era together. If a clean
   ablation is ever wanted: run the v8.1 chain on an L-based P1 tree and diff the panel. Not
   currently justified, every accepted result since v6.0 is X-based.

RTMO (one-stage) was evaluated on paper and **rejected**: COCO-17-only heads would lose the
feet (W5 research note).

### Detector findings (the part that actually moved the needle)

The detector, not the pose model, was the upstream bottleneck:
- **YOLO26x-pose** (`yolo26x-pose-full-db8`, kept in `data/derived/runs/`): historical
  comparison run; not adopted (RTMPose mandate + top-down pipeline).
- **Bake-off (`runs/bakeoff_w5/`)**: tiled RTMDet-m @640 beat native hi-res decisively —
  RTMDet only detects at its trained object scale (m1280/m2560 lost boxes; t640 was a strict
  superset, min box height 33 to 12 px). RTMDet-L @1280 marginal.
- **NMS 0.55** (from 0.3) lets both crossing players survive: +0.10-0.13 cross-camera
  agreement, the single largest identity gain of the campaign.
- Accepted production P1: tiled RTMDet-m 4×2 + full frame, NMS 0.55, IoM 0.7, fp16 fast path
  (worker-side crop prep): **18-25 fps for 9× detector work** on the L40S.

Open follow-up (see [`roadmap.md`](roadmap.md) A9): YOLO26-l / RF-DETR recall-oracle probes through the
same bake-off harness; clean L-vs-X ablation only if someone needs the number.

---

## Part F: Stabilization-order A/B (2026-07-16) — keep stab-first

Answers the recurring question ("stabilize the 2D first, or triangulate first then smooth the 3D?").
8_init, isolating only the ordering:

| | ARM A: stab-first (current) | ARM B: raw to 3D-smooth |
|---|---|---|
| cross-camera agreement | **0.9160** | 0.9114 |
| 3D-joint jitter | **0.0105 m** | 0.0117 m |
| teleports | **258** | 280 |
| reproj mean / p95 | 3.27 / 6.45 | 3.28 / 6.47 |

**Verdict: keep the current stab-first ordering** — 2D-stabilize-before-triangulate is better on every
axis (smoother 3D, higher agreement, fewer teleports). Removing per-view pixel jitter *before*
triangulation prevents the 3D depth-swimming that a post-hoc 3D smoother cannot fully recover. Validates
the status quo; no change to enable. (The related single-view sticky-hip lift and tracklet-id lock from
the same session are in the Rejected section above; the 1F teleport-source finding — 88% of emitted
teleports are single-camera and the `max=1220 m/s` outlier is position-source-invariant, so the real
lever is the A3 emission velocity gate, not a better hip source — motivates A3 in [`roadmap.md`](roadmap.md).)
