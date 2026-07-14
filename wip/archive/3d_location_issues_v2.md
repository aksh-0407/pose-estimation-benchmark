# 3D Location Issues — v2 (post-polish critical review)

Written as a critical read of the **v4** results (z0_reproj + Kalman-posterior emit + foot-v2 +
full 17-keypoint triangulation with occlusion fill), across all 8 deliveries. v1
(`3d_location_issues.md`) catalogued the original 12; several are now fixed. This v2 states what is
**resolved**, what **remains**, and — for each open issue — the **evidence**, **root cause**, and
**fix direction**. Companion: `id_issues.md` (identity, which is now the dominant ceiling).

## 0. Evidence (v4, all 8 deliveries)

| Delivery | single-cam rate | cluster cyc-consistency | 3D-pose accuracy (M11) | jitter p95 (m/f) |
|---|---|---|---|---|
| M1_1_14_1 | 0.39 | 0.82 | per-joint reproj 2–4 px; 100% multi-cam complete | ~0.04 |
| M1_1_14_2 | 0.48 | 0.85 | — | ~0.04 |
| M1_1_14_3 | 0.52 | 0.90 | — | ~0.04 |
| M1_1_14_4 | 0.51 | 0.85 | — | ~0.04 |
| M1_1_14_5 | 0.56 | 0.72 | — | ~0.05 |
| M1_1_14_6 | 0.46 | 0.88 | — | ~0.05 |
| M1_1_14_7 | 0.52 | 0.70 | — | ~0.07 |
| M2_1_12_1 | 0.61 | 0.68 | — | ~0.09 |

Calibration is cm-accurate (ball reproj ≤4.5 px p95) and, where ≥2 cameras see a player, full-body
triangulation is 2–4 px per joint. **The location machinery is sound; the losses are coverage and
input.**

## 1. RESOLVED in v4 (evidence)
- **R1 — emitted ground position accuracy.** z0_reproj: distance-to-triangulated-foot 0.211 → 0.147 m
  mean over 8 (−36%), largest on the hardest deliveries. (v1 ISSUE-2/3.)
- **R2 — emitted-trajectory jitter / teleport-jumps.** Kalman-posterior emit: disp p95 ~halved; worst
  emitted per-frame jump 14.0 → 0.36 m on _7; collisions 0. (v1 ISSUE-5, part of ISSUE-9.)
- **R3 — calibration suspicion.** Refuted; calibration is cm-accurate. (v1 ISSUE-6.)
- **R4 — full-body 3D pose.** All 17 keypoints triangulated (2–4 px) + occlusion fill → 100% of
  multi-camera frames emit a complete 3D pose (was 0 — P6 never ran). (v1 ISSUE-3 fully; new capability.)

## 2. OPEN issues (ranked by impact on 3D location)

### V2-L1 — ~50% of player-frames are single-camera → NO triangulation, weakly-corrected position  ★★★
- **Evidence:** single-camera rate **0.39–0.61** across 8; P6 shows only **61%** of player-frames have
  ≥2 views (3238/5301 on delivery 1). Single-camera clusters get a lone ray→z=ankle-height point
  (M5) with no cross-camera check; the ~39% single-camera frames get **no 3D pose at all**.
- **Root cause:** (a) the co-observing pairs are the **low-parallax facing pairs** (C1↔C4, C2↔C6,
  C3↔C5), where the cross-camera *association* is hardest, so many detections never get a second view
  bound to them; (b) genuine occlusion / limited per-camera FOV; (c) tight facing-pair ground gate
  (2.5 m) can split a correct 2-view merge, dropping it to single-camera.
- **Fix direction:** (1) **single-view pose reconstruction** — fit the identity's canonical 3D skeleton
  (from its multi-view frames) to the lone 2D view at its z0 ground position (PnP-style), so a
  single-camera player still gets a plausible full 3D pose; (2) raise the *multi-camera binding rate*
  in P3 (this is really an identity problem — see `id_issues.md` ID-1); (3) widen/adapt the facing-pair
  gate using the per-view ground covariance rather than a hard 2.5 m.

### V2-L2 — 10–32% of ≥3-view clusters fail reprojection cycle-consistency  ★★
- **Evidence:** `cycle_consistency_rate` **0.68–0.90** (worst _5 0.72, _7 0.70, M2 0.68). A failing
  cluster means the members do **not** triangulate to a single consistent 3D foot within 12 px.
- **Root cause:** either (a) a **chimera** — two different people merged (an identity error), or (b)
  a badly wrong member foot pixel. The emitted median/z0 point for such a cluster is between two
  bodies and meaningless.
- **Fix direction:** (1) use the M11 full-skeleton triangulation reprojection (not just the foot) as a
  stricter cluster-acceptance test — a chimera fails the torso/limb reprojection hard; (2) split
  clusters that fail cycle-consistency instead of emitting a blended point; (3) feed the pose-shape /
  anthropometric check as a merge veto (currently soft). Much of this is identity (see ID-2/ID-5).

### V2-L3 — flat z = 0 assumption for the ground point; airborne feet mislocated  ★
- **Evidence:** triangulated **ankle z p95 = 0.56 m** (p50 0.11 m) — a meaningful tail of raised feet
  (running stride, bowler load, jump). The P3/P4 ground point forces z=0, so an airborne foot's ray
  hits z=0 beyond the true position (metres at grazing angle).
- **Fix direction:** when the M11 skeleton is available, take the ground position from the triangulated
  **pelvis vertical-projection** (robust to a raised foot) rather than the z=0 foot ray; flag airborne
  frames (ankle z ≫ 0) and inflate their position covariance.

### V2-L4 — single-camera position accuracy is unverifiable by current proxies  ★★ (measurement gap)
- **Evidence:** proxy A (dist-to-triangulated-foot) needs ≥3 views; proxy B reprojection is ~0 for a
  self-consistent single view. So the ~50% single-camera positions are **unmeasured**. The M5 ankle-
  height correction is physically right (validated: triangulated ankle z ≈ 0.11 m) but its per-frame
  benefit on single-cam is not directly measurable.
- **Fix direction:** (1) hand-label a few dozen single-camera foot points against the triangulated
  position when the SAME player is briefly multi-camera (transition frames) → a real error sample;
  (2) the `eval_ground_accuracy.py` tool now reports the measurable multi-cam proxies per run — extend
  it with a transition-consistency check (does a track's position stay continuous when it drops from
  multi- to single-camera?).

### V2-L5 — foot-v2 / temporal smoothing are marginal and left opt-in  ★
- **Evidence:** foot-v2 +6% proxy A / −5% reproj (disagreeing proxies); F7 smoothing small on already-low
  jitter. Kept `default OFF`.
- **Fix direction:** revisit only if the single-view reconstruction (V2-L1) or a real error sample
  (V2-L4) shows they help; otherwise they are noise-level.

## 3. Verdict
The **3D location objective is largely met where the geometry is observable**: cm-accurate multi-view
pose, −36% ground error, halved jitter, 100% multi-cam pose completeness. The remaining location losses
are **coverage (single-camera, V2-L1) and cluster purity (V2-L2)** — and *both are fundamentally
identity/association problems*, not geometry. **The single highest-value next step for location is to
raise the multi-camera binding rate and split chimeras — i.e. fix identity (`id_issues.md`).** Pure-
geometry levers (EKF cov-R, offline smoothing) are now incremental.
