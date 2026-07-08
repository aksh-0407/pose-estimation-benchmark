# 3D Ground-Position Issues — PipeTrack v3

**Scope:** everything that degrades a player's **(X, Y) position on the pitch (z = 0 plane)** —
the quantity P3 consensus produces and P4a Kalman-tracks and emits as
`p4/diagnostics/ground_tracks.jsonl`. This is *not* about identity/global-ID (covered in
`implementation_plan.md`); it is about metric localisation accuracy.

**Method:** measured on the current on-disk results for **all 8 deliveries**
(`benchmarks/runs/pipetrack_v3/deliveries/*`) plus a full read of the ground-projection
code path. Every claim below is backed by a number from the runs or a `file:line`.

---

## 0. Evidence — the current state, measured

Per-delivery, from `p3/association_metrics.json`, `p3/diagnostics/correspondences.jsonl`,
and `p4/global_id_metrics.json`:

| Delivery | %single-cam clusters | multi-cam ground-spread p50 / p95 (m) | % multi-cam spread >2 m | % multi-cam reproj >12 px | reproj p95 (px) | X-cam agreement | teleports |
|---|---|---|---|---|---|---|---|
| M1_1_14_1 | 39% | 1.94 / 3.69 | 48% | 53% | 64 | 0.95 | 11 |
| M1_1_14_2 | 48% | 2.18 / 3.81 | 57% | 50% | 75 | 0.98 | 7 |
| M1_1_14_3 | 52% | 1.87 / 2.93 | 46% | 45% | 64 | 0.87 | 19 |
| M1_1_14_4 | 51% | 2.13 / 3.50 | 55% | 34% | 69 | 0.86 | 15 |
| M1_1_14_5 | 56% | 2.07 / 5.07 | 53% | 48% | 237 | 0.77 | 48 |
| M1_1_14_6 | 46% | 2.03 / 3.42 | 51% | 53% | 82 | 0.80 | 52 |
| M1_1_14_7 | 52% | 2.13 / 5.08 | 54% | 51% | 448 | **0.50** | 59 |
| M2_1_12_1 | 61% | 2.42 / 3.99 | 70% | 61% | 99 | 0.78 | **171** |

**What this says in one line:** roughly **half of all player observations are placed on the
pitch from a single camera** (no cross-camera correction), and among the observations that
*do* have two or more cameras, **the cameras disagree about where the player is standing by a
median of ~2 m** (p95 up to 5 m). ~half of multi-camera clusters reproject worse than the
12 px consistency tolerance, with p95 tails of 60–450 px (max seen: 4004 px in _7). Ground
localisation is currently **~1–2 m accurate at best, several metres at worst**, and the pipeline
does not know which is which.

Note also `p3` `cue_d_prime.appearance ≈ 0.0` in 5/8 deliveries — the appearance cue that is
supposed to help disambiguate is dead on this footage, so ground geometry is carrying almost
everything and its errors propagate directly.

---

## 1. How a ground position is computed today (the chain that can fail)

```
P1  bbox + 17 keypoints (2D, per camera)
      │
P2  ground_contact_pixel(bbox, ankles)                 scripts/tracking/jsonl_io.py:88
      │   → lower confident ankle, else bbox-bottom-centre   geometry.py:143
      │   → GroundPlaneCalibrator.image_to_ground_xy         tracking/calibration.py:20
      │        H = inv(P[:, [0,1,3]])   (z=0 homography)      geometry.py:188
      ▼   stored as detection.ground_xy
P3  _detection_ground_xy per member                     association/associator.py:177
      │   _ground_consensus_members = MEDIAN of member points   associator.py:460
      │   gate: max pairwise spread ≤ ground_cluster_gate_m (2.75)
      │   (≥3 views) RANSAC reproj ≤ cycle_reproj_tol_px (12)     associator.py:430
      ▼   correspondence.ground_xy  = that median
P4a SingerGroundKalman.update(correspondence.ground_xy) ground_kalman.py:102
      │   fixed R = measurement_noise² (0.3–0.4 m), role-invariant  ground_kalman.py:96
      ▼   per (id, frame) = MEAN of member ground obs          global_id/runner.py:230
P4b stitched ground = MEAN of segment points             global_id/runner.py:254
      ▼   ground_tracks.jsonl
```

Every "MEDIAN"/"MEAN"/"fixed R" in that chain is an unweighted step that throws away known
uncertainty. That is the through-line of the issues below.

---

## 2. Issues (ranked by impact on ground accuracy)

### ISSUE-1 — ~50% of positions come from a single camera with zero cross-camera correction  ★★★ (highest impact, most prevalent)
- **Evidence:** single-camera cluster rate **0.39–0.61** across deliveries (`single_camera_rate`);
  `cluster_camera_support` p50 = 1.0 in 5/8 deliveries.
- **Root cause:** a single-camera cluster's `ground_xy` is one ray-to-plane back-projection
  (`_detection_ground_xy` → `pixel_to_ground_xy`, `associator.py:186`). With the long broadcast
  lenses, a few-pixel foot error becomes **metres** on the ground, and there is no second view
  to correct it. These are then Kalman-tracked with the *same* small fixed R as a well-triangulated
  point (ISSUE-4), so the filter trusts them far too much.
- **Why so many are single-camera:** the facing (co-observing) pairs are also the low-parallax
  pairs; 7 of 21 pairs are flagged **degenerate** (`degenerate_pairs` incl. cam_02-cam_05,
  cam_03-cam_06, cam_05-cam_06), and ~30–36% of pair evaluations land on degenerate pairs
  (`degenerate_pair_usage_rate`). Where cameras can't be paired reliably, the player stays single-camera.
- **Fix direction:** (a) attach a per-observation ground covariance to single-camera points so
  downstream trusts them appropriately (the machinery exists — `ground_covariance`, `geometry.py:302`);
  (b) prefer the height-plane anchor (hips/shoulders) over bbox-bottom for close/cut-off single-camera
  figures; (c) let the Kalman coast on prediction rather than snap to a lone noisy ray.

### ISSUE-2 — cross-camera fusion is an unweighted median, ignoring the per-camera covariance the code already computes  ★★★
- **Evidence:** `_ground_consensus_members` (`associator.py:460`) returns
  `np.median(member_points)` — every camera weighted equally regardless of distance/grazing angle.
  Multi-cam ground-spread p50 ≈ 2 m, p95 up to 5 m (§0) shows members genuinely disagree at that scale,
  so an unweighted median is often 0.5–2 m off the best member.
- **Root cause / irony:** `association_mode: tracklet_graph` *does* build anisotropic ground
  covariances via `ground_covariance` for the identity **edges** (`tracklet_graph.py:336`,
  params `ground_sigma_px_*`), and a correct inverse-covariance combiner **already exists**
  (`fuse_ground_estimates`, `geometry.py:362`) — but the **emitted position** still uses the plain
  median. The uncertainty investment never reaches the output coordinate.
- **Fix direction:** replace the median in `_ground_consensus_members` with `fuse_ground_estimates`
  (inverse-covariance weighting) so the close, head-on camera dominates the distant grazing one;
  carry the fused 2×2 covariance forward as the P4 measurement noise (ISSUE-4).

### ISSUE-3 — the RANSAC/triangulated foot is computed for gating but discarded; the median is emitted instead  ★★
- **Evidence:** for ≥3-view clusters, `_multiview_reprojection_consistency` /
  `ransac_triangulate_point` compute a reprojection-consistent 3D foot (`associator.py:430,492`),
  but the value stored as `merged_point`/`correspondence.ground_xy` is the **median** from
  `_ground_consensus_members` (`associator.py:425,438`), not the triangulated point.
- **Impact:** even when a good multi-view estimate is available it is thrown away for gating only;
  the emitted coordinate is the coarser median. Also, **34–61% of multi-cam clusters exceed the
  12 px cycle tolerance** (§0) — i.e. the merged members are geometrically inconsistent — yet
  a median is still emitted for the 2-view clusters (the 12 px check only runs for ≥3 views,
  `associator.py:430`), so 2-view chimeras/misprojections pass through unmeasured.
- **Fix direction:** emit the RANSAC-triangulated foot (projected to z=0) as the position when it
  passes; fall back to covariance-weighted fusion; record per-cluster reprojection error on the
  emitted point (2-view included) so bad positions are flagged not hidden.

### ISSUE-4 — P4a Kalman measurement noise R is fixed and distance-invariant  ★★★
- **Evidence:** `SingerGroundKalman.R = eye(2) * measurement_noise²` set once at init
  (`ground_kalman.py:96`), `measurement_noise` = 0.3–0.4 m for every role
  (`p4_global_id.yaml:38-45`). Never updated per observation.
- **Root cause:** a foot 40 m away seen at a grazing angle has metres of ground uncertainty; a
  close head-on foot has ~0.2 m. The filter treats them identically, so it (a) snaps hard onto
  noisy far/single-camera measurements and (b) produces the teleports we see (7–171 per delivery;
  171 in M2_1_12_1). `changes_tbd` lists "distance-scaled measurement noise" — **not implemented**.
- **Fix direction:** feed the per-observation ground covariance (from ISSUE-2 fusion, or
  `ground_covariance` for single-camera) into `update()` as R. This is the single highest-leverage
  Kalman fix and directly attacks teleports.

### ISSUE-5 — emitted `ground_tracks` re-average across cameras and across stitched segments (unweighted, twice)  ★★
- **Evidence:** P4 accumulates per-(id, frame) as `np.mean` of member ground obs
  (`global_id/runner.py:230-233`) and again `np.mean` when merging stitched segments (`runner.py:254`).
- **Impact:** a third and fourth unweighted averaging on top of the median (ISSUE-2). If one camera's
  ray is 3 m off, it drags the emitted point ~1.5 m for a 2-camera id. Compounds ISSUE-2/4.
- **Fix direction:** use the Kalman posterior `pos_world_xy` as the emitted position (it already
  fuses over time and cameras optimally once R is correct), rather than re-meaning raw observations.

### ISSUE-6 — ~~calibration carries a 0.7–1.2 m same-player residual bias~~ → **REFUTED: calibration is excellent (~3 px)**  ✅ (corrected after audit)
- **What the audit actually shows (ran `audit_calibration('drive')`, 2026-07-08):** the **stored ball
  reprojection error** — the true quality signal, ball triangulated from these same 7 cameras — is
  **mean 1.2–1.9 px, p95 2.8–4.5 px, max ~5 px across all 8 deliveries**. That is sub-5-pixel,
  ~centimetre-level 3D. The extrinsics are **not** the bottleneck.
- **Correcting the earlier claim:** the `ground_var_floor_m: 0.4` config comment ("~0.7–1.2 m
  same-player residuals") attributes those residuals to "calibration bias", but the audit proves the
  calibration itself is tight. **Those residuals are from the 2D foot input (ISSUE-7) and the
  single-camera grazing-angle homography projection (ISSUE-1), not from the projection matrices.**
  The 892 px "projected-vs-stored delta" the audit prints is a **crop/normalization convention
  mismatch** in the stored ball artifact (audit warns this explicitly), *not* calibration error.
- **Consequence (this reframes the whole effort):** because calibration is cm-accurate, the right move
  is to **use it fully — proper uncertainty-aware multi-view triangulation/fusion — instead of
  median-of-homographies.** Sub-meter is trivially achievable; the realistic target is
  **~centimetres for well-observed joints and sub-0.3 m for ground position** (cf. Pose2Sim /
  multi-view sports literature: 30–50 mm joint-center error with good calibration). No calibration
  refinement work is needed; C7's different resolution is already handled (`load_image_sizes_from_drive`).
- **Residual action:** still emit per-emitted-point reprojection error (ISSUE-12) so any *future*
  per-camera drift is caught; use the ~3 px ball reprojection as the standing calibration anchor.

### ISSUE-7 — foot-contact pixel is coarse and pose-noise-prone  ★★
- **Evidence:** `ground_contact_pixel` (`geometry.py:143`) uses the lower confident ankle only if
  within `max(20 px, 0.25·bbox_h)` of the bbox bottom, else the **bbox bottom-centre**
  (`x + w/2, y + h`). Bottom-centre assumes the ground contact is at the horizontal centre of the box —
  wrong for a bowler's delivery stride, a diving fielder, or a running batter (feet spread / one foot
  planted off-centre), and the box bottom itself is a detector artifact.
- **Impact:** this is the *input* pixel to every projection; under the lens geometry a 10–20 px error
  here is the dominant term in the ~2 m spreads. There is **no temporal smoothing** of the foot pixel,
  so per-frame detector jitter injects per-frame ground jitter (ISSUE-9).
- **Fix direction:** use ankle+heel keypoints when confident; use the planted (lower, slower-moving)
  foot; smooth the foot pixel per tracklet before projecting; widen the ankle-plausibility logic to
  the actual foot rather than bbox centre.

### ISSUE-8 — height-prior fallback uses fixed population anthropometrics  ★
- **Evidence:** `upper_body_ground_estimate` (`geometry.py:248`) projects hips/shoulders/head at fixed
  heights **0.93 / 1.42 / 1.78 m** (`p3_association.yaml:90-92`); used whenever feet are unusable
  (cut-off bboxes, `runner.py:214`).
- **Impact:** ground error ≈ (true − assumed height) / tan(camera elevation). For low-elevation
  broadcast cameras tan is small, so a 10 cm height mismatch (crouching keeper, tall/short player,
  bowler in load-up) is 0.5–1 m of ground error. Applied exactly to the close-to-camera figures that
  matter most.
- **Fix direction:** estimate per-track standing height once (from confident upright frames /
  triangulation) and reuse it; inflate the covariance of these anchors (already has
  `approx_var_floor_m: 0.8`) and make sure that inflation reaches the Kalman R (ISSUE-4).

### ISSUE-9 — no temporal smoothing of the per-camera ground point before fusion  ★
- **Evidence:** the ground XY path has no EMA/smoother; `confidence_ema_smooth`
  (`triangulation.py:198`) exists but is only used by the P6 3D-skeleton export
  (`triangulate_predictions.py`), which isn't even run on disk (no `p6/` dirs).
- **Impact:** per-frame detector/foot jitter passes straight into the Kalman as measurement noise; the
  filter then either over-smooths (laggy) or chases jitter, depending on the fixed R (ISSUE-4).
- **Fix direction:** short causal smoothing of the per-camera foot pixel (or ground point) per tracklet
  before it becomes a measurement; the Kalman then only handles genuine motion.

### ISSUE-10 — flat z = 0 plane assumption  ★
- **Evidence:** all projection is onto the single plane z = 0 (`ground_homography_from_projection`,
  `geometry.py:188`; `pixel_to_plane_xy` supports other heights but is only used for the height-prior
  anchor).
- **Impact:** a jumping bowler/fielder, a batter airborne, or any real outfield undulation/pitch camber
  means the true foot is above/below z = 0, and the ray-plane intersection then lands off-target
  (error grows with grazing angle). Silent — nothing flags it.
- **Fix direction:** accept as a modelling limit but bound it: cross-check the z of the RANSAC foot
  (ISSUE-3) against 0 and inflate covariance when the contact point is likely airborne.

### ISSUE-11 — role-invariant dynamics: every track runs as `unknown`  ★
- **Evidence:** P5 roles exist (`p5/roles.json`) but the deep-dive states role wiring into P4 is a
  dormant hook — "every track currently runs as `unknown`" — so all tracks use α=1.0, σₐ=2.0,
  R=0.4 (`ground_kalman.py:35`). Roles are computed *after* P4 and never fed back.
- **Impact:** an umpire (near-static) and a sprinting fielder get identical process noise; the filter
  is simultaneously too loose for the umpire (jitter) and too stiff for the sprinter (lag/teleport on
  re-acquire). Not the biggest lever but real.
- **Fix direction:** feed P5 role (or an online velocity-based proxy) into `switch_role`
  (`ground_kalman.py:122`) so process noise matches motion.

### ISSUE-12 — no direct ground-accuracy metric; validation is proxies only  ★★ (blocks measuring any fix)
- **Evidence:** the only ground numbers are `ground_spread_m` (internal consistency, not accuracy),
  `cross_camera_agreement_rate` (computed from independent bbox-bottom projections — itself subject to
  ISSUE-7), and teleport counts. There is **no ground-truth position** and no
  survey-point / ball-trajectory accuracy check wired into the pipetrack metrics. p6 triangulation
  (which would give reprojection-in-metres) has not been run (no `p6/` dirs).
- **Impact:** we cannot currently say "positions improved by X metres" — only that proxies moved.
- **Fix direction:** (1) use the tracked-ball 3D trajectory + `compare_ball_reprojection`
  (`calibration.py:157`) as an independent metric anchor; (2) emit per-emitted-point reprojection
  error into all member views as a first-class accuracy proxy; (3) render the top-down minimap for a
  few deliveries and eyeball metric plausibility (players inside the field, keeper behind stumps, etc.).

---

## 3. Root-cause summary — the one-paragraph version

The ground-position machinery has the right *primitives* (anisotropic `ground_covariance`,
inverse-covariance `fuse_ground_estimates`, RANSAC triangulation, a Singer Kalman) but the actual
**emitted coordinate is produced by a chain of unweighted median/mean steps that discard all of that
uncertainty** (ISSUE-2, 3, 5), fed into a Kalman with **fixed, distance-blind measurement noise**
(ISSUE-4), on top of a **coarse foot pixel** (ISSUE-7) and a **calibration that carries ~1 m of
uncorrected bias** (ISSUE-6) — and about **half of all observations have only one camera** to begin
with (ISSUE-1). The covariance-aware code paths built for *identity* (`tracklet_graph`) were never
routed into the *position* output.

## 4. Recommended order of attack (value-for-effort)

1. **ISSUE-6 first (measure calibration):** run `audit_calibration` + attribute the reprojection tail
   per camera. Sets the accuracy ceiling; may be the whole story on _5/_7.
2. **ISSUE-4 + ISSUE-2 (uncertainty end-to-end):** fuse ground points by inverse covariance and feed
   that covariance into the Kalman R. Biggest algorithmic lever; directly kills teleports.
3. **ISSUE-3 (emit the triangulated foot, record reproj on the emitted point).**
4. **ISSUE-7 + ISSUE-9 (better + smoothed foot pixel).**
5. **ISSUE-1 single-camera handling, ISSUE-5 stop re-averaging, ISSUE-8/10/11 modelling refinements.**
6. **ISSUE-12 throughout:** stand up a real accuracy metric so each step above is provable.

## 5. Key file references

- Geometry primitives: `pose_estimation/cricket/geometry.py` (`pixel_to_ground_xy` :204,
  `ground_covariance` :302, `fuse_ground_estimates` :362, `ground_contact_pixel` :143,
  `upper_body_ground_estimate` :248)
- Triangulation: `pose_estimation/triangulation.py`
- P2 ground calibrator: `scripts/tracking/calibration.py` (`GroundPlaneCalibrator` :13,
  `build_ground_calibrators` :109)
- P3 consensus/gating: `scripts/association/associator.py` (`_detection_ground_xy` :177,
  `_ground_consensus_members` :460, cluster loop :417-447)
- P3 covariance edges: `scripts/association/tracklet_graph.py` (:336)
- P4 Kalman: `pose_estimation/cricket/ground_kalman.py`
- P4 runner (ground emit): `scripts/global_id/runner.py` (:112, :183-233, :249-255)
- Config: `configs/p3_association.yaml`, `configs/p4_global_id.yaml`
- Calibration audit: `pose_estimation/cricket/calibration.py` (`audit_calibration` :262,
  `compare_ball_reprojection` :157)
- Metrics source: `p3/association_metrics.json`, `p3/diagnostics/correspondences.jsonl`,
  `p4/global_id_metrics.json`, `p4/diagnostics/ground_tracks.jsonl` per delivery
