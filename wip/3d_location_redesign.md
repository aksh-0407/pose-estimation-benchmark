# 3D Ground-Location Redesign — research-backed, robust, sub-meter

Companion to `wip/3d_location_issues.md`. This maps recent/SOTA multi-view work to each of the 12
issues and specifies the replacement algorithm. It is grounded in the fact — newly established by the
Phase-0 calibration audit — that **our 7-camera calibration is cm-accurate** (ball reprojection
mean 1.2–1.9 px, p95 ≤ 4.5 px on all 8 deliveries). The problem is therefore **not** geometry we lack;
it is that the pipeline throws the geometry away with median/mean fusion and a distance-blind Kalman.

## 1. What the literature says (and how it maps to us)

| Source | Method | What we take |
|---|---|---|
| Iskakov et al., *Learnable Triangulation of Human Pose*, ICCV 2019 ([arXiv 1905.05754](https://arxiv.org/abs/1905.05754)) | Differentiable **algebraic (confidence-weighted DLT)** + volumetric triangulation | Confidence-weighted DLT is the right per-joint estimator; weight each view by keypoint confidence (we already do √conf — keep, extend to covariance). |
| Lee & Civera, *Robust Uncertainty-Aware Multiview Triangulation*, 2020 ([arXiv 2008.01258](https://arxiv.org/abs/2008.01258)) | RANSAC/**IRLS + robust (Huber) reweighting**; refine minimizing reprojection/Mahalanobis; midpoint biased at low parallax | Make fusion **robust** (down-weight the one bad view), not a plain mean/median. Prefer reprojection/covariance cost over midpoint on our low-parallax facing pairs. |
| LOSTU, 2023 ([arXiv 2311.11171](https://arxiv.org/html/2311.11171v2)); UPose3D, 2024 ([arXiv 2404.14634](https://arxiv.org/pdf/2404.14634)) | **Uncertainty-aware** triangulation; propagate 2D keypoint covariance; temporal cues | Attach a real **2×2 covariance** to every ground estimate and carry it downstream (into the Kalman R and fusion weights). |
| **Pose2Sim** Part 1 Robustness / Part 2 Accuracy ([PMC8512754](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8512754/), [PMC9002957](https://pmc.ncbi.nlm.nih.gov/articles/PMC9002957/)) | Calibrated multi-cam markerless **sports** kinematics: weighted DLT → RANSAC → **4th-order low-pass Butterworth** → IK. Tolerant to 1 cm calib error. Prior work 30–50 mm joint error. | This is our exact problem class. Adopt: weighted+robust DLT, then **offline zero-phase low-pass temporal filter** of 3D/ground trajectories. Sets the accuracy target (cm-level). |
| VoxelPose / Faster VoxelPose, ECCV 2020/2022 ([arXiv 2207.10955](https://arxiv.org/pdf/2207.10955)) | Aggregate features in 3D voxel space; "operate in 3D to avoid committing to noisy 2D" | Philosophy only: decide position in 3D/metric space with all views jointly. Full learned volumetric net is unnecessary given our calibration — too heavy, needs training. |
| Multi-camera sports/soccer tracking (e.g. [Graph-Based Multi-Camera Soccer Tracker](https://arxiv.org/pdf/2211.02125); homography ground-plane fusion surveys) | Fuse per-camera foot points on the ground plane weighted by **distance-dependent uncertainty**; particle/Kalman on the ground plane | Direct support for: per-camera ground covariance that grows with distance/grazing, inverse-covariance fusion, and ground-plane temporal filtering. |
| [Analytical Correction of Distance Error in Homography Ground-Plane Mapping, 2026](https://arxiv.org/pdf/2604.10805) | Models the grazing-angle/distance error of homography foot mapping | Confirms the homography Jacobian blow-up is the dominant single-camera error and must be modelled as covariance (our `ground_covariance`), not ignored. |
| Spatiotemporal Bundle Adjustment ([arXiv 2007.12806](https://arxiv.org/pdf/2007.12806)) | Joint temporal+spatial reprojection-min refinement | Optional stretch: an offline reprojection-min refinement of the ground trajectory. Not needed for v1. |

**Net recipe the field converges on for calibrated multi-view sports:**
`per-view estimate + covariance → robust (IRLS/RANSAC) inverse-covariance fusion → feed covariance to a
temporal filter (Kalman online + zero-phase low-pass / RTS offline)`. Our repo already has every
primitive; it just doesn't use them on the output path.

### 1a. EMPIRICAL PIVOT — measured on our data, 2026-07-08 (this overrides the recipe above)

The literature's "inverse-covariance fusion of per-camera ground points" was implemented (`robust_cov`)
and **A/B-tested against the calibration-optimal triangulated foot on delivery 1 (1836 reference
clusters)**. Result:

| emitted-position estimator | dist to triangulated foot (mean / p95, m) |
|---|---|
| historical median | 0.176 / 0.376 |
| `robust_cov` (inverse-cov IRLS) | **0.248 / 0.516** ← worse |
| `z0_reproj` (z=0 robust reprojection min) | **0.145 / 0.302** ← best |

**Why covariance fusion lost:** it minimises *metric position variance* under the homography Jacobian,
which pulls the estimate toward near cameras — but with cm-accurate calibration the *reprojection-optimal*
point is what matters, and pixel-reprojection over-weights far cameras (long lever). The two criteria
diverge exactly on our grazing/facing geometry. **Because our calibration is excellent, use the
calibration directly (reprojection minimisation), not a covariance model of homography noise.** This is
the key lesson: validate the paper recipe on your rig before adopting it.

**Chosen algorithm — `z0_reproj`** (`ground_from_reprojection`, `geometry.py`): solve
`argmin_{x,y} Σ_c w_c · ρ_Huber(‖project_c([x,y,0]) − foot_c‖)` by Gauss–Newton, then a hard inlier
refit (RANSAC-lite) to fully reject a gross outlier foot (a hallucinated ankle 50–200 px off). Depth is
removed by the ground constraint so it is well-posed on the low-parallax facing pairs where free
triangulation is degenerate. Verified: reprojection of the emitted point vs the actual foot pixels used
drops 84→74 px (p50 37→32); clustering/identity byte-unchanged (the merge gate is untouched).
`ground_covariance`/`robust_fuse_ground` are retained (unit-tested) for the P4 Kalman-R work, where a
per-observation covariance IS the right object — but the emitted *position* now comes from `z0_reproj`.

## 2. The replacement algorithm (per player, per frame → per track)

```
foot pixel (robust: planted foot, per-tracklet smoothed)          ← ISSUE-7,9
    │  per camera
    ├─ pixel_to_ground_xy(P)                → ground point (x,y)
    └─ ground_covariance(P, σ_px(conf,bbox))→ anisotropic 2×2 Σ    ← ISSUE-2 (exists, unused)
    │
ROBUST INVERSE-COVARIANCE FUSION over a cluster's cameras:         ← ISSUE-2,3,1
    IRLS: start = inverse-cov mean; iterate:
      w_i = Huber(mahalanobis(point_i, current, Σ_i))   # down-weight the bad view
      re-fuse with w_i · Σ_i⁻¹
    → fused (x,y) + fused Σ_f ; also record max reprojection into member views  ← ISSUE-12
    where parallax ≥ θ across ≥2 views: also RANSAC-triangulate the foot in 3D;
      use its (x,y) and its z (airborne flag inflates Σ)                          ← ISSUE-3,10
    single-camera: keep point but with its large anisotropic Σ (no false precision) ← ISSUE-1
    │
P4a SingerGroundKalman.update(z=fused_xy, R=Σ_f)                   ← ISSUE-4 (was fixed R)
    role from P5 (or online speed proxy) → dynamics                ← ISSUE-11
    emit the Kalman posterior, not a re-mean of raw obs            ← ISSUE-5
    │
OFFLINE (whole delivery is offline): RTS smoother / 4th-order zero-phase
    low-pass (Butterworth, ~6–8 Hz @ 50 fps) over each track's ground trajectory  ← ISSUE-9 (Pose2Sim)
    → ground_tracks.jsonl (+ per-point Σ and reprojection error)   ← ISSUE-12
```

## 3. Issue-by-issue resolution (better fixes than the original file)

- **ISSUE-1 (single-camera ~50%):** don't emit a bare point — emit point **+ anisotropic covariance**
  from `ground_covariance`; the Kalman then trusts it proportionally and **coasts on prediction** when
  the lone ray is far/grazing. Where a 2nd view exists even at low parallax, ground-plane fusion (not
  full triangulation) still tightens it.
- **ISSUE-2 (unweighted median):** replace `_ground_consensus_members` median with **robust IRLS
  inverse-covariance fusion** (`fuse_ground_estimates` + Huber reweighting). *(Lee & Civera; sports fusion.)*
- **ISSUE-3 (triangulated foot discarded):** where parallax suffices, **emit the RANSAC-triangulated
  foot** (projected to z=0) as position; record reprojection on the emitted point for **all** clusters
  incl. 2-view. *(Iskakov; Pose2Sim.)*
- **ISSUE-4 (fixed distance-blind R):** feed fused Σ_f into `SingerGroundKalman.update` as R. This is
  the biggest anti-teleport lever; standard in multi-camera sports tracking (distance-dependent R).
- **ISSUE-5 (re-mean twice):** emit the **Kalman/smoother posterior**, delete the raw `np.mean` re-average.
- **ISSUE-6:** ✅ refuted — calibration is cm-accurate; use it, don't refine it.
- **ISSUE-7 (coarse foot):** planted (lower, confident) ankle/heel; per-tracklet temporal smoothing of the
  foot pixel before projecting; σ_px scaled by keypoint confidence and bbox height.
- **ISSUE-8 (fixed height prior):** estimate **per-track standing height** from confident triangulated
  upright frames and reuse it; inflate Σ for height-anchored estimates. *(anthropometrics only as prior.)*
- **ISSUE-9 (no temporal smoothing):** offline zero-phase low-pass Butterworth / RTS smoother on the
  ground trajectory (Pose2Sim's 4th-order 6 Hz, scaled to 50 fps). Also smooth the foot pixel.
- **ISSUE-10 (flat z=0):** triangulated foot z ≠ 0 ⇒ airborne ⇒ inflate Σ / hold; flag it.
- **ISSUE-11 (role-invariant dynamics):** feed P5 role (or velocity-based online proxy) into `switch_role`.
- **ISSUE-12 (no accuracy metric):** emit per-point **reprojection error into member views** + fused Σ
  trace as first-class accuracy proxies; anchor on the ~3 px ball reprojection; report ground-spread,
  reprojection, teleport distributions before/after per delivery.

## 4. Implementation order (behind config flags; flags off ⇒ byte-identical baseline)

1. **Geometry core (unit-testable, no pipeline run):** robust IRLS inverse-covariance ground fusion in
   `pose_estimation/cricket/geometry.py` (+ tests). *[highest leverage, lowest blast radius]*
2. **Wire into P3 consensus** (`associator._ground_consensus_members` → robust fusion; attach Σ +
   reprojection to the emitted correspondence).
3. **Wire Σ into P4a Kalman R** and emit posterior (`ground_kalman`, `global_id/runner`).
4. **Offline ground-trajectory smoothing** (RTS/Butterworth) for `ground_tracks.jsonl`.
5. **Foot-pixel + height-prior + role + airborne** refinements.
6. **Metrics:** reprojection-on-emitted-point + Σ-trace into `association_metrics`/`global_id_metrics`.

## 4b. Real-time feasibility (pt5)

Can this run online / real-time (QT-style)? **Yes for the core**, with specific changes. Stage-by-stage:

| Stage | Causal? | Cost / note |
|---|---|---|
| P1 RTMPose top-down (RTMDet + RTMPose-L) | ✅ real-time on GPU | the throughput bottleneck; scales with #players (top-down). ~7–8 fps mosaic-render-bound here, pose inference is faster on GPU |
| P2 per-camera tracking | ✅ online | BYTE-style, causal |
| P3 z0_reproj ground solve | ✅ online | 2-DOF Gauss-Newton per player/frame = microseconds |
| P3 association *mode* | ⚠️ | `per_frame`/sliding-window is online; the current `tracklet_graph` default is **offline** (whole delivery) |
| P4a Singer Kalman + **posterior emit** | ✅ online | causal filter; posterior is exactly what an online consumer wants |
| P4b stitching | ❌ non-causal | short offline pass |

**To go real-time:** (1) association `per_frame` or a short sliding-window graph instead of whole-delivery;
(2) drop P4b (or window it); (3) the posterior emit, z0 solve, foot fixes are all already causal. The
**GPU is used for encoding (NVENC)** and pose inference; the geometry/Kalman math is negligible cost.
Trade-off: offline maximises accuracy/stability; online gives ~equal *position* accuracy with slightly
more identity flicker. QT itself is real-time, so the target is proven achievable.

## 5. Success criteria (verify against baseline in `wip/3d_location_issues.md` §0)

| Metric (per delivery) | Baseline | Target |
|---|---|---|
| multi-cam ground-spread p50 / p95 (m) | ~2.0 / 3.4–5.1 | ≤ 0.5 / ≤ 1.0 |
| % multi-cam reproj > 12 px | 34–61% | < 15% |
| teleport events | 7–171 | ≤ a few |
| single-cam positions with covariance | 0% (bare points) | 100% |
| emitted-point reprojection error recorded | none | yes (accuracy proxy) |
| cross-camera agreement | 0.50–0.98 | ≥ 0.90, no regression |
| same-camera collisions | 0 | 0 (hard invariant) |
