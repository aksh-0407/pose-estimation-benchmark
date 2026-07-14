# Triangulation / 3D lift — and its optimal placement

> **Stage 04** — the 3D lift, now run **before** global-id (Associate→Triangulate→Track). Code: `src/identity/p4_lift/run_triangulation.py`.

## Role & intuition

This stage turns the multi-view 2D keypoints of one identified player into a single **3D world
skeleton**. Today it runs **last (P6)**, purely to emit poses for Unreal Engine; it does not feed
identity or tracking. This doc analyses the method and argues — professionally — that its optimal
home is **04 (binding lift): immediately after association, before global ID**, so it can (a) let P4 track in 3D
and (b) supply the chimera-split signal P3's clustering lacks.

## I/O & config

| | |
|---|---|
| **Input (today)** | P4 run; calibration. **Input (proposed)** | P3 run (correspondences) |
| **Output** | `pose_3d.keypoints_world_m` written back into each camera stream |
| **Core** | `src/identity/p4_lift/run_triangulation.py`; `src/identity/common/triangulation.py` |
| **Knobs** | `--reprojection-threshold-px 10`, `--min-views 2`, `--ema-alpha 0.65` |

## Flowchart

```mermaid
flowchart TD
  C["correspondences (same player, >=2 views)"] --> T["per-joint RANSAC DLT<br/>triangulate_skeleton_ransac:162"]
  T --> R["reprojection inlier refit<br/>threshold 10px"]
  R --> F["temporal occlusion fill<br/>fill_occluded_joints:208 (gap<=25)"]
  F --> S["skeletal-prior fill<br/>fill_from_skeletal_prior:259"]
  S --> E["confidence-aware EMA<br/>confidence_ema_smooth:296 (alpha 0.65)"]
  E --> OUT["pose_3d.keypoints_world_m"]
  R -. per-cluster reprojection / cycle-consistency .-> SPLIT["chimera split signal (proposed -> P3/P4)"]
```

## Methods walkthrough

**Weighted DLT — `triangulate_point_dlt` ([triangulation.py:31](../../src/identity/common/triangulation.py#L31)).**
The classic linear triangulation: for each view stack the two rows `x·P₃−P₁`, `y·P₃−P₂`, weight
each by `√conf`, and solve `A X = 0` by SVD (the 3D point is the smallest right singular vector,
dehomogenised). Confidence weighting is the differentiable-triangulation idea from **Iskakov et al.,
Learnable Triangulation, ICCV 2019** ([arXiv 1905.05754](https://arxiv.org/abs/1905.05754)).

**RANSAC over views — `ransac_triangulate_point:90` / `triangulate_skeleton_ransac:162`.**
Triangulate every camera pair, count inliers by reprojection error ≤ `reprojection_threshold_px`,
keep the best inlier set, and **re-fit** the DLT on inliers. This robustly rejects a single bad
view — the practical robustification recommended by **Lee & Civera 2020**
([arXiv 2008.01258](https://arxiv.org/abs/2008.01258)) and used in markerless sports capture
(**Pose2Sim**, [PMC8512754](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8512754/)).

**Occlusion / prior fill + smoothing.** `fill_occluded_joints:208` linearly interpolates NaN
joints within a 25-frame gap; `fill_from_skeletal_prior:259` places a never-seen joint from its
parent + a bone vector scaled to the identity's median bone length; `confidence_ema_smooth:296`
applies confidence-weighted temporal EMA (α=0.65). Together these take multi-camera completeness to
100% of ≥2-view frames (per-joint reprojection 2–4 px; `../diagnosis/README.md` R4).

## Pros

- **Right estimator for a calibrated rig** — confidence-weighted DLT + reprojection-RANSAC is the
  field-standard for cm-accurate multi-view capture; it is cheap and needs no training.
- **Robust to one bad view** — the inlier refit rejects a hallucinated joint instead of averaging
  it in.
- **Complete skeletons** — occlusion/prior fill + EMA yield a full, temporally smooth 3D pose on
  every multi-view frame.
- **The reprojection residual is a free purity signal** — a chimera (two people merged) fails
  torso/limb reprojection hard; this is a clean, unused split signal.

## Cons

- **Needs ≥2 views** — the ~39% single-camera frames get *no* 3D pose at all (V2-L1). This is the
  biggest coverage gap.
- **Flat z=0 for the ground point** — an airborne foot (running stride, bowler load, jump) with
  z≫0 is mislocated when forced to the plane (V2-L3; ankle z p95 = 0.56 m).
- **Skeletal-prior fill can fabricate plausible-but-wrong joints** — a never-seen limb placed from
  a prior is a guess, low-confidence but still emitted.
- **Runs last (terminal P6)** — the full 3D skeleton, the single richest geometric signal, is
  computed *after* identity is already decided, so it cannot help tracking or split chimeras. This
  is a **sequencing** weakness, not a method weakness.

## Issues

- **V2-L1 (★★) Single-camera → no 3D pose.** ~39% of player-frames (`../diagnosis/README.md`
  V2-L1). No triangulation possible with one ray.
- **V2-L3 (★) Flat z=0 airborne error.** Triangulated ankle z p95 0.56 m; the ground point forced
  to z=0 lands beyond the true position at grazing angle.
- **T-1 (★★) Triangulation runs terminal, so its reprojection/cycle-consistency never feeds
  identity.** The exact signal that would split ID-5 chimeras and enable 3D tracking is discarded.
- **T-2 (★) Skeletal-prior fabrication risk** for never-seen joints on long single-view stretches.

## Fixes (all, priority-ordered)

| # | Fix | Priority | Reasoning | Expected effect | Effort | Source |
|---|---|---|---|---|---|---|
| 1 | **Move triangulation to 04 (binding lift)** (after association, before P4) and feed its per-cluster reprojection / cycle-consistency back as a **chimera-split signal** and as P4's 3D observation. | ★★★ | Correspondences exist by P3; running the lift here makes the richest geometric signal available *before* identity is finalised and *before* tracking — the single change that unlocks both 3D-aware tracking and splittable clustering (ID-5). | Fewer chimeras; 3D-aware P4; no extra model. | Medium (re-sequencing + wiring) | VoxelPose "decide in 3D" [Faster VoxelPose 2207.10955]; Iskakov [1905.05754] |
| 2 | **Single-view → canonical-skeleton lift (PnP)** for the ~39% single-camera frames: fit the identity's canonical 3D skeleton (learned from its multi-view frames) to the lone 2D view at its z0 ground position. | ★★ | Half of coverage is single-camera; a PnP/optimisation lift gives a plausible full 3D pose where triangulation can't. | 3D pose on single-camera frames → far higher completeness. | Medium-High | monocular lift / SMPLify-style fitting; UPose3D [2404.14634] |
| 3 | **Uncertainty-aware triangulation** — propagate 2D keypoint covariance into the DLT weights and emit a per-joint 3D covariance to carry downstream (into P4's Kalman R). | ★★ | Weighting by real uncertainty (not just √conf) is the modern robust-triangulation recipe and gives P4 a principled measurement noise. | Better fusion + anti-teleport R. | Medium | LOSTU [2311.11171]; UPose3D [2404.14634]; Lee & Civera [2008.01258] |
| 4 | **Airborne handling** — take the ground position from the triangulated **pelvis vertical projection** (robust to a raised foot), flag airborne frames (ankle z≫0) and inflate their covariance. | ★ | Removes the z=0 grazing-angle error on jumps/strides. | Correct location for airborne feet. | Low-Medium | Pose2Sim [PMC9002957] |
| 5 | **Offline zero-phase temporal filter (4th-order Butterworth / RTS)** on the whole-delivery 3D trajectory for the non-real-time render path. | ★ | The delivery is offline; a zero-phase low-pass is the sports-capture standard for the smoothest final trajectory. | Smoother final 3D with no lag. | Low | Pose2Sim [PMC8512754] |
| 6 | **Gate skeletal-prior fill** — cap how long a joint may be prior-filled and down-weight/flag it, or prefer the single-view PnP lift (fix 2) over pure priors. | ★ | Avoids emitting fabricated limbs on long single-view stretches. | Fewer wrong emitted joints. | Low | — |

Cross-phase: fix 1 here is the enabler for P3 fix 2 (splittable clustering) and P4 3D tracking —
see [fixes-roadmap.md](../changes_tbd.md).
