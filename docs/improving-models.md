# Improving the pipeline

Getting a good model is the starting point, not the goal. Off-the-shelf 2D keypoints
jitter, drop out under occlusion, and get swapped in crowds; multi-view fusion and identity
are where the production value is. This page is the map of *where the quality lives now* and
how to work on it. The definitive, evidence-backed analysis is in the
[pipeline reference](pipeline/README.md); this is the practical loop.

## The improvement loop

```
baseline run  ->  apply one change (behind a flag)  ->  re-run affected stage(s)  ->  diff the panel
                  (detector / smooth / cue / gate)      (same deliveries)            (broad win? no regressions?)
```

Every intervention should be **proven, not assumed**: run the baseline, apply the change,
re-run, and compare the committed proxy metrics
(`python -m identity.id_pipeline` prints them jointly). A change
is a win only if it helps broadly across the 8 deliveries without regressing a hard
invariant or another clip. See [reference/metrics.md](reference/metrics.md).

## Where the quality lives (measured)

The calibration is centimetre-accurate and **3D location is largely solved** (the emitted
ground error is down ~36% and jitter is halved). **Identity is now the dominant ceiling** —
mosaics place players correctly but their IDs swap and fragment. In rough priority:

1. **Cross-camera under-merge** on the low-parallax facing pairs (`cam_01↔04`, `02↔06`,
   `03↔05`) — the epipolar geometry is weak and the colour cue is dead, so two views of one
   player become two IDs.
2. **Fragmentation / over-segmentation** — players lost through occlusion are re-born as new
   IDs; stitching under-merges. 18–25 IDs vs a ~13 roster.
3. **The colour-appearance cue is effectively dead** (d′ ≈ 0) — both teams wear near-identical
   kit and the footage is desaturated. Body proportions (pose-shape) and a learned,
   kit-robust re-ID embedding are the substitutes to invest in.
4. **Teleports** — an ID jumps to a different nearby person in crowds/occlusion.

The open issues, their evidence, and prioritised fixes are enumerated per phase in the
[pipeline reference](pipeline/README.md) and in the measured [diagnosis/](diagnosis/README.md) (
`wip/3d_location_issues_v2.md`.

## The levers the code already gives you

- **2D denoising** — the P1 output feeds everything, so cleaner 2D helps every stage.
  Confidence gating, outlier rejection, and temporal smoothing of keypoints (One-Euro /
  Savitzky-Golay) are the front-line jitter levers.
- **Multi-view geometric denoising** — `src/identity/common/triangulation.py`
  (`ransac_triangulate_point`, `triangulate_skeleton_ransac`) rejects a bad view using the
  geometry of the others; `confidence_ema_smooth` smooths the 3D trajectory.
- **Ground-plane solving** — `src/identity/common/geometry.py` (`ground_from_reprojection`
  = the `z0_reproj` emitter, `ground_covariance`, `robust_fuse_ground`) turns multi-view foot
  pixels into a position with uncertainty on the low-parallax facing geometry.
- **Cue fusion** — the P3 tracklet graph fuses ground/epipolar/appearance/pose-shape/motion
  as log-likelihood ratios; the 05 global-id Singer-KF + min-cost-flow stitcher maintain identity.
  Most improvements are new/stronger cues or better gates behind a `configs/*` flag.

## Practical starting points

- Reduce 2D jitter at the source (a stabilization pass on the P1 stream) and re-measure
  `temporal_jitter` before/after.
- Promote **pose-shape / a learned re-ID descriptor** from a soft tie-breaker to a primary
  cross-camera cue where colour is dead (fixes under-merge on facing pairs).
- Give the clustering a way to **split** a chimera (correlation clustering / a reprojection
  -gated refine pass), since single-linkage can merge but never un-merge.

Each of these is spelled out with SOTA references and expected effect in the per-phase docs
under [pipeline/](pipeline/README.md).
