# Metrics & quality proxies

There is **no per-player identity or 3D-location ground truth** for the cricket
deliveries, so almost every quality number the pipeline reports is an explicitly-labelled
**proxy** derived from geometry and self-consistency. This page defines them. (If you add
labels, `--ground-truth` unlocks real MOTA/IDF1 in P4.)

## Calibration accuracy (the anchor)

The one hard, externally-checkable number: **ball reprojection error**. Projecting the
surveyed/known ball position into each camera lands within **mean 1.2–1.9 px, p95 ≤ 4.5 px**
on all deliveries. This is why identity and location are solved directly on the calibrated
ground plane — the calibration is centimetre-accurate and can be trusted.

## Identity proxies (P3 / P4)

Reported in `association_metrics.json` and `global_id_metrics.json`:

- **cross-camera agreement rate** — fraction of frames where independent bbox-bottom ground
  projections agree on the clustering. Low agreement ⇒ the same player is split across
  cameras (under-merge). Ranges ~0.50–0.98 across deliveries.
- **distinct global-ID count** — total IDs minted. A cricket scene has ~13–15 people
  (≤7 visible per camera); counts of 18–25 indicate **over-segmentation / fragmentation**.
- **teleport event count** — an ID moving faster than 9 m/s between frames (a mis-assignment
  to a different person). Observed 7–155 per delivery.
- **same-camera identity collisions** — two detections in one camera-frame sharing a global
  ID. This is a **hard invariant**: it is 0 by construction and must stay 0.
- **cluster cycle-consistency rate** — fraction of ≥3-view clusters that triangulate to one
  consistent point (≤12 px). Failures are likely **chimeras** (two people merged).
- **cue d′ (separability)** — how well a cue (appearance colour histogram, ground distance,
  pose-shape) separates same-vs-different players. The colour-appearance cue's d′ ≈ 0 on
  most deliveries (near-identical kit, desaturated footage) — i.e. it carries almost no
  identity information.

## 3D-location proxies (P3 / P6)

From `eval_ground_accuracy.py` and the P6 triangulation:

- **emitted-point reprojection error (px)** — reproject the emitted ground/3D point into the
  member camera views; distance to the actual foot pixels used.
- **distance-to-triangulated-foot (m)** — for ≥3-view clusters, the emitted ground point vs
  the calibration-optimal triangulated foot. The current emitter (`z0_reproj`) achieves
  ~0.147 m mean (down from ~0.211 m).
- **per-joint triangulation reprojection** — full-skeleton 3D lift is 2–4 px per joint
  (p95 5–7 px) where ≥2 cameras see the player.
- **ground-spread** — max pairwise disagreement (m) between cameras' ground projections for
  a cluster; a purity/coverage signal.
- **single-camera rate** — fraction of player-frames seen by only one camera (~0.39–0.61).
  These get no triangulation and a weakly-corrected position.

## Temporal quality

- **temporal jitter** — mean frame-to-frame joint displacement (the "how noisy" number),
  measured on 2D keypoints and on 3D/ground trajectories. `pose_estimation/metrics.py`
  provides `temporal_jitter`, `reprojection_error`, `mpjpe`/`p_mpjpe`.
- **emitted-trajectory jump p95** — worst per-frame position jump after the Kalman-posterior
  emit (halved vs the raw re-mean; worst case cut from 14.0 m to 0.36 m on the hardest clip).

## How to use them

No single proxy is optimised in isolation — the batch driver
([`run_id_pipeline.py`](scripts.md#the-batch-identity-driver)) prints them **jointly** across
all deliveries and diffs against a frozen baseline. A change is a "win" only if it improves
the panel broadly without regressing the hard invariants (same-camera collisions = 0) or
another delivery. See [improving-models.md](improving-models.md).
