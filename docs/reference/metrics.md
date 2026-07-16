# Metrics & quality proxies

There is **no per-player identity or 3D-location ground truth** for the cricket
deliveries, so almost every quality number the pipeline reports is an explicitly-labelled
**proxy** derived from geometry and self-consistency. This page defines them. (If you add
labels, `--ground-truth` unlocks real MOTA/IDF1 in global identity (05).)

## Calibration accuracy (the anchor)

The one hard, externally-checkable number: **ball reprojection error**. Projecting the
surveyed/known ball position into each camera lands within **mean 1.2–1.9 px, p95 ≤ 4.5 px**
on all deliveries. This is why identity and location are solved directly on the calibrated
ground plane — the calibration is centimetre-accurate and can be trusted.

## Identity proxies (03 association / 05 global identity)

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

## 3D-location proxies (03 association / 04 lift)

From `tools/diagnosis/eval_ground_accuracy.py` and the 04 lift (triangulation):

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
  measured on 2D keypoints and on 3D/ground trajectories via the diagnostic tools under
  `tools/diagnosis/` (e.g. `emit_smoothness.py`, `occupancy_and_3d.py`). Reprojection error is
  computed geometrically in `src/identity/common/triangulation.py`
  (`reprojection_errors_for_point`) and summarised per cluster in `metrics.py`. There is no MPJPE /
  P-MPJPE (those require 3D ground truth, which does not exist for this footage).
- **emitted-trajectory jump p95** — worst per-frame position jump after the Kalman-posterior
  emit (halved vs the raw re-mean; worst case cut from 14.0 m to 0.36 m on the hardest clip).

## Diagnostic tools (2026-07-16, read-only, no GT)

- **`tools/diagnosis/coverage_audit.py`** — true per-location camera coverage (single-link cluster the
  raw pre-association detections by ground proximity, count distinct cameras) vs the P3 achieved cluster
  size, to separate *detection-recall* gaps from *association under-merge*, plus same-camera collisions
  and the unbound camera-pair tally (tests the facing-pair hypothesis). This tool root-caused the ID-1
  facing-pair under-merge and the `graph_llr_positive_cap` fix.
- **`tools/diagnosis/camera_robustness.py`** — per-camera reprojection contribution (which camera
  disagrees most with the multi-view 3D consensus), leave-one-camera-out hip-shift (robustness to losing
  any one camera), and monocular(2-view)-vs-multiview dispersion. On this rig: no pathological camera
  (cam_06 ~8.5 px best, cam_03/cam_07 ~12–13 px worst), triangulation robust to any single camera
  (5–7 cm), multi-view ~8 cm tighter than 2-view.

## How to use them

No single proxy is optimised in isolation — the batch driver
(`python -m identity.id_pipeline`) prints them **jointly** across
all deliveries and diffs against a frozen baseline. A change is a "win" only if it improves
the panel broadly without regressing the hard invariants (same-camera collisions = 0) or
another delivery. See [improving-models.md](../improving-models.md).
