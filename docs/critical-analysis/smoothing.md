# How smoothing works in the pipeline

Quick reference (2026-07-14). We smooth at **three** stages, each on a different signal. All
are flag-gated; defaults are the v8.1 production values.

| Stage | What it smooths | Method | Where |
|---|---|---|---|
| **P1b** | 2D keypoints (pixels), per camera | One-Euro filter + spike clamp | `src/identity/p1_stabilization/linker.py`, `configs/01_stabilization.yaml` |
| **P4a** | Ground position (x,y world), per player | Singer-acceleration Kalman + χ²-gated posterior | `src/identity/p5_global_id/ground_kalman.py`, `configs/05_global_id.yaml` |
| **P6** | 3D skeleton (X,Y,Z world), per joint | Zero-phase Butterworth (+ occlusion fill) | `src/identity/common/triangulation.py::butterworth_smooth`, `src/identity/p4_lift/run_triangulation.py` |

---

## 1. P1b — 2D keypoint stabilization (One-Euro)
Runs between P1 (inference) and P2 (tracking). It links each camera's detections into short
**IoU micro-tracks** (purely for smoothing, not identity: `iou_min 0.3`, bridge
`max_gap_frames 2`) and runs a **One-Euro filter** on every keypoint's pixel trajectory.

- **One-Euro** is a speed-adaptive low-pass: it smooths hard when the joint is slow (kills
  jitter) and loosens when the joint moves fast (avoids lag on a real swing). Params:
  `min_cutoff 1.7` (lower = smoother at rest), `beta 0.30` (higher = less lag on fast
  motion), `d_cutoff 1.0`.
- **Confidence-gated spike clamp**: on a *low-confidence* keypoint (`confidence_min 0.30`), a
  single-frame jump beyond `120 px` **or** `0.5 ×` the bbox diagonal is treated as a detector
  spike and replaced by the filter's **predicted** position instead of the raw value.
- Applies to both COCO-17 and the full Halpe-26 block (`smooth_native: true`).
- Effect: −20–34 % keypoint jitter (panel `jitter_px` 1.2–3.4). `enabled: false` is a
  byte-identical pass-through for A/B.

## 2. P4a — ground-position smoothing (Singer Kalman)
Every player's fused ground point is filtered by a **Singer-acceleration Kalman filter**
(constant-position + velocity + correlated-acceleration state). It is **role-tuned** — each
role has its own manoeuvrability and noise (`ground_kalman.py:29-35`):

| role | α (manoeuvre) | σ_a accel (m/s²) | R meas. noise |
|---|---|---|---|
| bowler | 2.0 | 3.0 | 0.30 |
| striker | 1.5 | 2.5 | 0.30 |
| non-striker | 0.5 | 1.0 | 0.30 |
| wicketkeeper | 0.3 | 0.5 | 0.20 |
| umpire | 0.2 | 0.3 | 0.20 |
| fielder / unknown | 1.0 | 2.0 | 0.40 |

- A **χ² gate (5.991, 2-dof)** rejects measurements too far from the prediction before the
  update, so a single wild foot projection cannot yank the track.
- With `emit_kalman_posterior: true` (production), `ground_tracks.jsonl` carries the
  **filtered posterior**, not the raw per-frame solve — this is the temporal smoothing of the
  delivered ground track.
- On re-acquisition the process noise resets so the filter re-locks quickly.

## 3. P6 — 3D skeleton smoothing (zero-phase Butterworth + fill)
The terminal 3D lift triangulates each joint (RANSAC-DLT, cheirality-gated), then:

- **Occlusion fill** first (`triangulation.py:701`, `max_gap_frames 25`): short gaps between
  real triangulated frames are filled — linear for a joint, or **bone-vector offset** from a
  valid parent joint — but only *within* 25 frames of real data; sequence ends are held, and
  longer gaps stay NaN (never fabricated). `--dense-fill` is default ON.
- **Zero-phase Butterworth** (`butterworth_smooth`, default `cutoff_hz 6.0`, `order 4`,
  applied per X/Y/Z with `filtfilt` = forward+backward so there is **no phase lag**). Removes
  content above 6 Hz — the sports-capture standard (cf. Pose2Sim). NaN gaps are preserved,
  never bridged by the filter. The driver default `--tri-smoother butterworth`.

---

## What smoothing can and cannot fix (ties to the diagnosis)
- Smoothing removes **jitter** (small high-frequency noise) very well — that's why P6
  skeletons are clean (pelvis p95 1.6–3.8 m/s) and 2D jitter is low.
- Smoothing **cannot fix a teleport whose cause is a wrong/multi-modal input.** The emitted
  ground teleports (`docs/diagnosis/04-...`) survive because P4 emits the **mean of two
  far-apart fragment positions** (`runner.py:348`) — averaging a bimodal signal is not
  smoothing, and the Kalman only smooths *within* a track, not across a wrong merge. The fix
  is at the emission/association level (`docs/changes_tbd.md` C2/C3), not a stronger filter.
- P6 Butterworth is applied **only to triangulated (multi-view) joints**, so single-camera
  frames get no smoothing *and* no 3D — hence smooth-but-sparse coverage (`docs/diagnosis/08`).
