# changes_tbd.md — deferred work for global-ID stability

> **2026-07-11 status.** Re-triaged item-by-item during the pipetrack_v7 fix campaign
> (`docs/critical-analysis/fixes-log.md` is the full audit trail; `to-do.md` the plan).
> Most items below are now DONE or SUPERSEDED; the remainder are annotated with why they
> stay open. Current best composed run: `pipetrack_v7-rc2` (stitcher live for the first
> time — M2 IDs 16→11, id-persistence 0.699→0.956; `_7` agreement 0.603→0.703, IDs 18→13);
> the P1.5 isolation run (`v7-rc3`) decides the final default stack.

## Disposition of the original items

### Higher value

- **Weight/gate tuning** — PARTIALLY SUPERSEDED. The facing-pair gate concern was fixed by
  `graph_facing_gate_scale` (×1.3, v5) + the corroboration merge; the `ground_weight` /
  `epipolar_weight` / `appearance_weight` sweep only affects the `per_frame` fallback mode
  (default is `tracklet_graph`) — OPEN, low priority, sweep only if the fallback is ever
  load-bearing again.
- **P3 temporal cluster memory** — PARTIALLY DONE. `TemporalLinkMemory` now supports
  exponential decay (`temporal_link_decay`, H4 fix; default legacy 1.0). The
  confirmed-track spatial prior idea remains OPEN (per-frame mode only; low value while
  tracklet_graph is the default).
- **Continuous adaptive lost-window (role-free)** — DONE (two halves): hits-scaled window
  (`adaptive_lost_window`, v5) + **density-at-loss scaling** (`density_lost_window`,
  `density_radius_m`, `density_bonus_frames`; landed 2026-07-11 with unit test — a track
  lost inside a pack earns a longer window). Note the bowler special-case is no longer
  dead code: the F5 online role proxy assigns roles during tracking.
- **Pose tie-breaker at P4b stitching** — DONE and EXTENDED: v5 added the descriptor gate +
  `w_pose` + Kalman-smoothed exit/entry velocities; F12 added the billboard-posture key
  (works on facing pairs); and the G7 fix made the stitcher actually able to select links
  beyond 0.6 s gaps (it was mathematically dead — the root cause of
  `stitched_id_switch_proxy = 0`).

### Medium value

- **Cross-camera pose cluster gate/veto at P3 merge time** — SUPERSEDED by a safer design:
  F13 purity-driven *post-merge eviction* (torso-reprojection chimera signature names the
  intruding camera; conservative thresholds; unit-tested surgical split). This delivers the
  "can split" property without the merge-time false-reject risk this item warned about.
- **Online role wiring** — DONE differently (F5): run-up detection + end classification
  inside P4 (`online_role_proxy`), activating the role-aware Singer params live. The DRS
  `events.json` route was evaluated 2026-07-11 and DEFERRED: the event artifacts are 2D
  per-camera normalized ball points (airborne), so "nearest player at release" needs a 3D
  ball solve — poor ROI while the run-up detector works.
- **Distance-scaled measurement noise** — DONE and exceeded (F10): full 2×2 ground
  covariance from the z0 solver (F9a) as per-measurement Kalman R, eigen-clamped, applied
  asymmetrically (conservative admission gates, uncertainty-weighted updates) after the
  symmetric variant measurably loosened gates.
- **Clustering-algorithm refinement (split capability)** — DONE (F13 + the existing
  `_refine` move/split machinery).

### Lower value / robustness

- **Cheirality check** — DONE (F3), converging on exactly this item's prescription: the
  det(M) formula fails on this rig's handedness; the origin-referenced sign test (pitch
  centre in front of every camera) is implemented and verified on real calibration.
- **Foot-projection robustness** — DONE: v2 plausibility windows, F4 Halpe heel/toe
  ground contact (`foot_contact_mode: v3`), H5 NaN-confidence guard in the ground solve.
- **Full proxy panel + GT** — PARTIALLY DONE: panel now carries id-persistence, excess
  fragments, chimera counts, per-cue d′. OPEN: the ID-switch-without-cause proxy, and
  MOTA/IDF1 via `evaluate_ground_truth` (blocked on hand labels — flagged to the team as a
  parallel workstream).
- **Pose-descriptor perf** — OPEN (unchanged): P3 wall time is acceptable (~7 min for 8
  deliveries in parallel); revisit only if it grows. Related perf backlog (P3 appearance
  decode threading, observe_frame vectorization) tracked in
  `docs/critical-analysis/review-triage.md`.

## Still genuinely open (consolidated)

1. Identity ground truth labelling (few hundred frames on `_7`/`M2`) → real IDF1/HOTA.
2. ID-switch-without-cause proxy metric.
3. Per-frame-fallback weight sweep + spatial prior (only if the fallback returns).
4. Perf: P3 appearance decode threading; `observe_frame` vectorization; descriptor caching.
5. Wave 5 probe: tiled/hi-res detection for small/dark-subject recall (research-backed,
   highest upstream ROI). Wave 6: role-focused suppression of low-confidence peripheral
   identities (user directive; last).
