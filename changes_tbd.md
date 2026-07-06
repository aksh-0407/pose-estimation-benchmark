# changes_tbd.md — deferred work for global-ID stability

> **2026-07-05 status.** The tracklet-graph identity layer
> (`association_mode: tracklet_graph`, `scripts/association/tracklet_graph.py`)
> landed and supersedes several items below: "P3 temporal cluster memory"
> (bindings replace the nudge), "clustering-algorithm refinement" (LLR
> agglomeration + move-refinement + rescue pass can reconsider),
> "distance-scaled measurement noise" (per-detection ground covariance via the
> ray-plane Jacobian), and the pose cue is now the ground-anchored billboard
> posture layer with empirically calibrated tolerances (`cue_calibration.py`).
> Panel on this delivery: agreement 0.689→0.952, teleports 113→11, IDs 22→15,
> collisions 0. Still open below: weight/gate tuning for the per_frame fallback,
> adaptive lost-window, P4b descriptor stitching, role wiring, cheirality.

Captured while landing the safe, cross-camera-focused work (measurement guardrails +
degeneracy/C07 fix + view-invariant pose tie-breaker). These are the *next* levers,
ordered roughly by value-for-risk. None should be started before the current work is
validated on the 7-camera mosaic + ground video. There is **no identity ground truth**,
so every item below is tuned against the mosaic and the (jointly-read, individually
gameable) proxy metrics, never a single number.

## Higher value

- **Weight/gate tuning (empirical).** `ground_weight` / `epipolar_weight` /
  `appearance_weight` / `temporal_link_bonus`, and the gates
  `ground_distance_gate_m` (3.5), `opposite_pair_ground_gate_m` (2.5),
  `ground_cluster_gate_m` (2.75). Note the facing-pair gate is *tighter* (2.5 m) than
  the general gate — under foot-projection noise this can itself cause the
  under-merges that produce cross-camera disagreement. Sweep against the mosaic.
- **P3 temporal cluster memory (safe, one-way).** Strengthen the self-contained
  `TemporalLinkMemory` (`scripts/association/associator.py`) — longer confirm window,
  decay, higher `temporal_link_bonus` — and/or feed a *confirmed-track spatial prior*
  (a ground-distance nudge toward predicted positions) into P3 that never carries
  identity labels. Reduces frame-to-frame cluster flicker without coupling the stages.
- **Continuous adaptive lost-window (role-free).** Replace the dead bowler special-case
  in `GlobalTrack.should_delete` with a window driven by confirm-count (capped) and
  local detection-density at the moment of loss (a scrum/dive lengthens it). Keeps
  well-tracked players alive across occlusions without a role classifier.
- **Pose tie-breaker at P4b stitching.** Carry the accumulated `pose_proportions` into
  `scripts/global_id/stitching.py` and add a low-weight `descriptor_distance` term when
  two segment endpoints are spatially close at a gap — same soft, abstaining rule as
  P4a. Also use the Kalman's own smoothed exit/entry velocity instead of raw last-two-
  frame velocity.

## Medium value

- **Cross-camera pose CLUSTER GATE / veto at P3 merge time** (torso reprojection +
  anthropometric). *Deliberately deferred as risky:* the co-observing pairs are the
  low-parallax pairs, so a hard gate false-rejects correct merges; single-linkage never
  splits, so a blocked correct 3rd-camera join *spawns* a fragment; it can regress the
  bowler during the stride; and it is weakest on along-axis (depth) swaps. Only revisit
  with parallax-adaptive tolerance + mandatory abstention, and as a soft confidence
  penalty (already prototyped via `torso_anthropometric_ok`) before ever a hard veto.
- **Online role wiring.** Call `TrackManager.propose_role` from the DRS `events.json`
  release frame (bowler = nearest to ball at release), then let the existing role-aware
  Singer params / `bowler_lost_window_frames` / P4b `incompatible_role_pairs` /
  role-compatible re-entry veto activate. Separate feature; unvalidatable without GT;
  not the cause of non-bowler instability, so do it for role *labels*, not as the fix.
- **Distance-scaled measurement noise (capped).** Inflate P4a `measurement_noise` for
  distant/low-parallax fielders using a triangulation-uncertainty proxy, capped so two
  fielders' gates can never overlap (that case routes to the pose tie-breaker instead).
- **Clustering-algorithm refinement.** Replace greedy single-linkage union-find in
  `_constrained_cluster` with a method that can reconsider/split (correlation clustering
  / iterative refine), so an early wrong merge is recoverable.

## Lower value / robustness

- **Cheirality check in triangulation.** Reject points behind a camera. This rig uses
  the negative-depth-in-front convention, so `w>0` is wrong; derive a per-camera
  in-front sign from a known reference (e.g. the pitch origin all cameras look at) and
  gate on it in `ransac_triangulate_point`. Low impact today (association is gated on
  z=0 ground, not raw 3D) but correct.
- **Foot-projection robustness.** Tighten ankle-plausibility in `ground_contact_pixel`
  so hallucinated ankles don't move the ground point metres under the long lenses.
- **Full proxy panel + GT.** Add an ID-switch-without-cause proxy; wire the existing
  `evaluate_ground_truth` (MOTA/IDF1) if identity labels ever exist. Keep reporting the
  proxy panel *together* — each single metric is gameable by a different pathology.
- **Pose-descriptor perf.** If P3 slows with `pose_descriptor_enabled`, cache per-camera
  centres and short-circuit the per-joint parallax scan; consider triangulating the
  descriptor only every N frames.
