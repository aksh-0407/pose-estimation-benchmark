# Global-ID Perfection Plan — PipeTrack v3

Goal: **every physical player carries exactly one `global_player_id`, identical in all 7
cameras, for the whole delivery** — judged by you on one mosaic video, guarded by the
proxy-metric panel so we never hand over a regression.

This plan is grounded in the actual numbers of the run behind the mosaic you watched
(`benchmarks/runs/pipetrack_v3/p4`, delivery `CCPL080626M1_1_14_1`, 600 frames, 7 cams).

---

## 1. Why the mosaic looked very poor (measured, not guessed)

The mosaic you reviewed is the main `p4` run. Its own metrics describe exactly what you
saw:

| Symptom in mosaic | Metric behind it |
|---|---|
| Same player, different ID in different cams | `cross_camera_agreement_rate = 0.689` → ~1 in 3 co-observed pairs disagree |
| IDs keep changing / new IDs appear | **22 distinct IDs** minted for ~8 visible people; 27 spawns, 33 deletions in 12 s |
| ID jumps from one player to another | **113 teleport events** (IDs moving up to 104 m/s) |
| Chips flicker between cameras | P3 re-clusters every frame from scratch; 67.5% of all clusters are single-camera |
| Fragmented tracks | 236 confirmed segments; P4b stitched only 3 links |

Mechanism (verified in code, not speculation):

1. **P3 is per-frame with almost no memory.** `associate_frame` re-clusters every frame;
   the only temporal signal is `TemporalLinkMemory`, a bounded −0.25 cost nudge. Under
   foot-projection noise (cluster `ground_spread_m` p95 = 2.46 m, gates 2.5/3.5 m) the
   same two detections merge one frame and split the next.
2. **P4a ownership is first-come-and-permanent.** `_local_owners` binds a (camera, P2
   tracklet) to the first global track that claims it and never lets go. When P3 flickers,
   the split half of a player can't reach its owner (already "hit" that frame) → a **new
   track is spawned → new ID**. The counters prove the contention: 2,711
   `local_track_reassignment_conflicts_prevented` and 1,214 `local_identity_ambiguous`
   events in 600 frames.
3. **Identity is decided where the signal is weakest.** All cues (ground distance,
   appearance, pose) are evaluated **per frame**, at their noisiest, with fixed scalar
   gates. The `ab_gate` experiment (facing gate 2.5→3.5 m: 0.646→0.720, +7.4 pts) proves
   the gate model — not the soft cues — is the binding constraint.
4. **The strongest identity substrate is barely used.** P2 produces just **26 long,
   clean tracklets** across all 7 cameras (0 intra-camera switch proxy). Deciding
   identity per tracklet-pair aggregates hundreds of co-visible frames of evidence
   (noise ∝ 1/√N); deciding per frame throws that away.
5. **Appearance is underexploited.** Frames show three visually distinct groups at the
   crease (batting blue/orange, fielding navy/yellow, umpires maroon) — yet appearance
   is a 0.2-weight per-frame histogram on washed-out, desaturated footage whose
   saturation mask (S≥24) likely discards most pixels. Never aggregated per tracklet.

**Conclusion:** this is an architecture problem, not a tuning problem. The fix is to
decide identity at the **tracklet-graph level** (offline over the whole delivery — which
matches your validation workflow), with per-cue uncertainty calibrated from the data
itself, and to make P4 identity bindings revocable instead of first-come-permanent.

---

## 2. Target architecture

```
P1 (unchanged)   per-camera detect + pose
P2 (unchanged*)  per-camera tracklets            *purity audit + dedup fix only if audit demands
P3 pass A        per-frame pairwise cue evidence (ground-Mahalanobis, appearance,
                 epipolar, pose) accumulated per cross-camera tracklet pair
P3 pass B  NEW   TRACKLET GRAPH: nodes = tracklet chunks, edges = calibrated
                 log-likelihood-ratio (LLR) fusion of aggregated cues,
                 cannot-link = same-camera temporal overlap,
                 solved by constrained agglomerative clustering + local refinement
P3 pass C        per-frame correspondences EMITTED from the stable bindings
                 (per-frame fallback only for untracked detections)
P4a              consumes bindings: one binding = one global ID for life;
                 Kalman keeps bridging gaps; ownership becomes evidence-based/revocable
P4b              descriptor-aware stitching of the remaining temporal fragments
```

Everything lands behind config flags; with flags off the pipeline reproduces today's
output byte-identically. Each phase has its own A/B + metric gate.

---

## 3. Phases

### Phase 0 — Measure, calibrate, and upgrade the review loop (no behavior change)

*New diagnostics (in `pose_estimation/cricket/tracking_metrics.py` + runners):*
- **Binding-churn metric**: how often a (camA tracklet, camB tracklet) co-membership
  toggles between consecutive frames. This is the direct number for "chips flicker".
- **Tracklet purity scan** (`scripts/tracking/` diagnostic): within-tracklet ground/
  appearance/height discontinuity detection → is P2 really clean? (Your "I hope tracking
  is fine" gets a number. dormant re-ID joins, 9 events, get audited too.)
- **Same-camera duplicate scan**: two simultaneous overlapping tracklets on one person.
- **Empirical residual study** (`scripts/association/calibrate_cues.py`, new): from
  unambiguous same-player cross-camera pairs (isolated players, e.g. bowler/non-striker)
  and definite different-player pairs (>4 m apart), fit per-camera-pair distributions for
  every cue: cross-camera ground residual (vs distance-to-camera), appearance distance,
  height difference, pose-descriptor distance. Output: `cue_calibration.json` —
  the variances and LLR curves everything downstream uses. **This is where "put in
  numbers so view variation is accounted for" happens — measured, not hand-tuned.**
- **Cue power report**: per-cue same-vs-different AUC/d′ on this very footage, so we
  know which cues carry signal before trusting them.

*Review-loop upgrades (`scripts/visualization/render_phase1_videos.py`):*
- Global-ID → stable color everywhere; per-camera ID chips already exist.
- New **ID timeline strip** (which IDs alive when, per camera) + per-frame
  cross-camera agreement flag on the summary tile, so 30 seconds of scrubbing shows
  where identity breaks instead of you hunting for it.

*Gate:* baseline metrics locked and reproduced; calibration file exists; no output change.

### Phase 1 — Uncertainty-aware ground geometry (replaces fixed scalar gates)

- `pose_estimation/cricket/geometry.py`: `ground_covariance(pixel, P, sigma_px)` —
  propagate foot-pixel noise (σ scaled by bbox height + keypoint confidence) through the
  ray–plane Jacobian → per-detection 2×2 ground covariance (elongated along the view
  ray; exactly the anisotropy the crude 3.5 m gate approximated). Floor/scale calibrated
  from Phase-0 residuals.
- P3 pair gating/cost: fixed-gate Euclidean → chi²-gated **Mahalanobis** under Σa+Σb.
  Cluster consensus: median + max-spread → **covariance-weighted fusion + chi² check**.
- P4a: measurement noise R from the fused cluster covariance (capped) instead of one
  scalar — the changes_tbd "distance-scaled measurement noise" item, done properly.
- Config: `ground_model: fixed | covariance` (default `fixed` until validated).

*Gate:* with `covariance` on, agreement ≥ ab_gate's 0.720 (it must subsume the gate win),
collisions 0, teleports not worse, bowler tracks intact.

### Phase 2 — Tracklet-graph identity (the centerpiece)

New module `scripts/association/tracklet_graph.py`:

- **Nodes**: P2 tracklets, pre-split at purity breakpoints from Phase 0 (guards against
  any P2 identity switch; over-splitting is harmless — the graph re-joins).
- **Evidence accumulation (pass A)**: for every cross-camera tracklet pair that ever
  falls inside the (Mahalanobis) gate, log per-frame cue values: ground Mahalanobis,
  appearance distance, epipolar residual (non-degenerate pairs only), pose descriptors,
  ground velocity.
- **Edge score (LLR fusion)**: each cue → log-likelihood ratio via `cue_calibration.json`;
  robust aggregation over the overlap (median/trimmed mean + support count);
  cues abstain (LLR 0) when unobservable — same fail-open philosophy as now, but
  quantitative. Includes a **negative** term: consistently-far co-visible pairs push the
  edge strongly negative.
- **Constraints**: hard cannot-link for same-camera tracklets overlapping > 2 frames;
  one tracklet per camera per cluster at any instant.
- **Solve (pass B)**: constrained agglomerative merging in descending LLR with
  consistency re-check at each merge, then a **local refinement loop** (move/split/swap
  single tracklets while total LLR improves) — this fixes the changes_tbd
  "single-linkage can never reconsider" defect.
- **Within-camera re-ID for free**: clusters may contain non-overlapping same-camera
  tracklets, so a broken P2 track re-joins itself through cross-camera evidence —
  global optimization the current P4b (3 links) can't do.
- **Emit (pass C)**: per-frame correspondences built from cluster bindings (+ current
  per-frame path only for untracked detections, which stay low-confidence).
- Runs offline over the whole delivery (the runner already loads all 600 frames into
  memory; validation is offline anyway). The accumulate/solve API is windowed so an
  online sliding-window variant can follow later without redesign.
- Config: `association_mode: per_frame | tracklet_graph` (+ merge threshold, min
  co-visibility, refinement iterations).

*Gate:* binding churn ≈ 0; agreement materially ↑ from 0.689 (expect ≥0.85 already);
single-camera rate ↓; collisions 0; distinct-ID count drops toward the true roster.

### Phase 3 — Identity lifecycle hardening in P4

- **Revocable ownership** (`track_manager.py`): `_local_owners` → per-(cam, tracklet)
  vote history with decay; transfer after sustained contradiction (N consecutive
  frames of counter-evidence), every transfer counted in diagnostics. Kills the
  "one bad frame welds two players together forever" failure — the likely source of
  the bowler ID switching in the recent runs.
- **Binding-first assignment**: in `tracklet_graph` mode a binding maps 1:1 to a global
  track for its lifetime; geometry Hungarian only handles unbound leftovers.
- **Re-entry discipline**: descriptor agreement (appearance + pose aggregate) required
  on top of the kinematic gate; tighten `reentry_kinematic_slack`. 104 m/s teleports
  must be structurally impossible, not just rare.
- **Capacity-aware births**: soft roster prior from rolling max simultaneous count —
  when the roster is full, strongly prefer reviving a recently-lost ID over minting
  P023. Plus the changes_tbd **continuous adaptive lost-window** (confirm-count-driven,
  detection-density-aware) replacing the dead bowler special-case.
- **P4b stitching v2** (changes_tbd item): add `descriptor_distance` (pose) +
  appearance terms to `build_link_costs`, use Kalman-smoothed exit/entry velocities
  instead of raw last-two-frame velocity.

*Gate:* teleports ≈ 0; distinct IDs ≈ visually-counted people (I count them from
sampled frames first); zero collisions; every dominant track from baseline still whole.

### Phase 4 — Cue strengthening (each independently flagged + ablated)

- **Pose v2 — your pose-based ID request, made view-proof.** New
  `ground_anchored_skeleton()` in `pose_estimation/cricket/pose_shape.py`: each
  detection's keypoints are lifted to metric 3D on a vertical "billboard" at the
  player's own ground depth — **per camera, no cross-camera triangulation, so it works
  on the facing pairs where triangulation is degenerate**. From it: standing height,
  shoulder-, hip-height, vertical limb extents. Vertical quantities are
  foreshortening-free; widths are comparable exactly on facing (anti-parallel) pairs.
  Per-tracklet robust aggregation (e.g. p90 standing height over upright frames,
  SE = σ/√N). Comparison via z-scores against Phase-0 calibrated per-camera-pair
  variances → LLR term. Two players differing 5+ cm in height become separable even
  when standing 1 m apart in identical kit; when heights genuinely match, the cue
  abstains instead of lying. The existing triangulated 3D descriptor stays as a second
  channel where parallax is good.
- **Appearance v2**: per-tracklet aggregated descriptor (median histogram over
  confident frames); fix the desaturated-footage problem (CLAHE / lower saturation
  cutoff / Lab-chroma histogram — chosen by which maximizes same-vs-different d′ in the
  Phase-0 harness); torso/legs split descriptor. Used in graph edges, re-entry, and
  stitching. (Optional stretch, only if color saturates: tiny ONNX ReID embedding —
  decided by ablation, not by default.)
- **Motion cue**: ground-velocity correlation over the overlap window (same person seen
  from two cameras moves identically; abstains when both are static).

*Gate:* per-cue ablation table (graph on, cue on/off) — a cue ships only if it does not
reduce agreement/churn and helps at least one failure case visible in frames.

### Phase 5 — Tuning, self-validation, delivery

- Small sweep harness (`scripts/benchmarks/sweep_p3p4.py`): grid over the few real
  knobs (merge LLR threshold, chi² levels, capacity prior), emitting the full panel —
  agreement, churn, teleports, ID count, collisions, single-camera rate — into one
  table. Panel is read **jointly**; no single metric is optimized.
- **Self-inspection before your time is spent**: I render mosaic frames at fixed
  timestamps (and around every remaining disagreement event), read the images myself,
  and only when frame-level identity looks right do I render the one mosaic video.
- Deliverable to you: **one mosaic** + a half-page "what changed, what to look for"
  note + the metric table (baseline → final).

---

## 4. Success criteria (final run vs baseline `p4`)

| Metric | Baseline | Target |
|---|---|---|
| cross_camera_agreement_rate | 0.689 | ≥ 0.90 (proxy saturates ~0.95; mosaic is the judge) |
| distinct_global_id_count | 22 | ≈ true visible roster (~8–11, confirmed by frame count) |
| teleport_event_count | 113 | ~0 |
| same-camera collisions | 0 | 0 (hard invariant) |
| binding churn | (new) | ~0 |
| Human verdict on mosaic | "very poor" | every player one stable ID in all cameras |

## 5. Guardrails (the "don't ruin what works" contract)

1. Every change behind a config flag; all flags off ⇒ byte-identical baseline (proven
   in Phase 1/2 gates by rerunning and diffing).
2. Phase-by-phase A/B on the same frozen P2 output (`benchmarks/runs/pipetrack_v3/p2`);
   a phase lands only if no guard metric (collisions, teleports, dominant-track
   integrity, agreement) regresses.
3. Proxy metrics are read as a panel, never singly (each alone is gameable).
4. Tests stay green (47 now, ~30 added: covariance math, LLR calibration edge cases,
   graph solver invariants — cannot-link, one-per-camera, determinism — ownership
   transfer, teleport regression).
5. Env: pipeline runs in `cricket-rtmpose-l`; pytest via `cricket-yolo26x-pose` with
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=""`.

## 6. Known risks, stated honestly

- **One 12-second delivery** is our only tuning data → threshold choices are calibrated
  on it. Mitigation: all thresholds derive from measured distributions (not magic
  numbers), so they transfer as procedures; flag defaults stay conservative.
- The agreement proxy counts two genuinely different players within 1.5 m as "expected same"
  — near 0.95 it saturates; final judgment is your mosaic pass, by design.
- If the Phase-0 purity audit shows P2 tracklets are dirtier than the proxies claim,
  Phase 2's chunk-splitting absorbs it, but I'll report it and we may add one P2 fix
  (appearance-gated dormant re-ID) before proceeding.

## 7. Execution order & rough effort

0. Phase 0 — ~½ day (diagnostics, calibration harness, renderer upgrades)
1. Phase 1 — ~1 day (covariance model + gates + tests)
2. Phase 2 — ~2 days (graph module + emit path + tests)  ← biggest single lever
3. Phase 3 — ~1 day (ownership, re-entry, capacity, stitching v2)
4. Phase 4 — ~1–1.5 days (pose v2, appearance v2, motion, ablations)
5. Phase 5 — ~½ day + your mosaic review

Checkpoint for your eyes: **after Phase 5** (one mosaic). If anything looks off earlier
at my frame-level self-checks, I fix before rendering. If you prefer an early mosaic
after Phase 3 (architecture-only, before cue upgrades), say so — otherwise I'll bring
the finished one.
