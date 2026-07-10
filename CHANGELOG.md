# CHANGELOG — PipeTrack global-ID overhaul

All changes are behind config flags; `association_mode: per_frame` reproduces the
old pipeline byte-for-byte. Reference delivery: `CCPL080626M1_1_14_1` (600 frames,
7 cameras). Baseline = the run behind the first "very poor" mosaic.

---

## Round 1 — Tracklet-graph identity (the architecture fix)

**Problem.** Cross-camera IDs disagreed for everyone except the bowler
(agreement 0.689, 22 IDs for ~9 people, 113 teleports). Verified mechanism:
P3 re-clustered every frame from scratch (its only memory was a −0.25 cost
nudge), so cluster membership flickered under ~1 m foot-projection noise; P4a's
first-come-permanent tracklet ownership then turned every flicker into fresh
IDs and welded wrong merges together forever (2,711 blocked re-claims / 1,214
ambiguous-owner events in 600 frames).

**Changes.**
- `scripts/association/tracklet_graph.py` — identity is decided once per
  P2-tracklet pair over the whole delivery: per-frame cue evidence (ground
  Mahalanobis under a per-detection covariance, appearance, posture, motion) is
  aggregated over every co-visible frame, fused as calibrated log-likelihood
  ratios, and solved by constrained agglomerative clustering (cannot-link:
  same-camera temporal overlap) with move-refinement. Per-frame correspondences
  are then EMITTED from the stable bindings, so cross-camera agreement holds by
  construction.
- `scripts/association/cue_calibration.py` — no hand-tuned tolerances: same- and
  different-player cue distributions are bootstrapped per delivery from
  unambiguous geometric anchors (tight ground agreement + spatial isolation).
  Cues that cannot separate the populations abstain instead of guessing —
  measured verdicts on this rig: kit-colour appearance d′≈0.09 globally
  (useless: inter-camera colour processing differs more than kits do → later
  fixed with per-camera-pair fits), ground residual same-player median ~1.0 m
  (calibration bias, absorbed into the covariance floor).
- `pose_estimation/cricket/pose_shape.py` — the pose-shape identity layer:
  keypoints lifted to metric 3D on a vertical "billboard" plane at the player's
  own ground depth (per camera, no triangulation → works on the low-parallax
  facing pairs). Tracklet-aggregated stature/shape quantities compared as
  z-scores. This cue separates the crouching keeper from the striker at 0.7 m
  (z≈14-18) where ground position is blind — and needed a squat-aware upright
  test (hips ≥ 0.6 m) plus view-aware abstention for crouched bodies between
  oblique views.
- `scripts/global_id/track_manager.py` — Stage-0 binding continuity (one binding
  = one persistent track), revocable tracklet ownership (TTL) instead of
  first-come-permanent.
- Asymmetric cue clipping everywhere: agreement on any single cue is weak
  evidence of identity (players can share position, kit, and build); strong
  disagreement is near-conclusive. Merges need corroboration from ≥2 cues.

**Result on the reference delivery.** Agreement 0.689 → 0.952 (remaining
"disagreements" are striker-vs-keeper proxy blindness — genuinely different
people <1.5 m apart, correctly separated), 22 → 15 IDs, 113 → 11 teleports,
236 → 45 track fragments, collisions 0. Verified visually across all 7 cameras
including the batsmen-crossing moment.

## Round 2 — Feet approximation + synthetic tracklets (cut-off players)

**Problem.** Players cut off at the frame bottom (cam_01's striker close-up) or
never tracked by P2 at all (umpires: dark, cut off, all keypoints ~0.06) had
garbage or missing ground positions — the bbox-bottom projects the FRAME edge,
not the player — so they could not bind and carried stray IDs.

**Changes.**
- `upper_body_ground_estimate` (geometry): when feet are unusable, intersect a
  ray through the hips (z=0.93) / shoulders (1.42) / bbox-top-as-head (1.78)
  with that height's horizontal plane — lands directly above the feet, most
  accurate exactly for the close-to-camera subjects that get cut off. Applied
  STICKY per tracklet (per-frame triggering flip-flops anchors by ~1 m and
  shatters tracks); a failed estimate yields NO position, never the garbage
  fallback.
- Synthetic tracklets in the graph: persistent untracked detections chained by
  ground continuity so umpires become bindable nodes (long occlusion memory,
  capped re-acquisition radius; only untracked detections can ever join).
- Appearance recalibrated PER CAMERA PAIR (each pair's own anchors); pairs
  without a separable fit abstain — this un-sabotaged every cam_07 pairing.
- Evidence floors on rescues and refinement moves (≥30 co-visible frames) after
  an 18-frame rescue built a fielder chimera.

**Result.** Striker bound across 5 views including the cut-off one; main umpire
one ID in cam_01+cam_04 end-to-end; keeper (= PARNATE #5) unified across 4 views
including cam_01 bottom-edge fragments. Agreement 0.9525, 13 IDs, teleports 9.

## Round 3 — Calibration-derived mosaic + occlusion ghosts

- `scripts/visualization/mosaic_layout.py` — no hardcoded camera ids: facing
  pairs (from calibration) form columns; the end camera looking WITH the
  delivery goes top-left; side tiles mirror so the delivery reads one direction;
  pano bottom-middle. Bowling direction = pitch axis (stump mid-bases in the
  pitch calibration) + fastest early run along it (per-camera tracklets, never
  fused tracks whose inter-camera bias reads as motion). Flipping the bowling
  end flips the whole layout — validated for real on M2, which is bowled from
  the other end.
- Mirror-safe overlays: flip the image first, draw text on transformed
  coordinates — labels never mirror.
- Global-roster panel: every ID in the delivery, live-now highlight with the
  cameras that see it, reserved ROLE column.
- Occlusion ghost markers: a player undetected in one camera but tracked by the
  others gets its fused position reprojected into the blind tile as a dashed
  "occluded" marker (render-side only; no synthetic data enters the pipeline).
- Hardcode audit: `cam_0[1-7]` regexes generalized to `cam_\d{2}`.

## Round 4 — Anti-shadow-ID + clip verdicts (all-8-deliveries round) — CURRENT

**Problem (from the 8-delivery batch).** Deliveries 1-2 scored 0.95-0.98 with
13-14 IDs; the rest over-minted badly (up to 54 IDs / 140 teleports on M2).

**Why the first two videos had good ID-work while others failed** — measured,
not guessed: in deliveries 1-2, P2 produced ~24 tracklets with **median length
600** (unbroken, full clip), so one tracklet-cluster ≈ one person and the graph
had rich pairwise evidence. In the weak clips (darker footage, diving run-outs,
repeated running between wickets) P2 tracklets shatter — median length 156-259,
up to 127 chunks — and the Round-1 design silently assumed long tracklets:
**every chunk-cluster earned a binding ID and P4 confirmed all of them**
(geometry-spawned IDs were 0-3 per clip; the explosion was binding-spawned
fragments). Separately, the "ghost + different ID on top" artifact: a track
already updated this frame by other cameras could not receive this camera's
unbound detection, which then birthed a duplicate at the exact same spot.

**Changes.**
- Trajectory attach (graph): fragments that ride a binding's fused trajectory
  (the same information the ghost markers draw) for most of their life get
  attached to it — cannot-link respected, posture-vetoed, unique-candidate.
- Binding demotion: only multi-camera clusters or one long stable single-camera
  track (≥150 frames — an umpire) earn a binding ID; short fragments emit as
  unbound low-confidence detections.
- P4a Stage 2.5 "absorb, don't birth": an unmatched observation inside the chi2
  gate of a track already updated this frame joins it identity-only — the
  duplicate-on-top-of-the-ghost can no longer be born.
- Shadow-suppressed confirmation + cricket roster cap: a tentative sitting
  within 1.2 m of a confirmed track does not confirm until it separates or
  persists ≥30 hits (real second player, e.g. batsmen together); at the
  15-player cap a new ID additionally needs ≥3 m separation.
- Clip quality verdict in `global_id_metrics.json` (`quality_verdict`):
  pass/warn/fail from the two blind-trustable numbers — distinct IDs vs the
  15-player roster bound, and teleport events. Failing clips are flagged for
  work instead of review.
- P5 roles phase scaffold: `scripts/roles/run_role_assignment.py` writes
  `p5/roles.json` per the phase-folder convention; the mosaic roster reads
  roles only from that artifact. `scripts/roles/assigner.py` holds a documented
  heuristic v0 (bowler from the run-up signal, keeper/umpire/batsmen from
  stump-relative positions) with honest confidences — designed to be replaced
  rule-by-rule by the improved role work.

**Acceptance test for this round:** the full 8-delivery before/after table —
deliveries 1-2 must not regress; the over-minting clips must collapse toward
the true roster; anything still failing is declared FAILED by its own metrics
rather than shipped.

**Measured result (all 8 deliveries, P2 reused):**

| delivery | agreement | ids before -> after | teleports | verdict |
|---|---|---|---|---|
| M1_1_14_1 | 0.952 | 13 -> 12 | 11 | pass |
| M1_1_14_2 | 0.977 | 14 -> 11 | 7  | pass |
| M1_1_14_3 | 0.870 | 20 -> 18 | 19 | pass |
| M1_1_14_4 | 0.857 | 15 -> 13 | 15 | pass |
| M1_1_14_5 | 0.767 | 22 -> 15 | 48 | warn |
| M1_1_14_6 | 0.802 | 41 -> 25 | 52 | warn |
| M1_1_14_7 | 0.498 | 36 -> 22 | 59 | warn |
| M2_1_12_1 | 0.778 | 54 -> 20 | 171 | fail |

Non-regression on the good clips held (and they got cleaner). The kill-chain
counters show demotion carried most of the load (87 clusters demoted on M2),
absorption caught 2-35 shadows per clip, shadow-blocking 0-72 confirmations.
Teleports ROSE on the hard clips: consolidation moves noisy fragment positions
under one id, so their jumps now count against that id — the honest price of
fewer ids, and exactly what the verdict system is for (M2 self-flags as fail;
its true fix is upstream P1/P2 quality in low light). Known issue handed to the
roles work: v0 role assignments on delivery 1 look end-swapped — the assigner
derives the bowling-direction sign from FUSED tracks (unreliable; use
per-camera tracklets as mosaic_layout does) and scores the run with abs().

## Round 5 — v5 ID overhaul: under-merge + fragmentation + ghost markers — CURRENT

All changes behind config flags (`configs/p3_association_v5.yaml`,
`configs/p4_global_id_v5.yaml`); flags off ⇒ byte-identical baseline (proven on
delivery 1) and all 152 unit tests green. Baseline frozen at
`benchmarks/runs/pipetrack_v3/_baseline_snapshot`; runs land in `pipetrack_v5`.
Batch driver: `scripts/pipetrack/run_id_pipeline.py`. Method log:
`wip/methods_log.md`.

**ID-1 cross-camera under-merge (the 0.50 agreement on _7).** Root cause: the graph
merge threshold (2.0) exceeds the single-cue positive cap (1.5), so on the
low-parallax facing pairs — where appearance/motion abstain — ground alone can
never merge a genuine same-player pair. Fix (`tracklet_graph.py`): a
corroboration-aware second merge pass (`graph_corrob_merge`) that admits a
below-threshold edge only with full support, no disagreeing cue, mutual-unambiguous
best, and cannot-link respected; plus a parallax-adaptive facing-pair gate
(`graph_facing_gate_scale`). **Result: _7 agreement 0.498 → 0.600 (+0.102)**,
teleports −13, single-camera rate −0.051; easy clips byte-identical; collisions 0.

**ID-2 fragmentation (18–25 ids for a ~13–15 roster).** Root cause, measured: the
graph already yields ~10–11 clean bindings, but P4 emitted 18–25 ids because P4b
stitching selected **0** links (its dummy "new-trajectory" cost 0.5 undercut every
real stitch) and many *demoted* clusters briefly confirmed as ultra-short ids
(6–25 frames). Fixes: stitching v2 (`stitching.py`) — pose-shape descriptor per
segment + hard pose gate (`p4b.pose_stitch_max_distance`), Kalman-smoothed
exit/entry velocities, and a usable `new_traj_cost_factor`; a cardinality prior
(`p4a.min_emit_frames`) dropping any id whose whole-clip span is < 30 frames (a
fragment, not a player present the full 12 s); adaptive lost-window, pose veto in
the chi² gate, and descriptor-gated re-entry in P4a. **Result (all 8): every clip's
id count collapsed toward the roster** (e.g. _6 25→16, _7 22→15, M2 20→14, _3 18→13)
and **teleports fell on every clip** (_1 11→2, _4 15→6, _6 52→40, _7 59→44), with
agreement stable-or-up and same-camera collisions still 0.

**Ghost markers v2 + mosaic/BEV modernization.** `geometry.ground_point_visible_in`
(cheirality + in-frame per camera) drives ghost markers for *disappeared* ids in
**every** camera that frames that ground (occluded vs lost, aged/faded), a
last-known-position store, and greyed ghost dots in the bird's-eye view. The BEV
tile was rebuilt as a metric cricket-field radar (uniform scale, 30-yard ring,
pitch strip + creases, scale bar). Colour system unified on `identity_colors`
(standalone BEV join bug + golden-ratio hash retired); NVENC is now reported
correctly in the manifest. The same pose-corroborated fragment merge doubles as the
in-pipeline "ghost verification" (a lost id's fragment only re-joins when body shape
agrees). Audit fixes: `ground_kalman.cap_covariance` (both axes), fragment
posture-veto aggregate.

**Remaining:** M2 teleports (166) are largely a teleport-*proxy* artifact on M2's
noisy single-camera foot projections (worst single-cam rate 0.61), not emitted-
trajectory jumps; the emitted Kalman posterior stays smooth. Config promotion to the
committed defaults is held for the WS5 review.
