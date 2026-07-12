# Status Report — Multi-Camera 3D Pose & Identity Pipeline
### Fix-campaign snapshot, 2026-07-10 (for technical review)

---

## 1. What the pipeline does today (as-built order)

Seven synchronized 50 fps cameras → per-delivery outputs: 2D skeletons, per-camera tracks,
cross-camera identity, world-frame 3D skeletons and positions, roles, diagnostic mosaics.

| # | Phase | Method (current) |
|---|---|---|
| P1 | 2D pose | RTMDet-m person detector + **RTMPose-X** top-down, **Halpe-26** skeleton (COCO-17 + head/neck/hip + 6 foot keypoints), all 8 deliveries × 7 cams |
| P1.5 | 2D stabilization | One-Euro filter + confidence-gated spike clamp over IoU micro-tracks (−20–34% keypoint jitter). *Opt-in pending composition verdict* |
| P2 | Per-camera tracking | ByteTrack-style two-stage Hungarian (IoU + pose-cosine), CV-Kalman; **new:** zero-IoU fast movers matched by normalized motion cost; process noise resets on re-acquisition |
| P3 | Cross-camera identity | Whole-delivery **tracklet graph**: calibrated per-cue log-likelihood ratios (ground distance/Mahalanobis, appearance, billboard posture, motion) fused per tracklet-pair; union-find merge + corroboration pass for facing pairs + refine/rescue/attach; ground position from **z=0-constrained robust reprojection** (Gauss–Newton + Huber), now with **2×2 posterior covariance** per cluster |
| P3.5 | 3D lift (new stage) | Per-**binding** RANSAC-DLT triangulation of the full skeleton (optionally all 26 Halpe joints), per-joint 3D covariance, **chimera purity report** (one-sided torso reprojection bias names the intruding camera), pooled bone-ratio descriptor + stature |
| P4 | Global identity | Singer-acceleration ground Kalman, staged assignment (binding → exact tracklet → χ²-gated Hungarian → shadow absorb → re-entry); **new:** measurement-covariance-aware update (asymmetric: conservative gate, uncertainty-weighted fusion), billboard-posture veto, online role proxy, occupancy-licensed stitching, posture stitch key; min-cost-flow fragment stitching |
| P5 | Roles | Positional/kinematic heuristics (bowler run-up now direction-signed) |
| P6 | Terminal 3D | Same lift keyed on final global IDs; occlusion fill (**frame-aware gap gating**), zero-phase Butterworth option, cheirality-gated RANSAC |
| R | Render | Calibration-derived mosaic + bird's-eye view + roster, colours by global ID |

Evaluation: 8-delivery joint metric panel vs a frozen baseline (`pipetrack_v6.0`), primary
axes per the project objective: **cross-camera agreement, distinct-ID count vs the ~13–15
roster, ID persistence (mean confirmed-frame completeness), excess fragments**; teleport
count is secondary (double-counts acceptable occlusion transients). Collisions must stay 0
(held everywhere, all runs).

## 2. Before → after (measured, all 8 deliveries)

Baseline `v6.0` → best current composed run `v7-rc1` (full panels in `fixes-log.md`):

| Axis | Baseline v6.0 | v7-rc1 | Verdict |
|---|---|---|---|
| Hardest clip `_7` agreement | 0.603 | **0.713** | ✅ +0.110 |
| `_7` distinct IDs (roster ~13) | 18 | **13** | ✅ at roster |
| `_7` teleports / fragments | 42 / 12 | 32 / 7 | ✅ |
| ID persistence (mean, 8 clips) | 0.78–0.96 | up on 6/8 (`M2` +0.077) | ✅ |
| 3D reprojection (mean px) | 3.2–3.6 | **2.9–3.3** | ✅ |
| Cycle-consistency (worst clips) | 0.67–0.70 | +0.04…+0.20 | ✅ |
| Chimera suspects (per clip) | unmeasured | measured 1–5, splittable | ✅ new capability |
| Agreement `_3`/`_5`/`_6` | 0.86 / 0.90 / 0.65 | 0.74 / 0.67 / 0.57 | ❌ regression |
| Multi-view binding (`_1` single-cam rate) | 0.271 | 0.670 | ❌ regression |

Key intermediate findings (each caught by an A/B, with root cause):

- **Uncertainty-aware Kalman R**: symmetric use loosens admission gates (wide R makes far
  wrong candidates look close) → split to *gate on fixed role R, update on measurement R*.
- **Cheirality test**: textbook det(M) sign formula is wrong on this rig's world handedness
  → replaced by an origin-referenced sign test (pitch centre is in front of every camera).
- **Bone-ratio shape cue**: self-calibrates then **abstains** on all 8 clips (d′<0.5) — body
  proportions do not separate players in this footage; consistent with 2026 literature
  (identical-kit ReID unsolved). The billboard **stature/posture** channel is the live path.
- **Chimera splitting works** (suspects 1–5 → 1–2 per clip; per-camera bias correctly names
  the intruder) but split pieces need re-absorption or they inflate ID counts.
- **P2 defects fixed** (external review, verified by execution): dead zero-IoU gate — the
  sprinting bowler always fragmented; process noise stuck ~57× after any occlusion.

## 3. Current issues (open, prioritized)

1. **Composition regression under attribution (active right now).** The composed stack
   collapses multi-view binding on the easy clips (`_1` 0.27→0.67 single-cam) while
   transforming the hard clip. Diagnostics localize it inside P3's merge passes
   (corroboration merges 2→0 on `_1`); prime suspect is an unflagged review fix (H3: posture
   stature-sample policy) shifting the calibrated posture distributions, plus stricter
   fragment-attach margins (H6). Two ablation runs (FR-P3-code × baseline-P2; old-P2-code ×
   P1.5) are running to separate code-side from data-side causes. Resolution path: demote
   H3/H6 to config-gated policies, recalibrate, re-compose (v7-rc2).
2. **Facing-pair identity remains the structural ceiling.** Colour is dead (d′≈0), bone
   ratios abstain, epipolar geometry is degenerate on C1↔C4 / C2↔C6 / C3↔C5. Current
   leverage is ground geometry + billboard stature + corroboration merging.
3. **M2 teleport proxy** stays elevated (single-cam rate 0.63–0.76): known to be proxy
   artifact of noisy lone-camera foot projections; persistence on M2 *improved* (+0.077).
4. **Detector recall on small/dark subjects** (umpires, deep fielders) — upstream cause of
   the synthetic-tracklet machinery. Research verdict: tiled/hi-res inference over the
   existing RTMDet is the highest-ROI probe (players are ~10 px at 640-scale); RTMO is
   rejected (COCO-17-only would lose the feet).
5. **No identity ground truth** — all identity numbers are proxies read jointly;
   `evaluate_ground_truth` (IDF1/MOTA) is implemented but needs a few hundred hand-labelled
   frames on `_7`/`M2`.

## 4. Where mentor input would help most

1. **Facing-pair identity evidence**: given dead colour + abstaining bone ratios + degenerate
   epipolar geometry, are there additional cues worth the effort that we have not weighed
   (e.g. skeleton-gait embeddings pretrained on GREW/Gait3D as a weak self-calibrated cue;
   temporal motion signatures), or should effort go to the geometric channel (tiled
   detection → better feet → tighter ground gates)?
2. **Ground-truth strategy**: cheapest labelling protocol that yields usable IDF1/HOTA on 2–3
   deliveries (frame sampling density, tooling), so tuning stops being proxy-guided.
3. **Cue-fusion statistics**: our LLR fusion sums per-cue evidence assuming independence, but
   ground residual and billboard posture share the same foot anchor (positively correlated
   errors). Is fitting an inter-cue correlation from the anchor pairs (shrinking the summed
   evidence) worth it, or is the per-cue positive cap an adequate guard?
4. **Composition methodology**: with ~15 interacting flags, full factorial A/B is infeasible;
   we use evidence-ordered greedy composition with targeted ablations on regression. Any
   recommended lightweight design-of-experiments discipline for this scale?
5. **Teleport metric**: we down-weighted raw teleport counts in favour of ID-persistence per
   the occlusion-tolerance objective — sanity-check this reweighting.

## 5. Deliverables in hand

- Frozen baseline `pipetrack_v6.0` (all phases, 8 mosaics, metrics snapshot) — documented
  for colleague use in `docs/shared-data.md`.
- Candidate `pipetrack_v7-rc1` (full chain incl. P3.5 + native-26 3D); mosaics for
  `_1`, `_2`, `_7`, `M2` in `artifacts/pipetrack_v7-rc1/mosaics/`.
- Complete audit trail: `fixes-log.md` (every fix: mechanism, panel, verdict),
  `review-triage.md` (external review: verified/fixed/deferred), `to-do.md` (campaign plan,
  Waves 5–6 pending), 192 unit tests green.
