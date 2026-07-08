# Identity Issues — critical review (all 8 deliveries, v4)

Identity is now the **dominant quality ceiling** of the pipeline: the 3D *locations* are good
(`3d_location_issues_v2.md` §1), but the mosaics/BEV show players placed correctly while their **IDs
swap and fragment**. This document states the identity failure modes with evidence, root cause, and
fix direction. No identity ground truth exists, so all figures are explicitly labelled proxies.

## 0. Evidence (v4, all 8 deliveries)

| Delivery | distinct IDs | max simul/cam | excess-fragment proxy | X-cam agreement | appearance d′ | ground d′ | teleports |
|---|---|---|---|---|---|---|---|
| M1_1_14_1 | 12 | 5 | 7 | 0.95 | 0.09 | 5.02 | 11 |
| M1_1_14_2 | 11 | 6 | 5 | 0.98 | 0.00 | 5.29 | 7 |
| M1_1_14_3 | 18 | 6 | 12 | 0.87 | 0.00 | 4.74 | 19 |
| M1_1_14_4 | 13 | 5 | 8 | 0.86 | 0.04 | 4.54 | 15 |
| M1_1_14_5 | 16 | 5 | 11 | 0.77 | 0.00 | 4.00 | 43 |
| M1_1_14_6 | 25 | 6 | 19 | 0.80 | 0.00 | 5.48 | 52 |
| M1_1_14_7 | 25 | 7 | 18 | **0.50** | 0.57 | 4.09 | 56 |
| M2_1_12_1 | 21 | 4 | 17 | 0.77 | 0.96 | 3.29 | **155** |

A cricket scene has ~13 people (11 fielders + 2 batsmen + 2 umpires ≈ 15 max) and **≤7 visible per
camera at once**. So **distinct-IDs of 18–25 (deliveries 3/5/6/7/M2) is 40–90% over-segmentation.**
Same-camera collisions are **0** everywhere (hard invariant holds by construction).

## 1. Issues

### ID-1 — Cross-camera UNDER-merge: one player gets different IDs in different cameras  ★★★
- **Evidence:** cross-camera agreement **0.50 on _7**, 0.77–0.80 on _5/_6/M2 (independent bbox-bottom
  ground projections, so this judges the clustering, not echoes it). The BEV shows a stationary deep
  fielder labelled P009 in some frames, P014 in others.
- **Root cause:** the co-observing pairs are the **low-parallax facing pairs** (C1↔C4, C2↔C6, C3↔C5,
  per `config/facing.jpeg`) — exactly where epipolar geometry is ill-conditioned (7/21 pairs flagged
  degenerate) and the appearance cue is dead (below). Ground proximity alone cannot bind two views of
  a player standing near others, and the facing-pair ground gate (2.5 m) is tight enough to *split* a
  correct merge under foot-pixel noise → the two views become two IDs.
- **Fix direction:** (1) **parallax-adaptive** cross-camera cost + gate (loosen where geometry is weak,
  lean on pose-shape); (2) promote the **pose-shape descriptor** (limb-ratio, view-invariant) from a
  soft tie-breaker to a primary cross-camera cue — it is the only discriminative signal on identical
  kit; (3) stronger temporal binding memory so a pair confirmed once stays bound.

### ID-2 — Identity FRAGMENTATION (over-segmentation): tracks re-acquired as new IDs  ★★★
- **Evidence:** excess-fragment proxy **5–19**; 18–25 distinct IDs vs a ~13 roster (deliveries 3/5/6/7/M2).
  `stitched_id_switch_proxy = 0` on all → **P4b stitching is not merging the fragments it should.**
- **Root cause:** a player lost through occlusion past the lost-window (30 frames; 60 for bowler) is
  deleted and re-born with a fresh `P###`; re-entry/re-ID is weak because (a) appearance is dead, (b)
  the pose-shape descriptor needs many frames to mature, (c) P4b's feasibility gates (temporal 120,
  kinematic) + occupancy-safety may be too conservative to bridge real gaps.
- **Fix direction:** (1) **adaptive lost-window** scaled by track maturity + local detection density
  (keep well-established players alive far longer); (2) a **stronger re-ID** using the mature pose-shape
  descriptor + kinematic prediction at re-entry; (3) loosen P4b bridging where occupancy proves two
  segments cannot be simultaneous; (4) a cricket **roster prior** (cap at ~15, penalise minting a new ID
  when the roster is full and an existing track just went missing nearby).

### ID-3 — Teleports: an ID jumps between two different people  ★★
- **Evidence:** teleport events **7–155** (M2 155, _6 52, _7 56, _5 43) — an ID moving faster than
  9 m/s between frames, i.e. re-assigned to a detection of a *different* person elsewhere.
- **Root cause:** in crowds/occlusion the χ²-gated Mahalanobis assignment admits a wrong nearby cluster
  when the true one is missing that frame; the emitted-posterior fix (M8) stops the *reported position*
  from teleporting but not the underlying mis-assignment.
- **Fix direction:** tighten the assignment with a **pose-shape veto inside the gate**, a hard kinematic
  reachability check on re-entry, and hysteresis (don't hand a track to a new tracklet on a single frame).

### ID-4 — The appearance cue is effectively dead  ★★ (root enabler of ID-1/2/3)
- **Evidence:** appearance **d′ ≈ 0.00–0.09** on 5/8 deliveries (only _7 0.57, M2 0.96). A d′ near 0
  means the colour-histogram distance cannot separate "same" from "different" players at all.
- **Root cause:** both teams wear near-identical kit and the footage is desaturated — colour histograms
  carry almost no identity information. The cue weight (0.20) is therefore mostly noise.
- **Fix direction:** stop relying on colour. Make **body proportions (pose-shape)** the appearance
  substitute, and evaluate a **learned re-ID embedding** (a small person-ReID net) as a stronger, still
  kit-robust descriptor; down-weight or disable the colour histogram where its per-delivery d′ ≈ 0.

### ID-5 — Greedy single-linkage clustering can merge but never split  ★★
- **Evidence:** cluster cycle-consistency **0.68–0.90** (`3d_location_issues_v2.md` V2-L2): 10–32% of
  ≥3-view clusters are geometrically inconsistent = likely chimeras that were merged and cannot be undone.
- **Root cause:** P3 uses union-find single-linkage; once two tracklets merge, no mechanism ever splits
  them, so an early wrong merge is permanent and propagates.
- **Fix direction:** a clustering that can **reconsider** — correlation clustering or an iterative
  refine/split pass gated on the M11 full-skeleton reprojection (a chimera fails torso/limb reprojection
  hard, giving a clean split signal).

### ID-6 — No identity ground truth → cannot report MOTA/IDF1  ★ (measurement)
- **Evidence:** every figure above is a proxy; `evaluate_ground_truth` exists but has no labels to run on.
- **Fix direction:** hand-label global IDs on a few hundred frames of 2–3 deliveries (including _7/M2)
  to get real **IDF1 / ID-switch** numbers, so identity work is measurable rather than proxy-guided.

## 2. Priority
The location work is largely done; **identity is where the remaining quality lives.** Highest value,
in order: **ID-4 → ID-1** (make pose-shape/learned-ReID the primary cross-camera cue, since colour is
dead and geometry is weak on the facing pairs), then **ID-2** (adaptive lost-window + stronger re-ID to
kill fragmentation), then **ID-5** (splittable clustering). ID-3 teleports and ID-6 labels follow.
This is the `implementation_plan.md` workstream; the evidence here should re-prioritise it around the
dead appearance cue and the facing-pair under-merge.
