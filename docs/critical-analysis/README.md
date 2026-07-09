# Critical Analysis — Cricket 3D-Pose & Identity Pipeline

An expert, evidence-grounded review of the pipeline that turns 7-camera cricket footage into
per-player 3D pose, stable global identity, and the mosaic render. For every phase it walks
the methods and their math, separates the pros from the cons, enumerates the issues (with
`file:line` evidence and the measured proxy that proves each one), and prescribes **all** the
fixes in priority order, each backed by a verifiable source.

Nothing here is invented: quantitative claims trace to the repo's own logs (`wip/*.md`,
committed `*_metrics.json`) and every external method is cited in [references.md](references.md).

## The one-paragraph thesis

The calibration is **centimetre-accurate** (ball reprojection p95 ≤ 4.5 px), and the 3D
**location** problem is largely solved: the `z0_reproj` ground solver cut emitted ground error
~36% (0.211 → 0.147 m mean) and the Kalman-posterior emit halved trajectory jitter (worst
per-frame jump 14.0 → 0.36 m). **Identity is now the dominant quality ceiling.** Players are
placed correctly but their IDs swap and fragment: cross-camera agreement falls to 0.50 on the
hardest clip, distinct-ID counts run 18–25 against a ~13-person roster (40–90% over-
segmentation), teleports reach 7–155/clip, and the colour-appearance cue is **statistically
dead** (d′ ≈ 0) because both teams wear near-identical kit in desaturated footage. The two
structural causes are (1) the co-observing camera pairs are **low-parallax facing pairs**
(C1↔C4, C2↔C6, C3↔C5) where epipolar geometry is ill-conditioned, and (2) the association
clustering is **single-linkage** — it can merge but never split, so an early wrong merge is
permanent. Sources: `wip/id_issues.md`, `wip/3d_location_issues_v2.md`.

## Evidence at a glance (v4/v5, all 8 deliveries)

| Delivery | single-cam | X-cam agreement | distinct IDs | teleports | appearance d′ | cluster cyc-consistency |
|---|---|---|---|---|---|---|
| M1_1_14_1 | 0.39 | 0.95 | 12 | 11 | 0.09 | 0.82 |
| M1_1_14_5 | 0.56 | 0.77 | 16 | 43 | 0.00 | 0.72 |
| M1_1_14_6 | 0.46 | 0.80 | 25 | 52 | 0.00 | 0.88 |
| M1_1_14_7 | 0.52 | **0.50** | 25 | 56 | 0.57 | 0.70 |
| M2_1_12_1 | 0.61 | 0.77 | 21 | **155** | 0.96 | 0.68 |

(Source: `wip/id_issues.md` §0, `wip/3d_location_issues_v2.md` §0. Same-camera collisions are 0
everywhere — a hard invariant held by construction.)

## Reading order

1. [phases.md](phases.md) — the ordered pipeline (current **and** the proposed re-order) with flowcharts.
2. Per-phase deep dives:
   - [phase-1-inference.md](phase-1-inference.md) — P1: detection + 2D pose (RTMDet + RTMPose-X).
   - [phase-1b-2d-stabilization.md](phase-1b-2d-stabilization.md) — P1.5: 2D temporal stabilization (new).
   - [phase-2-tracking.md](phase-2-tracking.md) — P2: per-camera tracking.
   - [phase-3-association.md](phase-3-association.md) — P3: cross-camera association (the identity core).
   - [phase-triangulation-3d.md](phase-triangulation-3d.md) — the 3D lift and its optimal placement.
   - [phase-4-global-id.md](phase-4-global-id.md) — P4: global identity + stitching.
   - [roles-render-export.md](roles-render-export.md) — P5 roles, UE export, mosaic/BEV render.
3. [fixes-roadmap.md](fixes-roadmap.md) — the cross-cutting, priority-ordered fix roadmap.
4. [references.md](references.md) — repo anchors + verified external sources.

## How to read a phase doc

Each follows the same template: **Role & intuition → I/O & config → flowchart → methods
walkthrough (with math and `file:line`) → pros / cons (separately) → issues (with evidence) →
fixes (all of them, priority-ordered)**. "Priority" is impact-on-final-quality × confidence,
not effort; effort/blast-radius is a separate column.
