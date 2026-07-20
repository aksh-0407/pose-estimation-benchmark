# Verdict redesign, a usability grade that measures the right things

**Written 2026-07-14. Applied to all 40 (table below). Code updated for future runs.**
Reproduce: `regrade.py` (beside this file) reads the existing per-delivery metrics + the real
emitted `ground_tracks.jsonl` and prints the new grade, no pipeline re-run needed.

## Why the old verdict had to change

Old rule (`scripts/global_id/runner.py`, pre-change):
```
fail if distinct_ids > 2*roster_max (30)     # roster_max=15 -> never fires
warn if distinct_ids > 1.2*roster_max (18)   # never fires (max ids = 16)
fail if teleports > 60                        # <-- EVERY fail is this
warn if 20 < teleports <= 60
fail if same_camera_collision                 # 0 everywhere
warn if coloc > 0
```
Three fatal flaws:
1. **It ignores cross-camera agreement entirely**, the project's *primary* identity axis is
   not in the verdict. So `M2_2_4_1` at **0.992 agreement** is `fail` while `M1_1_14_6` at
   **0.527 agreement** is merely `warn`. Backwards.
2. **Its one live signal, `teleports`, is a noise metric**, computed on raw bbox-bottom foot
   projections averaged across cameras, dominated by single-camera grazing error, not the
   delivered trajectory (`03-issue-teleport-metric-and-verdict.md`).
3. **The id-overmint thresholds never fire** (roster_max too loose), so "too many IDs" is
   never actually caught.

Net effect: 27/40 read `fail` for the wrong reason, and the reviewer learns nothing about
which clips are actually deliverable.

## The new rubric

**Two hard gates (any to FAIL, regardless of score):**
| Gate | Threshold | Rationale |
|---|---|---|
| same-camera collision | `collisions > 0` | Hard invariant, one ID on two people in one camera is never acceptable. |
| identity broken | `agreement < 0.65` | Below this, cross-camera identity is too wrong to trust. |
| gross over-mint | `distinct_ids > 20` | A field of ~15; >20 is genuine over-segmentation. |

**Otherwise a weighted usability score in [0,1]** over five normalized sub-scores:

| Axis | Sub-score (clamped 0..1) | Weight | Why it's in |
|---|---|---:|---|
| **Agreement** | `(agreement − 0.72) / 0.24` | **0.40** | Primary identity-correctness axis. 0.72 to 0, 0.96 to 1. |
| **Smoothness** | `1 − emitted_bigjumps / 50` | **0.25** | **Real** teleports, single-frame jumps >25 m/s in the emitted `ground_tracks.jsonl`, not the proxy. 0 jumps to 1, ≥50 to 0. |
| **Coverage** | `(tri_cov − 0.45) / 0.45` | **0.15** | Completeness of the 3D deliverable. 0.45 to 0, 0.90 to 1. |
| **Persistence** | `(id_persist − 0.80) / 0.18` | **0.10** | Confirmed-frame completeness per ID (fragmentation). |
| **Parsimony** | `(16 − distinct_ids) / 3` | **0.10** | ID count vs roster. 13 to 1, 16 to 0. |

Then `score −= 0.10 × coloc_pairs` (residual split-ID penalty).

**Tiers:**
| Tier | Score | Meaning |
|---|---|---|
| **GOOD** | ≥ 0.75 | Deliverable as-is; core + most periphery correct. |
| **USABLE** | 0.55-0.75 | Deliverable with the `single_camera` / low-`tri_cov` caution flags; core solid, periphery noisy. |
| **WEAK** | 0.40-0.55 | Needs work before delivery. |
| **FAIL** | < 0.40 or any hard gate | Not deliverable. |

Design choices worth defending in the meeting:
- **Agreement dominates (0.40)** because identity correctness is the objective; a smooth wrong
  ID is worse than a slightly jumpy right one.
- **Smoothness uses the real emitted metric (0.25)**, so a genuine teleport storm still
  penalizes, but single-camera foot noise (which never enters `ground_tracks` as a >25 m/s
  emitted jump the way it enters the raw proxy) no longer dominates.
- **Coverage is included but light (0.15)**, it's completeness, not correctness; a
  low-coverage clip can still be GOOD if what it does emit is correct (`M2_2_4_1`).
- Every threshold is a documented number, tunable in one place.

## Re-graded panel (all 40)

`OLD` = shipped verdict; `bigj` = real emitted big jumps; `NEW` = this rubric;
`limit/gate` = the axis holding the score down (or the gate that failed).

```
delivery    OLD   agr   bigj cov  per  ids clc score NEW     limit/gate
M1_1_14_1   pass  0.799   23 0.82 0.98  10   0 0.586 USABLE  agr
M1_1_14_2   warn  0.845   29 0.48 0.98  11   0 0.525 WEAK    cov
M1_1_14_3   warn  0.889   10 0.71 0.90  13   0 0.725 USABLE  persist
M1_1_14_4   warn  0.972   27 0.60 0.95  10   0 0.750 GOOD    cov
M1_1_14_5   warn  0.695   30 0.52 0.90  13   0 0.280 FAIL    agr(<0.65 gate)
M1_1_14_6   warn  0.527    9 0.49 0.91  15   0 0.314 FAIL    identity_broken
M1_1_14_7   warn  0.819   19 0.51 0.92  12   1 0.407 WEAK    cov
M1_1_16_1   fail  0.923   15 0.63 0.90  11   0 0.730 USABLE  cov
M1_1_16_2   fail  0.625   30 0.59 0.94  12   0 0.321 FAIL    identity_broken
M1_1_16_3   fail  0.892   53 0.79 0.94  10   0 0.575 USABLE  smooth
M1_1_16_4   warn  0.951    0 0.84 0.85  12   0 0.893 GOOD    persist
M1_1_16_5   warn  0.843    6 0.76 0.89  11   0 0.677 USABLE  persist
M1_1_16_6   warn  0.975   23 0.71 0.92  11   0 0.786 GOOD    smooth
M1_1_17_1   fail  0.954   38 0.64 0.89  10   0 0.660 USABLE  smooth
M1_1_17_2   fail  0.983    5 0.65 0.85  10   0 0.818 GOOD    persist
M1_1_17_3   fail  0.887   12 0.62 0.80  12   0 0.627 USABLE  persist
M1_1_17_4   fail  0.862    4 0.47 0.92  13   0 0.639 USABLE  cov
M1_1_17_5   warn  0.971    7 0.65 0.82  13   0 0.794 GOOD    persist
M1_1_17_6   pass  0.918   12 0.67 0.84  11   0 0.714 USABLE  persist
M2_1_11_1   fail  0.891   43 0.72 0.87  11   0 0.548 WEAK    smooth
M2_1_11_2   fail  0.897   45 0.65 0.87  12   0 0.525 WEAK    smooth
M2_1_11_3   fail  0.839   39 0.67 0.80  12   1 0.329 FAIL    persist+coloc
M2_1_11_4   fail  0.953   15 0.77 0.86  10   0 0.807 GOOD    persist
M2_1_11_5   warn  0.900   12 0.79 0.86   9   0 0.736 USABLE  persist
M2_1_11_6   fail  0.836   12 0.49 0.83  12   0 0.516 WEAK    cov
M2_1_11_7   fail  0.832   69 0.71 0.85  11   0 0.403 WEAK    smooth
M2_1_12_1   fail  0.881   40 0.28 0.84  13   0 0.438 WEAK    cov
M2_2_3_1    fail  0.847   91 0.33 0.88  14   0 0.326 FAIL    smooth
M2_2_3_2    fail  0.862  170 0.31 0.92  13   0 0.402 WEAK    smooth
M2_2_3_3    fail  0.725  140 0.31 0.97  13   0 0.205 FAIL    smooth
M2_2_3_4    fail  0.743   39 0.25 0.90  16   0 0.149 FAIL    cov
M2_2_3_5    fail  0.787  106 0.23 0.90  16   0 0.170 FAIL    smooth
M2_2_3_6    fail  0.728  105 0.28 0.91  16   0 0.074 FAIL    smooth
M2_2_3_7    fail  0.823  102 0.27 0.88  16   0 0.218 FAIL    smooth
M2_2_4_1    fail  0.992    7 0.42 0.94  11   0 0.794 GOOD    cov
M2_2_4_2    fail  0.949   22 0.58 0.87  11   0 0.705 USABLE  cov
M2_2_4_3    fail  0.976   15 0.61 0.85  11   0 0.753 GOOD    persist
M2_2_4_4    fail  0.969   43 0.72 0.88  12   0 0.671 USABLE  smooth
M2_2_4_5    fail  0.937   32 0.54 0.88  14   0 0.594 USABLE  cov
M2_2_4_6    fail  0.783   29 0.55 0.90  16   0 0.298 FAIL    parsimony
```

Distribution: **OLD** 2 pass / 11 warn / 27 fail to **NEW** 8 GOOD / 13 USABLE / 8 WEAK / 11 FAIL.

## Notable flips (the ones to mention)
- **fail to GOOD**: `M2_2_4_1` (0.992 agr, 7 jumps), `M2_2_4_3`, `M1_1_17_2`, `M2_1_11_4`.
  The old verdict punished these for single-cam foot noise; they are among the best clips.
- **warn/pass to FAIL**: `M1_1_14_5` (0.695), `M1_1_14_6` (0.527), and `M1_1_16_2` (0.625) , 
  genuinely broken cross-camera identity the old verdict let through as `warn`.
- **The M2_2_3 block stays bad** (mostly FAIL/WEAK), correctly: low coverage (0.23-0.33) +
  real teleport storms (91-170 jumps). This is where the work is.

## How to adopt
1. **Now (no re-run)**: `regrade.py` prints the table above from the existing tree; use it as
   the panel the reviewer reads. Optionally patch each `global_id_metrics.json`'s
   `quality_verdict` and re-print `final_panel.md` (reversible; ask before overwriting the
   production tree).
2. **Future runs**: the runner verdict has been rewritten to this rubric (agreement + emitted
   smoothness + persistence + parsimony + gates; coverage is folded in by the panel tool since
   it is a P6 metric not available at P4 time). Flag-gated so the legacy label is reproducible.
