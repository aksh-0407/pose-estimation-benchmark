# Per-delivery categorization, where the pipeline works, partially works, or fails

Grades combine the measured axes (agreement, emitted big jumps, P6 coverage, 2D flicker,
distinct ids) rather than the panel verdict, which is teleport-metric-dominated. See
`00-executive-summary.md` for the two failure modes (A = split identity/low agreement,
B = ground teleport/non-smooth).

## Grade key
- **A (clean / near-perfect)**: agreement ≥ 0.9, emitted big jumps ≤ ~15, coverage ≥ 0.8,
  flicker ≤ ~10. Deliverable as-is; core + most periphery correct.
- **B (core-good, periphery-noisy)**: core roles (batsman/bowler/keeper) solid; deep
  fielders/umpires split or teleport. Usable with the caveat flags.
- **C (weak)**: pervasive split identity and/or teleport storms; low coverage. Needs work
  before delivery.

## First innings, M1

| delivery | agr | bigjmp | cov | flick | ids | mode | grade | note |
|---|---:|---:|---:|---:|---:|---|:--:|---|
| M1_1_14_1 | 0.799 | 23 | 0.91 | 2 | 10 |, | B+ | clean core; a few foot spikes |
| M1_1_14_2 | 0.845 | 29 | 0.76 | 1 | 11 | B | B | spikes; low coverage |
| M1_1_14_3 | 0.889 | 10 | 0.78 | 11 | 13 |, | B | flicker on periphery |
| M1_1_14_4 | 0.972 | 27 | 0.84 | 11 | 10 | B | B | high agreement, foot spikes |
| M1_1_14_5 | 0.695 | 30 | 0.74 | 5 | 13 | A | C | split identity |
| M1_1_14_6 | 0.527 | 9 | 0.69 | 18 | 15 | A | C | **worst agreement**, pack, deep catchers |
| M1_1_14_7 | 0.819 | 19 | 0.75 | 8 | 12 | A | B− | facing-pair split; 1 residual coloc |
| M1_1_16_1 | 0.923 | 15 | 0.85 | 17 | 11 | B | B | flicker |
| M1_1_16_2 | 0.625 | 30 | 0.79 | 21 | 12 | A | C | **cam_04 split** (see 05) |
| M1_1_16_3 | 0.892 | 53 | 0.85 | 8 | 10 | B | B− | many foot spikes |
| **M1_1_16_4** | 0.951 | **0** | 0.91 | 3 | 12 |, | **A** | **cleanest delivery, 0 emitted big jumps** |
| M1_1_16_5 | 0.843 | 6 | 0.88 | 10 | 11 |, | B+ | good |
| M1_1_16_6 | 0.975 | 23 | 0.85 | 13 | 11 | B | B | high agreement, spikes |
| M1_1_17_1 | 0.954 | 38 | 0.86 | 7 | 10 | B | B− | teleport spikes (e_max 1511) |
| M1_1_17_2 | 0.983 | 5 | 0.87 | 5 | 10 |, | A− | high agreement, few jumps |
| M1_1_17_3 | 0.887 | 12 | 0.84 | 11 | 12 | B | B | proxyT 208 mostly single-cam noise |
| M1_1_17_4 | 0.862 | 4 | 0.75 | 31 | 13 |, | B− | **highest flicker (31)** |
| M1_1_17_5 | 0.971 | 7 | 0.79 | 10 | 13 |, | B+ | good |
| M1_1_17_6 | 0.918 | 12 | 0.81 | 10 | 11 |, | B+ | panel `pass` |

## Second innings, M2

| delivery | agr | bigjmp | cov | flick | ids | mode | grade | note |
|---|---:|---:|---:|---:|---:|---|:--:|---|
| M2_1_11_1 | 0.891 | 43 | 0.87 | 24 | 11 | B | B− | teleport + flicker |
| M2_1_11_2 | 0.897 | 45 | 0.84 | 23 | 12 | B | B− | teleport (e_max 1379) |
| M2_1_11_3 | 0.839 | 39 | 0.85 | 11 | 12 | A/B | B− | 1 residual coloc |
| M2_1_11_4 | 0.953 | 15 | 0.91 | 11 | 10 | B | B | good coverage |
| M2_1_11_5 | 0.900 | 12 | 0.92 | 5 | 9 |, | B+ | good |
| M2_1_11_6 | 0.836 | 12 | 0.74 | 12 | 12 | A | B− | some split |
| M2_1_11_7 | 0.832 | 69 | 0.87 | 18 | 11 | B | C | **teleport storm** (e_p99 82) |
| M2_1_12_1 | 0.881 | 40 | 0.59 | 28 | 13 | A/B | C | low coverage, high flicker |
| M2_2_3_1 | 0.847 | 91 | 0.58 | 10 | 14 | A+B | C | deep field, single-cam 0.77 |
| M2_2_3_2 | 0.862 | 170 | 0.62 | 13 | 13 | A+B | C | **most big jumps (170)** |
| M2_2_3_3 | 0.725 | 140 | 0.53 | 5 | 13 | A+B | C | low agreement + teleport |
| M2_2_3_4 | 0.743 | 39 | 0.48 | 11 | 16 | A+B | C | **lowest coverage (0.48)** |
| M2_2_3_5 | 0.787 | 106 | 0.49 | 9 | 16 | A+B | C | cam_07 split (see 05) |
| M2_2_3_6 | 0.728 | 105 | 0.54 | 26 | 16 | A+B | C | worst compound |
| M2_2_3_7 | 0.823 | 102 | 0.51 | 22 | 16 | A+B | C | proxyT 301, flicker 22 |
| M2_2_4_1 | 0.992 | 7 | 0.73 | 9 | 11 | B | B | **highest agreement (0.992)** but proxyT 158 |
| M2_2_4_2 | 0.949 | 22 | 0.81 | 3 | 11 | B | B | agreement good, foot spikes |
| M2_2_4_3 | 0.976 | 15 | 0.83 | 19 | 11 | B | B | flicker |
| M2_2_4_4 | 0.969 | 43 | 0.88 | 15 | 12 | B | B− | teleport spikes |
| M2_2_4_5 | 0.937 | 32 | 0.79 | 26 | 14 | B | B− | high flicker |
| M2_2_4_6 | 0.783 | 29 | 0.77 | 15 | 16 | A/B | C | split + teleport |

## What the grade distribution says

- **A / A−**: 3 (`M1_1_16_4`, `M1_1_17_2`, borderline `M1_1_14_1`). Truly clean is rare.
- **B band**: ~22. Core identities good; the residual problems are **peripheral fielders and
  umpires** (single-camera, deep, small), foot-projection teleport spikes and occasional
  splits. These are deliverable if consumers respect the `single_camera` / low-`tri_cov`
  caution flags.
- **C band**: ~15, concentrated in **M2 second-innings deep-field overs** (`M2_2_3_*` all
  seven, `M2_1_11_7`, `M2_1_12_1`, `M2_2_4_6`, `M1_1_14_5/6`, `M1_1_16_2`). These share high
  single-camera fraction and/or dense packs.

## The clean vs broken contrast (why M1_1_16_4 works and M2_2_3_4 does not)

| | M1_1_16_4 (A) | M2_2_3_4 (C) |
|---|---|---|
| single-camera fraction | 0.260 | 0.790 |
| P6 coverage | 0.91 | 0.48 |
| emitted big jumps | 0 | 39 |
| agreement | 0.951 | 0.743 |
| distinct ids | 12 | 16 |

The delta is not the algorithm, it is **how many players each camera set actually
co-observes.** Raising multi-camera coverage on the deep field (detection + F16 single-view
lift) would move C-band clips toward B/A more than any identity-cue tuning.
