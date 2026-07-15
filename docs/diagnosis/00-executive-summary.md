# 40-delivery production diagnosis — executive summary

Written 2026-07-14. Scope: a full, from-scratch diagnosis of the v8.1 production tree
(`/home/ubuntu/pipetrack_v8/`, 40 deliveries × 7 cameras × 600 frames) on the L40S,
answering "why is the output not perfect": many IDs, ID switching, non-smooth output,
teleports, and cross-camera identity splits. All numbers here were **measured on the
actual production output**, not reasoned from the panel. Measurement scripts are checked
in beside this file (`emit_smoothness.py`, `jump_classify.py`, `occupancy_and_3d.py`,
`idswitch_2d.py`, `split_identity.py`); rerun them on the box with the
`cricket-rtmpose-l` env.

## Headline: what is actually true

1. **The panel verdict is not measuring what it claims.** 27/40 deliveries read `fail`,
   2 `pass`, 11 `warn`. Every single `fail` is the `teleport_storm` rule (teleports > 60);
   the `id_overmint` rule never fires (roster_max = 15; the worst delivery mints 16 ids).
   The teleport metric runs on **raw per-detection bbox-bottom ground projections averaged
   across cameras** — not on the emitted trajectory — so it is dominated by single-camera
   grazing-angle foot noise. See `03-issue-teleport-metric-and-verdict.md`.

2. **But the emitted output genuinely does teleport.** The delivered fused ground track
   (`p4/diagnostics/ground_tracks.jsonl`, the Kalman posterior consumers read) contains
   **1528 single-frame jumps faster than 25 m/s** across the 40 deliveries, with peak
   "speeds" of 140–1500 m/s (tens of metres in one 20 ms frame). So the user is right —
   this is not only a proxy artifact. Root cause is over-merge/mis-assignment amplified by
   a **mean-over-fragments emission** (`runner.py:339-348`). See
   `04-issue-emitted-ground-teleports.md`.

3. **The 3D skeleton output (P6) is smooth but sparse.** The triangulated skeletons
   (`p6_3d`, the actual 3D deliverable) are clean — pelvis p95 speed 1.6–3.8 m/s, ~0 big
   jumps — because 3D is only emitted where ≥2 cameras triangulate and RANSAC + Butterworth
   reject outliers. The price is **coverage 0.48–0.92 (median 0.80)**: single-camera frames
   get no skeleton at all. See `08-issue-3d-coverage-gaps.md`.

4. **Cross-camera split identity is real and systematic** (the user's specific report: one
   camera disagrees with the other three). 19–87 % of multi-camera ground clusters carry
   more than one global id. The odd-camera-out is consistently the geometrically hard camera:
   **cam_04 (end-on, grazing)** and **cam_07 (panoramic)**. The single biggest disagreeing
   pair across all 40 is the facing pair **cam_01↔cam_04**. Concrete: on `M1_1_16_2` cam_04
   labels a player P011/P013 while cam_01/02/06 all call him P005. See
   `05-issue-cross-camera-split-identity.md`.

5. **Visible ID-switch flicker exists but is modest.** Within a stable per-camera P2
   tracklet (one physical person in one camera), the global id flips **517 times total
   across the 40** (~13/delivery; ~5.5 tracklets/delivery carry >1 id). This is the mosaic
   colour flicker. See `07-issue-2d-id-switch-flicker.md`.

6. **ID count is NOT the primary problem.** Distinct ids are 9–16 against a roster max of 15.
   The pipeline *over-mints internally* (P4a spawns e.g. P001–P024 for ~13 people) and then
   stitches down — and **each stitch/merge seam is a teleport risk**. So "many IDs" is
   mostly resolved at the metric level but leaves teleport scars. See
   `06-issue-id-overmint-and-stitch-seams.md`.

## The single most important structural fact

Everything scales with the **single-camera fraction** of a delivery. When a player is seen
by only one camera he cannot be triangulated (no 3D, coverage drops), his ground position
is a grazing foot projection (noisy → teleport spikes), and he cannot be cross-checked for
identity (→ split ids and over-mint). The second-innings deep-field clips **M2_2_3_\*** run
single-cam 0.76–0.82 and are the worst on every axis (coverage 0.48–0.62, teleports
180–301, 15–16 ids). The clean first-over clips **M1_1_14_\*** run single-cam 0.39–0.65 and
behave. This is a **detection/geometry coverage problem first, an identity-algorithm problem
second.**

## Two distinct failure modes (they need different fixes)

- **Type A — split identity / low agreement** (`M1_1_14_6` 0.527, `M1_1_16_2` 0.625,
  `M1_1_14_5` 0.695, `M2_2_3_3/4/6`): cross-camera association fails on facing pairs and in
  packs. One camera's tracklet gets its own id. Fix lives in **P3** (association cues +
  facing-pair geometry).
- **Type B — ground teleport / non-smooth** (`M2_2_4_\*` agreement 0.94–0.99 but teleports
  121–236; `M2_1_11_\*`; `M1_1_17_\*`): identity is largely correct but the emitted ground
  position jumps from single-camera foot noise and stitch seams. Fix lives in **P4 emission
  + smoothing** (velocity-gated emission, drop mean-over-fragments, single-cam damping).

`M2_2_3_\*` suffers both at once.

## Where it works, partially works, fails

Full table: `02-per-delivery-categorization.md`. Summary:
- **Cleanest (closest to "perfect")**: `M1_1_16_4` (0 emitted big jumps, agreement 0.951,
  coverage 0.91, 3 flickers), `M1_1_14_1`, `M1_1_17_6`.
- **Core-good, periphery-noisy (usable with caveats)**: most of `M1_1_14`, `M1_1_16`,
  `M2_2_4`. Batsman/bowler/keeper solid; deep fielders and umpires split/teleport.
- **Weak (needs work before delivery)**: `M2_2_3_\*`, `M2_1_11_\*`, `M2_1_12_1`,
  `M1_1_14_6`. High single-cam, dense field, deep fielders.

## What to change (pointer)

Prioritized, evidence-backed change list: `../../wip/to_do.md`. Top three:
1. Fix the emitted ground teleports (P4 emission: per-id velocity gate + drop the
   mean-over-fragments; single-cam foot damping). Biggest visible-quality win.
2. Fix the verdict/teleport metric so it measures the emitted trajectory on multi-cam
   segments only — the current metric mislabels 27 deliveries and hides the real problem.
3. Attack split identity at the hard cameras (cam_04/cam_07): geometry-aware association,
   grazing-angle down-weighting done right, F16 single-view PnP lift to raise coverage.
