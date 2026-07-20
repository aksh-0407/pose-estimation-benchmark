# Issue 07, visible 2D ID-switch flicker (mosaic colour changes)

**Severity: medium (visually salient in the render; identity mostly recovers).**

## Symptom
In the mosaic, a player's skeleton overlay colour (= `global_player_id`) flickers, changes
for a stretch of frames then reverts. The user reports "a lot of ID switching."

## Measured (from `idswitch_2d.py`)
Within one `(camera, local_track_id)`, a P2 tracklet, i.e. one physical person tracked
continuously in one camera, we count frame-to-frame `global_player_id` changes:
- **517 switch events total across the 40** (~13/delivery).
- **~5.5 tracklets/delivery carry more than one global id** over their life.
- Worst: `M1_1_17_4` (31), `M2_1_11_1` (24), `M2_2_3_7` (22), `M2_2_4_5` (26). Best:
  `M1_1_14_1` (2), `M1_1_16_4` (3), `M2_2_4_2` (3).

Context: ~25-38 tracklets per delivery, so ~15-35 % of tracklets flicker at least once. It is
real but not pervasive, most tracklets hold one id the whole time.

## Root cause
A P2 tracklet is stable (it *is* one person in one camera), so the flicker is entirely a
**P3/P4 cross-camera assignment changing under it frame to frame**:
1. **Membership churn in P3.** The tracklet joins cluster X ( to  global id A) on some frames and
   cluster Y ( to  id B) on others, because the per-frame association is re-decided and the
   marginal cue (ground distance on a grazing camera, weak posture) tips differently. The
   panel `churn` column (0.000-0.004) and `cycle_cons` (0.47-0.95) track this; the low
   `cycle_cons` clips (`M2_1_12_1` 0.466, `M2_2_3_1` 0.732) flicker more.
2. **P4a re-assignment.** When a track is briefly lost and re-acquired, the online Hungarian
   can hand the tracklet to a different global track before P4b stitches it back, so the
   colour flips mid-clip and the stitch only fixes the *id label*, not necessarily every frame
   in between.
3. **Late merges.** A colocated/stitch merge relabels one side's history to the winner id;
   frames near the boundary can show the pre-merge id if the remap is applied per-segment.

So flicker and split identity (05) share a root: weak cross-camera binding on the hard
cameras, decided per-frame rather than locked per-tracklet.

## Why it is not worse
P2 tracklets are stable and P3 uses a whole-delivery tracklet graph (not purely per-frame),
which is why most tracklets keep one id. The flicker is concentrated on the
marginal/peripheral tracklets that the graph binds weakly.

## Fix direction (see ../roadmap.md item A7)
- **Lock global id per P2 tracklet, not per frame.** Once a `(camera, local_track_id)` is
  bound to a global id with confidence, hold it for the tracklet's life unless strong evidence
  overrides, a tracklet is one person, so its global id should be piecewise-constant by
  construction. This directly removes the intra-tracklet flicker.
- Apply id remaps as a **final relabel over whole tracklets** so no boundary frames keep the
  losing id.
- Raise `cycle_cons` by penalising inconsistent triangles in the tracklet graph (A=B, B=C,
  A≠C) at solve time.
