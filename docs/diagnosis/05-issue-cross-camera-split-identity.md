# Issue 05, cross-camera split identity (same person, different ID per camera)

**Severity: high (this is the user's explicit report and the biggest identity-correctness
gap).** Directly reported: "the same person visible in four cameras seems to have the same ID
in three cameras but a different ID in one camera."

## Symptom (measured)
`split_identity.py`: per frame, cluster detections by ground proximity (< 1.5 m) across
cameras; count multi-camera clusters carrying > 1 global id.

```
delivery    split_clusters / multicam   worst camera pairs
M1_1_14_1   443 / 2376   (19%)          01-04, 04-05, 03-04
M2_2_3_3   1299 / 2343   (55%)          05-07, 01-04, 03-04
M2_1_11_6  1355 / 2850   (48%)          04-05, 01-04, 03-04
M2_2_3_5   1165 / 2342   (50%)          05-07, 04-06, 01-04
M1_1_16_2  1966 / 2338   (84%)          01-04, 02-04, 04-06, 03-04
M1_1_14_6  1984 / 2273   (87%)          01-04, 02-04, 04-05, 03-04
```
The rate is inflated in packs (two real people inside 1.5 m are *supposed* to differ), so
read the **camera-pair breakdown and the recorded examples**, not the raw percentage.

## The split is real, not a projection artifact
The pipeline's own `cross_camera_disagreement_examples` (in each `global_id_metrics.json`)
show whole-tracklet splits where one camera is the odd one out:

```
M1_1_16_2:  cam_04 -> P011/P013   while  cam_01, cam_02, cam_06 -> P005   (same player)
M2_2_3_5:   cam_07 -> P016        while  cam_05 -> P009                   (same player)
M1_1_14_6:  P005 vs P006/P007 scattered across cam_01/02/04/06           (pack)
```
`M1_1_16_2` is the textbook case the user described: three cameras agree (P005), cam_04
dissents (P011/P013). It is not two nearby people, it is one player whose cam_04 tracklet
never associated into his cluster.

## Which cameras split, and why
Aggregated over the sample (`01-methodology` Table 3), the offending camera is almost always
one of two:

- **cam_04 (end-on, down the pitch)**, dominates every top pair (01-04, 02-04, 03-04,
  04-05, 04-06). Two compounding reasons:
  - **Facing-pair epipolar degeneracy.** cam_01-cam_04 is a *facing pair* (co-observing, near
    head-on). The epipolar constraint is near-degenerate, so the geometry cue that normally
    binds two cameras' detections is weak exactly here. This is the documented "facing-pair
    ceiling" (`docs/critical-analysis/status-report.md` §3.2). cam_01-cam_04 is the single
    biggest disagreeing pair across all 40 (5030 events).
  - **Grazing ground projection.** End-on, a player's bbox-bottom projects to the ground with
    ~1 m depth error, so the ground-distance cue that P3 fuses is noisy for cam_04 to its
    tracklet's ground position does not land close enough to the cluster to bind.
- **cam_07 (panoramic 3775×960, oblique)**, players are tiny (10-20 px), pose confidence is
  low, appearance is uninformative, and it sees a different slice of the field. Its tracklets
  frequently fail to associate (`M2_2_3_5`: cam_07 to P016 vs cam_05 to P009). cam_05-cam_07 is
  a top pair on the M2_2_3 clips.

## Why the existing cues do not catch it (from the campaign record, re-confirmed)
P3 fuses per-cue log-likelihood ratios; on this footage:
- **kit colour is dead** (d′ ≈ 0.09, identical kit, desaturated broadcast footage); the cue
  auto-abstains. The panel `d_app` column is 0.0-2.7 and often 0.
- **bone-ratio shape abstains** (d′ < 0.5, body proportions don't separate players).
- **billboard stature/posture** is the only facing-pair-capable shape channel, and it is weak
  when one view is grazing (cam_04) or tiny (cam_07).
- that leaves **ground geometry**, which is precisely the cue that is degenerate on the facing
  pairs and noisy on cam_04/cam_07.
So the hard cameras have *no* strong binding cue, and P3 leaves them as separate ids.

## W9's partial remedy and its limit
The W9 union-lift merge (P3) and colocated-id merge (P4b) were built for exactly this
("ghost-under-player split"). They fire when one triangulated skeleton explains all member
views, or two ids stay < 0.75 m apart in disjoint cameras for ≥ 25 frames. They resolved the
`coloc` metric to 0 on 38/40. **But they only merge when the two ids are already co-located on
the ground.** A cam_04 split whose grazing ground position is > 0.75 m off never satisfies the
merge gate, so it survives as a distinct id, and inflates the id count *and* the split rate.
Two production deliveries still carry a residual coloc pair (`M1_1_14_7`, `M2_1_11_3`).

## Consequence chain
split identity to extra distinct ids (over-mint) to the id also appears as the "other" player's
ghost in the render to and if a later merge *does* fire late, the seam becomes an emitted
teleport (`04-...`). So this issue feeds both the "many IDs" and the "teleport" complaints.

## Fix direction (see ../../wip/open-work.md items A4, A5)
- **Geometry-aware association weighting**: down-weight the ground-distance cue for grazing
  cameras (cam_04 end-on, cam_07) by their calibrated depth-uncertainty, and *up-weight* the
  cross-view triangulation-consistency test (union-lift) which is the facing-pair-capable cue.
- **Relax the merge gate to a depth-aware radius** (per the projecting camera's uncertainty)
  instead of a flat 0.75 m, so a cam_04 split whose foot is 1 m off can still merge.
- **F16 single-view PnP lift** raises coverage so cam_04/cam_07 lone tracklets get a
  triangulation-consistency check instead of only a ground-distance check.
- Consider the **skeleton-gait embedding** cue (deferred, needs sign-off) as the one remaining
  facing-pair-capable appearance channel.
