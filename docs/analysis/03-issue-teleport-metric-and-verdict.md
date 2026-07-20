# Issue 03, the teleport metric and the panel verdict mislead

**Severity: high (it hides the real problem and cries wolf on the multi-cam clips).**

## Symptom
27/40 deliveries read `fail`. A reviewer glancing at the panel concludes the pipeline is
broken almost everywhere. That is not what the output shows.

## What the verdict actually tests
`scripts/global_id/runner.py:414-447`:
```
if distinct_ids > 2 * roster_max:      verdict = fail   # roster_max=15 -> >30, NEVER hit
elif distinct_ids > 1.2 * roster_max:  verdict = warn   # >18, NEVER hit (max ids = 16)
if teleports > 60:                     verdict = fail   # <-- every fail is THIS
elif teleports > 20:                   verdict = warn
if same_camera_collision:              verdict = fail   # 0 everywhere
if coloc > 0:                          verdict = warn   # 2 deliveries
```
So the verdict is **almost entirely a function of one number: `teleport_event_count > 60`.**
The id-overmint rules are dead (roster_max is too loose relative to the ~11 minted ids).

## What the teleport metric measures
`pose_estimation/cricket/tracking_metrics.py:293-347` (`teleport_proxy`) is fed
`detection_ground_positions` built at `runner.py:211-249`:
- one point **per detection**, = `calibrator.bbox_bottom_center_ground_xy(bbox)` (or an
  upper-body estimate for feet-unusable detections), i.e. the **raw bbox-bottom projected to
  z = 0 by calibration alone**;
- per `(id, frame)` it takes the **mean over all cameras** that saw the id;
- flags a "teleport" when that mean moves > 9 m/s × 1.5 slack = 13.5 m/s between consecutive
  frames.

Two structural problems:
1. **Grazing-angle foot noise.** A bbox bottom projected to the ground at a grazing camera
   (cam_04 end-on; cam_07 panoramic) has ~1 m depth error. For a single-camera id this is
   the *only* source, and 1 m in 20 ms = 50 m/s to an instant false teleport. This is why
   `M2_2_4_1` scores agreement **0.992** (identity essentially perfect) yet **158 teleports**
   to `fail`. The identity is right; the metric is measuring foot-projection noise.
2. **Camera-set-change discontinuity.** The per-frame position is a mean over contributing
   cameras. When the set of cameras seeing an id changes frame to frame (very common at the
   field edges), the mean jumps even if every underlying detection is correct.

Neither of these is the emitted trajectory. The metric never looks at
`ground_tracks.jsonl`.

## Evidence it is a proxy artifact (not the delivered output)
- `M2_2_4_*` group: agreement 0.94-0.99, teleports 121-236 to all `fail`. High agreement is
  incompatible with genuine identity teleporting; the teleports are single-cam foot noise.
- Correlation with `single_cam` in the panel is tight: `M2_2_3_*` (single_cam 0.76-0.82)
  and `M2_2_4_*` carry the highest proxy teleports.

## But the metric is ALSO hiding a real problem
Because everyone learned "teleports are a known proxy artifact", the genuinely-teleporting
**emitted** trajectory (1528 non-physical single-frame jumps, `04-...`) was written off as
the same artifact. It is not, the proxy runs on raw detections, the real one is in the
emitted Kalman track. **The metric being noisy does not mean the output is clean.** Both are
true and they are different measurements.

## Fix direction (see ../roadmap.md item A1)
Replace / supplement the proxy with a metric on the **emitted** `ground_tracks.jsonl`,
velocity-gated, and computed **only on multi-camera segments** (skip frames where the id is
single-camera, since those carry ~1 m irreducible foot uncertainty). Keep the raw proxy as a
secondary tripwire but stop letting it drive the verdict. Re-tune the id-overmint thresholds
to the real roster (~13) so that rule can actually fire.
