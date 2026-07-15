# Issue 08 — 3D coverage gaps (the smooth output is sparse)

**Severity: medium (the 3D deliverable is trustworthy but incomplete).**

## Symptom
The P6 triangulated 3D skeletons are **smooth and clean** (pelvis p95 1.6–3.8 m/s, ~0 big
jumps — `occupancy_and_3d.py`), which is the good news. The cost is **coverage**: fraction of
identity-frames that actually carry a `pose_3d` is **0.48–0.92, median 0.80**
(`idswitch_2d.py`). On the worst clips nearly half of all player-frames have no 3D skeleton.

```
p6 coverage:  M2_2_3_4 0.48   M2_2_3_5 0.49   M2_2_3_7 0.51   M2_2_3_3 0.53
              M2_1_12_1 0.59   ...   M1_1_16_4 0.91   M2_1_11_5 0.92
```

## Root cause
P6 only triangulates a joint where **≥ 2 cameras** see it with enough confidence and pass the
cheirality-gated RANSAC. A **single-camera** player-frame therefore gets **no 3D at all**.
Coverage is thus a direct function of multi-camera visibility, i.e. `1 − single_cam`:
`M2_2_3_*` run single_cam 0.76–0.82 → coverage ~0.5; `M1_1_16_4` runs single_cam 0.26 →
coverage 0.91. This is the same single-camera-fraction driver behind every other issue.

The coverage gap is **by design and correct** — fabricating a 3D skeleton from one view would
be worse than emitting none. But it means downstream consumers see a player "blink out" of 3D
whenever he is only in one camera, which reads as missing/stuttering data even though it is
honest.

## The two are a trade
- P4 ground track: **dense** (every id, every frame) but **teleports** on single-cam ids.
- P6 3D skeleton: **smooth** but **sparse** (single-cam ids absent).
Neither currently gives a dense-and-smooth positional output. The lever that fixes both is
raising real multi-camera coverage.

## Fix direction (see ../../wip/to_do.md item A8)
- **F16 single-view PnP lift** (deferred experiment, `wip/to_do.md` A8): fit the
  identity's canonical skeleton (bone lengths learned from its multi-view frames) to the lone
  2D view with honest covariance. Turns single-camera frames from "no 3D" into "3D with a
  wider error bar" — raises coverage and gives the ground track a physically-constrained
  position instead of the noisy foot ray (also damps the `04-...` teleport spikes).
- **Better detection recall on the deep field** (tiled RTMDet already helps; a 3×2 tiling or a
  stronger detector probe on `M2_2_3_*`) converts single-camera players into multi-camera
  ones at the source.
- Emit a per-frame **coverage/confidence flag** so consumers can interpolate or hide gaps
  deliberately rather than seeing a hard blink.
