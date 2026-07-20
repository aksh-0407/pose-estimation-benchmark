# Issue 04, the emitted ground track genuinely teleports

**Severity: high (this is the real "not smooth / teleports" the user sees).**

## Symptom
`p4/diagnostics/ground_tracks.jsonl`, the fused world position per id per frame that
consumers and the BEV render read, contains **1528 single-frame jumps > 25 m/s** across the
40 deliveries, peaks of 140-1500 m/s (tens of metres in one 20 ms frame). Worst:
`M2_2_3_2` (170), `M2_2_3_3` (140), `M2_2_3_5/7` (~105), `M2_1_11_7` (69). Even the clean
`M1_1_14` group has 9-30 each. Only `M1_1_16_4` has **zero**.

This is despite `emit_kalman_posterior: true` (config `p4_global_id.yaml:12`), which is
supposed to emit the chi²-gated Kalman posterior, "cannot jump faster than the process
model" (config comment `config.py:134`). It jumps anyway.

## Jump shape (from `jump_classify.py`)
For every big jump we asked: **spike** (out-and-back, position leaps away for one frame then
returns near the origin, i.e. one outlier measurement) or **step** (persistent). Result on
the high-teleport clips:

```
M2_2_3_2  spikes=79  steps=91   (max step 28.8 m)   -- repeated ~14/28 m oscillation on P014
M2_2_3_7  spikes=29  steps=73   (max step 22.5 m)   -- P017 oscillates every few frames
M2_1_11_7 spikes=21  steps=48   (max step 20.4 m)   -- P010
M1_1_17_1 spikes=18  steps=20   (max step 30.2 m)   -- P005
M1_1_16_4 spikes=0   steps=0                         -- clean
```
The steps are **not one-time seams**, a single id (e.g. `M2_2_3_2` P014) flips ~14 m and
~28 m repeatedly over consecutive frames. That signature means one global id is receiving
observations from **two spatially separated sources at the same time**, and the emitted
value oscillates as the contributing set changes.

## Root cause (code)
Emission, `scripts/global_id/runner.py:339-349`:
```python
stitched_ground[(final_id, frame_index)].append(point)      # after id_remap
...
ground_rows_by_frame[frame_index].append(
    {"global_player_id": final_id,
     "ground_xy": np.mean(points, axis=0).tolist()})          # <-- MEAN over all points
```
`points` is every emitted ground observation whose id (after P4b stitch + colocated remap)
maps to `final_id` in that frame. Upstream (`runner.py:254-261`) `ground_positions` is
already a **mean over cameras** per `(P4a-id, frame)`. So there are two averaging layers, and
**averaging is exactly wrong for position**: if one id legitimately or wrongly has two
observations at A and B in a frame, the emitted point is the midpoint of A and B; in the next
frame only A contributes to the emitted point snaps from midpoint to A. Repeat to oscillation.

The two spatially-separated observations under one id arise from (in rough order of impact):
1. **P4a assigns one global track to two concurrent clusters** (disjoint camera sets) in the
   same frame, the ground observations for that `(id, frame)` then hold two far-apart points
   (`runner.py:255-257`). The occupancy invariant only forbids two ids **sharing a
   (camera,frame) cell**; it does **not** forbid one id spanning two clusters in *different*
   cameras that are actually two different people.
2. **Kalman fed alternating far measurements.** When the correspondence feeding a track
   alternates between two locations, the posterior itself lurches (the Singer process noise +
   loose chi² gate 5.991 admit the far measurement after a short gap), so even the
   posterior-emit path is not immune.
3. **Cross-space temporal stitch** (`stitching.py:139-226`): edges link `source.end <
   target.start` when `distance ≤ v_max·gap·slack`. With a 100-frame gap that budget is ~30 m
  , enough to stitch a fragment that ended at one end of the field to one that starts at the
   other. That produces a one-time **step** at the seam (a real ID error if the two fragments
   are different people, an acceptable occlusion-restore if they are the same person).

The **spikes** (~half of all events) are the single-frame foot-projection / bad-triangulation
outliers, same family as the proxy metric's noise, but here they survive into the emitted
posterior because the gate does not reject a single far measurement hard enough.

## Why P6 3D does NOT show this (important contrast)
`occupancy_and_3d.py`: the P6 triangulated pelvis has p95 1.6-3.8 m/s and **~0 big jumps**.
P6 only emits a joint where ≥2 cameras triangulate with RANSAC agreement and then applies a
zero-phase Butterworth. The single-camera and mis-associated observations that teleport the
P4 ground track are simply **dropped** by P6 (hence coverage 0.48-0.92, `08-...`). So the 3D
skeleton is the trustworthy positional output today; the P4 ground dot is not.

## Fix direction (see ../roadmap.md items A2, A3)
- **Do not emit `np.mean` of multi-modal points.** If an id has ≥2 observations > (say) 2 m
  apart in a frame, that is a bug flag, not a position, emit the Kalman posterior only, or
  the observation closest to the prior, and log the split.
- **Per-id velocity gate on emission**: reject/hold any emitted point implying > ~10-12 m/s
  from the previous emitted point on a *multi-camera* segment; on single-camera segments
  damp toward the posterior instead of snapping to the noisy foot ray.
- **Tighten the concurrent-cluster guard in P4a**: forbid one track from claiming two clusters
  whose ground positions differ by more than a body radius in the same frame.
- **Cap the cross-space stitch budget** independent of gap (an absolute metres ceiling), so a
  long occlusion cannot license a cross-field merge.
