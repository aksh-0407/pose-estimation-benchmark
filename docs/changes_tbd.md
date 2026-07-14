# Changes to be done — prioritized, evidence-backed

Derived from the 40-delivery production diagnosis (`docs/diagnosis/`, measured 2026-07-14).
Ordered by (visible-quality impact ÷ effort). Every item cites the issue doc, the code
location, and how to verify. Follow the project standard: **flag-gated, flags-off
byte-identity proven by execution, A/B on all 8 benchmark deliveries (or 40 for a production
claim), accept only significant generalized improvement.**

Legend: 🔴 high impact · 🟡 medium · 🟢 low/cleanup · (S)mall / (M)edium / (L)arge effort.

---

## C1 🔴 (S) Fix the teleport metric + verdict — stop mislabeling 27 deliveries
**Why**: every `fail` is `teleport>60` on a metric computed over **raw bbox-bottom foot
projections averaged across cameras** — dominated by single-cam grazing noise; `M2_2_4_1` is
0.992 agreement yet `fail`. (`diagnosis/03`)
**Do**:
- Add a second teleport metric on the **emitted** `ground_tracks.jsonl`, velocity-gated,
  computed **only on multi-camera segments** (skip frames where the id is single-camera).
- Demote the raw proxy to a secondary tripwire; drive the verdict off the emitted metric +
  agreement + coloc.
- Re-tune id-overmint thresholds to the real roster (~13) so that rule can fire.
**Code**: `src/identity/common/metrics.py:293` (`teleport_proxy`);
`src/identity/p5_global_id/runner.py:414-447` (verdict).
**Verify**: re-run `--panel-only`; verdict distribution should track the `docs/diagnosis/02`
grades, not the single-cam fraction.

## C2 🔴 (S) Stop emitting `np.mean` of multi-modal fragment positions
**Why**: the emitted oscillating teleports (1528 big jumps) come from averaging two far-apart
observations that share one final id in a frame. (`diagnosis/04`)
**Do**: at `runner.py:339-349`, when an `(id, frame)` has ≥2 points spread > ~2 m, do NOT
average — emit the Kalman posterior only (or the point nearest the prior) and log a
`split_observation` diagnostic.
**Code**: `src/identity/p5_global_id/runner.py:339-349`.
**Verify**: re-run `emit_smoothness.py`; emitted big-jump total should collapse from 1528
toward the `M1_1_16_4`-class baseline.

## C3 🔴 (M) Per-id velocity gate on the emitted ground track + single-cam damping
**Why**: even with C2, single-cam foot spikes and posterior lurches remain (`diagnosis/04`
spikes ≈ half the events).
**Do**:
- Reject/hold any emitted point implying > ~10–12 m/s from the previous emitted point on a
  **multi-camera** segment.
- On **single-camera** segments, damp toward the Kalman posterior instead of snapping to the
  noisy foot ray (the foot ray carries ~1 m grazing error).
- Tighten the P4a χ² gate / process noise so a single far measurement after a short gap is
  not admitted wholesale.
**Code**: `src/identity/p5_global_id/runner.py:110-141` (emission), `p4_global_id.yaml` (gates),
`src/identity/p5_global_id/ground_kalman.py`.
**Verify**: `emit_smoothness.py` e_max should drop from 100–1500 m/s toward < ~30 m/s.

## C4 🔴 (M) Depth-aware association weighting for grazing/facing cameras
**Why**: split identity — cam_04 (end-on) and cam_07 (panoramic) fail to bind because the
only live cue (ground distance) is noisy exactly there. (`diagnosis/05`)
**Do**: weight the ground-distance LLR by each camera's calibrated depth-uncertainty
(down-weight grazing cam_04/cam_07), and **up-weight the triangulation-consistency
(union-lift) cue**, which is the facing-pair-capable channel.
**Code**: P3 cue fusion in `src/identity/p3_association/` (tracklet_graph + associator);
config `configs/03_association.yaml`.
**Verify**: `split_identity.py` cam_04/cam_07 split tallies down; panel `agreement` up on
`M1_1_14_6`, `M1_1_16_2`, `M2_2_3_*` without collisions.

## C5 🟡 (S) Depth-aware colocated-merge radius (replace flat 0.75 m)
**Why**: a cam_04 split whose foot is ~1 m off never satisfies the 0.75 m merge gate → split
survives; 2 residual coloc pairs. (`diagnosis/05`, `remaining-work.md` §5b)
**Do**: make `colocated_radius_m` a function of the projecting camera's depth-uncertainty
(larger for grazing views) with the same disjoint-camera + posture guards.
**Code**: `src/identity/p5_global_id/stitching.py:328` (`merge_colocated_ids`), `p4_global_id.yaml`.
**Verify**: `coloc` → 0 on all 40; no new same-camera collisions.

## C6 🟡 (S) Cap the cross-space stitch budget (absolute metres, not gap-scaled)
**Why**: `distance ≤ v_max·gap·slack` grows with the gap → a long occlusion licenses a
cross-field stitch → emitted step / possible wrong-person merge. (`diagnosis/06`)
**Do**: add an absolute metres ceiling on stitch seam distance independent of gap; prefer an
honest extra id over a teleport. Emit a per-merge `seam_distance`/`overlap` quality flag.
**Code**: `src/identity/p5_global_id/stitching.py:139-226` (`build_link_costs`).
**Verify**: `jump_classify.py` step count down; id count may tick up slightly (acceptable).

## C7 🟡 (M) Lock global id per P2 tracklet, not per frame
**Why**: 517 flicker events are intra-tracklet global-id flips — a tracklet is one person, so
its id should be piecewise-constant. (`diagnosis/07`)
**Do**: once a `(camera, local_track_id)` binds to a global id with confidence, hold it for
the tracklet's life unless strongly overridden; apply id remaps as a final whole-tracklet
relabel so boundary frames don't keep the losing id.
**Code**: P3 membership + `runner.py` relabel; `src/identity/p3_association/`.
**Verify**: `idswitch_2d.py` total < ~150 (from 517); no agreement regression.

## C8 🔴 (L) F16 single-view PnP lift — raise multi-camera coverage
**Why**: the single-camera fraction is the root driver (coverage 0.48, teleports, splits).
(`diagnosis/08`, `remaining-work.md` §3.2)
**Do**: fit the identity's canonical skeleton (bone lengths from its multi-view frames) to a
lone 2D view, PnP-style, with honest covariance. Turns single-cam frames into "3D with a
wider error bar" → coverage up, ground position physically constrained (damps teleports too).
**Code**: new module consuming P3.5 lift + P6; `wip` V2-L1/F16 notes.
**Verify**: P6 coverage up on `M2_2_3_*` (from ~0.5); emitted single-cam spikes down.

## C9 🟡 (M) Detection recall on the deep field / small subjects
**Why**: upstream cause of single-camera players (`diagnosis/09` P1-1/P1-2).
**Do**: probe 3×2-grid tiling and/or a stronger detector (YOLO26-l, RF-DETR) through the
existing bake-off harness; recall-oracle on `M2_2_3_*` + cam_07 first.
**Code**: `tools/detector_bakeoff/detector_bakeoff.py` + `detector_bakeoff_report.py`.
**Verify**: recall on deep fielders up; single_cam fraction down on M2 innings-2 clips.

## C10 🟢 (S) Coverage/confidence flag per emitted frame
**Why**: consumers currently see a hard "blink" when 3D drops on single-cam frames.
(`diagnosis/08`)
**Do**: emit a per-frame coverage/confidence field so downstream can interpolate or hide gaps
deliberately.
**Verify**: schema check; no numeric change to existing fields.

---

## Sequencing suggestion
1. **This week (cheap, high visible impact)**: C2 → C1 → C3 → C6 → C5. These kill the visible
   teleports and fix the misleading verdict with small, flag-gated edits.
2. **Next (identity correctness)**: C4 → C7 → C9.
3. **Larger lever**: C8 (F16) — the structural fix for coverage/single-cam.

Each lands with a `fixes-log.md` entry (mechanism, panel, verdict) the moment it's A/B'd.
