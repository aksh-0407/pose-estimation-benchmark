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
**Code**: `src/identity/p5_global_id/runner.py:110-141` (emission), `configs/05_global_id.yaml` (gates),
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
**Code**: new module consuming the 04 binding lift + terminal 3D lift; `wip` V2-L1/F16 notes.
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

---

## Restructure follow-ups & bugs found (2026-07 docs/scrub pass)

These are repo-hygiene items from the `src/{core,identity}` + `tools/` restructure — not
algorithm changes. Bugs marked **FIXED** were corrected in the scrub pass; the rest are notes.

- **FIXED — `tools/` kept old imports.** Root cause: the restructure's import-rewrite pass scoped
  only `src/` + `tests/`, so `tools/` modules still imported the removed `pose_estimation.*` /
  `scripts.*` names and would crash at runtime. Fixed:
  - `tools/diagnosis/eval_ground_accuracy.py` → `identity.common.triangulation`, `core.calibration`.
  - `tools/diagnosis/split_identity.py` → `core.calibration`, `identity.p2_tracking.runner`;
    replaced the hardcoded `/home/ubuntu/pose-estimation-benchmark` `sys.path` insert with a
    repo-relative `src/` insert.
  - `src/identity/visualization/render_all_mosaics.py` → `RENDERER` now points at
    `src/identity/visualization/render_videos.py` (was the removed `scripts/.../render_phase1_videos.py`).
  - `tools/run_v8_l40s.sh` → `python -m main`, `pose-lab` env, `configs/0N_*.yaml`; also fixed a
    `REPO=$(… /../..)` depth bug (script moved one level shallower into `tools/`).
- **NOTE — `--show p4` render flag.** `src/main.py`'s render call passes `--show p4` as a *semantic*
  stage selector consumed by `render_videos.py` (P2/P3/P4 id vocabulary), not a run-dir name — left
  as-is intentionally; revisit if the renderer's `--show` vocabulary is ever renumbered.
- **NOTE — deferred algorithm items unchanged.** The Associate→Triangulate→Track payoff (feed the
  triangulated 3D into `05_global_id` instead of the z=0 reproj ground point) and dropping the
  redundant terminal `07_lift3d` re-triangulation remain open (see the code-restructure notes); the
  reorder made them *possible* but they need the standard 8-/40-delivery A/B before adoption.
- **FIXED — orchestrator render-skip token.** The stage renumber renamed the render stage to
  `08_render` in `STAGE_ORDER` but left the compute-loop skip as `if stage == "render"` in
  `src/main.py`, so a default `python -m main` run (window ends at `08_render`) would raise
  `AssertionError("08_render")` before rendering. Now `if stage == "08_render": continue` (render is
  handled by `run_render`). Uncovered by tests (they exercise `_stage_window`/`DeliveryPlan`, not the
  subprocess compute loop) — a future `tests/test_main.py` should drive the loop with a stubbed `_run_stage`.

---

## Data-layout + skeleton + pipeline restructure (2026-07-15)

Landed locally (unit-verified, 201 tests); end-to-end + A/B pending on the L40S box.

- **DONE — dataset abstraction.** `--data-root/--dataset/--version` + `configs/datasets.yaml` +
  `src/core/datasets.py`; one tree under `DATA_ROOT` (laptop `data/` == L40S `~/bits-pose-data`),
  borrow-aware calibration (40_full reads 8_init). `dataset.py`/`calibration.py` treat the passed
  root as the dataset root (no `/dataset` segment).
- **DONE — Halpe-26 canonical (replaces COCO-17).** Contract `halpe26`/26/`g1_player_frame/v1`;
  `pose_2d`/`pose_3d` are 26; `*_native` dropped; `pose_3d_named` (root-relative, named) added;
  viz draws 26 (feet). **Consumer-facing** — coordinate the v1 bump with downstream groups.
- **DONE — 04 the single triangulation; 07_lift3d deleted.** 04 copies correspondences forward;
  05 reads 04 and carries `pose_3d` forward; 06 emits the terminal role-stamped, suppression-filtered
  predictions (the merged handoff); `export_ue_packets` retargeted to the 26-joint run-dir.
- **OPEN — decide-in-3D (flag-gated A/B).** 05/06 should *consume* the 3D (pelvis_ground_xy +
  pelvis_cov_m2 as measurement/R from 04's `lift3d.jsonl`; 3D re-ID). Implement behind `--track-in-3d`
  (default OFF = current ground-plane path), A/B on 8 then 40, adopt only on generalised gain.
- **OPEN — docs sync.** `docs/{pipeline/04-lift,05-global-id,07-export-and-render,meeting-debug-reference,
  architecture,shared-data}.md`, README, CHANGELOG still describe `07_lift3d` / COCO-17 / old paths —
  refresh to 04-single-triangulation, Halpe-26, `data/derived/<dataset>/pipetrack_v<n>` + `data/viz`,
  and the `pipetrack` repo name.
- **NOTE — secondary de-hardcoding deferred.** `run_phase1_{l40s,parallel}.py` `/home/ubuntu/...`
  defaults + `id_pipeline.py --drive-root` default still legacy; harmless (main.py/P1 derive paths),
  finish during the L40S pass.
- **NOTE — YOLO/generic P1 path.** `phase1_outputs.coerce_coco17_keypoints` now pads to 26 (feet = 0)
  under the halpe26 contract — fine for the benchmark-only YOLO path; RTMPose-x (the mandate) emits real 26.

- **FINDING (v9 8_init run, 2026-07-15) — 26-joint 3D coverage regression.** `tri_cov` dropped to
  0.27–0.57 (v8 COCO-17 was ~0.48–0.92). Cause: the lift skips a player-frame when the smoothed
  26-point skeleton isn't fully finite (`run_triangulation.py` ~L347), so a frame where the feet
  aren't multi-view-triangulable loses ALL 3D — even the body joints that were fine. Fix options:
  (a) allow per-joint `null` in `pose_3d.keypoints_world_m` (relax the contract to nullable points)
  and emit whatever triangulated; or (b) gate the skip on the COCO-17 core only, prior-fill/flag feet.
  Recommend (a) — it preserves body 3D and marks un-triangulated feet honestly. Verify tri_cov recovers.

- **CORRECTION (measured 2026-07-15) — the nullable-joint fix did NOT recover tri_cov.** After
  re-running 04–06 on 8_init with the fix, tri_cov was unchanged (0.27–0.57). Root cause was
  mis-diagnosed: feet are prior-filled, so almost no player-frames were dropped for missing feet.
  The real coverage gap vs v8 is (1) a modest 26-vs-17 effect at the binding lift (v8 p3_5=0.646 →
  v9 04_lift=0.566 on M1_1_14_1) and (2) **dropping the global-keyed terminal re-triangulation** —
  v8's p6_3d pooled all views per FINAL global id (0.817) whereas v9's single 04 lift is binding-keyed
  (0.566). Recovering (2) means re-triangulating per global id after 05 (what 07_lift3d did) or
  pooling 3D in 05 — i.e. part of the **deferred decide-in-3D** work, not a quick fix. The 8_init
  set also includes weak deliveries (M2_1_12_1 etc.) that drag the mean down. The nullable-joint
  change is kept (correct/honest per-joint null), just not a coverage lever.
