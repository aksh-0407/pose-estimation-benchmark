# Audit change ledger (2026-07-17 campaign)

Every change made by the full-codebase audit campaign, grouped by batch, with its
verification evidence. Companion to [bugs.md](bugs.md) (defect register) and the
plan of record. Rules of the campaign:

- No behavior-affecting changes to pipeline outputs. Refactors are verified
  byte-identical against a golden reference run; mechanical fixes touch only
  comments, docstrings, help text, labels, metadata, and logging.
- The single sanctioned exception: the global-id diagnostics guard fix
  (bugs.md BUG-A2), which changes diagnostic metric values only, never emitted
  tracks. Its before/after deltas are recorded here.
- One commit per batch item; the owner runs the commits.

## Pre-campaign working-tree state

The tree carried uncommitted work that predates the campaign and is preserved
as-is: detector presets for both P1 runners (`--detector rtmdet_l | rtmdet_x |
dino`, resolver, metadata threading) in `run_phase1_rtmpose_inference.py` and
`run_phase1_l40s.py`, plus the untracked fetch script
`tools/detector_bakeoff/fetch_detectors.py`. This work also fixed the
hardcoded detector metadata label (bugs.md BUG-D2). Recommended to commit
separately before the campaign commits.

## Batch 0: preflight

- Baseline test suite recorded (pose-lab environment, PYTHONPATH cleared to
  keep ROS site-packages out of collection).
- Golden determinism reference: stages 02_tracking through 07_refine re-run
  twice on delivery CCPL080626M1_1_14_1 from the frozen
  `data/derived/8_init/pipetrack_v90` stabilization output; the two runs are
  diffed to prove determinism, and run A is kept as the byte-identical
  reference for all refactor batches.
- This document and bugs.md created.

## Batch 1: legacy YOLO stack removal

Planned: delete `src/core/inference/run_phase1_yolo.py`, `phase1_runner.py`,
`phase1_outputs.py`, `yolo_adapter.py`, `run_yolo26x_final.sh`,
`tests/test_cricket_p1.py`, and the `models/yolo26x_pose/` weights; sweep and
fix all referrers. Rationale: production is RTMPose by mandate; the YOLO stack
is a self-contained legacy island whose output pads COCO-17 to Halpe-26 with
fabricated zero-coordinate joints labeled as real (a correctness hazard if it
ever fed the pipeline).

## Batch 2: structural refactors (byte-identical)

### 2a: phase1_common extraction (DONE, verified)

New `src/core/inference/phase1_common.py` now owns everything the two RTMPose
runners shared: detector presets and resolution, model construction, batched
detection and pose inference, decode prefetching, camera-target discovery
helpers, the per-frame record schema (`build_frame_record`, single definition
of the P1 JSONL line for both runners, key order preserved), resume-state
reading (now warning on corrupt lines instead of silently recomputing), and
path/time helpers. `run_phase1_rtmpose_inference.py` (1314 to 715 lines) keeps
its CLI, discovery, overlays, run loop, and metrics; `run_phase1_l40s.py` keeps
its CLI, native-layout discovery, tiled detection, sweeps, and metrics, and now
imports the shared module instead of reaching into the other runner script.
The L40S runner's duplicated `resolve_skeleton` and `git_sha` are gone.

Verification:
- rtmpose runner, 40 frames, fixed run id: predictions JSONL byte-identical
  before vs after the refactor.
- L40S runner, deterministic settings (--no-amp --no-perf): player data
  identical to the rtmpose runner on all 40 frames (shared numeric path).
- L40S runner in its default mixed-precision mode showed a one-frame, one-field
  wobble (detection_confidence 0.8213 vs 0.8208) against the pre-refactor
  reference; two identical post-refactor runs are byte-identical to each other.
  This is cross-process nondeterminism inherent to fp16 autocast plus
  cudnn.benchmark kernel selection, not a refactor effect; recorded in
  docs/pipeline/00-inference.md.
- --help output: unchanged for the L40S runner; the rtmpose runner's help
  changed only in the intended docstring correction (the old text described the
  pre-restructure drive/dataset path and flat output layout).

### 2f: L40S per-delivery output layout (DONE, verified)

`run_phase1_l40s.py` gains `--layout {per-delivery,flat}`, default per-delivery:
predictions land at `<output-dir>/<DELIVERY>/00_inference/predictions/`, the
run-tree layout `src/main.py` consumes directly. The historical flat layout
remains available. The run manifest records the layout. Wrapper updates:
`run_rtmpose_x_l40s.sh` output default is now the run root;
`tools/run_v8_l40s.sh` rewritten for the current driver (it was broken: it
passed `--input-tree`, an option `src/main.py` does not have; see bugs.md).

Verification: same 40 frames in both layouts produce byte-identical prediction
content, and the flat output is byte-identical to the 2a deterministic reference.

### 2b: render_videos split and viz dedup (DONE, verified)

`render_videos.py` (1982 lines) is now the CLI/orchestration layer only
(~700 lines); the building blocks moved to four new modules in
`src/identity/visualization/`:

- `video_io.py`: `VideoSink` (NVENC/libx264/OpenCV encoders), GPU JPEG decode,
  `load_image_for_record`, `parse_size`.
- `loaders.py`: every JSONL/side-artifact reader (`iter_jsonl`, `load_records`,
  `load_cluster_badges`, `load_ball_positions`, `load_roles`,
  `load_suppression`, `load_pitch_extents`, roster and extents derivations).
- `overlays.py`: drawing primitives, chip placement, skeleton body paint,
  player boxes, ghosts, ball trail, headers/footers, `render_feed_frame`.
- `panels.py`: roster, info, and bird's-eye tiles.

Duplicates removed from the sibling renderers: `render_phase1_overlays.py` and
`render_bird_eye_view.py` now import `iter_jsonl` / `load_cluster_badges` /
`stage_from_manifest` from `loaders` instead of their own copies. The
bird's-eye `stable_color` was deliberately left alone (replacing it with the
shared id-color helper would change that tool's rendered colors).

Verification: mosaic render determinism proven (two pre-split renders with the
CPU encoder produce identical frame md5s); the post-split render of the same
120 frames is frame-for-frame identical to the pre-split reference (ffmpeg
framemd5). Renderer unit tests pass. One stdout label changed ("suppressed ids
(P5b):" is now "suppressed ids:").

### 2c: core/dataset.py renamed to core/frames.py (DONE, verified)

Ends the `core.dataset` / `core.datasets` module-name collision: `frames.py`
holds the frame-layout helpers (frame filename parsing, camera dirs, repo
paths), `datasets.py` remains the dataset/path registry. Three importers
updated. Verification: compile plus full test suite (only the two pre-existing
stale-expectation failures remain, fixed in batch 3).

### 2d: global-id package stage-05 naming (DONE, verified)

`P4Config` / `P4AConfig` / `P4BConfig` are now `GlobalIdConfig` /
`GlobalTrackingConfig` / `StitchingConfig`, and the `p4a` / `p4b` attributes are
`tracking` / `stitching` (about 136 sites across 10 files). Backward
compatibility: the old class names remain as module aliases; the YAML loader
accepts both the new `tracking:`/`stitching:` sections and the old
`p4a:`/`p4b:` spellings (rejecting a mix); the roles stage reads both manifest
key spellings so archived run trees keep working. `configs/05_global_id.yaml`
and the experiment YAML migrated.

Consequence, by design: the stage run-manifest's config echo now uses the new
key names (`config.tracking` / `config.stitching`).

Verification: stages 05 through 07 re-run against the golden reference;
predictions, diagnostics, and metrics byte-identical; the only manifest deltas
are the intended key rename and tree-path strings. An old-key-spelling YAML
produces output identical to the new spelling.

### 2e: emit-path performance, provable subset only (DONE, verified)

Applied: the tracklet graph's `_overlap_frames` no longer rebuilds both frame
sets on every call (it was called inside nested member loops during
agglomeration and refinement); frame sets are now cached per chunk, keyed by
(chunk, frame count) since chunk frame lists only ever grow. Stage 03 re-run
against golden: fully byte-identical, wall clock for 03 to 05 on one delivery
68 s.

Deferred as proposals (cannot be proven byte-identical within this campaign,
recorded in bugs.md deferred section): memoizing the double foot computation
between `smooth_emit_feet` and the emit path; collapsing the three
near-identical iterated-reweighting assembly loops in geometry.py and
triangulation.py (reordering float operations risks sub-ulp output drift);
hoisting the per-cluster-per-frame reprojection RANSAC out of the emit path.

## Batch 3: mechanical fixes (DONE, verified)

- 3a: the four contradictory `--tri-*` help strings in `src/main.py` fixed
  (defaults were flipped on when the fixes were accepted; help still said
  off); `--p4-config`/`--p5-config` help now states the stage each flag
  configures. The two stale tests (written before 07_refine/08_render joined
  the chain) updated; the suite is fully green (210 passed).
- 3b: whole-repo em-dash purge: 71 files, zero remain across src, tools,
  tests, configs, models, docs, README, CHANGELOG.
- 3c: insider-jargon translation: all campaign markers (wave numbers, fix
  numbers, issue ids, person names) rewritten as plain self-explanatory
  rationale, with docs/pipeline/fixes-log.md as the pointer where history
  matters. Module docstrings across the global-id package now use stage-05
  naming. `id_pipeline.py` defaults updated from the v8 era to the v9 trees.
- 3d: mosaic footer corrected from "COCO-17" to "Halpe-26" (rendered label
  only); dead `if False` ternary in relift.py simplified; contract camera-id
  formatting zero-padded; redundant re-import removed.
- 3e: the global-id diagnostics "feet unusable" guard fixed (required exactly
  17 confidence values; production is 26, so the ankle check never ran).
  Neutral before/after on the golden delivery: emitted tracks and predictions
  byte-identical; cross-camera agreement diagnostic 0.9327 to 0.9297
  (agreeing pairs 14572 to 13634); teleports, ids, persistence, and verdict
  unchanged. Recorded in docs/methods_log.md Part A0.
- 3f: warning logs added to five silent-failure sites (facing-pair
  derivation, suppression-file read, three calibration fallbacks in the
  global-id metrics path, GPU-decode fallback, resume corrupt-line skip).
  Fallback behavior unchanged.
- 3g: contradicting config comments corrected: `configs/06_roles.yaml`
  suppression comment now matches the enabled value; the abandoned
  positive-cap merge invariant is documented at the config site and in
  known-bugs.md.

Verification: full pytest green; stages 02 to 07 re-run against the golden
reference: 02/03/04/06/07 predictions, diagnostics, and metrics byte-identical
(manifest deltas are tree paths and the intended stage-05 key rename only);
the only content change anywhere is the 3e diagnostics delta above.

## Batch 4: docs refresh (DONE)

- Glyph purge: zero emoji/status glyphs remain in docs (was 106 emoji lines
  plus 162 star markers plus arrows); severity and status are now text.
- known-bugs.md rebuilt with text statuses; six new entries from the audit
  (BUG-9 foot-contact guard, BUG-10 finalize gate, BUG-11 abandoned cap
  invariant, BUG-12 abs-speed slot cost, BUG-13 exit-code convention, BUG-14
  render source), NB-2 expanded with the measured dataclass-vs-YAML drift
  tables, and BUG-8 corrected (the driver does run stabilization by default).
- New docs: docs/reference/legacy-and-dead-code.md (every kept dead path with
  status and reactivation notes) and docs/reference/glossary.md (newcomer
  glossary: cricket, skeleton, identity-pipeline, evaluation, and layout
  terms).
- Stage-doc accuracy: 00-inference.md documents phase1_common, the detector
  presets, the per-delivery L40S layout, and the mixed-precision
  reproducibility note; 05-global-id.md documents the tracking/stitching key
  rename and compatibility; 08-export-and-render.md documents the renderer
  module split and the driver exit-code convention; architecture and
  references updated for the frames.py rename and the viz split.
- docs/methods_log.md gained Part A0 (this campaign, with the one
  metric-definition change and its numbers).

## Verification log

- Determinism baseline: two identical 02-07 runs on CCPL080626M1_1_14_1
  differ only in timestamps and embedded paths (2026-07-17).
- P1 refactor: 40-frame GPU runs byte-identical (standard runner); L40S
  deterministic mode player-identical to the standard runner on all frames;
  mixed-precision wobble characterized and documented.
- Render split: 120-frame mosaic frame-md5 identical pre/post split.
- Stage-05 rename: outputs identical under both YAML key spellings.
- Post-Batch-3 golden rerun: everything byte-identical except the published
  3e diagnostics delta.
- Test suite: 210 passed, 0 failed (the 2 pre-existing failures were stale
  expectations, fixed in 3a).
