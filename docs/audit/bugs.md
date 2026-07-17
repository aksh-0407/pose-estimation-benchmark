# Audit bug register (2026-07-17 campaign)

Running register of every defect found during the full-codebase audit. Nothing here is
fixed silently: each entry states what is wrong, the evidence, the impact, and its
disposition. Entries marked "report only" are deliberately left in the code per the
owner's decision and await a separate go-ahead; entries marked "fixed in campaign"
were resolved by this campaign with the batch noted.

Severity: S1 = wrong output or silently disabled feature; S2 = correctness risk or
misleading behavior; S3 = fragility, edge case, or hygiene.

## S1

### BUG-A1: All ground positions come from the bounding-box bottom, not the feet
- Where: `src/identity/common/geometry.py:187` (`ground_contact_pixel_ex`)
- What: the function guards `points.shape != (17, 2) or confidence.shape != (17,)`
  and returns the bbox bottom-center when the guard fails. Since the Halpe-26
  migration, every caller passes 26-keypoint arrays (`src/core/contract.py`
  `KEYPOINT_COUNT = 26`; `src/identity/p3_association/jsonl_io.py:79` validates 26
  and passes through), so the guard always fails and the function always returns
  the bbox bottom.
- Impact: every ground point in cross-camera association (clustering gate,
  emitted feet and heights, the z0 reprojection solver input) is the bbox
  bottom-center. The three foot-contact modes (`legacy` ankle, `v2`, `v3`
  heel/toe) and their fixes are all dead code in production. The production
  config value `foot_contact_mode: v3` in `configs/03_association.yaml` is inert.
  All published baselines (verdict 0.689 on the 8-delivery set, the 40-delivery
  numbers) were measured with bbox-bottom grounds.
- Also affects: `src/identity/p2_tracking/jsonl_io.py:88` (same function).
- Disposition: REPORT ONLY (owner decision 2026-07-17). Fixing changes every
  ground position and invalidates tuned thresholds downstream; the fix must be
  an evaluated change, not a cleanup. Proposed design: make the guard
  length-aware (accept 17 or more rows; Halpe indices 0-16 are the COCO-17
  subset so ankle indices 15/16 remain valid), gate emit-path, clustering-path,
  and tracking-path activation behind three separate config flags, and A/B each
  on the 8-delivery set before any production enablement.

### BUG-A2: Global-id diagnostic "feet unusable" gate never consults ankle confidence
- Where: `src/identity/p5_global_id/runner.py:402-409` (`_feet_unusable_for_metrics`)
- What: same 17-versus-26 pattern as BUG-A1: `conf.shape != (17,)` is always true
  for production 26-length confidence arrays, so the ankle-confidence branch is
  unreachable and any bounding box touching the frame bottom is unconditionally
  treated as "feet unusable" and anchored on the upper-body height plane.
  Sibling code in the same file (lines 88, 454-457) checks 26 correctly.
- Impact: diagnostics only (teleport/agreement anchoring); emitted tracks are
  unaffected. The diagnostic is stricter than designed for bottom-of-frame boxes.
- Disposition: fixed in campaign (batch 3e), because it only changes diagnostic
  metric values, never emitted tracks. Before/after metric deltas on the local
  8-delivery trees are published in the changes ledger.

## S2

### BUG-B1: Track finalization can mint IDs that the shadow gate would have blocked
- Where: `src/identity/p5_global_id/track_manager.py:669-676` (`finalize`)
- What: end-of-delivery promotion mints a global id for any tentative track with
  2 or more hits, without running `_confirmation_blocked` (the shadow-duplicate
  and roster-cap gates that normal promotion applies).
- Impact: a late shadow duplicate of an already-confirmed player can be minted
  as a fresh id in the last frames of a delivery, inflating the id count.
- Disposition: report only (behavior-affecting). Proposed fix: route finalize
  through the same gates as `_promote_and_prune`, evaluated on the 8-delivery set.

### BUG-B2: The "a capped ground cue alone can never merge" invariant is no longer true
- Where: `src/identity/p3_association/config.py` (`graph_llr_positive_cap`),
  `src/identity/p3_association/tracklet_graph.py` merge threshold.
- What: comments in both files assert that the per-cue positive cap keeps a
  single ground-plane cue below the merge threshold (cap 1.5 < threshold 2.0).
  Production YAML raises the cap to 3.5, above the threshold 2.0, so a single
  strongly-supported ground cue can clear the merge threshold alone. The YAML
  change was a deliberate, measured tune; the invariant prose was never updated.
- Impact: merge safety now rests on cue quality and support weighting rather
  than the documented structural bound. Not a regression by itself, but the
  code comments actively mislead maintainers.
- Disposition: comments corrected in campaign (batch 3g); the abandoned
  invariant is recorded here and in docs/pipeline/known-bugs.md. Any cap retune
  is a separate evaluated change.

### BUG-B3: Bowler slot cost uses absolute speed, partially undoing the signed-direction fix
- Where: `src/identity/p6_roles/assigner.py:205` (`_slot_cost`)
- What: `_windowed_axis_speed` was deliberately made signed so that a sprint in
  the wrong axis direction cannot look like a bowler run-up, but the slot cost
  uses `abs(speed)`, reintroducing that ambiguity at slot-assignment level. The
  two-direction axis trial in the runner mitigates but does not eliminate it.
- Disposition: report only (behavior-affecting). Proposed fix: sign-aware slot
  cost consistent with the windowed speed, evaluated on role accuracy.

### BUG-B4: Batch driver misclassifies stage failures that exit 1
- Where: `src/main.py:250-258`, `src/identity/id_pipeline.py:102`
- What: exit code 1 is interpreted as a "warn verdict" for 03_association and
  05_global_id, and a crash is distinguished from a warn only by the presence of
  the stage's metrics artifact. `id_pipeline.py` applies the same convention to
  P3 but never gates the global-id return code at all. Any new stage (or any
  unexpected exception path) that exits 1 is misclassified as a soft warning.
- Impact: a genuinely failed stage can be treated as "ran with warnings" and the
  chain continues on incomplete output.
- Disposition: report only (behavior-affecting error-handling change). The
  convention is now documented in docs/pipeline/08-export-and-render.md and the
  stage docs. Proposed fix: distinct exit codes for warn-verdict (for example 3)
  versus failure, applied across stage CLIs and both drivers in one change.

### BUG-B5: The mosaic always renders from the global-id stage, ignoring roles and refinement
- Where: `src/main.py:262-271` (`run_render`)
- What: the render step reads `05_global_id` even when `06_roles` (role stamps,
  suppression) and `07_refine` (physically-constrained 3D) produced the terminal
  predictions. Role chips and suppression reach the mosaic through side files,
  but the refined 3D never reaches the render.
- Impact: rendered overlays show pre-refinement 3D; role/suppression handling
  works but relies on side-channel files rather than the terminal stage output.
- Disposition: report only (output-affecting). Proposed fix: render from the
  latest completed stage in the window, with side files resolved from it.

### BUG-B6: Tiled detection on the L40S runner depends on an out-of-tree import
- Where: `src/core/inference/run_phase1_l40s.py` (lazy `from detector_bakeoff import ...`)
- What: tiled mode imports from `tools/detector_bakeoff`, which is only
  importable when the process starts at the repo root (or with a tuned
  PYTHONPATH). A plain `--tiled-det` invocation from elsewhere raises
  ImportError after startup.
- Disposition: report only (packaging decision). Proposed fix: move the tiling
  helpers into `src/core/inference/` or make `tools` a proper package the
  runner adds to `sys.path` explicitly.

### BUG-B7: Suppression file errors are silently swallowed by the 3D lift
- Where: `src/identity/p4_lift/run_triangulation.py:176`
- What: `OSError` and `JSONDecodeError` when reading the suppression file are
  caught and ignored, so a corrupt or unreadable suppression file silently
  lifts every player instead of failing or warning.
- Disposition: warning log added in campaign (batch 3f); behavior (continue
  without suppression) unchanged.

### BUG-B8: Facing-pair derivation failure silently disables the adaptive facing gate
- Where: `src/identity/p3_association/tracklet_graph.py:302`
- What: any exception while deriving facing camera pairs from calibration is
  caught bare and results in an empty facing-pair set, silently disabling the
  parallax-adaptive gate scaling.
- Disposition: warning log added in campaign (batch 3f); fallback behavior unchanged.

## S3

### BUG-C1: Contract camera-id formatting breaks at 10 or more cameras
- Where: `src/core/contract.py:291` (`f"cam_0{index}"`)
- What: builds camera ids by string concatenation that produces `cam_010` for
  index 10, which the contract's own `cam_\d{2}` pattern rejects. Correct today
  (7 cameras), wrong the day the rig grows.
- Disposition: fixed in campaign (batch 3d) with zero-padded formatting.

### BUG-C2: Resume-scan swallows corrupt JSONL lines silently
- Where: `src/core/inference/run_phase1_rtmpose_inference.py` resume scan
- What: any exception while parsing a completed-predictions line is ignored,
  silently shrinking the resume set and recomputing frames.
- Disposition: warning log added in campaign (batch 3f); resume semantics unchanged.

### BUG-C3: Velocity gate blind spots in ground-track emission (documented trade-offs)
- Where: `src/identity/p5_global_id/runner.py` (`_velocity_gate_ground_rows`)
- What: (a) the first emitted frame of an id always anchors the gate, so a
  teleported first frame survives; (b) after 5 consecutive drops the gate
  re-anchors to the rejected position, admitting a sustained mis-association
  as a "relocation".
- Disposition: report only; both are documented design trade-offs, recorded so
  they are weighed in any future teleport work.

### BUG-C4: Identity-only bridge hits inflate confirmation credit
- Where: `src/identity/p5_global_id/global_track.py:121-123`
- What: identity-only hits (no position update) still increment `hits` and mark
  the track single-camera, so confirmation and the adaptive lost window reward
  evidence that never moved the filter.
- Disposition: report only (behavior-affecting).

### BUG-C5: Stitch cost terms that are inert in practice
- Where: `src/identity/p5_global_id/stitching.py`
- What: (a) the role-incompatibility penalty treats `unknown` as free, and the
  online role proxy labels most players unknown, so the large `w_role` weight
  rarely fires; (b) velocity continuity contributes zero for near-static or
  short links, exactly where co-located ghosts need discrimination.
- Disposition: report only; recorded as tuning-context, not defects.

### BUG-C9: tools/run_v8_l40s.sh was broken against the current batch driver
- Where: `tools/run_v8_l40s.sh`
- What: the script passed `--input-tree`, an option `src/main.py` does not
  define (argparse exits with "unrecognized arguments"). Stale since the
  per-delivery run-tree rework replaced the flat P1 input tree.
- Disposition: fixed in campaign (batch 2f); rewritten for the per-delivery
  layout the L40S runner now writes by default.

### BUG-C6: Batch drivers' stale defaults and unbounded fan-out
- Where: `src/identity/id_pipeline.py` (default trees point at pipetrack_v8;
  8 hardcoded delivery ids), `src/main.py` (`--jobs` times `--p2-max-workers`
  processes with no global cap), `src/core/inference/run_phase1_parallel.py`
  (shard subprocess has no timeout; a hung shard blocks forever).
- Disposition: id_pipeline defaults updated in campaign (batch 3c). Fan-out cap
  and shard timeout are report only (operational behavior).

### BUG-C7: Diagnostic-metrics calibration fallbacks are silent
- Where: `src/identity/p5_global_id/runner.py:381-396`
- What: three broad excepts fall back to empty projection/image-size maps, so
  the teleport/agreement metrics silently degrade to approximate anchoring.
- Disposition: warning log added in campaign (batch 3f).

### BUG-C8: Renderer silent excepts hide real errors
- Where: `src/identity/visualization/render_videos.py:1758` (OpenCL), `:1902`
  (ghost projection load)
- What: `except Exception: pass` hides all failure detail.
- Disposition: warning logs added in campaign (batch 3f).

## Fixed-label defects (viewer- or metadata-visible, zero pipeline change)

### BUG-D1: Mosaic footer labels the skeleton "COCO-17" while rendering Halpe-26
- Where: `src/identity/visualization/render_videos.py:759-763`
- Disposition: fixed in campaign (batch 3d). Rendered pixels change (the label),
  predictions do not.

### BUG-D2: Detector name hardcoded in P1 metadata
- Where: `src/core/inference/run_phase1_rtmpose_inference.py` metadata blocks
- What: metadata said `rtmdet_m_person` even when another detector preset was
  selected.
- Disposition: already fixed by the detector-preset working-tree change that
  predates this campaign (kept intact and folded through the batch 2a refactor).

### BUG-D3: Dead conditional in relift nearest-point selection
- Where: `src/identity/p7_refine/relift.py:83`
- What: `x - anchor if False else x - prev` always evaluates `x - prev`; the
  dead branch confuses readers into thinking anchor distance is considered.
- Disposition: fixed in campaign (batch 3d); expression simplified, identical result.

### BUG-D4: Stale pre-restructure paths in user-facing help text
- Where: `src/identity/p6_roles/suppress_peripherals.py:51`,
  `src/identity/visualization/render_videos.py:219`
- What: help text cites `p5/roles.json` layouts from before the stage-numbered
  restructure; the actual default is `06_roles/roles.json`.
- Disposition: fixed in campaign (batch 3c).

### BUG-D5: Batch-driver help text contradicts its own defaults
- Where: `src/main.py:406-419`
- What: four `--tri-*` options say "default off" or "default ema" in help while
  the argparse defaults are on/butterworth (defaults were flipped when the
  fixes were accepted; help was never updated).
- Disposition: fixed in campaign (batch 3a).

## Deferred structural observations

- `configs/06_roles.yaml` says suppression is "disabled here" directly above
  `suppression_enabled: true`; comment corrected in campaign (batch 3g) after
  verifying production run manifests enable it.
- Known-bugs doc cites `runner.py#L286` for the posterior-emission branch; the
  code now lives near line 352. Anchor corrected in the docs pass (batch 4b).
- The L40S runner projects run ETA from a hardcoded 134400-frame full-campaign
  count; wrong the moment the dataset changes. Report only.
