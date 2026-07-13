# To-do — implementing the critical-analysis fixes (pipetrack_v6 campaign)

Execution plan for the fixes catalogued in this directory. Companion documents:
[fixes-roadmap.md](fixes-roadmap.md) (the ranked fix list this plan executes) and
`fixes-log.md` (the accept/reject record, created with the v6.0 baseline).

## Context

The critical-analysis review concluded: calibration and 3D location are largely solved;
**identity is the dominant quality ceiling** (x-cam agreement down to 0.50, 18–25 distinct IDs
vs ~13 roster, 7–155 teleports/clip, colour cue dead d′≈0). `fixes-roadmap.md` ranks 15 fixes.
Deliverables: (1) a ground baseline `pipetrack_v6.0` built from the existing
`benchmarks/runs/rtmpose-x` P1 data (verified complete: 8 deliveries × 7 cams) with mosaics in
`artifacts/pipetrack_v6.0/`; (2) all fixes implemented, flag-gated, A/B'd on all 8 deliveries;
(3) `docs/critical-analysis/fixes-log.md` maintained in the `wip/methods_log.md` lab-notebook
format; (4) literature/web research wherever a listed fix looks unsatisfactory. Final judge:
human mosaic review after the numbers are good.

**Scope decisions (2026-07-10):** baseline = current default (v5 configs, P1.5 off; P1.5 wiring
is Fix F1); detector/P1 work in scope locally (RTX 4060); **mosaics rendered only on explicit
request** (the v6.0 ground mosaics are the one standing request; otherwise report data panels).

**Fresh wip/ state (reorganized 2026-07-10, post-dates the critical-analysis docs):**
`wip/methods_log.md` now consolidates location (M0–M11) + identity (ID-0–ID-6) methods; the v5
identity stack is already **validated on all 8 deliveries** (`_7` agreement 0.498→0.600, distinct
IDs 10–16 vs roster, teleports down on every clip, collisions 0) but **held out of committed
configs pending mosaic-review sign-off** — our v6.0 baseline (v5 configs) therefore starts from
these validated numbers, not the older v3 figures in the critical-analysis tables.
`wip/recommendations.md` sets priorities the plan must honor:
- Next identity step #2: **feed the billboard posture descriptor into the P4a teleport veto**
  (works on facing pairs, unlike the triangulated descriptor; most direct route to residual M2
  teleports) → added as fix F6b.
- Next identity step #3: cross-delivery prior calibration for anchor-starved clips → matches F8.
- **Trainable person-ReID is explicitly rejected** ("no identity GT to train on; geometry plus
  pose-shape carry the available signal") → F14 demoted to parked-unless-user-approves.
- M2's 166 residual teleports are the teleport-*proxy* reacting to noisy single-camera foot
  projections (single-cam rate 0.61), not the emitted trajectory → F9a/F10 (covariance/R)
  attack exactly this; M2 teleport count must be read with that caveat in A/B verdicts.
- Pending model study: RTMPose-L vs RTMPose-X (COCO-17 vs Halpe-26) write-up in
  `wip/model_comparison.md` — the v6.0-vs-v5 panel diff supplies the X-vs-L data point;
  record observations there when the baseline lands.

**Verification standard** (from `wip/3d_location_methods_log.md` + `implementation_plan.md` §5):
frozen baseline; evaluate across all 8 deliveries; accept only significant, *generalized*,
non-regressing improvement; every change behind a config flag (flags off ⇒ byte-identical);
metrics read as a joint panel, never singly; collisions must stay 0 always.
Envs: pipeline via `/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/python`; P1 inference in
`cricket-rtmpose-l` (RTMO in `cricket-rtmo-l`); pytest in `cricket-yolo26x-pose` with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=""`.

---

## Part A — Ground baseline: pipetrack_v6.0

### A1. Frozen config set `configs/v6/` (immutable after baseline; committed)
- `p1b_stabilization.yaml` — copy of `configs/p1b_stabilization.yaml` with `enabled: false`
- `p2_tracking.yaml` — copy of `configs/p2_tracking.yaml`
- `p3_association.yaml` — copy of `configs/p3_association_v5.yaml`
- `p4_global_id.yaml` — copy of `configs/p4_global_id_v5.yaml`
- Per-fix variants: `configs/experiments/v6_fNN_<slug>__pX.yaml` (full copies, flags flipped,
  diffable against `configs/v6/`). Accepted flags roll into a future `configs/v7/` cut, never
  edits to v6.

### A2. Metrics patch: `scripts/export/triangulate_predictions.py` (~line 205)
Add aggregates to `triangulation_metrics.json`: `mean_reprojection_error_px`,
`p95_reprojection_error_px`, `triangulation_coverage` (= fully_valid / candidate frames).

### A3. New driver `scripts/pipetrack/run_full_pipeline.py`
Keep `run_id_pipeline.py` untouched (fast P3→P4 inner loop); import its helpers
(`_blas_capped_env`, `_run_stage`, `_dig`, `_fmt`, `ALL_DELIVERIES`, panel-diff pattern).
- Stage registry `p1b→p2→p3→p4→p5→p6_3d→render`, writing
  `benchmarks/runs/<tree>/deliveries/<D>/{p1b,p2,p3,p4,p5,p6_3d,logs}`.
- CLI: `--deliveries`, `--input-tree` (P1 run dir, default `benchmarks/runs/rtmpose-x`),
  `--output-tree`, `--base-tree` (reuse upstream stage dirs directly — the A/B workhorse:
  `--from-stage p3 --base-tree v6.0` skips P2), `--from-stage/--until-stage/--skip-render`,
  `--enable-stabilization` (p1b default off), per-stage `--pX-config`, triangulation knobs,
  `--artifacts-root`, `--jobs 8`, `--p2-max-workers 2`, `--render-jobs 2`, `--panel-only`,
  `--baseline`.
- Exit codes: rc=1 fatal for P2 (real camera failure), warn-continue for P3/P4 (verdict),
  fatal elsewhere.
- Two-phase execution: compute chains fan out in ThreadPool(`--jobs`) with BLAS caps; render
  submitted per delivery on chain completion into a separate `--render-jobs 2` pool.
- Render call: `--run-dir .../p4 --mode mosaic --show p4` (probes sibling `p5/roles.json`,
  `render_phase1_videos.py:1738`; `--roles-path` flag exists as fallback).
- `pipeline_manifest.json` per tree: per-stage source dir, config path + sha256.
- Extended panel: existing 7 cols + `cycle_cons`, `d_prime`, `jitter_px`, `reproj_px`, `tri_cov`.

### A4. Baseline run + snapshot
1. Smoke: 1 delivery, `--skip-render --jobs 1`; verify roles.json placement, panel row, p6_3d
   metrics; render that delivery, eyeball the mosaic.
2. Full run all 8 (`--jobs 8 --p2-max-workers 2 --render-jobs 2`) — est. 30–40 min:
   fresh P2 from rtmpose-x → P3/P4 (v6 configs) → P5 → triangulation → 8 mosaics in
   `artifacts/pipetrack_v6.0/mosaics/<D>/`.
3. Snapshot: rsync metrics-only tree to `pipetrack_v6.0/_baseline_snapshot/` (panel-readable
   layout); `chmod -R a-w` the deliveries tree. User runs the git commit (no AI co-authorship).
4. Seed `docs/critical-analysis/fixes-log.md`: Executive Summary, Evaluation Standard, the v6.0
   ground panel (all 8 deliveries), master fix table skeleton.

## Part B — fixes-log.md protocol (mirrors `wip/methods_log.md`)

Structure: `## Executive Summary` → `## Evaluation Standard` → `## Summary of Fixes` master
table `| ID | Fix | Result | Verdict | Rationale |` → `## Detailed Results` with
`### F<n> - <Name>` entries (Purpose / Implementation (files, flags) / Result — 8-delivery
panel table with baseline deltas / Conclusion; optional Caveat/Finding) → `## Current Default /
Recommended State` (default / opt-in / rejected) → `## Next Work` → `## Environment Notes`.
Verdicts: Validated / Accepted / Opt-in / Rejected / No change / Baseline. Every entry cites its
run tree (`benchmarks/runs/pipetrack_v6.<k>-fNN`) and config diff. Log updated the moment each
A/B completes, not at the end.

## Part C — Fix waves

Each fix: implement flag-gated → unit tests → A/B run tree `pipetrack_v6.<k>-fNN` via
`--base-tree` reuse → panel vs current frozen baseline → log entry → accept/reject. Accepted
fixes that change upstream stage outputs (F1) re-freeze the working baseline (v6.1) so later
A/Bs inherit them; pure-P3/P4 fixes chain within the wave and get a combined wave-checkpoint
rebaseline. Checkpoints are reported as data panels; no mosaic renders unless the user asks.

### Wave 0 — F1: wire P1.5 stabilization (first logged fix)
Driver already supports `--enable-stabilization`; run it: p1b on rtmpose-x → fresh P2→P4.
Gate: jitter −~30%, P2 fragmentation/churn + P3 agreement non-regressing. If accepted: freeze
**v6.1** (new working baseline; no mosaics unless requested — report the panel).

### Wave 1 — F2–F8: low-effort correctness batch (independent flags, one combined A/B, bisect by flag on regression)
- **F2 C07 per-camera image size**: audit `image_w/h` consumers in `scripts/association/config.py`
  users + `mosaic_layout.py` (R-1 aspect); thread per-camera sizes. Gate: bit-identical on
  cams 01–06 paths; C07-pair agreement non-regressing.
- **F3 Cheirality check** in `ransac_triangulate_point` (`pose_estimation/triangulation.py:90`):
  reject non-positive-depth candidates/inliers. Gate: triangulation success + reproj
  non-regressing; synthetic unit test.
- **F4 Halpe-26 feet as ground contact**: `geometry.py` `ground_contact_pixel_ex` mode `v3`
  (heel/toe midpoint when conf ≥ 0.5, ankle fallback); pass `pose_2d_native` through
  `scripts/association/jsonl_io.py`. Gate: `eval_ground_accuracy.py` error down; spread/teleports
  non-regressing.
- **F5 Online role proxy → P4a dynamics**: factor role classification out of
  `scripts/roles/assigner.py`; call existing `track_manager.propose_role:571` from confirmed
  tracks (≥50 frames) so `switch_role` sets role-aware Singer params. Gate: teleports/IDs
  non-regressing everywhere, improved on ≥2 deliveries.
- **F6 P4b occupancy-licensed bridging**: `stitching.py` — when two segments' (camera,frame)
  occupancies are disjoint + kinematics hold, allow `temporal_gate_frames_occupancy: 300`.
  Gate: distinct IDs toward roster on ≥4 deliveries; collisions 0.
- **F6b Billboard posture → P4a teleport veto** (repo's own top identity next-step,
  `wip/recommendations.md`): thread the billboard `PostureAggregate` / `posture_distance_z`
  (facing-pair-capable, unlike the triangulated descriptor that rarely fires in the ID-4
  guardrails) into the P4a Stage-2 pose veto and re-entry gate in `track_manager.py`.
  Gate: teleports down on `_5/_6/_7/M2` with agreement/IDs non-regressing.
- **F7 Offline Butterworth 3D smoothing**: `--smoother {ema,butterworth}` (filtfilt, 4th order,
  ~6 Hz @50 fps, per contiguous finite segment) in triangulate script. Gate: 3D jitter p95 down,
  reproj not up.
- **F8 Cue cold-start robustness**: staged anchor relaxation before default-Gaussian fallback +
  optional `calibration_prior_path` (reuse `CueCalibration.save/load`). Gate: fallback count → 0
  with agreement non-regressing.
→ Wave checkpoint: freeze v6.2; report the 8-delivery panel with deltas.

### Wave 2 — F9: P3.5 triangulation re-sequencing (keystone enabler)
- **F9a ground covariance out of P3**: `geometry.py` `ground_from_reprojection_ex` returns
  `([x,y], cov_2x2)` (GN normal matrix, ~free); `Correspondence.ground_cov` emitted in
  `correspondences.jsonl`, parsed in `scripts/global_id/jsonl_io.py`; single-cam fallback uses
  existing per-view `ground_covariance` (geometry.py:368) × inflation; airborne flag (2D proxy
  now, lift-based later) inflates cov + uses pelvis projection. Schema-additive; flags-off
  byte-identical.
- **F9b binding-keyed lift stage (P3.5)**: refactor `triangulate_canonical_run` grouping into
  `--id-source {global,binding}` reusing `scripts/global_id/jsonl_io.py` member resolution;
  emit `diagnostics/lift3d.jsonl` (per-joint 3D + covariance `σ̂²(AᵀA)⁻¹` + inlier mask +
  reproj errors, pelvis ground, airborne flag) and `diagnostics/lift_purity.json` (torso-residual
  chimera signature per binding + bone-ratio descriptor + metric stature). Driver inserts p3_5
  stage; terminal export stops re-triangulating (stamps P3.5 3D into P4 records via the 1:1
  binding→global map; `export_ue_packets.py` unchanged input contract).
- **F9c in-runner lift feedback hook**: factor core into `scripts/association/cluster_lift.py`;
  P3 runner computes per-provisional-cluster residuals/descriptors post-solve behind
  `graph_lift_feedback: false` (logged, unused). Gate: <10% runtime overhead; identical outputs
  flag-off; purity-flag rate correlates with cycle-consistency proxy.

### Wave 3 — F10–F12: uncertainty R + pose-shape primary cue
- **F10 per-measurement R → Singer KF**: `SingerGroundKalman.update/mahalanobis_sq` accept
  optional 2×2 R (default `self.R`); plumb `ground_cov` through `GroundObservation` at all five
  call sites (`track_manager.py:206,256,380,410`, `global_track.py:81`) so gating and update use
  the same R; eigenvalue-floored/ceilinged (`r_floor_m 0.15`, `r_ceiling_m 2.0`). Gate:
  teleports down materially (M2 155 is the target); agreement/IDs non-regressing; expect small
  `chi2_gate_2dof` re-tune.
- **F11 `llr_shape` cluster-round cue in P3**: bone-ratio descriptor needs 3D (verified), so
  this is a second, cluster-level corroboration round after the F9c lift: calibrated
  `descriptor_distance` + stature term between compatible cluster pairs; same LLR/veto/abstain
  machinery (`cue_calibration` key `shape_dist`); `_refine` chunk moves use billboard stature vs
  cluster metric scale. Gate: facing-pair agreement up (`_7` 0.50 → ≥0.60), single-cam rate
  down, cycle-consistency non-regressing.
- **F12 stature into P4 re-entry + P4b stitch cost**; re-raise F6 aggressiveness with the
  stronger descriptor. Gate: distinct IDs → ~13–16 with agreement/teleports held.
→ Wave checkpoint: freeze v6.3; report the panel.

### Wave 4 — F13: splittable clustering (correlation-clustering local search)
Extend `_refine` (`tracklet_graph.py:891`) — no ILP solver dependency:
chimera-veto LLR injection from F9 purity report (override within-cluster pair LLRs to −6.0),
swap moves + explicit bipartition move seeded at the most-negative internal pair, accept on
objective gain ≥ `graph_split_min_gain`, deterministic pass caps; optional re-lift between
passes so splits and F11 re-merges cooperate. Gate: cycle-consistency ≥0.85 floor, purity flags
down, agreement non-regressing, distinct IDs must not explode (F6/F12 absorb fragments) —
panel read jointly.

### Wave 5 — decision-gated extensions
- **F14 pretrained ReID cue** — DEMOTED: `wip/recommendations.md` rejects a trainable ReID
  (no identity GT; geometry + pose-shape carry the signal). Revisit only if the descriptor path
  (F11/F12/F6b) plateaus with `_7` agreement <0.65, and only as a *pretrained, self-calibrated*
  cue with explicit user sign-off; preceded by web research.
- **F15 3D-informed P4 costs** (pelvis-height continuity + 3D shape distance in Stage-2/re-entry
  from lift3d) — full 3D KF only if teleports persist after F10.
- **F16 single-view PnP lift** for single-cam 3D completeness — after association gains plateau.
- **F17 P2 OC-SORT modules** (Kalman isolated in `scripts/tracking/kalman.py`, low-friction) —
  only if per-camera fragmentation still dominant. CMC skipped (static rig).
- **F18 detector/P1 evaluation block** (user: in scope, local): one dedicated block after
  Wave 4 — RTMO-l (weights + env already on disk) vs RTMDet-m, optional RTMDet-l/RT-DETR weight
  download, plus per-camera `bbox_thr` sweep, batched into one local P1 re-run set (overnight);
  umpire-recall probe on 1–2 hard deliveries first. Any accepted P1 swap triggers a full
  rebaseline (v6.4); v6.0-vs-v5 panel diff also feeds `wip/model_comparison.md` (X-vs-L study).

### Wave 6 — role-focused output shaping (user directive; LAST wave)

The production output prioritizes **bowler, striker, non-striker, wicketkeeper**. When
low-confidence poses / low-confidence tracking of peripheral identities (part-visible
umpires, distant fielders — today often extrapolated from priors) measurably hinder the
final output (identity churn, chimera fuel, render noise), suppress rather than guess:

- Per-identity output confidence score (pose confidence x track stability x view count);
  below a floor, drop the identity from the emitted streams/render (config-gated tiers:
  core roles always kept, fielders/umpires droppable).
- Skeletal-prior extrapolation (T-2) disabled for droppable identities — no fabricated limbs
  on peripheral players.
- Role source: P5 roles (fixed H2 sign + fused-track caution) + F5 online proxy.
- Includes the deferred H1 (P5 direction inference from fused tracks) since roles become
  load-bearing here.

### FR — external review fixes (landed mid-campaign)

See [review-triage.md](review-triage.md): C1/C2 (P2 fragmentation producers), C3/C4 (Singer
role-switch), C5 (single-cam emit), C6 (frame-aware fills), H2/H3/H5/H6/H7 + minor. C1-C3,
H2/H3/H5/H6 are unflagged BUG FIXES — flags-off byte-identity vs v6.0 is intentionally
retired from the FR batch onward; the grand-analysis best stack re-baselines.

### DONE (2026-07-11): grand analysis concluded, v7 cut
rc2 accepted as default (see fixes-log GRAND ANALYSIS CONCLUSION); `configs/v7/` +
driver defaults ship the new logical order. Remaining: Wave 5 probe, Wave 6, open list
in `wip/changes_tbd.md`.

### Final deliverable ordering (user directive)

After Waves 0-4 + FR: the grand cross-wave analysis composes the best stack, A/Bs it as one
run, cuts `configs/v7/` + driver defaults so the NEW logical order (P1 -> P1.5 -> P2 -> P3 ->
P3.5 -> P4 -> P5 -> 3D -> render) is the followed default, and updates `phases.md` to mark
it current. Only then Wave 5 probes (tiled detection) and Wave 6.

### Parked (logged in fixes-log with rationale)
Detector fine-tune (needs labels+training), self-supervised association (research-scale),
SmoothNet (marginal after P1.5), GT labeling → MOTA/IDF1/HOTA (needs human labeling — flagged
to user as a parallel workstream; proxy panel remains the gate until then).

### Deep-research checkpoints (user-invited)
Targeted web research before: F11 (latest view-invariant body-shape/ReID cues for identical-kit
sports, 2025–26), F13 (correlation-clustering/multicut local-search practice), F14 (ReID model
choice), F18 (detector SOTA for small/dark persons). Findings cited in fixes-log entries and
`references.md`.

## Verification (end-to-end)

1. Per fix: pytest (new tests per fix: cheirality synthetic, per-measurement-R, lift grouping,
   split move determinism, driver stage-chaining smoke) + flags-off byte-identity check.
2. Per A/B: full 8-delivery panel vs frozen baseline via driver `--baseline`; joint read;
   collisions 0 invariant.
3. Checkpoints (v6.1, v6.2, v6.3, …): report the full 8-delivery panel with baseline deltas
   in-chat and in fixes-log.md — data numbers only. **Mosaics are rendered only when the user
   explicitly asks** (the v6.0 ground mosaics are the one standing request).
4. M2 teleport-count verdicts always carry the proxy caveat (single-camera foot noise, per
   `wip/methods_log.md`); pair it with trajectory-jitter p95 when judging.
5. Final: user watches mosaics (on request) and gives the human verdict — the final judge.

## Critical files

New: `scripts/pipetrack/run_full_pipeline.py`, `configs/v6/*`, `configs/experiments/*`,
`scripts/association/cluster_lift.py`, `docs/critical-analysis/fixes-log.md`.
Modified: `scripts/export/triangulate_predictions.py`, `pose_estimation/triangulation.py`,
`pose_estimation/cricket/{geometry.py,ground_kalman.py}`, `scripts/association/{tracklet_graph.py,
associator.py,jsonl_io.py,cue_calibration.py}`, `scripts/global_id/{jsonl_io.py,track_manager.py,
global_track.py,stitching.py}`, `scripts/tracking/kalman.py` (Wave 5), `scripts/roles/assigner.py`,
`scripts/visualization/mosaic_layout.py`.
