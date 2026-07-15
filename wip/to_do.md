# To-do — consolidated backlog

Single source of truth for everything deferred, parked, or pending. Supersedes and folds in the
former `docs/changes_tbd.md` (the C1–C10 fix list — renumbered A1–A10 here — + restructure notes) and `remaining-work.md` (the v6→v8.1
campaign backlog), plus the 2026-07-15 restructure follow-ups. Current state:
`.claude/context/01-current-state.md`. History: `docs/pipeline/fixes-log.md`, `docs/runs/`,
`docs/diagnosis/`.

**Working standard for anything here:** flag-gated, flags-off byte-identity proven by execution,
A/B on all 8 benchmark deliveries (or 40 for a production claim), accept only significant generalised
improvement, one `fixes-log.md` entry per landing.

Legend: 🔴 high impact · 🟡 medium · 🟢 low/cleanup · (S)mall/(M)edium/(L)arge effort.

---

## A. Pipeline / algorithm — the quality levers

### A0 🔴 (L) decide-in-3D + 3D-coverage recovery  *(the current top item)*
The single-triangulation rewire is structural only: 04 triangulates once (binding-keyed), 05/06
carry the 3D forward but do **not** consume it. Two coupled pieces:
- **Consume the 3D in 05** behind `--track-in-3d` (default OFF = today's ground-plane path): feed
  04's `lift3d.jsonl` `pelvis_ground_xy` as the measurement and `pelvis_cov_m2` as the Kalman R
  (uncertainty-aware), add a 3D pose-shape re-ID for re-entry, and use the per-cluster
  reprojection/cycle-consistency as a chimera-split signal. (`05-global-id.md` fixes #1+#3; `04-lift.md` fix #1.)
- **Coverage recovery**: measured `tri_cov` dropped v8→v9 (M1_1_14_1: v8 terminal 0.817 → v9 04 0.566)
  mainly because we dropped the global-keyed terminal re-triangulation (it pooled more views per
  final ID). Recover by re-triangulating per `global_player_id` after 05 (what `07_lift3d` did) or by
  the 3D-pooling above — NOT by the nullable-joint change (that didn't move coverage; feet are
  prior-filled). A/B on 8 then 40; adopt only on a generalised gain.

### A1 🔴 (S) Fix the teleport metric + verdict — stop mislabeling deliveries
Every `fail` is `teleport>60` computed on **raw bbox-bottom foot projections averaged across cameras**
(single-cam grazing noise; `M2_2_4_1` is 0.992 agreement yet fail). Add a velocity-gated teleport
metric on the **emitted** `ground_tracks.jsonl`, multi-camera segments only; demote the raw proxy to a
tripwire; re-tune id-overmint to the ~13 roster. `src/identity/common/metrics.py:293`;
`src/identity/p5_global_id/runner.py:414-447`. (`diagnosis/03`)

### A2 🔴 (S) Stop emitting `np.mean` of multi-modal fragment positions
The 1528 emitted big jumps come from averaging two far-apart observations sharing one ID in a frame.
At `runner.py:339-349`, when an `(id, frame)` has ≥2 points spread > ~2 m, emit the Kalman posterior
(or nearest-to-prior), not the mean; log a `split_observation`. (`diagnosis/04`)

### A3 🔴 (M) Per-id velocity gate on the emitted track + single-cam damping
Reject/hold emitted points implying > ~10–12 m/s on multi-cam segments; on single-cam segments damp
toward the posterior (the foot ray carries ~1 m grazing error); tighten the χ² gate / process noise.
`runner.py:110-141`, `configs/05_global_id.yaml`, `ground_kalman.py`. (`diagnosis/04`)

### A4 🔴 (M) Depth-aware association weighting for grazing/facing cameras
Split identity: cam_04 (end-on) + cam_07 (panoramic) fail to bind. Weight the ground-distance LLR by
each camera's calibrated depth-uncertainty and up-weight the triangulation-consistency (union-lift)
cue. `src/identity/p3_association/` (tracklet_graph + associator), `configs/03_association.yaml`. (`diagnosis/05`)

### A5 🟡 (S) Depth-aware colocated-merge radius (replace flat 0.75 m)
Make `colocated_radius_m` a function of the projecting camera's depth-uncertainty (disjoint-camera +
posture guards kept). `stitching.py:328` (`merge_colocated_ids`). Clears the 2 residual coloc pairs
(`M1_1_14_7`, `M2_1_11_3` in the v8 production tree). (`diagnosis/05`)

### A6 🟡 (S) Cap the cross-space stitch budget (absolute metres, not gap-scaled)
`distance ≤ v_max·gap·slack` grows with the gap → a long occlusion licenses a cross-field stitch. Add
an absolute metres ceiling; emit a per-merge `seam_distance` flag. `stitching.py:139-226`. (`diagnosis/06`)

### A7 🟡 (M) Lock global id per P2 tracklet, not per frame
517 flicker events are intra-tracklet global-id flips. Once a `(camera, local_track_id)` binds with
confidence, hold it for the tracklet's life; apply remaps as a whole-tracklet relabel. P3 membership +
`runner.py` relabel. (`diagnosis/07`)

### A8 🔴 (L) F16 single-view PnP lift — raise multi-camera coverage
Single-camera frames get no 3D (coverage 0.23–0.84). Fit the identity's canonical skeleton (bone
lengths from its multi-view frames) to a lone 2D view, PnP-style, with honest covariance → 3D on
single-cam frames, ground position physically constrained (damps teleports). New module consuming the
04 lift. The structural lever behind coverage/single-cam. (`diagnosis/08`)

### A9 🟡 (M) Detection recall on the deep field / small subjects (F18)
Upstream cause of single-camera players. Probe 3×2 tiling and stronger detectors (YOLO26-l, RF-DETR)
through `tools/detector_bakeoff/`; recall-oracle on `M2_2_3_*` + cam_07 first. (`diagnosis/09`)

### A10 🟢 (S) Per-frame coverage/confidence flag
Emit a per-frame coverage field so consumers interpolate/hide gaps instead of seeing a hard blink.
(pose_3d is already per-joint nullable; this is a per-frame summary.) (`diagnosis/08`)

### A11 Pack-handling for in-pack peripherals — the quality floor
`_6` is weakest (agreement 0.527). Close catchers enter P3 as low-parallax tracks. Ideas: pack-aware
clustering (spatial pack = one super-cluster, joint membership), per-pack chimera passes, stricter
peripheral birth in packs.

### A12 Flag-gated OFF, awaiting A/B (implemented, dormant)
- **G1 Hartley conditioning + G3 parallax-ordered RANSAC** (`triangulation.py` `hartley=`,
  `parallax_order=`; CLI on `run_triangulation.py`). A/B reproj p95 without coverage loss.
- **Airborne pelvis-emit (V2-L3)** — `associator.py::_triangulated_pelvis_xy` behind
  `airborne_pelvis_emit`; emit-only. A/B teleports/jitter on jump clips.
- **`density_lost_window`**, **cross-delivery prior calibration (F8)**, **`temporal_link_decay` (H4)**.
  Keep OFF (documented harmful): `posture_keep_upright_unknown` (H3), contested-camera machinery.

### A13 Other deferred experiments (decision-gated)
F15 (3D-informed 05 costs — subsumed by A0), F17 (OC-SORT in P2, only if fragmentation resurfaces),
skeleton-gait embedding cue (needs sign-off), G4 inter-cue LLR correlation, per-track standing height
(v1 ISSUE-8), A1–A7 maintainability refactors (library extraction, config dedup, god-config split).

---

## B. Runs / verification pending
- **Full 40_full run** — `raw/40_full` symlinks are set up + verified on L40S; needs P1 (26-kpt) +
  the pipeline on all 40, then the panel.
- **decide-in-3D A/B** (pending A0), on 8 then 40.
- **Identity ground truth → IDF1/MOTA/HOTA** — `metrics.py::evaluate_ground_truth` is ready; label a
  few hundred frames on 2–3 hard deliveries (`_7`, `M2_1_12_1`, `_6`). Until then tuning is proxy-guided.
- **Mosaic sign-off** — arbitrate roles end-orientation (striker vs non-striker end confirmed only on
  `_2`) and keeper-pick ambiguity (P002-vs-P003 class); spot-check more clips.
- **Vedant's `global_id/` rewrite (parked, needs sign-off)** — a parallel `track_manager` lineage
  (differs 1000+ lines). His `roles/` contribution was merged (roles v1/v1.2); the global_id rewrite
  must NOT displace the validated stack without his changelog, per-change flag-gating, and the
  standard 8-delivery A/B. Removed from the tree in the restructure; recoverable from git / his branch.

## C. L40S / infrastructure
- **Physical consolidation** (deferred, needs sign-off): `pose_data` (85 G), `events_data`,
  `pipetrack_v8` (25 G), the old clone's `benchmarks/runs` + `artifacts` still sit in `~/`; referenced
  by symlinks, not moved. Any move is **copy → verify → then delete**, never `mv`.
- **Delete `~/pose-estimation-benchmark`** (14 G incl. 7.3 G bloated `.git`) once everything is verified.
- **Loose-item decisions**: the duplicate `scripts/roles/roles_distance (1).py` + the git stash
  `box-local-w9w10-pre-ff-pull` in the old clone.
- Done this session: `raw/40_full` symlinks · `.claude/context` updated (local + L40S) · `pose-lab`
  env (rename) · `#9` stale-file cleanup · `~/pipetrack` clone.

## D. Docs — long-tail sync
Core done (shared-data, architecture, getting-started, index, README, cli, 00-inference I/O+flowchart,
04/05/07/meeting-debug, new skeleton-halpe26 + localization-error). **Still stale** (old paths /
`07_lift3d` / COCO-17 / `pose_*_native` / old repo name / pre-`00_inference` layout):
`docs/pipeline/{01,02,03}-*.md`, `pipeline/README.md`, `references.md`, `00-inference.md` narrative
(§60-62 / Issues P1-5 / Fix 4), `docs/reference/{configuration,metrics,data-inventory}.md`,
`rtmpose-x-runbook.md`, `CHANGELOG.md`, `CONTRIBUTING.md:16`, `docs/diagnosis/README.md:29`.

## E. Consumer / schema / output
- **`g1_player_frame/v1` (Halpe-26) is consumer-facing** — the character team accepted it; coordinate
  any further change with biomechanics + officiating. `pose_3d` is now per-joint nullable.
- **UE packet export** (`identity/export/export_ue_packets.py`, retargeted to the 06_roles run-dir) —
  run per delivery when UE-cm packets are needed.
- **All-40 mosaic batch** — render on demand (~3 h, 2 parallel).
- **v8-selected mosaics** are v8-rendered (17-joint overlay); a 26-joint re-render is optional.

## F. Code hygiene / performance / metrics debt
- **Performance (flat profile)**: JSONL parse/write (orjson-class, ~15–20% of P3 wall);
  `observe_frame` per-frame pair loop; emit-side pose-descriptor bookkeeping (18.8 s/delivery);
  box `--jobs` tuning (probe `--jobs 5` prefetch 2–3 vs oversubscribing 8 vCPUs).
- **Metrics debt**: union-lift synthetic unit test (integration-proven only); swap-event
  `--dump-frames` crop emission (`diagnose_colocated_ids.py`, done manually, not wired); the cleaner
  teleport metric (= A1).
- **YOLO/generic P1 path**: `phase1_outputs.coerce_coco17_keypoints` pads to 26 (feet = 0) under the
  halpe26 contract — fine for the benchmark-only YOLO path; RTMPose-x emits real 26.
- Done this session: secondary de-hardcoding (`run_phase1_{l40s,parallel}.py`, `id_pipeline.py`,
  `tools/diagnosis/*` → env-overridable) · `tests/test_main.py` (compute-loop coverage) · the
  `08_render` AssertionError fix.

## G. Explicitly deferred by the user
- **Laptop conda recovery**: `balltrack`/`quadruped` are safe & runnable at `~/anaconda3_quarantine/envs/`;
  recover by registering that dir or recreating from `~/conda_env/2026-06-06/*.yml` (+ `kratos.yml`).
- **`presentables/`** refresh — out of scope for the restructure; still un-updated.
- **`.git` bloat** (7.5 G committed mosaics/docx in history) — accepted; history-rewrite/shallow declined.

## H. Measured reference facts (don't re-derive)
- **Reprojection**: panel (RANSAC-inlier, pre-smoothing) 3.07–3.56 px; post-smoothing vs all confident
  2D views mean 6.8 / p95 24.5 px; hips worst (11–12 px, cross-view keypoint-definition), camera spread
  mild → calibration healthy. The "1 px" applies to calibration targets, not pose-model keypoints
  (own 2D noise 2–3 px).
- **px → ground metres**: `docs/reference/localization-error.md` (anisotropic: cross ~3 mm/px, depth
  25–60 mm/px at 5–9° grazing).
- **Calibration**: one session for both matches (team-confirmed); cm-accurate.
- **Box-vs-local panel variance is expected, not a bug**: the same code on the two P1 binaries (the
  L40S fp16 fast path vs the laptop generic path) gives slightly different panels on the 8 overlap
  deliveries — e.g. `_7` local 0.962 agreement / 10 ids vs box 0.819 / 12. Same algorithm, different
  detector/pose numerics upstream; treat it as a variance class, don't chase it as a regression.
