# legacy_code.md — legacy / dead / superseded code inventory (handover audit, 2026-07-20)

Everything in `src/` (+ `configs/experiments`, `tools`, `tests`) that is **not on the active
production path** but still lives in the tree. For each: what it is, what it superseded / was
superseded by, whether it is truly dead vs off-but-reachable, and a **disposition**
(SCRAP / RELOCATE / KEEP) with a one-line reason. Final scrap/relocate calls are the team's — this
is the decision-ready inventory.

Active path = `src/main.py` + `configs/*.yaml`. Reachability confirmed by grep for importers/callers.

---

## A. DEAD — truly unreachable (no importer, no caller, not selected by any config)

### A-1. `src/identity/p6_roles/realtime_bowler_tracker.py` (whole file, ~202 lines)
- **What:** standalone real-time bowler-lock state-machine prototype (own SVD triangulation, own y-history heuristic). Superseded by `p5_global_id/role_proxy.py` (`OnlineRoleProxy`) feeding the v1 epoch role solver.
- **Reachability:** truly dead — `grep -rn "realtime_bowler\|RealTimeBowlerTracker"` over the whole repo returns only the file itself; `p6_roles/__init__.py` exports nothing.
- **Also wrong if run** (see BUG-2): triangulates normalized coords through pixel projection matrices; hardcodes COCO-17 `range(17)`; `__main__` uses random projection matrices; docstrings falsely claim "PRODUCTION PIPELINE."
- **→ SCRAP.** Zero references and incorrect; its function is fully covered elsewhere.

### A-2. `src/identity/visualization/animation_viz.py` (whole file)
- **What:** 3D matplotlib animation dev script (module-level `TARGET_PLAYER="P001"`, `FLIP_FORWARD`).
- **Reachability:** not imported/wired anywhere; `plt.cm.get_cmap("tab10")` (`:101`) is removed in matplotlib ≥3.9 and a hardcoded Windows path sits in `__main__` — it would crash on the current stack.
- **→ SCRAP** (or RELOCATE to a `scratch/` if anyone still uses it interactively, but it's broken as-is).

### A-3. `triangulate_legacy` + flat-JSONL arg group — `src/identity/p4_lift/run_triangulation.py:526-595` (+ args `:75-78`)
- **What:** pre-restructure code path that triangulated from flat `--predictions/--calibration/--output` files instead of the run-dir layout. Superseded by `triangulate_canonical_run` (run-dir layout).
- **Reachability:** runs only when all three flat flags are passed; `main.py` always uses run-dir layout + `--id-source binding`. No other caller found in repo. Also calls `triangulate_skeleton_ransac` with **no cheirality/smoother** — a strictly worse algorithm than the canonical path.
- **→ SCRAP** the function + its arg group (no external flat-JSONL feeder found). Team confirms no off-repo script relies on it.

### A-4. `velocity_toward_crease` — `src/identity/p5_global_id/global_track.py:163-168`
- **What:** self-described "Dormant P5 hook"; computes crease-approach velocity. No caller.
- **→ SCRAP** (dead method) — or KEEP only if a near-term role/impact feature will consume it; otherwise it's noise.

---

## B. OFF-BUT-REACHABLE — gated alternatives / fallbacks (selected only by non-default config)

### B-2. OC-SORT tracker path — `src/identity/p2_tracking/` (tracker.py OCM `:111-122`, OCR `:231-234`; kalman.py ORU `:101`; track.py `_obs_history`/last-obs `:63-216`; config.py `ocm_*`/`ocr_enabled`/`oru_enabled` `:57-96`)
- **What:** Observation-Centric SORT mechanisms. Production `tracker: bytetrack`; all OC-SORT code guarded by `if config.tracker == "ocsort"` so bytetrack is byte-identical.
- **Reachability:** reachable only via `configs/experiments/02_tracking_ocsort.yaml` (docs-only; never referenced by main/tests). `_obs_history`/`_last_obs_bbox` are *populated* for every track but only *read* under ocsort.
- **→ KEEP** (coherent gated A/B alternative, low risk). If bytetrack is final, RELOCATE the experiment config + document the branch as an experiment.

### B-3. Per-frame association mode + per-frame entry points of `associator.py`
- **What:** `association_mode: per_frame` historical per-frame clustering (`associate_frame`, `_associate_pairwise_anchor`, `_associate_multiway_cycle`, `select_anchor`, `TemporalLinkMemory`, epipolar/Sampson scoring). Superseded by `tracklet_graph`.
- **Reachability:** off-but-reachable. **Do NOT delete `associator.py`** — its `Detection3`/`Correspondence` dataclasses + cost/geometry helpers are shared by the *active* graph path and by `p5_global_id`. Only the per-frame entry points are dormant. Callers: `runner.py` (both branches), `tests/test_cricket_association.py`.
- **→ KEEP the file; fix the misleading dataclass default** (`config.py:95` default `per_frame` contradicts production `tracklet_graph` — flip or annotate).

### B-4. Roles v0 heuristic solver — `src/identity/p6_roles/assigner.py::assign_roles` (`:76-160`)
- **What:** original positional/kinematic role heuristic. Superseded by `assign_roles_epoched` (v1). Production `role_assignment_version: v1`.
- **Reachability:** off-but-reachable — `run_role_assignment.py:151` calls it only in the non-v1 branch (omit `--config` or set v0). Docs/CLI advertise "omit --config for legacy v0."
- **→ KEEP** as documented v0 fallback, **or SCRAP** if the team commits v1-only (then also drop the `v0` dataclass default). Team decision. Note the signed/`abs()` comment mismatch (BUG-15) lives here.

### B-5. EMA temporal smoother — `--smoother ema` path (`p4_lift/run_triangulation.py:337 confidence_ema_smooth`, helper in `common/triangulation.py`)
- **What:** causal EMA smoothing of lifted 3D, explicitly labelled "(legacy)". Superseded by Butterworth.
- **Reachability:** off-but-reachable — `main.py` always passes `--smoother butterworth`; ema only fires if run_triangulation is invoked directly without the flag. But the **module's own default is still `ema`** (`:84`).
- **→ KEEP** (cheap documented fallback) **but change the CLI default to `butterworth`** to match production/main.py.

### B-6. Robust/Huber IRLS refit — `--robust-refit` (`run_triangulation.py`, `common/triangulation.py` IRLS path)
- **What:** iteratively-reweighted per-joint triangulation polish. Off = byte-identical baseline; never adopted.
- **→ KEEP** as a gated experiment flag (documented, off by default). Low risk.

### B-7. In-file "legacy / byte-identical" reproducibility knobs (NOT separate files)
Consciously-kept config-selectable switches left in place for reproducibility — flag for a possible
config-schema cleanup pass, but not dead code:
- `p3_association/config.py`: `foot_contact_mode: "legacy"` default, historical emit-source modes ("median").
- `p5_global_id/config.py:54-326`: many "0/False = legacy byte-identical" gates; `runner.py:778` "Legacy teleport-proxy rule (recoverable for reproducibility)".
- `p4_lift/run_triangulation.py:113` `--native-skeleton` "(Deprecated no-op)"; `:89-92,271-278` `--hartley`/`--parallax-order` (G1/G3 A/B, not wired from main.py — cannot fire).
- **→ KEEP** unless the team wants a deliberate config-schema slimming; document that these are reproducibility switches, not live tuning.

---

## C. SUPERSEDED DRIVERS — still runnable, but the whole-pipeline role moved to `main.py`

### C-7. `src/identity/id_pipeline.py` — the inner-loop (03+05 only) batch driver
- **What:** standalone driver that re-runs only association (03) + global-id (05) across the 8 deliveries, with its own arg parser, panel, and `main()`. `src/main.py` is the superset full-chain driver (00→08).
- **Reachability: split.** `main.py` **imports 4 helpers from it** — `ALL_DELIVERIES`, `_dig`, `_fmt`, `_run_stage` (`main.py:54`) — so the file must stay. But its *driver* surface (`main()`, `run_delivery()`, `build_arg_parser()`, `read_panel_row()`, `print_panel()`, its own `PANEL_COLUMNS`) is superseded by richer equivalents inside `main.py` and is only reachable via `python -m identity.id_pipeline`.
- **→ RELOCATE the 4 shared helpers** into a small `identity/_driver_common.py` (or `core/`), then **SCRAP** id_pipeline's duplicated driver/panel code. Leaving it as-is means maintaining two parallel `main()`/panel implementations.

### C-8. P1 inference variants — `src/core/inference/`
- `run_phase1_rtmpose_inference.py` — **ACTIVE** repo-local runner (docs + `detector_bakeoff` import `DETECTOR_PRESETS`). KEEP.
- `run_phase1_l40s.py` — **ACTIVE** remote-GPU runner (all GPU runs go to L40S; runbook + `tools/run_v8_l40s.sh`). KEEP.
- `run_phase1_parallel.py` — thin sharding wrapper fanning out `run_phase1_l40s.py`. Reachable but **not called by main/tests/tools**; docs note a previously-broken runner path now fixed (dry-run only). **→ KEEP as an operational utility, but confirm it's still used** — weakest-verified of the three.
- `phase1_common.py` — **ACTIVE** shared building blocks. KEEP.
- Note: `main.py` never invokes P1 — it consumes pre-written `00_inference/`. All P1 runners are out-of-band manual drivers.

---

## D. EXPERIMENT CONFIGS — off-path by design (referenced only in docs)

### D-9. `configs/experiments/02_tracking_ocsort.yaml`
- Selects the OC-SORT branch (B-2). Docs-only. **→ KEEP** as the paired A/B config, or RELOCATE/scrap together with the OC-SORT branch if bytetrack is final.

### D-10. `configs/experiments/05_global_id_presmooth.yaml`
- Near-duplicate of production `05_global_id.yaml` (`presmooth_ground_enabled: true` + `online_role_proxy: true`; header still says "v8.1 DEFAULT p4"). Docs-only, largely overlaps production. **→ VERIFY it still differs meaningfully; if it has converged onto production, SCRAP** as dead weight.

---

## E. INERT — computed and threaded through, but the result is never read

The most insidious category: code that *runs* every delivery but whose output nothing consumes. Not
"off", not "dead" — wired and executing, just unread. (Confirmed against the July-17
`this file` register + re-verified this pass.)

### E-11. The entire fine-score cue-calibration subsystem (p3)
- **What:** an older cross-camera scoring scheme — `mu_fine_score`, `sigma_fine_score`, `w_epi`, `w_tri` (`config.py`); `CalibrationStats` (`cue_calibration.py`); `GeometryCache.stats`, `GeometryCache.huber_delta`, `PairGeometry.w_epi/w_tri/huber_delta` (`geometry_cache.py`); `config.huber_delta()`.
- **Status:** **inert.** Computed and threaded through the whole geometry cache, but `build_cost_matrix` recomputes its weights from `pg.is_degenerate` and never reads any of it. A ~whole legacy scoring scheme left wired but unread.
- **→ SCRAP** (remove the fields + `CalibrationStats` + the huber/w_epi/w_tri plumbing) once someone confirms `build_cost_matrix` is the only consumer — it is, per the register. Biggest single de-slop win in p3.

### E-12. `coco17_indices` / `select_coco17_pose` threading (P1)
- **What:** `player_records` receives `coco17_indices` and calls `select_coco17_pose`, but always emits the model's native (Halpe-26) keypoints unsliced.
- **Status:** **inert** (kept as the documented COCO-17 slicing reference). A non-Halpe model would fail contract validation anyway, so P1 is Halpe-26-only in practice.
- **→ SCRAP or RELOCATE to a docstring** — the live path never slices.

### E-13. Visualization inert bits
- `ROLE_TAGS` (`overlays.py:46`) — roles shown only in the roster panel by design; kept as the single place to change on-screen role wording. **KEEP (documented) or fold into the roster code.**
- `seen_here_nearby` inner check (`render_videos.py` mosaic loop) — deliberate no-op retained with its intent comment (lost ghosts shown regardless within the decay window). **KEEP or delete the dead branch.**
- `draw_players(roles=...)` param (`overlays.py:390`) — accepted, never used. **SCRAP the param.**

### E-14. Global-id ownership revive no-ops
- `_revive_owned_track(...)  # no-op if already active` (`track_manager.py:429,465,548`) — called defensively where the track is usually already active. **KEEP** (cheap guard, correct) — noted only for completeness.

---

## F. DEAD HELPERS & CONFIG FIELDS — no caller / no reader anywhere in src|tools|tests

### F-15. Dead functions in `common/geometry.py`
- `condition_number_dlt` — no callers. **SCRAP.**
- `ground_point_and_cov` — no callers. **SCRAP.**
- `huber_cost` — no callers. **SCRAP.**
- `parallax_weight` — imported by `associator.py` but never called. **SCRAP** (+ drop the import).
- `fuse_ground_estimates` — internal-only (called only by `robust_fuse_ground`). **KEEP** (real internal helper).

### F-16. Dead config fields in `p3_association/config.py`
- `cycle_xy_tol_m`, `dummy_cost_scale`, `parallax_min_deg`, `parallax_full_deg` — **dead**, no code reads them anywhere in `src/`. **SCRAP** from the dataclass + YAML.
- `image_w`, `image_h` (default 2560×1440) — **dead**: `load_image_sizes_from_drive` supplies each camera's true native size (incl. cam_07's ~3776×960) into the epipole test, feet check, and `keypoints_norm`; the `config.image_w/h` defaults have no consumers. (Was NB-1 "not-a-bug" in the pipeline tracker; recorded here as the dead-config item it actually is.) **SCRAP** from the dataclass + YAML.
- **Effort:** all F-16 fields **XS** (grep-confirm no reader, then delete).

### F-17. `src/core/schemas.py` — whole module unimported
- `CameraCalibration`, `PosePacket` dataclasses — **no importers** found in `src/`, `tools/`, or `tests/`. Appears to be an aspirational export-packet schema for the graphics layer.
- **→ RELOCATE** (to a clearly-marked `contracts/` or the export package) **or SCRAP** if the UE export path doesn't intend to adopt it. Do NOT leave it looking like a live core schema.

### F-18. `detect_person_boxes` (single-image path, `phase1_common.py`)
- **near-dead** — only used as the fallback inside `detect_person_boxes_batch` (the silent bare-except, `resolvebugs.md` BUG-list). **KEEP** (it is the fallback body) but fix the silent except.

### F-19. `run_triangulation.py:33-38` `_most_complete_pose` empty-guard returns `(17,3)`
- Returns a 17-row array only on empty input "callers never hit"; wrong width vs the 26-row pipeline. Effectively unreachable. **KEEP + fix width to (26,3)** for consistency, or drop the guard.

### F-20. `global_track.py:163-168` `velocity_toward_crease`
- Already listed A-4; self-described "Dormant P5 hook," no caller. **SCRAP.**

---

## G. DEPRECATED ALIASES & CLI COMPAT — kept for interface compatibility only

### G-21. Stage-05 config aliases `P4Config` / `P4AConfig` / `P4BConfig`
- Compatibility aliases for the stage-05 names (`GlobalIdConfig`, `GlobalTrackingConfig`, `StitchingConfig`); the YAML loader still accepts old `p4a:`/`p4b:` section spellings.
- **→ KEEP short-term** (old run manifests use them) **then SCRAP** with the P4/P5/P6 → 04/05/06 naming cleanup (`resolvebugs.md` stage-label skew).

### G-22. Legacy verdict rule — `usability_verdict: false` path (p5)
- Teleport-proxy grading superseded by the usability rubric (`runner.py:778` "Legacy teleport-proxy rule (recoverable for reproducibility)"). **KEEP** as reproducibility switch, off-by-config.

### G-23. "accepted for CLI uniformity; unused" args
- `--drive-root` / `--expected-frames` on `run_stabilization.py:32-33`; `--expected-frames` on `run_refinement.py:34`. Accepted so every stage CLI is uniform, but unused by those stages. **KEEP** (harmless uniformity) — documented here so a reader doesn't hunt for their effect.

### G-24. `p6_roles` `posture_keep_upright_unknown` (H3), p3 `temporal_link_decay`, `matching_mode: pairwise_anchor`
- Off-by-config reproducibility/A-B sub-branches of already-listed legacy paths (per-frame engine, posture policy). **KEEP** with the parent path (B-3 / B-4).

### G-25. Visualization duplication-by-design
- Two bird's-eye renderers: cv2 tile (`panels.draw_bev_panel`, in the mosaic) vs matplotlib tool (`render_bird_eye_view.py`, standalone diagnostics) — different outputs, only exact-duplicate helpers were merged. **KEEP.**
- `draw_info_panel` (`panels.py`) — **dead** (bird's-eye tile replaced the text monitor tile); kept as the generic panel primitive. **KEEP or SCRAP** (team call).

---

## Prior authoritative register

`this file` (2026-07-17 audit) is the existing register; this file
**supersedes and extends it** (adds the drivers, experiment configs, the always-fallback framing, and
per-item SCRAP/RELOCATE/KEEP dispositions). Keep them in sync, or fold the docs one into a pointer here.

---

## Suggested handover disposition (one screen)

Effort: **XS** <30 min (grep-confirm + delete) · **S** ~1–2 h (subtractive across files + run tests) · KEEP items have no effort. Dead-code deletion needs no A/B — it can't change output; verify by running `tests/` + one delivery.

| Item | Disposition | Effort | Reason |
|---|---|---|---|
| A-1 realtime_bowler_tracker.py | **SCRAP** | XS | dead + geometrically wrong + falsely labelled production |
| A-2 animation_viz.py | **SCRAP** | XS | dead + crashes on current matplotlib |
| A-3 triangulate_legacy | **SCRAP** | XS | dead, worse algorithm, no flat-JSONL feeder |
| A-4 velocity_toward_crease | **SCRAP** | XS | dead method, no caller |
| B-2 OC-SORT branch (+D-9) | KEEP (or relocate as experiment) | — | coherent gated A/B, low risk |
| B-3 per-frame association | KEEP file, fix default | XS (default) | dataclasses shared by active path; default mislabels prod |
| B-4 v0 roles | KEEP or scrap (team) | XS if scrap | documented fallback vs v1-only commitment |
| B-5 EMA smoother | KEEP, fix CLI default→butterworth | XS | cheap fallback; default disagrees with prod |
| B-6 robust-refit | KEEP | — | gated experiment, off by default |
| B-7 in-file legacy knobs | KEEP (optional schema cleanup) | S | reproducibility switches |
| C-7 id_pipeline.py | RELOCATE helpers + SCRAP duplicate driver | S | two parallel main()/panels to maintain |
| C-8 run_phase1_parallel.py | KEEP, confirm still used | — | operational, weakest-verified |
| D-10 presmooth experiment cfg | VERIFY then likely SCRAP | S | may have converged onto production |
| E-11 fine-score calibration subsystem | **SCRAP** | S | whole legacy scoring scheme wired but never read (biggest p3 de-slop) |
| E-12 coco17_indices/select_coco17_pose | SCRAP/RELOCATE | S | inert slicing; live path never slices |
| E-13 viz inert bits (ROLE_TAGS, seen_here_nearby, draw_players roles) | SCRAP param / KEEP documented | XS | unread |
| F-15 dead geometry helpers (condition_number_dlt, ground_point_and_cov, huber_cost, parallax_weight) | **SCRAP** | XS | no callers |
| F-16 dead p3 config fields (cycle_xy_tol_m, dummy_cost_scale, parallax_min/full_deg, image_w/h) | **SCRAP** | XS | no reader |
| F-17 core/schemas.py | RELOCATE or SCRAP | XS | whole module unimported |
| F-19 _most_complete_pose (17,3) guard | KEEP + fix width | XS | unreachable, wrong width |
| G-21 P4/P4A/P4BConfig aliases | KEEP then SCRAP w/ naming cleanup | S | old-manifest compat |
| G-22 legacy verdict rule | KEEP | — | reproducibility switch |
| G-25 draw_info_panel | KEEP or SCRAP (team) | XS | dead panel primitive |

**Total effort to clear the whole SCRAP list ≈ half a day** (all XS/S, no A/B — dead code can't change output). This is the single biggest handover-readability win and the safest change in the whole plan.

**Also reconcile the default-vs-production mismatches** (listed in `resolvebugs.md`) while doing the
above — several live in these same files (`association_mode`, `role_assignment_version`, tri
`--smoother`, `lowconf_can_spawn`, `suppression_enabled`).
