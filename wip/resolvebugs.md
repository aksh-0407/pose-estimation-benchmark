# resolvebugs.md — open bug & error register, with fix + effort (handover audit, 2026-07-20)

Full-codebase audit of every `src/core` and `src/identity` script. **Every item here is OPEN in the
current tree** — bugs already resolved by the 2026-07-17 campaign have been deliberately excluded (they
are not the team's concern; history is in git). Each item carries a concrete **Fix** and an **Effort**
estimate so the next session can plan the cleanup.

Effort scale (the cost is almost always *verification*, not typing — this pipeline has no ground truth,
so any behaviour-affecting change is gated on an 8- or 40-delivery A/B + human mosaic review):

| Tag | Wall-clock | Meaning |
|---|---|---|
| **XS** | < 30 min | one-line / comment / dead-code delete; no behaviour change, no re-run. |
| **S** | ~1–2 h | localized code change + smoke test / single-delivery check. |
| **M** | ~½–1 day | behaviour-affecting; needs an 8-delivery A/B + panel read. |
| **L** | days | needs 8+40 A/B, retune sweeps, or a cross-stage change. |

Full sequencing and cross-references live in `remediation_plan.md`; the living open-bug tracker is
`resolvebugs.md` (IDs reconciled there).

---

## HIGH — correctness, load-bearing

### BUG-1 — `ground_contact_pixel_ex` shape guard rejects Halpe-26 → every ground point silently degrades to bbox-bottom
**`src/identity/common/geometry.py:188`** — `if bbox.shape != (4,) or points.shape != (17, 2) or confidence.shape != (17,): return bottom, 0.0, "bbox_bottom"`.
- The contract mandates 26 keypoints; P2 (`p2_tracking/jsonl_io.py:88-94`) and P3 (`p3_association/jsonl_io.py:92`, `associator.py:219,236`) all pass `(26,2)`. The guard always fails → bbox-bottom for every detection. Empirically verified. Kills `foot_contact_mode v2/v3`, ankle knobs, `single_cam_height_emit`, and the z0_reproj grazing correction; all P2/P3 grounds are the bbox bottom edge. No exception, nothing logged.
- **Fix:** make the guard length-aware (accept ≥17 rows; Halpe 0-16 = COCO-17, so ankle indices 15/16 stay valid). Gate the **emit path, clustering path, and tracking path behind three separate config flags** (all default off) and A/B each independently on the 8-set before any production enablement. Downstream thresholds (`clustering gates m`, global-id `r_ceiling_m`, `confidence_discard`) were tuned on bbox-bottom grounds and likely need a retune sweep after.
- **Effort: L.** The guard edit is XS; the cost is three separate A/Bs + a probable retune. **Owner decision first** — this is currently the report-only BUG-9/BUG-A1 by the 2026-07-17 owner call; every published baseline (0.689 verdict, all 40 numbers) was measured with bbox-bottom grounds, so "fixing" it moves the numbers. Reconfirm the decision before touching it.

### BUG-2 — `realtime_bowler_tracker.py` is a dead prototype, geometrically wrong if run, docstring claims "PRODUCTION"
**`src/identity/p6_roles/realtime_bowler_tracker.py`** (whole file). Not imported anywhere (grep-confirmed); superseded by `p5_global_id/role_proxy.py::OnlineRoleProxy` + `assign_roles_epoched`. If run: triangulates normalized coords through pixel projection matrices (`:174`), hardcodes `range(17)` COCO-17 (`:108`), `__main__` uses random projection matrices, docstring says "PRODUCTION PIPELINE".
- **Fix:** `git rm` the file (see `legacy_code.md` A-1).
- **Effort: XS.** Zero references; deletion is safe.

---

## MEDIUM — correctness (masked in production) or misleading docs on live code

### BUG-3 — P2-stab standalone runs silently ignore the tuned config (stale `DEFAULT_CONFIG` path)
**`src/identity/p1_stabilization/config.py:17`** — `DEFAULT_CONFIG = "configs/p1b_stabilization.yaml"` doesn't exist; `load_stabilization_config(None)` silently returns dataclass defaults. Docstring + CLI help claim the default is `configs/01_stabilization.yaml`. Masked in production (`main.py:403` passes `--p1b-config` explicitly).
- **Fix:** point `DEFAULT_CONFIG` at `configs/01_stabilization.yaml`, or make `load_stabilization_config(None)` raise.
- **Effort: XS.**

### BUG-4 — legacy (non-posterior) global-id emit path emits filter-rejected outlier feet
**`src/identity/p5_global_id/runner.py:349-357`** — appends `correspondence.ground_xy` whenever the track has finite ground, without checking whether the tracker fused that measurement; for identity-only outlier frames it would emit the rejected outlier. Masked because `emit_kalman_posterior=true` takes the posterior instead.
- **Fix:** delete the non-posterior branch (or gate it to skip identity-only frames). Tie to the teleport-metric work (A1) since it's the same emit path.
- **Effort: S** (code) — but re-verify emitted `ground_tracks.jsonl` unchanged with posterior on (should be, since the branch is currently dead).

### BUG-5 — `finalize()` end-of-delivery promotion bypasses shadow/roster gating (over-mint)
**`src/identity/p5_global_id/track_manager.py:669-677`** — any tentative with `hits>=2` is confirmed and minted with no `_confirmation_blocked` check; a late shadow duplicate can be minted as a fresh id at the tail. Partly masked by `min_emit_frames=30`. (= known-bugs BUG-10.)
- **Fix:** route `finalize()` through the same gates as `_promote_and_prune`.
- **Effort: M.** Verify id count + colocated-pair diagnostics on the 8-set (expect flat/lower, no new collisions).

### BUG-6 — P1 `--sweep`/`--grid` autotuner ignores the production detection path
**`src/core/inference/run_phase1_l40s.py`** (`run_sweep`/`_grid_sweep`, ~`:500-740`) always calls the plain-640 `detect_person_boxes_batch` and never wraps forwards in `_amp_context`, even with `--tiled-det`/`--amp`. So the "BEST" batch sizes + `projected_full_run_minutes` don't reflect the real (several×-heavier tiled) path.
- **Fix:** run the sweep through the same tiled+AMP code path the production run uses (share the detect closure).
- **Effort: S.**

### BUG-7 — resume + append leaves corrupt partial JSONL lines in the file permanently
Both P1 runners open in `"a"` on resume (`run_phase1_rtmpose_inference.py:449`, `run_phase1_l40s.py:807`); `read_resume_state` (`phase1_common.py:665-686`) counts a truncated last line as corrupt but doesn't add its frame to `done`, so it recomputes + re-appends while the corrupt line stays forever, re-counted every resume.
- **Fix:** on resume, truncate the file to the last newline-terminated record before appending.
- **Effort: S.**

### BUG-8 — `resolve_skeleton` is filename-token-fragile; a renamed checkpoint aborts P1 at frame 1
**`src/core/inference/phase1_common.py:468-484`** — skeleton inferred from config-path substrings (`halpe` before `coco/body8/body7`); a Halpe-26 config path lacking "halpe" resolves to `coco_17`, gets stamped, and `validate_group1_frame` aborts. Self-checking but brittle. (See `keypoint_contract_handling.md`.)
- **Fix:** resolve the skeleton from the model registry (`model_registry.yaml` `skeleton:` field) rather than the path string; emit an explicit "expected halpe26, got X" error.
- **Effort: S.**

### BUG-9 — `ground_kalman.mahalanobis_sq` docstring contradicts the production design
**`src/identity/p5_global_id/ground_kalman.py:120-125`** — docstring says gating R "must match" the update R; production deliberately violates this (gating uses role R, update uses measurement R — the anti-teleport asymmetry).
- **Fix:** rewrite the docstring to describe the asymmetric-R design.
- **Effort: XS.**

### BUG-10 — `suppress_peripherals` config defaults are defined twice (two sources of truth)
**`src/identity/p6_roles/suppress_peripherals.py:38`** duplicates the suppression defaults already in `p6_roles/config.py`. They agree today but will silently diverge.
- **Fix:** delete the module-level `DEFAULTS` dict; read from `config.py`'s dataclass.
- **Effort: XS.**

### BUG-11 — 3D-lift suppression is dead in the production (binding) path but the flag/help imply it's live
**`src/identity/p4_lift/run_triangulation.py:166-223`** — `suppressed_ids` consulted only in the `--id-source global` branch; production forces `--id-source binding`, so `--suppression-path` + the `06_roles/suppression.json` probe are inert.
- **Fix:** either apply suppression in the binding path too, or remove the flag + help + probe and document that lift precedes roles.
- **Effort: XS** (remove/doc) or **M** (wire it in + A/B, if suppression-before-lift is actually wanted).

### BUG-12 — new reprojection-error metric is oversold as a "fair" fidelity measure (uncommitted p7 work)
**`src/identity/p7_refine/runner.py:_reproj_errors` + `:315-318`** — restricting to reliable (conf≥`vis_conf`) views means raw triangulation already reprojects near-perfectly there, so the physics/hinge constraints will typically *raise* the after-error on well-seen joints. Honest refinement can register as a regression, opposite of the docstring claim. Also `reproj_sample_count` reports the before-count only.
- **Fix:** reword the docstring (the +1–3 px rise is the expected rigid-bone trade, not a regression); report both before/after sample counts.
- **Effort: XS.**

### BUG-13 — misleading (17,·) shape comments across P3 that mask BUG-1
- **`associator.py:66-67`** (`Detection3.keypoints_px # (17, 2)` / `keypoint_conf # (17,)`), **`cluster_lift.py:48-50` + `lift_frame` docstring `:118`** — arrays are 26; comments say 17. The stale comment is what makes BUG-1's guard read as correct.
- **Fix:** correct the comments to (26,·)/halpe26.
- **Effort: XS.**

### BUG-14 — dead/duplicated "native keypoints" path implies a COCO-17 mode that cannot exist
**`src/identity/p3_association/jsonl_io.py:108-121`** — `native = player.get("pose_2d")` (same object as `pose`), so `native_keypoints_px` is a byte-copy of `keypoints_px`, and the `Detection3` "None on COCO-17-only runs" claim (`associator.py:80-83`) can never occur. Harmless (only feeds v3, disabled by BUG-1) but dead/confusing.
- **Fix:** drop `native_keypoints_px`/`native_keypoint_conf` from `Detection3` and the read path.
- **Effort: S.**

### BUG-15 — bowler slot cost uses `abs(speed)`, partially undoing the signed-direction fix
**`src/identity/p6_roles/assigner.py:205`** (`_slot_cost`, v1 epoch solver) uses `abs(speed)` while `_windowed_axis_speed` was deliberately made signed, so a sprint in the wrong axis direction can still look like a run-up at slot level. Also de-slop the v0 `assigner.py:69-71` comment (says `abs()`, code is signed). (= known-bugs BUG-12.)
- **Fix:** make `_slot_cost` sign-aware, consistent with the signed windowed speed.
- **Effort: M.** Verify role accuracy / core-role coverage on the 8-set (two-direction axis trial mitigates today, so small movement expected).

### BUG-16 — batch driver misclassifies stages that exit 1
**`src/main.py:250-258`, `src/identity/id_pipeline.py:102`** — exit code 1 is read as a "warn verdict" for 03/05 (a crash distinguished only by the missing metrics artifact); `id_pipeline.py` applies it to 03 and never gates 05's return code. Any other stage (or unexpected exception path) exiting 1 is misclassified as a soft warning and the chain continues on incomplete output.
- **Fix:** reserve a distinct exit code (e.g. 3) for a warn-verdict across the stage CLIs (03, 05) and interpret it in both drivers; treat any other non-zero as failure.
- **Effort: M** (touches several CLIs + two drivers; test with a forced-fail stage).

### BUG-17 — mosaic render always reads stage 05, ignoring roles (06) and refinement (07)
**`src/main.py:262-271` (`run_render`)** — renders from `05_global_id` even when 06 (role stamps, suppression) and 07 (physics-constrained 3D) produced the terminal predictions. Roles/suppression reach the mosaic via side files, but the **refined 3D never reaches the render**, so the human mosaic review (the final quality gate) sees pre-refinement 3D.
- **Fix:** render from the latest completed stage in the window (07 → else 06 → else 05), resolving side files from it.
- **Effort: S–M** (verification is visual — re-render a couple of deliveries).

---

## LOW — dead code, slop, stale labels (de-slopping); each XS unless noted

- **Stage-label skew (pervasive).** `run_triangulation.py:591` prints "P6:"; `run_global_id.py:41-48` "P4:"; `p6_roles/*` say "P5"; docstrings say "P4"/"P4a" (`p5_global_id/jsonl_io.py:1`, `global_track.py:1`, `track_manager.py:1`). **Fix:** rename to 04/05/06/07. **Effort: S** (many files, mechanical).
- **`smooth_native` inert flag + stale comment** (`p1_stabilization/config.py:45,70,87`, `01_stabilization.yaml:19`). **Fix:** remove flag + "pose_2d_native" comment (never read). **XS.**
- **Dead field `_prev_match`** (`p2_tracking/tracker.py:64`). **Fix:** delete. **XS.**
- **Redundant in-function imports** (`common/metrics.py:363-365`). **Fix:** drop. **XS.**
- **Dead `coco17_indices` param + `select_coco17_pose`** (`phase1_common.py:495-521,537`). **Fix:** remove the function + stop threading the param through both runners. **S.**
- **Bare `except Exception` in P1 detect** (`phase1_common.py:407-409`) — batched→per-image fallback with no log (F-D). **Fix:** log the exception once. **XS.**
- **Bare `except` in mosaic layout** (`render_videos.py:576`) swallows calibration breakage silently. **Fix:** add a WARN print (the sibling projections except at `:737` already has one). **XS.**
- **`--native-skeleton` documented no-op still threaded + written to manifest** (`run_triangulation.py:112-114`, `main.py:425`). **Fix:** drop the flag (or stop writing it to the manifest). **XS.**
- **`--hartley` / `--parallax-order` not wired from main.py** (`run_triangulation.py:89-92,271-278`) — can't fire in production. **Fix:** keep as documented dormant A/B (see `../docs/roadmap.md` A12) or remove. **XS.**
- **`velocity_toward_crease` dead hook** (`global_track.py:163-168`). **Fix:** delete. **XS.**
- **`animation_viz.py` dead** (crashes on matplotlib ≥3.9). **Fix:** `git rm`. **XS.**
- **`overlays.py:390 draw_players(roles=…)` unused param; `ROLE_TAGS` unused** (`:49`). **Fix:** drop the param; keep `ROLE_TAGS` as the documented single-source for role wording or fold into roster. **XS.**
- **`match_id_from_delivery` single-digit assumption** (`phase1_common.py:176-178`) — `M10` leaves the suffix. **Fix:** regex the match number. **XS.**
- **`output_skeleton` runner inconsistency** — stock passes `source_skeleton`, l40s passes `P1_SKELETON`. **Fix:** use `P1_SKELETON` in both. **XS.**
- **Duplicated failure cap** — `MAX_FAILURE_RECORDS=2000` vs hard-coded `2000` (`run_phase1_l40s.py:943`). **Fix:** reuse the constant. **XS.**
- **Capture fps hardcoded twice** — `run_triangulation.py:87` module default + `butterworth_smooth` default 50; `main.py` never passes `--capture-fps`. **Fix:** thread `frame_rate_fps` through. **S.**
- **`emit_kalman_posterior` comment overstates** ("removes double-averaging"; the two `np.mean` layers remain) (`config.py:141-143`). **Fix:** reword. **XS.**
- **`min_emit_frames` semantics mismatch** — comment says "span", code drops on `len(frames)` (`config.py:132-137` vs `runner.py:551-561`). **Fix:** reword or change to span. **XS.**
- **`one_euro_smooth` "zero-phase" wording** overstates (nonlinear filter; fwd/bwd averaging is only approximately zero-phase) — dormant code (`refine.py:374-404`). **Fix:** soften wording. **XS.**
- **`export_ue_packets.py:70 timestamp_ns=0`** hardcoded. **Fix:** derive from frame_index × frame_period, or confirm with the UE ingest team + document. **XS.**
- **Dead guard `< 1`** in shape round (`tracklet_graph.py:1215`) — never true; likely meant `< 2`. **Fix:** correct or remove. **XS.**
- **`l40s` module docstring says "RTMPose-L / Body8"** but default is `rtmpose_x_body8` (RTMPose-X Halpe-26); `:6-7` stale `/home/ubuntu/pose_data` path. **Fix:** correct docstring. **XS.**
- **`_default_incompatible_roles` non-exhaustive** (`p5_global_id/config.py:26-35`). **Fix:** document that it's intentionally partial. **XS.**

---

## Cross-stage / driver / operational issues (report-only, from the merged pipeline tracker)

Report-only items formerly tracked in `resolvebugs.md` / `resolvebugs.md`, folded here so this is the single register.

- **Posterior teleport guard is weak** (`p5_global_id/runner.py` ~`:352`, `emit_kalman_posterior`) — active and changes emission (verified off-vs-on differs 8/8), but weak: the chi²-gated posterior still follows a mis-associated measurement, so emitted teleports persist (33/8-set, 367/40-set) with it on. The effective fix that shipped is the emitted-track velocity drop-gate (A3, off by default). **Fix:** tighten the gate to reject the outlier, or drop the "this flag is *the* teleport fix" framing; tie to BUG-4 + roadmap A1/A3. **Effort: S–M.**
- **Cap-invariant abandoned (live design caveat, not a fix)** — production `graph_llr_positive_cap 3.5` sits above the merge threshold 2.0, so a single strong ground cue can merge alone; safety now rests on support-weighting + the confident-contradiction veto. Remember it; any retune is roadmap A4. **Effort: —.**
- **Tiled detection depends on an out-of-tree import** (`run_phase1_l40s.py`, lazy `from detector_bakeoff import …`) — importable only from repo root / tuned PYTHONPATH; a plain `--tiled-det` from elsewhere raises ImportError after startup. **Fix:** move tiling helpers into `src/core/inference/` or make `tools` a package on `sys.path`. **Effort: S.**
- **Velocity-gate blind spots** (`p5_global_id/runner.py`, `_velocity_gate_ground_rows`) — the first emitted frame always anchors the gate (a teleported first frame survives); after 5 consecutive drops it re-anchors to the rejected position. Documented trade-offs; inert while the emit-velocity gate is off. **Effort: S** (when A3 is enabled).
- **Identity-only bridge hits inflate confirmation credit** (`global_track.py:121-123`) — identity-only hits (no position update) still increment `hits` and mark single-camera, rewarding evidence that never moved the filter. **Effort: S.**
- **Inert stitch cost terms** (`stitching.py`) — the role penalty treats `unknown` as free (and the proxy labels most players unknown, so `w_role` rarely fires); velocity continuity is zero for near-static/short links, exactly where co-located ghosts need discrimination. Folds into the 05b work. **Effort: M** (with 05b).
- **Unbounded driver fan-out + no shard timeout** — `main.py` `--jobs`×`--p2-max-workers` uncapped; `run_phase1_parallel.py` shard subprocess has no timeout (a hung shard blocks forever). **Fix:** global process cap + per-shard timeout. **Effort: S.**
- **L40S runner hardcodes a 134400-frame ETA** — wrong the moment the dataset changes. **Fix:** derive from the discovered frame count. **Effort: XS.**

## Algorithmic quality limitations (not code defects — fix directions in the roadmap)

Pinned measured ceilings of the pipeline, not bugs to patch. Improvement directions: [`../docs/roadmap.md`](../docs/roadmap.md); measured analysis: [`../docs/analysis/`](../docs/analysis/README.md). Listed so the register is complete.

- **Detector recall bound (P1)** — dark/distant/occluded subjects missed; a miss is unrecoverable downstream. → roadmap A9. **L–XL.**
- **Constant-velocity tracker breaks under manoeuvre (02)** — tracks fragment on acceleration/turn/dive; 05 must stitch. → OC-SORT ablation. **L.**
- **Chimera split is on but conservative (03)** — sub-threshold chimeras persist; re-measure the residual rate first. → roadmap (graduated split). **M–L.**
- **Stitching under-merges (05)** — distinct-id 18–25 vs ~13 roster; min-cost-flow gates too conservative. → roadmap 05b. **L.**
- **Single-camera coverage gap (04)** — ~39% of player-frames single-camera, no 3D. → roadmap A8 (single-view PnP). **XL.**

---

## Default-vs-production mismatches to reconcile (traps, not bugs)

These dataclass/CLI defaults do **not** match production YAML (deliberate — defaults preserve the historical
baseline). The hazard is reading `config.py` as production truth. **Fix (recommended): keep the defaults,
but make each stage's standalone entrypoint require `--config` or print a loud banner naming the config it
loaded** — flipping defaults would silently change every standalone/CI/test baseline. **Effort: S.**

| Setting | Code default | Production | Where |
|---|---|---|---|
| `association_mode` | `per_frame` | `tracklet_graph` | `p3_association/config.py:95` |
| `role_assignment_version` | `v0` | `v1` | `p6_roles/config.py:19` |
| tri `--smoother` (standalone) | `ema` | `butterworth` | `run_triangulation.py:84` vs `main.py:421` |
| `lowconf_can_spawn` | `True` | `false` | `p2_tracking/config.py` |
| `suppression_enabled` | `False` | `true` | `p6_roles/config.py` + `suppress_peripherals.py:38` |
| P1 `--nms-thr` | `0.3` | `0.55` | runners |
| P1 `--tiled-det` | off | on | `run_phase1_l40s.py` |

(This is NB-2 in `resolvebugs.md`; the full divergent-field list lives there.)

---

## Verified-correct (checked, NOT bugs) — recorded so they aren't re-investigated

- Calibration column-selection `[0,1,3]`, homography inversion, bbox-bottom grounding math, world→UE axis swap (`ue.x=world.y*100`) — all correct.
- Kalman Joseph-form update + `K = solve(S, H@P).T` (P2 and P5) — correct.
- `ground_from_reprojection_ex` Jacobian (quotient rule) — correct.
- Van Loan `Q_d`, eigenvalue clamp on measurement R, same-camera-frame occupancy vetoes — correct.
- `triangulation._project_point` deliberately does not clamp negative depth — correct for mixed depth conventions.
- Hinge-angle clamp Rodrigues sign (`p7_refine`) — correct; `HALPE26_BONES` ordering valid (parent-before-child).
- The two `(17,3)` allocations (`associator.py:814`, `run_triangulation.py:36`) — deliberate COCO-17 pose-shape descriptor / unreachable empty-guard, NOT BUG-1 siblings.
- Uncommitted p7 One-Euro + reproj diff — correctly plumbed, inert on production config.
- `metrics.py` optimistic-empty defaults — intentional tripwire conventions.
