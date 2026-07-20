# fallback_methods.md — every fallback path, and whether it's silently always-on (handover audit, 2026-07-20)

Motivated by **BUG-1**: `ground_contact_pixel_ex` (`common/geometry.py:188`) guards on
`points.shape == (17,2)`, but the pipeline is Halpe-26 and always passes `(26,2)`, so it silently
returns the **bbox-bottom fallback for every detection** — the primary ankle/foot logic never runs.
This file inventories *every* fallback in `src/core` + `src/identity` and, for each, judges whether the
**primary path actually runs in production** or the fallback is the de-facto only path.

**Verdict legend**
- **PRIMARY-RUNS** — primary path executes in production; fallback is a genuine rare safety net.
- **ALWAYS-FALLBACK** — the trigger is (near-)always true, so the fallback is the real behaviour (BUG-1 class). **Danger.**
- **SILENT-ON-FAILURE** — primary runs on the benchmark data, but the fallback triggers *silently* (no log) if an input/config is missing → latent risk on a new dataset.
- **CONDITIONAL** — fires on a real, bounded subset of frames/tracks by design (expected).
- **UNKNOWN** — frequency is data-dependent; needs a runtime metric to settle (metric named where one exists).

Bugs cross-reference `resolvebugs.md`. Legacy cross-references `legacy_code.md`.

---

## 0. THE DANGEROUS ONES — silent, (near-)always-on fallbacks (read first)

| # | Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|---|
| **F-A** | `common/geometry.py:188` (`ground_contact_pixel_ex`) | ankle/midpoint/heel-toe ground contact → **bbox-bottom, height 0** | `points.shape != (17,2)` — always true (Halpe-26 is 26) | **ALWAYS-FALLBACK** | **NO** — silently corrupts all P2/P3 ground; defeats foot_contact v2/v3, ankle-height, single_cam_height_emit, z0_reproj grazing correction. = **BUG-1 (HIGH)** |
| **F-B** | `p1_stabilization/config.py:17` (`DEFAULT_CONFIG`) | load tuned `01_stabilization.yaml` → **hard-coded dataclass defaults** | `load_stabilization_config(None)` + path `configs/p1b_stabilization.yaml` doesn't exist | **SILENT-ON-FAILURE** (standalone runs only; `main.py` passes `--config` explicitly so prod is safe) | Partly — prod OK, but any standalone `run_stabilization` silently drops the tuned filter. = **BUG-3 (MED)** |
| **F-C** | `p7_refine/runner.py:144-148` | visibility-aware re-lift with real projections → **refine existing pose_3d, NO re-lift** | `load_projection_matrices_from_drive(...)` throws `OSError/ValueError/KeyError` → `projections={}` **with no warning** | **SILENT-ON-FAILURE** (benchmark calibration loads → relift RUNS; a bad calib path silently disables the whole re-lift feature) | Mechanism valid, but the **silence** is the risk — add a warning. (MED) |
| **F-D** | `phase1_common.py:407-409` (`detect_person_boxes_batch`) | batched list-input detection → **per-image loop** | bare `except Exception` on the batched call, per frame, **no log** | PRIMARY-RUNS (batched path works on mmdet) but a real detector fault degrades to slow path invisibly | Mechanism valid; the silent bare-except is the smell. (LOW–MED) |
| **F-E** | `cue_calibration.py:232-236` (`fit_cue_calibration`) | per-cue fitted same/diff Gaussians → **conservative default distribution** (per cue) | `_robust_gaussian(samples)` returns None (too few anchor/diff pairs for that cue) → `continue` | **UNKNOWN** — data-dependent; runtime signal = `association_metrics.json` `cue_d_prime.appearance` (`d_app` panel col) + `anchor_pair_count`. Ground cue dominates merges anyway (see `graph_llr_positive_cap` story), so appearance/posture defaulting has bounded impact | Valid *when* it fires (better a flat cue than an inverted one), but if it fires broadly the LLR fusion silently collapses toward ground-only |

**The one to fix/confirm is F-A.** F-C is the next-most-insidious (silent feature-off). F-E is worth a one-time check against the panel's `d_app` column on all 8/40 to confirm cue calibration isn't routinely thin.

### Fixes & effort for the dangerous fallbacks
Effort scale: **XS** <30 min · **S** ~1–2 h · **M** ½–1 day (needs 8-delivery A/B) · **L** days (8+40 A/B + retune). Verification dominates cost — see `remediation_plan.md`.

- **F-A (BUG-1) — HIGH.** *Fix:* make the guard length-aware (accept ≥17 rows; ankles 15/16 valid in both skeletons), then gate emit/clustering/tracking activation behind three separate flags and A/B each on the 8-set; expect a downstream threshold retune. *Effort:* **L** (guard edit is XS; verification + retune is the cost). **Owner decision required first** (report-only BUG-9). Do NOT silently flip it — every baseline was measured on bbox-bottom grounds.
- **F-B (BUG-3) — MED.** *Fix:* point `DEFAULT_CONFIG` at `configs/01_stabilization.yaml` or make `load_stabilization_config(None)` raise. *Effort:* **XS.**
- **F-C — MED.** *Fix:* add a one-line WARN in `p7_refine/runner.py:144-148` naming the delivery + reason when the re-lift calibration load fails (so a silently-disabled re-lift is visible). *Effort:* **S.**
- **F-D — LOW–MED.** *Fix:* log the exception once in the `phase1_common.py:407` batched→per-image fallback. *Effort:* **XS.**
- **F-E — UNKNOWN, verify first.** *Fix (diagnostic, not code):* read `association_metrics.json → cue_d_prime.appearance` (panel `d_app`) + `anchor_pair_count` across all 8/40; if cues are routinely thin, the LLR fusion is quietly ground-only and the `graph_llr_positive_cap 3.5` tune is doing all the work — then the real fix is the depth-aware association weighting (`../docs/roadmap.md` A4, **L**). *Effort to check:* **S.**

---

## 1. `src/core/`

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `calibration.py` (homography build) | invert ground homography | singular / non-finite projection matrix | raises (NOT a fallback — fails loud) | Correct — no silent degrade |
| `calibration.py` `image_to_ground_xy` | homogeneous divide | `|w| < 1e-12` degeneracy guard | CONDITIONAL (rare) | Valid |
| `keypoints.py:126` (`named_root_relative`) | root = mid-hip idx 19 → **mid of COCO l/r hip → any finite joint** | root joint non-finite | CONDITIONAL | Valid graceful cascade |
| `contract.py` | validate → **raise** on bad schema | shape/enum/finite violation | fails loud (not a fallback) | Correct |
| `datasets.py` | `calibration_source` indirection (40_full borrows 8_init) | dataset declares a borrow | PRIMARY-RUNS (by design) | Valid |
| `ue_transform.py` | axis-swap world→UE → **null** | non-finite coordinate | CONDITIONAL | Valid |
| `phase1_common.py:665-686` (`read_resume_state`) | resume from written frames → recompute last line | truncated last JSONL line counted corrupt | CONDITIONAL — **but leaves the corrupt line in the file forever** = **BUG-7** | Partly — recompute OK, residue not truncated |
| `phase1_common.py:468-484` (`resolve_skeleton`) | skeleton from config-path tokens → **`coco_17`** | path lacks `halpe`/`body8`/`body7` | SILENT default → then `validate` aborts at frame 1 = **BUG-8** | Brittle; fails loud downstream but for the wrong reason |
| `phase1_common.py:495-521,506` (`select_coco17_pose`) | slice source→COCO-17 with `coco17_indices or range(17)` | dead since Halpe-26 migration | dead code (never called on active path) | N/A — SCRAP |
| `phase1_common.py:176-178` (`match_id_from_delivery`) | strip `M<digit>` | two-digit match (`M10`) leaves suffix | CONDITIONAL (≥10 matches) = LOW bug | Wrong for ≥10 |

## 1b. `src/core/inference/` runners

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `--resume` append | skip done frames → recompute | resume on | PRIMARY-RUNS | Valid (batch-invariant) except BUG-7 residue |
| l40s `--tiled-fast` | fast forward path → generic per-image pipeline | `--no-tiled-fast` | PRIMARY-RUNS (fast on) | "parity-checked" per comment |
| l40s `--amp`/`--perf` | fp16 AMP + TF32 → fp32 | flag off | PRIMARY-RUNS (on) | parity-verified per comment |
| l40s sweep/grid | autotune batch → **plain 640 detect, no AMP** | always (sweep ignores tiled+AMP) | ALWAYS-FALLBACK *within the sweep* = **BUG-6** | NO — mis-tunes vs the real path |
| `run_phase1_parallel.py` | shard deliveries across GPU subprocesses | — | operational utility, weakest-verified | confirm still used |

---

## 2. Stage 01 — stabilization

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `runner.py` (`enabled`) | One-Euro smooth → **byte-identical passthrough** | `enabled: false` | PRIMARY-RUNS (enabled) | Valid A/B switch |
| `smoothing.py` spike clamp | filter → **replace jump with last filtered pos** | low-conf jump > max(120px, 0.5·bbox-diag) AND `conf < 0.30` | CONDITIONAL (rare, by design) | Valid |
| `smoothing.py` (0,0)/non-finite | advance filter → **pass-through placeholder, don't advance** | missing keypoint | CONDITIONAL | Valid |
| `linker.py` IoU link | continue micro-track → **end track / new track** | IoU < 0.3 or gap > 2 | CONDITIONAL | Valid (smoothing-only, not identity) |
| `config.py` `smooth_native` | (claims to also smooth native block) | — | **inert no-op** (never read; comment stale) | dead flag |

---

## 3. Stage 02 — per-camera tracking

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `jsonl_io.py:88-94` ground_xy | ankle ground contact → **bbox-bottom** | inherits **F-A / BUG-1** (26≠17) | **ALWAYS-FALLBACK** | NO — ankle knobs inert here too |
| `jsonl_io.py:45` pose-conf fallback (diagnostic counter `detection_confidence_pose_fallbacks`) | detection conf → pose-derived conf | missing detection confidence | CONDITIONAL (counted) | Valid; the counter tells you how often |
| `tracker.py` Stage-2 | high-conf IoU+pose match → **low-conf IoU-only recovery** | Stage-1 unmatched + `stage2_confidence_min 0.1` | PRIMARY-RUNS (ByteTrack by design) | Valid |
| `tracker.py` `lowconf_can_spawn` | spawn new track → **recover-only, no spawn** | `false` in prod | PRIMARY-RUNS (fallback-by-config) | Valid (suppresses fragments) |
| `_try_dormant_reid` | pose-cosine re-ID → **no re-ID (stay dormant/expire)** | best cosine < 0.25 or ambiguity margin < 0.05 or > 60 frames | CONDITIONAL | Valid (ambiguity guard) |
| `_apply_ground_cost` gate | ground reachability gate → **skip ground cost** | ground unavailable (and it usually IS bbox-bottom via F-A) | ALWAYS-degraded input | gate still runs on bbox-bottom point; weaker than intended |
| Kalman dormant | measurement update → **process-noise inflation, predict-only** | track dormant | CONDITIONAL | Valid |
| pose gallery `medoid` incremental | O(K) medoid cache → **full O(K²)** | — | PRIMARY-RUNS (incremental) | "bit-identical" per comment |
| **OC-SORT** OCM/ORU/OCR | (ocsort mechanisms) | only if `tracker: ocsort` | inert (bytetrack) | off; see `legacy_code.md` B-2 |

---

## 4. Stage 03 — cross-camera association (fallback-dense)

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `jsonl_io.py:92`, `associator.py:219,236` foot pixel / ground_xy | ankle/heel-toe ground + height → **bbox-bottom, height 0** | **F-A / BUG-1** | **ALWAYS-FALLBACK** | NO — the stage's ground accuracy is built on this |
| z0_reproj emit height correction (`associator.py:922`) | grazing-angle height correction → **project onto z=0** | `height > 1e-9` never true (height always 0 via F-A) | **ALWAYS-FALLBACK** | NO — correction dead |
| `tracklet_graph.py:573-581` cue calibration source | this delivery's harvested calibration → **`calibration_fallback_path` file → conservative defaults** | `calibration_fallback_path` empty (it is) → thin-cal defaults | PRIMARY-RUNS (auto-harvest) then per-cue F-E | Valid; watch F-E |
| `cue_calibration.py:232-236` | fitted per-cue Gaussian → **default distribution** | thin samples (see **F-E**) | UNKNOWN (check `d_app`) | Valid-when-fires |
| `cue_calibration.py:240-245` | separating cue → **collapse to zero-information (mu_same=mu_diff)** | `mu_diff <= mu_same` (cue carries no direction) | CONDITIONAL per cue | Valid (avoids inverted cue) |
| `pose_shape.py:252` posture sigmas | fitted posture systematic sigma → **default sigmas** | no posture calibration deltas | CONDITIONAL | Valid |
| appearance cue (`appearance.py`, `fit_pair_distribution`) | per-camera-pair colour stats → **abstain (None)** | either side too thin | CONDITIONAL (per pair) | Valid (won't borrow another pair's colour) |
| posture `_posture_z` (`tracklet_graph`) | full stature/width posture → **STATURE-only** | crouched / non-upright body | CONDITIONAL | Valid |
| pose descriptor / shape cues | LLR contribution → **abstain** | < min shared segments / parallax | CONDITIONAL — makes ground the sole cue on facing pairs (the reason `graph_llr_positive_cap` was raised to 3.5) | Valid but structurally load-bearing on ground |
| `associator.py:482` per-frame anchor | anchor-camera match → **busiest camera** | anchor empty this frame | inert (per_frame mode OFF) | N/A |
| `approx_feet` / `upper_body_ground_estimate` | foot ground → **hip/shoulder/bbox-top height-plane ray** | feet unusable (cut-off bbox) | CONDITIONAL — **and this path does NOT go through F-A, so it actually works** | Valid |
| triangulation `< min_views` | DLT triangulate joint → **NaN** | joint seen by < 2 cams | CONDITIONAL | Valid |

## 4b. Note on the two `(17,3)` allocations (checked — NOT BUG-1 siblings)
- `associator.py:814-815` (`_pose_descriptor_for_members`) — deliberate **COCO-17-only reduced skeleton** for the view-invariant bone-ratio descriptor (`points3d[_BODY_JOINTS]`, `_BODY_JOINTS ⊆ [0,16]`). Does NOT touch ground contact. Valid by design; only the surrounding `(17,·)` comments mislead (BUG-13).
- `run_triangulation.py:36` (`_most_complete_pose`) — `(17,3)` returned only on empty sequence, which "callers never hit"; effectively unreachable. Should be `(26,3)` for consistency (LOW).

---

## 5. Stage 04 — 3D lift

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `run_triangulation.py` id grouping | group by P3 `binding_id` → **`global_player_id`** | `--id-source global` | PRIMARY-RUNS (binding forced by main.py) | Valid; global branch legacy |
| RANSAC per-joint | inlier triangulation → **`< min_views` → NaN joint** | joint under-observed | CONDITIONAL | Valid |
| fill chain | **occlusion temporal fill → skeletal-prior fill → NaN** | gap ≤ 25 → else prior → else null | CONDITIONAL cascade (both fills active) | Valid graceful cascade; NaN only when truly unseen |
| emit gate | ship frame → **drop frame** | hips 11 & 12 not both finite | CONDITIONAL | Valid (mid-hip is the anchor) |
| `--smoother` | Butterworth zero-phase → **causal EMA** | `--smoother ema` | PRIMARY-RUNS (butterworth) | EMA legacy (B-5) |
| `--robust-refit` | non-IRLS batched → (IRLS Huber if ON) | off in prod | PRIMARY-RUNS (non-robust) | robust-refit is the *opt-in*, not a fallback |
| suppression (`:166-223`) | apply `suppressed_ids` | only in `--id-source global` branch | inert in binding prod = **BUG-11** | dead in prod |

---

## 6. Stage 05 — global identity

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `track_manager.py:151-179` `_measurement_R` (**update**) | P3 ground covariance R → **fixed role R** | `ground_cov` None / not (2,2) / non-finite / eigh fails | CONDITIONAL — measurement-R used when cov present (P3 `emit_ground_cov=true`); role R otherwise | Valid; but the cov is derived from F-A-degraded ground |
| `track_manager.py:164` `_measurement_R` (**gating**) | measurement R → **role R** | `use_measurement_covariance_for_gating` False (prod) | **ALWAYS-FALLBACK by design** (anti-teleport asymmetry) | Valid — intentional |
| `runner.py:349-357` emit | Kalman posterior → **raw correspondence ground_xy** | `emit_kalman_posterior` false | PRIMARY-RUNS (posterior on) — the raw branch would emit filter-rejected outliers = **BUG-4** | posterior path valid; legacy branch latently wrong |
| `runner.py:281` ground source | triangulated foot → **warn + fall back to feet** | lift ran `--id-source global` (no binding ground) | CONDITIONAL (warns) | Valid, logged |
| `runner.py:387` metrics anchoring | exact agreement/teleport anchoring → **approximate** | anchor data missing | CONDITIONAL (warns) | Valid, logged |
| `runner.py:584-595` hip emit | triangulated hip ground → **foot** | `emit_ground_source=triangulated_hip` + `single_view_hip_fallback` (both OFF in prod) | inert | off by default |
| four-stage association | binding→tracklet→geometric→shadow→**re-entry/birth** | each stage no-match | CONDITIONAL cascade | Valid |
| `adaptive_lost_window` | grow window with hits → base 30 | `adaptive_lost_window` (on) | PRIMARY-RUNS | Valid |
| stitching no-solution | min-cost-flow remap → **identity remap** | no improving flow | CONDITIONAL | Valid |
| `finalize()` `:669-677` | gated confirm → **promote any hits≥2, no shadow/roster gate** | end of delivery | **ALWAYS at tail** = **BUG-5** (over-mint) | partly masked by min_emit_frames |
| ground Kalman | Joseph update → (no PD-failure fallback; clamp on R) | — | PRIMARY-RUNS | Valid |

---

## 7. Stage 06 — roles

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `run_role_assignment.py:114-149` bowling sign | run-detected direction → **cost-flip on ±axis (pre-shot window)** → **`no_axis_fallback`** | run-up missing → cost-flip; **axis itself None** → no_axis_fallback | run-up-missing is COMMON and handled by cost-flip (designed, not degraded); no_axis_fallback only if pitch axis underivable = CONDITIONAL(rare) | Valid, layered |
| `assigner.py:282` `_no_axis_fallback` | epoch Hungarian roles → **fastest-mover → bowler, rest fielders** | axis None | CONDITIONAL(rare) | degraded but bounded; only when calibration can't give a pitch axis |
| `assigner.py:334-351` latch | assign slot → **stay unknown** | latch < 3 epochs or cost > `max_cost 8.0` | CONDITIONAL | Valid (debounce) |
| `suppress_peripherals.py` | drop low-quality peripheral → **keep** | core role, or above conf/completeness thresholds | CONDITIONAL — core roles never dropped (fails safe = keep) | Valid |
| `run_role_assignment.py:82` | v1 epoch solver | requires input `online_role_proxy=true`; **raises** if absent | fails loud (not silent) | Correct |
| v0 `assign_roles` | (legacy heuristic) | `role_assignment_version=v0` | inert (v1 prod) | legacy B-4 |

---

## 8. Stage 07 — refine

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `runner.py:144-148` projections | load calibration for re-lift → **`{}` (skip re-lift)** | calib load throws, **silently** | **SILENT-ON-FAILURE** = **F-C** | mechanism valid, silence risky |
| `runner.py:196` relift gate | `relift_sequence` → **refine existing pose_3d** | `projections` empty | PRIMARY-RUNS on benchmark; = F-C otherwise | Valid |
| `relift.py:142` per-joint | ≥2 reliable views → weighted DLT | — | CONDITIONAL | Valid |
| `relift.py:156-166` root | Pass-A root → **single-view neck-anchored ray → existing lifted pose** | root non-finite | CONDITIONAL cascade | Valid |
| `relift.py:169-177` Pass-B | multi-view joint → **single-view bone-length ray from parent** | joint seen reliably in 1 cam | CONDITIONAL | Valid |
| `refine.py:481-494` low-conf | trust raw joint → **NaN → temporal fill → skeletal-prior fill** | `conf < conf_floor 0.5` | CONDITIONAL cascade | Valid |
| `refine.py:501` FK rebuild | FK-recomposed pose → **filled input pose** | FK could not place root | CONDITIONAL | Valid |
| `refine.py` `limb_smoother` | moving-average/butterworth limb dir → (One-Euro if set) | `limb_smoother=one_euro` (OFF) | PRIMARY-RUNS (default) | One-Euro inert (human-rejected) |
| `estimate_canonical_bones` | per-player median bone → clamp to `HALPE26_BONE_LIMITS_M` | out-of-range median | CONDITIONAL | Valid |
| `clamp_joint_angles` | Rodrigues clamp into [15°,178°] → no-op | angle in band | CONDITIONAL | Valid (sign verified) |

---

## 9. Export & visualization

| Site | Primary → Fallback | Trigger | Verdict | Valid? |
|---|---|---|---|---|
| `export_ue_packets.py:24` source | `07_refine` predictions → **`06_roles`** | refinement disabled | CONDITIONAL | Valid |
| `export_ue_packets.py:70` | real timestamp → **`timestamp_ns=0`** | always | ALWAYS (hardcoded) | confirm intended; frame_id is the only temporal key |
| `render_videos.py:207,577` layout | calibration-derived mosaic layout → **alphabetical fallback layout** | calibration unavailable | SILENT-ish (prints a note) | Valid but hides calib breakage behind a print |
| `render_videos.py:619` delivery id | manifest → **lone delivery in run** | id not given | CONDITIONAL | Valid |
| `video_io.py:143` decode | GPU/NVENC decode → **CPU decode** | GPU/codec hiccup (says so once) | CONDITIONAL | Valid (logged once) |
| `overlays.py:372` label placement | preferred slot → **below, clamped** | everything crowded | CONDITIONAL | Valid |
| `mosaic_layout.py:62` pitch axis | stump mid-base → **PCA of all marking points** | stumps not found | CONDITIONAL | Valid |
| `render_phase1_overlays.py:50` sampler | `--frame-ids`/`--row-indices` → **JSONL row position** | neither given | CONDITIONAL | Valid |

---

## 10. Cross-cutting patterns & recommendations

1. **Shape/skeleton guards are the highest-risk fallback class.** Only **F-A (BUG-1)** is a live always-fallback of this kind — the other `(17,·)` sites are either deliberate COCO-17 descriptors (associator.py:814) or unreachable (run_triangulation.py:36) or comment-only (BUG-13). Fixing F-A is the single highest-value action; re-verify ground output on all 8/40 after.
2. **Silent bare-excepts hide real failures.** F-C (relift calib), F-D (detector batch), and the two viz excepts all degrade with little/no logging. Handover fix: make each print a one-line WARNING so a new dataset's missing calibration/detector fault is visible, not silent.
3. **Config-default fallbacks disagree with production** (F-B and the mismatch table in `resolvebugs.md`). Standalone/CI invocations that omit `--config` silently run a *different stack*. Fix the defaults to match production or fail loud on a missing config.
4. **Cue-calibration (F-E) needs one runtime confirmation.** Check `association_metrics.json → cue_d_prime.appearance` (panel `d_app`) and `anchor_pair_count` across all 8/40; if cues are routinely thin, the LLR fusion is quietly ground-only and the `graph_llr_positive_cap 3.5` tuning is doing all the work.
5. **Most fallbacks are healthy graceful cascades** (triangulation fill chains, relift per-joint, role sign resolution, KF association stages). They fire on a real bounded subset and degrade sensibly. The danger is confined to the handful above.

**Bottom line:** of ~60 fallback sites, exactly one is a silent always-on corruption (**F-A/BUG-1**); three are silent-on-failure risks worth a warning (**F-B, F-C, F-D**); one needs a runtime check (**F-E**). Everything else is a valid, bounded safety net.

**Fixes for the other problematic (non-"valid") rows** — the `NO`/degraded rows in the per-stage tables above (P2/P3 ground_xy inheriting F-A, z0_reproj height correction, non-posterior emit, finalize over-mint, dead suppression, `timestamp_ns=0`) are all themselves catalogued bugs with a fix + effort in `resolvebugs.md` (BUG-1/4/5/11 + LOW list) and sequenced in `remediation_plan.md`. The many rows marked **valid** graceful cascades need **no fix** — they are the healthy safety nets and should be left as-is. Total remediation for the *bad* fallbacks: F-B/F-D/`timestamp` are XS, F-C is S, F-A is L (owner-gated); the rest fold into the Tier-2 bug fixes.
