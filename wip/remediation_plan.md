# remediation_plan.md — how to fix every open issue, with effort & reasoning (handover, 2026-07-20)

The actionable companion to the audit docs. Every **open** issue from `wip/resolvebugs.md`,
`wip/fallback_methods.md`, `wip/legacy_code.md`, `resolvebugs.md`, `resolvebugs.md`,
and the `docs/roadmap.md` A0–A13 backlog — each with a concrete fix, an effort estimate, and the
reasoning behind doing it (and behind the effort). Grouped into execution tiers so the quick wins can be
batched and the campaign-scale levers planned separately.

## Effort scale
| Tag | Wall-clock | Meaning |
|---|---|---|
| **XS** | < 30 min | one-line / comment / dead-code delete. No behavior change, no re-run. |
| **S** | ~1–2 h | localized code change + a smoke test or single-delivery check. |
| **M** | ~½–1 day | behavior-affecting change requiring an **8-delivery A/B** + panel read. |
| **L** | days | needs 8+40 A/B, retune sweeps, or a cross-stage change. |
| **XL** | 1+ week (campaign) | new algorithm/subsystem; design + implement + 8/40 + human mosaic sign-off. |

The dominant cost for anything behavior-affecting is **verification, not typing** — this pipeline has no
ground truth, so every real change is gated on an 8- (or 40-) delivery A/B and human mosaic review
(the working standard in `../docs/roadmap.md`). That is why a one-line guard fix (BUG-1) is rated M/L, not XS.

## ID reconciliation (same issue, different registers)
| This plan | resolvebugs | fallbacks | audit/bugs | known-bugs |
|---|---|---|---|---|
| R1 ground guard | BUG-1 | F-A | BUG-A1 | BUG-9 |
| R2 finalize over-mint | BUG-5 | — | BUG-B1 | BUG-10 |
| R3 render reads 05 | (noted) | — | BUG-B5 | BUG-14 |
| R4 exit-1 misclassify | — | — | BUG-B4 | BUG-13 |
| R5 slot-cost abs speed | BUG-15 | — | BUG-B3 | BUG-12 |
| R6 cap invariant (live caveat, no fix) | (noted) | — | — | BUG-11 |
| R7 config default skew | mismatch table | (F-B related) | deferred | NB-2 |
| R8 posterior weak guard / non-posterior emit | BUG-4 | §6 F-4 | — | BUG-1 |

---

## TIER 0 — Trivial de-slop & dead-code removal (XS each; ~half a day total, zero re-run)

**Why this tier first:** none of it changes pipeline output, all of it directly serves the handover goal
(no slop, nothing that reads as live-but-isn't). It can be one commit, reviewed by eye, no A/B. Do it in a
branch, run the test suite, done.

### R-T0.1 — Delete dead modules — **XS**
- `p6_roles/realtime_bowler_tracker.py` (A-1/BUG-2): unused, geometrically wrong, docstring says "PRODUCTION". `grep` confirms zero importers. **Fix:** `git rm`.
- `visualization/animation_viz.py` (A-2): unused, crashes on matplotlib ≥3.9. **Fix:** `git rm`.
- `p4_lift/run_triangulation.py::triangulate_legacy` + its flat-JSONL arg group (A-3): dead, worse algorithm. **Fix:** delete function + args + the `__main__` dispatch to it.
- `global_track.py::velocity_toward_crease` (A-4): dead hook, no caller. **Fix:** delete.
- **Why:** dead code is the single clearest handover hazard — a successor cannot tell it from live code. Each is grep-verified unreachable, so deletion is safe.

### R-T0.2 — Delete dead helpers & config fields — **XS–S**
- `common/geometry.py`: `condition_number_dlt`, `ground_point_and_cov`, `huber_cost`, `parallax_weight` (F-15) — no callers; also drop `parallax_weight` from `associator.py` imports. **Fix:** delete + drop import. **XS.**
- `p3_association/config.py`: `cycle_xy_tol_m`, `dummy_cost_scale`, `parallax_min_deg`, `parallax_full_deg` (F-16) — no reader. **Fix:** remove from dataclass + YAML. **XS** (grep-confirm no reader first).
- `phase1_common.py`: `select_coco17_pose` + `coco17_indices` threading (E-12) — inert since Halpe-26. **Fix:** remove the function and stop threading the param through `player_records` and both runners. **S** (touches 3 files, but purely subtractive).
- `p2_tracking/tracker.py:64` `_prev_match` dead field; `overlays.py:390` `draw_players(roles=…)` unused param (E-13). **Fix:** delete. **XS.**
- **Why:** these are "inert but wired" — they read as functional. Removing them shrinks the surface a maintainer must understand.

### R-T0.3 — Remove the inert fine-score calibration subsystem — **S**
- `mu_fine_score`, `sigma_fine_score`, `w_epi`, `w_tri`, `CalibrationStats`, `GeometryCache.stats/.huber_delta`, `PairGeometry.w_epi/w_tri/huber_delta`, `config.huber_delta()` (E-11). **Fix:** confirm `build_cost_matrix` is the only consumer (it recomputes from `pg.is_degenerate`), then strip the whole scheme from `config.py`, `cue_calibration.py`, `geometry_cache.py`.
- **Why:** this is the biggest single de-slop in p3 — a whole legacy scoring scheme threaded through the geometry cache but never read. Bigger than XS because it's woven through three files, but purely subtractive and testable (association output must be byte-identical; the per-frame engine that used it is off).

### R-T0.4 — Comment / label / docstring de-slop — **XS each, batch to S**
Fix the misleading text (no code behavior changes):
- Stage-label skew everywhere: P4/P5/P6/P4a → 04/05/06 in prints & docstrings (`run_triangulation.py:591` "P6:", `run_global_id.py:41-48` "P4:", `p5/*` docstrings, `assigner.py`/`suppress_peripherals.py`/`run_role_assignment.py` "P5").
- `(17,·)` shape comments that mask BUG-1: `associator.py:66-67`, `cluster_lift.py:48-50,118`, `tracklet_graph.py:156` → say (26,·)/halpe26 (BUG-13).
- `run_phase1_l40s.py:1-3` "RTMPose-L/Body8" → RTMPose-**X** Halpe-26; `:6-7` stale `/home/ubuntu/pose_data` path.
- `01_stabilization.yaml:19` + `config.py` `smooth_native` — remove the dead flag and its "pose_2d_native" comment, or wire it (it's a no-op today).
- `emit_kalman_posterior` comment "removes double-averaging" (config.py:141-143); `min_emit_frames` "span" vs count (config.py:132-137); `one_euro_smooth` "zero-phase" wording; `_reproj_errors` "can't look like a regression" framing (BUG-12) + `reproj_sample_count` after-count.
- `--native-skeleton` documented no-op still threaded (run_triangulation) — either drop the flag or keep the "(Deprecated no-op)" and stop writing it to the manifest.
- **Why:** every one of these actively misleads a new owner about what runs. This is the literal de-slop deliverable.

### R-T0.5 — `core/schemas.py` disposition — **XS decision + XS move**
- Whole module unimported (F-17). **Fix:** either `git rm`, or if the UE export path intends to adopt `PosePacket`, move it into `identity/export/` and wire it. **Why:** it currently looks like a live core schema and isn't.

---

## TIER 1 — Safety & config hygiene: make silent failures loud (XS–S; low risk, high safety value)

**Why this tier:** these don't change the happy-path output at all; they make the pipeline *tell you* when
it silently degraded (the class of problem that hid BUG-1 for weeks). Pure risk-reduction for the successor.

### R-T1.1 — Fix `DEFAULT_CONFIG` stale path (R7-adjacent) — **XS**
- `p1_stabilization/config.py:17` points at non-existent `configs/p1b_stabilization.yaml` → standalone runs silently use dataclass defaults (BUG-3 / F-B). **Fix:** set it to `configs/01_stabilization.yaml`, or make `load_stabilization_config(None)` raise. **Why:** a standalone stage run silently ships the wrong (untuned) filter; one-character-class fix removes a silent-wrong-output trap.

### R-T1.2 — Add warnings to the remaining silent fallbacks — **S**
- `p7_refine/runner.py:144-148` (F-C): calibration load for re-lift fails → `{}` → whole re-lift silently skipped. **Fix:** `print`/log a one-line WARNING naming the delivery + reason.
- `phase1_common.py:407-409` (F-D): batched-detect bare-except → per-image path, no log. **Fix:** log the exception once.
- `render_videos.py:576` (mosaic layout): bare except degrades to alphabetical layout with no warning (the sibling projections except at `:737` already prints one). **Fix:** add the matching WARN.
- **Why:** F-C is the most insidious surviving one — on a new dataset with a calibration path hiccup, the entire re-lift quality feature vanishes with no signal. A warning turns a silent regression into a visible one. (Several other silent excepts already received warnings in the prior campaign and are not repeated here.)

### R-T1.3 — Truncate corrupt trailing JSONL on resume — **S**
- BUG-7: resume counts a truncated last line as corrupt but leaves it in the file forever. **Fix:** on resume, truncate the file to the last newline-terminated record before appending. **Why:** otherwise every downstream reader must tolerate a permanent bad line, and it re-counts corrupt on every resume.

### R-T1.4 — Config default-vs-production skew (R7 / NB-2) — **S (decision) + S (impl)**
- Dataclass defaults deliberately preserve the historical baseline (so `load_*_config(None)` reproduces it); production truth is the YAML. The hazard: reading `config.py` alone mis-describes production (it caused two mis-filed bugs already). **Fix (recommended):** keep the defaults, but make each stage's standalone entrypoint **require `--config` or emit a loud banner** naming which config it loaded; and add a one-line "production = configs/*.yaml, NOT these defaults" header to each `config.py` (several already have it). **Why:** flipping defaults to match production would silently change every standalone/CI/test invocation's baseline — worse. Making the config source explicit at run time is the safe fix. **Do NOT just flip defaults blindly.**
- `match_id_from_delivery` M10 (LOW) + confirm `contract.py` cam-id zero-pad (BUG-C1, already fixed): **XS** — zero-pad / regex the match number for ≥10 matches; harmless today (single match), fix opportunistically.

### R-T1.5 — Confirm-or-fix `export_ue_packets timestamp_ns=0` — **XS**
- Every packet gets `timestamp_ns=0`. **Fix:** derive from frame_index × frame_period, or confirm with the UE ingest team that frame_id is the only temporal key and document it. **Why:** silent-zero timestamps could bite a downstream consumer; cheap to settle.

---

## TIER 2 — Behavior-affecting code fixes (each M: 8-delivery A/B + human panel/mosaic read)

**Why this tier is M not S:** the code change is small; the cost is the mandatory A/B and sign-off. Each is
an isolated, well-understood fix with a proposed design already on record.

### R1 — Ground-contact shape guard (BUG-1/A1/BUG-9) — **M code + L verification. THE headline.**
- **Fix (already designed, `resolvebugs.md#bug-9`):** make the `(17,2)` guard length-aware (accept ≥17 rows; Halpe 0-16 are COCO-17 so ankle 15/16 stay valid). Gate the **emit path, clustering path, and tracking path behind three separate flags**, each defaulted off, and A/B each independently on the 8-set before any production enablement.
- **Effort:** the guard edit is XS; the real work is **L** — three separate A/Bs, and downstream thresholds (`clustering gates in m`, global-id `r_ceiling_m`, `confidence_discard`) were tuned on bbox-bottom grounds and likely need a retune sweep after.
- **Why:** it silently defeats the entire foot-contact subsystem the stage-03 ground accuracy is *supposed* to use, and every published baseline (0.689 verdict, all 40 numbers) was measured with bbox-bottom grounds. This is simultaneously the highest-value correctness item and the one most likely to move numbers — hence flag-gated and evaluated, never a "cleanup." **Owner decision required first** (it is currently REPORT-ONLY by the 2026-07-17 owner call). Reconfirm that decision before touching it.

### R2 — `finalize()` bypasses shadow/roster gates (BUG-5/B1/BUG-10) — **M**
- **Fix:** route `finalize()` promotion through the same `_confirmation_blocked` gates as `_promote_and_prune`. **Verify:** id count + colocated-pair diagnostics on the 8-set (expect id count flat or lower, no new collisions). **Why:** a late shadow duplicate can be minted as a fresh id in the last frames, inflating the distinct-id count — directly against the roster-cardinality goal. Low-risk, well-scoped.

### R3 — Render reads stage 05, ignoring 06/07 (BUG-B5/BUG-14) — **S–M**
- **Fix:** in `main.py run_render`, render from the latest completed stage in the window (07 refine when present, else 06, else 05), resolving role/suppression side files from it. **Why:** rendered mosaics currently show *pre-refinement* 3D and reach roles/suppression only via side channels — the human mosaic review (the final quality gate) isn't seeing the terminal output. Verification is visual (re-render a couple of deliveries), so closer to S.

### R4 — Batch driver misclassifies exit-1 (BUG-B4/BUG-13) — **M**
- **Fix:** reserve a distinct exit code (e.g. 3) for a warn-verdict across the stage CLIs (03, 05) and interpret it in both `main.py` and `id_pipeline.py`; treat any other non-zero as failure. **Why:** today a genuinely failed stage that exits 1 is read as "ran with warnings" and the chain continues on incomplete output — a correctness-of-orchestration risk. Touches several CLIs + two drivers, so M; test with a forced-fail stage.

### R5 — Bowler slot cost uses `abs(speed)` (BUG-15/B3/BUG-12) — **M**
- **Fix:** make `_slot_cost` sign-aware, consistent with the signed `_windowed_axis_speed`. Also de-slop the v0 `assigner.py:69-71` comment (says `abs()`, code is signed). **Verify:** role accuracy / core-role coverage on the 8-set (the two-direction axis trial mitigates today, so expect small movement). **Why:** the signed-speed fix is partially undone at slot level, reintroducing the "wrong-direction sprint looks like a run-up" ambiguity. S3, do it when roles are next touched.

### R6 — Live design caveat: the "capped ground cue alone can't merge" invariant is abandoned (BUG-11) — **no fix, remember it**
- Not a bug to fix — a design fact to carry forward. Production `graph_llr_positive_cap 3.5` sits above the merge threshold 2.0, so a single strongly-supported ground cue can merge alone; merge safety now rests on support-weighting + the confident-contradiction veto, not the old structural bound. Any **cap retune** is part of A4 (Tier 3). Keep `methods_inventory.md`'s `graph_llr_positive_cap` note in sync.

### R8 — Posterior teleport guard is weak / non-posterior emit latent (known-bugs BUG-1 + my BUG-4) — **S–M**
- Two linked things: (a) `emit_kalman_posterior` is active but a *weak* teleport guard (the chi²-gated posterior still follows the mis-associated measurement); the real fix that shipped is the emitted-track velocity drop-gate (A3, off by default). (b) The legacy non-posterior emit branch (`runner.py:349-357`) would emit filter-rejected outlier feet — latent because posterior is on. **Fix:** delete the non-posterior branch (or gate it correctly), and drop the doc framing that `emit_kalman_posterior` is *the* teleport fix; point to the velocity gate. **Why:** removes a latent wrong-output branch and a misleading claim. Tie the decision to A1/A3 (teleport metric + emit gate).

---

## TIER 3 — Algorithmic quality levers (L–XL; design + 8/40 A/B + mosaic sign-off)

**Why separate:** these are not "fixes," they are the roadmap items from `../docs/roadmap.md`. Each is a project
with a human keep/enable decision at the end. Listed with effort and the one-line reasoning; full context
in `../docs/roadmap.md` and `docs/analysis/`.

| Item | Fix direction | Effort | Why |
|---|---|---|---|
| **A0 decide-in-3D + coverage recovery** | Consume 04's 3D in 05 behind `--track-in-3d` (pelvis_ground_xy as measurement, pelvis_cov as R, 3D pose-shape re-ID, reproj chimera-split); re-triangulate per global_id after 05 to recover `tri_cov` (v8 0.817→v9 0.566). | **XL** | The top structural item — the single-triangulation rewire left 05/06 carrying 3D but not using it; recovering coverage + 3D-native identity is the biggest quality lever. |
| **A8 / BUG-7 single-view PnP lift** | Fit the identity's canonical skeleton (bone lengths from its multi-view frames) to a lone 2D view, PnP-style, honest covariance. | **XL** | ~39% of player-frames are single-camera and get no 3D at all. Principled version of the rejected sticky-hip attempt. |
| **A4 depth-aware association weighting** | Weight the ground-distance LLR by each camera's calibrated depth-uncertainty; up-weight union-lift consistency. Also folds the `graph_llr_positive_cap` retune. | **L** | Split identity persists on low-parallax facing pairs; the cap bump only partially closed it. The principled fix for the facing-pair problem. |
| **BUG-6 / 05b stitch under-merge** | Loosen min-cost-flow bridging where occupancy proves two segments can't be simultaneous; add pose/appearance to the stitch cost; fix the inert `w_role` (unknown treated free) and zero velocity-continuity terms (BUG-C5). | **L** | Distinct-id count stays 18–25 vs ~13 roster; stitcher barely fires (`stitched_id_switch_proxy=0`). The id-inflation lever independent of detection. |
| **BUG-5(pipeline) manoeuvre tracking** | Enable OC-SORT (already wired) — but ablate: drop the aggressive OCR recovery, keep ORU+OCM, re-A/B (net-negative as a whole today). | **L** | CV-Kalman fragments tracks exactly on acceleration/turn/dive; 05 then must stitch the pieces. |
| **BUG-4(pipeline) / A9 detector recall** | Detector bake-off (`--detector rtmdet_l/x/dino` presets exist) + per-camera adaptive `bbox_thr`; pair tiling with A3 emit-gate + W6 suppression. | **L–XL** | A miss at detection is unrecoverable; the "dark umpire" class. Note: no stronger detector weights on the box yet; tiling is +agreement but +teleports at 3× cost (decision pending). |
| **BUG-3 residual chimera** | Re-measure residual chimera rate with split on; if material, graduated split threshold or reproj-gated correlation clustering. | **M–L** | Split is on but conservative (torso-residual 30px, frame-fraction 0.6); mild/short chimeras persist. First step is just re-measurement (M). |
| **A1 teleport metric & verdict** | Velocity-gated teleport metric on emitted `ground_tracks.jsonl`, multi-camera only; demote the raw bbox-bottom proxy to a tripwire. | **S–M** | Every `fail` verdict rides on raw bbox-bottom single-camera grazing noise — the verdict is currently measuring the wrong thing. Cheap, high-signal. |
| **A5 depth-aware colocated radius** | Replace flat `colocated_radius_m 0.75` with a function of projecting-camera depth-uncertainty. | **S** | Clears residual coloc pairs. |
| **A6 absolute stitch-budget ceiling** | Cap cross-space stitch in absolute metres, not gap-scaled. | **S** | A long occlusion currently licenses a long cross-field stitch (ghost teleports). |
| **A7 tracklet-level global-id lock** | Lock id at the cross-camera *assignment* level, not post-hoc per-tracklet relabel. | **L** | ~517 flicker events are intra-tracklet id flips — BUT the naive per-tracklet lock was already rejected (stable wrong-person id). Do the assignment-level version only. |
| **Flag cleanup (graph_shape / graph_split)** | Disable/remove `graph_shape_enabled` (inert on all 40) and reconsider `graph_split_enabled` (slight agreement drag). | **S** | Decision-gated cleanup from the 40-set A/B; removes inert/negative flags. Human keep-decision required. |
| **A12 dormant flag A/Bs** | G1 Hartley / G3 parallax-order (`triangulation.py`), `airborne_pelvis_emit`, `density_lost_window`, `temporal_link_decay`. | **M each** | Implemented, off, awaiting A/B — each a bounded evaluation, not new code. |

---

## Suggested execution order (the cleaning sequence)

1. **Tier 0 in one branch** (dead code + de-slop) — no A/B, review by eye, run `tests/`. Half a day, biggest handover readability win. *(Note: `this file` says removals are a per-item owner decision — get the nod on the SCRAP list first; it's short.)*
2. **Tier 1** (silent-failure warnings + `DEFAULT_CONFIG` + resume truncation) — low risk, high safety. A day.
3. **Owner decision gate on R1 (BUG-1 ground guard)** — the one call that determines whether the next work touches ground handling at all. If yes, it's the first Tier-2 A/B.
4. **Tier 2 fixes** (finalize, render-05, exit-1, slot-cost, posterior branch) — each an isolated 8-delivery A/B; can be pipelined.
5. **Tier 3 levers** — planned as projects against `../docs/roadmap.md`, A0/A8/A4 first (highest impact), each ending in a human keep decision.

**Rough totals:** Tier 0 ≈ 0.5 day · Tier 1 ≈ 1 day · Tier 2 ≈ 1 A/B-day per item (5 items, parallelizable) · Tier 3 = weeks (the actual roadmap). The repo-cleanliness handover goal is essentially **Tiers 0+1 (≈2 days)**; Tier 2 is correctness; Tier 3 is the product roadmap.
