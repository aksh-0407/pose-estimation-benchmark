# Remaining work — consolidated backlog (2026-07-14, post v8.1 / 40-delivery production)

Single source of truth for everything deferred, parked or pending after the v6→v8.1 fix
campaign and the 40-delivery production run. Each item carries its full context so it can be
picked up cold. Authoritative history: `docs/critical-analysis/fixes-log.md` (every A/B,
verdict and incident); archived run data: `docs/runs/`.

Current state for orientation: default stack = **v8.1** (`configs/v8/`): tiled RTMDet-m +
NMS 0.55 + fp16 fast path → RTMPose-X Halpe-26 → P1.5 One-Euro → P2 (no-spawn) → P3 tracklet
graph + union-lift merge → P3.5 binding lift → P4 Singer-KF + stitching + colocated merge →
P5 roles v1.2 + peripheral suppression → P6 26-joint 3D (Butterworth, cheirality).
Production dataset: `/home/ubuntu/pipetrack_v8/` on the L40S (40 deliveries, all stages,
README inside). Panel: mean agreement 0.862 across 40, reproj 3.07–3.56 px, coloc 0 on 38/40.

---

## 1. Needs a human / external input

### 1.1 Calibration provenance confirmation (flag raised 2026-07-14)
The box's chain uses calibration copied from the laptop
(`drive/dataset/calibration-data/CCPL080626/` → `~/render_drive/dataset/calibration-data/`).
Empirically verified correct: frame md5s identical across machines; reprojection flat at
3.07–3.56 px mean across all 7 dataset segments including the never-before-processed M2
innings-2 overs. **Ask the team: was there exactly one calibration session for CCPL080626?**
If multiple calibrations exist, obtain and A/B them (expect no change; the numbers already
fit). No re-run needed unless a materially different calibration appears.

### 1.2 Identity ground-truth labelling → IDF1 / MOTA / HOTA
All identity numbers are proxies (agreement / persistence / coloc / teleports, read jointly).
`pose_estimation/cricket/tracking_metrics.py::evaluate_ground_truth` is implemented and takes
a hand-labelled JSONL. Needed: a few hundred labelled frames on 2–3 deliveries (suggest `_7`,
`M2_1_12_1`, `_6` — hard clips). Until then tuning stays proxy-guided. Recurring ask across
`wip/` and the campaign docs since v5.

### 1.3 Vedant's `global_id/` rewrite — parked for his changelog
`vedant2/scripts/global_id/*` is a parallel rebuild lineage (track_manager differs from every
commit by 1000+ lines). His `roles/` contribution was evaluated, de-bugged (uniqueness latch,
standing-back keeper, crease anchors, 2 umpire slots) and merged as roles v1/v1.2. The
global_id rewrite must NOT displace the validated stack without: his changelog, per-change
flag-gating, and the standard 8-delivery A/B. `vedant2/` kept in-tree for reference.

### 1.4 User mosaic sign-off of v8.1
Human review is the final judge. Delivered so far: `artifacts/pipetrack_v8/mosaics/`
(`_14_4`, `_14_7`). The full 8-delivery batch + all-40 batch render on demand
(`~/render_v8_quick.sh` pattern on the box; renderer = collision-free chips, body paint,
roles in roster panel only). While reviewing, arbitrate two open visual questions:
- **Roles end-orientation**: striker vs non-striker end is visually confirmed only on `_2`;
  v1.2 chooses per delivery via plausible-band run detection (3–9.5 m/s) with a pre-shot
  geometric cost fallback (`bowling_direction_source` recorded in every roles.json).
- **Keeper pick ambiguity** (P002-vs-P003 class on `_2`-era data): v1.2 chose the visually
  correct one on `_2`; spot-check a few more clips.

## 2. Implemented, flag-gated OFF — awaiting A/B measurement

### 2.1 G1 Hartley conditioning + G3 parallax-ordered RANSAC
`pose_estimation/triangulation.py` (`hartley=`, `parallax_order=`; CLI `--hartley
--parallax-order` on `scripts/export/triangulate_predictions.py`). G1 row-equilibrates DLT
systems; G3 orders RANSAC seed pairs by ray parallax so exact ties resolve toward good
geometry. A/B on the reproj/coverage panel (1–2 deliveries suffice); accept if reproj p95
drops without coverage loss. Note: the batched fast paths support `hartley` but fall back to
the per-joint loop when `parallax_order=True`.

### 2.2 Airborne pelvis-emit (V2-L3)
`scripts/association/associator.py::_triangulated_pelvis_xy` behind `airborne_pelvis_emit`
(P3 config). When a majority of a cluster's views flag airborne (`_airborne_2d_proxy`), the
emitted ground position becomes the triangulated hip-midpoint vertical projection instead of
the biased z=0 foot ray (measured bias: ankle-z p95 ≈ 0.56 m ⇒ ~0.5–1 m ground overshoot).
Emit-only (clustering gate untouched). A/B: teleports/jitter on jump-heavy clips.

### 2.3 Long-standing opt-ins never promoted
- `density_lost_window` (+`density_radius_m`, `density_bonus_frames`): longer lost-window for
  tracks lost inside a pack. Unit-tested, never composed-A/B'd.
- Cross-delivery prior calibration (F8, `calibration_prior_path` + `CueCalibration.save/load`):
  fit cue distributions on clean clips, reuse on anchor-starved ones (`_7` had 7 anchor pairs
  and the appearance cue abstained — this is the designed remedy).
- `temporal_link_decay` (H4, per-frame mode only), `posture_keep_upright_unknown` (H3,
  experimentally convicted as harmful in v7-rc1 — keep OFF; documented).
- Contested-camera machinery (`contested_iou` etc.): REJECTED on ablation evidence
  (−0.08 agreement on its target clip); dormant by design, do not enable without new evidence.

## 3. Larger deferred experiments (decision-gated)

### 3.1 Pack-handling for in-pack peripherals — the current quality floor
`_6` remains the weakest clip (agreement 0.527; 16–18 ids). Root cause measured: tiled
detection makes close catchers visible; they enter P3 as genuinely hard low-parallax tracks.
W9's union-lift lifted `_6` 0.477→0.625 on the 8-delivery A/B but the pack class remains.
Ideas on file: pack-aware clustering (treat a spatial pack as one super-cluster and solve
membership jointly), per-pack chimera passes, stricter peripheral birth policy in packs.

### 3.2 F16 single-view PnP lift — the 3D-coverage lever
Coverage (fully-triangulated frames / candidate frames) is 0.23–0.84 per delivery because
single-camera frames get NO skeleton. Fit the identity's canonical skeleton (bone lengths
from its multi-view frames) to the lone 2D view, PnP-style, with honest covariance. Big win
for downstream consumers; sizeable experiment (`wip` V2-L1 / F16 since v6 planning).

### 3.3 Detector follow-ups (F18)
Tiled RTMDet-m + NMS 0.55 is the accepted baseline. Queued probes: YOLO26-l and RF-DETR-class
detectors through the same bake-off harness (`scripts/inference/detector_bakeoff.py` +
`detector_bakeoff_report.py`), recall-oracle on `_7`+`M2` first. Also 3×2-grid tiling probe if
P1 time ever matters more than recall.

### 3.4 F15 3D-informed P4 costs / F17 OC-SORT P2 modules
F15: pelvis-height continuity + 3D shape distance in P4 Stage-2/re-entry (P3.5 lift3d is
already emitted per binding). F17: OC-SORT observation-centric re-update in P2 — only if
per-camera fragmentation resurfaces as dominant.

### 3.5 Skeleton-gait embedding cue (needs explicit sign-off)
Pretrained (GREW/Gait3D-class) skeleton-gait embedding as a weak, self-calibrated cue for the
facing-pair identity ceiling. Trainable ReID remains rejected (no GT). Sign-off required
before any work.

### 3.6 G4 inter-cue LLR correlation; per-track height; A1–A7 refactors
- G4: ground residual and billboard posture share the foot anchor (correlated errors); fit a
  correlation shrinkage on anchor pairs or keep the per-cue positive cap as the guard.
- Per-track standing height (v1 ISSUE-8): replace population anthropometrics in the
  height-prior fallback with per-track height learned from confident upright frames.
- A1–A7 (external review): library extraction, config dedup, god-config split — unblocked now
  that the campaign is stable; pure maintainability.

## 4. Performance backlog (W10-PERF residuals)

Shipped and bit-identical: vectorized reprojection/DLT, batched 2-view and multi-view
skeleton RANSAC (P6 10.4 s/delivery; P3 −19%). Remaining (flat profile, no dominant hotspot):
- JSONL parse/write (orjson-class swap; ~15–20% of P3 wall).
- `observe_frame` per-frame pair loop (6.8 s/delivery local).
- Emit-side pose-descriptor bookkeeping (18.8 s/delivery — the RANSAC inside is now fast;
  remainder is python assembly. A descriptor stride would help but changes outputs → A/B).
- Box parallelism tuning: `--jobs 7` × (solver + 4 decode threads) oversubscribes 8 vCPUs;
  probe `--jobs 5` with prefetch 2–3.

## 5. Metrics / evaluation debt

- **Teleport proxy**: raw counts double-count identity re-acquisition (accepted per the
  occlusion-restore objective) and single-cam foot noise (`_3` at 70, `M2_2_4_*` at 100+ with
  0.94+ agreement — clearly proxy artifacts). The coloc metric now covers the
  switch-without-cause class; a cleaner teleport metric (velocity-gated on multi-cam segments
  only) is still worth building. Sources: fixes-log W5-C/W9 caveats.
- **Union-lift synthetic unit test**: the pass is integration-proven (bit-level A/Bs + per-pair
  rejection diagnostics `union_lift_rejects`) but has no synthetic fixture test.
- **Swap-event frame-crop auto-dump**: `diagnose_colocated_ids.py` reports events; the planned
  `--dump-frames` annotated-crop emission was done manually for the exhibits, never wired in.

## 5b. Production-tree residuals (found at final reconciliation, 2026-07-14)

- **2/40 deliveries retain 1 colocated-id pair each** (`M1_1_14_7`, `M2_1_11_3`) in the
  production tree — the W9 merges fired on 38/40 but missed these two on the production P1
  variant (the box P1 used the fp16 fast path; the locally-validated A/B used the earlier
  generic-path P1 — near-parity inputs, threshold-sensitive solve). Options: relax
  `colocated_*` gates slightly and re-run P4 for just those two deliveries, or leave and let
  the mosaic review arbitrate. The `coloc` panel column tracks it either way.
- **Box-vs-local panels are near but not identical** for the 8 overlap deliveries (same code,
  different P1 binaries): e.g. `_7` local 0.962/10 ids vs box 0.819/12. Expected variance
  class, documented here so nobody chases it as a bug.

## 6. Product/output tasks not yet requested

- **UE packet export** (`scripts/export/export_ue_packets.py`) has not been run on v8
  production data — run it per delivery if the company needs UE-format packets rather than
  the JSONL 3D (input contract unchanged; consumes the P6 output).
- **All-40 mosaic batch** — render on demand (~3 h on the box, 2 parallel).
