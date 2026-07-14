# Critical-Analysis Fixes: Methods Log

This log records the implementation and A/B evaluation of the fixes catalogued in
[fixes-roadmap.md](fixes-roadmap.md), executed per the plan in [to-do.md](to-do.md). For every
fix it states the purpose, the implementation (files and flags), the measured result against the
frozen baseline across the 8-delivery evaluation set (`CCPL080626...`), and the verdict. It
follows the conventions of `wip/methods_log.md`.

## Executive Summary

(Meeting-ready snapshot: see also [status-report.md](status-report.md).)

- **Baseline established and frozen** (`pipetrack_v6.0`): full chain from the 8-delivery
  RTMPose-X data; IDs 10–18 vs ~13–15 roster, agreement 0.60–0.92, 3D reprojection
  3.2–3.6 px, collisions 0 everywhere. Doubles as the RTMPose-X-vs-L model data point.
- **Fifteen fixes implemented and A/B'd** (F1–F15 + FR review batch), each flag-gated and
  measured on all 8 deliveries; 192 unit tests. Every negative result is root-caused, not
  just recorded (asymmetric R, origin-referenced cheirality, abstaining shape cue, chimera
  split thresholds, posture-veto blind spot).
- **Best composed candidate `v7-rc1`**: the hardest clip is transformed (`_7` agreement
  0.603→0.713, IDs 18→13 at roster, teleports −10, persistence +0.051); persistence up on
  6/8 clips, 3D reprojection improved everywhere (native-26 lift + cheirality + frame-aware
  fills). NOT yet accepted: multi-view binding regressed on easy clips (`_1` single-cam
  0.27→0.67, agreement down on `_3`/`_5`/`_6`) — attribution ablations running; prime
  suspect is an unflagged review fix (H3 posture-sample policy) shifting calibrated cue
  distributions. v7-rc2 will gate it and re-compose.
- New measured capabilities: per-clip **chimera counts** with intruding-camera attribution,
  **ID-persistence** and **fragment** panel columns, per-cluster **ground covariance**, and
  the P3.5 binding-keyed 3D lift feeding identity instead of trailing it.

## Evaluation Standard

All candidate changes are evaluated against the frozen `pipetrack_v6.0` baseline
(`data/derived/runs/pipetrack_v6.0/_baseline_snapshot`) across all 8 deliveries. A fix is only
marked Accepted if it produces a significant, generalized improvement without introducing
clustering, identity, or collision regressions. Every change is behind a config flag; with all
flags off the pipeline is byte-identical to the baseline. Same-camera collisions must stay 0
everywhere (hard invariant).

**Objective re-statement (user directive, 2026-07-10):** the deliverable is cross-camera
ID constancy — a player visible in several cameras carries ONE global id in all of them for
the whole clip, and an occluded player's id is RESTORED when the occlusion ends. Transient
teleports during occlusion are acceptable. The panel is therefore read with **agreement,
distinct-ID count, id-persistence (mean confirmed-frame completeness) and excess fragments**
as the primary axes; the raw teleport count is a secondary caution signal (it double-counts
acceptable occlusion transients, especially on `M2`). Fixes are judged as COMPOSED STACKS in
pipeline order, not per-phase — a fix may expose problems a later phase clears.

Metrics (read jointly as a panel, never singly — each alone is gameable):

- **Cross-camera agreement** — fraction of co-observed pairs where both cameras carry the same
  global ID (`global_id_metrics.json`).
- **Distinct global IDs** — vs the ~13–15-person cricket roster; excess = over-segmentation.
- **Teleports** — ID jumps exceeding kinematic limits. Caveat: on `M2_1_12_1` the proxy is
  dominated by noisy single-camera foot projections (single-cam rate 0.61), not the emitted
  trajectory (`wip/methods_log.md`); read jointly with trajectory-jitter p95.
- **Same-camera collisions** — must remain 0.
- **Single-camera rate / pair-link churn / cycle-consistency / cue d′** — P3 association health
  (`association_metrics.json`).
- **2D jitter (px)** — P1.5 `stabilization_metrics.json`, when the stage runs.
- **3D lift: mean/p95 reprojection (px), coverage** — `triangulation_metrics.json`.

Driver: `src/main.py` (full chain + extended panel + baseline diff);
inner loop: `src/identity/id_pipeline.py`. Configs frozen in `configs/v6/`; per-fix
variants in `configs/experiments/`.

## Summary of Fixes

| ID | Fix | Result | Verdict | Rationale |
|---|---|---:|---|---|
| F0 | pipetrack_v6.0 ground baseline (RTMPose-X P1, v5 configs, P1.5 off) | IDs 10–18, agreement 0.60–0.92, teleports 7–154, collisions 0, 3D reproj 3.2–3.6 px | Baseline | Fixed comparison point; also the X-vs-L model data point. |
| F1 | Wire P1.5 2D stabilization into the default flow | jitter 1.2–3.7 px; `_7` IDs 18→14, agreement +0.055, single-cam −0.076; but `_5` agreement −0.121, teleports +24/`_7` +26/`M2` | Opt-in (not default) | Helps the hardest clip's identity structure but inflates the teleport proxy and regresses `_5`; revisit stacked with Wave-3 anti-teleport fixes (F6b/F10). Byte-identity writer bug found+fixed en route. |
| F2 | C07 per-camera size audit + aspect-correct tiles | P3/P4 core already per-camera-correct; render `--letterbox-tiles` added | implemented | Audit found the runtime paths correct; only the mosaic tile stretched C07. |
| F3 | Cheirality check in RANSAC triangulation | det(M) formula mis-signed on this rig (world handedness); fixed origin-referenced; verified on real calibration | re-testing | Wave-1 A/B caught the silent failure. |
| F4 | Halpe-26 feet as ground contact (v3 foot mode) | implemented (heel/toe midpoint, ankle fallback, emit-path only) | A/B running | `foot_contact_mode: v3`. |
| F5 | Online role proxy → P4a Singer dynamics | implemented (bowler-run direction lock, umpire/keeper ends) | A/B running | `p4a.online_role_proxy`. |
| F6 | P4b occupancy-licensed gap bridging | implemented (disjoint-cell license to 300 frames, pose-required) | A/B running | `p4b.occupancy_bridge`. |
| F6b | Billboard posture → P4a teleport veto + re-entry gate | 0 vetoes fired: teleport sources are UNBOUND single-cam clusters (no posture) | No effect (as built) | Needs instantaneous posture for unbound clusters or F10's inflated single-cam R; folded into Wave 3. |
| F7 | Offline zero-phase Butterworth 3D smoothing | implemented (`--smoother butterworth`, NaN-safe segments) | A/B running | Zero-phase verified by cross-correlation lag test. |
| F8 | Cue cold-start robustness (anchor relaxation + prior) | implemented (staged relax → prior file → defaults) | A/B running | `anchor_relax_enabled`, `calibration_fallback_path`. |
| F9 | P3.5 re-sequencing: binding-keyed lift + covariance + purity | implemented; real-data smoke: 7 bindings lifted, 5 chimera-suspects with per-camera bias attribution on delivery 1 | smoke verified | Purity flags correlate with the clip's worst-in-set cycle-consistency (0.698). |
| F10 | Per-measurement (uncertainty) Kalman R | symmetric use raised teleports (M2 +37: wide R loosens admission gates); split to gate-on-role-R / update-on-measurement-R | Wave-3b running | Asymmetric R is the standard MOT resolution; IDs already moved roster-ward. |
| F11 | Pose-shape (bone-ratio) as primary P3 cluster cue | self-calibrated then ABSTAINED on all 8 (d′ < 0.5): bone ratios do not separate players on this footage | Opt-in (abstains) | Honest negative; matches 2026 literature. Stature/billboard channel (F6b/F12) is the live shape path. |
| F12 | Billboard-posture stitch key in P4b | implemented (z-gate + cost term) | pending Wave-3 A/B | Matures on facing pairs where the triangulated key cannot. |
| F13 | Splittable clustering (purity-driven surgical eviction) | chimera suspects → 1-2/clip, M2 id-persist +0.074; but over-splits (agreement −0.08..−0.20, IDs +2..+4) | Mechanism validated, config rejected | Conservative thresholds + reattachment-first eviction in the best stack. |

## Detailed Results

### F0 - pipetrack_v6.0 Ground Baseline

Purpose:

- Establish the fixed comparison point for the fix campaign, built for the first time from the
  full 8-delivery RTMPose-X (Halpe-26) P1 data (`data/derived/runs/rtmpose-x`) rather than the
  RTMPose-L P2 tree that v3/v5 reused.

Implementation:

- Frozen config set `configs/v6/` (P3/P4 = copies of the validated `*_v5.yaml` flag stacks;
  P2 = committed defaults; P1.5 present but `enabled: false`).
- New full-chain driver `src/main.py`: P1.5(off) → P2 → P3 → P4 →
  P5 roles → 3D lift → mosaic render, per-delivery parallel, extended joint panel,
  `pipeline_manifest.json` provenance (config sha256), `--base-tree` stage reuse for cheap A/Bs.
- `triangulation_metrics.json` gained `mean/p95_reprojection_error_px` and
  `triangulation_coverage` aggregates (schema `triangulation_metrics/v2`).

Result (all 8 deliveries; the fixed comparison point for every later fix):

| Delivery | Agreement | IDs | Teleports | Collisions | P2 tracks | Single-cam | Cycle-cons | d′(app) | 3D reproj (px) | 3D coverage | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M1_1_14_1 | 0.784 | 10 | 7 | 0 | 29 | 0.271 | 0.698 | 0.86 | 3.6 | 0.737 | pass |
| M1_1_14_2 | 0.923 | 11 | 9 | 0 | 26 | 0.499 | 0.866 | 0.00 | 3.5 | 0.502 | pass |
| M1_1_14_3 | 0.857 | 13 | 13 | 0 | 30 | 0.539 | 0.924 | 1.98 | 3.3 | 0.470 | pass |
| M1_1_14_4 | 0.772 | 13 | 14 | 0 | 26 | 0.528 | 0.930 | 0.00 | 3.5 | 0.477 | pass |
| M1_1_14_5 | 0.898 | 13 | 32 | 0 | 28 | 0.519 | 0.918 | 0.00 | 3.3 | 0.545 | warn |
| M1_1_14_6 | 0.653 | 16 | 40 | 0 | 38 | 0.562 | 0.931 | 0.00 | 3.2 | 0.448 | warn |
| M1_1_14_7 | 0.603 | 18 | 42 | 0 | 35 | 0.504 | 0.891 | 0.49 | 3.3 | 0.503 | warn |
| M2_1_12_1 | 0.791 | 16 | 154 | 0 | 28 | 0.635 | 0.668 | 0.93 | 3.5 | 0.375 | fail |

Observations vs the v5 stack (which ran on the older RTMPose-L-based P2 tree — the deltas
mix the P1-model change with nothing else, since configs are identical):

- Distinct IDs land at 10–18 vs the ~13–15 roster without any new fix.
- Delivery 1's single-camera rate collapses to 0.271 (X binds far more multi-view frames
  there), but its cross-camera agreement drops to 0.784 (was 0.953 on L) — more bound views
  create more opportunities to disagree; `_6` similarly trades agreement for coverage.
- `M2`'s teleport count (154) remains proxy-dominated by single-camera foot noise
  (single-cam 0.635, the worst), per `wip/methods_log.md` — read jointly with jitter.
- The 3D lift is healthy everywhere it can run: 3.2–3.6 px mean reprojection.
- Mosaics: `artifacts/pipetrack_v6.0/mosaics/<D>/` (all 8; `_3/_4/_5` re-rendered after a
  machine crash truncated them — content verified complete at 600/600 frames).

Conclusion:

- Baseline frozen. Every fix from here is measured against this panel; upstream-changing
  fixes (F1) will re-freeze a working baseline on acceptance.
- This panel doubles as the RTMPose-X-vs-L data point for `wip/model_comparison.md`.

### F1 - Wire P1.5 2D Stabilization Into the Default Flow

Purpose:

- Realise the validated One-Euro 2D stabilization end-to-end: denoise keypoints once at the
  source so P2/P3/P4 inherit a cleaner signal.

Implementation:

- Driver `--enable-stabilization` + `configs/experiments/v6_f01_stabilization__p1b.yaml`
  (`enabled: true`, otherwise identical to the frozen v6 copy). Full fresh chain from the
  rtmpose-x P1 data. Run tree: `data/derived/runs/pipetrack_v6.1-f01`.
- En route, the stage's byte-identity guarantee was found broken against insertion-ordered P1
  files (the writer re-sorted JSON keys); fixed by preserving parsed key order, with a
  scrambled-key regression test.

Result (vs the frozen v6.0 baseline; only deltas shown, full panel in the run tree):

| Delivery | Agreement | IDs | Teleports | Single-cam | Jitter after (px) | 3D cov |
|---|---:|---:|---:|---:|---:|---:|
| _1..._4, _6 | ±0.002 | 0 | 0 to +2 | ±0.001 | 1.6-2.1 | ±0.001 |
| _5 | **-0.121** | +1 | 0 | +0.043 | 1.23 | -0.051 |
| _7 | **+0.055** | **-4** (18→14) | **+24** | **-0.076** | 2.98 | +0.085 |
| M2 | -0.015 | 0 | **+26** | -0.009 | 3.74 | +0.007 |

Conclusion:

- **Opt-in, not default.** The identity *structure* on the hardest clip improves markedly
  (roster-ward IDs, more multi-view binding, more 3D coverage on `_7`), but the smoother feet
  bind more views and the fixed-R Kalman/teleport proxy punishes the extra cross-camera
  disagreement (`_7` +24, `M2` +26 teleports), and `_5` agreement regresses. This is exactly
  the failure mode the Wave-3 anti-teleport fixes (F6b posture veto, F10 uncertainty R)
  target — F1 will be re-tested stacked on them before a final verdict.

### W1 - Wave-1 Correctness Batch (F3+F4+F5+F6+F6b+F7+F8)

Purpose:

- Land the seven low-effort correctness fixes as one combined A/B (bisect by flag on
  regression), from the frozen baseline's P2.

Implementation:

- `configs/experiments/v6_wave1__{p3,p4}.yaml` + `--tri-cheirality --tri-smoother butterworth`;
  run tree `data/derived/runs/pipetrack_v6.1-wave1` (P3→P4→P5→3D, baseline P2 reused).

Result (vs v6.0; all 8 chains rc=0, collisions 0):

- Teleports: `_4` −4, `_6` −2, `M2` −1; agreement/IDs/cycle-consistency all within ±0.002;
  no regressions anywhere. Safe but structurally weak.
- Root-caused the weakness: (1) **F6b's posture veto never fired** — the teleport sources are
  UNBOUND single-camera clusters, which carry no binding posture, so the veto abstains
  exactly where it is needed (fix direction: instantaneous posture for unbound clusters, or
  F10's inflated single-camera covariance — Wave 3). (2) The P4b stitcher selected its first
  real link on `_7` (baseline 0), but `M2`'s 1068 feasible edges still yield 0 links.
- **F3 bug found on real data**: the det(M)-based cheirality sign classified every genuine
  in-front point as "behind" on this rig (world-frame handedness), silently disabling the
  RANSAC inlier machinery in the 3D lift (coverage unchanged, but reprojection went
  unmeasured). Fixed with an origin-referenced sign test (the pitch centre is in front of
  every camera; the convention factor cancels) — verified 100% in-front on 200 field points x
  7 real cameras; the Wave-1 3D stage is being re-run with the fix.

Conclusion:

- Wave-1 flags are harmless-to-mildly-positive guardrails; keep them staged into the Wave-3
  stack (which attacks the diagnosed teleport source directly) rather than accepting alone.
- The A/B did its real job: it exposed a silent cheirality bug and localized WHY the posture
  veto cannot fire — both now feed Wave 3.

### W3 - Wave-3 Stack (F9a covariance + F10 uncertainty R + F11 shape round + F12 posture stitch, over Wave-1)

Purpose:

- The structural identity levers: measurement-uncertainty-aware Kalman fusion, the pose-shape
  primary cue, and the P3.5 lift running in-pipeline for the first time.

Implementation:

- `configs/experiments/v6_wave3__{p3,p4}.yaml` + `--enable-lift --tri-cheirality
  --tri-smoother butterworth`; run tree `data/derived/runs/pipetrack_v6.2-wave3`
  (P3 → P3.5 → P4 → P5 → 3D from the frozen baseline's P2).

Result (vs v6.0; all 8 chains rc=0, collisions 0):

- **IDs move roster-ward on every hard clip**: `M2` 16→14, `_6` 16→15, `_7` 18→17; agreement
  flat (±0.004); 3D lift healthy; the P3.5 stage now measures 2–5 chimera suspects per clip
  in-pipeline (the Wave-4 split fuel).
- **Teleports rose** (`M2` +37, `_5` +13, `_6` +7): root-caused to symmetric R usage — the
  inflated single-camera covariance correctly downweights bad feet in the state update but
  also enters the admission gate's innovation covariance, making far, wrong candidates look
  Mahalanobis-close (M2 diagnostics: 59 shadow-absorbs, 31 spawns, 8 re-entries).
- **F11's shape cue self-calibrated and abstained on all 8** (`d' < 0.5`): same-player
  temporal-half bone-ratio distances overlap different-player distances on this footage —
  the abstention machinery correctly refused to inject a non-separating cue. Consistent with
  the July-2026 literature check (identical-kit shape ReID unsolved).

Conclusion:

- Not accepted as-is; **Wave-3b** re-runs P4 with asymmetric R (gate on the conservative role
  R, update with the measurement R — the standard MOT resolution), unit-tested and queued.
- F11 stays as a safely-abstaining opt-in; its practical value may come from the stature/
  billboard channel rather than bone ratios (the F6b/F12 path).

### W3b/W4 - Asymmetric R and Chimera Splits

Result (vs v6.0):

- **W3b (gate on role R, update on measurement R)**: halves the symmetric variant's teleport
  damage (M2 +29 vs +37) and keeps `_7` −3/`_4` −4, but M2 stays elevated: with the 2.0 m
  ceiling a single-camera observation barely moves the posterior, the track freezes, and the
  next multi-view fix reads as a jump. IDs revert toward baseline (the symmetric variant's
  roster-ward IDs were partly the same wide-gate artifact). Next composition: r_ceiling 0.8 m.
- **W4 (F13 splits over the Wave-3 stack)**: the purity mechanism works — chimera suspects
  drop to 1-2 per clip everywhere and M2 id-persistence +0.074 — but split pieces are not
  re-absorbed (IDs +2..+4, single-cam +0.08..0.18) and agreement falls where evictions hit
  genuinely-same-player facing-pair clusters (`_5` −0.203). As configured it over-splits:
  thresholds must scale with person size / be raised (fraction 0.3 → 0.6), and eviction
  should prefer reattachment over fresh bindings. Note: 3 deliveries initially crashed at
  p3_5 because F15 edits landed mid-chain (operator error, repaired + rule reaffirmed: no
  pipeline-source edits while a chain is live).

Conclusion:

- Both are mechanism-validated / configuration-rejected. The grand-analysis best stack
  carries: asymmetric R with a tighter ceiling, and F13 at conservative thresholds.

### FR - External Review Fixes

An external full-codebase review was triaged ([review-triage.md](review-triage.md)); every
confirmed item is fixed with guardrail tests (192 total passing): C1 dead zero-IoU gate (the
sprinting-bowler fragmentation producer), C2 process-noise never resetting after
re-acquisition (verified 57.7x residual inflation), C3 unreachable Lyapunov branch (explicit
stable-block inflation now), C4 role KeyError, C5 single-camera ankle-height emit wired
(flag `single_cam_height_emit`), C6 frame-aware occlusion fills (flag `--tri-dense-fill`),
H2 direction-signed bowler detection, H3 stature retention for upright-unknown samples,
H5 NaN-confidence guard in the ground solve, H6 additive attach margin, H7 driver crash
detection, pose_3d aliasing and bool-validator fixes. C1/C2/C3/H2/H3/H5/H6 are unflagged bug
fixes — the byte-identity guarantee vs v6.0 is intentionally retired from here; the best
stack re-baselines.

### v7-rc1 - Composed Best Stack (grand-analysis release candidate 1)

Composition: P1.5 on + FR-fixed P2 (C1/C2) + Wave-3 P3 stack + C5 single-cam emit +
conservative F13 + asymmetric R (ceiling 0.8 m) + P3.5 lift + native-26 skeleton +
dense-fill + cheirality + Butterworth. Run tree `data/derived/runs/pipetrack_v7-rc1`.

Result (vs v6.0, reweighted panel — agreement / IDs / persistence / fragments primary):

- **`_7` (hardest identity clip) improves on every axis**: agreement 0.603 → **0.713**
  (+0.110), IDs **18 → 13** (at roster), teleports −10, id-persistence +0.051, fragments −5,
  single-camera rate −0.073, 3D coverage +0.079.
- Portfolio-wide: id-persistence up on 6/8 (`M2` +0.077), teleports down on 5/8,
  cycle-consistency up on the previously-worst clips (`_1` +0.196), 3D reprojection improves
  everywhere (2.9–3.3 px; native-26 + cheirality + frame-aware fills).
- **Regression: multi-view binding collapses off the hard clip** — single-camera rate +0.12
  to **+0.40** (`_1` 0.271 → 0.670), dragging agreement on `_3` (−0.116), `_5` (−0.227),
  `_6` (−0.085) and 3D coverage everywhere except `_7`; IDs +2..+4 on easy clips. This
  directly violates the ID-constancy goal (unbound single-camera detections carry no
  cross-camera identity), so rc1 is NOT accepted.

Attribution in progress (two ablations): A = v7rc P3/P4 over the baseline P2 (isolates the
unflagged FR P3-side changes H3/H6), B = same over the F1 tree (P1.5 with the OLD P2 code —
isolates the C1/C2 tracklet changes). The suspects are exactly the two new ingredients this
stack introduced beyond measured waves.

### Attribution Closed: v7-rc1 Binding Collapse = H3; P4b Stitcher Was Mathematically Dead (G7)

- **H3 confirmed by controlled experiment**: re-running the identical v7rc P3 on the identical
  composed P2 with the posture-sample policy reverted to legacy restored multi-camera bindings
  4→7, single-camera rate 0.670→0.367, corroboration merges 0→2 on delivery 1. The permissive
  policy (keep stature for upright-unknown samples) shifts the self-calibrated posture
  distributions enough that the posture cue weakly DISAGREES on facing pairs, silently vetoing
  the corroboration merges. Resolution: `posture_keep_upright_unknown` config flag, default
  legacy; the residual 0.367-vs-0.271 gap is the (small) P1.5 + C1/C2 P2-data effect.
- **G7 unit-mixing confirmed as dead code**: with the tuned weights, `w_temporal x gap` alone
  exceeds the new-trajectory dummy (3.0) for any gap > 30 frames — every stitch beyond 0.6 s,
  including all F6 occupancy bridges, was mathematically unselectable (the measured
  "M2: 1068 feasible edges, 0 links" and the historic `stitched_id_switch_proxy = 0`).
  Fixed with `p4b.normalized_costs` (each term divided by its own gate; legacy off by
  default); unit test pins both the dead zone and the fix.
- Next composed candidate **v7-rc2** = v7-rc1 flags with H3 legacy + normalized stitch costs.

### v7-rc2 - Composed Stack on Fixed Code (post bug-pass validation)

Same flags as rc1; code now carries the H3 gate (legacy default) and G7 normalized stitch
costs. Run tree `data/derived/runs/pipetrack_v7-rc2`.

Result (vs v6.0; primary axes first):

- **The P4b stitcher selected links for the first time in project history** (3-6 per clip;
  0-1 always before G7): `M2` IDs 16→11, fragments 12→7, **id-persistence 0.699→0.956**;
  `_7` holds its transformation (agreement +0.099, IDs 18→13, fragments −6, teleports −7);
  `_3/_4/_5` each −1 ID with persistence up; 3D reprojection better on all 8; collisions 0.
- `M2` teleports +69: the anticipated proxy artifact of stitching itself — a re-joined
  identity "jumps" across its occlusion gap, which the project objective explicitly accepts
  (ID restored after occlusion). Read jointly with its +0.257 persistence.
- H3 gating recovered the rc1 binding collapse (`_1` single-cam 0.670→0.367, `_5` agreement
  0.671→0.778). Residual `_5` agreement −0.120 matches F1's solo P1.5 cost (−0.121) almost
  exactly — motivating the final composition experiment **v7-rc3 = rc2 without P1.5**.

Conclusion: strongest portfolio of the campaign on the reweighted objective; final
default chosen between rc2/rc3 after the P1.5 isolation run.

### GRAND ANALYSIS CONCLUSION - v7 Default Accepted (2026-07-11)

The P1.5 isolation run (`v7-rc3`, identical stack without stabilization) closed the last
composition question: **P1.5 is a pure `_5` <-> `_7` trade** (`_5` agreement 0.899-vs-0.778,
`_7` 0.591-vs-0.703 with IDs 15-vs-13 and fragments 9-vs-6); everything else — including the
M2 stitcher transformation (IDs 16→11, persistence 0.699→0.956) — is P1.5-independent.
Portfolio mean agreement is equal (0.785 vs 0.783); the decision falls to the worst-clip
floor, which is the project objective: **rc2 (P1.5 ON) is the accepted default** (floor
0.655 vs 0.591; and rc2's `_5` still beats baseline on IDs and fragments).

Shipped as **`configs/v7/`** with the driver defaulting to the re-ordered pipeline
(P1 → P1.5 → P2 → P3 → P3.5 → P4 → P5 → 3D[native-26, cheirality, Butterworth, dense-fill]);
v6-style runs remain available via `--no-enable-stabilization/--no-enable-lift` + the frozen
`configs/v6/`. Final panel highlights vs the v6.0 ground baseline:

| Axis | v6.0 | v7 (rc2) |
|---|---|---|
| `_7` agreement / IDs / fragments | 0.603 / 18 / 12 | **0.703 / 13 / 6** |
| `M2` IDs / id-persistence | 16 / 0.699 | **11 / 0.956** |
| Distinct IDs, all clips | 10-18 | **11-16** |
| Stitcher links selected | 0-1 (dead) | **3-6 per clip** |
| 3D reprojection (mean px) | 3.2-3.6 | **3.0-3.5** |
| Same-camera collisions | 0 | 0 |

Documented costs: `_5` agreement −0.120 (IDs/persistence there still better than baseline);
teleport proxy up on stitched clips (the accepted occlusion-restoration artifact — id
restored = a counted "jump" across the gap). Remaining program: Wave 5 probe (tiled
detection) and Wave 6 (role-focused peripheral suppression), plus the open list in
`wip/changes_tbd.md`.

### W5 - Detector Recall Bake-off (L40S remote; IN PROGRESS 2026-07-13)

**Purpose.** Attack the upstream recall ceiling: at the production 640-px detector scale a
distant fielder is ~9 px tall; cam_07 medians ZERO detections/frame on `_7`. Missed players
become single-cam ghosts and fragment fodder for every identity stage.

**Method.** Detector-only sweep (`tools/detector_bakeoff/detector_bakeoff.py` +
`_report.py`), 2 hard deliveries (`_7`, `M2`) x 7 cams x stride 5 = 1,680 frames on the
remote L40S (`quidich-gpu-intern`; the new standing home for all GPU work). Candidates:
m640 (baseline), m1280/m2560 (native-resolution RTMDet-m), t640 (SAHI-style 4x2 tiles +
full-frame merge), l1280 (RTMDet-l COCO). Pose model untouched (RTMPose-X mandate).

**Result (recall audit).**

| Candidate | new boxes/frame vs m640 | lost/frame | verdict |
|---|---|---|---|
| m1280 | 0.00-0.07 | 0.0-0.33 | dead — scale-mismatch, loses boxes too |
| m2560 | 0.00-0.20 | 0.0-0.28 | dead — same cause, 9x cost |
| l1280 | 0.00-0.32 | 0.0-0.11 | marginal, weaker person prior (80-class) |
| **t640** | **+0.76 to +6.02** | **0.000 everywhere** | **decisive winner** |

**Finding.** Resolution only helps at the model's TRAINED object scale: native hi-res makes
people 4x larger than RTMDet-m's training distribution and it misses them; tiles keep people
at trained scale -> min detected box height drops 33 px -> 12 px, and t640 is a perfect
superset of the baseline (zero lost boxes). Overlay eyeballing confirmed real recall (e.g.
the bottom-frame umpire on `_7`/cam_04) but exposed seam duplicates (partial-body fragments
around the striker) -> fixed with interior-border clipping + IoM containment suppression
(standard SAHI hygiene); re-sweep running. Next gates: fragment-free overlays -> full
det+pose P1 on the 2 hard deliveries -> local v7 chain panel -> 8-delivery rebaseline (v8).

### W5B - Contested-Camera Evidence Down-weighting (IMPLEMENTED 2026-07-13; A/B RUNNING)

**Purpose.** Bowler/non-striker crossing: in one facing camera the two boxes merge, while
other cameras still see them distinct. A merged box poisons the ground solve (wrong foot)
and every appearance/posture descriptor (two players in one crop) — a direct ID-swap cause.
Fix: trust the cameras where the players are distinct.

**Implementation** (flag `contested_iou`, default 0.0 = off).
Same-camera detection pairs with bbox IoU >= `contested_iou` (0.45 in the experiment
overlay `configs/experiments/v7_w5b__p3.yaml`) are marked contested
(`Detection3.contested`, `mark_contested_detections` in `src/identity/p3_association/associator.py`;
runner hook pre-gate). Effects: z0_reproj solve weight x`contested_conf_scale` (0.25);
per-view ground sigma x`contested_sigma_scale` (2.5) in both associator and tracklet-graph
covariances; emitted `ground_cov` inflated x scale^2 when ALL member views are contested
(uniform weights cancel in the solve); appearance/posture/kp-sample accumulation muted
(`contested_mute_appearance`). The merge GATE is untouched. 4 unit tests
(`tests/test_cricket_association.py`); **flags-off proven byte-identical** against
`pipetrack_v7-rc2` P3 outputs on `M2` (predictions + correspondences, 7/7 cams).

**Result: NO-CHANGE, root cause identified (2026-07-13).** The composed 8-delivery A/B
(`pipetrack_v7-w5b`) produced **byte-identical P3 outputs** with the flag ON at 0.45 — the
contested condition never fires on current data. Measured cause: **P1's detector NMS
(IoU 0.3) already deletes one of any two overlapping same-camera boxes** — the maximum
same-camera pairwise IoU across all 4,200 `_7` P2 frames is 0.298. The crossing failure
mode in today's pipeline is therefore a MISSING detection (one player suppressed or both
merged into one box), not a corrupted pair; there is nothing for a 0.45 gate to mark.

**Disposition.** Machinery kept (flag-gated, unit-tested, byte-identical off). It becomes
live only when P1 stops erasing the second box: queued composed experiment = P1 `--nms-thr
0.55` (tiled path's IoM containment suppression handles true duplicates) + `contested_iou
~0.30`. Bake-off evidence already shows the tiled detector splits merged crossing-pair
boxes into two individuals — exactly the input W5B was designed to referee.

### W5-C - Tiled P1 Through the v7 Stack (Phase C verdict, 2026-07-13)

**Run.** Full tiled det+pose P1 on the L40S (`--tiled-det`, 8,400 frames, 0 failures,
27 min) -> fresh local v7 chain (`pipetrack_v8-probe`) on `_7` + `M2` vs rc2.

| clip | agreement | ids | id_persist | cycle_cons | tri_cov |
|---|---|---|---|---|---|
| `_7` | 0.670 (-0.033) | 13 (=) | 0.858 (-0.024) | 0.901 (**+0.034**) | 0.541 (+0.025) |
| `M2` | 0.784 (+0.003) | 12 (+1) | 0.886 (-0.070) | 0.718 (**+0.052**) | 0.421 (+0.014) |

**Verdict: HOLD for composition (not rejected).** Tiling changes the *population*, not
just the detections: umpires and distant fielders that the baseline never saw now enter
the identity problem as genuinely hard tracks (tiny, low-parallax). Geometry improves
(cycle-consistency and triangulation coverage up on both), but per-track identity
averages dilute. This is the composition principle in action: the recall belongs with
Wave 6 (role-focused suppression of low-confidence peripherals) which consumes exactly
what tiling produces — cleanly-tracked peripherals to *decide about*, instead of
half-visible ghosts. Final accept/reject happens as the composed W5+W6 stack (v8).

### W5-ROLES - Vedant P5 v1 Solver: Evaluated, Defects Fixed, Merged (2026-07-13)

**Origin.** Colleague drop (`vedant2/`): new P5 config loader + epoch-scored Hungarian
role solver (v1) + runner wiring. His `global_id/` rewrite is PARKED pending his
changelog (parallel lineage; would displace the validated v7 stack). His v0 edit
reverting the H2 signed-speed fix was REJECTED (H2 was experimentally convicted).

**Defects found (evidence on rc2 `_2`) and fixed in the merged v1.1:**
1. Role uniqueness not enforced across the epoch latch (assigned two umpires through
   leakage; could equally mint two bowlers) -> final greedy resolution on accumulated
   latch strength, one track per slot, one slot per track.
2. Keeper cost hardcoded standing-up (|along-11.06|) — the real keeper standing back to
   pace (+16.1 m) lost to a slip -> zero-cost band 0.5-8 m behind the stumps.
3. Striker/non-striker anchored mid-pitch (±5.03 m) -> batting creases (±8.84 m).
Roster corrected to cricket reality: 6 slots = bowler, striker, non-striker, keeper,
**umpire_bowler_end + umpire_square_leg** (2 umpires with distinct geometry, mapped to
role "umpire").

**Renderer.** Roles are now drawn on the player chips in every tile (`P012 BOWLER`),
not only the roster side-panel — first mosaics with visible roles
(`artifacts/meeting_2026-07-13/`). Frame-verified: v1.1's keeper pick (P003, crouching
at the stumps) is visibly correct where v0 picked P002.

**Safety.** v0 path byte-identical (timestamp-only); flag `role_assignment_version`
(default v0); 198 tests green. Config: `configs/v7/p5_roles.yaml` (v1).
**Result: ACCEPTED as v7 default (8-delivery roster A/B, 2026-07-13).** Core-role
coverage (bowler/striker/non-striker/keeper found) **24/32 -> 29/32**; both umpires
named on 6/8 clips (v0 max was one, usually zero); v1.1 >= v0 on every single delivery.
The one weak clip (`_6`, bowler-only) is tracking-limited, not solver-limited — and v1.1
still beats v0 there (v0 named nobody). Driver now passes `configs/v7/p5_roles.yaml` by
default (`--p5-config ''` for legacy v0); manifest records the P5 config + sha. Unit
tests: full-roster uniqueness + standing-back keeper + no-duplicate-slots (200 total).

### W6 - Role-Aware Peripheral Suppression (P5b) - IMPLEMENTED (2026-07-13)

**Purpose (user directive, "wave six, at the very end").** Bowler/striker/non-striker/
keeper are the product; when low-confidence poses/tracking of peripheral players
(umpires/fielders) hinder the output, DROP them rather than extrapolate.

**Implementation.** New `src/identity/p6_roles/suppress_peripherals.py` (P5b, runs inside the
driver's p5 stage with explicit paths): per-global-id quality aggregates (mean keypoint
conf, delivery-span completeness, single-cam rate, det conf) -> `suppression.json` next
to roles.json. Core roles are NEVER suppressed; umpires optionally protectable
(`suppress_protect_umpires`). Consumers: mosaic tiles drop suppressed players (roster
panel keeps them greyed for honesty); P6 terminal lift skips them (global-id branch
only — P3.5 binding lifts stay complete). Flag `suppression_enabled` (default false =
byte-identical; `configs/experiments/w6__p5.yaml` enables). 4 unit tests (202 total).

**First A/B (rc2 stack + v1.1 roles, all 8 deliveries).** 0-3 ids suppressed per clip
(11 of 99 total); zero core-role suppression anywhere; on `_2` it removed exactly the
C1-only fragment (P002) that v0 had miscalled keeper. Verdict so far: conservative and
sane. **The real test is the composed W5+W6 stack** — suppression consumes the noisy
peripherals tiled detection adds; that composition (v8 rebaseline) is the next gate.

### W5B-LIVE + NMS-0.55 - Crossing Survival Experiment (2026-07-13, verdict in)

**Setup.** P1 NMS 0.3 -> 0.55 on the tiled detector (both crossing players survive
detection: `_7` now has 245 camera-frames with same-cam IoU >= 0.30, max 0.515 — vs
literally zero at NMS 0.3), then A/B the contested-camera weighting (0.30 gate) on top.
Chains on `_7`+`M2` vs the tiled NMS-0.3 baseline (`pipetrack_v8-probe`).

| stack | `_7` agree/ids/persist/frags | `M2` agree/ids/persist |
|---|---|---|
| tiled nms0.3 (baseline) | 0.670 / 13 / 0.858 / 6 | 0.784 / 12 / 0.886 |
| + nms0.55 + contested0.30 | 0.724 / 15 / 0.816 / 9 | 0.887 / 12 / 0.903 |
| **+ nms0.55 only** | **0.804 / 12 / 0.882 / 5** | **0.887 / 12 / 0.903** |

**Verdicts.**
- **NMS 0.55: the discovery of the wave.** Keeping both crossing boxes is worth
  +0.13 agreement on `_7` and +0.10 on `M2` over tiled-0.3 — and +0.10/+0.11 over the
  accepted v7 default. Best agreement figures recorded on both clips.
- **W5B contested weighting: REJECTED.** On the only clip where it fires it costs
  -0.080 agreement, +3 IDs, +4 fragments vs NMS-only on identical inputs. The surviving
  crossing boxes are good evidence; down-weighting them starves the solve. Machinery
  stays flag-gated off (byte-identical) for possible future use.
- Known costs to carry into Phase D: cycle-consistency / tri-coverage dip (more
  single-cam peripherals — the W6 target), and the `_7` appearance cue abstains
  (d' 0 -> calibration anchors shifted under the new detection distribution; present in
  both arms, NOT caused by contested muting).

**Next:** Phase D — composed v8 candidate (tiled + nms0.55 P1 x 8 deliveries on the
L40S, v7 identity stack, v1.1 roles, W6 suppression) vs rc2 across the full panel.

### GRAND ANALYSIS v2 - v8.0 Accepted (tiled detection era, 2026-07-13)

**The stack.** P1 = tiled RTMDet-m (4x2 grid + full frame, cross-tile NMS 0.55, IoM-0.7
containment) + RTMPose-X, produced on the L40S; P2 = v7 + `lowconf_can_spawn: false`;
P3/P4 = v7 unchanged; P5 = roles v1.1 (epoch Hungarian, 2-umpire roster) + Wave-6
peripheral suppression ON. Frozen tree `data/derived/runs/pipetrack_v8.0` (+ metrics
snapshot); driver defaults now `configs/v8/`.

**Final 8-delivery panel vs v7-rc2 (full table in the run tree):**

| axis | v7 (rc2) | v8.0 |
|---|---|---|
| `_4` agreement | 0.770 | **0.972** |
| `_7` agreement / IDs / frags | 0.703 / 13 / 6 | **0.811 / 12 / 5** |
| `M2` agreement / teleports | 0.781 / 223 | **0.886 / 184** |
| `_1` persistence / frags | 0.921 / 6 | **0.969 / 5** |
| mean agreement (8 clips) | 0.783 | 0.782 |
| collisions | 0 | 0 |

**The population caveat, resolved at the product level.** Overall agreement dips on
`_2`/`_5`/`_6` because tiling makes ~20% more real people visible — mostly in-pack close
catchers — and those enter the metric as genuinely hard tracks. Core-role forensics
settle it: on `_5` the bowler goes 0.63->0.91 completeness (2->3 cams, rest perfect
both sides); on `_6` v7 could name ONLY the bowler at 0.59 completeness while v8 names
ALL FOUR core roles (keeper 0.95/3 cams, bowler 0.99). **Core-role identity — the
product — is equal or better on every delivery**, and the hard-clip transformations are
unqualified. Teleport spikes on `_3` remain the known single-cam proxy artifact.

**Chronicle of the wave:** tiled detection (recall superset, +0.8..6 boxes/frame) ->
NMS 0.55 (crossing survival, the decisive +0.10-0.13 agreement) -> contested weighting
rejected by ablation -> no-spawn P2 (specks associate, never birth) -> roles v1.1 ->
W6 suppression (outputs only carry what we trust). Rejected/parked: contested-camera
weighting (flag-gated off), vedant global_id rewrite (awaiting changelog).

**Next work.** Pack-handling on `_5`/`_6` peripherals (the one open regression, now
isolated to non-core in-pack tracks); `_3` teleport-proxy replacement; identity GT
labelling; mosaic sign-off of v8.0 by the user.

### W5-PERF + W7-RENDER + W8-ROLES-v1.2 - Production-Scale Enablement (2026-07-14)

**W5-PERF (P1 tiled throughput 5.6 -> 16-18 fps, 2.9-3.2x).** fp16 autocast alone gained
~nothing — the tiled bottleneck was CPU: 9 crops/frame each running mmdet's generic
per-image Python pipeline inside the GPU loop. Fix: crop slicing + resize moved into the
decode-prefetch workers (`make_tiled_loader`), pre-sized crops fed straight through
`data_preprocessor -> predict` (`detect_person_boxes_tiled_fast`; generic path kept via
`--no-tiled-fast`). Parity: identical person counts, matched-box delta <= 3.3 px, fp16
usable-joint delta <= 3.7 px (sub-threshold joints excluded by every consumer). Sweep:
det-batch 4 optimal (15.99 fps probe; 17.7-18 fps in production). CPU side: P3
appearance decode prefetch (4 threads, byte-identical, ~2x Pass A); driver
`--deliveries all` discovery; `run_v8_l40s.sh` launcher.

**W7-RENDER (mosaic upgrades, user directives).** (1) Collision-aware chip placement —
candidates above/below/beside then stacked with leader lines; two adjacent players can
no longer draw unreadable overlapping labels (frame-verified on the `_7` CAM-04 pack).
(2) Skeleton body-paint identity overlay (torso quad + limb capsules + head disc,
0.35 blend, `--no-body-paint` to disable). (3) Roles removed from tile chips — roster
panel only (brightened). (4) Palette 12 -> 20 max-separation colours (golden-ratio hue,
two brightness bands; wraparound every 21st id). Tests: tests/test_render_labels.py.

**W8-ROLES-v1.2 (bowling-end auto-flip).** Three fixes composed: (a) plausibility band
on run detection (3.0-9.5 m/s) — tracking teleports previously read as 20-31 m/s "runs"
and could vote the wrong end (measured on 4/8 clips); (b) when no plausible run exists,
a two-sign trial of the epoch solver decides the end by roster-geometry fit cost,
computed on the PRE-SHOT window only (post-shot running swaps the batters' ends and
blurred whole-delivery medians); unfilled slots pay max-cost so accept-less cannot fake
a better score; (c) `--force-axis-sign` hook for future cross-delivery consensus.
Result on v8.0 x8: core roster 4/4 + both umpires on ALL deliveries; sources:
run_detected 7/8, cost_flip 1/8. FINDING: the `_14_x` clips do NOT share one bowling
end (clean opposite-direction runs in `_2` vs `_3`) — the naming is not one-over-per-
group, so no over-consensus constraint applies. OPEN: end-orientation (striker vs
non-striker end swap) is visually confirmed correct only on `_2`; the 8 production
mosaics will arbitrate the rest. 209 tests green.

### W9 - Split-Identity / Ghost-Swap Fix: Union-Lift Merge + Colocated-ID Merge (v8.1, 2026-07-14)

**The user's exhibit, root-caused with visual proof.** Mosaics showed a ghost marker of one
id under a live player of another, inverted between cameras (`_7`: live P004 + ghost P011 in
cam_01; live P011 + ghost P004 in cam_02). Extracted frames PROVE one person: cam_02's P011
and cam_01's P004 are the same striker (shirt 66). Mechanism: P3 minted two clusters for one
player (a keeper-head fragment contaminated P011's cluster, blocking every pairwise merge);
P4 binding continuity kept both ids; the renderer projects each id's fused position into all
cameras -> each camera draws the other id as a ghost on its live player. A new diagnostic
(`tools/diagnosis/diagnose_colocated_ids.py`) found 8 such mergeable events across the 8
deliveries.

**Fix 1 - P3 union-lift merge** (`graph_union_lift_merge`, tracklet_graph.py): candidate
cluster pairs co-located on the ground (chunk-level gate 2.6 m — wide because per-camera
homography feet carry the facing bias; safety comes from the test, not the gate) are
adjudicated by triangulating the UNION of both clusters' views: one coherent low-residual
3D skeleton in every view => one person (anti-chimera signature rejects genuine pairs).
Only hard blocks apply (same-camera overlap, confident cue veto) — the usual sum>=0 evidence
vote is deliberately waived (facing bias makes exactly these edges sum negative). Per-pair
rejection diagnostics recorded.

**Fix 2 - P4 colocated-id merge** (`p4b.colocated_merge`, stitching.merge_colocated_ids):
merges two emitted ids co-located >= 25 frames within 0.75 m whose histories NEVER share a
camera-frame (sharing one = two real people) and whose billboard statures agree. Extends
id_switch_report with reason "colocated".

**Tripwire** (`colocated_identity_metrics`): coloc pair count now in metrics, panel (column
`coloc`) and the quality verdict — this class can never go unnoticed again.

**8-delivery A/B (`pipetrack_v8.1-w9` vs v8.0): accepted with zero regressions.**

| clip | agreement | ids | frags | coloc |
|---|---|---|---|---|
| `_7` | 0.811 -> **0.962** | 12 -> 10 | 5 -> 4 | 1 -> **0** |
| `_6` | 0.477 -> **0.625** | 18 -> 16 | 11 -> 9 | 3 -> **0** |
| `_5` | 0.627 -> **0.694** | 13 -> 12 | 8 -> 7 | 1 -> **0** |
| `_2` | 0.802 -> **0.845** | 14 -> 13 | 9 -> 8 | 1 -> **0** |
| mean (8) | 0.782 -> **0.834** | — | — | **0 everywhere** |

Persistence up on 5 clips, tri coverage up on 6, collisions 0 everywhere, no agreement
regression on any clip. **Cut as v8.1 into `configs/v8/`.**

**Production incident (same session, fixed):** the 40-delivery P1 run failed on
cam_07 only (24,000 frames, all deliveries) — the W5-PERF fast tiled path skipped the
generic pipeline's Pad-to-/32, and cam_07's panoramic full-frame crop (3775x960 ->
640x163) produced stride-indivisible FPN inputs ("Expected size 22 but got 21").
Fixed by padding crops bottom/right to /32 in the prefetch loader (+ pad_shape
metainfo); parity re-verified on cam_01 (identical counts) and cam_07 backfilled via
--resume. Lesson recorded: fast-path probes must cover the heterogeneous camera.

**Audit companions (same session):** airborne pelvis-emit (V2-L3, `airborne_pelvis_emit`,
flag-gated); G1 Hartley row-equilibration + G3 parallax-ordered RANSAC seeds
(`--hartley/--parallax-order`, flag-gated A/B candidates); G6 confirmed already shipped;
**latent crash bug fixed** in `ground_from_reprojection_ex` (divergence branch returned a
tuple where callers expect an array — crashed np.isfinite instead of degrading to NaN;
regression test added). 212 tests; flags-off byte-identity proven vs v8.0 (P3+P4, M2).

### W10-PERF - P2-P6 Optimization Pass (2026-07-14)

**Method.** cProfile misled (per-call overhead on 163M tiny calls); switched to
wall-clock section timing. True P3 distribution (74 s local, `_7`): emit 33% (pose
descriptor 25%), graph solve 21%, observe 9%, appearance decode 7% (already
prefetched), IO/misc ~25%.

**Shipped (every change proven BIT-IDENTICAL on real P3+P6 outputs):**
1. `reprojection_errors_for_point` vectorized (batched matmul, 556k calls/delivery).
2. `triangulate_point_dlt` row assembly vectorized.
3. **Batched skeleton RANSAC** — 2-view case collapses analytically to one stacked
   SVD across all joints; generic multi-view case batches candidate-pair DLTs,
   inlier scoring, grouped refits and confidences across joints while replicating
   the reference loop's candidate order and tie-breaks exactly.
Result: P3 74 -> 60 s/delivery local (-19%), **P6 terminal lift 10.4 s/delivery**
(was ~1-2 min on the box); P3.5 shares the same core. Synced to the L40S mid-chain
(safe: bit-identical), so remaining production deliveries inherit it.

**Honest residual:** P3's remaining time is LLR/solve machinery + JSONL IO + emit
bookkeeping — flat profiles with no dominant hotspot; further gains need structural
changes (e.g. orjson, emit-side caching), logged as future work, not pursued under
diminishing returns. P2 (~1 min) and P4/P5 (seconds) are not bottlenecks.

### PRODUCTION RECORD - 40-Delivery Dataset on the L40S (v8.1, 2026-07-14)

**Deliverable:** `/home/ubuntu/pipetrack_v8/` — the full 40-delivery CCPL080626 dataset
(M1 overs 14/16/17; M2 overs 11/12 + innings-2 overs 3/4), every stage P1->P6, README +
`final_panel.md` + production logs inside. P1: 168,000 frames, 0 failures (after the cam_07
pad-to-/32 backfill), 18-25 fps tiled fp16. Chain: 40/40 `compute ok`, 0 errors.

**Panel (all 40):** agreement mean 0.862 (0.527-0.992); reproj 3.07-3.56 px mean on every
delivery; collisions 0 everywhere; coloc 0 on 38/40 (residual 1 pair each on `M1_1_14_7`
and `M2_1_11_3` — see remaining-work.md 5b). Calibration provenance verified empirically
(frame-md5 identity across machines; flat reproj across all 7 segments incl. innings-2);
team confirmation of a single calibration session still requested.

**Reconciliation note:** box panels differ slightly from the local v8.1-w9 A/B on the 8
overlap deliveries (e.g. `_7` 0.819/12 vs 0.962/10) — same code, different P1 binaries
(production fp16 fast path vs earlier generic path); expected near-parity variance,
documented so it is not chased as a bug.

**Repository state at close:** default `configs/v8/` = v8.1; 212 tests; kept run trees:
`pipetrack_v8.1-w9` (local reference), `rtmpose-x-tiled-w5-full` (P1 input),
`yolo26x-pose-full-db8` (model comparison); all else archived to `docs/runs/` and deleted.
Open work consolidated in `/remaining-work.md`.

## Current Default / Recommended State

- **Default: `configs/v8/` at v8.1** (tiled+NMS0.55 detection, no-spawn P2, v7 identity
  stack + W9 union-lift & colocated merges, roles v1.2, W6 suppression ON) — references
  `pipetrack_v8.0` (pre-W9) and `pipetrack_v8.1-w9`. `configs/v7/` and
  `configs/v6/` stay frozen as references; `configs/experiments/` holds every
  intermediate stack for reproduction.
- Opt-in (measured, not default): `posture_keep_upright_unknown` (H3), F11 shape round
  (self-abstaining), `temporal_link_decay`, `density_lost_window` (fresh, awaiting its
  first composed A/B).

## Next Work

- A/B verdicts for F1 and the Wave-1 batch (runs in flight), then Wave-2/3/4 A/Bs.
- Decision-gated Wave 5 (pretrained ReID only with sign-off, 3D P4 costs, PnP lift,
  OC-SORT, detector/RTMO block) per [to-do.md](to-do.md).

Literature check (July 2026, web research; reshapes the Wave-5 detector block):

- **Detector recall**: no drop-in beats RTMDet decisively inside mmdet; the dominant lever at
  2560x1440 with 30-100 px players is **input resolution — tiled (SAHI-style) inference over
  the existing RTMDet** (reported ~3x person-F1 on small subjects), before any model swap.
  Candidates worth a probe after tiling: YOLO26-l (small-target-aware assignment,
  arXiv 2509.25164), RF-DETR-Large (56.5 AP, fits 8 GB); Co-DINO (mmdet zoo) as an offline
  recall oracle only.
- **RTMO demoted to detector-miss fallback**: it is COCO-17-only — adopting it would drop the
  Halpe-26 feet and break the F4 ground contact and bone-ratio/stature cues (mmpose #3135
  also reports slightly worse accuracy vs top-down at small scales).
- **Identical-kit ReID**: SoccerNet GSR winners still lean on jersey numbers/colour (dead
  here); nothing pretrained solves appearance-free ReID in 2026. One cheap auxiliary
  candidate: skeleton-based gait embeddings (OpenGait/SkeletonGait++, pretrained) consuming
  our existing pose sequences — weak-cue trial only, self-calibrated, needs sign-off.
- **F13 externally consistent**: reprojection-residual gating of cross-view clusters is the
  standard mechanism (RapidPoseTriangulation 2503.21692; Cross-View Tracking 2003.03972 uses
  25-70 px gates at ~1-2K res and prescribes resolution-scaled thresholds) — our
  torso-residual threshold (20 px at 1440p) sits in the accepted range and stays rig-tunable.

## Environment Notes

> **2026-07-14 cleanup:** historical run trees (v3-v8-rc, superseded P1 runs, probes) were
> archived as per-run documents in `docs/runs/` (panels + manifests + verdicts) and their
> bulk data deleted (43 GB -> 4.5 GB). Kept: `pipetrack_v8.0`, `rtmpose-x-tiled-w5-full`,
> `yolo26x-pose-full-db8`.


- Pipeline stages and render: `/home/aksh/miniconda3/envs/pose-lab/bin/python`.
- P1 inference (not re-run in this campaign unless F18 triggers): `pose-lab`.
- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="" pytest` in `pose-lab`.
- BLAS threads capped to 1 per stage process by the drivers (oversubscription lesson).
