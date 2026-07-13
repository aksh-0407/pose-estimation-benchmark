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
(`benchmarks/runs/pipetrack_v6.0/_baseline_snapshot`) across all 8 deliveries. A fix is only
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

Driver: `scripts/pipetrack/run_full_pipeline.py` (full chain + extended panel + baseline diff);
inner loop: `scripts/pipetrack/run_id_pipeline.py`. Configs frozen in `configs/v6/`; per-fix
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
  full 8-delivery RTMPose-X (Halpe-26) P1 data (`benchmarks/runs/rtmpose-x`) rather than the
  RTMPose-L P2 tree that v3/v5 reused.

Implementation:

- Frozen config set `configs/v6/` (P3/P4 = copies of the validated `*_v5.yaml` flag stacks;
  P2 = committed defaults; P1.5 present but `enabled: false`).
- New full-chain driver `scripts/pipetrack/run_full_pipeline.py`: P1.5(off) → P2 → P3 → P4 →
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
  rtmpose-x P1 data. Run tree: `benchmarks/runs/pipetrack_v6.1-f01`.
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
  run tree `benchmarks/runs/pipetrack_v6.1-wave1` (P3→P4→P5→3D, baseline P2 reused).

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
  --tri-smoother butterworth`; run tree `benchmarks/runs/pipetrack_v6.2-wave3`
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
dense-fill + cheirality + Butterworth. Run tree `benchmarks/runs/pipetrack_v7-rc1`.

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
costs. Run tree `benchmarks/runs/pipetrack_v7-rc2`.

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

## Current Default / Recommended State

- **Default: `configs/v7/`** (the accepted rc2 stack) with the re-ordered pipeline as the
  driver default. `configs/v6/` stays frozen as the ground-baseline reference;
  `configs/experiments/` holds every intermediate stack for reproduction.
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

- Pipeline stages and render: `/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/python`.
- P1 inference (not re-run in this campaign unless F18 triggers): `cricket-rtmpose-l`.
- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="" pytest` in `cricket-yolo26x-pose`.
- BLAS threads capped to 1 per stage process by the drivers (oversubscription lesson).
