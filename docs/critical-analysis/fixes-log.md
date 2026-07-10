# Critical-Analysis Fixes: Methods Log

This log records the implementation and A/B evaluation of the fixes catalogued in
[fixes-roadmap.md](fixes-roadmap.md), executed per the plan in [to-do.md](to-do.md). For every
fix it states the purpose, the implementation (files and flags), the measured result against the
frozen baseline across the 8-delivery evaluation set (`CCPL080626...`), and the verdict. It
follows the conventions of `wip/methods_log.md`.

## Executive Summary

(Updated as fixes land.)

- `pipetrack_v6.0` ground baseline: pending first full run.

## Evaluation Standard

All candidate changes are evaluated against the frozen `pipetrack_v6.0` baseline
(`benchmarks/runs/pipetrack_v6.0/_baseline_snapshot`) across all 8 deliveries. A fix is only
marked Accepted if it produces a significant, generalized improvement without introducing
clustering, identity, or collision regressions. Every change is behind a config flag; with all
flags off the pipeline is byte-identical to the baseline. Same-camera collisions must stay 0
everywhere (hard invariant).

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
| F0 | pipetrack_v6.0 ground baseline (RTMPose-X P1, v5 configs, P1.5 off) | pending | Baseline | Fixed comparison point for the whole campaign. |
| F1 | Wire P1.5 2D stabilization into the default flow | — | pending | — |
| F2 | C07 per-camera image size everywhere | — | pending | — |
| F3 | Cheirality check in RANSAC triangulation | — | pending | — |
| F4 | Halpe-26 feet as ground contact (v3 foot mode) | — | pending | — |
| F5 | Online role proxy → P4a Singer dynamics | — | pending | — |
| F6 | P4b occupancy-licensed gap bridging | — | pending | — |
| F6b | Billboard posture → P4a teleport veto | — | pending | — |
| F7 | Offline zero-phase Butterworth 3D smoothing | — | pending | — |
| F8 | Cue cold-start robustness (anchor relaxation + prior) | — | pending | — |
| F9 | P3.5 re-sequencing: binding-keyed 3D lift + covariance + purity | — | pending | — |
| F10 | Per-measurement (distance/uncertainty) Kalman R | — | pending | — |
| F11 | Pose-shape (bone-ratio) as primary P3 cluster cue | — | pending | — |
| F12 | Stature into P4 re-entry + P4b stitch cost | — | pending | — |
| F13 | Splittable clustering (correlation local search) | — | pending | — |

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

Result:

- Pending first full run.

Conclusion:

- Pending.

## Current Default / Recommended State

- Baseline: `configs/v6/` as frozen; no fixes accepted yet.

## Next Work

- Run the v6.0 baseline; then Wave 0 (F1) per [to-do.md](to-do.md).

## Environment Notes

- Pipeline stages and render: `/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/python`.
- P1 inference (not re-run in this campaign unless F18 triggers): `cricket-rtmpose-l`.
- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="" pytest` in `cricket-yolo26x-pose`.
- BLAS threads capped to 1 per stage process by the drivers (oversubscription lesson).
