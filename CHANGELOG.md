# Changelog

Notable changes to the Group-1 cricket 3D-pose & identity pipeline. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). The detailed A/B lab-notebook (every
experiment, verdict and incident) lives in
[`docs/critical-analysis/fixes-log.md`](docs/critical-analysis/fixes-log.md) — this file is the
high-level summary only.

## Current — v8.1 (default stack)

The pipeline turns synchronised 7-camera footage into per-player 3D pose + stable global
identity and a diagnostic mosaic. Stage order (`src/main.py`, `python -m main`):

**P1** 2D inference (tiled RTMDet-m + NMS 0.55 → RTMPose-X Halpe-26) → **01** stabilization
(One-Euro) → **02** per-camera tracking (ByteTrack-style, no-spawn) → **03** cross-camera
association (tracklet-graph LLR + W9 union-lift) → **04** 3D lift (RANSAC-DLT + cheirality +
Butterworth, binding-keyed, *before* global identity) → **05** global identity (Singer-KF +
min-cost-flow stitching + colocated-id merge) → **06** roles (v1.2 epoch solver + peripheral
suppression) → UE export + mosaic/BEV render.

Production: 40 deliveries (`CCPL080626`), mean cross-camera agreement 0.862, reprojection
3.07–3.56 px, same-camera collisions 0, colocated-id pairs 0 on 38/40. Identity remains the
dominant quality ceiling (facing-pair split identity, single-camera coverage). Open work:
[`wip/to_do.md`](wip/to_do.md) (the consolidated backlog + prioritized algorithm
fix list); measured 40-delivery diagnosis: [`docs/diagnosis/`](docs/diagnosis/README.md).

## 2026-07 — Repository restructure

- Code reorganised into a `src/` package: `src/core/` (contract, calibration, keypoints,
  dataset, schemas, UE transform, + `inference/` = P1), `src/identity/common/` (geometry,
  triangulation, pose_shape, metrics), `src/identity/pN_<stage>/` (the six identity stages),
  `src/identity/{export,visualization}`, and `src/main.py` (orchestrator). Dev/ops tooling in
  `tools/`. Imports are `core.*` / `identity.*` / `tools.*` (`pip install -e .`).
- Stages renumbered to execution order (`01`–`06`); the 3D lift now runs **before** global
  identity (Associate → Triangulate → Track). Run-dir stage folders are `0N_<stage>`.
- Configs consolidated to one numbered YAML per stage (`configs/0N_<stage>.yaml`); legacy
  version dirs removed.
- Single conda env **`pose-lab`** (mm-stack + PyTorch + Ultralytics) replaces the per-model envs.
- Pipeline outputs live under gitignored `data/derived/{runs,mosaics}/`. Benchmarking is off
  `main` (on the `benchmark` branch); `benchmarks/` and `artifacts/` were removed from the repo.
- Documentation overhauled: per-phase reference under `docs/pipeline/`, an `architecture.md`
  overview, and a scrub of every old-structure reference across the live codebase and docs.

## History (pre-restructure, ≤ v8.0)

The v6 → v8.1 fix campaign (tracklet-graph identity, cross-camera association hardening,
Singer-KF global-id + min-cost-flow stitching, tiled detection, roles) is recorded
experiment-by-experiment in [`docs/critical-analysis/fixes-log.md`](docs/critical-analysis/fixes-log.md)
and the methods lab-notebook [`wip/methods_log.md`](wip/methods_log.md). Archived per-run
reports are under [`docs/runs/`](docs/runs/README.md). Those are dated historical records; their
file paths reflect the structure in place at the time.
