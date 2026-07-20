# Changelog

Notable changes to the Group-1 cricket 3D-pose and identity pipeline. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). The detailed A/B lab notebook (every method, its
before/after evidence, verdict, pros and cons) lives in [`docs/methods_log.md`](docs/methods_log.md).
This file is the high-level summary only.

## Current: pipetrack_v9 (default stack)

The pipeline turns synchronised 7-camera footage into per-player 3D pose plus a stable global identity
and a diagnostic mosaic. Stage order (`src/main.py`, run as `python src/main.py` or `python -m main`):

P1 2D inference (RTMDet-m detector at 640, then RTMPose-X Halpe-26 top-down, one 26-joint pose per box)
then 01 stabilization (One-Euro) then 02 per-camera tracking (ByteTrack-style two-pass plus
constant-velocity Kalman) then 03 cross-camera association (tracklet-graph LLR plus union-lift merge, cap
3.5) then 04 3D lift (single binding-keyed triangulation, RANSAC plus weighted DLT plus cheirality plus
Butterworth, run before global identity) then 05 global identity (Singer-acceleration Kalman plus
min-cost-flow stitching plus colocated-id merge) then 06 roles (epoch Hungarian solver plus peripheral
suppression) then 07 refine (physics-constrained 3D skeleton rebuild plus hip de-wobble plus low-confidence
refill, new stage) then 08 render (mosaic and BEV).

Run layout is per delivery: `<run>/<DELIVERY>/{00_inference,01_stabilization,...,07_refine,logs}`.

Production reference (40 deliveries, v8.1 panel that v9 builds on): mean cross-camera agreement 0.862,
reprojection 3.07 to 3.56 px, same-camera collisions 0, colocated-id pairs 0 on 38 of 40. Identity remains
the dominant quality ceiling (facing-pair split identity, single-camera coverage). Open work:
[`docs/roadmap.md`](docs/roadmap.md). Measured 40-delivery diagnosis: [`docs/analysis/`](docs/analysis/README.md).

## 2026-07-16 to 2026-07-17: optimization and A/B campaign

Post-v9-push work on the L40S box. Full detail in [`docs/methods_log.md`](docs/methods_log.md) Part A and
[`docs/reference/performance.md`](docs/reference/performance.md).

- Script optimization: six fixes to the run scripts. The data-parallel P1 launcher
  (`run_phase1_parallel.py`) had a broken runner path and failed on every shard; fixed and dry-run
  validated, restoring the roughly 2x GPU-throughput lever. Plus thread-oversubscription fixes in the
  render (cv2 thread cap) and the P1 shards (auto-clamped io-workers) for the 8-core box.
- 40-set flag verification: the first hard-set measurement of the flags that ship on. `graph_shape_enabled`
  is fully inert on all 40; `graph_split_enabled` is a slight agreement drag; distance-R, the facing gate,
  and the adaptive lost window each suppress real teleport events (71, 46, 40 across 40) at negligible
  agreement cost. Collisions held at 0 under every toggle.
- Tiled detection A/B: agreement improves on all 8 hardest deliveries (mean +0.115, attributable to tiling
  and not NMS) but underlying teleport events regress (+704 total) at about 3x GPU cost. Two-edged, pending
  a human decision, off by default.
- OC-SORT tracker: implemented as a config-selectable stage-02 option (OCM, ORU, OCR), with the `bytetrack`
  default verified byte-identical. 40-set A/B: it reduces fragmentation (`p2_tracks` minus 26) but is
  net-negative downstream (agreement minus 0.0129, teleports plus 151). Rejected as implemented, off by
  default.

No production defaults were changed. Every keep, enable, and disable decision is deferred to human review.

## 2026-07: repository restructure (pipetrack_v9)

- Halpe-26 (26-keypoint) skeleton throughout; a single binding-keyed triangulation (stage 04) replaces the
  earlier multiple lift paths and runs before global identity (associate, then triangulate, then track).
- New stage 07 refine: physics-constrained 3D skeleton rebuild (constant, left-right-symmetric bone
  lengths, backward-bend clamps), hip de-wobble (a lower smoothing cutoff on the root than the limbs), and
  low-confidence joint drop-and-refill. Runs after identity is frozen and rewrites only `pose_3d`, so IDs
  are unchanged.
- Per-delivery run layout `<run>/<DELIVERY>/<stage>/`; P1 writes `<run>/<DELIVERY>/00_inference/`.
- Configs consolidated to one numbered YAML per stage (`configs/0N_<stage>.yaml`).
- Single conda env `pose-lab` (mm-stack plus PyTorch plus Ultralytics).
- Pipeline outputs live under gitignored `data/derived/`.
- Documentation overhauled: per-phase reference under `docs/pipeline/`, an `architecture.md` overview, and
  the combined method ledger `docs/methods_log.md`.

## History (pre-restructure, v8.1 and earlier)

The v6 to v8.1 fix campaign (tracklet-graph identity, cross-camera association hardening,
Singer-acceleration global-id plus min-cost-flow stitching, tiled detection, roles, the W9 union-lift and
colocated-id merges) is recorded method by method in [`docs/methods_log.md`](docs/methods_log.md) Part D.
Archived per-run reports are under [`docs/runs/`](docs/runs/README.md); those are dated historical records
and their file paths reflect the structure in place at the time.
