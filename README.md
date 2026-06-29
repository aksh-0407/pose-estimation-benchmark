# Pose Estimation Benchmarking Workspace

A shared workspace for benchmarking 2D and 3D human pose estimation models. The
end target is cricket production, but the framework runs against standard public
datasets (starting with COCO) so different people can measure models the same way
and compare results.

This repo is built for a team: several people benchmark models on their own
machines, commit the small result files, and merge them into `main`. A clean
comparison view is then produced automatically.

Benchmarking is the starting point. The project's main work is improving model
outputs: reducing jitter, fusing cameras, smoothing over time, and fine-tuning for
the domain. See [docs/improving-models.md](docs/improving-models.md).

Laptops are for functional validation, not speed conclusions. Real
latency/throughput numbers should come from stronger NVIDIA GPUs, and they are only
comparable when the run records its full hardware/software metadata (every run does).

## Start here

New to this repo? Read [docs/getting-started.md](docs/getting-started.md). It takes you
from a fresh checkout to your first real benchmark result, step by step.

## How it works

- `configs/` holds the source of truth: which models, datasets, metrics, and skeletons
  exist. Everything else is driven from it.
- `scripts/benchmark.py` is the entry point. It has five stages:
  `prepare → smoke → run → aggregate → report`.
- Heavy files stay local and are never committed: model weights, datasets, upstream code
  clones, and raw per-image predictions and logs.
- Each benchmark produces a small run folder under `benchmarks/runs/<run_id>/`
  (manifests plus metrics JSON). These are committed. Their unique timestamped names mean
  two people never conflict.
- The comparison tables and reports are derived, not hand-written. CI regenerates them on
  `main`, so nobody merges a CSV by hand.
- Each model runs in its own Conda environment, so conflicting dependencies (MMPose,
  Ultralytics, MediaPipe, OpenPose) can coexist.

See [docs/concepts.md](docs/concepts.md) for the full picture and a diagram.

## What works today

Every model has a smoke adapter (a one-image readiness check). Four models also have a
benchmark adapter that produces real COCO-17 scores: `yolo26x_pose` (end-to-end) and
`rtmw_l`, `rtmw_x`, `rtmpose_l_wholebody` (top-down with ground-truth boxes, a different
and not directly comparable protocol). The remaining models are smoke-ready, with
benchmark adapters tracked as work. The distinction and the readiness matrix are in
[docs/models.md](docs/models.md).

The cricket delivery path also includes the modular PipeTrack stages: per-camera
tracking (P2), cycle-consistent cross-camera association (P3), global identity and
tracklet stitching (P4), and a light multi-view 3D lift (P6). Each stage consumes and
produces canonical run directories, so it can be inspected or rerun independently.
See [docs/scripts.md](docs/scripts.md#pipetrack-cricket-tracking-p2p6).

## Documentation

Read the Start here docs in order; use the Reference docs as needed. The full index is
in [docs/index.md](docs/index.md).

Start here:
1. [docs/getting-started.md](docs/getting-started.md): install and run your first benchmark
2. [docs/concepts.md](docs/concepts.md): how the repo works
3. [docs/workflow.md](docs/workflow.md): the five-stage pipeline in depth
4. [docs/collaboration.md](docs/collaboration.md): what goes to git, and the team workflow

Reference:
- [docs/improving-models.md](docs/improving-models.md): reducing noise and improving outputs (the core work)
- [docs/scripts.md](docs/scripts.md): every script explained
- [docs/configuration.md](docs/configuration.md): the config YAML files
- [docs/models.md](docs/models.md), [docs/datasets.md](docs/datasets.md), [docs/metrics.md](docs/metrics.md)
- [docs/results-format.md](docs/results-format.md): run output layout
- [docs/adding-a-model.md](docs/adding-a-model.md): integrate a new model
- [docs/troubleshooting.md](docs/troubleshooting.md): when something breaks
