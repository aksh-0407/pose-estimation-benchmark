# Documentation

This is the **Group-1 cricket 3D-pose & identity pipeline**: 7-camera cricket footage →
per-player 2D pose → cross-camera identity → 3D pose & ground location → mosaic render.

If you only read one page, read **[getting-started.md](getting-started.md)**. If you want
the engineering depth, read the **[critical analysis](critical-analysis/README.md)**.

## Start here

| # | Doc | What you get |
| - | --- | ------------ |
| 1 | [getting-started.md](getting-started.md) | From a fresh checkout to a rendered mosaic on one delivery, step by step. |
| 2 | [critical-analysis/phases.md](critical-analysis/phases.md) | The ordered pipeline (current **and** proposed), inputs/outputs, and flowcharts. |
| 3 | [critical-analysis/README.md](critical-analysis/README.md) | Per-phase methods, math, weaknesses, and the prioritised fix roadmap. |

## Reference

| Doc | Use it when… |
| --- | ------------ |
| [scripts.md](scripts.md) | …you want to know exactly what a pipeline script does, its I/O, and key flags. |
| [rtmpose-x-runbook.md](rtmpose-x-runbook.md) | …you're installing/running/tuning P1 (RTMPose-X) on a new or remote machine. |
| [configuration.md](configuration.md) | …you need to read or edit a `configs/*.yaml` (p2/p3/p4, model envs, keypoint maps). |
| [metrics.md](metrics.md) | …you want to know what a reported number or proxy means. |
| [improving-models.md](improving-models.md) | …you're reducing jitter, fixing identity, or improving 3D location. |
| [troubleshooting.md](troubleshooting.md) | …something broke (model download, env, CUDA). |

## The pipeline stages

`P1` 2D inference → `P2` per-camera tracking → `P3` cross-camera association → `P4` global
identity → `P5` roles → `P6` 3D lift → UE export / mosaic render. Each stage reads and
writes a canonical run directory. Full detail in
[critical-analysis/phases.md](critical-analysis/phases.md).

## Common commands

```bash
# P1 — 2D pose over a delivery (RTMPose-X, top-down); emits COCO-17 + Halpe-26
conda run -n pose-lab python src/core/inference/run_phase1_rtmpose_inference.py \
  --model-id rtmpose_x_body8 --deliveries CCPL080626M1_1_14_1 \
  --run-id rtmpose-x --run-dir data/derived/runs/rtmpose-x

# P2→P4 identity (batch driver over deliveries), then inspect the metric panel
PY=/home/aksh/miniconda3/envs/pose-lab/bin/python
$PY -m identity.id_pipeline \
  --input-tree data/derived/runs/pipetrack_v3 --output-tree data/derived/runs/pipetrack_v5 --jobs 8

# Render the mosaic (7 tiles + bird's-eye monitor + roster), coloured by global ID
$PY -m identity.visualization.render_videos \
  --drive-root drive --run-dir <p4-run> --delivery-id CCPL080626M1_1_14_1 --mode mosaic --show p4
```

See [scripts.md](scripts.md) for the full per-stage command sequence (P2→P3→P4→P6).
