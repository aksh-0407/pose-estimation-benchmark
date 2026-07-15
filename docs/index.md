# Documentation

The **Group-1 cricket 3D-pose & identity pipeline**: 7-camera footage → per-player 2D pose →
cross-camera identity → 3D pose & ground location → roles → mosaic render.

## Start here

| # | Doc | What you get |
| - | --- | ------------ |
| 1 | [getting-started.md](getting-started.md) | From a fresh checkout to a rendered mosaic on one delivery, step by step. |
| 2 | [architecture.md](architecture.md) | The shared concepts: camera rig & facing pairs, calibration, the data contract, skeletons, run-dir layout, metrics, the causal chain. |
| 3 | [pipeline/README.md](pipeline/README.md) | The ordered pipeline + per-stage deep dives (method, config, what's been tried, current state). |

## Reference

| Doc | Use it when… |
| --- | ------------ |
| [reference/cli.md](reference/cli.md) | …you want the exact command, I/O and flags for a stage or tool. |
| [reference/configuration.md](reference/configuration.md) | …you need to read or edit a `configs/0N_*.yaml`. |
| [reference/metrics.md](reference/metrics.md) | …you want to know what a reported number or proxy means. |
| [reference/data-inventory.md](reference/data-inventory.md) | …you want the dataset inventory. |
| [rtmpose-x-runbook.md](rtmpose-x-runbook.md) | …you're installing/running/tuning P1 (RTMPose-X) on a new or remote machine. |
| [troubleshooting.md](troubleshooting.md) | …something broke (model download, env, CUDA). |
| [shared-data.md](shared-data.md) | …a downstream group needs to consume a run's outputs. |

## Status & analysis

| Doc | What it is |
| --- | ---------- |
| [../wip/to_do.md](../wip/to_do.md) | The single consolidated backlog — everything deferred / parked / pending, incl. the prioritized algorithm fix list (A1–A10). |
| [diagnosis/README.md](diagnosis/README.md) | The measured 40-delivery production diagnosis (teleports, split identity, coverage). |
| [pipeline/fixes-log.md](pipeline/fixes-log.md) | The dated A/B campaign ledger (historical). |

## The pipeline at a glance

P1 2D inference → **01** stabilization → **02** per-camera tracking → **03** cross-camera
association → **04** 3D lift → **05** global identity → **06** roles → UE export / mosaic render.
The 3D lift runs **before** global identity (Associate → Triangulate → Track). Each stage reads
and writes a canonical run directory; full detail in [pipeline/README.md](pipeline/README.md).

## Common commands

Everything runs under the single `pose-lab` conda env, invoked as a module.

```bash
# P1 — 2D pose over a delivery (RTMPose-X, top-down); emits Halpe-26 (26 joints)
python -m core.inference.run_phase1_rtmpose_inference --model-id rtmpose_x_body8 \
  --dataset 8_init --version 9 --deliveries CCPL080626M1_1_14_1

# Whole chain over one delivery (phase-select with --from-stage/--until-stage)
python -m main --dataset 8_init --version 9 --deliveries CCPL080626M1_1_14_1

# Render the mosaic (7 tiles + bird's-eye monitor + roster), coloured by global ID
python -m identity.visualization.render_videos \
  --drive-root drive --run-dir <05_global_id-run> --delivery-id CCPL080626M1_1_14_1 \
  --mode mosaic --show p4
```

See [reference/cli.md](reference/cli.md) for the full per-stage command sequence.
