# Contributing

This repository is the **Group-1 cricket 3D-pose & identity pipeline**. Changes land on
`main` through pull requests so they can be reviewed and so the pipeline stays runnable
end-to-end. (The COCO benchmarking framework lives on the `benchmark` branch.)

## Prerequisites

A Linux machine with an NVIDIA GPU and conda. Cloning gives you code, configs, and docs;
model weights and the frame dataset are **not** committed and are set up locally. See
[docs/getting-started.md](docs/getting-started.md) for the full walkthrough.

## One-time setup

```bash
git clone <this-repo> pose-estimation && cd pose-estimation
pip install -r requirements.txt

# P1 model env + weights (RTMPose-X + RTMDet detector)
python3 tools/setup_model_envs.py --models rtmpose_x_body8 --download-assets
python3 tools/check_assets.py --models rtmpose_x_body8 --fail-missing
```

The tracking/association/global-ID/triangulation stages (P2–P6) run in an env with
NumPy ≥ 1.23.5 and SciPy ≥ 1.10 (e.g. `pose-lab`). See the per-stage commands
in [docs/scripts.md](docs/scripts.md).

## Workflow

1. Create a branch: `git checkout -b <your-name>/<task>`.
2. Do the work. Run the relevant stage(s) on the reference delivery
   `CCPL080626M1_1_14_1` and eyeball the mosaic / bird's-eye render.
3. Before committing, run the checks:
   ```bash
   python3 -m pytest -q
   python3 tools/audit_repo.py --fail
   ```
4. Commit only source, configs, docs, and the small committed run metrics
   (`data/derived/runs/<run>/**/​*_metrics.json`, manifests). Do **not** commit weights,
   frames, or raw per-frame prediction dumps — they are gitignored.
5. Push and open a pull request.

## Ground rules for pipeline changes

- **Behaviour-changing stage logic goes behind a config flag** where practical, so an
  A/B against the frozen baseline is possible (the pattern used throughout `configs/`).
- **Prove it, don't assume it.** Re-run the affected stage(s) and compare the committed
  proxy metrics (cross-camera agreement, distinct-ID count, teleports, ground-reprojection)
  against the baseline snapshot before claiming a win. See
  [docs/metrics.md](docs/metrics.md) and [docs/improving-models.md](docs/improving-models.md).
- **The same-camera identity invariant is hard**: two detections in the same camera-frame
  must never share a global ID. Don't weaken it.
