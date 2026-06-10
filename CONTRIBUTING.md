# Contributing

This repository is a shared benchmarking workspace. Results land on `main` through pull
requests so they can be reviewed and so the published comparison report stays current.

## Prerequisites

A Linux machine with an NVIDIA GPU and conda. Cloning gives you code, configs, and docs;
model weights and datasets are not committed and are downloaded locally during setup.
See [docs/getting-started.md](docs/getting-started.md) for the full walkthrough.

## One-time setup

```bash
git clone https://github.com/aksh-0407/pose-estimation-benchmark
cd pose-estimation-benchmark
pip install -r requirements.txt
python3 scripts/benchmark.py prepare --models all --datasets coco17_val2017
python3 scripts/check_assets.py --models all --fail-missing
python3 scripts/benchmark.py smoke --models all
```

The `prepare` step builds a conda environment per model and downloads weights plus COCO.
It takes a while and some mirrors can be flaky; see
[docs/troubleshooting.md](docs/troubleshooting.md) if a download fails.

## Workflow

1. Create a branch: `git checkout -b <your-name>/<task>`.
2. Do the work:
   - Run a benchmark: `python3 scripts/benchmark.py run --models <id> --datasets coco17_val2017`.
   - Or add a model adapter: see [docs/adding-a-model.md](docs/adding-a-model.md).
3. Before committing, run the checks:
   ```bash
   python3 scripts/audit_repo.py --fail
   python3 -m pytest -q
   ```
4. Commit only source and compact evidence: the new `benchmarks/runs/<run_id>/` folder and
   any config changes. Do not commit weights, datasets, raw artifacts, or generated
   `results/` and `benchmarks/reports/` files. They are gitignored, and CI regenerates the
   report.
5. Push your branch and open a pull request. A maintainer merges it into `main`, and CI
   republishes the report.

## What goes in git

The full policy is in [docs/collaboration.md](docs/collaboration.md). Short version:
commit code, configs, docs, and `benchmarks/runs/<run_id>/`; keep weights, datasets,
upstream clones, and raw artifacts local.
