# Collaboration and git policy

This is the authoritative reference for what goes to GitHub, what stays on your machine,
and how a team benchmarks without stepping on each other. If another doc disagrees with
this one, this one wins.

## Policy

Commit source and compact evidence; keep assets and raw artifacts local; let CI build
the comparison view.

## What goes to GitHub vs. stays local

| Commit | Keep local |
| ------ | ---------- |
| `pose_estimation/`, `scripts/`, `tests/` | `models/<id>/weights/` (checkpoints) |
| `configs/*.yaml` | `models/<id>/checksums/` (generated hashes) |
| `docs/` | `data/raw/`, `data/derived/` (datasets) |
| `models/<id>/model.yaml`, `README.md` | `external/` (upstream clones) |
| `benchmarks/runs/<run_id>/` (compact evidence) | `benchmarks/artifacts/<run_id>/` (raw predictions, logs) |
| `data/splits/` (IDs only, no media) | `results/*.csv`, `benchmarks/reports/*.html` (derived; CI builds these) |
| `.gitkeep` placeholders, `.github/workflows/` | `Papers/`, caches, `wheels/`, `.model_env_stamps/` |

Two distinctions worth stating explicitly:

- `benchmarks/runs/` is committed; `benchmarks/artifacts/` is not. The run folder is the
  small, reviewable evidence (manifests and metrics). The artifacts folder is the bulky
  raw prediction and log dump: useful locally for debugging and resume, too large and
  unnecessary for comparing machines in git.
- `results/` and `benchmarks/reports/` are not committed. They are derived entirely from
  the committed run folders. See "Why CI owns the aggregates" below.

This is machine-checked by the pre-commit checklist.

## Run ids: why two people don't conflict

`benchmark.py run` names each run folder with a unique id:

```
models-<model_ids>__bench-<dataset_ids>__scope-<scope>__<YYYYMMDDTHHMMSSZ>
```

Examples:

```
models-yolo26x_pose__bench-coco17_val2017__scope-full__20260610T061500Z
models-yolo26x_pose__bench-coco17_val2017__scope-n100__20260610T061500Z
models-rtmw_x+yolo26x_pose__bench-coco17_val2017__scope-start200-n100__20260610T061500Z
```

Because the name encodes models, dataset, scope, and a UTC timestamp, two contributors
produce different folders and their commits merge cleanly. Use the generated id for
normal work; pass `--run-id` only for a fixed publication or handoff name.

Adding a benchmark adapter is conflict-free for the same reason: the runner is declared
per model with a `benchmark_runner` field in `configs/model_envs.yaml`
([adding-a-model.md](adding-a-model.md)), so two people adding adapters for different
models edit different config blocks rather than a shared table in code.

## The team workflow

1. **Work on a branch**, not directly on `main`.
2. **Run your benchmark.** Commit only the new `benchmarks/runs/<run_id>/` folder(s).
3. **Don't touch anyone else's run folder.** Runs are immutable — to change a result,
   produce a new run.
4. **Don't commit raw artifacts or derived aggregates** (`benchmarks/artifacts/`,
   `results/*.csv`, `benchmarks/reports/*.html`). `.gitignore` already excludes them.
5. **Open a PR and merge to `main`.** On merge, CI regenerates the aggregate CSV +
   HTML report from all committed runs and publishes them (below).
6. **Include hardware/software context when comparing.** Every run folder carries its
   own `hardware.json` / `software.json`; a laptop number and an A100 number are only
   comparable with that context.

Two people benchmarking **different** models never collide. Two people benchmarking the
**same** model just get different timestamped ids — also fine.

## Pre-commit checklist

Run these before you commit. They take seconds and prevent the two classic mistakes
(committing a weight file, or a generated CSV):

```bash
python3 scripts/audit_repo.py --fail   # no tracked weights/datasets/artifacts/derived files
python3 -m pytest -q                    # code still green
```

`audit_repo.py` is the guard rail: it fails if anything under `models/*/weights`,
`models/*/checksums`, `data/raw`, `data/derived`, `benchmarks/artifacts`,
`results/*.csv`, or `benchmarks/reports/*` has been tracked by mistake (placeholders
and `README.md` files are allowed).

## Why CI owns the aggregates

`results/aggregate_metrics.csv` is **one file with one row per run**, and
`benchmarks/reports/index.html` is just that CSV rendered. They're derived entirely
from the run folders. If everyone regenerated and committed them, every parallel merge
would conflict on those two files for no benefit.

So instead: **contributors commit only run folders, and CI regenerates the aggregates
on `main`.** The shared comparison view is always current, and no human ever
hand-merges a derived file. To preview your own numbers locally before pushing, just
run `aggregate` + `report` yourself (the outputs stay uncommitted).

### The CI workflow

[`.github/workflows/aggregate-report.yml`](../.github/workflows/aggregate-report.yml)
runs on every push to `main`:

1. Checks out the repo and installs the small dependency set (`requirements.txt` —
   `aggregate`/`report` only read JSON and write CSV/HTML; no weights, datasets, or GPU
   needed).
2. Runs `benchmark.py aggregate` then `benchmark.py report`.
3. Publishes `results/` + `benchmarks/reports/` to **GitHub Pages**. The derived files
   never land in the tracked tree, so there's nothing to merge and nothing to conflict.

> **Note:** this repo has no Git remote yet. Push it to GitHub and enable Pages
> (Settings → Pages → "GitHub Actions") for the workflow to take effect. Until then,
> preview locally with `aggregate` + `report`.

## First-time GitHub setup (maintainer)

When you create the GitHub repository and push:

- The history has already been cleaned of large weight blobs, so the clone stays small.
- Enable **Settings → Pages → Build and deployment → Source: GitHub Actions** so the
  aggregate report publishes.
- Protect `main` (require PRs) so runs land via review and CI runs on each merge.
