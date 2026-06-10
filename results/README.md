# results/

Small, **derived** comparison tables generated from the committed run folders under
`benchmarks/runs/`:

```text
aggregate_metrics.csv    # one row per run (benchmark.py aggregate)
model_ranking.csv        # weighted ranking (score_models.py)
```

These are **not committed.** They're a pure function of the run folders, so committing
them would only create merge conflicts. You can regenerate them anytime to preview your
own numbers:

```bash
python3 scripts/benchmark.py aggregate
python3 scripts/benchmark.py report
```

On `main`, CI regenerates and publishes them. See
[docs/collaboration.md](../docs/collaboration.md#why-ci-owns-the-aggregates).

Only this `README.md` and the `.gitkeep` placeholder are tracked here.
