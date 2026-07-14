# data/

All pipeline data lives here under a single `DATA_ROOT` — this `data/` directory locally,
or `~/bits-pose-data/` on the L40S box (select with `--data-root` / `$PIPETRACK_DATA`).
The tree is identical on every machine; only the base differs.

```text
data/
  raw/<dataset>/                     # footage + calibration + events            (LOCAL, ignored)
    8_init/                          #   the originally-shared 8-delivery set
      bt_01/ bt_02/ bt_03/           #   capture groups -> <delivery>/cameraNN/*.jpg
      calibration-data/  events-data/
    40_full/                         #   the 40-delivery campaign (L40S only; borrows 8_init calibration)
  derived/<dataset>/pipetrack_v<n>/  # P1 (p1/) + stage outputs (deliveries/<D>/0N_*)  (LOCAL, ignored)
  viz/<dataset>/pipetrack_v<n>/      # mosaics: <delivery>/<delivery>__all_cameras.mp4 (LOCAL, ignored)
```

Datasets are declared in [`../configs/datasets.yaml`](../configs/datasets.yaml) — each has a
`calibration_source` (so `40_full` reuses `8_init`'s single calibrated session). Keep `raw/`
immutable; everything under `raw/`, `derived/`, `viz/` is machine-local and gitignored — only
this `README.md` is committed.

Run against a dataset with `--dataset <name> --version <n>` (see
[`../docs/getting-started.md`](../docs/getting-started.md)).
