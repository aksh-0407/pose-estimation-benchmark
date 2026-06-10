# Adding a model

This is the end-to-end recipe for benchmarking a model that isn't in the registry yet.
There are two levels of done:

- **Smoke-ready** (steps 1–6): the model installs, its weights download, and one image
  runs through it. Enough to register a candidate.
- **Benchmark-ready** (step 7): a full dataset adapter exists, so `run` produces real
  scores instead of an `adapter_pending` placeholder.

Read [concepts.md](concepts.md) and [configuration.md](configuration.md) first — this
guide assumes you know what the config files are.

## 1. Register the model's identity

Add an entry under `models:` in
[`configs/model_registry.yaml`](configuration.md#model_registryyaml). The `id` is the
stable key used everywhere (CLI, paths, env config).

```yaml
  - id: my_pose_model
    name: My Pose Model
    family: <upstream family>
    role: primary_candidate            # candidate / teacher / reference / baseline
    framework: <mmpose|ultralytics|...>
    skeleton: coco_17                  # the model's NATIVE skeleton
    input_size: [640, 640]
    checkpoint: models/my_pose_model/weights/<file>
    export_targets: [onnx]
    license: <check terms>
    status: candidate
    expected_strengths: [...]
    expected_risks: [...]
    sources: [<paper / repo URLs>]
```

## 2. Describe how to install and feed it

Add an entry under `models:` in
[`configs/model_envs.yaml`](configuration.md#model_envsyaml). Reuse an existing
`profile` if the model shares an ecosystem (e.g. `mmpose_v1`, `ultralytics_pose`);
otherwise add a new profile with its Conda/pip recipe.

```yaml
  my_pose_model:
    env_name: cricket-my-pose-model
    profile: ultralytics_pose          # reuse, or add a new profile above
    smoke_profile: ultralytics         # which smoke path in run_model_smoke.py drives it
    checkpoint: models/my_pose_model/weights/<file>
    smoke_image: external/mmpose/.../human-pose.jpeg
    assets:
      - kind: url                       # url | hf | hf_repo
        url: https://.../<file>
        path: models/my_pose_model/weights/<file>
        required_for_smoke: true
        # fallback_urls: [...]          # optional mirror if the primary host is flaky
        # large: true                   # only downloaded with --download-large-assets
```

## 3. Map its skeleton to COCO-17

If the model's native skeleton isn't already in
[`configs/keypoint_mappings.yaml`](configuration.md#keypoint_mappingsyaml), add a
`source_to_coco_17` entry giving the source index for each of the 17 COCO joints, in
COCO order. Models that already emit COCO-17 use the identity slice `[0..16]`. This is
what makes the model comparable to every other model.

## 4. Create the env and download weights

```bash
python3 scripts/setup_model_envs.py --models my_pose_model --download-assets
```

Add `--download-large-assets` if you marked any asset `large: true`.

## 5. Generate the model-store metadata

```bash
python3 scripts/sync_model_store.py
```

This writes `models/my_pose_model/{model.yaml, README.md, checksums/}` and the `.gitkeep`
placeholders. `model.yaml` and `README.md` are committed; `weights/` and `checksums/`
stay local.

## 6. Verify assets, then smoke

```bash
python3 scripts/check_assets.py --models my_pose_model --fail-missing
python3 scripts/benchmark.py smoke --models my_pose_model
```

If your framework isn't one the smoke runner already handles, add a `smoke_profile`
branch in [`scripts/run_model_smoke.py`](scripts.md#run_model_smokepy) (it dispatches on
the `smoke_profile` field: `mmpose`, `dwpose`, `ultralytics`, `mediapipe`, `vitpose`,
`sapiens2`, `openpose`). The branch loads the model in its env, runs one image, and
returns the standard JSON (status, instances, keypoints, latency).

**At this point the model is smoke-ready.** `run` will work but produce an
`adapter_pending` metrics file until you do step 7.

## 7. Add a full dataset adapter (benchmark-ready)

Which runner drives a model is declared in that model's own `configs/model_envs.yaml`
entry, via a `benchmark_runner` field. Two runner kinds exist today: `yolo` (end-to-end)
and `mmpose_topdown` (top-down, GT boxes).

If your model fits an existing runner, this is a one-line config change with no code:

```yaml
  my_pose_model:
    env_name: cricket-my-pose-model
    profile: mmpose_v1
    smoke_profile: mmpose
    benchmark_runner: mmpose_topdown   # <-- makes it benchmark-ready
    ...
```

Most MMPose top-down whole-body/body models work this way. Because the field lives in
the model's own config block, two people adding adapters for different models do not
touch shared code and will not conflict on merge.

If your model needs a new runner kind:

1. Write a runner modeled on
   [`run_yolo_coco_benchmark.py`](scripts.md#run_yolo_coco_benchmarkpy) (end-to-end) or
   [`run_mmpose_coco_benchmark.py`](scripts.md#run_mmpose_coco_benchmarkpy) (top-down).
   Reuse the shared helpers in `pose_estimation/coco_keypoint_eval.py` (image selection,
   resumability, COCO OKS evaluation) and reduce native keypoints to COCO-17 with
   `pose_estimation.keypoints.map_keypoints`. Record an `eval_protocol` in the metrics so
   results stay comparable (see
   [models.md](models.md#evaluation-protocols)).
2. Register the runner kind in `RUNNER_SCRIPTS` in
   [`scripts/benchmark.py`](scripts.md#benchmarkpy) and handle any runner-specific flags
   in `run_native_benchmark`.
3. Set `benchmark_runner: <your_kind>` on the model in `configs/model_envs.yaml`.
4. Emit `PredictionRecord`s (native + reduced `coco_17`); see
   `pose_estimation/predictions.py`.

## 8. Run it for real and share

```bash
python3 scripts/benchmark.py run --models my_pose_model --datasets coco17_val2017 --limit 100
python3 scripts/benchmark.py run --models my_pose_model --datasets coco17_val2017   # full
```

Then commit the `configs/` changes + the new `benchmarks/runs/<run_id>/` folder on a
branch and open a PR — the team flow is in [collaboration.md](collaboration.md). Run the
[pre-commit checklist](collaboration.md#pre-commit-checklist) first.
