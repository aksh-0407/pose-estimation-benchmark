# RTMPose-x runbook — install, run, and tune on a new machine

RTMPose-x (`rtmpose_x_body8`) is the largest RTMPose body model and the accuracy-first
Phase-1 2D-pose model. This is the end-to-end guide to stand it up on a fresh machine
(e.g. a remote GPU server) and run the full detection over every delivery at peak
throughput.

See also: [models.md](models.md#rtmpose-x-rtmpose_x_body8) for the model-identity note
and the Halpe-26 vs COCO-17 explanation.

---

## 0. What you get

- **Model:** RTMPose-x, 384×288, top-down (paired with an RTMDet-m person detector).
- **Output skeleton:** Halpe-26 (26 keypoints). Its first 17 are exactly COCO-17 in
  COCO order; 18–26 add head/neck/hip + 6 foot keypoints (heels, big/small toes).
- **Per-frame record:** every player carries **both**
  - `pose_2d` — COCO-17 (17 kpts), the contract the rest of the pipeline consumes, and
  - `pose_2d_native` — the full Halpe-26 (26 kpts) for future phases (feet etc.).
  `pose_2d.keypoints_px == pose_2d_native.keypoints_px[0:17]`.
- Runs in the shared `cricket-rtmpose-l` Conda env (mmpose 1.3.2 / mmcv 2.1.0 /
  mmdet 3.2.0 / torch 2.1.0-cu121) — **no separate env**.

---

## 1. Prerequisites

- NVIDIA GPU + driver (CUDA 12.x runtime bundled with the torch wheel; system `nvcc`
  not required), `conda`, `git`.
- The mmpose source tree vendored at `external/mmpose` (configs resolve `_base_`
  relative to it). If missing:
  ```bash
  git clone -b v1.3.2 https://github.com/open-mmlab/mmpose.git external/mmpose
  ```
- The dataset laid out as `<drive-root>/dataset/bt_01|bt_02|bt_03/<delivery>/camera<NN>/frame_*.jpg`
  (2560×1440). Default `--drive-root drive`.

## 2. Create the environment (skip if `cricket-rtmpose-l` already exists)

```bash
python3 scripts/setup/setup_model_envs.py --models rtmpose_x_body8
```

This creates/reuses the `cricket-rtmpose-l` env from the `mmpose_v1` profile in
[`configs/model_envs.yaml`](../configs/model_envs.yaml) (torch is auto-selected for the
detected driver). It is idempotent — an existing env is left in place.

## 3. Download the weights + detector

```bash
python3 scripts/setup/setup_model_envs.py --models rtmpose_x_body8 --skip-envs --download-assets
python3 scripts/setup/sync_model_store.py    # regenerate model.yaml / README / checksums
```

Downloads (~200 MB pose + shared RTMDet-m detector) into `models/rtmpose_x_body8/weights/`
and `models/rtmdet_m_person/weights/`. Weights and checksums stay local (git-ignored);
`model.yaml`/`README.md` are tracked. Verify:

```bash
python3 scripts/setup/check_assets.py --models rtmpose_x_body8 --fail-missing
```

## 4. Smoke test (one image)

```bash
python3 scripts/benchmark/benchmark.py smoke --models rtmpose_x_body8
```

Expect `status: ok`, `instances: 1`, `keypoints: 26`.

## 5. Tune batch/io/prefetch for THIS machine (a few minutes, writes no predictions)

Throughput depends on the GPU, CPU core count, and disk speed, so tune on the actual
box. The sweep runs `--benchmark-only` (no output written) and ranks by median FPS:

```bash
conda run -n cricket-rtmpose-l python scripts/tuning/tune_rtmpose_batches.py \
  --model-id rtmpose_x_body8 \
  --det-batches 16 24 32 --pose-batches 96 160 256 \
  --io-workers-list 8 16 24 --prefetch-list 2 4 \
  --repeats 2
```

It prints the top settings and writes `results/rtmpose_batch_tuning.csv` +
`.summary.json`. Note the winning `det_batch / pose_batch / io_workers / prefetch`.

> **Why sweep io/prefetch too?** Top-down RTMDet+RTMPose are batch-invariant in eval
> (batch size changes speed only, never output). On this dataset the frames are large
> 2560×1440 JPEGs read cold once (~50 GB), so on many machines the **disk-read + decode**
> stage, not the GPU, is the limiter. `--io-workers` (parallel readers) and
> `--prefetch-batches` (read-ahead depth that overlaps decode with GPU compute) are
> therefore first-class tuning knobs — see the performance notes below.

## 6. Run the full detection over all deliveries

Plug the tuned numbers into the wrapper (resumable; safe to re-run):

```bash
DET_BATCH_SIZE=32 POSE_BATCH_SIZE=96 IO_WORKERS=16 PREFETCH_BATCHES=3 \
  bash scripts/inference/run_rtmpose_x_final.sh
```

or call the runner directly:

```bash
conda run -n cricket-rtmpose-l python scripts/inference/run_phase1_rtmpose_inference.py \
  --model-id rtmpose_x_body8 \
  --det-batch-size 32 --pose-batch-size 96 \
  --io-workers 16 --cv2-threads 2 --prefetch-batches 3 \
  --run-id rtmpose-x --run-dir benchmarks/runs/rtmpose-x
```

Output → `benchmarks/runs/rtmpose-x/predictions/bt_XX__<delivery>__cam_YY.jsonl`
(one per camera; 56 cameras = 8 deliveries × 7 cameras spread across bt_01/02/03).
`run_manifest.json` records config, timings, and FPS on completion.

Useful flags: `--groups/--deliveries/--cameras/--frame-limit` (filter), `--list`
(preview scope), `--no-resume` (recompute), `--overlay` (render sample overlays).

## 7. Verify completeness

```bash
# each finished camera file has exactly 600 lines
for f in benchmarks/runs/rtmpose-x/predictions/*.jsonl; do echo "$(wc -l < "$f") $f"; done
# expect 56 files
ls benchmarks/runs/rtmpose-x/predictions/*.jsonl | wc -l
```

---

## Performance notes (how the runner is optimised)

The runner is built to keep the GPU fed while keeping the CPU light:

- **Overlapped I/O ↔ GPU pipeline (`--prefetch-batches`, default 3).** A single
  persistent thread-pool reads + decodes the next N detector batches *while the GPU runs
  detection + pose on the current batch*, instead of the old read → detect → pose stall
  where the GPU idled during every cold-disk read. On a laptop 4060 this took cold-data
  throughput from ~3.4 → ~8.6 FPS.
- **No thread oversubscription (`--cv2-threads`, default 2).** OpenCV otherwise spawns a
  thread per core inside *every* io-worker (e.g. 16 workers × 32 cores → thrash that
  pegs the CPU and starves the GPU). Capping OpenCV/torch CPU threads keeps loadavg low.
  The wrapper also exports `OMP/MKL/OPENBLAS_NUM_THREADS=2`.
- **One decode pool for the whole run**, not one per batch.
- **GPU JPEG decode is deliberately NOT used** — on these images nvjpeg (~19 ms/frame,
  incl. copy-back) is *slower* than CPU `cv2.imread` (~14 ms), and it competes with the
  pose model for the GPU. CPU decode overlapped with GPU compute wins.

**Rules of thumb for a beefier server:** more CPU cores → raise `--io-workers`
(16–32) and `--prefetch-batches` (4–8) to hide cold-disk latency; a bigger GPU → raise
`--pose-batch-size` (256–512) since ~35 person-crops/frame batch well. Always confirm
with the tuner. Keep `--cv2-threads` at 2–4 regardless.

**Numerics are batch-invariant:** changing any of det/pose batch, io-workers, or
prefetch changes speed only, never the predicted keypoints — so tuning is free of
accuracy risk and runs stay comparable.
