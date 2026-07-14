# RTMPose-x runbook — install, run, and tune on a new machine

RTMPose-x (`rtmpose_x_body8`) is the largest RTMPose body model and the accuracy-first
Phase-1 2D-pose model. This is the end-to-end guide to stand it up on a fresh machine
(e.g. a remote GPU server) and run the full detection over every delivery at peak
throughput.

RTMPose-X emits **Halpe-26** (26 keypoints): the first 17 are exactly COCO-17 in COCO
order; 18–26 add head/neck/hip + 6 foot keypoints (heels, big/small toes).

---

## 0. What you get

- **Model:** RTMPose-x, 384×288, top-down (paired with an RTMDet-m person detector).
- **Output skeleton:** Halpe-26 (26 keypoints). Its first 17 are exactly COCO-17 in
  COCO order; 18–26 add head/neck/hip + 6 foot keypoints (heels, big/small toes).
- **Per-frame record:** every player carries **both**
  - `pose_2d` — COCO-17 (17 kpts), the contract the rest of the pipeline consumes, and
  - `pose_2d_native` — the full Halpe-26 (26 kpts) for future phases (feet etc.).
  `pose_2d.keypoints_px == pose_2d_native.keypoints_px[0:17]`.
- Runs in the shared `pose-lab` Conda env (mmpose 1.3.2 / mmcv 2.1.0 /
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

## 2. Create the environment (skip if `pose-lab` already exists)

```bash
python3 tools/setup_model_envs.py --models rtmpose_x_body8
```

This creates/reuses the `pose-lab` env from the `mmpose_v1` profile in
[`configs/model_envs.yaml`](../configs/model_envs.yaml) (torch is auto-selected for the
detected driver). It is idempotent — an existing env is left in place.

## 3. Download the weights + detector

```bash
python3 tools/setup_model_envs.py --models rtmpose_x_body8 --skip-envs --download-assets
python3 tools/sync_model_store.py    # regenerate model.yaml / README / checksums
```

Downloads (~200 MB pose + shared RTMDet-m detector) into `models/rtmpose_x_body8/weights/`
and `models/rtmdet_m_person/weights/`. Weights and checksums stay local (git-ignored);
`model.yaml`/`README.md` are tracked. Verify:

```bash
python3 tools/check_assets.py --models rtmpose_x_body8 --fail-missing
```

## 4. Smoke test (one image)

```bash
python3 tools/run_model_smoke.py --model rtmpose_x_body8
```

Expect `status: ok`, `instances: 1`, `keypoints: 26`.

## 5. Tune batch/io/prefetch for THIS machine

Throughput depends on the GPU, CPU core count, and disk speed, so tune on the actual box.
The P1 runner itself exposes the knobs directly — do a short scoped run
(`--deliveries <one> --frame-limit 200 --no-resume`) at a few `--det-batch-size` /
`--pose-batch-size` / `--io-workers` / `--prefetch-batches` settings and compare the FPS
recorded in `run_manifest.json`. Note the winning combination for the full run below.

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
  bash src/core/inference/run_rtmpose_x_final.sh
```

or call the runner directly:

```bash
conda run -n pose-lab python src/core/inference/run_phase1_rtmpose_inference.py \
  --model-id rtmpose_x_body8 \
  --det-batch-size 32 --pose-batch-size 96 \
  --io-workers 16 --cv2-threads 2 --prefetch-batches 3 \
  --run-id rtmpose-x --run-dir data/derived/runs/rtmpose-x
```

Output → `data/derived/runs/rtmpose-x/predictions/bt_XX__<delivery>__cam_YY.jsonl`
(one per camera; 56 cameras = 8 deliveries × 7 cameras spread across bt_01/02/03).
`run_manifest.json` records config, timings, and FPS on completion.

Useful flags: `--groups/--deliveries/--cameras/--frame-limit` (filter), `--list`
(preview scope), `--no-resume` (recompute), `--overlay` (render sample overlays).

## 7. Verify completeness

```bash
# each finished camera file has exactly 600 lines
for f in data/derived/runs/rtmpose-x/predictions/*.jsonl; do echo "$(wc -l < "$f") $f"; done
# expect 56 files
ls data/derived/runs/rtmpose-x/predictions/*.jsonl | wc -l
```

---

## Running on the L40S / remote capture machine

The remote capture box stores frames in a different native layout —
`/home/ubuntu/pose_data/{bt1,bt2,bt3}/<delivery>/camera<NN>/frame_*.jpg` — and writes to
a caller-chosen output dir. Use the dedicated runner
[`run_phase1_l40s.py`](../src/core/inference/run_phase1_l40s.py) (it reuses the exact same
mmdet/mmpose building blocks and P1 schema, incl. the 26-keypoint `pose_2d_native`, and
has the same prefetch + thread-cap optimisation). RTMPose-x is fully wired: just pass
`--model-id rtmpose_x_body8`.

**One-time setup on the remote machine** (same as §1–4 above): clone the repo, create the
`pose-lab` env, download the RTMPose-x weights + RTMDet detector,
`sync_model_store.py`, then smoke. Confirm the GPU: `run_phase1_l40s.py --list` prints the
selection with no GPU needed; add `--device cuda:0` runs to check CUDA.

**1. Tune batch sizes for the L40S** (in-process, single model load, writes only `best.json`):

```bash
conda activate pose-lab
python src/core/inference/run_phase1_l40s.py \
  --model-id rtmpose_x_body8 --output-dir /home/ubuntu/pose-rtm-x \
  --sweep --grid          # --grid = real end-to-end det x pose ranking
```

`best.json` lands in the output dir with the winning `det_batch / pose_batch / io_workers`
and a projected full-run time. (The grid measures GPU + decode; with the prefetch overlap
the real run is a bit faster than the projection.)

**2. Full run over all data → `/home/ubuntu/pose-rtm-x/`** using the wrapper
[`run_rtmpose_x_l40s.sh`](../src/core/inference/run_rtmpose_x_l40s.sh):

```bash
DET_BATCH_SIZE=<B> POSE_BATCH_SIZE=<P> IO_WORKERS=<W> \
  bash src/core/inference/run_rtmpose_x_l40s.sh
```

or directly:

```bash
python src/core/inference/run_phase1_l40s.py \
  --pose-data /home/ubuntu/pose_data --output-dir /home/ubuntu/pose-rtm-x \
  --model-id rtmpose_x_body8 \
  --det-batch-size <B> --pose-batch-size <P> \
  --io-workers <W> --cv2-threads 2 --prefetch-batches 4
```

Output → `/home/ubuntu/pose-rtm-x/predictions/bt_XX__<delivery>__cam_YY.jsonl`, plus
`run_manifest.json` + `p1_metrics.json`. Runs are **resumable** — re-run the same command
to continue after any interruption. Run it under `tmux`/`nohup` so an SSH drop doesn't kill
it:

```bash
tmux new -s posex
DET_BATCH_SIZE=<B> POSE_BATCH_SIZE=<P> IO_WORKERS=<W> \
  bash src/core/inference/run_rtmpose_x_l40s.sh 2>&1 | tee /home/ubuntu/pose-rtm-x/run.log
# detach: Ctrl-b d ; reattach: tmux attach -t posex
```

Verify completeness: every finished camera file has 600 lines; `camera_count` in
`run_manifest.json` equals the number of cameras discovered by `--list`.

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
