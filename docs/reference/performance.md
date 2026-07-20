# Pipeline optimization findings (L40S box)

Goal: run every stage as fast as possible on the L40S box and maximise hardware use, without changing
results. Standing rule: a speed change must not change the pipeline's decisions or numbers. Accuracy and
results always take precedence over speed.

The box is 8 CPU cores, 61 GB RAM, one NVIDIA L40S (46 GB). Every stage is compute-bound (CPU cores or
GPU SMs), never memory-bound, so "use more RAM" buys nothing; the real levers are reducing work per
delivery (algorithmic) and filling idle GPU or CPU with correctly-capped parallelism.

---

## Part 1: 2026-07-17 script-optimization pass (this session)

Six fixes to the run scripts, applied before any A/B, verified by compile and by a dry-run of the
parallel launcher. These were found by a full audit of the P1, render, and tracking run-scripts against
the 8-core plus L40S hardware.

| # | File | Change | Why | Output effect |
|---|---|---|---|---|
| 1 | `src/core/inference/run_phase1_parallel.py` | Fix the runner path (was `scripts/inference/…`, which does not exist; now derived from `__file__` as the sibling `run_phase1_l40s.py`) | The data-parallel P1 launcher, the only real lever to fill the idle L40S (about 2x throughput at 3 shards), failed on every shard | Enables a previously-broken path; the per-shard P1 output is unchanged |
| 2 | `src/core/inference/run_phase1_parallel.py` | Auto-clamp per-shard `--io-workers` to `cores // shards` and `--cv2-threads` to 1 when not set explicitly | Default 3 shards times 12 io-workers times 2 cv2 threads is 36 to 72 decode threads on 8 cores | Thread count only, output unchanged |
| 3 | `src/identity/visualization/render_videos.py` | `cv2.setNumThreads(1)` and disable OpenCL at the top of `main()` | The renderer runs one process per delivery under render-jobs; the BLAS cap does not cap OpenCV, so each render grabbed all cores | Thread count only; render pixels unchanged |
| 4 | `src/main.py` | Default `QT_RENDER_GPU_DECODE=0` in the render subprocess env (CPU JPEG decode) | Parallel renders otherwise each hammer the single GPU with a per-frame device sync | Switches the JPEG decode path (GPU to CPU); may produce marginally different render pixels, render only, not a metric stage; NVENC still encodes |
| 5 | `src/core/inference/run_rtmpose_x_l40s.sh` and `run_phase1_l40s.py` | Single-process defaults io-workers 8, cv2-threads 1 (was 16 or 12, and 2) | 8 cores; io-workers times cv2-threads should stay near the core count | Thread count only, output unchanged |
| 6 | `src/identity/p2_tracking/runner.py` | `os.environ.setdefault` the BLAS caps before the per-camera ProcessPool | Under `id_pipeline` these are already 1 (inherited); this makes a standalone stage-02 CLI run safe too | Standalone-only; the production path already capped |

Headline: fix 1 restores the roughly 2x GPU-throughput lever for full P1 re-runs (the parallel launcher).
Fixes 2, 3, 5, 6 are thread-count only and do not change any output. Fix 4 changes the render JPEG decode
path (GPU to CPU) and may produce marginally different render pixels; it is the render stage only and
touches no metric.

The already-good parts, confirmed and left untouched: the P1 runner already batches the detector and pose
forwards, uses AMP fp16 plus TF32 plus cudnn.benchmark, and overlaps JPEG decode with GPU compute through
a prefetch pool; the identity stages already cap BLAS threads to 1 per subprocess in
`id_pipeline._run_stage`; stage 02 already uses a process pool capped to `min(7, cpu_count)`; the render
already uses NVENC for encode.

Dry-run validation of fix 1: the launcher discovers all 40 deliveries, plans 3 balanced shards (14, 13,
13), and resolves the runner path. Confirmed on the box.

---

## Part 2: 2026-07-14 algorithmic findings (still in the code, byte-identical, carried forward)

These are the accuracy-neutral algorithmic speedups shipped earlier. They remain in the current code
(`pose_medoid_incremental` in `p2_tracking/config.py`, the vectorised triangulation kernels) and are all
proven byte-identical by execution.

### P2 incremental medoid cache (the large one)

`src/identity/p2_tracking/track.py` plus `config.py`. Sample-profiling showed about 86.9% of P2 self-time
in `masked_weighted_cosine`, driven by `Track.gallery_repr()` recomputing the O(K squared) gallery medoid
(K equals `pose_gallery_size`, 30) on every hit, even though each hit appends only one member.

Fix: cache the pairwise cosine matrix keyed by a monotonic per-member sequence id kept in lockstep with
the ring buffer, and recompute only the new member's row, O(K) per update instead of O(K squared).
Bit-identical by construction (the cosine is symmetric, values are memoised, row sums add the same
per-pair values in the same order with the same first-minimum tie-break). Flag `pose_medoid_incremental`
(default on) lets the legacy path be A/B'd.

- Per camera: 96.3 s to 6.0 s (16x). Full delivery (7 cameras): 154 s to 10 s local, 154 s to 12 s on
  the box.
- Byte-identical: 7 of 7 cameras vs baseline and vs box production output.

### P3 ground-anchored skeleton vectorised (bit-identical)

`src/identity/common/pose_shape.py`. Replaced a per-joint Python loop with batched numpy. Proven
byte-identical on 20,000 random cases (including NaN and edge cases) and on full-P3 correspondences.

### Combined single-delivery chain (warm cache): 348 s to 111 s, full chain byte-identical

| stage | baseline (s) | optimized (s) | note |
|---|---|---|---|
| 01 stabilization | 13.7 | ~13 | stateful One-Euro loop |
| 02 tracking | 154.6 | 10.0 | medoid cache, real algorithmic win |
| 03 association | 134.1 | 49.4 | ground-anchored plus warm page-cache on appearance decode |
| 03.5 lift | 24.3 | 20.2 | already batched |
| 05 global id | 6.3 | 6.3 | irreducible JSON IO |
| 06 3D | 12.2 | 10.8 | already batched |

The full optimized chain re-run from P1 on the box for 3 deliveries reproduced the shipped production
outputs byte-for-byte across all stages.

### Analysed, no byte-identical speedup available (documented, not forced)

- P3 appearance cue (about 60% of P3): d-prime about 0.09; disabling it is the only large P3 lever but it
  changes output (cycle-consistency 0.701 to 0.687 on one delivery). Dropped, accuracy first.
- Stage 05 JSON serialization (61% of its 6 s): swapping serializers changes float formatting, not
  byte-identical. Irreducible.
- One-Euro filter: a stateful K by T scalar loop with per-frame conditional skips; vectorising risks
  floating-point drift and it is about 4% of the chain. Not shipped.

---

## Throughput ceiling

The CPU batch is core-bound at 8 cores, already reached at `--jobs 8` with BLAS capped to 1 per
subprocess. Past that, the only levers are more algorithmic work reduction or more vCPUs (deliveries are
independent, so the batch scales roughly linearly with cores). The GPU lever is the now-fixed parallel P1
launcher (Part 1, fix 1).
