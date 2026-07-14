# Troubleshooting

Common failures and what to do. When in doubt, start with
`python3 tools/check_environment.py` to see what's actually installed.

## Model download

### `download.openmmlab.com` times out

The OpenMMLab host is unreliable on some networks — it serves the RTMPose-X pose weights
and the RTMDet person-detector checkpoint. Options:

- Retry — it's often transient.
- Download the file from another network or a verified mirror, then drop it at the exact
  `path` listed for that asset in [`configs/model_envs.yaml`](reference/configuration.md#shared).
- Re-check with `python3 tools/check_assets.py --models rtmpose_x_body8 --fail-missing`.

### mmpose config `_base_` not found

P1 resolves mmpose/mmdet configs relative to the vendored source tree at `external/mmpose`.
If it's missing:
`git clone -b v1.3.2 https://github.com/open-mmlab/mmpose.git external/mmpose`.

## Source control

### Checkpoints / weights show as untracked

**Expected and correct.** `models/*/weights/` and `models/*/checksums/` are gitignored so
multi-GB binaries never get committed; only `model.yaml` / `README.md` are tracked. Same for
frames under `drive/` and raw per-frame prediction dumps.

### `audit_repo.py` reports tracked artifacts

It found a weight, frame, or raw artifact that was committed by mistake. Un-track it with
`git rm --cached <path>` (the file stays on disk), then commit; `.gitignore` keeps it out
going forward.

## Pipeline runs

### Identity-stage (01–06) import errors (NumPy/SciPy)

The tracking → global-ID → triangulation stages need NumPy ≥ 1.23.5 and SciPy ≥ 1.10. Run
them in an env that has them (e.g. `pose-lab`), not the mmpose env.

### The mosaic render is missing tiles / roles / ground monitor

The renderer reads several artifacts from the 05_global_id run: `predictions/*.jsonl`,
`diagnostics/correspondences.jsonl` (03 association badges), `diagnostics/ground_tracks.jsonl` (the
bird's-eye monitor), and `../06_roles/roles.json` (the roster). A missing panel usually means the
corresponding stage didn't write its diagnostic — re-run that stage. Camera 07 has a
different native resolution (~3775×960); if a tile looks wrong, check the per-camera image
size handling.

### FFmpeg / NVENC

The renderer encodes via `h264_nvenc` with an `mp4v` fallback. If encoding fails, confirm
`ffmpeg` is on `PATH` (`check_environment.py`) and that the NVENC-capable driver is present;
the fallback path works without a GPU encoder, just slower.

## Hardware / speed sanity

Speed depends on GPU, CPU cores, and disk. P1 top-down inference scales with the number of
players per frame; on large 2560×1440 JPEGs read cold, disk-read + decode is often the
limiter, not the GPU — tune `--io-workers` / `--prefetch-batches` (see
[rtmpose-x-runbook.md](rtmpose-x-runbook.md)). Batch sizes change speed only, never the
predicted keypoints.
