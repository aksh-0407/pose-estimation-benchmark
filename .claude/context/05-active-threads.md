# Active threads (2026-07-14 evening)

## 1. NEXT UP (user's stated plan for a new session): VRAM-accelerated mosaic rendering
Goal: use the L40S's 46 GB VRAM to render all mosaics fast (8-batch, then all 40).
Current renderer is CPU-bound (~10 min/delivery: cv2 decode + numpy compositing + x264).
Leads, in expected-yield order:
- **NVENC encoding**: renderer already probes `_ffmpeg_has_nvenc()`
  (`render_phase1_videos.py:94`) and prefers `h264_nvenc` — but the static ffmpeg at
  `~/bin/ffmpeg` (johnvansickle) has NO nvenc. Install an nvenc-capable ffmpeg (conda
  `ffmpeg` with nvenc, or a BtbN/ffmpeg-builds GPL binary) → encoding offloaded to GPU.
- **GPU decode**: frame JPEG decode via nvJPEG (torchvision.io.decode_jpeg(device='cuda')
  in the cricket-rtmpose-l env) instead of cv2.imread — decode is a large slice of render
  time (7 cams × 600 × 2560×1440). Caveat: nvJPEG lost to cv2 on the laptop 4060 (~19 vs
  14 ms, GPU shared with pose) — but during renders the L40S GPU is IDLE, so measure fresh
  there rather than trusting that verdict.
- **Compositing on GPU**: torch tensor ops for resize/blend instead of cv2/numpy (biggest
  rewrite; only if the first two don't reach the target).
- Parallelism: renders are per-delivery independent — with GPU decode+encode, 4+ parallel
  renders become feasible (8 vCPUs stop being the ceiling).
Mosaic spec: upgraded renderer (collision-free chips, body paint, roles in roster only);
8 benchmark deliveries pulled to laptop for review first, then all 40 stay on the box.

## 2. Manager's reprojection questions (analysis DONE, answer in chat 2026-07-14)
Numbers measured on v8.1 (`_14_4`+`_14_7`, ~331k joint-view residuals):
- Panel metric (RANSAC inlier views, raw pre-smoothing triangulation): 3.07–3.56 px mean.
- Post-smoothing, all confident 2D views incl. fills: mean 6.8 px, MEDIAN 3.7 px,
  p95 24.5 px.
- WHERE high errors: hips 11–12 px mean (systematic cross-view keypoint-definition
  inconsistency, worst joint by far); fast limbs' tails (r_elbow p95 34 px); per-camera
  spread mild 6.0–8.3 px (no bad camera → calibration healthy).
- The "1 px" expectation is the CALIBRATION-TARGET standard (sub-pixel corners); ours is
  measured against POSE-MODEL keypoints whose own noise is 2–3 px (P1.5 jitter metric) plus
  cross-view definition offsets (hips!). Rig calibration itself was validated at ball-reproj
  p95 ≤ 4.5 px (wip/methods_log.md). 1 px mean vs detected keypoints is not achievable with
  any current 2D pose model at this resolution — the floor is the 2D noise, not calibration.
- Possible follow-ups if the manager wants movement: report median (3.7 px) + inlier metric
  consistently; exclude fills from the reported number; hip-specific handling (e.g. exclude
  hips from reproj reporting or add a hip-offset model); G1/G3 flags are implemented and
  un-A/B'd (remaining-work §2.1) — worth measuring, expected small.

## 3. Open user decisions
- Mosaic batch timing (blocked on the VRAM thread above, or run CPU renders now).
- UE packet export need (`export_ue_packets.py` never run on v8 data).
- Two production residual coloc pairs (M1_1_14_7, M2_1_11_3): relax colocated gates +
  re-run P4 for those two, or leave for mosaic arbitration (remaining-work §5b).
- Vedant global_id changelog still awaited; GT labelling still open.
