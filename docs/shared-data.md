# Shared Data Guide — `pipetrack_v6.0`

This documents the **pipetrack_v6.0** run so it can be consumed as-is while the fix campaign
(`docs/critical-analysis/to-do.md`) refines the pipeline. It is the current **frozen ground
baseline**: the first full-chain run built from the complete 8-delivery RTMPose-X (Halpe-26)
2D pose data, with per-stage outputs, quality metrics, and rendered mosaic videos for every
delivery. Treat it as read-only (`_baseline_snapshot` is the diff reference for all ongoing
A/B experiments).

## What this data is

Seven synchronized cameras (`cam_01 … cam_07`) film each cricket delivery (~600 frames at
50 fps, 2560×1440 px; `cam_07` is a panoramic ~3775×960). The pipeline turns the raw frames
into, per delivery:

- per-player **2D skeletons** in every camera (COCO-17 + native Halpe-26 with feet),
- per-camera **tracklets** (temporally linked detections),
- **cross-camera associations** (which detection in camera A is the same person as in B),
- **global player identities** (`P001`, `P002`, …) that persist across cameras and time,
- player **roles** (bowler / striker / non-striker / wicketkeeper / umpire / fielder),
- **3D skeletons** in world metres (triangulated from all cameras; z = 0 is the ground,
  origin at the pitch centre) and fused **ground positions**,
- a diagnostic **mosaic video** with all cameras, skeleton overlays coloured by global ID,
  a bird's-eye field view, ball trail, and the roster.

## Locations

| What | Path |
|---|---|
| Run data (all stages) | `benchmarks/runs/pipetrack_v6.0/deliveries/<DELIVERY>/` |
| Mosaic videos | `artifacts/pipetrack_v6.0/mosaics/<DELIVERY>/<DELIVERY>__all_cameras.mp4` |
| Frozen metrics snapshot | `benchmarks/runs/pipetrack_v6.0/_baseline_snapshot/` |
| Run provenance (configs + sha256) | `benchmarks/runs/pipetrack_v6.0/pipeline_manifest.json` |
| Input 2D poses (P1) | `benchmarks/runs/rtmpose-x/predictions/` |
| Frames / calibration / ball events | `drive/dataset/…` (see `docs/getting-started.md`) |

The 8 deliveries: `CCPL080626M1_1_14_1 … _7` and `CCPL080626M2_1_12_1`.

## Pipeline phases and their outputs

Each delivery directory contains one folder per phase. Every phase reads the previous one's
`predictions/` and enriches the same per-camera JSONL records, so any phase can be consumed
independently.

```
benchmarks/runs/pipetrack_v6.0/deliveries/CCPL080626M1_1_14_1/
├── p2/          per-camera tracking
│   ├── predictions/<group>__<delivery>__cam_NN.jsonl
│   ├── diagnostics/
│   ├── tracking_metrics.json
│   └── run_manifest.json
├── p3/          cross-camera association
│   ├── predictions/…                      (adds association fields)
│   ├── diagnostics/correspondences.jsonl  (per-frame cross-camera clusters)
│   ├── diagnostics/tracklet_graph.json    (whole-delivery identity bindings)
│   └── association_metrics.json
├── p4/          global identity
│   ├── predictions/…                      (adds global_player_id)
│   ├── diagnostics/ground_tracks.jsonl    (per-frame world positions per ID)
│   ├── diagnostics/correspondences.jsonl  (copy, for the renderer)
│   ├── global_id_metrics.json
│   └── id_switch_report.json
├── p5/          roles
│   ├── roles.json
│   └── run_manifest.json
├── p6_3d/       3D lift (terminal)
│   ├── predictions/…                      (adds pose_3d)
│   ├── triangulation_metrics.json
│   └── run_manifest.json
└── logs/        one log per stage
```

**Phase summary**

| Phase | What it does | Key output |
|---|---|---|
| P1 (input, `benchmarks/runs/rtmpose-x`) | RTMDet person detection + RTMPose-X pose per frame per camera | `pose_2d` (COCO-17) + `pose_2d_native` (Halpe-26 incl. feet) per player |
| P2 | Links detections into per-camera tracklets (ByteTrack-style Kalman + pose-cosine) | `local_track_id` per player |
| P3 | Decides which tracklets across cameras are the same person (tracklet-graph, ground-plane geometry + calibrated cues) | `correspondences.jsonl`, per-cluster world `ground_xy` |
| P4 | Persistent global IDs across the whole clip (Singer-Kalman ground tracker + min-cost-flow stitching) | `global_player_id`, `ground_tracks.jsonl` |
| P5 | Role per global ID from ground trajectories | `roles.json` |
| P6 | Triangulates each identified player's 17-joint 3D skeleton (RANSAC DLT, occlusion fill, temporal smoothing) | `pose_3d.keypoints_world_m` |
| Render | Mosaic diagnostic video (reads p4 + p5 + frames + calibration + ball events) | the mp4 in `artifacts/` |

## Record format (per-camera `predictions/*.jsonl`)

One JSON object per line = one camera-frame. Fields accumulate through the phases;
`p6_3d/predictions/` is the most complete:

```jsonc
{
  "schema_version": "g1_player_frame/v0",
  "match_id": "CCPL080626",
  "delivery_id": "CCPL080626M1_1_14_1",
  "capture_group": "bt_01",            // which batch folder the camera belongs to
  "camera_id": "cam_01",
  "frame_index": 470704,               // true synchronized frame number
  "frame_name": "frame_camera01_000470704.jpg",
  "metadata": {"image_size_px": [2560, 1440]},
  "players": [
    {
      "global_player_id": "P001",      // stable across cameras + time (null before P4)
      "local_track_id": "cam_01_trk_0003",  // per-camera tracklet (null before P2)
      "role": "wicketkeeper",          // from P5 (render reads roles.json instead)
      "detection_confidence": 0.93,
      "track_confidence": 0.87,        // P3 association confidence
      "single_camera": false,          // true = only this camera saw the player
      "bbox_xywh_px": [x, y, w, h],
      "bbox_xywh_norm": [...],
      "pose_2d":        {"skeleton": "coco17", "keypoints_px": [[x,y]×17],
                         "keypoints_norm": [...], "confidence": [c×17]},
      "pose_2d_native": {"skeleton": "halpe26", ... 26 keypoints incl. toes/heels},
      "pose_3d":        {"keypoints_world_m": [[X,Y,Z]×17],   // metres, z=0 ground
                         "confidence": [c×17],
                         "mean_reprojection_error_px": [e×17]} // 100.0 = extrapolated joint
    }
  ]
}
```

Notes for consumers:

- **Same person across cameras** = same `global_player_id` in the same `frame_index`
  across the camera files. Cross-camera membership per frame is also explicit in
  `p3/diagnostics/correspondences.jsonl` (`clusters[].members[]`, `binding_id`).
- **World ground positions per ID** are easiest from `p4/diagnostics/ground_tracks.jsonl`:
  one row per frame with `tracks: [{global_player_id, ground_xy, …}]`. These are Kalman
  posteriors (smoothed), in metres, pitch-centred, z = 0 = ground.
- `pose_3d` exists only where ≥2 cameras saw the player (~37–74% of identity-frames,
  see `triangulation_coverage`); joints with reprojection error `100.0` are
  occlusion-filled/prior-extrapolated, flagged by their low confidence.
- Halpe-26 index map: 0–16 = COCO-17, 17 head, 18 neck, 19 hip-mid, 20/21 big toes,
  22/23 small toes, 24/25 heels.

## Quality: what to trust, per delivery

From the baseline metric panel (details in `docs/critical-analysis/fixes-log.md` §F0):

| Delivery | X-cam agreement | Distinct IDs (roster ~13–15) | Teleports | 3D reproj (px) | Verdict |
|---|---:|---:|---:|---:|---|
| M1_1_14_1 | 0.784 | 10 | 7 | 3.6 | pass |
| M1_1_14_2 | 0.923 | 11 | 9 | 3.5 | pass |
| M1_1_14_3 | 0.857 | 13 | 13 | 3.3 | pass |
| M1_1_14_4 | 0.772 | 13 | 14 | 3.5 | pass |
| M1_1_14_5 | 0.898 | 13 | 32 | 3.3 | warn |
| M1_1_14_6 | 0.653 | 16 | 40 | 3.2 | warn |
| M1_1_14_7 | 0.603 | 18 | 42 | 3.3 | warn |
| M2_1_12_1 | 0.791 | 16 | 154 | 3.5 | fail |

Practical guidance:

- `_1 … _4` are the cleanest clips; `_6`, `_7` have cross-camera ID disagreement on the
  facing camera pairs; `M2` has many teleport-proxy events driven by noisy single-camera
  foot projections (the emitted Kalman trajectories stay smooth — see `wip/methods_log.md`).
- Same-camera ID collisions are **0 everywhere** (hard invariant): within one camera one ID
  never labels two people in the same frame.
- Calibration is centimetre-accurate (ball reprojection p95 ≤ 4.5 px); 3D joint positions
  where measured reproject at ~3.5 px mean.
- Per-player caution flags: `single_camera: true` positions carry ~1 m uncertainty at
  grazing angles; low `track_confidence` (< 0.5) clusters are weakly associated.

## Quick-start snippets

Read one camera stream:

```python
import json
rows = [json.loads(l) for l in open(
    "benchmarks/runs/pipetrack_v6.0/deliveries/CCPL080626M1_1_14_1/"
    "p6_3d/predictions/bt_01__CCPL080626M1_1_14_1__cam_01.jsonl")]
for r in rows:
    for p in r["players"]:
        if p["global_player_id"] and p.get("pose_3d"):
            ...  # p["pose_3d"]["keypoints_world_m"]
```

World trajectory of every ID:

```python
import json, collections
traj = collections.defaultdict(list)
for line in open(".../p4/diagnostics/ground_tracks.jsonl"):
    row = json.loads(line)
    for t in row["tracks"]:
        traj[t["global_player_id"]].append((row["frame_index"], t["ground_xy"]))
```

Re-print the metric panel (with deltas vs the frozen snapshot):

```bash
/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/python -m scripts.pipetrack.run_full_pipeline \
  --panel-only --output-tree benchmarks/runs/pipetrack_v6.0 \
  --baseline benchmarks/runs/pipetrack_v6.0/_baseline_snapshot
```

## Caveats & campaign status

- This baseline intentionally reproduces the **pre-campaign** behaviour (the validated v5
  identity flag stack on RTMPose-X data). The fix campaign has since concluded: the
  accepted default is **`configs/v7/`** (run tree `pipetrack_v7-rc2`;
  `run_full_pipeline.py` now defaults to it) — consult `docs/critical-analysis/fixes-log.md`
  (GRAND ANALYSIS CONCLUSION) for what changed and why before switching.
- Do not write into `pipetrack_v6.0` — experiments reference its stage outputs in place.
- The mosaic videos show `global_player_id` colours from P4; a colour flicker = an ID
  switch, which is exactly what the campaign is reducing. Known render quirk: `cam_07`'s
  panoramic tile is stretched to 16:9 in this baseline render (fixed upstream via
  `--letterbox-tiles` for future renders).
