# Pose Estimation Pipeline Data Details

## 1. Purpose and scope

This document describes the saved outputs for Phases 1 through 4 of the cricket
pose-estimation and player-tracking pipeline. It is intended as a technical handoff
for continuing development, evaluation, visualization, or downstream integration.

The two relevant data roots are:

```text
data/derived/runs/rtmpose-l-body8-full-db32-pb96
benchmarks/artifacts/pipetrack-all-deliveries-v2
```

The first directory contains the shared Phase 1 perception run. The second contains
the Phase 2, Phase 3, Phase 4, and rendered-video outputs, grouped by delivery.

Work on the tracking and global-identity pipeline is still in progress. If the
pipeline or its outputs are updated, a new data snapshot will be shared.

## 2. Pipeline overview

The phases form a sequential enrichment pipeline:

```text
Phase 1: per-frame person detection and 2D pose estimation
    |
    v
Phase 2: per-camera temporal tracking
    |
    v
Phase 3: cross-camera association at synchronized frames
    |
    v
Phase 4: persistent global identity assignment across cameras and time
```

Each phase writes a complete output snapshot. A later phase does not modify the
previous phase's directory in place. Instead, it reads the previous snapshot,
preserves existing pose and tracking information, adds its own fields and
diagnostics, and writes a new set of prediction JSONL files.

This separation provides reproducible phase boundaries, independent inspection and
debugging, phase-specific metrics, and protection of earlier output. The tradeoff is
intentional data duplication: P2, P3, and P4 each save complete prediction streams
rather than only their field-level changes.

## 3. Meaning of each phase

| Phase | Scope | Main result | Important fields/artifacts |
| --- | --- | --- | --- |
| **P1** | Each frame and camera independently | Person boxes and COCO-17 2D poses | `bbox_xywh_px`, `bbox_xywh_norm`, `pose_2d` |
| **P2** | Time within one camera | Stable camera-local tracks | `local_track_id`, per-camera tracking diagnostics |
| **P3** | Synchronized observations across cameras | Cross-camera correspondence clusters | `single_camera`, association `track_confidence`, `correspondences.jsonl` |
| **P4** | Cameras and time over one delivery | Persistent delivery-level player identities | `global_player_id`, `track_state`, `ground_tracks.jsonl` |

### 3.1 Phase 1: per-camera perception

Phase 1 runs RTMPose-L with a person detector and produces one record per source
camera frame. Each detected person contains a bounding box, 17 COCO body keypoints,
and one confidence value per keypoint.

At this stage:

```text
local_track_id   = null
global_player_id = null
role             = "unknown"
pose_3d          = null
```

P1 has no temporal memory. The same person in two consecutive frames is represented
by two independent detections.

### 3.2 Phase 2: per-camera tracking

Phase 2 links P1 detections over time, independently for each camera. A successfully
tracked person receives a camera-local identifier such as:

```text
cam_01_trk_0001
```

This identifier is meaningful only inside that camera and delivery. For example,
`cam_01_trk_0001` and `cam_04_trk_0001` must not be assumed to be the same player.
Some unmatched or low-confidence detections can retain a null `local_track_id`.

### 3.3 Phase 3: cross-camera association

Phase 3 processes synchronized camera frames and groups observations estimated to
show the same physical person. Association uses camera calibration, ground-plane
geometry, epipolar/reprojection consistency, and P2 tracking information.

The detailed association result is stored in:

```text
p3/diagnostics/correspondences.jsonl
```

Each line represents one synchronized frame and contains correspondence clusters.
A cluster records its frame-local `cluster_id`, member cameras and player indexes,
local track IDs, estimated `ground_xy`, single-camera status, camera support,
geometry diagnostics, and association confidence.

`cluster_id` is frame-local diagnostic information, not a persistent identity. It is
stored in `correspondences.jsonl`, not as a canonical player field in every P3
prediction record.

### 3.4 Phase 4: global identity tracking

Phase 4 carries P3 associations through time and assigns persistent identifiers:

```text
P001, P002, P003, ...
```

These IDs are intended to remain stable across cameras and frames within one
delivery. P4 records also include a `track_state`, such as `tentative` or
`confirmed`.

P4 saves ground-plane tracks, identity stitching/switch information, collision and
fragmentation checks, completeness summaries, and a rendered seven-camera mosaic.

P4 is not the 3D-pose phase. In these files, `pose_3d` remains null. A later
triangulation/export stage is responsible for producing 3D keypoints.

## 4. Current data inventory

The current snapshot covers eight deliveries:

```text
CCPL080626M1_1_14_1
CCPL080626M1_1_14_2
CCPL080626M1_1_14_3
CCPL080626M1_1_14_4
CCPL080626M1_1_14_5
CCPL080626M1_1_14_6
CCPL080626M1_1_14_7
CCPL080626M2_1_12_1
```

Each delivery has seven camera streams and 600 synchronized frame records per
camera.

| Data | Files or records |
| --- | ---: |
| P1 prediction streams | 56 JSONL files |
| P1 frame records | 33,600 |
| P1 detected-person records | 104,265 |
| P1 visual-QA images | 280 JPEGs |
| P2 prediction streams | 56 JSONL files |
| P3 prediction streams | 56 JSONL files |
| P4 prediction streams | 56 JSONL files |
| P2-P4 frame records combined | 100,800 |
| P4 delivery mosaic videos | 8 MP4 files |

Approximate on-disk sizes are:

```text
Phase 1 run:             480 MB
Phase 2-4 artifact root: 889 MB
```

## 5. Capture groups and prediction filenames

The cameras are stored under three source capture groups:

| Capture group | Cameras |
| --- | --- |
| `bt_01` | `cam_01`, `cam_04` |
| `bt_02` | `cam_02`, `cam_05`, `cam_07` |
| `bt_03` | `cam_03`, `cam_06` |

The `bt_01`, `bt_02`, and `bt_03` labels are capture groups, not pipeline phase
labels.

Prediction files use:

```text
<capture_group>__<delivery_id>__<camera_id>.jsonl
```

For example:

```text
bt_01__CCPL080626M1_1_14_1__cam_01.jsonl
```

identifies capture group `bt_01`, delivery `CCPL080626M1_1_14_1`, and camera
`cam_01`.

## 6. Phase 1 directory

Phase 1 is stored at:

```text
data/derived/runs/rtmpose-l-body8-full-db32-pb96/
```

The run name records the main inference configuration:

```text
rtmpose-l  RTMPose-L pose model
body8      Body8 training/configuration family
full       full dataset inference
db32       detector batch size 32
pb96       pose batch size 96
```

Directory layout:

```text
rtmpose-l-body8-full-db32-pb96/
|-- run_manifest.json
|-- p1_metrics.json
|-- predictions/
|   `-- <capture-group>__<delivery>__<camera>.jsonl
|-- delivery_metrics/
|   `-- <delivery>/
|       |-- run_manifest.json
|       `-- p1_metrics.json
`-- visualizations/
    |-- visual_qa_manifest.json
    `-- <capture-group>/<delivery>/<camera>/frame_*.jpg
```

### 6.1 `predictions/`

This is the canonical P1 data source. It contains 56 streams:

```text
8 deliveries x 7 cameras = 56 JSONL files
```

Each file contains 600 lines. Each line is one frame record, whose `players` array
contains zero or more detections.

### 6.2 Root manifest and metrics

`run_manifest.json` describes the inference run, including run/model IDs, model
configuration and checkpoints, device and batch settings, schemas, output locations,
timings, and aggregate summaries.

Relevant schema values are:

```text
run manifest schema: cricket_phase1_run/v2
prediction schema:   g1_player_frame/v0
```

`p1_metrics.json` aggregates all deliveries and contains per-camera counts, failures,
processed-record totals, detected-person totals, model settings, and visualization
references.

### 6.3 `delivery_metrics/<delivery>/`

Every delivery has its own manifest and P1 metrics, restricted to its seven camera
streams. The prediction streams are not duplicated here; they remain under the root
`predictions/` directory and are referenced by filename.

### 6.4 `visualizations/`

This directory contains still-frame visual-QA overlays. The current snapshot has
five images per camera stream:

```text
56 camera streams x 5 images = 280 JPEGs
```

These JPEGs are for qualitative inspection and are not prediction inputs.

## 7. Phase 2-4 artifact directory

The downstream tracking data is stored at:

```text
benchmarks/artifacts/pipetrack-all-deliveries-v2/
```

Unlike the shared P1 run, P2-P4 are organized first by delivery because synchronized
tracking and association are processed one delivery at a time.

```text
pipetrack-all-deliveries-v2/
|-- <delivery>/
|   |-- p2/
|   |   |-- predictions/
|   |   |-- diagnostics/
|   |   |-- tracking_metrics.json
|   |   `-- run_manifest.json
|   |-- p3/
|   |   |-- predictions/
|   |   |-- diagnostics/correspondences.jsonl
|   |   |-- association_metrics.json
|   |   `-- run_manifest.json
|   |-- p4/
|   |   |-- predictions/
|   |   |-- diagnostics/correspondences.jsonl
|   |   |-- diagnostics/ground_tracks.jsonl
|   |   |-- global_id_metrics.json
|   |   |-- id_switch_report.json
|   |   `-- run_manifest.json
|   |-- p2.log
|   |-- p3.log
|   |-- p4.log
|   `-- video.log
`-- videos/
    `-- <delivery>/
        |-- <delivery>__all_cameras.mp4
        `-- video_manifest.json
```

The `v2` suffix in `pipetrack-all-deliveries-v2` names this delivered artifact set.
Individual JSON files declare their own schema versions internally.

## 8. Phase-specific files

### 8.1 P2 files

For each delivery, `p2/` contains seven complete prediction streams, one diagnostic
JSON per camera, `tracking_metrics.json`, and `run_manifest.json`.

The P2 manifest points to the shared P1 run. The P2 predictions preserve P1 boxes and
poses and add camera-local tracking IDs. The metrics report per-camera tracker
diagnostics such as confirmed tracks, unmatched detections, calibration failures,
gating rejects, dormant-track re-identification, and an intra-camera switch proxy.

### 8.2 P3 files

For each delivery, `p3/` contains seven enriched prediction streams,
`diagnostics/correspondences.jsonl`, `association_metrics.json`, and
`run_manifest.json`.

The metrics summarize cluster counts, camera support, single-camera rate, ground
spread, reprojection error, cycle consistency, confidence, and calibration checks.

### 8.3 P4 files

For each delivery, `p4/` contains seven final prediction streams plus:

- `diagnostics/correspondences.jsonl`: P3 correspondence provenance;
- `diagnostics/ground_tracks.jsonl`: frame-by-frame `P###` ground positions;
- `global_id_metrics.json`: identity, collision, fragmentation, and completeness
  summaries;
- `id_switch_report.json`: delivery-level identity stitching events; and
- `run_manifest.json`: P3 input, P4 configuration, and artifact locations.

The P4 prediction streams are the appropriate input for work that requires
delivery-level player identities.

### 8.4 Logs and videos

`p2.log`, `p3.log`, `p4.log`, and `video.log` capture console output from processing
and rendering. They are useful when diagnosing incomplete or failed runs.

Each delivery has one P4-rendered mosaic:

```text
videos/<delivery>/<delivery>__all_cameras.mp4
```

The video is a synchronized 1920x1080 seven-camera layout with summary and roster
panels. `video_manifest.json` records frame count, FPS, encoder, CRF, dimensions,
keypoint threshold, source stage, and output path.

## 9. Canonical JSONL record

Prediction streams use a frame-oriented contract. A simplified P4 record is:

```json
{
  "schema_version": "g1_player_frame/v0",
  "match_id": "CCPL080626",
  "delivery_id": "CCPL080626M1_1_14_1",
  "capture_group": "bt_01",
  "camera_id": "cam_01",
  "frame_index": 212334,
  "frame_name": "frame_camera01_000212334.jpg",
  "metadata": {
    "model_id": "rtmpose_l_body8",
    "image_size_px": [2560, 1440]
  },
  "players": [
    {
      "bbox_xywh_px": [562.1, 463.7, 175.9, 450.0],
      "bbox_xywh_norm": [0.2196, 0.3220, 0.0687, 0.3125],
      "pose_2d": {
        "skeleton": "coco_17",
        "keypoints_px": [[682.3, 519.5]],
        "keypoints_norm": [[0.2665, 0.3608]],
        "confidence": [0.8471]
      },
      "local_track_id": "cam_01_trk_0001",
      "global_player_id": "P001",
      "track_confidence": 0.7582,
      "track_state": "confirmed",
      "single_camera": false,
      "role": "unknown",
      "pose_3d": null
    }
  ]
}
```

The keypoint arrays are abbreviated above. Actual `keypoints_px`, `keypoints_norm`,
and `confidence` arrays contain 17 entries each.

Field evolution by phase:

| Field | P1 | P2 | P3 | P4 |
| --- | --- | --- | --- | --- |
| Bounding box and `pose_2d` | populated | preserved | preserved | preserved |
| `local_track_id` | null | populated where tracked | preserved | preserved |
| `single_camera` | absent | absent | populated | preserved |
| Association `track_confidence` | null | generally null | populated | preserved |
| `global_player_id` | null | null | null | populated where assigned |
| `track_state` | absent | absent | absent | populated |
| `pose_3d` | null | null | null | null |

## 10. Linking records and diagnostics

### 10.1 Prediction to source image

Use `capture_group`, `delivery_id`, `camera_id`, and `frame_name`. Source images are
under:

```text
drive/dataset/<capture_group>/<delivery_id>/camera0N/<frame_name>
```

### 10.2 P3 cluster to prediction player

A correspondence member contains a `cam_id`, `player_index`, and optional
`local_track_id`. For the correspondence row's `frame_index`, select the matching
camera record and use `player_index` as the index into its `players` array.

### 10.3 P4 ground track to prediction player

Each `ground_tracks.jsonl` row contains a `frame_index` and tracks with
`global_player_id` and `ground_xy`. Join them to P4 players using `frame_index` and
`global_player_id`.

## 11. Manifests and lineage

The manifests establish the processing chain:

```text
P1 run manifest
  -> P2 run_manifest.input_run_dir
      -> P3 run_manifest.input_run_dir
          -> P4 run_manifest.input_run_dir
```

P2-P4 manifests also record delivery and match IDs, input files, calibration/drive
locations, full phase configuration, expected frame count, and output artifacts.

Some paths record the absolute path on the generation machine. After copying the
dataset, resolve files from the documented directory hierarchy rather than assuming
those stored absolute paths still exist.

## 12. Metrics interpretation

Metrics provide operational validation and geometry/tracking/identity summaries.
No labelled identity ground-truth file was supplied for these runs. Therefore,
identity-switch, association, fragmentation, and completeness values should be
treated as pipeline proxy metrics rather than labelled MOTA/IDF1 measurements.

A phase-level `"status": "pass"` means the phase passed its implemented validation
checks; it is not a ground-truth accuracy guarantee.

## 13. Recommended inputs for continuing work

| Downstream task | Recommended input |
| --- | --- |
| Person detection or 2D pose analysis | P1 `predictions/` |
| Single-camera tracking analysis | Delivery P2 `predictions/` |
| Cross-camera association analysis | P3 predictions and `correspondences.jsonl` |
| Persistent player-ID processing | Delivery P4 `predictions/` |
| Top-down movement analysis | P4 `ground_tracks.jsonl` |
| Qualitative identity review | Delivery mosaic MP4 |
| 3D triangulation or export | P4 predictions as identity-resolved input |

For most work continuing after global identity assignment, start from:

```text
benchmarks/artifacts/pipetrack-all-deliveries-v2/<delivery>/p4/
```

Do not mix predictions and diagnostics from different deliveries. Global IDs should
also be treated as delivery-scoped unless a separate match-level reconciliation stage
is introduced.

## 14. Data transfer and updates

The Phase 1 run is retained under `benchmarks/runs`. The generated
`benchmarks/artifacts` hierarchy is ignored by the repository's default Git policy,
so a source-code checkout alone may not include P2-P4 predictions, diagnostics, logs,
or videos. These artifacts must be transferred separately.

This document describes the current handoff snapshot. Tracking, association, and
global-identity work remains in progress. If processing logic, configuration, schema,
or generated data changes, an updated data snapshot and relevant documentation will
be shared.
