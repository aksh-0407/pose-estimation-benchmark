# Consuming a run's outputs

How a downstream consumer (broadcast biomechanics, event/officiating, the Unreal graphics layer)
reads a pipeline run. The interface is the **JSON contract**, not our code — see
[architecture.md](architecture.md) for the full `g1_player_frame/v0` schema.

## Where a run lives

A run is a tree under `data/derived/runs/<run_id>/`, one folder per delivery, one folder per stage:

```
data/derived/runs/<run_id>/deliveries/<DELIVERY>/
  01_stabilization/  02_tracking/  03_association/  04_lift/
  05_global_id/  06_roles/  07_lift3d/  logs/
  pipeline_manifest.json          # configs + sha256 + base-tree lineage
data/derived/mosaics/<run_id>/<DELIVERY>/*.mp4   # rendered mosaic / BEV
```

Each stage folder holds `predictions/*.jsonl` (one file per camera, one record per frame),
`diagnostics/`, and a `*_metrics.json`. The **final** per-player state is in `05_global_id/`
(identity) and `07_lift3d/` (terminal 3D); roles are in `06_roles/roles.json`.

## What each stage adds

| Stage | Produces | Key fields |
|---|---|---|
| P1 (input) | 2D pose per camera | `pose_2d` (COCO-17) + `pose_2d_native` (Halpe-26) |
| 02 tracking | per-camera tracklets | `local_track_id` |
| 03 association | cross-camera clusters | `correspondences.jsonl`, per-cluster `ground_xy`, `binding_id` |
| 04 lift | 3D skeleton (binding-keyed) | `pose_3d.keypoints_world_m` + `pose_3d_native` |
| 05 global identity | persistent IDs + ground tracks | `global_player_id`, `ground_tracks.jsonl` |
| 06 roles | cricket role per ID | `roles.json` (`role`, `bowling_direction_xy`) |
| render | diagnostic video | mosaic `.mp4` (reads 05 + 06 + frames + calibration) |

## Fields you most likely want

- **Who** — `global_player_id` (stable across cameras + time; `null` before global identity).
- **Where (3D)** — `pose_3d.keypoints_world_m` (17×[x,y,z] metres, world origin = pitch centre;
  `null` until triangulation) + `mean_reprojection_error_px` per joint. Full Halpe-26 in
  `pose_3d_native`.
- **Where (ground)** — `05_global_id/diagnostics/ground_tracks.jsonl` (fused world XY per ID/frame,
  the smooth bird's-eye channel).
- **Role** — `06_roles/roles.json` (the render reads roles from here, not from the frame records).
- **Quality** — `track_confidence`, `single_camera`, and the run's `*_metrics.json`.

Same-camera collisions are 0 by construction; identity/coverage caveats and current numbers are in
[diagnosis/](diagnosis/README.md).

## Unreal Engine packets

For UE-format packets rather than the JSONL 3D, run the exporter (world-metres → UE-cm transform):

```bash
python -m identity.export.export_ue_packets --input <07_lift3d/predictions.jsonl> \
  --output <ue_packets.jsonl> --model-version <v>
```

See [pipeline/07-export-and-render.md](pipeline/07-export-and-render.md).

## Reproducibility

`pipeline_manifest.json` records every stage's config path + sha256 and the base-tree lineage, so
any run is traceable to the exact configuration that produced it. Heavy payloads (predictions,
videos) live under gitignored `data/derived/`; only the compact metrics/manifests are ever small
enough to share directly — coordinate large-output sharing out of band.
