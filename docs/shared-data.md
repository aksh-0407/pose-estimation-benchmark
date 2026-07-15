# Consuming a run's outputs

How a downstream consumer (broadcast biomechanics, event/officiating, the Unreal graphics layer)
reads a pipeline run. The interface is the **JSON contract**, not our code — see
[architecture.md](architecture.md) for the full `g1_player_frame/v1` schema.

## Where a run lives

A run is a tree under `data/derived/<dataset>/pipetrack_v<num>/`, one folder per delivery, one
folder per stage (P1 lands as the per-delivery `00_inference` stage):

```
data/derived/<dataset>/pipetrack_v<num>/<DELIVERY>/
  00_inference/  01_stabilization/  02_tracking/  03_association/  04_lift/
  05_global_id/  06_roles/  logs/
data/derived/<dataset>/pipetrack_v<num>/pipeline_manifest.json   # configs + sha256 + base-tree lineage
data/viz/<dataset>/pipetrack_v<num>/<DELIVERY>/*.mp4             # rendered mosaic / BEV
```

Each stage folder holds `predictions/*.jsonl` (one file per camera, one record per frame),
`diagnostics/`, and a `*_metrics.json`. The **final, consume-this** per-player state is
`06_roles/predictions/` (2D + 3D + `global_player_id` + `role`); the role roster is also in
`06_roles/roles.json`.

## What each stage adds

| Stage | Produces | Key fields |
|---|---|---|
| 00 inference (P1) | 2D pose per camera | `pose_2d` (Halpe-26, 26 joints incl. feet) |
| 02 tracking | per-camera tracklets | `local_track_id` |
| 03 association | cross-camera clusters | `correspondences.jsonl`, per-cluster `ground_xy`, `binding_id` |
| 04 lift | 3D skeleton (binding-keyed) | `pose_3d.keypoints_world_m` (Halpe-26, 26) + `pose_3d_named` |
| 05 global identity | persistent IDs + ground tracks | `global_player_id`, `ground_tracks.jsonl` |
| 06 roles | role per ID + **terminal per-camera output** | `predictions/` (2D+3D+id+role), `roles.json` (`role`, `bowling_direction_xy`) |
| render | diagnostic video | mosaic `.mp4` (reads 05 + 06 + frames + calibration) |

## Fields you most likely want

- **Who** — `global_player_id` (stable across cameras + time; `null` before global identity).
- **Where (3D)** — `pose_3d.keypoints_world_m` (Halpe-26, 26×[x,y,z] metres, world origin = pitch
  centre; per-joint `null` where not triangulated) + `mean_reprojection_error_px` per joint. The
  self-describing `pose_3d_named` (root in world + joints root-relative) is the rig-friendly view.
- **Where (ground)** — `05_global_id/diagnostics/ground_tracks.jsonl` (fused world XY per ID/frame,
  the smooth bird's-eye channel), or `pose_3d_named.root_world_m`.
- **Role** — stamped on every player in `06_roles/predictions/`, and summarised in `06_roles/roles.json`.
- **Quality** — `track_confidence`, `single_camera`, and the run's `*_metrics.json`.

Same-camera collisions are 0 by construction; identity/coverage caveats and current numbers are in
[diagnosis/](diagnosis/README.md).

## Unreal Engine packets

For UE-format packets rather than the JSONL 3D, run the exporter (world-metres → UE-cm transform):

```bash
python -m identity.export.export_ue_packets --run-dir <06_roles> \
  --output <ue_packets.jsonl> --model-version <v>
```

See [pipeline/07-export-and-render.md](pipeline/07-export-and-render.md).

## Reproducibility

`pipeline_manifest.json` records every stage's config path + sha256 and the base-tree lineage, so
any run is traceable to the exact configuration that produced it. Heavy payloads (predictions,
videos) live under gitignored `data/derived/`; only the compact metrics/manifests are ever small
enough to share directly — coordinate large-output sharing out of band.
