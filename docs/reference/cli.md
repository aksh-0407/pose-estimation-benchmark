# CLI reference

Every entry point, its I/O and key flags. Stages are separate canonical runs — each reads an
`--input-run-dir` and writes an `--output-run-dir` (`predictions/*.jsonl` + `diagnostics/` +
`*_metrics.json`). Run everything under the `pose-lab` env as a module.

## Pipeline stages

| Command | Stage | Purpose |
| --- | --- | --- |
| `python -m core.inference.run_phase1_rtmpose_inference` | P1 | RTMDet detect + RTMPose-X pose → 2D keypoints (COCO-17 + Halpe-26). |
| `python -m core.inference.run_phase1_l40s` | P1 (remote) | Same P1 for the L40S box's `bt1/bt2/bt3` layout + tiled detector. |
| `python -m identity.p1_stabilization.run_stabilization` | 01 | One-Euro temporal smoothing of 2D keypoints. |
| `python -m identity.p2_tracking.run_per_camera_tracking` | 02 | Link detections into per-camera tracklets (`local_track_id`). |
| `python -m identity.p3_association.run_cross_camera_association` | 03 | Cluster the same player across cameras (tracklet graph → `binding_id`). |
| `python -m identity.p4_lift.run_triangulation` | 04 / terminal | Multi-view 2D → 3D world skeleton (binding-keyed, then global-keyed). |
| `python -m identity.p5_global_id.run_global_id` | 05 | Persistent `global_player_id` + tracklet stitching + ground tracks. |
| `python -m identity.p6_roles.run_role_assignment` | 06 | Batter/bowler/fielder/… roles from ground geometry. |
| `python -m identity.p6_roles.suppress_peripherals` | 06b | Role-aware peripheral suppression. |
| `python -m identity.export.export_ue_packets` | export | Triangulated 3D → Unreal Engine pose packets. |
| `python -m identity.visualization.render_videos` | render | Mosaic / per-camera / bird's-eye videos. |

Per-stage method, config and current state: [../pipeline/README.md](../pipeline/README.md).

## Orchestrators

- **`python -m main`** — full chain P1→06 + render across deliveries; `--from-stage`/`--until-stage`
  select the window, `--base-tree` reuses frozen upstream stages, then prints/diffs a joint metric
  panel. Stage dirs are `<D>/{00_inference,0N_<stage>}/`; configs default to `configs/0N_*.yaml`.
- **`python -m identity.id_pipeline`** — the identity-only inner loop (association → global-id over
  a frozen tracking tree) with the metric panel.

Common flags: `--dataset`, `--version`, `--deliveries`, `--output-tree`, `--artifacts-root`
(mosaics → `data/viz/<dataset>/pipetrack_v<num>/`), `--drive-root`, `--jobs`, `--skip-render`,
`--panel-only`, `--baseline`.

## Tools (`tools/`)

| Command | When | Purpose |
| --- | --- | --- |
| `python tools/setup_model_envs.py` | first | Provision the P1 weights (RTMPose-X + RTMDet). |
| `python tools/check_assets.py --fail-missing` | before runs | Report which checkpoints are present. |
| `python tools/check_environment.py` | debugging | Print Python, binaries, packages, GPU. |
| `python tools/sync_model_store.py` | after asset changes | Regenerate `models/<id>/` metadata + checksums. |
| `python tools/audit_repo.py --fail` | before commit | Fail if weights/frames/outputs/videos got tracked. |
| `python tools/dataset_layout.py <dir>` | ad hoc | Summarise an arbitrary dataset tree. |
| `tools/diagnosis/*` | analysis | The 40-delivery diagnosis scripts (see [../diagnosis/](../diagnosis/README.md)). |
| `tools/detector_bakeoff/*` | experiments | Detector recall bake-off harness. |
| `tools/archive_run_docs.py` | before deletion | Archive a run tree to `docs/runs/<run>.md`. |

## `--show` (renderer)

`--show {p2,p3,p4}` selects **which stage's IDs** to colour the render by — `p2` per-camera
tracks, `p3` cross-camera clusters, `p4` global identity. It is a semantic selector, not a
directory name.
