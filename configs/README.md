# configs/ - live pipeline configuration (v8.1 stack)

One numbered YAML per identity stage, matching the `src/identity/pN_<stage>/` packages and the
`data/derived/runs/<id>/deliveries/<D>/0N_<stage>/` run-dir layout. `src/main.py` defaults to
these; override per stage with `--pN-config`.

| File | Stage | Notes |
|---|---|---|
| `01_stabilization.yaml` | 01 stabilization | One-Euro 2D smoothing; `enabled: true` |
| `02_tracking.yaml` | 02 per-camera tracking | ByteTrack-style; `lowconf_can_spawn: false` (v8) |
| `03_association.yaml` | 03 cross-camera association | tracklet-graph LLR + W9 union-lift |
| `05_global_id.yaml` | 05 global identity | Singer-KF + min-cost-flow stitch + colocated merge |
| `06_roles.yaml` | 06 roles | v1.2 epoch solver + Wave-6 peripheral suppression |

Stage **04 (3D lift)** has no YAML - its parameters are CLI flags on
`src/identity/p4_lift/run_triangulation.py` (set by `src/main.py`).

Shared: `keypoint_mappings.yaml` (COCO-17 / Halpe-26 skeletons), `model_envs.yaml` +
`model_registry.yaml` (P1 model catalog), `reference/{ground_conventions,camera_layout}.jpeg`.

P1 (2D inference, upstream of identity) is produced with the tiled detector:
```
python -m core.inference.run_phase1_rtmpose_inference --model-id rtmpose_x_body8 ...
# on the L40S box (bt1/bt2/bt3 layout): src/core/inference/run_phase1_l40s.py --tiled-det --nms-thr 0.55
```
(RTMDet-m person on a 4×2 overlap-0.25 tile grid + full frame, cross-tile NMS 0.55 + IoM-0.7
containment suppression; RTMPose-X Halpe-26 pose.)
