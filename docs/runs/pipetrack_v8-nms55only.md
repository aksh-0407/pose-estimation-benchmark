# Run archive: `pipetrack_v8-nms55only`

- **Purpose:** Tiled NMS-0.55 ablation without contested (_7+M2).
- **Verdict:** Winner; became v8 detection spec
- **Full analysis:** methods_log W5B-LIVE
- Archived: 2026-07-14T02:30:49.015727+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-13T21:22:55.849899+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v8-nms55w5b
- stages_run: ['p3', 'p3_5', 'p4', 'p5', 'p6_3d']
- stabilization/lift: True/True
- config[p1b]: `configs/v7/p1b_stabilization.yaml` sha256 `d43a68a2c092eec3…`
- config[p2]: `configs/v7/p2_tracking.yaml` sha256 `d82a07f1c4b28fca…`
- config[p3]: `configs/v7/p3_association.yaml` sha256 `ce303573b7e76d1d…`
- config[p4]: `configs/v7/p4_global_id.yaml` sha256 `b20553ad3f385341…`
- config[p5]: `configs/v7/p5_roles.yaml` sha256 `231f076708d1aa51…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_7 | 0.804 | 12 | 57 | 0.882 | 5 | 7 | 3.154 | 0.519 |
| CCPL080626M2_1_12_1 | 0.887 | 12 | 196 | 0.903 | 7 | 6 | 3.361 | 0.271 |
