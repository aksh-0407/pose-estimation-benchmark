# Run archive: `_v8_nospawn_probe`

- **Purpose:** lowconf_can_spawn=false probe (_5,_6,_7,M2).
- **Verdict:** Adopted into v8.0 (strict improvement)
- **Full analysis:** methods_log GRAND ANALYSIS v2
- Archived: 2026-07-14T02:30:48.990588+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-14T00:19:17.765363+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v8-rc1
- stages_run: ['p2', 'p3', 'p3_5', 'p4', 'p5', 'p6_3d']
- stabilization/lift: True/True
- config[p1b]: `configs/v7/p1b_stabilization.yaml` sha256 `d43a68a2c092eec3…`
- config[p2]: `configs/experiments/v8_p2_nospawn.yaml` sha256 `56b5d3c652d45bb7…`
- config[p3]: `configs/v7/p3_association.yaml` sha256 `ce303573b7e76d1d…`
- config[p4]: `configs/v7/p4_global_id.yaml` sha256 `b20553ad3f385341…`
- config[p5]: `configs/experiments/w6__p5.yaml` sha256 `2e2ff66d58277c11…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_5 | 0.627 | 13 | 30 | 0.964 | 8 | 3 | 3.091 | 0.410 |
| CCPL080626M1_1_14_6 | 0.477 | 18 | 53 | 0.866 | 11 | 3 | 3.625 | 0.390 |
| CCPL080626M1_1_14_7 | 0.811 | 12 | 55 | 0.867 | 5 | 6 | 3.139 | 0.543 |
| CCPL080626M2_1_12_1 | 0.886 | 11 | 184 | 0.896 | 6 | 8 | 3.571 | 0.401 |
