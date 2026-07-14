# Run archive: `pipetrack_v8.1-w9`

- **Purpose:** ACCEPTED v8.1 reference (KEPT): W9 union-lift + colocated merges over v8.0.
- **Verdict:** Current local reference
- **Full analysis:** fixes-log W9
- Archived: 2026-07-14T07:45:14.837864+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-14T04:31:17.271440+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v8.0
- stages_run: ['p3', 'p3_5', 'p4', 'p5', 'p6_3d']
- stabilization/lift: True/True
- config[p1b]: `configs/v8/p1b_stabilization.yaml` sha256 `d43a68a2c092eec3…`
- config[p2]: `configs/v8/p2_tracking.yaml` sha256 `193c96d3e3be2107…`
- config[p3]: `configs/experiments/w9__p3.yaml` sha256 `32fddd2ce3740363…`
- config[p4]: `configs/experiments/w9__p4.yaml` sha256 `65b5ed0a518bc985…`
- config[p5]: `configs/v8/p5_roles.yaml` sha256 `2e2ff66d58277c11…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.798 | 10 | 7 | 0.977 | 4 | 1 | 3.458 | 0.818 |
| CCPL080626M1_1_14_2 | 0.845 | 13 | 6 | 0.940 | 8 | 1 | 3.304 | 0.464 |
| CCPL080626M1_1_14_3 | 0.891 | 11 | 63 | 0.950 | 6 | 7 | 3.173 | 0.622 |
| CCPL080626M1_1_14_4 | 0.972 | 10 | 20 | 0.948 | 5 | 5 | 3.282 | 0.556 |
| CCPL080626M1_1_14_5 | 0.694 | 12 | 30 | 0.983 | 7 | 5 | 3.195 | 0.524 |
| CCPL080626M1_1_14_6 | 0.625 | 16 | 36 | 0.852 | 9 | 4 | 3.393 | 0.459 |
| CCPL080626M1_1_14_7 | 0.962 | 10 | 34 | 0.887 | 4 | 7 | 3.258 | 0.635 |
| CCPL080626M2_1_12_1 | 0.886 | 11 | 184 | 0.896 | 6 | 8 | 3.571 | 0.401 |
