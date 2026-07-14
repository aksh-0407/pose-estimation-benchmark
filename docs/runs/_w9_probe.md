# Run archive: `_w9_probe`

- **Purpose:** W9 union-lift iteration probe (_7,_2,_6).
- **Verdict:** Diagnostic; superseded by v8.1-w9
- **Full analysis:** fixes-log W9
- Archived: 2026-07-14T07:45:14.838861+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-14T03:52:33.459099+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v8.0
- stages_run: ['p3', 'p3_5', 'p4']
- stabilization/lift: True/True
- config[p1b]: `configs/v8/p1b_stabilization.yaml` sha256 `d43a68a2c092eec3…`
- config[p2]: `configs/v8/p2_tracking.yaml` sha256 `193c96d3e3be2107…`
- config[p3]: `configs/experiments/w9__p3.yaml` sha256 `2fda344aa7ac61b7…`
- config[p4]: `configs/v8/p4_global_id.yaml` sha256 `b20553ad3f385341…`
- config[p5]: `configs/v8/p5_roles.yaml` sha256 `2e2ff66d58277c11…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_2 | 0.802 | 14 | 3 | 0.916 | 9 | 0 | - | - |
| CCPL080626M1_1_14_6 | 0.477 | 18 | 53 | 0.866 | 11 | 3 | - | - |
| CCPL080626M1_1_14_7 | 0.811 | 12 | 55 | 0.867 | 5 | 6 | - | - |
