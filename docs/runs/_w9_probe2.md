# Run archive: `_w9_probe2`

- **Purpose:** W9 gate-widening probe (_7).
- **Verdict:** Diagnostic
- **Full analysis:** fixes-log W9
- Archived: 2026-07-14T07:45:14.839278+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-14T03:59:07.033329+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v8.0
- stages_run: ['p3', 'p3_5', 'p4']
- stabilization/lift: True/True
- config[p1b]: `configs/v8/p1b_stabilization.yaml` sha256 `d43a68a2c092eec3…`
- config[p2]: `configs/v8/p2_tracking.yaml` sha256 `193c96d3e3be2107…`
- config[p3]: `configs/experiments/w9__p3.yaml` sha256 `b13f5a8a2c35aa06…`
- config[p4]: `configs/v8/p4_global_id.yaml` sha256 `b20553ad3f385341…`
- config[p5]: `configs/v8/p5_roles.yaml` sha256 `2e2ff66d58277c11…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_7 | 0.828 | 11 | 39 | 0.893 | 5 | 6 | - | - |
