# Run archive: `_p3_prefetch_check`

- **Purpose:** P3 appearance-prefetch byte-identity tree (M2).
- **Verdict:** Diagnostic (identity proven)
- **Full analysis:** fixes-log W10-PERF
- Archived: 2026-07-14T07:45:14.839874+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-14T02:54:42.450710+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v8.0
- stages_run: ['p3']
- stabilization/lift: True/True
- config[p1b]: `configs/v8/p1b_stabilization.yaml` sha256 `d43a68a2c092eec3…`
- config[p2]: `configs/v8/p2_tracking.yaml` sha256 `193c96d3e3be2107…`
- config[p3]: `configs/v8/p3_association.yaml` sha256 `ce303573b7e76d1d…`
- config[p4]: `configs/v8/p4_global_id.yaml` sha256 `b20553ad3f385341…`
- config[p5]: `configs/v8/p5_roles.yaml` sha256 `2e2ff66d58277c11…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}
