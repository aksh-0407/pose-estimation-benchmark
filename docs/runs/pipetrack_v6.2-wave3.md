# Run archive: `pipetrack_v6.2-wave3`

- **Purpose:** Wave-3 stack (F9a covariance, F10 R, F11 shape, F12 posture stitch).
- **Verdict:** Accepted into v7 lineage
- **Full analysis:** fixes-log W3
- Archived: 2026-07-14T02:30:49.002022+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-10T09:17:59.973927+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v6.0
- stages_run: ['p3', 'p3_5', 'p4', 'p5', 'p6_3d']
- stabilization/lift: False/True
- config[p1b]: `configs/v6/p1b_stabilization.yaml` sha256 `a3f4512b30dcbbf1…`
- config[p2]: `configs/v6/p2_tracking.yaml` sha256 `5f46b1a2b464603e…`
- config[p3]: `configs/experiments/v6_wave3__p3.yaml` sha256 `cbed196e4683af32…`
- config[p4]: `configs/experiments/v6_wave3__p4.yaml` sha256 `736695d79efa1646…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'ema_alpha': 0.65, 'min_views': 2, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.784 | 10 | 7 | 0.951 | 4 | 0 | 3.577 | 0.737 |
| CCPL080626M1_1_14_2 | 0.923 | 11 | 9 | 0.960 | 5 | 0 | 3.462 | 0.502 |
| CCPL080626M1_1_14_3 | 0.854 | 13 | 15 | 0.938 | 7 | 0 | 3.317 | 0.468 |
| CCPL080626M1_1_14_4 | 0.770 | 13 | 10 | 0.943 | 8 | 0 | 3.475 | 0.480 |
| CCPL080626M1_1_14_5 | 0.898 | 14 | 45 | 0.874 | 9 | 0 | 3.284 | 0.487 |
| CCPL080626M1_1_14_6 | 0.655 | 15 | 47 | 0.896 | 9 | 0 | 3.235 | 0.451 |
| CCPL080626M1_1_14_7 | 0.605 | 17 | 44 | 0.910 | 11 | 0 | 3.337 | 0.503 |
| CCPL080626M2_1_12_1 | 0.787 | 14 | 191 | 0.788 | 10 | 0 | 3.539 | 0.380 |
