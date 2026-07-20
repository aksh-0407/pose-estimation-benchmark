# Run archive: `pipetrack_v7-rc2`

- **Purpose:** Re-composed v7 on fixed code; stitcher live first time.
- **Verdict:** Accepted as v7 default (superseded by v8.0)
- **Full analysis:** methods_log v7-rc2 + GRAND ANALYSIS
- Archived: 2026-07-14T02:30:49.009593+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-13T07:45:59.057448+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v7-rc1
- stages_run: ['p3', 'p3_5', 'p4', 'p5', 'p6_3d']
- stabilization/lift: True/True
- config[p1b]: `configs/experiments/v7rc__p1b.yaml` sha256 `825da4ad97c9f020…`
- config[p2]: `configs/v6/p2_tracking.yaml` sha256 `5f46b1a2b464603e…`
- config[p3]: `configs/experiments/v7rc__p3.yaml` sha256 `dd959c21ae0641c7…`
- config[p4]: `configs/experiments/v7rc__p4.yaml` sha256 `734cf50fdc745bd1…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.785 | 11 | 5 | 0.921 | 6 | 1 | 3.306 | 0.637 |
| CCPL080626M1_1_14_2 | 0.933 | 12 | 12 | 0.927 | 6 | 1 | 3.334 | 0.486 |
| CCPL080626M1_1_14_3 | 0.860 | 12 | 20 | 0.959 | 6 | 3 | 3.213 | 0.469 |
| CCPL080626M1_1_14_4 | 0.770 | 12 | 19 | 0.952 | 7 | 4 | 3.332 | 0.489 |
| CCPL080626M1_1_14_5 | 0.778 | 12 | 45 | 0.928 | 7 | 6 | 2.966 | 0.450 |
| CCPL080626M1_1_14_6 | 0.655 | 16 | 44 | 0.848 | 10 | 3 | 3.049 | 0.448 |
| CCPL080626M1_1_14_7 | 0.703 | 13 | 35 | 0.881 | 6 | 3 | 3.069 | 0.517 |
| CCPL080626M2_1_12_1 | 0.781 | 11 | 223 | 0.956 | 7 | 6 | 3.511 | 0.408 |
