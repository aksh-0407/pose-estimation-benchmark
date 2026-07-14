# Run archive: `pipetrack_v7-rc3`

- **Purpose:** P1.5 isolation (no stabilization).
- **Verdict:** Rejected (worse worst-clip floor)
- **Full analysis:** fixes-log GRAND ANALYSIS
- Archived: 2026-07-14T02:30:49.011626+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-13T08:09:50.194941+00:00
- base_tree: None
- stages_run: ['p2', 'p3', 'p3_5', 'p4', 'p5', 'p6_3d']
- stabilization/lift: False/True
- config[p1b]: `configs/v6/p1b_stabilization.yaml` sha256 `a3f4512b30dcbbf1…`
- config[p2]: `configs/v6/p2_tracking.yaml` sha256 `5f46b1a2b464603e…`
- config[p3]: `configs/experiments/v7rc__p3.yaml` sha256 `dd959c21ae0641c7…`
- config[p4]: `configs/experiments/v7rc__p4.yaml` sha256 `734cf50fdc745bd1…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.785 | 11 | 5 | 0.921 | 6 | 1 | 3.305 | 0.637 |
| CCPL080626M1_1_14_2 | 0.933 | 13 | 12 | 0.943 | 7 | 1 | 3.323 | 0.354 |
| CCPL080626M1_1_14_3 | 0.858 | 12 | 17 | 0.959 | 6 | 3 | 3.217 | 0.468 |
| CCPL080626M1_1_14_4 | 0.769 | 12 | 10 | 0.958 | 7 | 3 | 3.352 | 0.490 |
| CCPL080626M1_1_14_5 | 0.899 | 12 | 48 | 0.909 | 7 | 7 | 3.106 | 0.485 |
| CCPL080626M1_1_14_6 | 0.655 | 16 | 42 | 0.848 | 10 | 3 | 3.051 | 0.448 |
| CCPL080626M1_1_14_7 | 0.591 | 15 | 43 | 0.902 | 9 | 7 | 3.176 | 0.373 |
| CCPL080626M2_1_12_1 | 0.790 | 11 | 228 | 0.956 | 7 | 6 | 3.504 | 0.400 |
