# Run archive: `pipetrack_v8-rc1`

- **Purpose:** Composed v8 candidate: tiled+NMS55 x8, v7 stack, W6.
- **Verdict:** Superseded by v8.0 (adds no-spawn)
- **Full analysis:** fixes-log GRAND ANALYSIS v2
- Archived: 2026-07-14T02:30:49.017709+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-14T00:03:14.889569+00:00
- base_tree: None
- stages_run: ['p1b', 'p2', 'p3', 'p3_5', 'p4', 'p5', 'p6_3d']
- stabilization/lift: True/True
- config[p1b]: `configs/v7/p1b_stabilization.yaml` sha256 `d43a68a2c092eec3…`
- config[p2]: `configs/v7/p2_tracking.yaml` sha256 `d82a07f1c4b28fca…`
- config[p3]: `configs/v7/p3_association.yaml` sha256 `ce303573b7e76d1d…`
- config[p4]: `configs/v7/p4_global_id.yaml` sha256 `b20553ad3f385341…`
- config[p5]: `configs/experiments/w6__p5.yaml` sha256 `2e2ff66d58277c11…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.798 | 11 | 7 | 0.969 | 5 | 0 | 3.273 | 0.720 |
| CCPL080626M1_1_14_2 | 0.802 | 12 | 4 | 0.959 | 7 | 2 | 3.413 | 0.501 |
| CCPL080626M1_1_14_3 | 0.882 | 12 | 70 | 0.950 | 7 | 5 | 3.161 | 0.662 |
| CCPL080626M1_1_14_4 | 0.972 | 10 | 19 | 0.931 | 5 | 4 | 3.359 | 0.614 |
| CCPL080626M1_1_14_5 | 0.627 | 13 | 33 | 0.961 | 8 | 4 | 3.091 | 0.410 |
| CCPL080626M1_1_14_6 | 0.477 | 19 | 57 | 0.862 | 12 | 1 | 3.634 | 0.373 |
| CCPL080626M1_1_14_7 | 0.812 | 13 | 49 | 0.836 | 6 | 6 | 3.152 | 0.528 |
| CCPL080626M2_1_12_1 | 0.886 | 11 | 201 | 0.895 | 6 | 7 | 3.572 | 0.401 |
