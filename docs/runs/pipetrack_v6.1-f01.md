# Run archive: `pipetrack_v6.1-f01`

- **Purpose:** Wave-0 A/B: P1.5 stabilization wired (F1).
- **Verdict:** Accepted into v7 lineage
- **Full analysis:** fixes-log F1
- Archived: 2026-07-14T02:30:48.997812+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-10T08:36:04.198795+00:00
- base_tree: None
- stages_run: ['p1b', 'p2', 'p3', 'p3_5', 'p4', 'p5', 'p6_3d']
- stabilization/lift: True/False
- config[p1b]: `configs/experiments/v6_f01_stabilization__p1b.yaml` sha256 `825da4ad97c9f020…`
- config[p2]: `configs/v6/p2_tracking.yaml` sha256 `5f46b1a2b464603e…`
- config[p3]: `configs/v6/p3_association.yaml` sha256 `6d6593b3653549df…`
- config[p4]: `configs/v6/p4_global_id.yaml` sha256 `ce283a214b491eaa…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': False, 'ema_alpha': 0.65, 'min_views': 2, 'reprojection_threshold_px': 10.0, 'smoother': 'ema'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.784 | 10 | 7 | 0.951 | 4 | 0 | 3.573 | 0.737 |
| CCPL080626M1_1_14_2 | 0.923 | 11 | 9 | 0.964 | 5 | 0 | 3.458 | 0.502 |
| CCPL080626M1_1_14_3 | 0.858 | 13 | 15 | 0.930 | 7 | 0 | 3.311 | 0.470 |
| CCPL080626M1_1_14_4 | 0.773 | 13 | 14 | 0.932 | 8 | 0 | 3.454 | 0.477 |
| CCPL080626M1_1_14_5 | 0.777 | 14 | 32 | 0.889 | 9 | 2 | 3.113 | 0.494 |
| CCPL080626M1_1_14_6 | 0.653 | 16 | 40 | 0.863 | 10 | 0 | 3.222 | 0.448 |
| CCPL080626M1_1_14_7 | 0.658 | 14 | 66 | 0.938 | 8 | 1 | 3.336 | 0.588 |
| CCPL080626M2_1_12_1 | 0.776 | 16 | 180 | 0.691 | 12 | 0 | 3.555 | 0.382 |
