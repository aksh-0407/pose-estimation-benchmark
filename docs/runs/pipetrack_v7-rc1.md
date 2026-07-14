# Run archive: `pipetrack_v7-rc1`

- **Purpose:** First composed v7 release candidate.
- **Verdict:** Rejected (H3 binding collapse; root-caused)
- **Full analysis:** fixes-log v7-rc1
- Archived: 2026-07-14T02:30:49.007509+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-10T11:06:19.742942+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v7-rc1
- stages_run: ['render']
- stabilization/lift: False/False
- config[p1b]: `configs/v6/p1b_stabilization.yaml` sha256 `a3f4512b30dcbbf1…`
- config[p2]: `configs/v6/p2_tracking.yaml` sha256 `5f46b1a2b464603e…`
- config[p3]: `configs/v6/p3_association.yaml` sha256 `6d6593b3653549df…`
- config[p4]: `configs/v6/p4_global_id.yaml` sha256 `ce283a214b491eaa…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': False, 'dense_fill': False, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': False, 'reprojection_threshold_px': 10.0, 'smoother': 'ema'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.779 | 14 | 5 | 0.942 | 9 | 0 | 3.096 | 0.332 |
| CCPL080626M1_1_14_2 | 0.951 | 13 | 7 | 0.972 | 7 | 0 | 3.345 | 0.348 |
| CCPL080626M1_1_14_3 | 0.741 | 15 | 13 | 0.961 | 9 | 0 | 3.102 | 0.315 |
| CCPL080626M1_1_14_4 | 0.770 | 15 | 10 | 0.965 | 10 | 0 | 3.266 | 0.315 |
| CCPL080626M1_1_14_5 | 0.671 | 15 | 26 | 0.929 | 10 | 0 | 2.860 | 0.326 |
| CCPL080626M1_1_14_6 | 0.567 | 19 | 43 | 0.853 | 13 | 0 | 3.121 | 0.319 |
| CCPL080626M1_1_14_7 | 0.713 | 13 | 32 | 0.935 | 7 | 0 | 3.068 | 0.582 |
| CCPL080626M2_1_12_1 | 0.782 | 16 | 177 | 0.776 | 12 | 0 | 3.205 | 0.246 |
