# Run archive: `pipetrack_v6.0`

- **Purpose:** Frozen ground baseline of the fix campaign (v5 configs, RTMPose-X P1).
- **Verdict:** Baseline (superseded by v8.0)
- **Full analysis:** methods_log F0
- Archived: 2026-07-14T02:30:48.995706+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-10T08:04:54.298008+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v6.0
- stages_run: ['render']
- stabilization/lift: False/False
- config[p1b]: `configs/v6/p1b_stabilization.yaml` sha256 `a3f4512b30dcbbf1…`
- config[p2]: `configs/v6/p2_tracking.yaml` sha256 `5f46b1a2b464603e…`
- config[p3]: `configs/v6/p3_association.yaml` sha256 `6d6593b3653549df…`
- config[p4]: `configs/v6/p4_global_id.yaml` sha256 `ce283a214b491eaa…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': False, 'ema_alpha': 0.65, 'min_views': 2, 'reprojection_threshold_px': 10.0, 'smoother': 'ema'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.784 | 10 | 7 | 0.951 | 4 | 0 | 3.577 | 0.737 |
| CCPL080626M1_1_14_2 | 0.923 | 11 | 9 | 0.964 | 5 | 0 | 3.462 | 0.502 |
| CCPL080626M1_1_14_3 | 0.857 | 13 | 13 | 0.930 | 7 | 0 | 3.314 | 0.470 |
| CCPL080626M1_1_14_4 | 0.772 | 13 | 14 | 0.932 | 8 | 0 | 3.471 | 0.477 |
| CCPL080626M1_1_14_5 | 0.898 | 13 | 32 | 0.882 | 8 | 2 | 3.288 | 0.545 |
| CCPL080626M1_1_14_6 | 0.653 | 16 | 40 | 0.862 | 10 | 0 | 3.232 | 0.448 |
| CCPL080626M1_1_14_7 | 0.603 | 18 | 42 | 0.884 | 12 | 1 | 3.331 | 0.503 |
| CCPL080626M2_1_12_1 | 0.791 | 16 | 154 | 0.699 | 12 | 0 | 3.536 | 0.375 |
