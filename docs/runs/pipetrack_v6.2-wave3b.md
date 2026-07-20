# Run archive: `pipetrack_v6.2-wave3b`

- **Purpose:** Wave-3b asymmetric-R refinement.
- **Verdict:** Accepted into v7 lineage
- **Full analysis:** methods_log W3b/W4
- Archived: 2026-07-14T02:30:49.004112+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-10T09:34:03.246831+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v6.2-wave3
- stages_run: ['p4']
- stabilization/lift: False/False
- config[p1b]: `configs/v6/p1b_stabilization.yaml` sha256 `a3f4512b30dcbbf1…`
- config[p2]: `configs/v6/p2_tracking.yaml` sha256 `5f46b1a2b464603e…`
- config[p3]: `configs/v6/p3_association.yaml` sha256 `6d6593b3653549df…`
- config[p4]: `configs/experiments/v6_wave3__p4.yaml` sha256 `736695d79efa1646…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': False, 'ema_alpha': 0.65, 'min_views': 2, 'reprojection_threshold_px': 10.0, 'smoother': 'ema'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.784 | 10 | 7 | 0.951 | 4 | 0 | - | - |
| CCPL080626M1_1_14_2 | 0.923 | 11 | 9 | 0.960 | 5 | 0 | - | - |
| CCPL080626M1_1_14_3 | 0.848 | 13 | 20 | 0.933 | 7 | 1 | - | - |
| CCPL080626M1_1_14_4 | 0.770 | 13 | 10 | 0.938 | 8 | 0 | - | - |
| CCPL080626M1_1_14_5 | 0.898 | 15 | 32 | 0.828 | 10 | 0 | - | - |
| CCPL080626M1_1_14_6 | 0.652 | 16 | 41 | 0.859 | 10 | 1 | - | - |
| CCPL080626M1_1_14_7 | 0.603 | 18 | 39 | 0.924 | 12 | 0 | - | - |
| CCPL080626M2_1_12_1 | 0.790 | 15 | 183 | 0.748 | 11 | 0 | - | - |
