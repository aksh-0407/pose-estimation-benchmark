# Run archive: `pipetrack_v6.3-wave4`

- **Purpose:** Wave-4 chimera splitting (F13).
- **Verdict:** Accepted into v7 lineage
- **Full analysis:** methods_log W3b/W4
- Archived: 2026-07-14T02:30:49.005622+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-10T09:52:04.981133+00:00
- base_tree: /home/aksh/quidich/Pose_Estimation/benchmarks/runs/pipetrack_v6.3-wave4
- stages_run: ['p3_5', 'p4']
- stabilization/lift: False/True
- config[p1b]: `configs/v6/p1b_stabilization.yaml` sha256 `a3f4512b30dcbbf1…`
- config[p2]: `configs/v6/p2_tracking.yaml` sha256 `5f46b1a2b464603e…`
- config[p3]: `configs/v6/p3_association.yaml` sha256 `6d6593b3653549df…`
- config[p4]: `configs/experiments/v6_wave3__p4.yaml` sha256 `736695d79efa1646…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': False, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': False, 'reprojection_threshold_px': 10.0, 'smoother': 'ema'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_1 | 0.671 | 14 | 9 | 0.899 | 8 | 0 | - | - |
| CCPL080626M1_1_14_2 | 0.952 | 13 | 7 | 0.968 | 7 | 0 | - | - |
| CCPL080626M1_1_14_3 | 0.777 | 15 | 33 | 0.938 | 9 | 1 | - | - |
| CCPL080626M1_1_14_4 | 0.689 | 15 | 14 | 0.947 | 10 | 0 | - | - |
| CCPL080626M1_1_14_5 | 0.695 | 17 | 23 | 0.872 | 12 | 0 | - | - |
| CCPL080626M1_1_14_6 | 0.678 | 18 | 42 | 0.832 | 12 | 1 | - | - |
| CCPL080626M1_1_14_7 | 0.589 | 20 | 39 | 0.904 | 14 | 0 | - | - |
| CCPL080626M2_1_12_1 | 0.758 | 16 | 154 | 0.773 | 12 | 0 | - | - |
