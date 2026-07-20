# Run archive: `pipetrack_v8-probe`

- **Purpose:** Phase C: tiled NMS-0.3 P1 through v7 stack (_7+M2).
- **Verdict:** Hold verdict; superseded by nms55
- **Full analysis:** methods_log W5-C
- Archived: 2026-07-14T02:30:49.017063+00:00 (data tree deleted after archival)

## Manifest / provenance
- created_at: 2026-07-13T11:05:40.667504+00:00
- base_tree: None
- stages_run: ['p1b', 'p2', 'p3', 'p3_5', 'p4', 'p5', 'p6_3d']
- stabilization/lift: True/True
- config[p1b]: `configs/v7/p1b_stabilization.yaml` sha256 `d43a68a2c092eec3…`
- config[p2]: `configs/v7/p2_tracking.yaml` sha256 `d82a07f1c4b28fca…`
- config[p3]: `configs/v7/p3_association.yaml` sha256 `ce303573b7e76d1d…`
- config[p4]: `configs/v7/p4_global_id.yaml` sha256 `b20553ad3f385341…`
- triangulation: {'butter_cutoff_hz': 6.0, 'cheirality': True, 'dense_fill': True, 'ema_alpha': 0.65, 'min_views': 2, 'native_skeleton': True, 'reprojection_threshold_px': 10.0, 'smoother': 'butterworth'}

## Per-delivery metric panel
| delivery | agreement | ids | teleports | id_persist | frags | stitch_links | tri_reproj_px | tri_cov |
|---|---|---|---|---|---|---|---|---|
| CCPL080626M1_1_14_7 | 0.670 | 13 | 51 | 0.858 | 6 | 8 | 3.153 | 0.541 |
| CCPL080626M2_1_12_1 | 0.784 | 12 | 228 | 0.886 | 7 | 10 | 3.529 | 0.421 |
