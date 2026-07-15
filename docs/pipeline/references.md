# References

Repo anchors (code the analysis cites) and external sources (verifiable — arXiv IDs / venues).
No claim in these docs rests on an unverified source; quantitative claims trace to the repo's own
logs.

## Repo evidence anchors

**Measured results / issue logs (the source of every metric quoted):**
- `../diagnosis/09-per-phase-issue-register.md` — identity failure modes ID-1…ID-6 with per-delivery evidence.
- `../diagnosis/README.md`, `../diagnosis/README.md` — 3D-location issues (12 → resolved/open).
- `../../wip/methods_log.md` — accept/reject lab notebook (M0…M11; z0_reproj win, calibration audit).
- `../diagnosis/README.md` — SOTA-mapped redesign + the empirical A/B pivot.
- `implementation_plan.md`, `wip/to_do.md`, `CHANGELOG.md` — plan, deferred work, round-by-round history.
- Committed metrics: `data/derived/<dataset>/pipetrack_v<num>/**/{association_metrics,global_id_metrics,stabilization_metrics}.json`.

**Key code locations:**
- P1: `src/core/inference/run_phase1_rtmpose_inference.py` (`boxes_from_det_result:485`, `inference_topdown_batch:528`, `player_records:619`).
- 01 (stabilization): `src/identity/p1_stabilization/{smoothing,linker,runner}.py`.
- 02: `src/identity/p2_tracking/{tracker.py:180, kalman.py:16}`.
- 03: `src/identity/p3_association/{tracklet_graph.py, geometry_cache.py:74, cue_calibration.py}`; `src/identity/common/geometry.py:540` (`ground_from_reprojection`), `pose_shape.py`.
- 3D lift: `src/identity/common/triangulation.py` (`triangulate_point_dlt:31`, `triangulate_skeleton_ransac:162`); `src/identity/p4_lift/run_triangulation.py`.
- 05: `src/identity/p5_global_id/{track_manager.py:322, stitching.py}`; `src/identity/p5_global_id/ground_kalman.py:39` (Singer/Van Loan).
- Render: `src/identity/visualization/{render_videos.py, mosaic_layout.py, identity_colors.py}`.

## External sources

**Detection / 2D pose**
- RTMPose — Jiang et al. 2023, [arXiv 2303.07399](https://arxiv.org/abs/2303.07399).
- RTMO (one-stage) — Lu et al., CVPR 2024, [arXiv 2312.07526](https://arxiv.org/abs/2312.07526).
- RTMDet — Lyu et al. 2022, [arXiv 2212.07784](https://arxiv.org/abs/2212.07784).
- RT-DETR — Zhao et al., CVPR 2024, [arXiv 2304.08069](https://arxiv.org/abs/2304.08069).
- Co-DETR (collaborative hybrid assignments) — Zong et al., ICCV 2023, [arXiv 2211.12860](https://arxiv.org/abs/2211.12860).
- SmoothNet (temporal pose refinement) — Zeng et al., ECCV 2022, [arXiv 2112.13715](https://arxiv.org/abs/2112.13715).
- One-Euro filter — Casiez, Roussel & Vogel, CHI 2012, [gery.casiez.net/1euro](https://gery.casiez.net/1euro/).

**Multi-object tracking (per-camera / motion)**
- ByteTrack — Zhang et al., ECCV 2022, [arXiv 2110.06864](https://arxiv.org/abs/2110.06864).
- BoT-SORT — Aharon et al. 2022, [arXiv 2206.14651](https://arxiv.org/abs/2206.14651).
- OC-SORT — Cao et al., CVPR 2023, [arXiv 2203.14360](https://arxiv.org/abs/2203.14360).
- Deep OC-SORT — Maggiolino et al., ICIP 2023, [arXiv 2302.11813](https://arxiv.org/abs/2302.11813).
- UCMCTrack (uniform camera-motion compensation) — Yi et al., AAAI 2024, [arXiv 2312.08952](https://arxiv.org/abs/2312.08952).
- Global data association / min-cost flow — Zhang, Li & Nevatia, CVPR 2008 (network-flow MOT).
- Correlation clustering / multicut for MOT — Tang et al., CVPR 2017 (multiple-object tracking as a graph multicut).

**Sports MOT / cross-camera / kit-robust ReID**
- Deep-EIoU (expansion-IoU + deep features, sports) — Huang et al. 2023, [arXiv 2306.13074](https://arxiv.org/abs/2306.13074).
- GTA: Global Tracklet Association in sports — 2024, [arXiv 2411.08216](https://arxiv.org/abs/2411.08216).
- Self-supervised multi-view multi-human association & tracking — 2024, [arXiv 2401.17617](https://arxiv.org/abs/2401.17617).
- A Unified Multi-view Multi-person Tracking Framework — 2023, [arXiv 2302.03820](https://arxiv.org/abs/2302.03820).
- Robust Online Multi-Camera People Tracking (geometric consistency + state-aware ReID) — CVPRW 2024.
- Multi-task ReID / team affiliation / role for sports tracking — 2024, [arXiv 2401.09942](https://arxiv.org/abs/2401.09942).
- SoccerNet Game State Reconstruction (tracking + jersey + minimap) — 2024, [arXiv 2404.11335](https://arxiv.org/abs/2404.11335).
- SoccerNet Re-Identification challenge/dataset — [github.com/SoccerNet/sn-reid](https://github.com/SoccerNet/sn-reid).

**Multi-view triangulation / 3D**
- Learnable Triangulation (confidence-weighted DLT) — Iskakov et al., ICCV 2019, [arXiv 1905.05754](https://arxiv.org/abs/1905.05754).
- Robust Uncertainty-Aware Multiview Triangulation — Lee & Civera 2020, [arXiv 2008.01258](https://arxiv.org/abs/2008.01258).
- Pose2Sim (calibrated markerless sports capture) — [PMC8512754](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8512754/), [PMC9002957](https://pmc.ncbi.nlm.nih.gov/articles/PMC9002957/).
- Faster VoxelPose (operate in 3D) — Ye et al., ECCV 2022, [arXiv 2207.10955](https://arxiv.org/abs/2207.10955).
- LOSTU (uncertainty-aware triangulation) — 2023, [arXiv 2311.11171](https://arxiv.org/abs/2311.11171).
- UPose3D (uncertainty-aware, multi-view + temporal) — 2024, [arXiv 2404.14634](https://arxiv.org/abs/2404.14634).

**Evaluation**
- CLEAR-MOT (MOTA) — Bernardin & Stiefelhagen 2008; IDF1 — Ristani et al., ECCVW 2016.
- HOTA — Luiten et al., IJCV 2021, [arXiv 2009.07736](https://arxiv.org/abs/2009.07736).

> Sources surfaced/confirmed via literature search (July 2026). arXiv IDs are authoritative; where a
> venue is named without an arXiv ID, the paper is a workshop/journal entry located by title.
