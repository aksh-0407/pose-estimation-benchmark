# Legacy and dead code register

Code paths that are present in the tree but unreachable or unread under the
production configuration. Per the 2026-07-17 audit decision they are KEPT (git
history alone was judged insufficient for rediscovery) and documented here so
nobody mistakes them for live behavior. Production truth is `configs/*.yaml`
plus each run's `run_manifest.json`, never dataclass defaults.

Status meanings: `dead` = no code path can reach it under any config;
`inert` = computed but its result is never read; `off-by-config` = reachable
only by flipping a config value away from production; `deprecated` = kept for
interface compatibility only.

## Cross-camera association (src/identity/p3_association/)

| Item | Where | Status | Notes |
|---|---|---|---|
| Per-frame association engine | `associator.py`: `associate_frame`, `_associate_multiway_cycle`, `_associate_pairwise_anchor`, `_constrained_cluster`, `build_cost_matrix`, `solve_optional_assignment`, `TemporalLinkMemory` | off-by-config | Production sets `association_mode: tracklet_graph`; the per-frame engine (about 500 lines) runs only with `association_mode: per_frame`. Kept as the historical A/B baseline. |
| `matching_mode: pairwise_anchor` | `associator.py` | off-by-config | A sub-branch of the per-frame engine above. |
| `temporal_link_decay` | `config.py`, per-frame engine | off-by-config | Only meaningful in per-frame mode. |
| Fine-score calibration subsystem | `config.py` (`mu_fine_score`, `sigma_fine_score`, `w_epi`, `w_tri`), `cue_calibration.py` (`CalibrationStats`), `geometry_cache.py` (`GeometryCache.stats`, `.huber_delta`, `PairGeometry.w_epi/w_tri/huber_delta`), `config.huber_delta()` | inert | Computed and threaded through, but `build_cost_matrix` recomputes weights from `pg.is_degenerate` and never reads them. A whole legacy scoring scheme left wired but unread. |
| `cycle_xy_tol_m`, `dummy_cost_scale`, `parallax_min_deg`, `parallax_full_deg` | `config.py` | dead | No code reads these fields anywhere in `src/`. |

## Shared geometry (src/identity/common/geometry.py)

| Item | Status | Notes |
|---|---|---|
| `condition_number_dlt` | dead | No callers. |
| `ground_point_and_cov` | dead | No callers. |
| `huber_cost` | dead | No callers. |
| `parallax_weight` | dead | Imported by `associator.py` but never called. |
| `fuse_ground_estimates` | internal-only | Called only by `robust_fuse_ground`. |
| Foot-contact modes `legacy` / `v2` / `v3` in `ground_contact_pixel_ex` | dead in production | The 17-keypoint shape guard makes every production (Halpe-26) call return the bbox bottom, so none of the three modes executes. This is the audit's headline finding (bugs.md BUG-A1, report-only by owner decision); `foot_contact_mode: v3` in `configs/03_association.yaml` is inert until the guard is fixed. |

## Phase 1 inference (src/core/)

| Item | Status | Notes |
|---|---|---|
| `select_coco17_pose`, `coco17_indices` threading | inert | `player_records` receives `coco17_indices` but always emits the model's native keypoints unsliced. Kept as the documented COCO-17 slicing reference. A non-Halpe pose model would emit non-26 keypoints and fail contract validation, so the P1 runners are Halpe-26-only in practice. |
| `core/schemas.py` (`CameraCalibration`, `PosePacket`) | dead | No importers found in `src/`, `tools/`, or `tests/`. Appears to be an aspirational export-packet schema for the graphics layer. |
| `detect_person_boxes` (single-image path) | near-dead | Only used as the fallback inside `detect_person_boxes_batch` when a detector build rejects list input. |

## 3D lift (src/identity/p4_lift/)

| Item | Status | Notes |
|---|---|---|
| `--native-skeleton` flag | deprecated | Explicit no-op; the lift always triangulates the full native skeleton. Still threaded into the run manifest for compatibility. |
| `triangulate_legacy` flat-JSONL entry point | dead | Parallel unused entry point from the pre-restructure layout. |
| `depth_signs`, `irls_huber_refit` (common/triangulation.py) | off-by-config | Reachable via `--cheirality` (production on) and `--robust-refit` (production off). |

## Global identity (src/identity/p5_global_id/)

| Item | Status | Notes |
|---|---|---|
| `P4Config`, `P4AConfig`, `P4BConfig` aliases | deprecated | Compatibility aliases for the stage-05 names (`GlobalIdConfig`, `GlobalTrackingConfig`, `StitchingConfig`). YAML loader accepts the old `p4a:`/`p4b:` section spellings. |
| Legacy verdict rule (`usability_verdict: false` path) | off-by-config | The teleport-proxy grading superseded by the usability rubric. |

## Roles (src/identity/p6_roles/)

| Item | Status | Notes |
|---|---|---|
| v0 role solver (`assign_roles`) | off-by-config | Production pins `role_assignment_version: v1` (epoch solver). v0 kept for reproducibility of old runs. |

## Visualization (src/identity/visualization/)

| Item | Status | Notes |
|---|---|---|
| `ROLE_TAGS` (overlays.py) | inert | Roles are shown only in the roster panel by design; kept as the single place to change on-screen role wording. |
| `draw_info_panel` (panels.py) | dead | The bird's-eye tile replaced the text monitor tile; kept as the generic panel primitive. |
| `seen_here_nearby` inner check (render_videos.py mosaic loop) | inert | Deliberate no-op retained with its intent comment: lost ghosts are shown regardless, within the decay window. |
| Two bird's-eye renderers | duplication by design | The cv2 tile (panels.draw_bev_panel, in the mosaic) and the matplotlib tool (render_bird_eye_view.py, standalone diagnostics) serve different outputs; only exact-duplicate helpers were merged. |

## Batch drivers

| Item | Status | Notes |
|---|---|---|
| `id_pipeline.py` | partially superseded | Drives association through global-id only (03 to 05); `src/main.py` is the full-chain driver. Kept for focused association/identity experiment loops. |

Removal of anything in this table is a deliberate future decision, to be taken
item by item with the owner; see docs/audit/changes.md for the campaign record.
