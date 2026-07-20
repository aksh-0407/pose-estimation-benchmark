# methods_inventory.md — what runs, what's flagged on/off, and why (handover audit, 2026-07-20)

Whole-codebase pass over every `src/core` and `src/identity` script. This is the map a new owner
needs: the methods that actually produce the final output, the methods that exist but are switched
**off**, and the reasoning behind each on/off state.

**Source of truth for flag state = `configs/*.yaml` + `run_manifest.json`, NOT the code defaults.**
Several code defaults disagree with production (see `resolvebugs.md` → "Default-vs-production mismatches").
Bugs referenced below (BUG-N) are detailed in `resolvebugs.md`. Legacy/off modules are inventoried in
`legacy_code.md`.

---

## 0. Pipeline shape (from `src/main.py`)

```
00_inference → 01_stabilization → 02_tracking → 03_association → 04_lift → 05_global_id → 06_roles(+suppress) → 07_refine → 08_render
```

Key structural fact: **the 3D lift (04) runs BEFORE global-id (05)** — order is Associate → Triangulate → Track.
Global-id and roles carry `pose_3d` forward; `06_roles` emits the terminal role-stamped,
suppression-filtered predictions consumed downstream; `07_refine` rewrites only `pose_3d`/`pose_3d_named`
(IDs byte-identical). P1 (00_inference) is NOT run by `main.py` — it is produced out-of-band by the
`core/inference` runners and consumed as pre-written `00_inference/` output.

Driver-level enable flags (`main.py`, all default **ON** except robust-refit):

| Flag | Default | Meaning |
|---|---|---|
| `--enable-stabilization` | ON | run stage 01 before P2 (v7 default) |
| `--enable-lift` | ON | run stage 04 binding-keyed 3D lift |
| `--enable-refine` | ON | run stage 07 physics refinement |
| `--tri-cheirality` | ON | in-front-of-camera gate in the lift (fix F3) |
| `--tri-smoother` | butterworth | zero-phase 3D smoother (EMA is the legacy off option) |
| `--tri-native-skeleton` | ON | triangulate all 26 Halpe kps — **documented no-op**, always 26 (BUG: dead flag) |
| `--tri-dense-fill` | ON | gap-gate temporal fills on REAL frame numbers (C6) |
| `--tri-robust-refit` | **OFF** | IRLS-Huber per-joint polish; off = byte-identical baseline |

---

## 1. `src/core/` — shared primitives

| File | Methods in use | Notes |
|---|---|---|
| `calibration.py` | Ground-plane homography from the 3×4 projection matrix (columns `[0,1,3]` for z=0), inverted for image↔ground; pixel→ground homogeneous divide with `|w|<1e-12` guard; bbox-bottom grounding; borrow-aware calibration routing (40_full borrows 8_init). | Geometry verified correct. |
| `contract.py` | Schema validator for `g1_player_frame/v1`, skeleton `halpe26`, **26** keypoints; per-frame global-id uniqueness; `final_handoff` gate. `pose_3d`/`pose_3d_named` may be null (un-triangulated). | The 26-keypoint contract is what BUG-1's `(17,2)` guard violates. |
| `datasets.py` | Path registry over one `DATA_ROOT` (`--data-root` > `$PIPETRACK_DATA` > `data`); `calibration_source` indirection; camera-group topology `bt_01:[01,04] bt_02:[02,05,07] bt_03:[03,06]`; frame-count / sync validation. | |
| `frames.py` | Frame/prediction filename parsing (`frame_cameraNN_<id>.jpg`, `<group>__<delivery>__<cam_NN>.jsonl`); integer frame sort. | |
| `keypoints.py` | Halpe-26 skeleton: COCO-17 in 0–16 + head/neck/hip + 6 feet; **root = mid-hip idx 19**; `HALPE26_BONES` (BFS kinematic tree for single-sweep FK), `HALPE26_BONE_LIMITS_M` (hard limb clamps), `HALPE26_HINGES` (knee/elbow); `map_keypoints` slices to COCO-17. | Bone ordering valid (parent-before-child). |
| `schemas.py` | Dataclasses only (`CameraCalibration`, `PosePacket`). No algorithm. | |
| `ue_transform.py` | World→UE: cricket (X=right,Y=fwd,Z=up) → UE (X=fwd,Y=right,Z=up) via X/Y swap + m→cm ×100; non-finite→null. | Verified correct. |

## 1b. `src/core/inference/` — P1 2D pose (out-of-band, upstream of identity)

Active production stack (per README + runbook): **RTMDet-m person detector on a 4×2 overlap-0.25 tile
grid + full-frame pass, cross-tile NMS 0.55 + IoM-0.7 containment suppression; RTMPose-X Halpe-26 pose.**

| Method | State | Reasoning |
|---|---|---|
| Top-down RTMDet→RTMPose, batched detect + flattened pose crop stream | ACTIVE | shared `phase1_common.py` core for both runners |
| Tiled (SAHI-style) detection `--tiled-det` | ON in production (code default OFF) | small/distant-player recall (v8 stack) |
| Tiled-fast bypass `--tiled-fast` | ON | "parity-checked against the generic path" |
| fp16 AMP `--amp`, cudnn.benchmark+TF32 `--perf` | ON | verified box/keypoint parity within tolerance on L40S |
| NMS threshold `--nms-thr` | 0.55 in production (code default 0.3) | crossing-survival (v8 discovery); code default is stale |
| Decode prefetch overlap + cv2/torch thread caps | ON | fixed the GPU-starvation (cold-disk I/O + thread oversubscription) |
| Data-parallel sharding (`run_phase1_parallel.py`) | operational utility | round-robins deliveries across GPU-sharing subprocesses; weakest-verified of the three runners (BUG-list: confirm still used) |
| Pose model swap | fixed to RTMPose (mandate); only detector swaps | RTMPose-X is the only x body checkpoint (Halpe-26 superset of COCO-17) |

Runners: `run_phase1_rtmpose_inference.py` (repo layout, ACTIVE) · `run_phase1_l40s.py` (L40S native
`bt1/bt2/bt3` layout, ACTIVE, all GPU runs go here) · `run_phase1_parallel.py` (sharding wrapper).
See BUG-6/7/8 for the sweep, resume-corruption, and skeleton-resolution issues.

---

## 2. Stage 01 — stabilization (`p1_stabilization/`)  ·  ENABLED

**Purpose:** reduce per-keypoint pixel jitter between P1 and P2 so it doesn't propagate into
tracking/association/triangulation. `enabled: false` → byte-identical pass-through (for A/B).

| Method | State | Reasoning |
|---|---|---|
| IoU micro-track linking (`linker.py`) | ON | short-range smoothing only — explicitly **NOT** identity; a mislink only co-smooths one frame |
| One-Euro filter per keypoint coord (`smoothing.py`) | ON, `method: one_euro` (only supported) | speed-adaptive low-pass (Casiez 2012); smooths hard when still, responsive when fast |
| Confidence-gated spike clamp | ON | engages **only** below `confidence_min=0.30`; replaces a low-conf jump (> max of `120px` / `0.5·bbox-diag`) with the last filtered position |
| `smooth_native` | inert no-op | declared/parsed but never read; stale comment references a non-existent `pose_2d_native` block (LOW slop) |

Config: `min_cutoff 1.7`, `beta 0.30`, `d_cutoff 1.0`, `iou_min 0.3`, `max_gap_frames 2`.

---

## 3. Stage 02 — per-camera tracking (`p2_tracking/`)  ·  tracker = bytetrack

**Purpose:** per-camera multi-object tracking, assigning `local_track_id`.

| Method | State | Reasoning |
|---|---|---|
| ByteTrack two-pass association (`tracker.py`) | ACTIVE | Stage-1 high-conf (IoU+pose cost), Stage-2 low-conf recovery (IoU only); Hungarian |
| Constant-velocity 8-state Kalman (`kalman.py`) | ACTIVE | Joseph-form update; process-noise inflation while dormant |
| Pose-cosine dormant re-ID (`pose_vector.py`) | ACTIVE | masked confidence-weighted cosine on root-relative scaled kps; `reid_threshold 0.25`, ambiguity margin `0.05`, `dormant_max_frames 60` |
| Scale-adaptive motion gate (Mahalanobis-or-distance) | ACTIVE | `chi2_gate 9.21`, `v_max 120 px/frame` |
| Calibration-assisted ground-plane gating | ACTIVE (partly hobbled) | `ground_cost_weight 0.15`, reachability radius from `ground_vmax 9 m/s`; **but** BUG-1 makes ground = bbox-bottom, so `ankle_confidence_min`/`max_ankle_above_bbox_fraction` are inert |
| Medoid pose gallery (incremental) | ACTIVE | O(K) cache, claimed bit-identical to O(K²) |
| `lowconf_can_spawn` | **OFF** (yaml false; dataclass True) | ByteTrack principle: low scores only *recover*, never *spawn* — suppresses spurious fragments (v8) |
| **OC-SORT path** (OCM cost / ORU re-update / OCR recovery) | **OFF** | `tracker: bytetrack`; all OC-SORT code guarded so bytetrack is byte-identical. Kept as gated A/B alt (see `legacy_code.md` B-2) |

---

## 4. Stage 03 — cross-camera association (`p3_association/`)  ·  association_mode = tracklet_graph

**The most complex stage.** Production decides who-is-who once per P2-tracklet pair over the whole
delivery (evidence aggregated across all co-visible frames, cues fused as calibrated log-likelihood
ratios), then emits per-frame correspondences from the stable bindings.

### ACTIVE methods
| Method | Reasoning / config |
|---|---|
| Tracklet-graph identity solve (`tracklet_graph.py`) | 3-pass: observe → harvest_calibration → solve (constrained union-find agglomeration + refine + singleton rescue + fragment attach + binding mint) |
| LLR cue fusion | per-pair `total = support · (llr_ground + llr_app + llr_posture + llr_motion)`; asymmetric clipping |
| z0_reproj ground solver | emit-only; joint z=0-constrained Huber+inlier-refit reprojection min; well-posed on low-parallax facing pairs. **Degraded by BUG-1** (plane heights always 0 → back-projects onto z=0) |
| Cluster lift + purity / chimera signature (`cluster_lift.py`) | feeds union-lift, shape round, split |
| Pose-shape descriptor cue | `pose_descriptor_enabled: true`; soft down-weight for implausible torso |
| Ground-anchored posture cue | `posture_enabled: true`; metric, per-camera, works on facing pairs |
| Cue auto-calibration (`cue_calibration.py`) | `calibration_mode: auto`; bootstraps same/diff anchors from this delivery |
| Union-lift co-location merge | `graph_union_lift_merge: true`; kills facing-pair split identities (ghost-under-player) |
| Corroboration merge | `graph_corrob_merge: true`; single-cue facing edge in [1.2, 2.0) only if fully supported + unambiguous |
| Shape corroboration round | `graph_shape_enabled: true`; self-calibrated bone-ratio |
| Chimera split/veto | `graph_split_enabled: true` at CONSERVATIVE thresholds (W4 over-split lesson): frame-fraction 0.6, torso-residual 30px |
| Synthetic tracklets | `synthetic_tracklets_enabled: true`; binds persistent untracked detections (umpires) |
| Approx-feet recovery | `approx_feet_enabled: true`; upper-body height-plane ray when feet unusable. **Does NOT go through the broken guard — works** |
| Purity split | `purity_split_enabled: true`; kinematic-jump chunking |
| Facing-pair gate widening | `graph_facing_gate_scale 1.3`; facing pairs (01-04, 02-06, 03-05) derived from calibration optical axes |
| emit_posture / emit_ground_cov | F6b / F9a, both true |

### The headline toggle: `graph_llr_positive_cap 3.5` (raised from 1.5)
The dataclass default 1.5 (< the 2.0 merge threshold) was a deliberate structural bound: one capped
ground cue alone can't clear the threshold, so a merge *needs* corroboration. Production raises it to
**3.5**, abandoning that bound, because on facing pairs appearance/motion/posture structurally abstain
and ground is the only strong cue — the old cap throttled genuinely-tight ground agreement (0.3–0.5 m)
and under-merged 16% of ≥3-camera locations. 3.5 is the measured agreement peak on 8_init
(0.880→0.916, under-merge 16%→7%, coloc 2→0, collisions 0); above 3.75 agreement turns over (false
ground-alone merges), so the cap clips there. Safety now rests on support-weighting, the
`graph_llr_veto -4.5` confident-contradiction block, and the `_cluster_compatible` same-camera-frame
cannot-link — **not** on the cap/threshold gap. Pending 40-delivery confirm.

### OFF / inert in production
- **Per-frame associator** (`associate_frame`, `multiway_cycle`/`pairwise_anchor` cost matrices, `TemporalLinkMemory`, epipolar/Sampson scoring) — graph mode never calls it; `matching_mode: multiway_cycle` is inert. File stays live (shared dataclasses/geometry). See `legacy_code.md` B-3.
- **Contested-camera down-weighting** — `contested_iou 0.0`, not in yaml → all `contested_*` branches dead.
- **`airborne_pelvis_emit`** (false) · **`foot_smooth_window 1`** (no-op) · **calibration_fallback_path** (empty).
- **`foot_contact_mode v3` + `single_cam_height_emit true`** — enabled in yaml but **INERT via BUG-1** (height always 0).

---

## 5. Stage 04 — 3D triangulation lift (`p4_lift/run_triangulation.py`)  ·  ENABLED, id-source = binding

| Method | State | Reasoning |
|---|---|---|
| Group observations by P3 `binding_id` | ACTIVE (`--id-source binding` forced by main.py) | F9b: lift before global-id so purity/descriptor evidence feeds identity; legacy `global_player_id` grouping is off |
| RANSAC skeleton triangulation, all 26 joints | ACTIVE | batched kernels, claimed bit-identical to reference loop |
| Cheirality gate | ACTIVE (`--cheirality`) | reject points behind a camera (F3) |
| Per-joint measurement covariance | ACTIVE (binding mode) | emits `cov_diag_m2`/`pelvis_cov_m2` → stage-05 R |
| Occlusion temporal fill + skeletal-prior fill | ACTIVE | linear interp / hold within `max_gap 25`, then median-bone-length last resort |
| Dense-fill on real frame numbers | ACTIVE (`--dense-fill`) | a 2-row gap spanning 300 real frames is not interpolated as adjacent (C6) |
| Butterworth zero-phase smoothing (order-4, 6 Hz) | ACTIVE (`--smoother butterworth`) | export quality (F7) |
| Per-joint nullable emit gated on mid-hip finite | ACTIVE | frame ships only if hips finite; other joints emit null + sentinel 100.0 |
| Chimera-purity diagnostics | ACTIVE (binding) | `lift_purity.json` |
| **Robust/Huber IRLS refit** | **OFF** (`--robust-refit` false) | off = byte-identical; not adopted |
| **EMA smoother** | **OFF** | legacy causal path; `--ema-alpha 0.65` threaded but inert |
| `--native-skeleton`, `--hartley`, `--parallax-order`, `triangulate_legacy`, `--id-source global` | dead/no-op | see `legacy_code.md` B-5/B-6, BUG-11 |

---

## 6. Stage 05 — global identity (`p5_global_id/`)  ·  stitching enabled

| Method | State | Reasoning |
|---|---|---|
| Singer-acceleration ground Kalman | ACTIVE | state `[x,y,vx,vy,ax,ay]`, Van Loan `Q_d`, Joseph update; per-role `alpha/sigma_a/measurement_noise` |
| Chi²-gated Mahalanobis Hungarian assignment | ACTIVE | gate 5.991 |
| Four-stage temporal association | ACTIVE | binding continuity → exact tracklet continuity (revocable ownership TTL 50) → geometric Hungarian → shadow-absorb → re-entry-or-birth → ground-less bridging |
| Measurement-covariance R (asymmetric) | ACTIVE (`use_measurement_covariance: true`) | P3 ground cov eigen-clamped to [0.15², 0.8²] m², applied to **update only**; gating stays on role R (anti-teleport; `use_measurement_covariance_for_gating` off — symmetric cost +37 teleports on M2) |
| Kalman posterior emit | ACTIVE (`emit_kalman_posterior: true`) | emitted ground = filtered posterior (dodges BUG-4's outlier-emit) |
| Online role proxy (`role_proxy.py`) | ACTIVE (`online_role_proxy: true`) | causal bowler/umpire/keeper classification; latches role switches into the Singer model; **required by v1 roles** |
| Adaptive lost window | ACTIVE (`adaptive_lost_window: true`) | grows with hits up to 90; density-half variant off |
| Min-cost-flow stitching (normalized costs) | ACTIVE (`stitching.enabled: true`) | same-camera-frame occupancy veto in `remap_ids` |
| Occupancy-licensed long bridge | ACTIVE (`occupancy_bridge: true`) | gate extended to 300 frames, `require_pose: true` |
| Colocated-id merge | ACTIVE (`colocated_merge: true`) | W9 ghost-under-player fix; radius 0.75 m, min 25 frames |
| Pose-shape + billboard-posture gating | ACTIVE | `pose_gate_veto 0.3`, `posture_gate_veto_z 3.0`, `w_pose 2.0`, `w_posture 0.5` |
| Cardinality drop | ACTIVE (`min_emit_frames 30`) | short ids dropped post-stitch |
| Roster-cap + shadow-confirm gating | ACTIVE | `expected_roster_max 15`, `roster_cap_min_separation 3.0`, `shadow_confirm_override_hits 30` |
| Verdict hard gates | ACTIVE | fail on collisions>0, distinct_ids>20, or agreement<0.65 |
| OFF/dormant | — | `emit_ground_source` hip machinery, `single_view_hip_fallback`, `emit_velocity_gate`, `drop_partial_singlecam`, `presmooth_ground_enabled`, `density_lost_window`, legacy teleport-proxy, non-posterior emit branch (BUG-4) |

---

## 7. Stage 06 — roles + suppression (`p6_roles/`)  ·  version = v1

| Method | State | Reasoning |
|---|---|---|
| Epoch-scored Hungarian role solver (`assign_roles_epoched`) | ACTIVE (v1) | per-epoch (40 frames) Hungarian over 6 roster slots; gated by `role_assignment_max_cost 8.0` |
| Latch debounce + final uniqueness | ACTIVE | (pid,slot) latches only after `latch_count 3` consecutive epochs; deterministic greedy final resolution |
| Bowling-end sign resolution (v1.2 auto-flip) | ACTIVE | solves both ±axis on the pre-shot window, keeps cheaper sign; detected run-up wins outright |
| No-axis fallback | ACTIVE (degraded) | fastest net-displacement id → bowler, rest fielders |
| Peripheral suppression (`suppress_peripherals.py`) | ACTIVE (`suppression_enabled: true`) | core roles never dropped; peripherals dropped on kp-conf/completeness/single-cam-det-conf; writes terminal role-stamped predictions that 07 reads. `suppress_protect_umpires: false` (umpires suppressible) |
| **v0 legacy heuristic (`assign_roles`)** | **OFF** | reachable only via `role_assignment_version: v0` / omit `--config`; `legacy_code.md` B-4 |
| **`realtime_bowler_tracker.py`** | **DEAD + wrong** | not imported; BUG-2 → SCRAP |

Requires input P4 `online_role_proxy: true` (hard-checked, `run_role_assignment.py:82`).

---

## 8. Stage 07 — refinement (`p7_refine/`)  ·  ENABLED, offline whole-clip

**Rewrites only `pose_3d`/`pose_3d_named` — never any identity field, so IDs are byte-identical.**
Runs after identity is frozen (06_roles). Non-causal (whole-clip) smoothing.

| Method | State | Reasoning |
|---|---|---|
| Visibility-aware re-lift (`relift.py`) | ACTIVE (`relift: true`) | per-joint reliable-view (conf≥`vis_conf 0.5`): ≥2 → weighted DLT, exactly 1 → single-view bone-length ray placement, 0 → NaN for fill. Fixes depth-stretched legs (umpire in one camera) |
| Low-conf predict-and-substitute | ACTIVE | `conf<conf_floor 0.5` → NaN → temporal fill (`max_gap 25`) → skeletal-prior fill |
| Canonical bone-length clamp | ACTIVE | per-player median lengths, L/R pooled, clamped to `HALPE26_BONE_LIMITS_M` |
| Bone-length-preserving zero-phase smoothing (`fk_smooth`) | ACTIVE | decompose → root low-pass (`root_cutoff 3.0`) + limb-dir low-pass (`limb_cutoff 6.0`) Butterworth; harder passes for face/foot/mid groups; renormalize; FK re-compose |
| Hinge-angle clamp | ACTIVE (`clamp_angles: true`) | Rodrigues rotation of distal subtree into [15°, 178°], preserving downstream bone lengths |
| **One-Euro limb smoother** (uncommitted) | **OFF** (`limb_smoother: moving_average`) | new base-limb selector; A/B showed −4/−6% jitter, reproj flat, but 70× slower raw; **human verdict 2026-07-18: keep default, do not enable** — code kept as off option. Inert on production config |
| **Reprojection-error metric** (uncommitted) | diagnostic only | see BUG-12 (framing caveat); does not affect emitted 3D |

---

## 9. Export & visualization

- **UE export** (`export_ue_packets.py`) — ACTIVE; one packet per (frame, global_player_id); world→UE transform verified. Caveat: `timestamp_ns=0` hardcoded (LOW).
- **Mosaic render** (`render_videos.py`, `overlays.py`, `panels.py`, `mosaic_layout.py`) — ACTIVE; calibration-derived layout, occlusion/lost ghosts (reproject last-known fused ground), single-camera ids drawn hollow, BEV + roster panels. CPU JPEG decode forced under the parallel batch driver (GPU decode contends).
- **Bird's-eye** (`render_bird_eye_view.py`) + **P1 QA overlays** (`render_phase1_overlays.py`) — ACTIVE utilities.
- **`animation_viz.py`** — DEAD dev script (crashes on matplotlib ≥3.9); not wired. SCRAP.

---

## 10. Cross-cutting "on/off" summary (quick reference)

**ON in production:** stabilization · bytetrack · tracklet_graph association · z0_reproj emit ·
union-lift/corrob/shape/split merges · approx-feet · synthetic tracklets · auto cue-calibration ·
binding-keyed lift · cheirality · butterworth (3D + refine) · dense-fill · Singer-KF · measurement-R
(update only) · kalman-posterior emit · online role proxy · adaptive lost window · min-cost-flow
stitch · occupancy bridge · colocated merge · v1 epoch roles · peripheral suppression · relift ·
hinge/bone clamp · P1 tiled-det + NMS 0.55 + AMP.

**OFF (kept as gated alternatives / fallbacks):** OC-SORT · per-frame association · EMA 3D smoother ·
robust-refit · v0 roles · One-Euro limb smoother · density lost-window · velocity-gate ·
drop-partial-singlecam · presmooth-ground · symmetric measurement-R gating.

**Inert due to bugs (LOOK LIKE on, actually off):** foot_contact_mode v3 · single_cam_height_emit ·
ankle-height knobs — all defeated by **BUG-1**. `smooth_native` · `--native-skeleton` — dead flags.
