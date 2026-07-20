# pipetrack_v9 state and the 2026-07-16 to 2026-07-17 A/B session

Current-state addendum to the 40-delivery diagnosis. It records where the pipeline stands after the
pipetrack_v9 restructure, every method measured in the post-v9 A/B session, the failures, and the levers
that can now be tried. The full per-method ledger with pros and cons is `docs/methods_log.md`; the
prioritized backlog is `roadmap.md`; known bugs are `analysis/README.md`.

## 1. Final state after pipetrack_v9

The v9 restructure keeps Halpe-26 canonical and a single binding-keyed triangulation (stage 04) run
before global identity, and adds stage 07 refine (physics-constrained 3D skeleton rebuild, hip de-wobble,
low-confidence refill) after identity. The stage chain is 00 inference, 01 stabilization, 02 tracking, 03
association, 04 lift, 05 global id, 06 roles, 07 refine, 08 render, with a per-delivery run layout.

The production reference panel (v8.1, 40 deliveries, the panel v9 builds on): mean cross-camera agreement
0.862 (range 0.527 to 0.992), reprojection 3.07 to 3.56 px, same-camera collisions 0 everywhere,
colocated-id pairs 0 on 38 of 40. Identity is still the dominant quality ceiling: facing-pair split
identity and single-camera coverage.

The two structural facts that drive everything downstream:
- Roughly 39% of frames are single-camera and get no triangulated 3D. Coverage is a detection problem
  first and an identity problem second.
- The co-observing camera pairs (C1-C4, C2-C6, C3-C5) face each other from opposite sides. This is low
  parallax, so the usual cross-camera geometry is unreliable there, and this is where identity is hardest.

## 2. This session's A/B campaign (all measured, decisions deferred to human review)

All identity A/Bs were run from a frozen 40-delivery all-flags-on baseline (`pipetrack_v91_base`) on the
box, at 8-wide with BLAS threads capped to 1. Detector A/Bs re-ran P1 and chained 01 to 05.

### 2.1 The shipped flags, measured on all 40 for the first time

These five flags ship on but had only been measured on the easy 8-clip set, where they were inert or
noise-level. Delta is flag-OFF minus flag-ON (a positive agreement delta means the flag was hurting).

| Flag turned OFF | mean dAgreement | sum dTeleports | sum dIDs | collisions |
|---|---|---|---|---|
| `graph_shape_enabled` | +0.0000 | +0 | +0 | 0 |
| `graph_split_enabled` | +0.0032 | +0 | -3 | 0 |
| `graph_facing_gate_scale` | +0.0010 | +46 | +2 | 0 |
| `use_measurement_covariance` (distance-R) | -0.0013 | +71 | +2 | 0 |
| `adaptive_lost_window` | -0.0002 | +40 | +0 | 0 |

- `graph_shape_enabled`: fully inert on all 40. Doing nothing on this data. Pro: harmless. Con: dead
  weight. Status ENABLED-INCONCLUSIVE; cleanup candidate.
- `graph_split_enabled`: a slight agreement drag here. Pro: conservative, no collision risk. Con: no
  measured benefit. Status ENABLED-INCONCLUSIVE; cleanup candidate.
- distance-R, the facing gate, and the adaptive lost window: each suppresses real underlying teleport
  events (71, 46, 40) at negligible agreement cost. Pro: teleport robustness. Con: these are underlying
  events, and the A3 emit-gate masks the visible ones regardless, so the value is robustness, not
  on-screen marker count. Status ENABLED-INCONCLUSIVE, but with a real teleport-suppression signal.

### 2.2 Tiled detection (the recall lever)

Recall-gap evidence (no ground truth): detections are large and high-confidence (14_7 p50 confidence
0.84, p50 box height 452 px), with almost nothing borderline (3% below 0.4 confidence, 2% below 100 px).
Missing distant and dark players score near zero, not just under threshold. This is a scale problem, not a
threshold problem, so tiling (re-scaling distant subjects to the detector's trained size) is the correctly
targeted lever, and lowering the confidence threshold would recover little. cam_07 sits at about one
person per frame; cam_02 and cam_06 are pinned at a constant count.

Broadened result (tiled plus NMS 0.3, clean isolation, the 8 hardest deliveries by baseline agreement):
agreement improves on all 8 (mean +0.115, range +0.002 to +0.236), but underlying teleport events regress
badly on crowded clips (sum +704; +148, +182, +222 on the M2_2_3 group; only 14_6 and 14_7 improved on
teleports). Collisions held at 0.

Isolation established that the win is tiling and not NMS: plain-640 plus NMS 0.55 (NMS changed alone) hurt
agreement on both tested clips and roughly doubled teleports despite producing the most detections.

Pros: agreement generalizes on the hardest clips. Cons: teleport regression on crowded clips, only +2 to
+4% detection recovery that misses the starved cameras, and about 3x GPU cost. Status PENDING, two-edged.
The honest next measurement is tiling with the A3 emit-gate on, since A3 masks exactly the teleports
tiling inflates. No stronger detector weights exist on the box (RTMDet-l/x, RTMO-l, YOLO are empty
placeholders), so tiling is the only runnable recall lever.

### 2.3 OC-SORT tracker (stage 02)

Built as a config-selectable alternative to the ByteTrack-plus-constant-velocity-Kalman tracker, targeting
the documented fragmentation on sharp manoeuvres (OCM velocity-consistency cost, ORU virtual-trajectory
re-update on recovery, OCR last-observation recovery pass). The `bytetrack` default was verified
byte-identical (control reproduced the baseline exactly on 14_7).

40-set A/B: fragmentation proxy `p2_tracks` minus 26 (fewer per-camera fragments, exactly what it targets),
but mean agreement minus 0.0129, teleports plus 151, collisions 0. Pro: it reduces fragmentation. Con: the
fragmentation reduction does not translate downstream; the OCR and ORU recovery reconnects fragments and,
on the low-parallax facing pairs, some reconnections are wrong-player merges that hurt agreement and spawn
teleports. Status REJECTED as implemented, off by default. Possible ablation: disable OCR, keep ORU and OCM.

### 2.4 Script optimization

Six run-script fixes verified by compile and dry-run. The data-parallel P1 launcher had a broken runner
path and failed on every shard; fixed and dry-run validated, restoring the roughly 2x GPU-throughput
lever. Plus render and P1-shard thread-oversubscription fixes for the 8-core box. Detail in
`reference/performance.md`. These do not change pipeline decisions (fix 4, the render
CPU-decode switch, may change render pixels marginally but touches no metric).

## 3. Levers that can now be tried (updated)

Highest value first, full list in `roadmap.md`:
- Decide-in-3D consumption in stage 05 (A0), plus re-triangulate per global id to recover the v8-to-v9
  `tri_cov` drop. The top item.
- Single-view PnP lift (A8) for the roughly 39% single-camera frames.
- Tiling plus A3 combined 40-set A/B, the honest way to read the tiling tradeoff.
- 05b stitching under-merge (distinct IDs 18 to 25 vs the roughly 11 roster).
- Depth-aware association weighting for the facing pairs (A4), the principled version of the cap fix.
- Flag cleanup: remove or disable the inert `graph_shape` and the slightly-negative `graph_split`.
- OC-SORT ablation: OCR off, keep ORU and OCM.

## 4. Known bugs and metric caveats (pointer)

See `analysis/README.md`. The key caveats: the raw teleport proxy reacts to noisy single-camera
foot projections rather than the emitted trajectory (A1 in the backlog is the metric fix); the
`emit_kalman_posterior` guard is active but ineffective as a teleport guard (BUG-1); and the dataclass
defaults for several flags disagree with the shipped YAML, which caused an earlier on-vs-on A/B error
(NB-2, verify production flag state from `configs/*.yaml` and `run_manifest.json`, not the dataclass
defaults).
