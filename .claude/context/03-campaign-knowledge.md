# Campaign knowledge — settled verdicts (do NOT re-litigate without new evidence)

Full record: docs/critical-analysis/fixes-log.md. Highlights:

**Detection**
- RTMDet only detects at its trained object scale: native hi-res (1280/2560) LOSES boxes;
  tiling wins (superset recall, min box 33→12 px).
- NMS 0.55 (from 0.3) lets both crossing players survive: +0.10–0.13 agreement — the
  largest single identity gain. IoM-0.7 containment kills seam fragments.
- Tiled fast path = crop slicing/resize in prefetch workers + direct
  data_preprocessor→predict (3.2× throughput; fp16 parity ≤3.7 px on usable joints).
  cProfile misleads on this codebase (per-call overhead) — use wall-clock section timing.

**Identity**
- Ghost-under-player = split identity (one player, two global ids in disjoint camera sets).
  Fixed by W9: P3 union-lift merge (one coherent 3D skeleton explaining all views = one
  person) + P4 colocated-id merge (co-located ≥25f within 0.75 m + never share a
  camera-frame + stature agrees). `coloc` panel column + verdict = permanent tripwire.
- Facing pairs C1↔C4, C2↔C6, C3↔C5 (co-observing, low parallax). Colour cue dead (d′≈0),
  bone ratios abstain; billboard posture (STATURE_QUANTITIES) is the facing-pair-capable
  shape channel.
- REJECTED with evidence: contested-camera down-weighting (−0.08 on its target clip);
  H3 posture policy (binding collapse); symmetric measurement-R (gate loosening);
  trainable ReID (no GT).
- P4b stitching is temporal-only; occupancy (same camera-frame) veto = the two-people test.

**Roles (v1.2)**: 6 Hungarian slots (bowler/striker/non-striker/keeper/2 umpires with
distinct geometry), latch + final uniqueness; direction = plausible-band run detection
(3–9.5 m/s — unbanded, tracking teleports fake 20–30 m/s "runs") else pre-shot two-sign
geometric cost. `_14_x` groups do NOT share one bowling end (proven: opposite clean runs).

**3D**: z=0 Gauss–Newton+Huber ground solve (cm-accurate calibration); cheirality =
origin-referenced sign test (det(M) formula wrong on this rig); triangulation core is
batched + bit-identical (W10-PERF); coverage gap = single-camera frames (F16 PnP lift is
the lever).

**Perf**: box CPU chain bottleneck is P3 (solve/IO, flat profile); P6 = 10 s/delivery.

**Incidents**: cam_07 pad-to-/32 (fast-path probes must cover the panoramic cam);
calibration provenance (copied laptop→box, verified + team-confirmed single session);
box-vs-local panels are near-parity not identical when P1 binaries differ.

## Earlier-era settled facts (from auto-memory, re-verified against v8.1 on 2026-07-14)

**Rig geometry**: world origin = pitch centre, +Y toward Far End, z=0 = ground; stump
mid-bases at (0, ±10.08 m). cam01/04 end-on pair, cams 02/03 east (+x), 05/06 west (−x),
cam07 oblique panoramic. Facing pairs C1↔C4, C2↔C6, C3↔C5 are the CO-OBSERVING pairs
(look-at point, NOT antipodal position) — the old config bug is fixed in `configs/v8/`.

**Ground fusion (v8 defaults, A/B-proven)**: `ground_fusion_mode: z0_reproj` (Huber z=0
reprojection minimisation) beat median 0.176→0.145 m; the literature covariance-fusion
recipe LOST (0.248 m) — with cm-accurate calibration, minimise reprojection, don't model
homography noise. P4 `emit_kalman_posterior: true` (chi2-gated) halved trajectory-disp
p95, worst emitted jump 14 m→0.36 m. BEV lesson: remaining teleports/low agreement are
IDENTITY, not location.

**Identity cues (measured)**: kit-colour appearance d′≈0.09 on this desaturated footage
(cue auto-abstains); appearance LLRs must be calibrated PER CAMERA PAIR (a global fit
punished all cam_07 pairs by −1.3…−1.8). Rescue/refinement merges need ≥30 co-visible
frames (an 18-frame rescue built a fielder chimera).

**Cut-off / untracked figures**: upper-body ground estimate (hips z 0.93 → shoulders 1.42
→ bbox-top 1.78) must be STICKY per tracklet — per-frame anchor switching flip-flops ~1 m
and shatters purity splits; failed approximations must yield NaN ground, never garbage
bbox-bottom. Persistent untracked detections (umpires) become synthetic tracklets
(`syn_min_confidence: 0.2` — the 0.24–0.29 conf tail otherwise mints rival IDs).

**Mosaic layout** (`scripts/visualization/mosaic_layout.py`): fully auto-derived per
delivery; bowling direction from PER-CAMERA tracklet motion projected on the pitch axis —
NEVER from fused cross-camera tracks (inter-camera bias reads as motion; that bug produced
a wrong direction). Flip rule validated on M2 (opposite end).

**P1 model history**: RTMPose-L body8 mandate (2026-07-08; no YOLO, no WholeBody) was
superseded by user-approved RTMPose-X. Key fact: the only x-size body checkpoint is
**Halpe-26** (no COCO-17 x exists); its first 17 keypoints are exactly COCO-17, joints
17–25 add head/neck/hip + 6 foot keypoints. "Don't switch to YOLO" still stands.

**Phase-1 perf lore**: batch/io/prefetch knobs are batch-invariant (speed only, never
keypoints); warm-cache benchmarks LIE — always measure on cold data; on the laptop 4060,
GPU nvJPEG decode was a dead end (~19 ms vs cv2 ~14 ms AND competes with pose for the GPU)
— but that verdict is laptop-specific; during L40S renders the GPU is idle, so re-measure
there (05 §1).
