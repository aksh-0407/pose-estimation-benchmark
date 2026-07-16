# 3D-rework session summary — for morning review (2026-07-16)

Everything below is **flag-gated and default-off except the cap fix**, which is the only change
enabled in production. Nothing else was enabled without your sign-off. A/B standard: 8_init for
iteration, 40_full to confirm a win. Mosaics are the final judge.

---

## ✅ ACCEPTED & ENABLED — the one production change

### Cap fix: `graph_llr_positive_cap 1.5 → 3.5` (configs/03_association.yaml)
The facing-pair (02-06 / 03-05 / 01-04) under-merge fix. The flat 1.5 cap throttled genuinely-tight
ground agreement on co-observing pairs; 3.5 is the measured agreement peak (4.0+ over-merges).

| | 8_init | **40_full (confirmed)** |
|---|---|---|
| cross-camera agreement | 0.880 → 0.916 | **0.853 → 0.883 (+0.030)** |
| central-player under-merge | 16% → 7% | **11% → 6%** |
| ghost/coloc pairs | 2 → 0 | **5 → 0** |
| same-camera collisions | 0 | **0** (held) |

Mosaic-confirmed by you on 14_3 ("after is definitely better"). **Stays landed.**

---

## ❌ REJECTED — reverted, code removed

### Tracklet-id lock (diagnosis-07 flicker)
Despike variant cut 2D id-switches 65→39 with agreement essentially held (−0.0016), BUT on mosaic
review it put a stable **wrong-person id** on a player — a regression baseline never had. **All code
removed** (helpers, flags, tests, configs; 205 tests, grep-clean). Logged in fixes-log. Lesson: never
stabilise identity by post-hoc per-tracklet relabel; any flicker fix must act at the cross-camera
assignment level.

---

## 🟡 BUILT + TESTED, AWAITING YOUR VERDICT (default-off)

### IMPACT-2: partial-detection drop (`p4a.drop_partial_singlecam`, P5 emission)
Your keeper-head / cut-off-umpire observation. A single-camera id that is predominantly **partial**
(median confident-keypoint count < 8 — head-only) is DROPPED at emission. **Drop-only, never a relabel
→ structurally cannot repeat the lock's wrong-person failure.** Full-body single-camera peripherals
(26 kpts) are spared. (First tried at P3-binding level — ineffective, P5 re-spawned the ghost; moved to
P5.)

- 8_init A/B: **4 ghost ids dropped**, distinct ids 89→85, agreement 0.9160→0.9161 (held), collisions 0,
  byte-identity(off) PASS.
- **40_full confirmed: 13 ghost ids dropped** (462→449), agreement 0.8831→0.8832 (held), collisions 0,
  coloc 0. Generalises cleanly — drops partial ghosts, touches nothing else.
- **Mosaics to review:** `data/viz/impact2_mosaics/14_3_BEFORE_ghost.mp4` vs `..._AFTER_ghostdropped.mp4`.
  Check: head-only ghost gone in AFTER; every real player unchanged.
- **Verdict needed:** enable `drop_partial_singlecam`? (I recommend yes if the mosaic shows the ghost
  gone with no real player touched — but your call.)

### A3: emitted-track velocity gate (`p4a.emit_velocity_gate`, P5 emission) — STRONG 8_init WIN
The direct fix for the "haywire ghost markers going crazy". Built after the 1F A/B proved the teleport is
**not** a position-source problem (foot/hip/sticky-hip and the built-in Kalman posterior all leave it
untouched) — it is an id-level jump. A3 walks each id's emitted ground track and DROPS any frame whose
implied speed from the last kept frame exceeds 12 m/s (a real cricketer never exceeds ~11); gap-scaled, and
re-anchors after 5 consecutive drops so a genuine relocation isn't deleted. **Drop-only — never moves or
relabels a position, so like IMPACT-2 it structurally cannot repeat the lock's wrong-person failure.**

- 8_init A/B (05-only, FOOT source, isolates the gate): **teleports 33 → 0**, catastrophic **max 1220 → 11.8
  m/s**, **p95 (1.4), distinct-id count (89), agreement (0.9160) ALL unchanged**, only 145/46781 steps (0.3%)
  dropped. **Flag-off byte-identical 8/8.** 5 unit tests, 217 green.
- **40-CONFIRMED (all 40):** teleports **367 → 0**, catastrophic max **2224.7 → 11.9 m/s**, distinct ids
  462→462 (none lost), agreement 0.8831 unchanged, p95 held, only 0.55% of steps dropped. Generalises the
  8-set perfectly.
- Teleports concentrate in **M2_1_12_1 (18)** and **M1_1_14_5 (12, incl the 1221 m/s max)** — the M2
  full-stack mosaic you have still shows them (no A3), so M2 before/after is the ideal A3 demo. Verified on
  the hip-emission stack tracks: M2 16→0, 14_5 11→0, no ids lost. Stack+A3 "after" trees built; before/after
  mosaic renders queued behind the pool.
- **Verdict needed after the mosaic:** enable `emit_velocity_gate`? This is the most surgical,
  lowest-collateral change of the campaign and targets exactly your ghost-marker complaint — **I recommend
  yes**, pending only your look at the before/after mosaic.

---

## 📊 ANALYSES you requested (read-only, no pipeline change)

### 1B: per-camera robustness + monocular-vs-multiview (`tools/diagnosis/camera_robustness.py`)
**Confirmed across all 40 deliveries:**
- **Per-camera reprojection (which camera is the outlier):** cam_06 **8.5px** (cleanest) / cam_04 9.6 /
  cam_02 9.9 / cam_01 10.3 / cam_05 10.7 / cam_03 12.1 / **cam_07 12.9px (worst)**. Only a ~1.5× spread →
  **no pathological camera; the rig is healthy.** cam_07 (oblique panoramic) and cam_03 are the
  highest-disagreement cameras — worth watching, but not a "bad camera" to exclude.
- **Leave-one-camera-out (all 40):** dropping any single camera shifts the triangulated hip only
  **5.7–7.0 cm** → the multi-view triangulation is robust to losing/mistrusting any one camera. No single
  camera dominates the solution.
- **Monocular (2-view) vs multi-view** (14_3+14_7 sample): a 2-camera estimate differs from full
  multi-view by **~8 cm mean, 30 cm p95** → multi-view is meaningfully tighter; quantifies the
  multi-camera advantage you asked about. (No monocular *lift* model exists yet; this is the 2-view
  proxy — a real monocular-lift comparison is part of 1F/F16.)

### 1D: stabilization-order A/B (stab-then-3D vs triangulate-raw-then-3D-smooth)
Answers the colleague's question ("stabilize first, or triangulate first then stabilize?"). 8_init:

| | ARM A: stab-first (current) | ARM B: raw→3D-smooth |
|---|---|---|
| cross-camera agreement | **0.9160** | 0.9114 |
| 3D-joint jitter | **0.0105 m** | 0.0117 m |
| teleports | **258** | 280 |
| reproj mean / p95 | 3.27 / 6.45 | 3.28 / 6.47 |

**VERDICT: keep the current stab-first ordering.** 2D-stabilize-before-triangulate is better on *every*
axis — smoother 3D, higher agreement, fewer teleports. The colleague's concern that 2D stabilization
distorts the 3D is not borne out; removing per-view pixel jitter *before* triangulation prevents 3D
depth-swimming that a post-hoc 3D smoother can't fully recover. (8-delivery; validates the status quo,
so no change to enable.)

---

## ⏭ NEXT (designed; will build/measure — some need your steer)

- **1F single-view sticky-hip lift — BUILT + A/B DONE → ❌ NEGATIVE, do NOT enable.** Flag
  `p4a.single_view_hip_fallback` (default-off; only active with `emit_ground_source: triangulated_hip`).
  Learns each id's sticky hip height (median hip-z over its multi-camera pose_3d frames) and, for
  single-camera frames, back-projects the hip pixel onto that height plane (`geometry.pixel_to_plane_xy`)
  instead of the noisy foot fallback. 3 unit tests, 212 green.

  **A/B result (8_init, 05-only off shared cap3.5+robust 04; same 46781 track-steps, only emitted
  position differs):**

  | arm | teleports >25 m/s | p95 | max | agreement |
  |---|---|---|---|---|
  | FOOT (base) | **33** | 1.4 | 1220.5 | 0.9160 |
  | HIP (1A) | 32 | 1.8 | 1220.5 | 0.9160 |
  | HIP+1F | **35** | 2.0 | 1220.5 | 0.9160 |

  **1F slightly *increases* teleports (33→35) and p95 (1.4→2.0).** Hypothesis fails: a single-view hip
  back-projected onto a sticky height plane swings with torso lean, so it is *noisier* frame-to-frame than
  the foot fallback (feet are physically on the ground). **Verdict: do not enable 1F; leave the flag as a
  default-off option.** 1A (hip) is teleport-neutral (33→32).

  **Key redirect this A/B gives us:** the `max = 1220 m/s` outlier is **identical across all three arms**,
  so the dominant teleport driver is *not* the emitted position source (foot/hip/sticky-hip) — it is an
  **id-level jump** (one id inheriting two players' positions across consecutive frames). Therefore the
  real teleport lever is a **per-tracklet emission velocity-gate / damping (backlog A3)**, not a better hip
  source. That is the next teleport-targeted change to build.

  Prior teleport-source finding (still holds): of 33 emitted teleports (8_init cap-3.5), 88% (29) are at
  single-camera frames — but the A/B above shows changing the single-camera *position source* doesn't move
  the count, so the jumps are anchor/id-driven, not plane-height-driven. Confirms A3 over 1F.

  Prior teleport-source finding (still holds):
  Teleport-source analysis (8_init cap-3.5): of 33 emitted teleports (>25 m/s), **88% (29) are at
  single-camera frames**, 12% multi-camera. So the residual "haywire ghost markers" ARE a single-camera
  emission problem — the right target. BUT the single-camera hip is *already* back-projected onto a fixed
  0.93 m plane, so the teleport cause is one of: (a) fixed-plane-height error, (b) anchor switching
  (hip→shoulder→bbox-top frame-to-frame), (c) single-view pixel noise. 1F (sticky *learned* height) only
  fixes (a); (b)/(c) need a per-tracklet emission velocity-gate / damping (backlog A3). NEXT: a
  finer teleport-cause classifier to pick the fix, THEN build flag-gated + A/B + mosaic. I did NOT build
  1F blind (1A was teleport-neutral; this is position-changing, and post-lock I'm being conservative).
- **IMPACT-4 splittable clustering** — replace single-linkage union-find with a splittable objective for
  residual over-merge. Bigger/riskier change; I'll design + A/B carefully (not enable unsupervised).
- **IMPACT-6 detector eval (RTMO / RT-DETR / Co-DETR)** + deep-field recall — heavy (model fetch +
  bakeoff); its own phase.

## Open verdicts I need from you
1. IMPACT-2 (`drop_partial_singlecam`): enable? (mosaic + numbers above)
2. 1A hip-emission (`emit_ground_source: triangulated_hip`) and 1C robust-triangulation
   (`--tri-robust-refit`): both built, ~neutral on metrics, default-off — enable, or leave as options?
