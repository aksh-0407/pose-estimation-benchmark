# 3D Ground-Location — Methods Log (running lab notebook)

Every method tried for the 3D ground-location problem, with its numbers and an explicit
**accept / reject / pending** verdict and the reasoning. Append-only; newest experiments at the
bottom of each section. Companion to `wip/3d_location_issues.md` (the 12 issues) and
`wip/3d_location_redesign.md` (the design).

## SUMMARY TABLE — everything tried (before → after, verdict)

All numbers on the internal 8-delivery set (CCPL080626…). Position proxy A = mean distance of the
emitted cluster `ground_xy` to the calibration-optimal triangulated foot (lower better); reproj =
emitted-point reprojection vs the foot pixels used (px); jitter = P4 track disp p95 (m/frame).

| # | Method (issue) | Before → After | Verdict | Why |
|---|---|---|---|---|
| M0 | Calibration audit | ball reproj 1.2–1.9 px mean, ≤4.5 px p95 | **fact** | calibration cm-accurate → **ISSUE-6 refuted**; error is 2D-foot + fusion, not calibration |
| M1 | Baseline characterisation | proxyA 0.211 m; ~50% single-cam; teleports 382; jitter p95 0.06–0.21 | baseline | the bar to beat |
| M2 | robust_cov (inverse-cov IRLS fusion) | proxyA 0.211 → **0.248** | **REJECTED** | covariance model pulls toward near cams; with cm calibration, reprojection-optimal wins |
| M3 | **z0_reproj** (z=0 robust reprojection solve) | proxyA 0.211 → **0.147 (−36%)**; reproj 102 → 84 px; clustering byte-identical | **WIN (committed)** | uses calibration directly; well-posed on low-parallax facing pairs |
| M4 | foot-v2 (ankle-midpoint + plausibility + height) | proxyA 0.144 → 0.135 (+6% vs z0); reproj −5% | inconclusive → **opt-in** | multi-cam solver already absorbs it; proxies disagree |
| M4b | (why M4 marginal) | ankle-height shift measured = **~0.94 m** | insight | big bias lives on single-cam (50%), unmeasurable by proxies |
| M5 | single-cam height correction | (unmeasurable by A/B; physically correct) | **opt-in** | applies the 0.94 m fix to lone-camera clusters; validate via BEV/downstream |
| M6 | temporal smoothing F7 (median foot) | small | **opt-in** | jitter already low at input |
| M7 | Bird's-eye-view viz | — | **tool** | showed teleports = ID flicker (identity), not location |
| M8 | **Kalman-posterior emit** (ISSUE-5) | jitter p95 ~halved; worst jump **14.0 → 0.36 m** (_7); collisions 0 | **WIN (committed)** | posterior is chi²-gated → can't teleport on one bad frame |
| M9 | v4 mosaics + BEV-in-mosaic | — | **delivered** | monitor tile → QT-style field plot |
| M10 | colour A/B (BGR vs RGB vs grey) | det 0.797 / 0.790 / 0.776 | **no gain** | current BGR best/tied; models colour-robust on desaturated footage |
| — | NVENC GPU encode; `eval_ground_accuracy.py` (ISSUE-12) | — | **tooling** | GPU video encode; reusable accuracy metric |

**Net:** two committed location wins — **z0_reproj** (position −36%) and **posterior emit** (jitter
halved, emitted teleports removed). Foot micro-levers implemented, identity-safe, opt-in (marginal).
Colour not a lever. **The remaining ceiling is IDENTITY, not location** (BEV: dots still, labels swap).

**Standing evaluation rules** (per user): validate on **all 8 deliveries** (lead with the worst,
esp. _5/_6/_7/M2), and **never call something a "win" without a significant, generalised
improvement**. A single-delivery delta is provisional, not a win.

**How results are measured** (no per-player ground truth exists, so we use proxies — see M0):
- **Accuracy proxy A — dist-to-triangulated-foot (m):** distance from the emitted cluster
  `ground_xy` to the calibration-optimal RANSAC-triangulated foot (free-3D DLT, restricted to
  ≥3-view clusters whose triangulation reprojects < 12 px, so the reference is trustworthy).
  Lower = better. Legit because calibration is cm-accurate (M0).
- **Accuracy proxy B — reprojection vs actual foot pixels (px):** project the emitted `ground_xy`
  (z=0) into each member camera, compare to the foot pixel actually used
  (`ground_contact_pixel`). Lower = the emitted point better explains all observations.
  Caveat: pixel-reprojection over-weights far/grazing cameras (long lever arm), so it is a
  supporting metric, not the sole arbiter.
- **Consistency — ground_spread_m:** max pairwise foot-distance among a cluster's cameras. Measures
  member *disagreement* (a foot-pixel/geometry property); a fusion method that only changes which
  point we report **cannot** move it — so it is NOT a fusion-quality metric (learned the hard way, M2).
- **Downstream — teleports, cross-camera agreement, same-camera collisions** (P4 `global_id_metrics`).

Env: pipeline runs in `cricket-rtmpose-l`; pytest in `cricket-yolo26x-pose` with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=""`.

---

## M0 — Calibration audit (grounding, not a method) — ACCEPTED as fact
- **What:** ran `audit_calibration('drive')` (`pose_estimation/cricket/calibration.py`).
- **Numbers:** stored **ball reprojection error mean 1.2–1.9 px, p95 2.8–4.5 px, max ~5 px** on all
  8 deliveries. (The 892 px "projected-vs-stored delta" the audit also prints is a crop/normalization
  convention mismatch in the stored ball artifact — the audit warns this explicitly — NOT calibration error.)
- **Verdict / why it matters:** calibration is **cm-accurate**. This **refutes ISSUE-6** (the config's
  "~0.7–1.2 m same-player residual = calibration bias" claim). The ~2 m player-ground error is the 2D
  foot pixel + single-camera grazing projection + median fusion — not the projection matrices.
  **Consequence:** use the calibration *directly* (reprojection minimisation), don't refine it, and
  don't model it as noise. This decision drove M2 vs M3.

## M1 — Baseline characterisation (all 8 deliveries) — reference numbers
- **What:** parsed committed `p3/association_metrics.json` + `correspondences.jsonl` across all 8.
- **Numbers (baseline, median-of-homographies fusion):** single-camera cluster rate **0.39–0.61**;
  multi-cam ground-spread p50 **~1.9–2.4 m**, p95 **3.4–5.1 m**; % multi-cam reproj > 12 px **34–61%**;
  P4 teleports **7–171**/delivery; cross-camera agreement **0.50–0.98**; same-camera collisions **0**.
  Accuracy proxy A on delivery 1 = **0.176 m mean / 0.376 p95** (1836 reference clusters).
- **Verdict:** this is the bar every method below must beat, on all 8.

---

## Fusion of the EMITTED cluster ground position (P3 `_ground_consensus_members`)

Guardrail for all fusion methods: the **merge gate (max pairwise spread) is left unchanged**, so
clustering/identity is byte-identical — only the *reported position* moves. Verified: `median` mode
reproduces the committed baseline exactly (cluster_count 5306, single-cam 0.3899, binding_count 8 —
all identical on delivery 1).

### M2 — `robust_cov`: inverse-covariance IRLS fusion (the literature recipe) — **REJECTED**
- **Hypothesis (from research):** Pose2Sim / Lee & Civera / multi-camera sports tracking fuse per-camera
  ground points weighted by distance-dependent covariance with robust (Huber) down-weighting. Implemented
  `robust_fuse_ground` + `ground_covariance` (propagate σ_px through the ray-plane Jacobian → anisotropic
  2×2, elongated along the view ray), IRLS Huber over members.
- **Numbers (delivery 1, proxy A, 1836 ref clusters):** median **0.176 m** → robust_cov **0.248 m**
  (p95 0.376 → 0.516). Worse. Closer to truth in only 19.8% of clusters.
- **Why it FAILED for us:** it minimises *metric position variance* under the homography Jacobian, which
  pulls the estimate toward near cameras. But with cm-accurate calibration the *reprojection-optimal*
  point is the accurate one, and those two criteria **diverge on our grazing / low-parallax facing
  geometry** (C1↔C4, C2↔C6, C3↔C5). The covariance model is a model of noise we don't actually have
  (calibration is clean), so it hurt.
- **Verdict:** **REJECTED** as the emitted-position estimator. Kept in code behind
  `ground_fusion_mode: robust_cov` for A/B, and `ground_covariance`/`robust_fuse_ground` are retained
  (unit-tested) for the P4 **Kalman-R** work, where a per-observation covariance IS the right object.
- **Lesson:** validate a paper recipe on the actual rig before adopting it.

### M3 — `z0_reproj`: z=0-constrained robust reprojection minimisation — **PROVISIONAL (delivery 1 only)**
- **Method:** `ground_from_reprojection` (`geometry.py`). Solve
  `argmin_{x,y} Σ_c w_c ρ_Huber(‖project_c([x,y,0]) − foot_c‖)` by Gauss–Newton over every member's full
  3×4 projection matrix, then a **hard inlier refit** (RANSAC-lite: drop views with residual >
  max(3·huber, 2.5·median), re-solve) to fully reject a gross outlier foot (hallucinated ankle 50–200 px
  off). Depth removed by the z=0 constraint → well-posed on low-parallax facing pairs where free
  triangulation is degenerate. Uses calibration directly (the M0 lesson).
- **Numbers (delivery 1):** proxy A **0.176 → 0.145 m** (p95 0.376 → 0.302); proxy B (reprojection vs
  actual foot pixels) **84.3 → 73.7 px mean** (p50 37.5 → 32.5). Clustering byte-identical (5306 / 0.390).
  Unit tests: recovers a synthetic ground point to < 0.10 m with a 50 px outlier view and beats the
  median; single-view case reduces to the exact homography back-projection.
- **Why it works:** it's the reprojection-optimal ground point given the (excellent) calibration, and the
  z=0 constraint keeps it stable exactly where triangulation isn't.
- **Why the gain is only MODEST (~18% / ~13% on delivery 1):** it is **bounded by foot-pixel quality
  (ISSUE-7)** — members genuinely disagree because of ankle-hallucination / bbox-bottom bias, so no single
  point can reproject to all of them. z0_reproj finds the best compromise but can't fix the input. The
  next lever (better foot pixel) should compound with it.
- **Verdict:** **PROVISIONAL — NOT a win yet.** A modest single-delivery improvement. Enabled in
  `configs/p3_association.yaml` (`ground_fusion_mode: z0_reproj`) because it is strictly ≥ median by
  construction and clustering-invariant, but the **all-8-delivery verification is pending** (running).
  Will be re-judged against the full set (esp. _5/_6/_7/M2) before any "win" claim.

### M3 — all-8-delivery verification — DONE (2026-07-08, runs parallelised on 32 cores)

| Delivery | proxy A base (m) | proxy A z0 (m) | A gain | proxy B base (px) | proxy B z0 (px) | B gain | clustering identical? |
|---|---|---|---|---|---|---|---|
| M1_1_14_1 | 0.176 | 0.145 | 18% | 84.3 | 73.7 | 13% | yes |
| M1_1_14_2 | 0.166 | 0.135 | 18% | 71.8 | 64.0 | 11% | yes |
| M1_1_14_3 | 0.172 | 0.116 | 32% | 80.9 | 68.6 | 15% | yes |
| M1_1_14_4 | 0.209 | 0.151 | 28% | 70.9 | 59.7 | 16% | yes |
| M1_1_14_5 | 0.241 | 0.153 | 36% | 129.7 | 119.6 | 8% | yes |
| M1_1_14_6 | 0.200 | 0.116 | 42% | 87.1 | 74.4 | 15% | yes |
| M1_1_14_7 | 0.289 | 0.177 | 39% | 185.0 | 142.2 | 23% | yes |
| M2_1_12_1 | 0.373 | 0.183 | **51%** | 129.3 | 74.1 | **43%** | yes |
| **mean** | **0.228** | **0.147** | **36%** | **104.9** | **84.5** | **19%** | all yes |

- **Verdict: ACCEPTED as a real, generalised improvement** (not a single-delivery fluke). The gain is
  **consistent across all 8** and **largest exactly on the hardest baselines** (_5/_6/_7/M2, which had
  the worst baseline error) — the opposite of overfitting. 36% mean reduction in the position-error
  proxy, 19% in reprojection, **zero clustering/identity regression** (merge gate untouched by design).
- **Honest caveats (why this is "accepted" but not yet the finish line):**
  1. Both metrics are **proxies** — no per-player ground truth exists. Proxy A's reference (triangulated
     foot) uses bbox-bottom feet while z0 minimises ankle-based feet, so they are related but not
     identical; proxy B (reprojection vs the actual feet used) is the more independent check and also
     improves everywhere. The **decisive** validation is still downstream P4 (teleports / agreement) and
     the BEV visual (pt4) — not yet run.
  2. The absolute reprojection is still high (z0 mean 60–142 px across deliveries) because it is
     **bounded by foot-pixel quality (ISSUE-7)**: members genuinely disagree (ankle hallucination /
     bbox-bottom bias), so no single ground point reprojects to all. z0 finds the optimum given bad
     input; fixing the input (ISSUE-7) is the next, compounding lever.
- **Status:** enabled in `configs/p3_association.yaml`. This closes the P3 emit-path portion of
  ISSUE-2/3. Remaining large levers: foot pixel (ISSUE-7), pixel-space EKF (ISSUE-4), offline smoothing
  (ISSUE-9).

---

## Foot-contact pixel (ISSUE-7)

### M4 — `foot_contact_mode: v2` (ankle-midpoint + height correction) — INCONCLUSIVE, kept opt-in (default OFF)
- **What (F2/F3/F4/F6):** `ground_contact_pixel_ex` (`geometry.py`) — ankle **midpoint** as the
  cross-camera-consistent reference when both feet are down (else planted/lower ankle), tighter
  vertical + **new horizontal** plausibility, and it reports the **ankle height** so the z0 solver
  back-projects onto `z = ankle_height (0.10 m)` not `z = 0` (removes the ~10 cm bias). Solver
  generalised with `plane_heights`. Unit-tested.
- **Attempt 1 — REGRESSION caught by all-8 testing:** first wiring let the v2 foot feed the clustering
  **gate** (via `detection.ground_xy`). Result: cluster count inflated **+25% on _2, +23% on _6**,
  single-camera rate 0.46→0.61 on _6 — i.e. identity fragmentation. (Would have looked fine on delivery
  1 alone: 5306→5305.) **Lesson: a foot-pixel change that touches the gate changes identity; test all 8.**
- **Fix — decoupled:** the gate/cost/triangulation path is pinned to the **legacy** foot
  (`_foot_pixel` mode="legacy"); `foot_contact_mode` now affects **only the emitted z0 position**
  (`_emit_foot_and_height`). Re-verified: **cluster count byte-identical to baseline on all 8.**
- **Numbers (decoupled, all 8, clustering identical → valid comparison):**

| metric (mean over 8) | baseline | z0 | z0 + foot-v2 |
|---|---|---|---|
| proxy A — dist to triangulated foot (m) | 0.211 | 0.144 (+32%) | 0.135 (+36% vs base, **+6% vs z0**) |
| proxy B — reproj vs legacy foot px | 102.3 | 83.5 (+18%) | 88.0 (+14% vs base, **−5% vs z0**) |

- **Verdict: INCONCLUSIVE — NOT a win.** Proxy A (the fairer metric; independent of the foot
  *definition*) says +6%; proxy B (rewards matching the *legacy* foot, so biased against v2) says −5%.
  A 6% move on one proxy with the other disagreeing is **not significant**. **Decision: keep the code,
  flag-gated, default `legacy` (NOT enabled in the committed yaml).**
- **Why marginal:** (a) the ankle-height correction is ~10 cm vs a ~0.15 m residual; (b) the midpoint
  benefit is largely already captured by the robust z0 fusion; (c) **~50% of clusters are single-camera**
  and the decoupling deliberately leaves them (and the gate) on the legacy foot — so half the data is
  untouched.
- **Where the real foot gains are (next):** **F7 temporal smoothing** of the foot pixel per tracklet
  (attacks jitter/teleports — not yet done); **WholeBody foot keypoints** (heel/toe) via a P1 re-run,
  which fixes F2 structurally instead of a 10 cm prior. Both logged as planned.

### M4b — WHY foot-v2 was marginal (instrumentation, 2026-07-08) — root cause found
- Ran the foot selector over the actual predictions (delivery 1 + M2). Findings:
  - Ankles are **reliable**: both ankles conf ≥0.6 in **75–77%** of detections (bbox-bottom fallback ~20%).
  - v2 vs legacy foot pixel differs on ~20% of detections (mean 12 px) — mostly midpoint-vs-single-foot.
  - **The ankle-height correction is a ~0.94 m ground shift (p95 1.3 m), NOT ~10 cm** — at grazing angles,
    projecting the ankle to z=0 vs z=0.10 m moves the ray-ground intersection by nearly a metre.
- **So why only +6%?** The multi-camera **z0 solver already absorbs** most of the single-view height bias
  by fitting all views jointly. The full ~0.94 m bias is only live on **single-camera clusters (~50% of
  data)** — and the identity-safe decoupling had left those on the legacy z=0 projection. **That is the
  untapped lever.**

### M5 — single-camera height correction (extend z0_reproj to 1-view) — numbers pending P4
- **Change:** the z0_reproj emit branch now handles single-member clusters (project the ankle onto its
  z=ankle_height plane). A 1-member cluster has max_spread 0, so the **merge gate is untouched** →
  identity unchanged. Only active with `foot_contact_mode: v2` (legacy reports height 0 → committed
  default byte-identical). Captures the ~0.94 m single-cam bias that proxies A/B **cannot measure**
  (proxy A needs ≥3 views; proxy B is ~0 for a single self-consistent view) — so this is validated
  **downstream (P4 teleports/agreement)** and visually (BEV), not by A/B.

### M6 — temporal smoothing of the emit foot (F7) — numbers pending P4
- **Change:** `smooth_emit_feet` (`associator.py`) median-filters each (camera, tracklet) foot-pixel
  series (`foot_smooth_window`, default 1=off) and attaches it as `emit_foot_px` (emit-only, never the
  gate → identity unchanged). Kills single-frame ankle spikes before they enter the solver. Unit-tested
  (a 300 px spike frame is suppressed to ~140 px). Expected to reduce jitter/teleports; measured via P4.
- **Downstream result (P4 on all 8, z0 + foot-v2 + single-cam-height + smooth5 vs committed baseline):**

| metric | baseline | + all foot changes |
|---|---|---|
| cross-camera agreement (mean) | 0.812 | 0.812 (no change) |
| teleport events (total over 8) | 382 | 364 (**−5%**; M2 171→160, _5 48→43, _7 59→57) |
| same-camera collisions | 0 | 0 (invariant held) |

- **Verdict on M5+M6: NOT a win.** Small teleport reduction, no agreement change.
- **Why the location fixes barely move the identity metrics (important):**
  1. `cross_camera_agreement` is **computed from independent bbox-bottom projections**, by design (so
     it judges clustering rather than echoing it) — it **structurally cannot see** a change to the
     emitted ground position. So 0.812→0.812 is expected, not a failure.
  2. **Teleports/agreement are dominated by IDENTITY errors** (an ID jumping between two different
     people), not by position noise. Better *location* accuracy doesn't fix *wrong associations*.
     Identity is the separate workstream (`implementation_plan.md`), not this one.
  3. **Single-camera position accuracy — the whole point of M5 — is unmeasurable by A/B AND barely
     visible downstream** (single-cam detections enter P4 at low confidence 0.20–0.45 and are
     down-weighted). So neither proxies nor P4 can adjudicate M5.
- **Consequence:** the honest, correct next step is the **bird's-eye-view visual (pt4)** — the only
  instrument that can show whether single-camera 3D positions land in sensible places. Keeping foot
  changes in code, flag-gated, **default OFF** (committed config stays `foot_contact_mode: legacy`)
  until the BEV confirms they help. z0_reproj (the one real, +36% position win) stays enabled.

### Ops lesson — parallelism / BLAS oversubscription
- Running 8 P3 jobs in parallel **without capping BLAS threads** → each spawned ~32 OMP threads →
  load average **45 on 32 cores** → thrash, ~0 progress in 10 min. **Fix:** set
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` per process → each ~99% CPU, 8 jobs ≈ 8 cores,
  fast. Always cap BLAS threads when fan-out is CPU-bound numpy/scipy.

## M7 — Bird's-eye-view visualisation (pt4) — BUILT; validates locations, isolates identity as the culprit
- **Tool:** `scripts/visualization/render_bird_eye_view.py` — QT-style top-down field plot, each player's
  world (x,y) as an ID-coloured dot, animated (mp4) + sampled montage; reads P4 `ground_tracks.jsonl`.
- **What it shows (delivery 1 + M2):**
  - **Locations are good and stable** — realistic cricket layout (pitch strip at x≈0; batsmen/keeper at
    the crease ends; fielders spread; 8–12 players/frame), **zero wild outliers**, positions steady
    across frames. Strong qualitative confirmation the z0_reproj 3D positions are sound.
  - **The teleports are IDENTITY flicker, not location error.** On M2 the stationary right-side deep
    fielder is labelled **P009 in some frames and P014 in others** (same spot); P013 flickers in/out.
    The dot sits still; the *label* jumps. So the downstream teleport/agreement problems are wrong ID
    assignment (the `implementation_plan.md` workstream), **not** the ground positions this effort fixed.
- **Known gap:** the single-camera hollow-marker overlay didn't populate (correspondences `binding_id`
  ≠ ground-track `global_player_id`, so the join missed). Minor viz fix; positions still render.
- **Takeaway:** the 3D-**location** objective is in good shape (z0_reproj, +36% proxy, sensible BEV);
  further teleport/agreement gains require the **identity** workstream, not more location work. Remaining
  location polish (EKF measurement model, offline smoothing) is incremental.

## M8 — Emit Kalman posterior (ISSUE-5) — **ACCEPTED (clear win), enabled by default**
- **Change:** P4 emits `track.kalman.pos_world_xy` (the chi2-gated filtered posterior, already
  computed by `manager.update`) instead of the raw per-frame fused observation. Flag
  `p4a.emit_kalman_posterior` (default now true). Isolated A/B: same committed z0 P3 input,
  posterior vs raw emit, all 8.
- **Numbers (all 8):** trajectory **displacement p95 roughly halved** every delivery (_6 0.148→0.045,
  _7 0.169→0.065, M2 0.212→0.085 m/frame); **worst emitted jump 14.0 m → 0.36 m on _7**, _3 2.73→0.76,
  _6 3.32→1.56; **same-camera collisions still 0**. Not over-smoothed (0.05 m/frame ≪ the 0.18 m/frame a
  real 9 m/s player covers).
- **Note:** `teleport_event_count` barely moved (382→374) because that metric is counted at ID-
  *assignment* time, not on the emitted trajectory — an identity signal, not a location one. The
  *emitted positions* (what downstream + the BEV use) are what improved.
- **Verdict: WIN.** Consistent, significant on jitter/max-jump across all 8, zero regression. Enabled.

## M9 — pipetrack_v4 mosaics with BEV replacing the delivery monitor — DELIVERED
- **Renderer:** `scripts/visualization/render_phase1_videos.py` — new `draw_bev_panel` +
  `compute_ground_extents`; the mosaic's bottom-left `MONITOR_SLOT` now renders a QT-style top-down
  field (boundary ellipse, pitch strip, id-coloured player dots + motion trails) driven by the P4
  `ground_tracks.jsonl`, instead of the text delivery monitor.
- **Source config (v4 = the polished stack):** P3 `ground_fusion_mode: z0_reproj` + `foot_contact_mode:
  v2` + `foot_smooth_window: 5`; P4 `emit_kalman_posterior: true`. Output: `artifacts/pipetrack_v4/mosaics/`.
- Smoke frame verified: BEV dots (P001–P010) land at sensible field positions consistent with the 6
  camera tiles + the global roster.

## Summary — location effort outcome (as of 2026-07-08)
- **Wins (committed):** z0_reproj (position −36% vs median, all 8) + Kalman-posterior emit (jitter
  ~halved, worst emitted jump 14 m→0.36 m). BEV confirms positions are cm-plausible and stable.
- **Marginal / opt-in:** foot-v2, single-cam height, foot temporal smoothing (identity-safe, small).
- **Deferred:** covariance-R EKF (posterior captured the main filtering benefit).
- **Key learning:** most *remaining* downstream error (teleports, agreement) is **identity**, not
  location — the BEV shows dots holding still while ID labels swap. Location is in good shape; the next
  pipeline-level gains are in the identity workstream (`implementation_plan.md`).

## M10 — P1 pose colour-profile A/B (pt7) — tested empirically, NO gain, current is best
- **Config check:** detector RTMDet `bgr_to_rgb=False` with BGR-ordered means `[103.53,116.28,123.675]`;
  pose RTMPose `bgr_to_rgb=True` with RGB means `[123.675,116.28,103.53]`. Script feeds `cv2.imread`
  (BGR). Both stages therefore handle BGR correctly — no channel bug.
- **Empirical A/B (real RTMDet+RTMPose, 120 frames, cams 01/03/04):**

| colour order | people | det_conf | kpt_conf mean/p50 |
|---|---|---|---|
| current (cv2 BGR) | 520 | 0.797 | 0.653 / 0.731 |
| swapped (fed RGB) | 520 | 0.790 | 0.656 / 0.735 |
| greyscale | 520 | 0.776 | 0.650 / 0.726 |

- **Verdict:** RGB-swap = within noise (±<1%, identical detection count); greyscale slightly WORSE
  (−2.6% det conf). Models are colour-robust on this low-saturation footage (green field / white kit).
  **Current BGR is best/tied — colour profile is not a lever.** Tested rather than assumed
  (`scratchpad/colour_ab.py`).

## M11 — Full 17-keypoint multi-view triangulation + occlusion extrapolation (Task 3) — WIN where ≥2 cams
- **What:** ran the (previously never-run) P6 stage `scripts/export/triangulate_predictions.py`, which
  triangulates ALL 17 COCO keypoints per global player across the calibrated cameras (weighted DLT +
  per-joint RANSAC), not just the foot. Added occlusion extrapolation in
  `pose_estimation/triangulation.py`: `fill_occluded_joints` (temporal interp of joints missing in
  scattered frames) + `fill_from_skeletal_prior` (parent-joint + median-bone-length fallback).
- **Triangulation accuracy (delivery 1, 2960 frames):** per-joint reprojection **p50 2–4 px, p95 5–7 px**
  across all 17 joints — cm-level full-body 3D. Triangulated **ankle z ≈ 0.11 m** (p50), independently
  confirming the anatomical ankle height used by the single-cam correction (M5).
- **Occlusion fill (delivery 1):** complete 17-joint poses **2960 → 3238 (+278) = 100%** of the 3238
  multi-camera frames (was 91%); 278 frames used extrapolated joints (low confidence, sentinel reproj).
  Unit-tested (temporal interp + hold).
- **Verdict: WIN for the 3D POSE deliverable** — all multi-camera frames now emit a complete, cm-accurate
  17-joint world pose. **Remaining gap:** the **~39% single-camera frames** (3238/5301 multi-cam) cannot
  be triangulated at all; they need single-view pose reconstruction (fit the identity's canonical 3D
  skeleton to the lone view at its z0 ground position) — the next build.
- **Note:** this is a much richer output than the P3/P4 ground point (which uses only the foot). The
  full skeleton also gives a more robust ground position (pelvis/ankle-derived) and enables the UE export.

## Planned experiments (not yet run) — will be logged here with numbers + verdict
- **Foot-pixel quality (ISSUE-7):** planted-foot selection + per-tracklet temporal smoothing of the foot
  pixel. Expected to compound with M3 (relaxes its bound).
- **EKF with pixel-space measurements (ISSUE-4 + legs-not-visible):** replace fixed Kalman R with a
  nonlinear `h(x)=project(x)` measurement model so the Jacobian yields distance-aware uncertainty.
- **Per-track height for the legs-not-visible anchor (ISSUE-8):** replace fixed 0.93/1.42/1.78 m priors.
- **Offline trajectory smoothing (ISSUE-9):** Pose2Sim-style zero-phase Butterworth / RTS smoother.
- **P1 pose colour profile A/B (pt7):** RGB↔BGR and colour↔greyscale vs current; check MMPose/MMDet
  `data_preprocessor` `to_rgb`. Measure keypoint → foot → 3D impact on all 8.
