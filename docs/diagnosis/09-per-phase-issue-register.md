# Per-phase issue register

Every phase of the chain, the issues found in the 40-delivery production tree, and the
downstream symptom each feeds. Cross-references the deep-dive issue docs (03–08) and the
change list (`../../wip/to_do.md`).

Pipeline order: **P1** (2D pose) → **P1b** (2D stabilization) → **P2** (per-camera tracking)
→ **P3** (cross-camera association) → **P3.5** (binding 3D lift) → **P4a** (online global id)
→ **P4b** (stitch + colocated merge) → **P5** (roles) → **P6** (terminal 3D) → **R** (render).

---

## P1 — 2D pose (RTMDet-m tiled + RTMPose-X Halpe-26)

| # | Issue | Evidence | Feeds |
|---|---|---|---|
| P1-1 | **Deep-field / small-subject recall** is the upstream driver of everything. Deep fielders and umpires are ~10–20 px; when a camera misses them they become single-camera → no triangulation, noisy foot ray, split id. | single_cam 0.76–0.82 on `M2_2_3_*`; coverage falls to 0.48. | 05, 08, 04 |
| P1-2 | **cam_07 panoramic** (3775×960) sees players tiny with low keypoint confidence; its 2D is the weakest and its tracklets fail to associate. | cam_05↔cam_07 top split pair on M2_2_3. | 05 |
| P1-3 | jitter_px 1.2–3.4 (panel) — 2D keypoint noise is 2–3 px, the floor for reprojection error; hips are the worst joint (11–12 px cross-view). | manager reprojection analysis (`wip/to_do.md` §H). | 3D reproj |

Verdict: P1 is **not** the identity algorithm's fault, but it is the **root cause** of the
coverage/single-cam problem that everything else inherits. Highest-leverage upstream fix.

## P1b — 2D stabilization (One-Euro)
No production issue found; jitter_px is low. Kept ON for the worst-clip floor. Not implicated
in teleports (those are position-fusion, not 2D-jitter, artifacts).

## P2 — per-camera tracking (ByteTrack-style, no-spawn)

| # | Issue | Evidence | Feeds |
|---|---|---|---|
| P2-1 | Per-camera tracklets are **stable** (this is a strength) — 25–38 tracklets/delivery, most hold one identity. Collisions 0 everywhere. | `idswitch_2d.py`. | — |
| P2-2 | Fragmentation on fast sprints / occlusion still produces multiple tracklets per person per camera, which P3/P4 must re-bind. | frags 4–12 (panel). | 06 |

Verdict: P2 is healthy. The visible flicker is **not** born here — it is P3/P4 re-labelling a
stable P2 tracklet (07).

## P3 — cross-camera association (tracklet graph + union-lift)  ⬅ **primary identity fault**

| # | Issue | Evidence | Feeds |
|---|---|---|---|
| P3-1 | **Facing-pair binding failure.** cam_01↔cam_04 (and cam_02↔cam_06, cam_03↔cam_05) are near-degenerate epipolar geometry; the ground-distance cue is the only one live and it is noisy on grazing cam_04. Whole tracklets fail to associate. | cam_01↔cam_04 = 5030 split events (#1); `M1_1_16_2` cam_04→P011 vs others→P005. | 05, 06 |
| P3-2 | **No strong facing-pair-capable cue.** colour d′≈0, bone-ratio abstains, posture weak when a view is grazing/tiny. Association leans on geometry exactly where geometry is degenerate. | panel `d_app` 0–2.7, often 0. | 05 |
| P3-3 | **Per-frame membership churn** tips a tracklet between clusters → the 2D flicker. | `cycle_cons` 0.47–0.95; low on `M2_1_12_1`, `M2_2_3_1`. | 07 |
| P3-4 | Ground position from the z=0 solve is **per-frame** and, for grazing/single-cam clusters, noisy → feeds the teleport spikes. | e_max spikes correlate with single_cam. | 04 |

Verdict: **P3 is where split identity lives.** union-lift (W9) helped but only merges
already-colocated ids; grazing-offset splits survive.

## P3.5 — binding 3D lift + chimera purity
Chimera suspects 1–7 per delivery (panel `chimera`). Splitting works but split pieces need
re-absorption or they inflate id count. `M2_2_4_4` chimera=7 is an outlier worth a look. Low
severity relative to P3/P4.

## P4a — online global id (Singer-Kalman, staged Hungarian)

| # | Issue | Evidence | Feeds |
|---|---|---|---|
| P4a-1 | **Over-mints** (P001–P024 for ~15 people) — one track per unassociated fragment/split. | `id_switch_report` shows 6+ merges on `M2_2_3_4`. | 06 |
| P4a-2 | **Concurrent-cluster claim**: one track can receive observations from two disjoint-camera clusters in one frame; only same-(camera,frame) collisions are vetoed, not same-id-two-places. | oscillating steps in `jump_classify.py`. | 04 |
| P4a-3 | **Loose chi² gate (5.991) + Singer process noise** admit a far measurement after a short gap → posterior lurches even with `emit_kalman_posterior`. | e_max 500–1500 on posterior-emit clips. | 04 |

## P4b — stitch (min-cost-flow) + colocated merge (W9)

| # | Issue | Evidence | Feeds |
|---|---|---|---|
| P4b-1 | **Gap-scaled spatial budget** (`distance ≤ v_max·gap·slack`) lets a long occlusion license a cross-field stitch → emitted step. | `stitching.py:171-176`. | 04, 06 |
| P4b-2 | **Mean-over-fragments emission** (`runner.py:348`) averages the positions of everything merged into an id per frame → oscillation when two are concurrent. | 1528 emitted big jumps. | 04 |
| P4b-3 | **Flat 0.75 m colocated-merge radius** misses grazing-offset splits (cam_04 foot 1 m off) → split id survives; 2 residual coloc pairs. | `M1_1_14_7`, `M2_1_11_3`. | 05 |

Verdict: **P4 is where teleports are manufactured** — the emission design amplifies upstream
mis-association into non-physical jumps.

## P5 — roles (v1.2)
No teleport/identity contribution. Open visual arbitration only: bowling-end orientation and
keeper pick (`wip/to_do.md` §B) — needs mosaic sign-off, not a code fix. Peripheral
suppression correctly never touches core roles.

## P6 — terminal 3D (RANSAC DLT + Butterworth)

| # | Issue | Evidence | Feeds |
|---|---|---|---|
| P6-1 | **Coverage gaps** 0.48–0.92: single-camera frames get no skeleton (by design). | `idswitch_2d.py`. | 08 |
| P6-2 | Rare residual 3D teleport on a bad cross-camera association (chimera frame). | `M2_2_3_7` pelvis max 137 m/s, 9 big jumps. | 08 |
| P6-3 | Hip keypoint cross-view inconsistency → 11–12 px reproj (worst joint). | reproj analysis. | reproj |

Verdict: P6 output is the **trustworthy positional channel** today (smooth), limited by
coverage. F16 single-view lift is the coverage lever.

## R — render / mosaic
Colour flicker = the P3/P4 id switches (07). BEV dots read the teleporting P4 ground track
(04) — the render will *show* the teleports until P4 emission is fixed. cam_07 letterboxing
handled. Not a source of error, but the surface where all the above become visible to the
reviewer.

---

## One-line causal chain
**P1 deep-field recall gap → single-camera players → (P3) no binding cue on grazing/facing
cameras → split identity + (P4a) over-mint → (P4b) risky stitch + mean-emission → emitted
teleports + flicker; (P6) drops the same frames → coverage gaps.** Fix upstream coverage and
P4 emission and the visible symptoms collapse together.
