# Meeting / debug reference — how the pipeline works + why the output isn't perfect

**One-page-per-section, skimmable, meeting-ready. Written 2026-07-14 for a live debug
session.** Part 1 explains every process we currently run. Part 2 is the fresh 40-delivery
diagnosis (measured on the L40S production tree). Part 3 is anticipated Q&A. Deep detail:
`docs/diagnosis/` (issue-by-issue) and `docs/critical-analysis/fixes-log.md` (every A/B).

---

# PART 1 — What the pipeline does (every process, in order)

Input: 7 synchronized cameras (`cam_01…cam_07`) film each cricket delivery — ~600 frames at
50 fps, 2560×1440 px (cam_07 is a panoramic 3775×960). Output per delivery: 2D skeletons,
per-camera tracks, one **global player ID consistent across all 7 cameras**, roles, world
3D skeletons and ground positions, and a diagnostic mosaic video.

Data lives on the L40S at `/home/ubuntu/pipetrack_v8/deliveries/<DELIVERY>/{p1b,p2,p3,p3_5,
p4,p5,p6_3d}`. 40 deliveries (M1 overs 14/16/17; M2 overs 11/12 + innings-2 overs 3/4).

| Stage | Plain-English job | Method we use | Key output |
|---|---|---|---|
| **P1** 2D pose | Find every person in every frame of every camera and estimate their skeleton | RTMDet-m person detector on a **4×2 overlapping tile grid + full frame** (small players survive), cross-tile **NMS 0.55** (both crossing players kept), fp16; then **RTMPose-X** top-down, **Halpe-26** skeleton (COCO-17 + head/neck/hip + 6 foot keypoints) | `pose_2d` (Halpe-26, 26 incl. feet) |
| **P1b** 2D stabilize | Remove per-frame keypoint jitter before it propagates | **One-Euro filter** + confidence-gated spike clamp over IoU micro-tracks | smoother 2D (−20–34% jitter) |
| **02** per-camera track | Link detections over time **within one camera** into tracklets (one person = one tracklet) | ByteTrack-style two-stage Hungarian (IoU + pose-cosine) + constant-velocity Kalman; low-confidence detections associate-only (no new births); zero-IoU fast movers matched by motion | `local_track_id` |
| **03** cross-camera ID | Decide which tracklet in camera A is the **same person** as which in camera B/C/… | Whole-delivery **tracklet graph**: per-pair calibrated log-likelihood-ratio cues (ground distance on the z=0 plane, appearance, billboard posture, motion) fused → union-find merge + facing-pair corroboration; ground position by **z=0-constrained robust reprojection** (Gauss–Newton + Huber); **W9 union-lift merge** (two clusters merge if ONE triangulated skeleton explains all views) | `correspondences.jsonl`, per-cluster world `ground_xy`, binding ids |
| **04 (binding lift)** binding 3D lift | Triangulate a full skeleton per binding + check for "chimeras" (two people fused) | per-binding RANSAC-DLT triangulation, per-joint 3D covariance, one-sided reprojection bias names an intruding camera | `lift3d.jsonl`, `lift_purity.json` |
| **05a** global ID (online) | Turn per-frame clusters into **persistent IDs** that live the whole clip | **Singer-acceleration ground Kalman** per player + staged assignment (binding → exact tracklet → χ²-gated Hungarian → re-entry); emits the **Kalman posterior** ground position | `global_player_id`, `ground_tracks.jsonl` |
| **05b** stitch + merge | Join fragments of the same person; merge a player who got two IDs | **min-cost-flow** fragment stitching (temporal + spatial + role + velocity cost; never merges two IDs that share a camera-frame) + **W9 colocated-id merge** (two IDs within 0.75 m in disjoint cameras for ≥25 frames = one person) | `id_switch_report.json` |
| **06** roles | Label each ID: bowler / striker / non-striker / keeper / 2 umpires / fielder | Role solver v1.2 — 6 Hungarian slots with distinct geometry, latch + uniqueness, **per-delivery bowling-end auto-flip**; peripheral suppression (never suppresses the 4 core roles) | `roles.json`, `suppression.json` |
| **R** render | Diagnostic mosaic: 7 cameras + bird's-eye view + roster, coloured by global ID | calibration-derived compositing; colour = global ID (colour flicker = an ID switch) | the mp4 |

**Rig geometry** (for questions): world origin = pitch centre, +Y toward the far end, z=0 =
ground/turf; stump mid-bases at (0, ±10.08 m). cam_01/cam_04 are the **end-on** pair looking
down the pitch; cam_02/03 east, cam_05/06 west; cam_07 oblique panoramic. **Facing (co-observing)
pairs: cam_01↔cam_04, cam_02↔cam_06, cam_03↔cam_05** — these look head-on at each other, so their
epipolar geometry is near-degenerate (this matters a lot below). Calibration is
centimetre-accurate (ball reprojection p95 ≤ 4.5 px), team-confirmed one session for both matches.

**How we evaluate** (the panel, `final_panel.md`): cross-camera **agreement** (do close
cross-camera detections share an ID), distinct **IDs** vs the ~13–15 roster, **teleports**
(ID jumps faster than a human), **id_persist**, **collisions** (same camera, one ID on two
people — must be 0, and is 0 everywhere), **coloc** (split-ID tripwire), 3D **reproj** px, 3D
**coverage**. Verdict pass/warn/fail is a rule over these.

---

# PART 2 — What's actually wrong (measured on all 40, 2026-07-14)

Full evidence + scripts: `docs/diagnosis/`. Headline numbers were measured on the real
output, not read off the panel.

### 2.1 The panel says "fail" on 27/40 — but that verdict is misleading
Every `fail` is the single rule **`teleport_event_count > 60`**. The ID-overmint rule never
fires (roster_max=15, we mint at most 16). And the teleport metric runs on **raw
bbox-bottom foot projections averaged across cameras**, *not* on our emitted track — so it is
dominated by grazing-angle foot noise on single-camera players. Proof: `M2_2_4_1` scores
agreement **0.992** (identity essentially perfect) yet 158 teleports → `fail`. **So "27 fails"
overstates how broken we are.** (`docs/diagnosis/03-...`)

### 2.2 …but the emitted ground track *does* genuinely teleport (the user is right)
The delivered `ground_tracks.jsonl` has **1528 non-physical single-frame jumps (>25 m/s)**
across the 40, peaks of 140–1500 m/s. Cause: after we stitch/merge IDs we emit the **mean of
all fragment positions** for that ID in a frame (`runner.py:348`); when one ID briefly holds
two observations (concurrent disjoint cameras, or a cross-field stitch), the emitted point
oscillates between them. **This is a 05 emission bug, fixable.** (`docs/diagnosis/04-...`)

### 2.3 The 3D skeletons (04 lift) are smooth but sparse
The triangulated 3D output — our actual 3D deliverable — is clean (pelvis p95 1.6–3.8 m/s,
~0 big jumps) because it only triangulates where ≥2 cameras agree. The price is **coverage
0.48–0.92 (median 0.80)**: single-camera players get no 3D. So today the **3D skeleton is the
trustworthy positional channel; the flat ground dot is not.** (`docs/diagnosis/08-...`)

### 2.4 Cross-camera split identity is real (same person, different ID in one camera)
19–87 % of multi-camera ground clusters carry >1 ID. The odd-camera-out is systematically
the **hard camera: cam_04 (end-on, grazing) and cam_07 (panoramic)**. Concrete:
`M1_1_16_2` — cam_04 calls a player P011/P013 while cam_01/02/06 all call him P005. Why: on
the facing pair cam_01↔cam_04 the epipolar geometry is degenerate, kit colour is dead
(d′≈0), bone-ratios abstain — so only the **ground-distance** cue is live, and it is exactly
the cue that's noisy on a grazing camera. The hard cameras have no strong binding cue.
(`docs/diagnosis/05-...`)

### 2.5 Visible ID-switch flicker exists but is modest
A stable per-camera tracklet flips its global ID **517 times total across 40** (~13/delivery);
~5.5 tracklets/delivery flicker. It's the same root as split identity — 03/05 re-decides
membership per frame instead of locking it per tracklet. (`docs/diagnosis/07-...`)

### 2.6 "Many IDs" is mostly a metric non-issue, but leaves scars
Final IDs are 9–16 vs roster 15 — in range. But internally 05a **over-mints** (e.g. P001–P024
for ~15 people) and stitches down, and **each stitch seam is a teleport risk**.
(`docs/diagnosis/06-...`)

### 2.7 The one structural fact behind all of it
**Everything scales with the single-camera fraction.** A single-camera player can't be
triangulated (no 3D), has a noisy foot position (teleport), and can't be cross-checked
(split ID / over-mint). Second-innings deep-field overs `M2_2_3_*` run single-cam 0.76–0.82
and are worst on every axis; clean first overs `M1_1_14_*` run 0.39–0.65 and behave. **This is
a detection/coverage problem first, an identity-algorithm problem second.**

### Where it works / partially works / fails (grades: `docs/diagnosis/02-...`)
- **Cleanest**: `M1_1_16_4` (0 emitted big jumps, agreement 0.951, coverage 0.91),
  `M1_1_17_2`, `M1_1_14_1`.
- **Core-good, periphery-noisy (most deliveries)**: batsman/bowler/keeper solid; deep
  fielders/umpires split or foot-teleport. Usable with the `single_camera`/low-coverage flags.
- **Weak**: `M2_2_3_*` (all 7), `M2_1_11_7`, `M2_1_12_1`, `M1_1_14_5/6`, `M1_1_16_2` — high
  single-cam, dense field, deep fielders.

---

# PART 3 — Anticipated meeting questions

**Q: Why are so many deliveries marked "fail"?**
A: The verdict is driven by one over-sensitive metric (teleports on raw foot projections). It
cries wolf on single-camera players. Real identity on those clips is often 0.9+. We're fixing
the metric to measure the emitted track on multi-camera segments only. (2.1)

**Q: But you admit it teleports — is the delivered data usable?**
A: The **3D skeletons are smooth and usable now**; the flat ground dot has ~1500 real jumps
we've root-caused to a mean-over-fragments emission step — a targeted 05 fix, not a redesign.
(2.2, 2.3)

**Q: Same player has different IDs in different cameras — why?**
A: The two "facing" cameras (esp. cam_01↔cam_04, end-on) have degenerate geometry, and our
only strong ID cue there — ground distance — is noisy because a grazing camera's foot
projection is ~1 m off. Colour and body-proportion cues are dead on identical-kit footage. So
the hard camera's tracklet doesn't bind and gets its own ID. (2.4)

**Q: What's the single biggest lever?**
A: **Multi-camera coverage on the deep field.** Better small-player detection + a single-view
3D lift (F16) converts single-camera players into checkable, triangulatable ones — that
simultaneously fixes teleports, split IDs, and 3D coverage. (2.7)

**Q: Is calibration the problem?**
A: No — it's cm-accurate (ball reproj p95 ≤ 4.5 px), flat 3.07–3.56 px across all 40, team
confirmed one session. Camera spread is even. The error floor is 2D pose-model noise (2–3 px,
hips worst), not calibration. (`wip/to_do.md` §H)

**Q: Are we minting garbage IDs / colliding people?**
A: Same-camera collisions are **0 everywhere** (hard invariant held). Final ID counts are in
roster range. The internal over-mint is cleaned by stitching. (2.6)

**Q: What are the immediate fixes?**
A: (1) Fix 05 emission (drop mean-over-fragments, velocity-gate, damp single-cam) → kills the
visible teleports. (2) Fix the verdict/teleport metric → stop mislabeling 27 deliveries.
(3) Attack split ID at cam_04/cam_07 (depth-aware association + F16 lift). Prioritized list
with code pointers: `wip/to_do.md`.
