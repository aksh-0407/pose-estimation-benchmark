# P5 roles, UE export, and the mosaic render

Three consumer stages that turn identified 3D poses into deliverables: **roles** (P5), the
**Unreal Engine** pose packets (a parallel production branch), and the **mosaic / bird's-eye
render** (the diagnostic video that is the visible end product).

## P5 — role assignment

**Role & intuition.** Classify each global player as bowler / striker / non-striker / keeper /
umpire / fielder from ground geometry relative to the pitch axis and bowling direction. The
mosaic roster reads roles only from `p5/roles.json`.

**Method — `scripts/roles/{assigner,run_role_assignment}.py`.** Uses `load_pitch_axis` /
`infer_bowling_direction` (from `mosaic_layout`) and each player's ground track to assign a role
by position/motion relative to the pitch. Runs **after** P4.

**Pros.** Simple, calibration-grounded, decoupled (the render depends only on the JSON artifact).

**Cons / issues.**
- **P5-1 (★) Roles are computed after P4 and never fed back.** The Singer Kalman supports
  role-aware dynamics (a bowler is agile, an umpire static), but because P5 runs last, P4 tracks
  every player as `unknown` (`wip/3d_location_issues.md` ISSUE-11). The role information that would
  improve tracking is produced too late to be used.
- **P5-2 (★) Geometry-only roles** are brittle at role transitions (a fielder walking through the
  pitch region) with no temporal smoothing or team-sheet prior.

**Fixes (priority-ordered).**

| # | Fix | Priority | Reasoning | Effect | Source |
|---|---|---|---|---|---|
| 1 | **Feed a role (or an online velocity-based role proxy) into P4a's `switch_role`** so the Kalman uses role-aware manoeuvrability during tracking. | ★★ | The dynamics support it; only the sequencing withholds it. Bowler/umpire dynamics differ hugely. | Better prediction → fewer teleports/fragments. | multi-task role+ReID [2401.09942] |
| 2 | **Temporal smoothing + team-sheet prior** on role labels. | ★ | Stabilises transitions; cricket rosters are known. | Fewer role flips in the roster panel. | game-state priors [2404.11335] |

## Mosaic / bird's-eye render

**Role & intuition.** Composite the seven calibration-ordered camera tiles + a **bird's-eye ground
monitor** + a **team roster** into one video, overlaying skeletons coloured by stable global ID,
the ball trail, roles, and occlusion "ghost" markers. This is where identity errors become
visible — a swapping colour is an ID switch.

**Method — `scripts/visualization/render_phase1_videos.py` (+ `mosaic_layout.py`,
`identity_colors.py`).** The tile layout is **derived from calibration** (no hardcoded camera IDs):
`derive_mosaic_layout` places columns/rows and mirrors by camera look-direction, with bottom-row
monitor + roster slots. It reads P4 `predictions` (poses + IDs), P3 `correspondences.jsonl`
(cluster badges), P4 `ground_tracks.jsonl` (the BEV monitor + occlusion ghosts), `p5/roles.json`,
raw frames, calibration (for ghost foot-projection), and ball events. Stable ID colours come from
`color_for_global_id`. Encoding is `h264_nvenc` with an `mp4v` fallback.

**Pros.**
- **Calibration-derived layout** — robust to camera changes, no magic per-rig constants.
- **Stable per-ID colour** makes identity errors immediately legible (the whole point of a
  diagnostic render).
- **Rich overlay** — BEV monitor, ghosts for occluded players, ball trail, roster — a genuinely
  useful debugging surface.
- **Decoupled inputs** — reads artifacts, so any stage can be swapped and re-rendered.

**Cons / issues.**
- **R-1 (★) Camera-07 aspect** — the heterogeneous ~3775×960 tile can distort in a grid built for
  2560×1440; verify the layout scales per-camera size.
- **R-2 (★) Render is the wall-clock bottleneck** (~7–8 fps, render-bound not pose-bound), which
  slows the iterate-measure loop.
- **R-3 (★) Ghost-marker correctness** depends on the calibration foot-projection; a wrong ghost
  can mislead debugging.

**Fixes (priority-ordered).**

| # | Fix | Priority | Reasoning | Effect | Source |
|---|---|---|---|---|---|
| 1 | **Per-camera aspect-correct tiles** (use the native size for C07). | ★ | Avoids a distorted/ misleading tile for the one heterogeneous camera. | Correct C07 overlay. | — |
| 2 | **Speed the render** (batch NVENC, lower preview resolution for the iterate loop). | ★ | The render bounds the measure loop; faster renders = faster iteration. | Faster A/B cycles. | — |
| 3 | **A minimap "game-state" view** (players + roles + ball on a top-down pitch, à la SoccerNet GSR) as the primary QA surface. | ★ | A clean minimap surfaces identity/location errors better than seven tiles. | Faster error triage. | SoccerNet GSR [2404.11335] |

## UE export

**Method — `scripts/export/export_ue_packets.py`** converts triangulated 3D JSONL into Unreal
Engine pose packets tagged with a model version. It is a parallel production branch (it does not
feed the mosaic).

**Pros.** Clean, versioned hand-off; decoupled from the render.
**Cons / issues.** **UE-1 (★)** it consumes the terminal P6 triangulation; if triangulation moves
to P3.5 (recommended), the export should read the P3.5/P4 3D poses instead — a one-line input
change to keep the production branch on the improved 3D.
