# 06, roles

> **Stage 06** (was P5), gives each persistent player a cricket **role** (bowler / striker /
> non-striker / wicketkeeper / umpire / fielder) from their movement and the pitch geometry, then
> drops clearly low-quality peripheral detections. Code: `src/identity/p6_roles/`, config
> `configs/06_roles.yaml`. Consumes stage 05's tracks; export + render are [07](07-export-and-render.md).

---

## 1. What this stage does (and why)

Every player now has a stable id (`P007`) but no *meaning*, the system doesn't yet know which id is
the bowler and which is the umpire. Stage 06 labels each id with a **role** from its ground trajectory
and where it sits relative to the pitch (creases, stumps), then applies **role-aware peripheral
suppression**: dropping obviously bad peripheral detections (a barely-seen fielder) before the render.

Roles are consumed only by the roster panel in the mosaic and by downstream groups, **they never
change identity or geometry.** So a role mistake is cosmetic, not a tracking error.

> **In plain words:** it's the "name tags" step, turn anonymous `P007` into "the bowler" using where
> people stand and move. Getting a name tag wrong doesn't corrupt anything upstream; it just mislabels
> a box in the roster.

---

## 2. Inputs and outputs

| | |
|---|---|
| **Input** | a 05 run (`diagnostics/ground_tracks.jsonl`) + pitch calibration |
| **Output** | `roles.json` (`{roles:{Pxxx:{role,confidence,source}}, bowling_direction_xy, …}`) and `suppression.json` |
| **Core** | `src/identity/p6_roles/{assigner.py, run_role_assignment.py, suppress_peripherals.py}` |

---

## 3. How it works

### 3a. Bowling direction

Derived from the **fastest plausible early run along the pitch axis** (`infer_bowling_direction`, capped
at 9.5 m/s; pitch axis from `load_pitch_axis`), the bowler running in gives the "which end is bowling"
signal.

> **In plain words:** whoever sprints in along the pitch first is almost certainly the bowler, and that
> tells you which end is the bowling end.

### 3b. Role assignment, the epoch Hungarian solver (`assign_roles_epoched`)

For each **40-frame epoch** (a short time-window), the six roster slots, bowler, striker, non-striker,
wicketkeeper, and two umpires (bowler's-end + square-leg), are assigned to player ids by a **Hungarian
assignment** over a **geometric slot cost**: how well each player's position matches where that role
*should* be (creases ± 8.84 m, stumps ± 10.06 m).

- **Hungarian** (again, the optimal one-to-one matcher from [02](02-tracking.md)): here it matches
  *players to role slots* at lowest total geometric cost, so each role gets exactly one player and vice
  versa.
- **Epoch + latch (debounce):** deciding per short window (not per frame) and requiring a role to
  persist for `role_epoch_latch_count=3` epochs before it "sticks" prevents the label flickering
  frame-to-frame. A final greedy pass enforces uniqueness.
  > **In plain words:** decide roles over short chunks of time and only lock a label in once it's been
  > stable for a few chunks, so the roster panel doesn't strobe between "striker" and "non-striker".

### 3c. Bowling-end auto-flip (v1.2)

The pitch axis has two possible sign conventions (which end is "positive"). The solver tries **both**
and keeps the sign whose roster fits best over the pre-shot window, with the bowler's run breaking ties
(`bowling_direction_source` is recorded per delivery). Overs don't share a bowling end, so each delivery
decides independently.

### 3d. Peripheral suppression (06b), `suppress_peripherals.decide`

**Core roles (bowler / striker / non-striker / keeper) are NEVER suppressed.** Only *peripherals*
(umpire / fielder / unknown) can be dropped, and only when clearly low-quality, low keypoint
confidence, low completeness, or a weak single-camera detection.

> **In plain words:** a conservative "hide the junk detections at the edges" pass that is forbidden from
> ever touching the four central players you actually care about.

---

## 4. Config knobs (`configs/06_roles.yaml`)

`role_assignment_version: v1`, `min_track_frames: 60`, `epoch_frames: 40`, `role_epoch_latch_count: 3`,
`role_assignment_max_cost: 8.0`; suppression: `suppression_enabled: true`, `suppress_min_kp_conf: 0.35`,
`suppress_min_completeness: 0.25`, `suppress_single_cam_det_conf: 0.40`, `suppress_protect_umpires:
false`. (v1 requires the 05 run to have `online_role_proxy: true`.)

---

## 5. What's been tried (verdicts)

- **v1 epoch solver, ACCEPTED** (fixes-log W5-ROLES): core-role coverage 24/32 to 29/32, both umpires
  resolved on 6/8 deliveries, ≥ v0 everywhere.
- **v1.2 auto-flip, ADDED** (W8): removes the hardcoded bowling-end assumption.
- **W6 suppression, ACCEPTED, conservative**: 0-3 ids/clip suppressed, zero core-role suppression.
- A parallel `global_id/` rewrite contributed alongside the roles work is **parked** pending its own
  changelog + 8-delivery A/B.

---

## 6. Strengths / weaknesses

**Strengths**
- **Purely geometric + downstream-only**, a role error can't corrupt identity or 3D.
- **Epoch + latch debounce** gives stable labels without frame flicker.
- **Auto-flip** removes a brittle hardcoded assumption.
- **Suppression is conservative and core-protected** by construction.

**Weaknesses**
- **Geometry-only**, no play-state / event awareness (e.g. who actually bowled), so an unusual field
  set can mis-slot a fielder as an umpire and vice-versa.
- **Depends entirely on 05's ids**, a fragmented/teleporting id upstream produces a noisy trajectory
  the geometric solver then mis-labels.
- **Fixed slot geometry**, crease/stump offsets are constants; an unusual camera/pitch setup needs
  re-tuning.

---

## 7. Known issues (severity, 1 low to 3 high)

- **R-1 (severity 1/3) Bowling-end / keeper picks need visual arbitration**, the open items here are *visual
  sign-off only* (bowling orientation, standing-back keeper), not code bugs. No teleport or identity
  contribution from this stage.
- **R-2 (severity 1/3) No event/play-state prior**, roles are inferred from geometry alone, with no ball/over
  context to disambiguate.

---

## 8. Candidate fixes (priority-ordered)

> **Implementation status (2026-07-16):** **neither implemented**, 06 runs the v1 epoch-Hungarian role
> solver + v1.2 auto-flip + conservative W6 suppression described above (all accepted earlier). The two
> rows below (play-state prior, confidence-scaled suppression) are future.

| # | Fix | Priority | Why | Effort | Source |
|---|---|---|---|---|---|
| 1 | **Fold in play-state / event context** (ball release, over number) as a prior on the role slots. | severity 1/3 | Geometry alone can't always separate a deep fielder from an umpire; event context can. | Medium | SoccerNet game-state [2404.11335] |
| 2 | **Confidence-scaled suppression** rather than fixed thresholds, and expose the keeper standing-back band as a knob. | severity 1/3 | Fixed thresholds under-/over-suppress across cameras. | Low |, |

---

## 9. Entry-point commands

```bash
python -m identity.p6_roles.run_role_assignment \
  --input-run-dir <05_global_id> --output-run-dir <06_roles> \
  --drive-root drive --delivery-id <D> --config configs/06_roles.yaml

python -m identity.p6_roles.suppress_peripherals \
  --input-run-dir <05_global_id> --roles-path <06_roles>/roles.json \
  --output-path <06_roles>/suppression.json --config configs/06_roles.yaml
```
