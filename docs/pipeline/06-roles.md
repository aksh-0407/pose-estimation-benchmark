# 06 — roles

> **Stage 06** (was P5) — code `src/identity/p6_roles/`, config `configs/06_roles.yaml`.
> Consumes stage 05's fused ground tracks; export + render are [07](07-export-and-render.md).

## Role & intuition

Assign each persistent `global_player_id` a cricket **role** (bowler / striker / non_striker /
wicketkeeper / umpire / fielder / unknown) from its ground trajectory and the pitch geometry,
then apply role-aware **peripheral suppression** (06b, Wave-6) to drop clearly low-quality
peripheral detections before the terminal 3D lift and render. Roles are consumed only by the
roster panel in the mosaic and by downstream groups — they never change identity or geometry.

## Inputs → outputs

| | |
|---|---|
| **Input** | 05 run dir (`diagnostics/ground_tracks.jsonl`) + pitch calibration |
| **Output** | `roles.json` (`{roles:{Pxxx:{role,confidence,source}}, bowling_direction_xy, bowling_direction_source}`) and `suppression.json` |
| **Core** | `src/identity/p6_roles/{assigner.py, run_role_assignment.py, suppress_peripherals.py}` |

## Method as implemented now

- **Bowling direction** is derived from the fastest plausible early run along the pitch axis
  (`mosaic_layout.infer_bowling_direction`, band ≤ 9.5 m/s; pitch axis from `load_pitch_axis`).
- **v1 epoch solver** (`assigner.assign_roles_epoched`): per 40-frame epoch, a Hungarian
  assignment over six roster slots — bowler, striker, non_striker, wicketkeeper, and two umpires
  (bowler's-end + square-leg) — with a geometric slot cost (creases ±8.84 m, stumps ±10.06 m),
  a latch-count debounce (`role_epoch_latch_count`), and a final greedy uniqueness pass.
- **v1.2 bowling-end auto-flip** (`run_role_assignment.py`): solve both axis signs and keep the
  sign whose roster fits best on the pre-shot window; run detection breaks ties
  (`bowling_direction_source` is recorded per delivery). Overs do not share a bowling end, so
  each delivery decides independently.
- **06b peripheral suppression** (`suppress_peripherals.decide`): core roles are **never**
  suppressed; only peripherals (umpire/fielder/unknown) are dropped, and only when clearly
  low-quality (low keypoint confidence / completeness / single-camera detection confidence).

## Config knobs (`configs/06_roles.yaml`)

`role_assignment_version: v1`, `min_track_frames: 60`, `epoch_frames: 40`,
`role_epoch_latch_count: 3`, `role_assignment_max_cost: 8.0`; suppression:
`suppression_enabled: true`, `suppress_min_kp_conf: 0.35`, `suppress_min_completeness: 0.25`,
`suppress_single_cam_det_conf: 0.40`, `suppress_protect_umpires: false`. v1 requires the 05 run
to have `online_role_proxy: true`.

## What's been tried

- **v1 epoch solver — accepted** (fixes-log W5-ROLES): core-role coverage 24/32 → 29/32, both
  umpires resolved on 6/8 deliveries, ≥ v0 everywhere. Colleague-contributed defects fixed
  (uniqueness latch, standing-back keeper zero-cost band, crease anchors, two-umpire roster).
- **v1.2 auto-flip — added** (W8): removes the hardcoded bowling-end assumption.
- **W6 suppression — accepted, conservative**: 0–3 IDs/clip suppressed, zero core-role suppression.
- A parallel `global_id/` rewrite (contributed alongside the roles work) is **parked** pending its
  own changelog + 8-delivery A/B ([`wip/to_do.md`](../../wip/to_do.md) §B).

## Current issues & measured state

No teleport or identity contribution (`../diagnosis/09-per-phase-issue-register.md`, P5). The
open items are **visual arbitration only** — bowling-end orientation (visually confirmed on `_2`;
spot-check more) and the keeper pick — and need mosaic sign-off, not code changes
([`wip/to_do.md`](../../wip/to_do.md) §B).

## Entry-point commands

```bash
python -m identity.p6_roles.run_role_assignment \
  --input-run-dir <05_global_id> --output-run-dir <06_roles> \
  --drive-root drive --delivery-id <D> --config configs/06_roles.yaml

python -m identity.p6_roles.suppress_peripherals \
  --input-run-dir <05_global_id> --roles-path <06_roles>/roles.json \
  --output-path <06_roles>/suppression.json --config configs/06_roles.yaml
```
