# Handover notes — operational, infrastructure, deferred

Internal operational context for the hand-over: box/infra layout, consumer/schema state, and the items
the user explicitly deferred. The forward-looking algorithm roadmap moved to
[`docs/roadmap.md`](../docs/roadmap.md); the pre-hand-over bug-fix and cleanup plan is
[`remediation_plan.md`](remediation_plan.md).

## Box and infrastructure

- The box home is `/home/ubuntu`; repo `~/pipetrack`, data `~/bits-pose-data`, env
  `~/miniconda3/envs/pose-lab/bin/python`.
- The 2026-07-16/17 A/B trees under `~/bits-pose-data/derived/40_full/`: `pipetrack_v90` (P1 only),
  `pipetrack_v91_base` (40-set baseline), `pipetrack_v91_{shape,split,facing,R,lostwin}_off` (flag
  toggles), `pipetrack_tiledAB`, `pipetrack_nms055AB`, `pipetrack_tiled03` (detector), `pipetrack_v91_ocsort`
  (tracker). A/B harness and helper scripts in `~/ab_work/`.
- The box clone was synced to pipetrack_v9 (origin/main); the prior uncommitted state is in a
  git stash `box-local-presync-pipetrack_v9` and three session-created files are backed up under
  `~/pipetrack_presync_backup/`.
- Physical consolidation (deferred, needs sign-off): old large trees still sit in the box home referenced
  by symlinks. Any move is copy, verify, then delete, never `mv`.

## Consumer, schema, output

- `g1_player_frame/v1` (Halpe-26) is consumer-facing and accepted by the character team; `pose_3d` is
  per-joint nullable. Coordinate any further schema change with the biomechanics and officiating teams.
- UE packet export runs per delivery when UE-cm packets are needed.
- All-40 mosaic batch renders on demand.
- Mosaic sign-off still pending: arbitrate roles end-orientation and keeper-pick ambiguity; spot-check
  more clips.

## Explicitly deferred by the user

- Laptop conda recovery of the `balltrack` / `quadruped` envs.
- The `.git` bloat (committed mosaics and docx in history) is accepted; no history rewrite.
