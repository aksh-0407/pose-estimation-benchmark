# docs/diagnosis — 40-delivery production diagnosis (2026-07-14)

A from-scratch, **measured** diagnosis of the v8.1 production tree
(`/home/ubuntu/pipetrack_v8/`, 40 deliveries) answering: many IDs, ID switching, non-smooth
output, teleports, cross-camera ID splits. Every number was measured on the real output by
the scripts checked in here (run on the L40S, env `cricket-rtmpose-l`).

**Meeting-ready single-file summary** (pipeline processes + diagnosis + Q&A):
`../critical-analysis/meeting-debug-reference.md`.
**Prioritized fixes**: `../../wip/to_do.md`.

## Read order
| File | What |
|---|---|
| `00-executive-summary.md` | The headline findings and the two failure modes. Start here. |
| `01-methodology-and-measurements.md` | How everything was measured + the full raw tables. |
| `02-per-delivery-categorization.md` | Works / partially works / fails, graded per delivery. |
| `03-issue-teleport-metric-and-verdict.md` | Why 27/40 read `fail` and why that's misleading. |
| `04-issue-emitted-ground-teleports.md` | The **real** teleports in `ground_tracks.jsonl` and their code cause. |
| `05-issue-cross-camera-split-identity.md` | Same person, different ID per camera (cam_04/cam_07). |
| `06-issue-id-overmint-and-stitch-seams.md` | "Many IDs" — internal over-mint then stitch. |
| `07-issue-2d-id-switch-flicker.md` | Mosaic colour flicker (per-tracklet id flips). |
| `08-issue-3d-coverage-gaps.md` | 3D is smooth but sparse (coverage). |
| `09-per-phase-issue-register.md` | Every phase P1→R, its issues, and the downstream symptom. |

## Reproduce
Scripts (copy to the box `/tmp`, run under `cricket-rtmpose-l`):
`emit_smoothness.py`, `jump_classify.py`, `occupancy_and_3d.py`, `idswitch_2d.py`,
`split_identity.py` (this one runs from `~/pose-estimation-benchmark` to import the repo's
calibration).

## Three-sentence summary
The panel's 27 `fail`s are a metric artifact (teleport proxy on raw foot projections), **but**
the emitted ground track genuinely teleports (1528 non-physical jumps, from a
mean-over-fragments emission bug) while the 3D skeletons are smooth-but-sparse; cross-camera
split identity is real and concentrated on the geometrically hard cameras (cam_04 end-on,
cam_07 panoramic). Everything scales with the **single-camera fraction** — it's a
detection/coverage problem first, an identity-algorithm problem second. Fixes, cheap-first,
are in `../../wip/to_do.md`.
