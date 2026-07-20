# docs/analysis: repository audit & 40-delivery diagnosis

The measured analytical study of the pipeline — the combined audit + diagnosis of where the system is
strong, where it is weak, and why. A from-scratch, measured diagnosis of the v8.1 production tree (40
deliveries) answering: many IDs, ID switching, non-smooth output, teleports, cross-camera ID splits.
Every number was measured on the real output by the scripts checked in here.

Current state after pipetrack_v9 and the 2026-07-16 to 2026-07-17 A/B session:
`11-v9-state-and-2026-07-session.md`. Start there for what is current; files 00 to 10 are the original
2026-07-14 diagnosis and remain valid as the measured root-cause analysis.

Companion docs:
- Method ledger with every A/B, pros and cons: `../methods_log.md`.
- What to try next (the improvement directions these findings motivate): `../roadmap.md`.
- The concrete code-defect register and pre-hand-over cleanup plan are tracked internally (not in the
  hand-over docs): see the repo's `wip/` working notes.

## Read order
| File | What |
|---|---|
| `00-executive-summary.md` | The headline findings and the two failure modes. Start here. |
| `01-methodology-and-measurements.md` | How everything was measured + the full raw tables. |
| `02-per-delivery-categorization.md` | Works / partially works / fails, graded per delivery. |
| `03-issue-teleport-metric-and-verdict.md` | Why 27/40 read `fail` and why that's misleading. |
| `04-issue-emitted-ground-teleports.md` | The **real** teleports in `ground_tracks.jsonl` and their code cause. |
| `05-issue-cross-camera-split-identity.md` | Same person, different ID per camera (cam_04/cam_07). |
| `06-issue-id-overmint-and-stitch-seams.md` | "Many IDs", internal over-mint then stitch. |
| `07-issue-2d-id-switch-flicker.md` | Mosaic colour flicker (per-tracklet id flips). |
| `08-issue-3d-coverage-gaps.md` | 3D is smooth but sparse (coverage). |
| `09-per-phase-issue-register.md` | Every phase P1 to render, its issues, and the downstream symptom. |
| `10-verdict-redesign.md` | The verdict-metric redesign proposal. |
| `11-v9-state-and-2026-07-session.md` | Current state after v9 plus every method measured in the 2026-07 A/B session. Read for what is current. |

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
cam_07 panoramic). Everything scales with the **single-camera fraction**, it's a
detection/coverage problem first, an identity-algorithm problem second. Fixes, cheap-first,
are in `../roadmap.md`.
