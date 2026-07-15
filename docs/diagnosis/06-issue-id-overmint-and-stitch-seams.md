# Issue 06 — ID over-minting and stitch seams ("many IDs")

**Severity: medium (mostly resolved at the count level, but leaves teleport scars).**

## Symptom
The user reports "many IDs". The *final* distinct-id count is actually in range: 9–16 against
a roster max of 15 (panel `ids` column). So the delivered id count is not the headline
problem. The real issue is **internal over-minting then stitching**, which is what produces
the teleport seams in `04-...`.

## Evidence of over-mint → stitch
`p4/id_switch_report.json` records the merges P4b/W9 applied. Example `M2_2_3_4`:
```
P017 -> P012,  P019 -> P016,  P020 -> P012,  P021 -> P012,  P023 -> P014,  P024 -> P014
```
So P4a minted at least P001–P024 (24 online tracks) for a field of ~15, and the stitcher +
colocated-merge collapsed them to 16. Every delivery shows 1–12 such merges. The min-cost
flow "links selected" going from 0–1 (baseline) to 3–6 (v7) was celebrated as the stitcher
finally being *live* — but a live stitcher that merges spatially-distant fragments is exactly
what plants the emitted oscillations.

## Why P4a over-mints
- **Split identity (05)** — every cam_04/cam_07 tracklet that fails to associate becomes a
  fresh global track.
- **Fragmentation** — a per-camera tracklet that breaks (occlusion, fast sprint, missed
  detection) yields a new cluster → new track; P4b is meant to re-stitch it.
- **Single-camera peripherals** — deep fielders/umpires seen by one camera enter as their own
  tracks; the synthetic-tracklet machinery (`syn_min_confidence`) deliberately mints these to
  avoid rival-id churn, but they are exactly the ones that teleport.

## Why the stitch-down is double-edged
`stitching.py:build_link_costs` links `source.end < target.start` (temporally disjoint) when
kinematically reachable. Two regimes:
- **Good stitch**: same player, short gap, spatially continuous seam → id count drops, no
  jump (e.g. `M1_1_16_4`: 3 merges, 0 big jumps).
- **Bad stitch**: two fragments whose seam spans a big spatial gap (allowed because
  `distance ≤ v_max·gap·slack` grows with gap) → an emitted **step** at the join. Or a
  concurrent-in-disjoint-cameras merge → an emitted **oscillation** (`04-...`).
The `id_switch_report` `at_frame` marks only the join frame; the positional consequence spans
the whole overlap, which is why `jump_classify.py` finds only ~15–20 % of steps *near* a
recorded merge frame — the rest are the ongoing oscillation the merge set up.

## Interaction with `min_emit_frames` drop
`runner.py:318-337` drops any final id whose total span < `min_emit_frames`. This keeps the
count honest but can **delete a short real fragment** (a briefly-seen fielder) rather than
stitch it — a silent recall loss on the periphery. Worth auditing which ids get dropped on the
C-band clips.

## Fix direction (see ../../wip/to_do.md items A3, A6)
- Reduce over-mint at the source by fixing split identity (05), so P4b has fewer fragments to
  gamble on.
- Make the stitcher **conservative about spatial gaps** (absolute-metres cap, not
  gap-scaled), preferring to leave a fragment un-stitched (honest extra id) over planting a
  teleport.
- Emit a **stitch-quality flag** per merge (seam distance, overlap) so downstream can trust or
  discard a stitched segment.
