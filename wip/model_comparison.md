# P1 model comparison — RTMPose-L vs RTMPose-X (and detector findings)

Written 2026-07-14 to close the long-open study item. Sources: archived run docs in
`docs/runs/` (rtmpose-l-body8-full-db32-pb96, rtmpose-x, yolo26x-pose-full-db8, bakeoff_w5),
`docs/critical-analysis/fixes-log.md` (W5/W5B-LIVE entries), meeting/production panels.

## Pose model: RTMPose-X (Halpe-26) is the accepted choice

| Aspect | RTMPose-L (body8, COCO-17) | RTMPose-X (body8, Halpe-26) |
|---|---|---|
| Skeleton | 17 joints, no feet | 26 joints: COCO-17 + head/neck/pelvis + 6 foot points |
| Ground contact | ankle-based only | heel/toe landmarks → `foot_contact_mode: v3` (F4) |
| Throughput (L40S, plain 640 det) | ~30+ fps | 27.5 fps (134k frames / 82 min) |
| Identity-era baseline | v5 stack: `_7` agreement 0.498→0.600 | v6.0 ground baseline onward: all campaign gains built on X |

Decision drivers (in order):
1. **Feet.** The entire ground-position channel (z=0 reprojection solve, F4 heel/toe contact,
   billboard posture anchoring) improves with real foot landmarks. Only the X body8 model
   ships Halpe-26; there is no COCO-17-only reason to stay on L.
2. **Accuracy-first mandate**: X is the accuracy flagship; the throughput delta (~10%) is
   irrelevant next to identity quality on this project.
3. The per-metric L-vs-X ablation on identical downstream configs was **never run in
   isolation** — the v5(L) → v6.0(X) jump changed model and campaign era together. If a clean
   ablation is ever wanted: run the v8.1 chain on an L-based P1 tree and diff the panel. Not
   currently justified — every accepted result since v6.0 is X-based.

RTMO (one-stage) was evaluated on paper and **rejected**: COCO-17-only heads would lose the
feet (fixes-log W5 research note).

## Detector findings (the part that actually moved the needle)

The detector, not the pose model, was the upstream bottleneck:
- **YOLO26x-pose** (`yolo26x-pose-full-db8`, kept in `data/derived/runs/`): historical
  comparison run; not adopted (RTMPose mandate + top-down pipeline).
- **Bake-off (docs/runs/bakeoff_w5/)**: tiled RTMDet-m @640 beat native hi-res decisively —
  RTMDet only detects at its trained object scale (m1280/m2560 lost boxes; t640 was a strict
  superset, min box height 33→12 px). RTMDet-L @1280 marginal.
- **NMS 0.55** (from 0.3) lets both crossing players survive: +0.10–0.13 cross-camera
  agreement — the single largest identity gain of the campaign.
- Accepted production P1: tiled RTMDet-m 4×2 + full frame, NMS 0.55, IoM 0.7, fp16 fast path
  (worker-side crop prep): **18–25 fps for 9× detector work** on the L40S.

## Open follow-ups
Queued in `remaining-work.md` §3.3: YOLO26-l / RF-DETR recall-oracle probes through the same
bake-off harness; clean L-vs-X ablation only if someone needs the number.
