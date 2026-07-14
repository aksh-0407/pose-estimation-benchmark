# Archived benchmark runs

Historical run trees were documented here and their bulk data deleted (2026-07-14 cleanup). Full analytical narrative: docs/critical-analysis/fixes-log.md.

| run | purpose | verdict | analysis pointer |
|---|---|---|---|
| [_v8_nospawn_probe](_v8_nospawn_probe.md) | lowconf_can_spawn=false probe (_5,_6,_7,M2). | Adopted into v8.0 (strict improvement) | fixes-log GRAND ANALYSIS v2 |
| [_w5b_id_check](_w5b_id_check.md) | W5b flags-off byte-identity check tree (M2). | Diagnostic only (identity proven) | fixes-log W5B |
| [bakeoff_w5](bakeoff_w5.md) | Detector-only recall bake-off (5 candidates, _7+M2 sampled). | t640 tiled won; native hi-res dead | fixes-log W5 |
| [pipetrack_v3](pipetrack_v3.md) | Historical identity stack (pre-campaign, v3 era). | Superseded | wip/methods_log.md |
| [pipetrack_v5](pipetrack_v5.md) | Validated v5 identity stack on RTMPose-L data. | Superseded | wip/methods_log.md ID-0..6 |
| [pipetrack_v6.0](pipetrack_v6.0.md) | Frozen ground baseline of the fix campaign (v5 configs, RTMPose-X P1). | Baseline (superseded by v8.0) | fixes-log F0 |
| pipetrack_v6.0.zip | Zip archive of pipetrack_v6.0. | Redundant archive | fixes-log F0 |
| [pipetrack_v6.1-f01](pipetrack_v6.1-f01.md) | Wave-0 A/B: P1.5 stabilization wired (F1). | Accepted into v7 lineage | fixes-log F1 |
| [pipetrack_v6.1-wave1](pipetrack_v6.1-wave1.md) | Wave-1 correctness batch (F3-F8). | Accepted into v7 lineage | fixes-log W1 |
| [pipetrack_v6.2-wave3](pipetrack_v6.2-wave3.md) | Wave-3 stack (F9a covariance, F10 R, F11 shape, F12 posture stitch). | Accepted into v7 lineage | fixes-log W3 |
| [pipetrack_v6.2-wave3b](pipetrack_v6.2-wave3b.md) | Wave-3b asymmetric-R refinement. | Accepted into v7 lineage | fixes-log W3b/W4 |
| [pipetrack_v6.3-wave4](pipetrack_v6.3-wave4.md) | Wave-4 chimera splitting (F13). | Accepted into v7 lineage | fixes-log W3b/W4 |
| [pipetrack_v7-ablA](pipetrack_v7-ablA.md) | Wave ablation helper tree. | Diagnostic only | fixes-log W3b/W4 |
| [pipetrack_v7-rc1](pipetrack_v7-rc1.md) | First composed v7 release candidate. | Rejected (H3 binding collapse; root-caused) | fixes-log v7-rc1 |
| [pipetrack_v7-rc2](pipetrack_v7-rc2.md) | Re-composed v7 on fixed code; stitcher live first time. | Accepted as v7 default (superseded by v8.0) | fixes-log v7-rc2 + GRAND ANALYSIS |
| [pipetrack_v7-rc3](pipetrack_v7-rc3.md) | P1.5 isolation (no stabilization). | Rejected (worse worst-clip floor) | fixes-log GRAND ANALYSIS |
| [pipetrack_v7-w5b](pipetrack_v7-w5b.md) | Contested-camera weighting composed A/B. | No-op proven (P1 NMS 0.3 caps same-cam IoU) | fixes-log W5B |
| [pipetrack_v8-nms55only](pipetrack_v8-nms55only.md) | Tiled NMS-0.55 ablation without contested (_7+M2). | Winner; became v8 detection spec | fixes-log W5B-LIVE |
| [pipetrack_v8-nms55w5b](pipetrack_v8-nms55w5b.md) | Tiled NMS-0.55 + contested-0.30 (_7+M2). | Contested rejected (-0.08 agreement) | fixes-log W5B-LIVE |
| [pipetrack_v8-probe](pipetrack_v8-probe.md) | Phase C: tiled NMS-0.3 P1 through v7 stack (_7+M2). | Hold verdict; superseded by nms55 | fixes-log W5-C |
| [pipetrack_v8-rc1](pipetrack_v8-rc1.md) | Composed v8 candidate: tiled+NMS55 x8, v7 stack, W6. | Superseded by v8.0 (adds no-spawn) | fixes-log GRAND ANALYSIS v2 |
| [pipetrack_v8.0](pipetrack_v8.0.md) | ACCEPTED v8.0 default tree (KEPT). | Current default | fixes-log GRAND ANALYSIS v2 |
| [rtmpose-l-body8-full-db32-pb96](rtmpose-l-body8-full-db32-pb96.md) | RTMPose-L body8 P1 full run. | Rejected (X chosen for Halpe-26 accuracy) | wip/model_comparison.md |
| [rtmpose-x](rtmpose-x.md) | RTMPose-X P1 at plain 640 detection (8 deliveries). | Superseded by tiled-w5-full | fixes-log F0/W5 |
| [rtmpose-x-tiled-nms55](rtmpose-x-tiled-nms55.md) | Tiled NMS-0.55 P1 probe (_7+M2). | Superseded by tiled-w5-full | fixes-log W5B-LIVE |
| [rtmpose-x-tiled-w5](rtmpose-x-tiled-w5.md) | Tiled NMS-0.3 P1 probe (_7+M2). | Superseded by nms55 | fixes-log W5-C |
| [rtmpose-x-tiled-w5-full](rtmpose-x-tiled-w5-full.md) | Tiled NMS-0.55 P1, all 8 benchmark deliveries (KEPT - v8 input). | Current best RTMPose P1 | fixes-log GRAND ANALYSIS v2 |
| rtmpose-x.zip | Zip archive of rtmpose-x. | Redundant archive | - |
| [yolo26x-pose-full-db8](yolo26x-pose-full-db8.md) | YOLO26x-pose P1 full run (KEPT - best YOLO data). | Kept for model comparison | wip/model_comparison.md |
