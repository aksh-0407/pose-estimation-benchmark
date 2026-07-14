# configs/v8 — tiled-detection default stack (cut 2026-07-13)

P1 is produced on the L40S box with the tiled detector:
    python scripts/inference/run_phase1_l40s.py --tiled-det --nms-thr 0.55 \
        --det-batch-size 8 --pose-batch-size 256
(RTMDet-m person on a 4x2 overlap-0.25 tile grid + full frame, cross-tile NMS 0.55
+ IoM-0.7 containment suppression; RTMPose-X Halpe-26 pose unchanged.)

Downstream: p2 = v7 + lowconf_can_spawn:false (specks associate, never birth);
p3/p4 = v7 unchanged; p5 = roles v1.1 epoch solver + Wave-6 peripheral
suppression ENABLED. Frozen reference tree: benchmarks/runs/pipetrack_v8.0.
