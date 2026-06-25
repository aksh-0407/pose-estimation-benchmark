# RTMPose-l Body8 384x288

Real-time Multi-Person Pose Estimator - Large RTMPose body model trained on the
combined Body8 setup and evaluated as the strongest official RTMPose COCO-17 row
in the local RTMPose project table.

## Model Details

- **Framework**: MMPose
- **Skeleton**: COCO-17 (body only)
- **Input Size**: 384x288
- **Training Data**: Body8 combined body datasets
- **Checkpoint**: `models/rtmpose_l_body8/weights/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth`
- **Config**: `external/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-384x288.py`

## Benchmark Status

Preferred accuracy-first RTMPose COCO-17 candidate. The official RTMPose project
table reports 78.3 COCO AP for this 384x288 Body8 model.

## References

- MMPose RTMPose: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- Paper: https://arxiv.org/abs/2303.07399
