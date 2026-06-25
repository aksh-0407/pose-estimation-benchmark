# RTMPose-l

Real-time Multi-Person Pose Estimator - Large variant from OpenMMLab for COCO body-only (17 keypoints) keypoint detection.

## Model Details

- **Framework**: MMPose
- **Skeleton**: COCO-17 (body only)
- **Input Size**: 384x288
- **Training Data**: COCO + AIC dataset
- **Checkpoint**: 420 epoch training

## Benchmark Status

Candidate model for body-only real-time pose estimation. Good speed/accuracy tradeoff for multi-person scenarios.

## References

- MMPose RTMPose: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- Paper: https://arxiv.org/abs/2303.07399
