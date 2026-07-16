# Comprehensive Pipeline Restructuring & Implementation Plan

> This document restructures the provided engineering transcript into a professional implementation specification while preserving all technical intent.

# 1. Executive Summary

The primary objective is to significantly improve the robustness, geometric accuracy, and overall reliability of the complete multi-camera 3D player tracking pipeline.

The emphasis is **accuracy first**. Computational optimization is explicitly a secondary concern and should only be considered after the highest-quality pipeline has been established.

Every proposed modification must be validated experimentally through rigorous **before-vs-after analysis**, rather than assumptions.

---

# 2. Guiding Principles

- Never blindly trust existing implementation.
- Never blindly trust suggestions from this document.
- Never blindly trust internet resources.
- Every proposed modification must be experimentally verified.
- Human visualization remains the final qualitative verification.
- Decisions must be driven by measured metrics.

---

# 3. Highest Priority Pipeline Changes

## 3.1 Complete 3D Pose Triangulation

Current approach:
- Single-point triangulation / lifting.

Required redesign:
- Triangulate the complete 26-keypoint skeleton.
- Use all seven calibrated cameras.
- Produce full 3D skeleton.
- Understand calibration pipeline thoroughly before modifying code.

Research tasks:
- Robust multi-view triangulation.
- Outlier rejection.
- Confidence weighting.
- Multi-view optimization.
- Robust aggregation (median/M-estimators instead of arithmetic mean).

Avoid simple averaging because one poor camera can significantly corrupt reconstruction.

---

## 3.2 Human Location Definition

Current:
- Feet used as player location.

Problem:
- Feet spacing introduces instability.

New definition:
- Hip/root joint becomes canonical body location.
- Project hip vertically onto ground plane.
- Ground projection becomes player position.

Research:
- Mathematical implications.
- Failure cases.
- Robust projection methods.

---

# 4. Stabilization Study

Question:

Should stabilization happen:

Pipeline A:
2D → Stabilization → Triangulation → 3D

Pipeline B:
2D → Triangulation → 3D Stabilization

Concern:
Early stabilization may distort geometry.

Required:
Implement both.
Benchmark both.
Compare metrics.

---

# 5. Robust Multi-Camera Design

Need robustness against:

- bad camera
- noisy camera
- occlusion
- missed detections
- outlier camera

Instead of averaging all cameras equally:

- identify outliers
- rely more on consistent cameras
- perform research-grade robust estimation

---

# 6. Per-Camera Analysis

Generate metrics for every camera individually.

Goals:

- identify weak cameras
- identify noisy viewpoints
- compare monocular lifting vs multi-view triangulation
- compare every camera's contribution

Deliverables:

- per-camera metrics
- per-camera visualizations
- per-ID statistics

---

# 7. Detector Evaluation

RTM currently appears strongest.

Still evaluate:

- RTMO
- RT-DETR
- any remaining detector listed in inference.md

Requirement:

No geometric precision loss during cropping.

Target:

~1 pixel joint localization error.

---

# 8. Ghost Marker Investigation

Current visualization frequently shows unstable 3D locations.

Investigate:

- reconstruction instability
- calibration issues
- synchronization
- poor triangulation
- stabilization effects

---

# 9. Documentation Cleanup

Remove throughout repository:

- Ground Truth references
- Labelled-data assumptions
- Fine-tuning suggestions requiring labels

Reason:

No labelled dataset exists.

Final verification:

Human visualization.

---

# 10. Evaluation Metrics

Primary metrics:

- Reprojection error
- ID swaps
- ID teleportations

Objectives:

Minimize all three.

Do not optimize for metrics requiring unavailable ground truth.

---

# 11. Temporal Improvements

Experiment with:

- SmoothNet
- Temporal 2D refinement

Measure improvements quantitatively.

---

# 12. Occlusion Robustness

When players overlap:

If one camera has poor visibility while others remain clear:

Treat poor view as outlier.

Depend more on clear observations.

Research robust fusion methods.

---

# 13. Tracking Improvements

SORT-like appearance methods may fail because uniforms are visually similar.

Instead:

- exploit pose geometry
- exploit skeleton structure
- evaluate fixes documented in tracking notes
- compare before vs after

---

# 14. Pose Shape as Primary Cue

Cross-camera identity should rely heavily on pose geometry.

Required:

- pose-shape based clustering
- geometry-aware matching
- adaptive parallax gating
- clustering capable of splitting identities
- fix C7 image-size handling

---

# 15. 3D-First Philosophy

Major design principle:

Everything important should ultimately be decided in 3D.

Core representation:

Full 3D skeleton.

Identity reasoning should prioritize:

- skeleton shape
- skeleton location

rather than isolated 2D observations.

---

# 16. Experimental Methodology

Every modification requires:

- baseline
- modified pipeline
- quantitative comparison
- visualization
- mosaics
- implementation log

No change should be accepted without evidence.

---

# 17. Large-Scale Validation

Use available compute (L40S).

Run extensive experiments over:

- 8-video dataset
- 40-delivery dataset
- additional representative subsets

Collect statistically meaningful numbers.

---

# 18. Documentation Review

Thoroughly review:

docs/diagnosis/

docs/pipeline/

Incorporate useful ideas after verification.

---

# 19. Human-in-the-Loop

Whenever uncertainty exists:

- ask for human review

Generate mosaics frequently.

Review intermediate outputs before large architectural changes.

---

# 20. Optimization Philosophy

Accuracy first.

Optimization second.

However:

Keep implementations efficient enough to rapidly generate mosaics and benchmark outputs.

---

# 21. Mandatory Deliverables

For every completed task:

- implementation notes
- methods log
- before vs after comparison
- numerical metrics
- qualitative visualization
- mosaics
- conclusions
- recommendations

# Final Objective

Build a research-grade, robust, multi-camera 3D player tracking pipeline whose decisions are based primarily on accurate 3D skeleton reconstruction and validated through rigorous experimentation rather than assumptions.
