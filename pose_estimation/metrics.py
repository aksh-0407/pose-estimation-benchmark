"""Pose-estimation metrics and model selection scoring."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_unit(value: float) -> float:
    """Accept either 0-1 or 0-100 style scores and normalize to 0-1."""

    if np.isnan(value):
        return 0.0
    if value > 1.0:
        value = value / 100.0
    return float(np.clip(value, 0.0, 1.0))


def masked_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if mask is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(values[valid]))


def pck(
    predicted_xy: np.ndarray,
    target_xy: np.ndarray,
    threshold: float,
    norm: float | np.ndarray = 1.0,
    visibility: np.ndarray | None = None,
) -> float:
    """Percentage of Correct Keypoints for 2D predictions."""

    predicted_xy = np.asarray(predicted_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if predicted_xy.shape != target_xy.shape or predicted_xy.shape[-1] != 2:
        raise ValueError("predicted_xy and target_xy must both have shape (..., 2)")

    distance = np.linalg.norm(predicted_xy - target_xy, axis=-1)
    norm_array = np.asarray(norm, dtype=float)
    normalized = distance / np.maximum(norm_array, 1e-12)
    correct = normalized <= threshold
    if visibility is None:
        mask = np.isfinite(normalized)
    else:
        mask = np.asarray(visibility, dtype=float) > 0
    return masked_mean(correct.astype(float), mask)


def mpjpe(predicted_xyz: np.ndarray, target_xyz: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean per-joint position error in the units of the input arrays."""

    predicted_xyz = np.asarray(predicted_xyz, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)
    if predicted_xyz.shape != target_xyz.shape or predicted_xyz.shape[-1] != 3:
        raise ValueError("predicted_xyz and target_xyz must both have shape (..., 3)")
    errors = np.linalg.norm(predicted_xyz - target_xyz, axis=-1)
    return masked_mean(errors, mask)


def procrustes_align(predicted_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """Rigid similarity alignment for P-MPJPE evaluation."""

    predicted_xyz = np.asarray(predicted_xyz, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)
    if predicted_xyz.shape != target_xyz.shape or predicted_xyz.ndim != 2 or predicted_xyz.shape[1] != 3:
        raise ValueError("inputs must both have shape (N, 3)")

    pred_mean = predicted_xyz.mean(axis=0, keepdims=True)
    target_mean = target_xyz.mean(axis=0, keepdims=True)
    pred_centered = predicted_xyz - pred_mean
    target_centered = target_xyz - target_mean

    covariance = pred_centered.T @ target_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        vt_matrix[-1, :] *= -1
        rotation = vt_matrix.T @ u_matrix.T

    pred_variance = np.sum(pred_centered**2)
    scale = np.trace(rotation @ covariance) / max(pred_variance, 1e-12)
    return scale * pred_centered @ rotation.T + target_mean


def p_mpjpe(predicted_xyz: np.ndarray, target_xyz: np.ndarray, mask: np.ndarray | None = None) -> float:
    """MPJPE after Procrustes similarity alignment."""

    predicted_xyz = np.asarray(predicted_xyz, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        aligned = predicted_xyz.copy()
        aligned[valid] = procrustes_align(predicted_xyz[valid], target_xyz[valid])
    else:
        aligned = procrustes_align(predicted_xyz, target_xyz)
    return mpjpe(aligned, target_xyz, mask)


def project_points(points3d: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Project 3D points with a 3x4 camera projection matrix."""

    points3d = np.asarray(points3d, dtype=float)
    projection_matrix = np.asarray(projection_matrix, dtype=float)
    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("points3d must have shape (N, 3)")
    if projection_matrix.shape != (3, 4):
        raise ValueError("projection_matrix must have shape (3, 4)")

    homogeneous = np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=1)
    projected = homogeneous @ projection_matrix.T
    xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
    return xy


def reprojection_error(
    points3d: np.ndarray,
    observed_xy: np.ndarray,
    projection_matrix: np.ndarray,
    visibility: np.ndarray | None = None,
) -> np.ndarray:
    """Per-keypoint reprojection error in pixels."""

    projected_xy = project_points(points3d, projection_matrix)
    errors = np.linalg.norm(projected_xy - np.asarray(observed_xy, dtype=float), axis=-1)
    if visibility is not None:
        mask = np.asarray(visibility, dtype=float) > 0
        errors = np.where(mask, errors, np.nan)
    return errors


def temporal_jitter(sequence_xyz: np.ndarray, fps: float = 1.0, mask: np.ndarray | None = None) -> float:
    """Mean frame-to-frame joint displacement scaled by FPS."""

    sequence_xyz = np.asarray(sequence_xyz, dtype=float)
    if sequence_xyz.ndim != 3 or sequence_xyz.shape[-1] != 3:
        raise ValueError("sequence_xyz must have shape (T, J, 3)")
    if sequence_xyz.shape[0] < 2:
        return 0.0
    deltas = np.linalg.norm(np.diff(sequence_xyz, axis=0), axis=-1) * float(fps)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        valid = valid[1:] & valid[:-1]
    else:
        valid = np.isfinite(deltas)
    return masked_mean(deltas, valid)


def weighted_model_score(
    cricket_2d_accuracy: float,
    occlusion_robustness: float,
    latency_p95_ms: float,
    jitter_score: float,
    integration_effort: float,
    latency_budget_ms: float = 200.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute the plan's weighted model-selection score on a 0-100 scale."""

    weights = weights or {
        "cricket_2d_accuracy": 0.40,
        "occlusion_robustness": 0.25,
        "latency": 0.20,
        "stability_jitter": 0.10,
        "integration_effort": 0.05,
    }
    latency_score = 1.0 - (float(latency_p95_ms) / max(float(latency_budget_ms), 1e-12))
    score = (
        weights["cricket_2d_accuracy"] * _as_unit(float(cricket_2d_accuracy))
        + weights["occlusion_robustness"] * _as_unit(float(occlusion_robustness))
        + weights["latency"] * float(np.clip(latency_score, 0.0, 1.0))
        + weights["stability_jitter"] * _as_unit(float(jitter_score))
        + weights["integration_effort"] * _as_unit(float(integration_effort))
    )
    return round(score * 100.0, 4)


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return masked_mean(array)

