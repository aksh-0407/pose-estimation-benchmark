"""2D benchmark metrics for normalized pose predictions."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from pose_estimation.metrics import masked_mean, pck


def pck_at_thresholds(
    predicted_xy: np.ndarray,
    target_xy: np.ndarray,
    norm: float | np.ndarray,
    visibility: np.ndarray | None = None,
    thresholds: Iterable[float] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    return {
        f"pck@{threshold:.2f}": pck(predicted_xy, target_xy, threshold, norm=norm, visibility=visibility)
        for threshold in thresholds
    }


def evaluate_2d_predictions(
    predicted: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray | None = None,
    norm: float | np.ndarray = 1.0,
) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    if predicted.shape != target.shape or predicted.shape[-1] < 2:
        raise ValueError("predicted and target must have matching shape (..., >=2)")
    errors = np.linalg.norm(predicted[..., :2] - target[..., :2], axis=-1)
    if visibility is None:
        visible = np.isfinite(errors)
    else:
        visible = np.asarray(visibility, dtype=float) > 0
    metrics = pck_at_thresholds(predicted[..., :2], target[..., :2], norm=norm, visibility=visible)
    metrics.update(
        {
            "mean_pixel_error": masked_mean(errors, visible),
            "missing_keypoint_rate": 1.0 - float(np.mean(visible)) if visible.size else 1.0,
            "detection_rate": float(np.any(visible)),
        }
    )
    return metrics

