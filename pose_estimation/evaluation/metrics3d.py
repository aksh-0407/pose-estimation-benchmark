"""3D benchmark metrics."""

from __future__ import annotations

import numpy as np

from pose_estimation.metrics import masked_mean


def pck3d(predicted_xyz: np.ndarray, target_xyz: np.ndarray, threshold: float = 150.0) -> float:
    predicted_xyz = np.asarray(predicted_xyz, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)
    if predicted_xyz.shape != target_xyz.shape or predicted_xyz.shape[-1] != 3:
        raise ValueError("predicted_xyz and target_xyz must both have shape (..., 3)")
    errors = np.linalg.norm(predicted_xyz - target_xyz, axis=-1)
    return masked_mean((errors <= threshold).astype(float))


def acceleration_error(sequence_xyz: np.ndarray, target_xyz: np.ndarray) -> float:
    sequence_xyz = np.asarray(sequence_xyz, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)
    if sequence_xyz.shape != target_xyz.shape or sequence_xyz.ndim != 3 or sequence_xyz.shape[-1] != 3:
        raise ValueError("inputs must have shape (T, J, 3)")
    if sequence_xyz.shape[0] < 3:
        return 0.0
    pred_acc = sequence_xyz[2:] - 2 * sequence_xyz[1:-1] + sequence_xyz[:-2]
    target_acc = target_xyz[2:] - 2 * target_xyz[1:-1] + target_xyz[:-2]
    return masked_mean(np.linalg.norm(pred_acc - target_acc, axis=-1))

