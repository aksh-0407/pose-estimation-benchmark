"""Cluster-level 3D lift + purity signals (F9b/F9c — the P3.5 core).

Shared by the standalone P3.5 stage (``triangulate_predictions --id-source binding``)
and the in-runner feedback hook (``graph_lift_feedback``). Given per-frame member
keypoints for one identity key (a P3 ``binding_id`` or a provisional cluster), it:

* triangulates the full skeleton per frame (RANSAC DLT, the repo's standard lift);
* derives the **purity signature**: a chimera — two people welded into one binding —
  fails torso reprojection *consistently and one-sidedly* (the intruding camera's
  member carries the bias), which clean clusters do not;
* pools a **bone-ratio descriptor** and a metric **stature** per key — the identity
  evidence Waves 3/4 use for the shape cue and the split moves.

Everything here is read-only over the association outputs; nothing feeds back into
clustering unless a caller explicitly consumes it (flag-gated).

Performance note: the per-joint solves are tiny (2x2/3x3); run with BLAS threads
capped to 1 (OMP/MKL/OPENBLAS_NUM_THREADS=1, as the pipetrack drivers already do)
or thread spawn overhead dominates by ~20x.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from identity.common.pose_shape import (
    PoseProportions,
    limb_proportion_descriptor,
    merge_descriptor,
)
from identity.common.triangulation import (
    point_covariance_3d,
    reprojection_errors_for_point,
    triangulate_skeleton_ransac,
)

_TORSO_JOINTS = (5, 6, 11, 12)   # shoulders + hips: present on every real person
_PELVIS_JOINTS = (11, 12)


@dataclass
class FrameLift:
    """One identity-frame's triangulated skeleton."""

    points3d: np.ndarray                 # (17, 3) world metres (NaN where <2 views)
    confidences: np.ndarray              # (17,)
    mean_reprojection_errors: np.ndarray  # (17,) px, NaN where unmeasured
    # cam_id -> mean torso reprojection residual of THIS member's 2D against the
    # cluster's 3D skeleton (px). The chimera signature lives here: an intruder
    # camera shows a consistent, one-sided bias.
    torso_residual_by_cam: dict[str, float] = field(default_factory=dict)
    # Optional per-joint 3D covariance diagonals (17, 3) m^2 and the full pelvis
    # covariance (3, 3): elongated along the rays on facing-pair view sets.
    cov_diag_m2: np.ndarray | None = None
    pelvis_cov_m2: np.ndarray | None = None

    @property
    def pelvis_xyz(self) -> np.ndarray:
        pelvis = self.points3d[list(_PELVIS_JOINTS)]
        finite = np.isfinite(pelvis).all(axis=1)
        if not finite.any():
            return np.full(3, np.nan)
        return pelvis[finite].mean(axis=0)

    @property
    def ankle_z(self) -> float:
        ankles = self.points3d[[15, 16], 2]
        finite = np.isfinite(ankles)
        return float(np.min(ankles[finite])) if finite.any() else float("nan")


@dataclass
class ClusterPurity:
    """Whole-delivery purity + identity evidence for one binding/cluster key."""

    frames_lifted: int = 0
    torso_residual_p50: float | None = None
    torso_residual_p95: float | None = None
    chimera_frame_fraction: float | None = None   # frames with residual > threshold
    # cam_id -> (mean torso residual px, frames observed): one-sided bias = intruder.
    per_camera_residual: dict[str, tuple[float, int]] = field(default_factory=dict)
    chimera_suspect: bool = False
    worst_camera: str | None = None
    descriptor: PoseProportions | None = None
    stature_m: float | None = None

    def to_json(self) -> dict:
        return {
            "frames_lifted": self.frames_lifted,
            "torso_residual_p50": self.torso_residual_p50,
            "torso_residual_p95": self.torso_residual_p95,
            "chimera_frame_fraction": self.chimera_frame_fraction,
            "per_camera_residual": {
                cam: {"mean_px": mean, "frames": count}
                for cam, (mean, count) in sorted(self.per_camera_residual.items())
            },
            "chimera_suspect": self.chimera_suspect,
            "worst_camera": self.worst_camera,
            "descriptor": self.descriptor.to_json() if self.descriptor is not None else None,
            "stature_m": self.stature_m,
        }


def lift_frame(
    member_keypoints: dict[str, np.ndarray],
    projections: dict[str, np.ndarray],
    *,
    reprojection_threshold_px: float = 10.0,
    min_views: int = 2,
    cheirality: bool = False,
    compute_cov: bool = False,
) -> FrameLift | None:
    """Triangulate one identity-frame from its member cameras' (17, 3) keypoints."""

    cams = [cam for cam in sorted(member_keypoints) if cam in projections]
    if len(cams) < min_views:
        return None
    keypoints = np.asarray([member_keypoints[cam] for cam in cams], dtype=float)
    proj = np.asarray([projections[cam] for cam in cams], dtype=float)
    points3d, confidences, errors = triangulate_skeleton_ransac(
        keypoints, proj,
        reprojection_threshold_px=reprojection_threshold_px,
        min_views=min_views,
        cheirality=cheirality,
    )
    lift = FrameLift(points3d, confidences, errors)
    if compute_cov:
        diag = np.full((points3d.shape[0], 3), np.nan)
        pelvis_covs = []
        for joint in range(points3d.shape[0]):
            cov = point_covariance_3d(
                points3d[joint], keypoints[:, joint, :2], proj, keypoints[:, joint, 2]
            )
            if cov is not None:
                diag[joint] = np.diag(cov)
                if joint in _PELVIS_JOINTS:
                    pelvis_covs.append(cov)
        lift.cov_diag_m2 = diag
        if pelvis_covs:
            lift.pelvis_cov_m2 = np.mean(np.asarray(pelvis_covs), axis=0)
    for view_index, cam in enumerate(cams):
        residuals = []
        for joint in _TORSO_JOINTS:
            if not np.isfinite(points3d[joint]).all():
                continue
            if keypoints[view_index, joint, 2] <= 0:
                continue
            err = reprojection_errors_for_point(
                points3d[joint],
                keypoints[view_index : view_index + 1, joint, :2],
                proj[view_index : view_index + 1],
            )[0]
            if np.isfinite(err):
                residuals.append(float(err))
        if residuals:
            lift.torso_residual_by_cam[cam] = float(np.mean(residuals))
    return lift


def cluster_purity(
    lifts: list[FrameLift],
    *,
    chimera_torso_residual_px: float = 20.0,
    chimera_frame_fraction: float = 0.3,
    descriptor_min_conf: float = 0.3,
) -> ClusterPurity:
    """Aggregate per-frame lifts into the whole-delivery purity/identity report."""

    purity = ClusterPurity(frames_lifted=len(lifts))
    if not lifts:
        return purity

    frame_residuals = []
    per_cam: dict[str, list[float]] = defaultdict(list)
    statures = []
    descriptor: PoseProportions | None = None
    for lift in lifts:
        if lift.torso_residual_by_cam:
            frame_residuals.append(max(lift.torso_residual_by_cam.values()))
            for cam, value in lift.torso_residual_by_cam.items():
                per_cam[cam].append(value)
        finite_z = lift.points3d[np.isfinite(lift.points3d).all(axis=1), 2]
        if finite_z.size >= 6:
            statures.append(float(np.max(finite_z)))
        frame_descriptor = limb_proportion_descriptor(
            lift.points3d, lift.confidences, min_conf=descriptor_min_conf
        )
        if frame_descriptor is not None and frame_descriptor.is_defined():
            descriptor = (
                frame_descriptor if descriptor is None
                else merge_descriptor(descriptor, frame_descriptor, rate=0.1)
            )

    if frame_residuals:
        values = np.asarray(frame_residuals, dtype=float)
        purity.torso_residual_p50 = float(np.percentile(values, 50))
        purity.torso_residual_p95 = float(np.percentile(values, 95))
        purity.chimera_frame_fraction = float(
            np.mean(values > chimera_torso_residual_px)
        )
        purity.chimera_suspect = purity.chimera_frame_fraction >= chimera_frame_fraction
    purity.per_camera_residual = {
        cam: (float(np.mean(vals)), len(vals)) for cam, vals in per_cam.items()
    }
    if purity.per_camera_residual:
        purity.worst_camera = max(
            purity.per_camera_residual, key=lambda cam: purity.per_camera_residual[cam][0]
        )
    purity.descriptor = descriptor
    purity.stature_m = float(np.median(statures)) if statures else None
    return purity
