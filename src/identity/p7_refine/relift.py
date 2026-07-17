"""Visibility-aware re-lift: fix joints triangulated from a partially-visible view.

Root cause it solves (real case: an umpire whose full body is in cam_01 but only whose
upper body is in cam_04): the 2D pose model still emits lower-body keypoints for cam_04,
crammed at the image edge with low confidence. Stage-04 triangulation gates only on
``conf > 0``, so it pairs cam_01's good legs with cam_04's *hallucinated* legs and the 3D
legs stretch along the depth ray - a pose that reprojects perfectly in both views yet is
physically impossible.

The fix reasons per joint about which cameras *reliably* see it (by confidence), then:

  * >= 2 reliable views  -> ordinary weighted-DLT triangulation (clean);
  * exactly 1 reliable view -> place the joint on that camera's back-projection ray at the
    canonical bone length from its already-placed parent (single-view bone-length lift), so
    the umpire's legs hang straight down and still reproject correctly in cam_01;
  * 0 reliable views -> left NaN for the temporal / skeletal-prior fill downstream.

Pure numpy (no scipy) so it runs anywhere. Never reads or writes identity - it only
recomputes ``pose_3d`` for the already-assigned player cluster.
"""

from __future__ import annotations

import numpy as np

from identity.common.triangulation import triangulate_point_dlt


def camera_center_ray(projection: np.ndarray, pixel) -> tuple[np.ndarray, np.ndarray]:
    """Camera centre C and the into-scene ray direction d for a pixel.

    For projection ``P = [M | p4]`` a world point projects as ``x ~ M X + p4``. The
    back-projection ray is ``X(λ) = C + λ d`` with ``C = -M^{-1} p4`` and
    ``d = M^{-1} [u, v, 1]``; because ``M d = [u, v, 1]`` the projective depth of ``C+λd``
    equals ``λ``, so λ > 0 is exactly "in front of the camera".
    """

    M = np.asarray(projection, dtype=float)[:, :3]
    p4 = np.asarray(projection, dtype=float)[:, 3]
    minv = np.linalg.inv(M)
    center = -minv @ p4
    direction = minv @ np.array([pixel[0], pixel[1], 1.0])
    return center, direction


def ray_bone_length_point(
    anchor: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
    bone_length: float,
    *,
    prev: np.ndarray | None = None,
) -> np.ndarray | None:
    """Point on the ray ``center + λ·direction`` at distance ``bone_length`` from ``anchor``.

    Solves ``|C + λ d - anchor| = L`` and picks the in-front (λ>0) root that (a) is nearest
    the previous frame's joint when available (temporal continuity → smooth) else (b) keeps
    the joint nearest the anchor's depth, which counters the depth-stretch. If the ray does
    not reach the sphere, falls back to the closest ray point projected to bone length.
    """

    o = center - anchor
    b = 2.0 * float(direction @ o)
    c = float(o @ o) - bone_length * bone_length
    a = float(direction @ direction)
    disc = b * b - 4 * a * c
    roots: list[float] = []
    if disc >= 0:
        s = np.sqrt(disc)
        roots = [(-b + s) / (2 * a), (-b - s) / (2 * a)]
    front = [lam for lam in roots if lam > 1e-6]
    if not front:
        # ray misses the sphere: closest approach, then scale to bone length from anchor.
        lam = -float(direction @ o) / a
        approach = center + lam * direction
        v = approach - anchor
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return None
        return anchor + v / n * bone_length
    candidates = [center + lam * direction for lam in front]
    if prev is not None and np.isfinite(prev).all():
        return min(candidates, key=lambda x: float(np.linalg.norm(x - prev)))
    anchor_depth = -float(direction @ (center - anchor)) / a  # lambda of anchor's closest point
    return min(candidates, key=lambda x: abs(float((x - center) @ direction) / a - anchor_depth))


def _reliable(conf: float, pixel, image_size, conf_min: float, margin: float) -> bool:
    """A view reliably sees a joint iff its confidence clears the floor (the trustworthy
    signal - a hallucinated edge keypoint is low-confidence even when nominally in-frame).
    Grossly out-of-bounds points are also rejected as a cheap extra guard."""
    if not np.isfinite(conf) or conf < conf_min:
        return False
    if image_size is not None:
        w, h = image_size
        x, y = pixel
        if x < -margin or y < -margin or x > w + margin or y > h + margin:
            return False
    return True


def _reliable_views(cam_obs, joint_count, conf_min, margin) -> list[list[int]]:
    reliable: list[list[int]] = [[] for _ in range(joint_count)]
    for view_i, (_, kp, conf, imgsz) in enumerate(cam_obs):
        for j in range(joint_count):
            if _reliable(float(conf[j]), kp[j], imgsz, conf_min, margin):
                reliable[j].append(view_i)
    return reliable


def _triangulate_reliable(cam_obs, reliable, joint_count) -> tuple[np.ndarray, np.ndarray]:
    """Weighted DLT for every joint with >= 2 reliable views; NaN otherwise."""
    pose = np.full((joint_count, 3), np.nan, dtype=float)
    conf_out = np.zeros(joint_count, dtype=float)
    for j in range(joint_count):
        if len(reliable[j]) >= 2:
            pts = np.array([cam_obs[v][1][j] for v in reliable[j]], dtype=float)
            projs = np.array([cam_obs[v][0] for v in reliable[j]], dtype=float)
            cfs = np.array([cam_obs[v][2][j] for v in reliable[j]], dtype=float)
            x = triangulate_point_dlt(pts, projs, cfs, min_views=2)
            if np.isfinite(x).all():
                pose[j] = x
                conf_out[j] = float(np.mean(cfs))
    return pose, conf_out


def relift_frame(
    cam_obs: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple]],
    canonical: dict[tuple[int, int], float],
    bones: list[tuple[int, int]],
    *,
    root_index: int,
    joint_count: int,
    conf_min: float,
    margin: float,
    prev_pose: np.ndarray | None,
    fallback_pose: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-lift one frame's skeleton. ``cam_obs`` = list of (P, kp(J,2), conf(J), image_size)."""

    reliable = _reliable_views(cam_obs, joint_count, conf_min, margin)
    pose, conf_out = _triangulate_reliable(cam_obs, reliable, joint_count)

    def _ray_for(j: int, anchor: np.ndarray, length: float) -> np.ndarray | None:
        view = reliable[j][0]
        P, kp, conf, _ = cam_obs[view]
        center, direction = camera_center_ray(P, kp[j])
        prev = prev_pose[j] if prev_pose is not None else None
        pt = ray_bone_length_point(anchor, center, direction, length, prev=prev)
        if pt is not None:
            conf_out[j] = max(conf_out[j], 0.5 * float(conf[j]) + 0.25)  # single-view: medium trust
        return pt

    # Root: prefer Pass-A; else single-view lift anchored to the (reliably placed) neck via
    # the trunk bone; else fall back to the existing lifted pose.
    if not np.isfinite(pose[root_index]).all():
        neck = None
        trunk_len = canonical.get((root_index, 18))
        if np.isfinite(pose[18]).all() and reliable[root_index] and trunk_len:
            neck = pose[18]
            pt = _ray_for(root_index, neck, trunk_len)
            if pt is not None:
                pose[root_index] = pt
        if not np.isfinite(pose[root_index]).all() and fallback_pose is not None and np.isfinite(fallback_pose[root_index]).all():
            pose[root_index] = fallback_pose[root_index]
            conf_out[root_index] = max(conf_out[root_index], 0.2)

    # Pass B: BFS over bones - fill still-missing joints from their placed parent + one ray.
    for parent, child in bones:
        if np.isfinite(pose[child]).all() or not np.isfinite(pose[parent]).all():
            continue
        length = canonical.get((parent, child))
        if not length or not reliable[child]:
            continue
        pt = _ray_for(child, pose[parent], length)
        if pt is not None:
            pose[child] = pt

    return pose, conf_out


def relift_sequence(
    obs_by_row: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple]]],
    bones: list[tuple[int, int]],
    symmetric_pairs,
    *,
    root_index: int,
    joint_count: int,
    conf_min: float,
    margin: float,
    limits,
    fallback_seq: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-pass visibility-aware re-lift of a whole identity sequence.

    Pass 1 triangulates only the confidently-multi-view joints and estimates the player's
    canonical (anatomically-clamped) bone lengths from those clean samples. Pass 2 re-lifts
    every frame, using single-view bone-length ray placement for joints seen reliably in
    only one camera - with the previous frame as a continuity prior so the result is smooth.
    """

    from identity.p7_refine.refine import estimate_canonical_bones

    frames = len(obs_by_row)
    pass_a = np.full((frames, joint_count, 3), np.nan, dtype=float)
    for t, cam_obs in enumerate(obs_by_row):
        reliable = _reliable_views(cam_obs, joint_count, conf_min, margin)
        pass_a[t], _ = _triangulate_reliable(cam_obs, reliable, joint_count)

    canonical = estimate_canonical_bones(pass_a, bones, symmetric_pairs, limits=limits)

    out = np.full((frames, joint_count, 3), np.nan, dtype=float)
    conf = np.zeros((frames, joint_count), dtype=float)
    prev = None
    for t, cam_obs in enumerate(obs_by_row):
        fb = fallback_seq[t] if fallback_seq is not None else None
        pose, cf = relift_frame(
            cam_obs, canonical, bones,
            root_index=root_index, joint_count=joint_count,
            conf_min=conf_min, margin=margin, prev_pose=prev, fallback_pose=fb,
        )
        out[t], conf[t] = pose, cf
        prev = pose
    return out, conf
