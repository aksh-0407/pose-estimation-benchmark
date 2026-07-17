"""Keypoint skeleton mapping utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = ROOT / "configs" / "keypoint_mappings.yaml"

# Canonical skeletons. Halpe-26 is the pipeline skeleton (COCO-17 in its first 17
# indices, then head/neck/hip + 6 foot joints). The named + root-relative export
# (pose_3d_named) keys by these names; the root joint is the mid-hip (index 19).
COCO17_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
HALPE26_KEYPOINTS = COCO17_KEYPOINTS + [
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
]
HALPE26_ROOT_INDEX = HALPE26_KEYPOINTS.index("hip")  # 19 - the export root joint

# Halpe-26 skeleton connectivity (for rendering the full body incl. feet).
HALPE26_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),                # face
    (17, 18), (18, 5), (18, 6), (18, 19),          # head-neck-shoulders-hip (spine)
    (5, 7), (7, 9), (6, 8), (8, 10),               # arms
    (19, 11), (19, 12),                            # hip -> l/r hip
    (11, 13), (13, 15), (12, 14), (14, 16),        # legs
    (15, 24), (15, 20), (15, 22),                  # left foot: heel, big toe, small toe
    (16, 25), (16, 21), (16, 23),                  # right foot: heel, big toe, small toe
]

# Kinematic tree as an ordered (parent -> child) bone list, rooted at the mid-hip
# (index 19 = HALPE26_ROOT_INDEX). The order is breadth-first from the root, so a
# forward pass over the list always sees a bone's parent joint already placed  - 
# this is what makes forward-kinematics rebuild / bone-length enforcement a single
# linear sweep (used by the 07 refinement stage). Covers all 26 joints (root has no
# incoming bone), a spanning tree of HALPE26_EDGES plus the face->head link.
HALPE26_BONES = [
    (19, 11), (19, 12), (19, 18),                          # pelvis -> hips + spine
    (18, 5), (18, 6), (18, 17),                            # neck -> shoulders + head
    (11, 13), (12, 14),                                    # hips -> knees
    (5, 7), (6, 8),                                        # shoulders -> elbows
    (17, 0),                                               # head -> nose
    (13, 15), (14, 16),                                    # knees -> ankles
    (7, 9), (8, 10),                                       # elbows -> wrists
    (0, 1), (0, 2),                                        # nose -> eyes
    (15, 20), (15, 22), (15, 24),                          # left ankle -> toes + heel
    (16, 21), (16, 23), (16, 25),                          # right ankle -> toes + heel
    (1, 3), (2, 4),                                        # eyes -> ears
]

# Left/right bone pairs whose lengths are anatomically equal - pooled to a single
# per-player length so the emitted skeleton is bilaterally symmetric.
HALPE26_SYMMETRIC_BONES = [
    ((19, 11), (19, 12)),   # pelvis width halves
    ((18, 5), (18, 6)),     # neck -> shoulder
    ((5, 7), (6, 8)),       # upper arm
    ((7, 9), (8, 10)),      # forearm
    ((11, 13), (12, 14)),   # thigh
    ((13, 15), (14, 16)),   # shank
    ((0, 1), (0, 2)),       # nose -> eye
    ((1, 3), (2, 4)),       # eye -> ear
    ((15, 20), (16, 21)),   # ankle -> big toe
    ((15, 22), (16, 23)),   # ankle -> small toe
    ((15, 24), (16, 25)),   # ankle -> heel
]

# Absolute anatomical bone-length limits in METRES: {(parent, child): (min, max, default)}.
# Adult cricketer proportions (~1.7-1.95 m). These are a hard biomechanical guardrail: a
# per-player median bone that falls outside its range is a triangulation artefact (a bad /
# chimera identity), NOT a real limb, so the refinement caps it - the emitted skeleton can
# never "go beyond physics" regardless of how bad the underlying 3D is. `default` is used
# when a bone has no reliable samples at all.
HALPE26_BONE_LIMITS_M = {
    (19, 11): (0.06, 0.17, 0.11), (19, 12): (0.06, 0.17, 0.11),   # pelvis half-width
    (19, 18): (0.40, 0.62, 0.52),                                 # hip -> neck (trunk)
    (18, 5): (0.12, 0.26, 0.18), (18, 6): (0.12, 0.26, 0.18),     # neck -> shoulder
    (18, 17): (0.10, 0.30, 0.20),                                 # neck -> head
    (11, 13): (0.35, 0.55, 0.45), (12, 14): (0.35, 0.55, 0.45),   # thigh
    (13, 15): (0.33, 0.50, 0.43), (14, 16): (0.33, 0.50, 0.43),   # shank
    (5, 7): (0.23, 0.38, 0.30), (6, 8): (0.23, 0.38, 0.30),       # upper arm
    (7, 9): (0.20, 0.32, 0.26), (8, 10): (0.20, 0.32, 0.26),      # forearm
    (17, 0): (0.05, 0.30, 0.15),                                  # head -> nose
    (0, 1): (0.02, 0.10, 0.05), (0, 2): (0.02, 0.10, 0.05),       # nose -> eye
    (1, 3): (0.03, 0.14, 0.09), (2, 4): (0.03, 0.14, 0.09),       # eye -> ear
    (15, 20): (0.10, 0.28, 0.19), (16, 21): (0.10, 0.28, 0.19),   # ankle -> big toe
    (15, 22): (0.10, 0.28, 0.19), (16, 23): (0.10, 0.28, 0.19),   # ankle -> small toe
    (15, 24): (0.03, 0.13, 0.08), (16, 25): (0.03, 0.13, 0.08),   # ankle -> heel
}

# Hinge joints as (proximal, joint, distal) triplets - the flexion angle is measured
# at ``joint`` between the (proximal - joint) and (distal - joint) vectors. Used to
# clamp anatomically impossible bends (backward knees / elbows).
HALPE26_HINGES = [
    (5, 7, 9),      # left elbow
    (6, 8, 10),     # right elbow
    (11, 13, 15),   # left knee
    (12, 14, 16),   # right knee
]


def named_root_relative(
    points: np.ndarray,
    names: list[str] = HALPE26_KEYPOINTS,
    root_index: int = HALPE26_ROOT_INDEX,
) -> dict[str, Any]:
    """Self-describing named 3D pose: root in world metres, every joint relative to root.

    ``points`` is ``(J, 3)`` world-metre coordinates aligned to ``names``. Returns
    ``{root_joint, root_world_m, joints_root_relative_m}`` where the root world
    position is chosen robustly (the named root joint, else the mid-hip, else the
    first finite joint). Non-finite joints are emitted as ``null``.
    """
    array = np.asarray(points, dtype=float)
    root = array[root_index] if root_index < len(array) else np.array([np.nan] * 3)
    if not np.isfinite(root).all():
        # Fall back to the mid-hip (COCO l/r hip), then any finite joint.
        hips = array[[11, 12]] if len(array) > 12 else array[:0]
        finite_hips = hips[np.isfinite(hips).all(axis=1)] if len(hips) else hips
        if len(finite_hips):
            root = finite_hips.mean(axis=0)
        else:
            finite = array[np.isfinite(array).all(axis=1)]
            root = finite[0] if len(finite) else np.zeros(3)
    joints: dict[str, list[float] | None] = {}
    for index, name in enumerate(names):
        if index < len(array) and np.isfinite(array[index]).all():
            joints[name] = (array[index] - root).tolist()
        else:
            joints[name] = None
    return {
        "root_joint": names[root_index],
        "root_world_m": [float(v) for v in root],
        "joints_root_relative_m": joints,
    }


def load_keypoint_mappings(path: str | Path = DEFAULT_MAPPING) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def map_keypoints(
    keypoints: list[list[float | None]] | np.ndarray,
    source_skeleton: str,
    target_skeleton: str = "coco_17",
    mapping_path: str | Path = DEFAULT_MAPPING,
) -> list[list[float | None]]:
    """Map an ``N x >=3`` keypoint array to the configured target skeleton."""

    mappings = load_keypoint_mappings(mapping_path)
    if target_skeleton != mappings.get("target_skeleton", target_skeleton):
        raise ValueError(f"Unsupported target skeleton: {target_skeleton}")
    source = mappings["source_to_coco_17"].get(source_skeleton)
    if source is None:
        raise KeyError(f"No mapping from {source_skeleton} to {target_skeleton}")

    array = np.asarray(keypoints, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("keypoints must have shape (N, >=3)")

    indices = source["source_indices"]
    output = np.full((len(indices), array.shape[1]), np.nan, dtype=float)
    for out_index, source_index in enumerate(indices):
        if source_index < array.shape[0]:
            output[out_index] = array[source_index]
    return _nullable(output[:, :3])


def _nullable(array: np.ndarray) -> list[list[float | None]]:
    result: list[list[float | None]] = []
    for row in array:
        result.append([None if not np.isfinite(value) else float(value) for value in row])
    return result

