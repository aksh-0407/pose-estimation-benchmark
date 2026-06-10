"""UE5/HyperView pose packet conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schemas import PosePacket


def cricket_world_to_ue_cm(points_world_m: np.ndarray) -> np.ndarray:
    """Convert cricket world meters to UE centimeters.

    Internal cricket world convention:
    - X: right
    - Y: forward
    - Z: up

    UE output convention:
    - X: forward
    - Y: right
    - Z: up
    """

    points_world_m = np.asarray(points_world_m, dtype=float)
    if points_world_m.ndim != 2 or points_world_m.shape[1] != 3:
        raise ValueError("points_world_m must have shape (N, 3)")
    ue = np.empty_like(points_world_m)
    ue[:, 0] = points_world_m[:, 1] * 100.0
    ue[:, 1] = points_world_m[:, 0] * 100.0
    ue[:, 2] = points_world_m[:, 2] * 100.0
    return ue


def _nullable_points(points: np.ndarray) -> list[list[float | None]]:
    output: list[list[float | None]] = []
    for point in np.asarray(points, dtype=float):
        output.append([None if not np.isfinite(value) else float(value) for value in point])
    return output


def build_pose_packet(
    frame_id: str,
    timestamp_ns: int,
    player_id: str,
    model_version: str,
    keypoints3d_world_m: np.ndarray,
    confidence: Iterable[float],
    calibration_id: str | None = None,
    visibility: Iterable[float | None] | None = None,
    occlusion_tags: Iterable[str] | None = None,
    keypoints2d: dict[str, list[list[float | None]]] | None = None,
) -> PosePacket:
    points_world = np.asarray(keypoints3d_world_m, dtype=float)
    points_ue = cricket_world_to_ue_cm(points_world)
    confidence_list = [float(value) if np.isfinite(value) else 0.0 for value in confidence]
    visibility_list = [] if visibility is None else [None if value is None else float(value) for value in visibility]
    return PosePacket(
        frame_id=str(frame_id),
        timestamp_ns=int(timestamp_ns),
        player_id=str(player_id),
        model_version=str(model_version),
        calibration_id=calibration_id,
        keypoints3d_world_m=_nullable_points(points_world),
        keypoints3d_ue_cm=_nullable_points(points_ue),
        confidence=confidence_list,
        visibility=visibility_list,
        occlusion_tags=list(occlusion_tags or []),
        keypoints2d=keypoints2d or {},
    )


def write_jsonl(path: str | Path, packets: Iterable[PosePacket | dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for packet in packets:
            payload = packet.to_dict() if isinstance(packet, PosePacket) else packet
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

