"""Run-artifact loading shared by the visualization renderers.

Readers for the pipeline's JSONL predictions and side artifacts (association
correspondences, ball events, roles, suppression, pitch calibration), plus
small derivations over them (delivery roster, ground extents, ball trails).
Every renderer (mosaic, per-camera, phase-1 overlays, bird's-eye) reads
through this module so the file formats are parsed in exactly one place.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: Path, *, sample_every: int, max_frames: int | None) -> list[dict[str, Any]]:
    records = []
    for index, record in enumerate(iter_jsonl(path)):
        if index % sample_every != 0:
            continue
        records.append(record)
        if max_frames is not None and len(records) >= max_frames:
            break
    return records


def stage_from_manifest(manifest: dict[str, Any]) -> str:
    """Map a stage run manifest's task name to the overlay identity mode."""
    return {
        "per_camera_tracking": "p2",
        "cross_camera_association": "p3",
        "global_id_tracking": "p4",
        "multi_view_3d_lift": "p4",
    }.get(manifest.get("task"), "p4")


def load_cluster_badges(path: Path) -> dict[tuple[int, str], dict[int, int]]:
    """Per (frame, camera) player-index to association-cluster-id map."""
    badges: dict[tuple[int, str], dict[int, int]] = {}
    if not path.exists():
        return badges
    for row in iter_jsonl(path):
        frame_index = int(row["frame_index"])
        for cluster in row.get("clusters", []):
            for member in cluster.get("members", []):
                key = (frame_index, member["cam_id"])
                badges.setdefault(key, {})[int(member["player_index"])] = int(cluster["cluster_id"])
    return badges


def load_pitch_extents(path: Path) -> tuple[float, float, float, float]:
    """World-plane view window from the pitch calibration file, padded; a wide
    default when the file is absent or unusable."""
    if not path.exists():
        return (-15.0, 15.0, -30.0, 30.0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(
        [value[:2] for value in payload.values() if isinstance(value, list) and len(value) >= 2],
        dtype=float,
    )
    if points.size == 0 or not np.isfinite(points).all():
        return (-15.0, 15.0, -30.0, 30.0)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_pad = max(3.0, 0.08 * (x_max - x_min))
    y_pad = max(3.0, 0.08 * (y_max - y_min))
    return (float(x_min - x_pad), float(x_max + x_pad), float(y_min - y_pad), float(y_max + y_pad))


def load_ball_positions(events_root: Path, delivery_id: str) -> dict[str, tuple[float, float]]:
    """Map frame_name -> (cx, cy) normalized ball center from the 2D event artifacts."""
    positions: dict[str, tuple[float, float]] = {}
    if not events_root.exists():
        return positions
    best_conf: dict[str, float] = {}
    for delivery_dir in sorted(events_root.glob(f"{delivery_id}_*")):
        two_d = delivery_dir / f"{delivery_dir.name}_2D.json"
        if not two_d.exists():
            continue
        payload = json.loads(two_d.read_text(encoding="utf-8"))
        for frame in payload.get("frames", []):
            for camera in frame.get("cameras", []):
                frame_name = camera.get("frame_name")
                if not frame_name:
                    continue
                for detection in camera.get("detections", []):
                    coords = detection.get("coords")
                    if not coords or len(coords) < 2:
                        continue
                    conf = float(detection.get("confidence_score") or 0.0)
                    if frame_name not in best_conf or conf > best_conf[frame_name]:
                        best_conf[frame_name] = conf
                        positions[frame_name] = (float(coords[0]), float(coords[1]))
    return positions


def ball_trail_for(
    records: list[dict[str, Any]],
    index: int,
    positions: dict[str, tuple[float, float]],
    trail_frames: int,
) -> list[tuple[float, float]]:
    if trail_frames <= 0 or not positions:
        return []
    trail: list[tuple[float, float]] = []
    start = max(0, index - trail_frames + 1)
    for cursor in range(start, index + 1):
        position = positions.get(records[cursor].get("frame_name"))
        if position is not None:
            trail.append(position)
    return trail


def load_suppression(path: Path) -> set[str]:
    """Suppressed global ids from a roles-stage ``suppression.json``; empty set when absent."""
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    if not payload.get("enabled"):
        return set()
    return {str(pid) for pid in (payload.get("suppressed") or {})}


def load_roles(path: Path) -> dict[str, str]:
    """Roles from a roles-stage ``roles.json`` artifact; {} when the stage has not run."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    roles = {}
    for player_id, entry in (payload.get("roles") or {}).items():
        role = (entry or {}).get("role")
        if role and role != "unknown":
            roles[str(player_id)] = str(role)
    return roles


def cam_short(camera_id: str) -> str:
    try:
        return f"C{int(camera_id.rsplit('_', 1)[1])}"
    except (ValueError, IndexError):
        return camera_id


def build_delivery_roster(
    records_by_camera: dict[str, list[dict[str, Any]]],
    roles: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Every global id seen anywhere in the delivery, in first-appearance order."""
    stats: dict[str, dict[str, Any]] = {}
    for records in records_by_camera.values():
        for record in records:
            frame_index = int(record["frame_index"])
            for player in record.get("players", []):
                player_id = player.get("global_player_id")
                if not player_id:
                    continue
                entry = stats.setdefault(
                    str(player_id), {"first": frame_index, "last": frame_index, "frames": 0}
                )
                entry["first"] = min(entry["first"], frame_index)
                entry["last"] = max(entry["last"], frame_index)
                entry["frames"] += 1
    roles = roles or {}
    return [
        {"id": player_id, "role": roles.get(player_id), **entry}
        for player_id, entry in sorted(stats.items(), key=lambda kv: (kv[1]["first"], kv[0]))
    ]


def compute_ground_extents(
    ground_positions: dict[int, dict[str, tuple[float, float]]],
    *,
    margin_m: float = 8.0,
) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for frame in ground_positions.values():
        for x, y in frame.values():
            if np.isfinite(x) and np.isfinite(y):
                xs.append(float(x))
                ys.append(float(y))
    if not xs:
        return -50.0, 50.0, -50.0, 50.0
    return min(xs) - margin_m, max(xs) + margin_m, min(ys) - margin_m, max(ys) + margin_m
