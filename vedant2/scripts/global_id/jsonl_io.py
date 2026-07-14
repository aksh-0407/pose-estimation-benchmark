"""JSONL conversion and write-back helpers for P4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from pose_estimation.cricket.contract import validate_group1_frame
from pose_estimation.cricket.pose_shape import PoseProportions, PostureAggregate
from scripts.association.associator import Correspondence, Detection3
from scripts.association.jsonl_io import RecordsByFrame
from scripts.tracking.runner import PredictionFile


def read_correspondence_rows(path: str | Path) -> list[dict]:
    rows = []
    previous: int | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame_index = row.get("frame_index")
            if type(frame_index) is not int:
                raise ValueError("correspondence row frame_index must be an integer")
            if previous is not None and frame_index <= previous:
                raise ValueError("correspondence frame_index values must be strictly increasing")
            if not isinstance(row.get("clusters"), list):
                raise ValueError("correspondence row clusters must be a list")
            previous = frame_index
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no correspondence rows found in {path}")
    return rows


def row_to_correspondences(row: dict, camera_records: dict[str, dict]) -> list[Correspondence]:
    correspondences = []
    for cluster in row.get("clusters", []):
        members: dict[str, Detection3] = {}
        for member in cluster.get("members", []):
            camera_id = member.get("cam_id")
            if camera_id not in camera_records:
                continue
            player_index = int(member["player_index"])
            players = camera_records[camera_id].get("players", [])
            if player_index < 0 or player_index >= len(players):
                raise IndexError(f"invalid P3 player index {player_index} for {camera_id}")
            player = players[player_index]
            pose = player.get("pose_2d")
            bbox = player.get("bbox_xywh_px")
            if pose is None or bbox is None:
                continue
            members[camera_id] = Detection3(
                cam_id=camera_id,
                player_index=player_index,
                bbox_xywh_px=[float(value) for value in bbox],
                keypoints_px=np.asarray(pose["keypoints_px"], dtype=float),
                keypoint_conf=np.asarray(pose["confidence"], dtype=float),
                confidence=float(player.get("detection_confidence") or 0.0),
                local_track_id=player.get("local_track_id"),
            )
        if not members:
            continue
        ground = cluster.get("ground_xy")
        ground_xy = np.asarray(ground, dtype=float) if ground is not None else np.full(2, np.nan)
        if ground_xy.shape != (2,):
            raise ValueError("correspondence ground_xy must be length 2 or null")
        reprojection = cluster.get("mean_reprojection_error_px")
        correspondences.append(
            Correspondence(
                cluster_id=int(cluster["cluster_id"]),
                members=members,
                ground_xy=ground_xy,
                track_confidence=float(cluster["track_confidence"]),
                single_camera=bool(cluster["single_camera"]),
                mean_reprojection_error_px=float(reprojection) if reprojection is not None else None,
                cycle_consistent=bool(cluster.get("cycle_consistent", True)),
                ground_spread_m=(
                    float(cluster["ground_spread_m"])
                    if cluster.get("ground_spread_m") is not None else None
                ),
                pose_descriptor=PoseProportions.from_json(cluster.get("pose_descriptor")),
                binding_id=cluster.get("binding_id"),
                posture=PostureAggregate.from_json(cluster.get("posture")),
                ground_cov=(
                    np.asarray(cluster["ground_cov"], dtype=float)
                    if cluster.get("ground_cov") is not None else None
                ),
            )
        )
    return correspondences


def write_prediction_streams(
    records_by_frame: RecordsByFrame,
    prediction_files: Iterable[PredictionFile],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in prediction_files:
        with (output_dir / item.path.name).open("w", encoding="utf-8") as handle:
            for frame_index in sorted(records_by_frame):
                record = records_by_frame[frame_index].get(item.camera_id)
                if record is None:
                    continue
                validate_group1_frame(record, final_handoff=False)
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
