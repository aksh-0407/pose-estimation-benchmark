"""P4a online global IDs followed by P4b delivery-level stitching."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pose_estimation.cricket.contract import validate_group1_frame
from pose_estimation.cricket.tracking_metrics import (
    cross_camera_agreement,
    evaluate_ground_truth,
    identity_collision_metrics,
    identity_fragmentation_proxy,
    teleport_proxy,
    track_completeness,
)
from scripts.association.jsonl_io import load_synchronized_records
from scripts.global_id.config import P4Config
from scripts.global_id.global_track import GlobalTrack
from scripts.global_id.jsonl_io import (
    read_correspondence_rows,
    row_to_correspondences,
    write_jsonl,
    write_prediction_streams,
)
from scripts.global_id.stitching import (
    build_link_costs,
    extract_segments,
    remap_ids,
    solve_flow,
)
from pose_estimation.cricket.geometry import upper_body_ground_estimate
from scripts.global_id.track_manager import TrackManager
from scripts.tracking.calibration import (
    build_ground_calibrators,
    load_image_sizes_from_drive,
    load_projection_matrices_from_drive,
)
from scripts.tracking.runner import discover_prediction_files, infer_match_id


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_global_id(
    input_run_dir: str | Path,
    output_run_dir: str | Path,
    drive_root: str | Path,
    delivery_id: str,
    config: P4Config,
    cameras: list[str] | None = None,
    expected_frames: int = 600,
    ground_truth: str | Path | None = None,
) -> dict:
    input_run_dir = Path(input_run_dir)
    output_run_dir = Path(output_run_dir)
    drive_root = Path(drive_root)
    prediction_files = discover_prediction_files(input_run_dir, delivery_id, cameras)
    records_by_frame = load_synchronized_records(prediction_files, delivery_id)
    correspondence_input = input_run_dir / "diagnostics" / "correspondences.jsonl"
    if not correspondence_input.exists():
        raise FileNotFoundError(f"missing P3 correspondence artifact: {correspondence_input}")
    correspondence_rows = read_correspondence_rows(correspondence_input)
    correspondence_by_frame = {row["frame_index"]: row for row in correspondence_rows}
    missing_rows = sorted(set(records_by_frame) - set(correspondence_by_frame))
    if missing_rows:
        raise ValueError(f"P3 correspondences are missing {len(missing_rows)} prediction frames")

    manager = TrackManager(config)
    # Keep object references outside contract records for final ID backfill.
    player_track_references: list[tuple[dict, GlobalTrack]] = []
    ground_observations: list[tuple[GlobalTrack, int, np.ndarray]] = []
    for frame_index in sorted(records_by_frame):
        camera_records = records_by_frame[frame_index]
        correspondences = row_to_correspondences(correspondence_by_frame[frame_index], camera_records)
        assignments = manager.update(correspondences, frame_index)
        touched: set[tuple[str, int]] = set()
        for correspondence in correspondences:
            track = assignments.get(correspondence.cluster_id)
            for camera_id, detection in correspondence.members.items():
                member_track = manager.last_member_assignments.get(
                    (correspondence.cluster_id, camera_id), track
                )
                player = camera_records[camera_id]["players"][detection.player_index]
                player["single_camera"] = correspondence.single_camera
                player["track_confidence"] = float(np.clip(correspondence.track_confidence, 0.0, 1.0))
                if member_track is None:
                    player["global_player_id"] = None
                    player["track_state"] = "tentative"
                else:
                    player["global_player_id"] = member_track.global_player_id
                    player["track_state"] = member_track.state
                    player_track_references.append((player, member_track))
                touched.add((camera_id, detection.player_index))
            if track is not None and np.isfinite(correspondence.ground_xy).all():
                ground_observations.append((track, frame_index, correspondence.ground_xy.copy()))
        for camera_id, record in camera_records.items():
            for player_index, player in enumerate(record.get("players", [])):
                if (camera_id, player_index) not in touched:
                    player["global_player_id"] = None
                    player["track_state"] = "tentative"
                    player["single_camera"] = True
                    player["track_confidence"] = 0.0

    manager.finalize()
    for player, track in player_track_references:
        if track.global_player_id is not None:
            player["global_player_id"] = track.global_player_id

    records = [
        records_by_frame[frame][camera]
        for frame in sorted(records_by_frame)
        for camera in sorted(records_by_frame[frame])
    ]

    # Independent per-detection ground points (bbox bottom-centre projected to
    # z=0 via calibration only, NOT via P3 clustering) feed the cross-camera
    # agreement and teleport tripwires -- so those metrics judge the clustering
    # rather than echo it.
    try:
        ground_calibrators = build_ground_calibrators(
            drive_root, infer_match_id(delivery_id), cameras
        )
    except (FileNotFoundError, ValueError):
        ground_calibrators = {}
    try:
        metric_projections = load_projection_matrices_from_drive(
            drive_root, infer_match_id(delivery_id)
        )
    except (FileNotFoundError, ValueError):
        metric_projections = {}
    try:
        metric_image_wh = load_image_sizes_from_drive(drive_root, infer_match_id(delivery_id))
    except (FileNotFoundError, ValueError):
        metric_image_wh = {}
    # A bbox cut off at the frame bottom with no confident ankle projects the
    # FRAME edge, not the player; anchor those on an upper-body height plane
    # instead (calibration + detection only — still independent of the P3
    # clustering these metrics judge). The decision is STICKY per tracklet so
    # borderline frames don't flip between anchors and read as teleports.
    def _feet_unusable_for_metrics(player: dict, image_h: int) -> bool:
        bbox = player.get("bbox_xywh_px")
        if not bbox:
            return False
        conf = np.asarray((player.get("pose_2d") or {}).get("confidence", []), dtype=float)
        return float(bbox[1]) + float(bbox[3]) >= image_h - 4 and (
            conf.shape != (17,) or float(np.max(conf[[15, 16]])) < 0.6
        )

    anchor_votes: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        camera_id = str(record["camera_id"])
        image_h = metric_image_wh.get(camera_id, (0, 1 << 30))[1]
        for player in record.get("players", []):
            local_track_id = player.get("local_track_id")
            if not local_track_id:
                continue
            tally = anchor_votes[(camera_id, local_track_id)]
            tally[0] += 1
            tally[1] += int(_feet_unusable_for_metrics(player, image_h))
    sticky_approx = {
        key for key, (total, unusable) in anchor_votes.items()
        if total > 0 and unusable / total >= 0.6
    }

    detection_ground_positions: dict[tuple[int, str, int], np.ndarray] = {}
    approximated_keys: set[tuple[int, str, int]] = set()
    for record in records:
        calibrator = ground_calibrators.get(str(record["camera_id"]))
        if calibrator is None:
            continue
        frame_index = int(record["frame_index"])
        camera_id = str(record["camera_id"])
        image_h = metric_image_wh.get(camera_id, (0, 1 << 30))[1]
        projection = metric_projections.get(camera_id)
        for player_index, player in enumerate(record.get("players", [])):
            if not player.get("global_player_id"):
                continue
            bbox = player.get("bbox_xywh_px")
            if not bbox:
                continue
            bbox = [float(v) for v in bbox]
            local_track_id = player.get("local_track_id")
            if local_track_id:
                approximate = (camera_id, local_track_id) in sticky_approx
            else:
                approximate = _feet_unusable_for_metrics(player, image_h)
            xy = None
            if approximate and projection is not None:
                pose = player.get("pose_2d") or {}
                keypoints = np.asarray(pose.get("keypoints_px", []), dtype=float)
                conf = np.asarray(pose.get("confidence", []), dtype=float)
                if keypoints.shape != (17, 2):
                    keypoints = np.zeros((17, 2))
                if conf.shape != (17,):
                    conf = np.zeros(17)
                estimate = upper_body_ground_estimate(keypoints, conf, bbox, projection)
                if estimate is not None:
                    xy = estimate[0]
            if xy is None:
                xy = calibrator.bbox_bottom_center_ground_xy(bbox)
            if xy is not None:
                key = (frame_index, camera_id, player_index)
                detection_ground_positions[key] = xy
                if approximate:
                    approximated_keys.add(key)

    # Ground-plane track table from the online (P4a) assignments.
    ground_positions_accumulator: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for track, frame_index, ground_xy in ground_observations:
        if track.global_player_id is not None:
            ground_positions_accumulator[(track.global_player_id, frame_index)].append(ground_xy)
    ground_positions = {
        key: np.mean(np.asarray(values, dtype=float), axis=0)
        for key, values in ground_positions_accumulator.items()
    }

    # P4b: bridge fragmented confirmed tracklets with a min-cost-flow path cover
    # (temporal + spatial + role + velocity-continuity costs). ``remap_ids`` forbids any
    # merge whose two histories ever share a (camera, frame) cell, so post-hoc stitching
    # can never manufacture a same-camera-frame identity collision.
    segments = extract_segments(records, ground_positions)
    edges = build_link_costs(segments, config) if config.p4b.enabled else []
    links = solve_flow(segments, edges, config) if config.p4b.enabled else {}
    switch_report = remap_ids(records, segments, links) if config.p4b.enabled else []

    id_remap = {entry["merged_id"]: entry["into_id"] for entry in switch_report}
    for player_id in list(id_remap):
        while id_remap[player_id] in id_remap:
            id_remap[player_id] = id_remap[id_remap[player_id]]
    stitched_ground: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for (player_id, frame_index), point in ground_positions.items():
        stitched_ground[(id_remap.get(player_id, player_id), frame_index)].append(point)
    ground_rows_by_frame: dict[int, list[dict]] = defaultdict(list)
    for (player_id, frame_index), points in stitched_ground.items():
        ground_rows_by_frame[frame_index].append(
            {"global_player_id": player_id, "ground_xy": np.mean(points, axis=0).tolist()}
        )

    for record in records:
        validate_group1_frame(record, final_handoff=False)
    write_prediction_streams(records_by_frame, prediction_files, output_run_dir / "predictions")
    output_correspondences = output_run_dir / "diagnostics" / "correspondences.jsonl"
    write_jsonl(correspondence_rows, output_correspondences)
    ground_tracks_path = output_run_dir / "diagnostics" / "ground_tracks.jsonl"
    write_jsonl(
        (
            {"frame_index": frame, "tracks": sorted(tracks, key=lambda item: item["global_player_id"])}
            for frame, tracks in sorted(ground_rows_by_frame.items())
        ),
        ground_tracks_path,
    )
    switch_report_path = output_run_dir / "id_switch_report.json"
    _write_json(switch_report_path, switch_report)

    state_counts = Counter(
        player.get("track_state", "unset")
        for record in records for player in record.get("players", [])
    )
    identity_proxy = identity_fragmentation_proxy(records, switch_report)
    collision_metrics = identity_collision_metrics(records)
    completeness = track_completeness(records)
    agreement_metrics = cross_camera_agreement(records, detection_ground_positions)
    # Teleports judge identity leaps from POSITION series; height-plane
    # approximated members carry ~1 m anchor bias that reads as hopping when
    # their cameras pop in/out of the mean, so only foot-anchored detections
    # vote on position (they still count for identity agreement above).
    teleport_metrics = teleport_proxy(
        records,
        {
            key: value for key, value in detection_ground_positions.items()
            if key not in approximated_keys
        },
        max_speed_mps=config.kinematic_v_max_mps,
        frame_rate_fps=config.frame_rate_fps,
    )
    # Regression guard: the longest-lived tracks are, empirically, the bowler and
    # other well-tracked players. A/B comparison must not let their camera-support
    # or confirmed completeness drop.
    dominant_tracks = sorted(
        (
            {"global_player_id": player_id, **stats}
            for player_id, stats in completeness["per_track"].items()
        ),
        key=lambda item: (item["observed_frames"], item["camera_count"]),
        reverse=True,
    )[:3]
    metrics = {
        "schema_version": "global_id_metrics/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "match_id": infer_match_id(delivery_id),
        "delivery_id": delivery_id,
        "status": (
            "pass"
            if identity_proxy["distinct_global_id_count"] > 0
            and collision_metrics["same_camera_identity_collision_frames"] == 0
            else "fail"
        ),
        "frames_processed": len(records_by_frame),
        "expected_frames": expected_frames,
        "camera_count": len(prediction_files),
        "track_state_record_counts": dict(sorted(state_counts.items())),
        "online_tracker": dict(sorted(manager.diagnostics.items())),
        "segment_count": len(segments),
        "feasible_stitch_edge_count": len(edges),
        "selected_stitch_link_count": len(switch_report),
        **identity_proxy,
        **collision_metrics,
        **agreement_metrics,
        **teleport_metrics,
        "dominant_tracks": dominant_tracks,
        "completeness": completeness,
    }
    if ground_truth is not None:
        metrics["ground_truth"] = evaluate_ground_truth(records, _read_jsonl(ground_truth))

    manifest = {
        "schema_version": "global_id_run/v1",
        "created_at": metrics["created_at"],
        "task": "global_id_tracking",
        "input_run_dir": str(input_run_dir),
        "output_run_dir": str(output_run_dir),
        "drive_root": str(drive_root),
        "match_id": metrics["match_id"],
        "delivery_id": delivery_id,
        "expected_frames": expected_frames,
        "config": config.to_dict(),
        "ground_truth": str(ground_truth) if ground_truth is not None else None,
        "inputs": [
            {
                "prediction_jsonl": str(item.path),
                "capture_group": item.capture_group,
                "camera_id": item.camera_id,
            }
            for item in prediction_files
        ],
        "artifacts": {
            "correspondences_jsonl": str(output_correspondences),
            "ground_tracks_jsonl": str(ground_tracks_path),
            "id_switch_report_json": str(switch_report_path),
        },
    }
    _write_json(output_run_dir / "run_manifest.json", manifest)
    _write_json(output_run_dir / "global_id_metrics.json", metrics)
    return metrics
