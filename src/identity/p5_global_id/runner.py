"""P4a online global IDs followed by P4b delivery-level stitching."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.contract import validate_group1_frame
from identity.common.metrics import (
    colocated_identity_metrics,
    cross_camera_agreement,
    identity_collision_metrics,
    identity_fragmentation_proxy,
    teleport_proxy,
    track_completeness,
)
from identity.p3_association.jsonl_io import load_synchronized_records
from identity.p5_global_id.config import P4Config
from identity.p5_global_id.global_track import GlobalTrack
from identity.p5_global_id.jsonl_io import (
    read_correspondence_rows,
    row_to_correspondences,
    write_jsonl,
    write_prediction_streams,
)
from identity.p5_global_id.stitching import (
    merge_colocated_ids,
    build_link_costs,
    extract_segments,
    remap_ids,
    solve_flow,
)
from identity.common.geometry import upper_body_ground_estimate, pixel_to_plane_xy
from identity.p5_global_id.role_proxy import OnlineRoleProxy
from identity.p5_global_id.track_manager import TrackManager
from core.calibration import current_calibration_dir
from identity.visualization.mosaic_layout import load_pitch_axis
from core.calibration import (
    build_ground_calibrators,
    load_image_sizes_from_drive,
    load_projection_matrices_from_drive,
)
from identity.p2_tracking.runner import discover_prediction_files, infer_match_id


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _single_view_hip_positions(
    records: list[dict], projections: dict, id_remap: dict,
    min_conf: float = 0.3, min_z_frames: int = 5,
) -> dict[tuple[str, int], np.ndarray]:
    """1F: for each FINAL global id, learn a STICKY hip height (median hip-z over its
    multi-camera triangulated pose_3d frames), then for single-camera frames back-project
    that frame's hip PIXEL onto the per-id height plane -> a stable hip-on-ground xy. Keyed
    by (final_id, frame); only single-camera frames of ids with enough triangulated hips
    get an entry (the caller uses it only where no triangulated hip exists that frame).
    """

    hip_z: dict[str, list[float]] = defaultdict(list)
    obs: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = defaultdict(list)
    for record in records:
        cam = str(record.get("camera_id", "unknown"))
        fi = int(record["frame_index"])
        for player in record.get("players", []):
            gid = player.get("global_player_id")
            if not gid:
                continue
            final = id_remap.get(gid, gid)
            kw = (player.get("pose_3d") or {}).get("keypoints_world_m")
            if kw:
                for j in (11, 12):
                    v = kw[j] if j < len(kw) else None
                    if v is not None and np.isfinite(v[2]):
                        hip_z[final].append(float(v[2]))
            pose2 = player.get("pose_2d") or {}
            kps = np.asarray(pose2.get("keypoints_px", []), dtype=float)
            conf = np.asarray(pose2.get("confidence", []), dtype=float)
            if kps.shape[:1] == (26,) and conf.shape == (26,):
                sel = [j for j in (11, 12) if conf[j] >= min_conf]
                if sel:
                    obs[(final, fi)].append((cam, kps[sel].mean(axis=0)))
    sticky = {g: float(np.median(zs)) for g, zs in hip_z.items() if len(zs) >= min_z_frames}
    out: dict[tuple[str, int], np.ndarray] = {}
    for (gid, fi), cams in obs.items():
        if len(cams) == 1 and gid in sticky:
            cam, hip_px = cams[0]
            P = projections.get(cam)
            if P is None:
                continue
            xy = pixel_to_plane_xy(hip_px, np.asarray(P, dtype=float), sticky[gid])
            if np.isfinite(xy).all():
                out[(gid, fi)] = xy
    return out


def _partial_singlecam_ids(records: list[dict], id_remap: dict, min_visible_kpts: int) -> set[str]:
    """IMPACT-2 (emission level): the set of FINAL global ids that are SINGLE-camera
    across the whole delivery AND predominantly partial (median confident-keypoint count
    below ``min_visible_kpts`` — a head-only view of the keeper, a cut-off umpire). Such a
    detection has an unreliable ground position and mints a rival id / ghost. The caller
    DROPS these (detections go unlabelled) — drop-only, never a relabel, so unlike the
    rejected tracklet lock this can never put an id on the wrong person. Full-body
    single-camera peripherals (many confident keypoints) are spared by the keypoint floor.
    """

    cameras: dict[str, set[str]] = defaultdict(set)
    visible: dict[str, list[int]] = defaultdict(list)
    for record in records:
        cam = str(record.get("camera_id", "unknown"))
        for player in record.get("players", []):
            gid = player.get("global_player_id")
            if not gid:
                continue
            final = id_remap.get(gid, gid)
            cameras[final].add(cam)
            conf = np.asarray((player.get("pose_2d") or {}).get("confidence", []), dtype=float)
            visible[final].append(int(np.sum(conf > 0.3)))
    return {
        gid for gid, cams in cameras.items()
        if len(cams) == 1 and visible[gid]
        and float(np.median(visible[gid])) < min_visible_kpts
    }


def _velocity_gate_ground_rows(
    ground_rows_by_frame: dict[int, list[dict]],
    *,
    frame_rate_fps: float,
    max_mps: float,
    max_consec_drops: int,
) -> tuple[dict[int, list[dict]], int]:
    """A3 emitted-track velocity gate (drop-only). Walk each global id's emitted ground
    track in frame order; drop any frame whose implied speed from the LAST KEPT frame
    exceeds ``max_mps`` (a physical cricketer never exceeds ~10-11 m/s). The gap between
    frames scales the allowance, so an id re-appearing after an absence may move
    proportionally. After ``max_consec_drops`` consecutive drops the gate RE-ANCHORS to
    the current position (a sustained move / re-acquisition is accepted rather than the
    whole tail deleted). Never moves or relabels a position — only removes teleport frames.
    Returns the filtered rows-by-frame (frames with no surviving row are omitted) and the
    number of dropped emissions.
    """

    # id -> ordered [(frame, xy, row)]
    series: dict[str, list[tuple[int, np.ndarray, dict]]] = defaultdict(list)
    for frame_index in sorted(ground_rows_by_frame):
        for row in ground_rows_by_frame[frame_index]:
            series[row["global_player_id"]].append(
                (frame_index, np.asarray(row["ground_xy"], dtype=float), row)
            )

    dropped: set[tuple[str, int]] = set()
    for gid, points in series.items():
        points.sort(key=lambda item: item[0])
        last_frame, last_pos, _ = points[0]  # always keep the first frame
        consec = 0
        for frame_index, pos, _row in points[1:]:
            gap = max(1, frame_index - last_frame)
            speed = float(np.linalg.norm(pos - last_pos)) * frame_rate_fps / gap
            if speed <= max_mps:
                last_frame, last_pos, consec = frame_index, pos, 0
            elif consec >= max_consec_drops:
                # sustained divergence -> accept as a genuine relocation, re-anchor
                last_frame, last_pos, consec = frame_index, pos, 0
            else:
                dropped.add((gid, frame_index))
                consec += 1

    if not dropped:
        return ground_rows_by_frame, 0
    filtered: dict[int, list[dict]] = {}
    for frame_index, rows in ground_rows_by_frame.items():
        kept = [r for r in rows if (r["global_player_id"], frame_index) not in dropped]
        if kept:
            filtered[frame_index] = kept
    return filtered, len(dropped)


def run_global_id(
    input_run_dir: str | Path,
    output_run_dir: str | Path,
    drive_root: str | Path,
    delivery_id: str,
    config: P4Config,
    cameras: list[str] | None = None,
    expected_frames: int = 600,
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

    # Triangulated-hip emission source (Phase-1A). The 04 lift writes the RANSAC-
    # triangulated pelvis (mean of COCO hips 11/12) projected to z=0 as
    # ``pelvis_ground_xy`` in lift3d.jsonl, keyed by (binding_id, frame). When the
    # flag is set we emit that single robust multi-view point instead of averaging
    # per-camera foot rays and then fragments (the ground-teleport source). Missing
    # file (e.g. 04 ran with --id-source global) -> warn and fall back to feet.
    hip_ground_by_binding_frame: dict[tuple[str, int], np.ndarray] = {}
    if config.p4a.emit_ground_source == "triangulated_hip":
        lift3d_path = input_run_dir / "diagnostics" / "lift3d.jsonl"
        if not lift3d_path.exists():
            print(
                f"WARNING: emit_ground_source=triangulated_hip but {lift3d_path} is "
                "missing (04 lift must run with --id-source binding); emitting feet.",
                flush=True,
            )
        else:
            with lift3d_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    frame_index = int(row["frame_index"])
                    for binding in row.get("bindings", []):
                        binding_id = binding.get("binding_id")
                        pelvis_xy = binding.get("pelvis_ground_xy")
                        if binding_id is not None and pelvis_xy is not None:
                            hip_ground_by_binding_frame[(str(binding_id), frame_index)] = (
                                np.asarray(pelvis_xy, dtype=float)
                            )

    manager = TrackManager(config)
    role_proxy: OnlineRoleProxy | None = None
    if config.p4a.online_role_proxy:
        # F5: role-aware Singer dynamics during tracking (bowler agile, umpire
        # near-static) instead of every player running the generic model.
        pitch_axis = load_pitch_axis(
            current_calibration_dir(drive_root, infer_match_id(delivery_id))
            / "pitch_calibration_config.json"
        )
        role_proxy = OnlineRoleProxy(
            pitch_axis,
            frame_rate_fps=config.frame_rate_fps,
            min_track_frames=config.p4a.proxy_min_track_frames,
            bowler_min_speed_mps=config.p4a.proxy_bowler_min_speed_mps,
            static_speed_max_mps=config.p4a.proxy_static_speed_max_mps,
        )
    # Keep object references outside contract records for final ID backfill.
    player_track_references: list[tuple[dict, GlobalTrack]] = []
    ground_observations: list[tuple[GlobalTrack, int, np.ndarray, str | None]] = []
    for frame_index in sorted(records_by_frame):
        camera_records = records_by_frame[frame_index]
        correspondences = row_to_correspondences(correspondence_by_frame[frame_index], camera_records)
        assignments = manager.update(correspondences, frame_index)
        if role_proxy is not None:
            role_proxy.observe(manager, frame_index)
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
                # manager.update() has already run this frame's Kalman update, so
                # track.kalman.pos_world_xy is the filtered (chi2-gated) posterior.
                if config.p4a.emit_kalman_posterior and np.isfinite(track.kalman.pos_world_xy).all():
                    emitted = track.kalman.pos_world_xy.copy()
                else:
                    emitted = correspondence.ground_xy.copy()
                binding_id = getattr(correspondence, "binding_id", None)
                ground_observations.append((track, frame_index, emitted, binding_id))
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
                if keypoints.shape != (26, 2):
                    keypoints = np.zeros((26, 2))
                if conf.shape != (26,):
                    conf = np.zeros(26)
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
    # (global_player_id, frame) -> binding_id, so the emit path can look up the
    # triangulated hip for this track/frame (Phase-1A hip emission).
    binding_by_gid_frame: dict[tuple[str, int], str] = {}
    for track, frame_index, ground_xy, binding_id in ground_observations:
        if track.global_player_id is not None:
            ground_positions_accumulator[(track.global_player_id, frame_index)].append(ground_xy)
            if binding_id is not None:
                binding_by_gid_frame[(track.global_player_id, frame_index)] = str(binding_id)
    ground_positions = {
        key: np.mean(np.asarray(values, dtype=float), axis=0)
        for key, values in ground_positions_accumulator.items()
    }

    # P4b: bridge fragmented confirmed tracklets with a min-cost-flow path cover
    # (temporal + spatial + role + velocity-continuity costs). ``remap_ids`` forbids any
    # merge whose two histories ever share a (camera, frame) cell, so post-hoc stitching
    # can never manufacture a same-camera-frame identity collision.
    # Accumulated pose-shape descriptor per P4a id, so stitching can require
    # body-shape agreement before merging two fragments (ID-2 / ghost-verification).
    pose_by_id: dict[str, object] = {}
    posture_by_id: dict[str, object] = {}
    for track in list(manager.tracks) + list(manager.deleted_pool):
        if track.global_player_id is None:
            continue
        if (
            track.pose_proportions is not None
            and track.pose_update_count >= config.p4a.pose_min_updates
        ):
            pose_by_id[track.global_player_id] = track.pose_proportions
        if track.posture is not None:
            posture_by_id[track.global_player_id] = track.posture
    segments = extract_segments(records, ground_positions, pose_by_id, posture_by_id)
    # (camera, frame) occupancy per identity for the F6 occupancy-licensed bridge
    # (same cells remap_ids uses for its merge veto).
    occupancy_by_id: dict[str, set[tuple[str, int]]] = defaultdict(set)
    if config.p4b.occupancy_bridge:
        for record in records:
            frame_index = int(record["frame_index"])
            camera_id = str(record.get("camera_id", "unknown"))
            for player in record.get("players", []):
                player_id = player.get("global_player_id")
                if player_id:
                    occupancy_by_id[player_id].add((camera_id, frame_index))
    edges = build_link_costs(segments, config, occupancy_by_id) if config.p4b.enabled else []
    links = solve_flow(segments, edges, config) if config.p4b.enabled else {}
    switch_report = remap_ids(records, segments, links) if config.p4b.enabled else []

    id_remap = {entry["merged_id"]: entry["into_id"] for entry in switch_report}
    for player_id in list(id_remap):
        while id_remap[player_id] in id_remap:
            id_remap[player_id] = id_remap[id_remap[player_id]]

    # W9: colocated-id merge — one physical player carrying two ids in disjoint
    # camera sets (the ghost-under-player swap). Runs after temporal stitching so
    # it sees final fragments; extends the same switch report.
    if config.p4b.colocated_merge:
        colocated_report = merge_colocated_ids(
            records, ground_positions, posture_by_id, id_remap,
            radius_m=config.p4b.colocated_radius_m,
            min_frames=config.p4b.colocated_min_frames,
            posture_max_z=config.p4b.colocated_posture_max_z,
        )
        switch_report = list(switch_report) + colocated_report

    # ID-2 cardinality prior: after stitching, drop any global id whose total
    # frame-span is below min_emit_frames (a fragment/shadow, not a real player who
    # is present the whole delivery). Applied to the FINAL (post-remap) ids so a
    # short fragment already stitched into a long track is safe.
    dropped_short_ids: set[str] = set()
    if config.p4a.min_emit_frames > 0:
        frames_by_id: dict[str, set[int]] = defaultdict(set)
        for record in records:
            frame_index = int(record["frame_index"])
            for player in record.get("players", []):
                pid = player.get("global_player_id")
                if pid is not None:
                    frames_by_id[id_remap.get(pid, pid)].add(frame_index)
        dropped_short_ids = {
            pid for pid, frames in frames_by_id.items()
            if len(frames) < config.p4a.min_emit_frames
        }
        if dropped_short_ids:
            for record in records:
                for player in record.get("players", []):
                    pid = player.get("global_player_id")
                    if pid is not None and id_remap.get(pid, pid) in dropped_short_ids:
                        player["global_player_id"] = None
                        player["track_state"] = "tentative"

    # IMPACT-2: drop single-camera predominantly-partial ids (head-only / cut-off ghosts)
    # at emission — drop-only, never relabel (see _partial_singlecam_ids).
    dropped_partial_ids: set[str] = set()
    if config.p4a.drop_partial_singlecam:
        dropped_partial_ids = _partial_singlecam_ids(records, id_remap, config.p4a.partial_min_visible_kpts)
        for record in records:
            for player in record.get("players", []):
                pid = player.get("global_player_id")
                if pid is not None and id_remap.get(pid, pid) in dropped_partial_ids:
                    player["global_player_id"] = None
                    player["track_state"] = "tentative"

    # 1F: single-camera hip-on-sticky-plane positions (only when hip emission is on).
    single_view_hip: dict[tuple[str, int], np.ndarray] = {}
    if config.p4a.emit_ground_source == "triangulated_hip" and config.p4a.single_view_hip_fallback:
        single_view_hip = _single_view_hip_positions(records, metric_projections, id_remap)

    stitched_ground: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for (player_id, frame_index), point in ground_positions.items():
        final_id = id_remap.get(player_id, player_id)
        if final_id in dropped_short_ids or final_id in dropped_partial_ids:
            continue
        emit_point = point
        if config.p4a.emit_ground_source == "triangulated_hip":
            # Prefer the single robust triangulated-hip point over the averaged
            # foot rays; fall back to the foot position when the hip was not
            # triangulable this frame (single-camera / <2 views).
            binding_id = binding_by_gid_frame.get((player_id, frame_index))
            hip_xy = (
                hip_ground_by_binding_frame.get((binding_id, frame_index))
                if binding_id is not None else None
            )
            if hip_xy is not None:
                emit_point = hip_xy
            elif single_view_hip:
                # 1F: no triangulated hip this frame (single-camera) -> hip pixel
                # back-projected onto the id's sticky hip-height plane.
                sv = single_view_hip.get((final_id, frame_index))
                if sv is not None:
                    emit_point = sv
        stitched_ground[(final_id, frame_index)].append(emit_point)
    ground_rows_by_frame: dict[int, list[dict]] = defaultdict(list)
    for (player_id, frame_index), points in stitched_ground.items():
        ground_rows_by_frame[frame_index].append(
            {"global_player_id": player_id, "ground_xy": np.mean(points, axis=0).tolist()}
        )

    # A3: drop teleport frames from the emitted ground tracks (drop-only, never relabel).
    if config.p4a.emit_velocity_gate:
        ground_rows_by_frame, n_gated = _velocity_gate_ground_rows(
            dict(ground_rows_by_frame),
            frame_rate_fps=config.frame_rate_fps,
            max_mps=config.p4a.emit_velocity_max_mps,
            max_consec_drops=config.p4a.emit_velocity_max_consec_drops,
        )
        if n_gated:
            print(f"  [A3] velocity gate dropped {n_gated} teleport emission(s)")

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
    # W9 tripwire: post-final occupancy per id + fused rows -> colocated swaps
    final_occupancy: dict[str, set] = defaultdict(set)
    for record in records:
        fi = int(record["frame_index"])
        cam = str(record.get("camera_id", "unknown"))
        for player in record.get("players", []):
            pid = player.get("global_player_id")
            if pid:
                final_occupancy[pid].add((cam, fi))
    colocated_metrics = colocated_identity_metrics(
        {
            frame: rows for frame, rows in ground_rows_by_frame.items()
        },
        final_occupancy,
    )
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
    # Clip-level verdict from the two metrics a human can trust blind: how many
    # identities were minted for a field that holds at most `expected_roster_max`
    # people, and how often an identity teleported. A clip failing here needs
    # work (usually upstream detection quality), not delivery to reviewers.
    roster_max = config.p4a.expected_roster_max
    distinct_ids = identity_proxy["distinct_global_id_count"]
    teleports = teleport_metrics["teleport_event_count"]
    collisions = collision_metrics["same_camera_identity_collision_frames"]
    coloc_pairs = colocated_metrics["colocated_disjoint_pair_count"]

    # Emitted big jumps: single-frame (>25 m/s) leaps in the EMITTED ground track
    # (the delivered posterior), NOT the raw-foot-projection teleport proxy. This is
    # the real smoothness signal (docs/diagnosis/04-issue-emitted-ground-teleports.md).
    _emit_series: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for frame_index, rows in ground_rows_by_frame.items():
        for row in rows:
            xy = np.asarray(row["ground_xy"], dtype=float)
            if np.isfinite(xy).all():
                _emit_series[row["global_player_id"]][frame_index] = xy
    emitted_big_jumps = 0
    for series in _emit_series.values():
        frames = sorted(series)
        for prev_f, curr_f in zip(frames, frames[1:]):
            if curr_f - prev_f != 1:
                continue
            speed = float(np.linalg.norm(series[curr_f] - series[prev_f])) * config.frame_rate_fps
            if speed > 25.0:
                emitted_big_jumps += 1

    verdict_reasons: list[str] = []
    if getattr(config.p4a, "usability_verdict", True):
        # Composite usability grade. Hard gates first, then a weighted score whose
        # dominant axis is cross-camera AGREEMENT (the primary identity axis the
        # legacy rule ignored). Coverage (a P6 metric) is folded in by the panel
        # re-grade tool; here we renormalize over the four P4-available axes.
        # Rubric + thresholds: docs/diagnosis/10-verdict-redesign.md.
        agreement = agreement_metrics["cross_camera_agreement_rate"]
        persist = (
            completeness.get("confirmed_frame_completeness", {}).get("mean", 0.0)
            if isinstance(completeness, dict) else 0.0
        )

        def _clamp(value: float) -> float:
            return max(0.0, min(1.0, value))

        sub = {
            "agreement": _clamp((agreement - 0.72) / 0.24),
            "smoothness": _clamp(1.0 - emitted_big_jumps / 50.0),
            "persistence": _clamp((persist - 0.80) / 0.18),
            "parsimony": _clamp((16 - distinct_ids) / 3.0),
        }
        weights = {"agreement": 0.45, "smoothness": 0.30, "persistence": 0.12, "parsimony": 0.13}
        score = sum(weights[k] * sub[k] for k in weights) - 0.10 * coloc_pairs
        score = max(0.0, score)

        gate = None
        if collisions > 0:
            gate = "same_camera_collision"
        elif distinct_ids > 20:
            gate = f"id_overmint: {distinct_ids} ids"
        elif agreement < 0.65:
            gate = f"identity_broken: agreement {agreement:.3f}"

        if gate is not None:
            verdict = "fail"
            verdict_reasons.append(gate)
        elif score >= 0.75:
            verdict = "good"
        elif score >= 0.55:
            verdict = "usable"
        elif score >= 0.40:
            verdict = "weak"
        else:
            verdict = "fail"
        limiting = min(sub, key=sub.get)
        verdict_reasons.append(f"limiting_axis: {limiting} ({sub[limiting]:.2f})")
        if coloc_pairs > 0:
            verdict_reasons.append(f"colocated_split_ids: {coloc_pairs} pairs")
        quality_verdict = {
            "verdict": verdict,
            "score": round(score, 3),
            "subscores": {k: round(v, 3) for k, v in sub.items()},
            "emitted_big_jumps": emitted_big_jumps,
            "reasons": verdict_reasons,
            "expected_roster_max": roster_max,
        }
    else:
        # Legacy teleport-proxy rule (recoverable for reproducibility).
        verdict = "pass"
        if distinct_ids > 2 * roster_max:
            verdict = "fail"
            verdict_reasons.append(f"id_overmint: {distinct_ids} ids for <= {roster_max} people")
        elif distinct_ids > int(1.2 * roster_max):
            verdict = "warn"
            verdict_reasons.append(f"id_overmint: {distinct_ids} ids for <= {roster_max} people")
        if teleports > 60:
            verdict = "fail"
            verdict_reasons.append(f"teleport_storm: {teleports} events")
        elif teleports > 20 and verdict != "fail":
            verdict = "warn"
            verdict_reasons.append(f"teleports_elevated: {teleports} events")
        if collisions > 0:
            verdict = "fail"
            verdict_reasons.append("same_camera_id_collision")
        if coloc_pairs > 0 and verdict != "fail":
            verdict = "warn"
            verdict_reasons.append(f"colocated_split_ids: {coloc_pairs} pairs")
        quality_verdict = {
            "verdict": verdict,
            "reasons": verdict_reasons,
            "expected_roster_max": roster_max,
        }

    metrics = {
        "schema_version": "global_id_metrics/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "match_id": infer_match_id(delivery_id),
        "delivery_id": delivery_id,
        "quality_verdict": quality_verdict,
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
        **colocated_metrics,
        "dominant_tracks": dominant_tracks,
        "completeness": completeness,
        "partial_singlecam_dropped": len(dropped_partial_ids),
    }

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
