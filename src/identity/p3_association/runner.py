"""Sequential per-delivery orchestration for P3 cross-camera association."""

from __future__ import annotations

import json
from dataclasses import replace
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import cv2

from identity.common.geometry import derive_facing_pairs
from identity.common.metrics import (
    association_proxy_metrics,
    pair_link_churn,
)
from identity.p3_association.associator import (
    AnchorState,
    TemporalLinkMemory,
    associate_frame,
    mark_contested_detections,
    select_anchor,
    smooth_emit_feet,
)
from identity.p3_association.config import P3AssociationConfig
from identity.p3_association.cue_calibration import CueCalibration
from identity.p3_association.geometry_cache import build_geometry_cache
from identity.p3_association.tracklet_graph import (
    TrackletGraphBuilder,
    apply_feet_approximation,
)
from identity.p3_association.jsonl_io import (
    apply_correspondences,
    correspondence_row,
    load_synchronized_records,
    record_to_detections,
    write_correspondence_rows,
    write_prediction_streams,
)
from core.calibration import (
    build_ground_calibrators,
    current_calibration_dir,
    load_image_sizes_from_drive,
    load_projection_matrices_from_drive,
)
from identity.p2_tracking.runner import discover_prediction_files, infer_match_id


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_association(
    input_run_dir: str | Path,
    output_run_dir: str | Path,
    drive_root: str | Path,
    delivery_id: str,
    config: P3AssociationConfig,
    cameras: list[str] | None = None,
    expected_frames: int = 600,
) -> dict:
    """Associate one delivery across all selected cameras and write a canonical P3 run."""

    input_run_dir = Path(input_run_dir)
    output_run_dir = Path(output_run_dir)
    drive_root = Path(drive_root)
    prediction_files = discover_prediction_files(input_run_dir, delivery_id, cameras)
    if len({item.camera_id for item in prediction_files}) < 2:
        raise ValueError("cross-camera association requires at least two camera streams")

    match_id = infer_match_id(delivery_id)
    all_projections = load_projection_matrices_from_drive(drive_root, match_id)
    camera_ids = sorted({item.camera_id for item in prediction_files})
    missing = sorted(set(camera_ids) - set(all_projections))
    if missing:
        raise ValueError(f"calibration is missing selected cameras: {missing}")
    projections = {camera: all_projections[camera] for camera in camera_ids}
    # Calibration is the source of truth for which cameras co-observe. Auto-derive the
    # FACING pairs (anti-parallel optical axes looking at the same ground strip) and
    # override any hand-edited opposite_camera_pairs so a wrong config cannot mislabel
    # the stricter opposite-pair ground gate. See src/identity/common/geometry.py.
    derived_facing_pairs = derive_facing_pairs(projections)
    if derived_facing_pairs:
        config = replace(
            config, opposite_camera_pairs=[list(pair) for pair in derived_facing_pairs]
        )
    ground_calibrators = build_ground_calibrators(drive_root, match_id, camera_ids)
    # Per-camera native image sizes so the epipole-in-image degeneracy test is
    # correct for the non-uniform rig (C07 is ~3775x960, not 2560x1440).
    try:
        image_wh_by_cam = load_image_sizes_from_drive(drive_root, match_id)
    except (FileNotFoundError, ValueError):
        image_wh_by_cam = None
    geometry = build_geometry_cache(projections, config, image_wh_by_cam=image_wh_by_cam)
    calibration_preflight = {
        "camera_count": len(projections),
        "pair_count": len(geometry.pairs),
        "finite_camera_center_count": sum(
            bool(np.isfinite(center).all()) for center in geometry.camera_centers.values()
        ),
        "finite_fundamental_matrix_count": sum(
            bool(np.isfinite(pair.F).all()) for pair in geometry.pairs.values()
        ),
    }
    if calibration_preflight["finite_camera_center_count"] != len(projections):
        raise ValueError("calibration preflight found non-finite camera centres")
    if calibration_preflight["finite_fundamental_matrix_count"] != len(geometry.pairs):
        raise ValueError("calibration preflight found non-finite fundamental matrices")
    records_by_frame = load_synchronized_records(prediction_files, delivery_id)

    # Pass A: build (and cache) per-frame detections once — including appearance
    # descriptors, which require the frame images. Frame decode is prefetched on a
    # small thread pool (W5-PERF): decoding 7 cameras x 2560x1440 serially left the
    # solver idle ~70% of Pass A; workers decode frame f+1..f+2 while f is processed.
    # Order and outputs are unchanged (byte-identical, verified) — only the imread
    # moves off the main thread.
    detections_by_frame: dict[int, dict] = {}
    appearance_images_missing = 0
    frame_order = sorted(records_by_frame)

    def _decode_frame_images(frame_index: int) -> dict[str, "np.ndarray | None"]:
        images: dict[str, "np.ndarray | None"] = {}
        if not config.appearance_enabled:
            return images
        for camera_id, record in sorted(records_by_frame[frame_index].items()):
            camera_number = int(camera_id.rsplit("_", 1)[1])
            image_path = (
                drive_root / "dataset" / record["capture_group"] / delivery_id
                / f"camera{camera_number:02d}" / record["frame_name"]
            )
            images[camera_id] = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        return images

    decode_pool = ThreadPoolExecutor(max_workers=4) if config.appearance_enabled else None
    inflight: "deque[tuple[int, Future]]" = deque()
    if decode_pool is not None:
        cv2.setNumThreads(1)  # 4 workers x 1 thread; never oversubscribe under --jobs N
        for frame_index in frame_order[:3]:
            inflight.append((frame_index, decode_pool.submit(_decode_frame_images, frame_index)))
    for position, frame_index in enumerate(frame_order):
        if decode_pool is not None:
            queued_index, future = inflight.popleft()
            assert queued_index == frame_index
            nxt = position + 3
            if nxt < len(frame_order):
                inflight.append(
                    (frame_order[nxt], decode_pool.submit(_decode_frame_images, frame_order[nxt]))
                )
            frame_images = future.result()
        else:
            frame_images = {}
        camera_records = records_by_frame[frame_index]
        detections = {}
        for camera_id, record in sorted(camera_records.items()):
            image = None
            if config.appearance_enabled:
                image = frame_images.get(camera_id)
                if image is None:
                    appearance_images_missing += 1
            detections[camera_id] = record_to_detections(
                record,
                ground_calibrators[camera_id],
                image,
                ankle_confidence_min=config.ankle_conf_min,
                max_ankle_above_bbox_fraction=config.max_ankle_above_bbox_fraction,
                # detection.ground_xy feeds the clustering GATE -> keep it on the legacy
                # foot so identity is invariant to foot_contact_mode (v2 affects only the
                # emitted z0_reproj position, in the associator).
            )
        if config.contested_iou > 0.0:
            # Wave-5b: flag same-camera overlapping detections (e.g. bowler crossing
            # the non-striker in a facing pair) so downstream evidence rides on the
            # cameras where the players are distinct.
            detections = mark_contested_detections(detections, config.contested_iou)
        detections_by_frame[frame_index] = detections
    if decode_pool is not None:
        decode_pool.shutdown(wait=False)
    if config.association_mode == "tracklet_graph":
        # Feet cut off at the frame edge project garbage ground points; re-anchor
        # those detections on upper-body height planes (sticky per tracklet).
        image_h_by_cam = {
            camera_id: wh[1] for camera_id, wh in (image_wh_by_cam or {}).items()
        }
        detections_by_frame = apply_feet_approximation(
            detections_by_frame, projections, image_h_by_cam, config
        )

    # Temporally smooth the emitted foot pixel per (camera, tracklet) (F7). Emit-only:
    # it never touches detection.ground_xy (the gate), so identity/clustering is
    # unchanged; it only steadies the reported ground position. No-op when window <= 1.
    detections_by_frame = smooth_emit_feet(detections_by_frame, config)

    # Pass B (tracklet_graph mode): accumulate tracklet-pair evidence over the
    # whole delivery, calibrate cue LLRs, and solve persistent identity bindings.
    graph_builder: TrackletGraphBuilder | None = None
    graph_solution = None
    if config.association_mode == "tracklet_graph":
        graph_builder = TrackletGraphBuilder(config, projections)
        for frame_index in sorted(detections_by_frame):
            graph_builder.observe_frame(frame_index, detections_by_frame[frame_index])
        if config.calibration_mode == "file" and config.cue_calibration_path:
            calibration = CueCalibration.load(config.cue_calibration_path)
        elif config.calibration_mode == "defaults":
            calibration = CueCalibration()
        else:
            calibration = graph_builder.harvest_calibration()
        graph_solution = graph_builder.solve(calibration)

    # Pass C: per-frame correspondences — from stable bindings in graph mode,
    # from the historical per-frame clustering otherwise.
    anchor: AnchorState | None = None
    temporal_memory = TemporalLinkMemory(
        confirm_frames=config.temporal_confirm_frames,
        decay=config.temporal_link_decay,
    )
    anchor_switch_frames: list[int] = []
    rows: list[dict] = []
    # F9c: multi-view member keypoints per binding, sampled every graph_lift_stride
    # frames, for the post-solve purity lift (read-only diagnostics).
    lift_samples: dict[str, list[dict]] = {}
    for frame_index in sorted(records_by_frame):
        camera_records = records_by_frame[frame_index]
        detections = detections_by_frame[frame_index]
        previous_anchor = anchor.anchor_id if anchor is not None else None
        if graph_builder is not None and graph_solution is not None:
            anchor = select_anchor(detections, anchor, config)
            correspondences = graph_builder.emit_frame(
                frame_index, detections, graph_solution, projections,
                geometry.camera_centers,
            )
        else:
            correspondences, anchor = associate_frame(
                detections, projections, geometry, anchor, config, temporal_memory
            )
        if previous_anchor is not None and anchor.anchor_id != previous_anchor:
            anchor_switch_frames.append(frame_index)
        if (
            config.graph_lift_feedback
            and graph_solution is not None
            and frame_index % config.graph_lift_stride == 0
        ):
            for corr in correspondences:
                if corr.binding_id is None or len(corr.members) < 2:
                    continue
                lift_samples.setdefault(corr.binding_id, []).append({
                    cam_id: np.column_stack(
                        (det.keypoints_px, det.keypoint_conf)
                    )
                    for cam_id, det in corr.members.items()
                })
        apply_correspondences(camera_records, correspondences)
        row = correspondence_row(frame_index, correspondences)
        row["anchor_camera_id"] = anchor.anchor_id
        rows.append(row)

    write_prediction_streams(records_by_frame, prediction_files, output_run_dir / "predictions")
    correspondence_path = output_run_dir / "diagnostics" / "correspondences.jsonl"
    write_correspondence_rows(rows, correspondence_path)
    if graph_solution is not None:
        _write_json(
            output_run_dir / "diagnostics" / "tracklet_graph.json",
            graph_solution.to_json(),
        )
    lift_purity_by_binding: dict[str, dict] = {}
    if config.graph_lift_feedback and lift_samples:
        from identity.p3_association.cluster_lift import cluster_purity, lift_frame

        for binding, frames in sorted(lift_samples.items()):
            lifts = [
                lift for members in frames
                if (lift := lift_frame(
                    members, projections,
                    reprojection_threshold_px=config.triangulation_reproj_threshold_px,
                    min_views=config.triangulation_min_views,
                )) is not None
            ]
            lift_purity_by_binding[binding] = cluster_purity(
                lifts,
                chimera_torso_residual_px=config.graph_chimera_torso_residual_px,
                chimera_frame_fraction=config.graph_chimera_frame_fraction,
            ).to_json()
        _write_json(
            output_run_dir / "diagnostics" / "lift_purity.json",
            {
                "schema_version": "lift_purity/v1",
                "delivery_id": delivery_id,
                "sampled_every_frames": config.graph_lift_stride,
                "thresholds": {
                    "chimera_torso_residual_px": config.graph_chimera_torso_residual_px,
                    "chimera_frame_fraction": config.graph_chimera_frame_fraction,
                },
                "bindings": lift_purity_by_binding,
            },
        )

    created_at = datetime.now(timezone.utc).isoformat()
    degenerate_pairs = [
        list(pair) for pair, pair_geometry in sorted(geometry.pairs.items())
        if pair_geometry.is_degenerate
    ]
    degenerate_pair_set = {tuple(pair) for pair in degenerate_pairs}
    matched_pair_count = 0
    degenerate_pair_usage_count = 0
    for row in rows:
        for cluster in row["clusters"]:
            cameras_in_cluster = sorted(member["cam_id"] for member in cluster["members"])
            for camera_pair in combinations(cameras_in_cluster, 2):
                matched_pair_count += 1
                degenerate_pair_usage_count += int(camera_pair in degenerate_pair_set)
    proxy = association_proxy_metrics(rows, anchor_switch_frames=anchor_switch_frames)
    churn = pair_link_churn(rows)
    frame_counts = {
        camera: sum(camera in per_camera for per_camera in records_by_frame.values())
        for camera in camera_ids
    }
    graph_metrics: dict = {"association_mode": config.association_mode}
    if lift_purity_by_binding:
        suspects = [
            binding for binding, report in lift_purity_by_binding.items()
            if report.get("chimera_suspect")
        ]
        graph_metrics["lift_bindings_evaluated"] = len(lift_purity_by_binding)
        graph_metrics["lift_chimera_suspect_count"] = len(suspects)
        graph_metrics["lift_chimera_suspects"] = suspects
    if graph_solution is not None:
        graph_metrics.update({
            "binding_count": len(graph_solution.clusters),
            "multi_camera_binding_count": sum(
                len({key[0] for key in keys}) >= 2
                for keys in graph_solution.clusters.values()
            ),
            "graph_edge_count": len(graph_solution.edges),
            "graph_diagnostics": graph_solution.diagnostics,
            "cue_d_prime": {
                cue: round(graph_solution.calibration.d_prime(cue), 3)
                for cue in sorted(graph_solution.calibration.distributions)
            },
            "calibration_anchor_pair_count": graph_solution.calibration.anchor_pair_count,
            "calibration_diff_pair_count": graph_solution.calibration.diff_pair_count,
        })
    metrics = {
        "schema_version": "association_metrics/v1",
        "created_at": created_at,
        "match_id": match_id,
        "delivery_id": delivery_id,
        "status": "pass" if proxy["cluster_count"] > proxy["single_camera_cluster_count"] else "fail",
        "frames_processed": len(records_by_frame),
        "expected_frames": expected_frames,
        "frames_per_camera": frame_counts,
        "camera_count": len(camera_ids),
        "facing_camera_pairs": [list(pair) for pair in derived_facing_pairs],
        "degenerate_pairs": degenerate_pairs,
        "degenerate_pair_count": len(degenerate_pairs),
        "matched_camera_pair_count": matched_pair_count,
        "degenerate_pair_usage_count": degenerate_pair_usage_count,
        "degenerate_pair_usage_rate": (
            degenerate_pair_usage_count / matched_pair_count if matched_pair_count else 0.0
        ),
        "calibration_preflight": calibration_preflight,
        "appearance_images_missing": appearance_images_missing,
        **proxy,
        **churn,
        **graph_metrics,
    }
    manifest = {
        "schema_version": "association_run/v1",
        "created_at": created_at,
        "task": "cross_camera_association",
        "input_run_dir": str(input_run_dir),
        "output_run_dir": str(output_run_dir),
        "drive_root": str(drive_root),
        "calibration_dir": str(current_calibration_dir(drive_root, match_id)),
        "match_id": match_id,
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
        "artifacts": {"correspondences_jsonl": str(correspondence_path)},
        "calibration_preflight": calibration_preflight,
    }
    _write_json(output_run_dir / "run_manifest.json", manifest)
    _write_json(output_run_dir / "association_metrics.json", metrics)
    return metrics
