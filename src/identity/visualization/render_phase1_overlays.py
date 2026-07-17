#!/usr/bin/env python3
"""Render visual QA overlays for cricket Phase 1 prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from core.keypoints import HALPE26_EDGES
from identity.visualization.identity_colors import color_for_player
from identity.visualization.loaders import (
    iter_jsonl,
    load_cluster_badges,
    stage_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="drive")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument(
        "--frame-ids",
        nargs="+",
        type=int,
        default=None,
        help="Render these exact frame IDs from each camera JSONL, e.g. 1 150 300 450 600.",
    )
    parser.add_argument(
        "--row-indices",
        nargs="+",
        type=int,
        default=None,
        help="Render these one-based JSONL row positions per camera, e.g. 1 150 300 450 600.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=None,
        help="Fallback sampler when neither --frame-ids nor --row-indices is set. Uses JSONL row position.",
    )
    parser.add_argument("--max-per-camera", type=int, default=5)
    parser.add_argument("--keypoint-threshold", type=float, default=0.2)
    parser.add_argument("--show", choices=["p2", "p3", "p4"], default=None)
    parser.add_argument(
        "--no-manifest-update",
        action="store_true",
        help="Do not add visualization metadata to run/delivery manifests.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def canonical_prediction_parts(path: Path) -> tuple[str, str, str] | None:
    parts = path.stem.split("__")
    if len(parts) != 3:
        return None
    group, delivery_id, camera_id = parts
    if not group or not delivery_id or not camera_id.startswith("cam_"):
        return None
    return group, delivery_id, camera_id


def camera_folder(camera_id: str) -> str:
    return f"camera{camera_id.replace('cam_', '')}"


def should_render_record(
    *,
    row_index: int,
    frame_index: int,
    row_indices: set[int] | None,
    rendered_count: int,
    frame_ids: set[int] | None,
    sample_every: int | None,
    max_per_camera: int,
) -> bool:
    if rendered_count >= max_per_camera:
        return False
    if frame_ids is not None:
        return frame_index in frame_ids
    if row_indices is not None:
        return (row_index + 1) in row_indices
    if sample_every is None:
        return rendered_count == 0
    return row_index % sample_every == 0


def display_id(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def confidence_text(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def pose_confidence_text(player: dict[str, Any]) -> str:
    confidence = player.get("pose_2d", {}).get("confidence") or []
    values = []
    for value in confidence:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.2f}"


def draw_label_block(image, lines: list[str], *, anchor: tuple[int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.46
    thickness = 1
    padding_x = 6
    padding_y = 5
    line_gap = 4
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    block_width = max(width for width, _ in text_sizes) + padding_x * 2
    block_height = sum(height for _, height in text_sizes) + line_gap * (len(lines) - 1) + padding_y * 2

    image_height, image_width = image.shape[:2]
    x = max(0, min(anchor[0], image_width - block_width - 1))
    y = anchor[1] - block_height - 6
    if y < 0:
        y = anchor[1] + 6
    y = max(0, min(y, image_height - block_height - 1))

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + block_width, y + block_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    cursor_y = y + padding_y
    for line, (_, text_height) in zip(lines, text_sizes):
        cursor_y += text_height
        cv2.putText(
            image,
            line,
            (x + padding_x, cursor_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += line_gap


def draw_record(
    image,
    record: dict[str, Any],
    *,
    keypoint_threshold: float,
    show: str = "p4",
    cluster_badges: dict[int, int] | None = None,
) -> None:
    for player_index, player in enumerate(record.get("players", []), start=1):
        if player.get("bbox_xywh_px") is None or player.get("pose_2d") is None:
            continue
        x, y, w, h = [int(round(value)) for value in player["bbox_xywh_px"]]
        color = color_for_player(player.get("global_player_id"), player.get("local_track_id"))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cluster_id = (cluster_badges or {}).get(player_index - 1)
        if show == "p4":
            identity_line = f"global id: {display_id(player.get('global_player_id'))}"
        elif show == "p3":
            identity_line = f"cluster: {cluster_id if cluster_id is not None else 'n/a'}"
        else:
            identity_line = f"local track: {display_id(player.get('local_track_id'))}"
        draw_label_block(
            image,
            [
                f"detection index: {player_index}",
                identity_line,
                f"local track id: {display_id(player.get('local_track_id'))}",
                f"role: {player.get('role') or 'unknown'}",
                f"track state: {player.get('track_state') or 'n/a'}",
                f"single camera: {player.get('single_camera') if player.get('single_camera') is not None else 'n/a'}",
                f"detection confidence: {confidence_text(player.get('detection_confidence'))}",
                f"track confidence: {confidence_text(player.get('track_confidence'))}",
                f"pose confidence mean: {pose_confidence_text(player)}",
            ],
            anchor=(x, y),
        )
        pose = player["pose_2d"]
        keypoints = pose["keypoints_px"]
        confidence = pose["confidence"]
        for left, right in HALPE26_EDGES:
            if confidence[left] < keypoint_threshold or confidence[right] < keypoint_threshold:
                continue
            p1 = tuple(int(round(value)) for value in keypoints[left])
            p2 = tuple(int(round(value)) for value in keypoints[right])
            cv2.line(image, p1, p2, (255, 180, 0), 2)
        for point, score in zip(keypoints, confidence):
            if score < keypoint_threshold:
                continue
            center = tuple(int(round(value)) for value in point)
            cv2.circle(image, center, 3, (0, 0, 255), -1)


def update_manifest_file(path: Path, updates: dict[str, Any]) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    write_json(path, payload)


def update_run_manifests(
    *,
    run_dir: Path,
    artifact_dir: Path,
    manifest_path: Path,
    overlay_count: int,
    overlays_by_delivery: dict[str, int],
) -> None:
    updates = {
        "visualizations": str(artifact_dir),
        "visual_qa_manifest": str(manifest_path),
        "visual_qa_overlay_count": overlay_count,
    }
    update_manifest_file(run_dir / "run_manifest.json", updates)
    update_manifest_file(run_dir / "p1_metrics.json", updates)
    delivery_root = run_dir / "delivery_metrics"
    for delivery_id, count in overlays_by_delivery.items():
        delivery_updates = {
            "visualizations": str(artifact_dir),
            "visual_qa_manifest": str(manifest_path),
            "visual_qa_overlay_count": count,
        }
        update_manifest_file(delivery_root / delivery_id / "run_manifest.json", delivery_updates)
        update_manifest_file(delivery_root / delivery_id / "p1_metrics.json", delivery_updates)


def main() -> int:
    args = parse_args()
    if args.max_per_camera < 0:
        raise SystemExit("--max-per-camera must be >= 0")
    if args.sample_every is not None and args.sample_every <= 0:
        raise SystemExit("--sample-every must be positive")
    if args.frame_ids is not None and args.row_indices is not None:
        raise SystemExit("Use either --frame-ids or --row-indices, not both")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    drive_root = Path(args.drive_root)
    if not drive_root.is_absolute():
        drive_root = ROOT / drive_root
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else run_dir / "visualizations"
    if not artifact_dir.is_absolute():
        artifact_dir = ROOT / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = (
        json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if (run_dir / "run_manifest.json").exists()
        else {}
    )
    show = args.show or stage_from_manifest(run_manifest)
    cluster_badges = load_cluster_badges(run_dir / "diagnostics" / "correspondences.jsonl")

    frame_ids = set(args.frame_ids) if args.frame_ids is not None else None
    row_indices = set(args.row_indices) if args.row_indices is not None else None
    written: list[str] = []
    skipped_missing_images: list[dict[str, Any]] = []
    overlays_by_camera: dict[str, int] = {}
    overlays_by_delivery: dict[str, int] = defaultdict(int)

    prediction_paths = sorted((run_dir / "predictions").glob("*.jsonl"))
    if not prediction_paths:
        raise SystemExit(f"No prediction JSONL files found under {run_dir / 'predictions'}")

    for prediction_path in prediction_paths:
        parts = canonical_prediction_parts(prediction_path)
        if parts is None:
            continue
        group, delivery_id, camera_id = parts
        image_dir = drive_root / group / delivery_id / camera_folder(camera_id)
        camera_output_dir = artifact_dir / group / delivery_id / camera_id
        camera_output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        camera_key = f"{group}/{delivery_id}/{camera_id}"
        for row_index, record in enumerate(iter_jsonl(prediction_path)):
            frame_index = int(record["frame_index"])
            if not should_render_record(
                row_index=row_index,
                frame_index=frame_index,
                row_indices=row_indices,
                rendered_count=count,
                frame_ids=frame_ids,
                sample_every=args.sample_every,
                max_per_camera=args.max_per_camera,
            ):
                continue
            image_path = image_dir / record["frame_name"]
            image = cv2.imread(str(image_path))
            if image is None:
                skipped_missing_images.append({
                    "camera": camera_key,
                    "frame_name": record["frame_name"],
                    "image_path": str(image_path),
                })
                continue
            draw_record(
                image,
                record,
                keypoint_threshold=args.keypoint_threshold,
                show=show,
                cluster_badges=cluster_badges.get((frame_index, camera_id)),
            )
            cv2.putText(
                image,
                f"{camera_key} frame={frame_index} detections={len(record['players'])}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            output_path = camera_output_dir / f"{Path(record['frame_name']).stem}.jpg"
            cv2.imwrite(str(output_path), image)
            written.append(str(output_path))
            count += 1
            overlays_by_delivery[delivery_id] += 1
        overlays_by_camera[camera_key] = count

    manifest_path = artifact_dir / "visual_qa_manifest.json"
    manifest_payload = {
        "schema_version": "cricket_pipetrack_visual_qa/v2",
        "run_id": json.loads((run_dir / "run_manifest.json").read_text()).get("run_id")
        if (run_dir / "run_manifest.json").exists()
        else run_dir.name,
        "artifact_dir": str(artifact_dir),
        "frame_ids": sorted(frame_ids) if frame_ids is not None else None,
        "row_indices": sorted(row_indices) if row_indices is not None else None,
        "sample_every": args.sample_every,
        "max_per_camera": args.max_per_camera,
        "keypoint_threshold": args.keypoint_threshold,
        "show": show,
        "overlay_count": len(written),
        "overlays_by_camera": overlays_by_camera,
        "overlays_by_delivery": dict(sorted(overlays_by_delivery.items())),
        "skipped_missing_images": skipped_missing_images[:200],
        "sample_overlays": written[:20],
    }
    write_json(manifest_path, manifest_payload)
    if not args.no_manifest_update:
        update_run_manifests(
            run_dir=run_dir,
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
            overlay_count=len(written),
            overlays_by_delivery=dict(overlays_by_delivery),
        )
    print(f"Wrote {len(written)} overlays under {artifact_dir}")
    print(f"Wrote {manifest_path}")
    if skipped_missing_images:
        print(f"Skipped {len(skipped_missing_images)} missing/unreadable source images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
