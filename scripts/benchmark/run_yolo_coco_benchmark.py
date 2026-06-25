#!/usr/bin/env python3
"""Run a batched, resumable YOLO pose benchmark on a COCO keypoint subset.

End-to-end protocol: YOLO performs its own person detection and pose in one pass
(no ground-truth boxes). Shared image selection, resumability, and COCO OKS
evaluation live in ``pose_estimation.coco_keypoint_eval``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from pose_estimation.coco_keypoint_eval import (
    append_jsonl,
    count_gt_person_instances,
    dataset_block,
    display_path,
    evaluate_coco,
    latency_percentiles,
    load_progress,
    load_yaml,
    read_jsonl,
    registry_model,
    resolve,
    selected_images,
    write_progress,
)
from pose_estimation.predictions import PredictionRecord
from pose_estimation.results_io import utc_now, write_manifest

DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model_envs.yaml"
DEFAULT_DATASET_CONFIG = ROOT / "configs" / "datasets.yaml"
DEFAULT_REGISTRY = ROOT / "configs" / "model_registry.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--model-id", default="yolo26x_pose")
    parser.add_argument("--dataset-id", default="coco17_val2017")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--dataset-config", default=str(DEFAULT_DATASET_CONFIG))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def device_arg(requested: str) -> tuple[str | int, dict[str, Any]]:
    import torch

    metadata = {
        "requested_device": requested,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    if requested.startswith("cuda") and torch.cuda.is_available():
        if ":" in requested:
            return int(requested.split(":", 1)[1]), metadata
        return 0, metadata
    return "cpu", metadata


def yolo_records(
    *,
    run_id: str,
    model_id: str,
    dataset_id: str,
    image: dict[str, Any],
    result: Any,
    timing_ms: dict[str, float],
) -> list[dict[str, Any]]:
    if result.keypoints is None or result.boxes is None:
        return []

    keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
    keypoint_conf = result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None else None
    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    box_conf = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones((len(boxes_xyxy),))
    count = min(len(keypoints_xy), len(boxes_xyxy), len(box_conf))
    records = []
    for index in range(count):
        conf = keypoint_conf[index] if keypoint_conf is not None else np.ones((keypoints_xy.shape[1],), dtype=float)
        native = [
            [float(x), float(y), float(score)]
            for (x, y), score in zip(keypoints_xy[index], conf, strict=True)
        ]
        bbox = [float(value) for value in boxes_xyxy[index].tolist()]
        coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        coco_keypoints = []
        for keypoint in native:
            coco_keypoints.extend(keypoint)
        record = PredictionRecord(
            run_id=run_id,
            model_id=model_id,
            dataset_id=dataset_id,
            sample_id=str(image["id"]),
            image_id=str(image["id"]),
            person_id=f"{image['id']}:{index}",
            source_skeleton="coco_17",
            target_skeletons={"native": native, "coco_17": native},
            bbox_xyxy=bbox,
            score=float(box_conf[index]),
            timing_ms=timing_ms,
            metadata={
                "file_name": image["file_name"],
                "coco_bbox_xywh": coco_bbox,
                "coco_keypoints": coco_keypoints,
            },
        ).to_dict()
        records.append(record)
    return records


def prediction_records_by_instance(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    records = {}
    for record in read_jsonl(path):
        records[(int(record["image_id"]), str(record["person_id"]))] = record
    return records


def coco_results_from_predictions(prediction_path: Path) -> list[dict[str, Any]]:
    results = []
    for record in prediction_records_by_instance(prediction_path).values():
        metadata = record.get("metadata", {})
        results.append(
            {
                "image_id": int(record["image_id"]),
                "category_id": 1,
                "keypoints": metadata["coco_keypoints"],
                "score": float(record.get("score") or 0.0),
                "bbox": metadata["coco_bbox_xywh"],
            }
        )
    return results


def latency_by_image(path: Path) -> dict[int, dict[str, Any]]:
    latencies = {}
    for record in read_jsonl(path):
        latencies[int(record["image_id"])] = record
    return latencies


def build_metrics(
    *,
    args: argparse.Namespace,
    registry_entry: dict[str, Any],
    dataset: dict[str, Any],
    selected: list[dict[str, Any]],
    prediction_path: Path,
    coco_results_path: Path,
    latency_path: Path,
    device_metadata: dict[str, Any],
) -> dict[str, Any]:
    image_ids = [int(image["id"]) for image in selected]
    gt_instances = count_gt_person_instances(resolve(dataset["annotation_file"]), image_ids)
    prediction_records = prediction_records_by_instance(prediction_path)
    predicted_instances = len(prediction_records)
    predicted_image_ids = {image_id for image_id, _ in prediction_records}
    latencies = latency_by_image(latency_path)
    end_to_end = [float(item["timing_ms"]["end_to_end"]) for item in latencies.values() if "timing_ms" in item]
    coco_metrics = evaluate_coco(resolve(dataset["annotation_file"]), coco_results_path, image_ids)
    metrics = {
        **coco_metrics,
        **latency_percentiles(end_to_end),
        "eval_protocol": "end_to_end",
        "evaluated_images": len(selected),
        "completed_images": len(latencies),
        "gt_person_instances": gt_instances,
        "predicted_instances": predicted_instances,
        "detection_rate": len(predicted_image_ids) / len(selected) if selected else 0.0,
        "predictions_per_image": predicted_instances / len(selected) if selected else 0.0,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
    }
    return {
        "schema_version": "pose_metrics/v1",
        "run_id": args.run_id,
        "model_id": args.model_id,
        "model_name": registry_entry.get("name") or args.model_id,
        "framework": "ultralytics",
        "source_skeleton": "coco_17",
        "dataset_id": args.dataset_id,
        "target_skeleton": dataset.get("target_skeleton", "coco_17"),
        "status": "ok" if len(latencies) == len(selected) else "partial",
        "precision": "fp32",
        "backend": "ultralytics",
        "limit": args.limit,
        "metrics": metrics,
        "artifacts": {
            "policy": "local_only_not_committed",
            "predictions_jsonl": display_path(prediction_path),
            "coco_results_json": display_path(coco_results_path),
            "latency_jsonl": display_path(latency_path),
        },
        "dataset": dataset_block(dataset, image_ids),
        "runtime": device_metadata,
        "created_at": utc_now(),
    }


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model_config = load_yaml(args.model_config)
    dataset_config = load_yaml(args.dataset_config)
    model = model_config["models"][args.model_id]
    registry_entry = registry_model(args.registry, args.model_id)
    dataset = dataset_config["datasets"][args.dataset_id]
    annotation_file = resolve(dataset["annotation_file"])
    image_dir = resolve(dataset["images"])
    run_dir = resolve(args.run_dir)
    artifact_dir = resolve(args.artifact_dir) if args.artifact_dir else run_dir
    prediction_path = artifact_dir / "predictions" / f"{args.model_id}__{args.dataset_id}.jsonl"
    coco_results_path = artifact_dir / "predictions" / f"{args.model_id}__{args.dataset_id}.coco_keypoints.json"
    latency_path = artifact_dir / "logs" / f"{args.model_id}__{args.dataset_id}.latency.jsonl"
    progress_path = artifact_dir / "logs" / f"{args.model_id}__{args.dataset_id}.progress.json"
    metrics_path = run_dir / "metrics" / f"{args.model_id}__{args.dataset_id}.json"

    if not args.resume:
        for path in [prediction_path, coco_results_path, latency_path, progress_path, metrics_path]:
            path.unlink(missing_ok=True)

    from pycocotools.coco import COCO

    coco = COCO(str(annotation_file))
    selected = selected_images(coco, image_dir, args.start_index, args.limit)
    processed = load_progress(progress_path, prediction_path, args.resume)
    remaining = [image for image in selected if int(image["id"]) not in processed]
    device, device_metadata = device_arg(args.device)
    yolo = YOLO(str(resolve(model.get("model_name", model["checkpoint"]))))

    for offset in range(0, len(remaining), args.batch_size):
        batch = remaining[offset : offset + args.batch_size]
        paths = [image["path"] for image in batch]
        start = time.perf_counter()
        results = yolo.predict(
            source=paths,
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            batch=args.batch_size,
            verbose=False,
        )
        batch_ms = (time.perf_counter() - start) * 1000.0
        per_image_ms = batch_ms / max(len(batch), 1)
        prediction_records = []
        latency_records = []
        for image, result in zip(batch, results, strict=True):
            speed = getattr(result, "speed", {}) or {}
            timing_ms = {
                "preprocess": float(speed.get("preprocess", 0.0)),
                "inference": float(speed.get("inference", 0.0)),
                "postprocess": float(speed.get("postprocess", 0.0)),
                "end_to_end": per_image_ms,
            }
            prediction_records.extend(
                yolo_records(
                    run_id=args.run_id,
                    model_id=args.model_id,
                    dataset_id=args.dataset_id,
                    image=image,
                    result=result,
                    timing_ms=timing_ms,
                )
            )
            latency_records.append(
                {
                    "schema_version": "pose_latency/v1",
                    "run_id": args.run_id,
                    "model_id": args.model_id,
                    "dataset_id": args.dataset_id,
                    "image_id": str(image["id"]),
                    "file_name": image["file_name"],
                    "timing_ms": timing_ms,
                }
            )
            processed.add(int(image["id"]))
        append_jsonl(prediction_path, prediction_records)
        append_jsonl(latency_path, latency_records)
        write_progress(
            progress_path,
            {
                "schema_version": "pose_benchmark_progress/v1",
                "run_id": args.run_id,
                "model_id": args.model_id,
                "dataset_id": args.dataset_id,
                "updated_at": utc_now(),
                "total_selected": len(selected),
                "completed_images": len(processed),
                "remaining_images": max(len(selected) - len(processed), 0),
                "batch_size": args.batch_size,
                "processed_image_ids": sorted(processed),
            },
        )
        print(f"{args.model_id}: processed {len(processed)}/{len(selected)} images")

    coco_results = coco_results_from_predictions(prediction_path)
    coco_results_path.parent.mkdir(parents=True, exist_ok=True)
    coco_results_path.write_text(json.dumps(coco_results, sort_keys=True) + "\n", encoding="utf-8")
    metrics = build_metrics(
        args=args,
        registry_entry=registry_entry,
        dataset=dataset,
        selected=selected,
        prediction_path=prediction_path,
        coco_results_path=coco_results_path,
        latency_path=latency_path,
        device_metadata=device_metadata,
    )
    write_manifest(metrics_path, metrics)
    print(json.dumps({"status": metrics["status"], "metrics_path": str(metrics_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
