#!/usr/bin/env python3
"""Resumable top-down MMPose benchmark on a COCO-17 keypoint subset.

Drives the MMPose whole-body / body top-down models (RTMW, RTMPose-WholeBody)
that share the ``mmpose`` framework. Inference uses the **ground-truth person
boxes** from the COCO annotations (a clean, reproducible top-down protocol), the
native keypoints are reduced to COCO-17 via configs/keypoint_mappings.yaml, and
the result is scored with COCO OKS AP/AR.

Note on comparability: these are GT-bbox top-down numbers. They isolate pose
quality from person detection and are therefore NOT directly comparable to an
end-to-end detector+pose model such as ``yolo26x_pose``. The protocol is recorded
as ``eval_protocol: topdown_gt_bbox`` in the metrics so this stays explicit.

Normally invoked via ``scripts/benchmark/benchmark.py run`` (inside the model's conda env).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from pycocotools.coco import COCO

from pose_estimation.coco_keypoint_eval import (
    append_jsonl,
    count_gt_person_instances,
    dataset_block,
    display_path,
    evaluate_coco,
    latency_percentiles,
    load_progress,
    load_yaml,
    person_boxes_xyxy,
    read_jsonl,
    registry_model,
    resolve,
    selected_images,
    write_progress,
)
from pose_estimation.keypoints import map_keypoints
from pose_estimation.predictions import PredictionRecord
from pose_estimation.results_io import utc_now, write_manifest

DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model_envs.yaml"
DEFAULT_DATASET_CONFIG = ROOT / "configs" / "datasets.yaml"
DEFAULT_REGISTRY = ROOT / "configs" / "model_registry.yaml"
MMPOSE_DIR = ROOT / "external" / "mmpose"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", default="coco17_val2017")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--dataset-config", default=str(DEFAULT_DATASET_CONFIG))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Recorded only; top-down runs per image.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def device_and_metadata(requested: str) -> tuple[str, dict[str, Any]]:
    import torch

    metadata = {
        "requested_device": requested,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    if requested.startswith("cuda") and not torch.cuda.is_available():
        metadata["runtime_device"] = "cpu"
        metadata["device_fallback_reason"] = "CUDA requested but not visible to torch"
        return "cpu", metadata
    metadata["runtime_device"] = requested
    return requested, metadata


def instances_from_results(results: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flatten MMPose top-down output to a list of (keypoints[K,2], scores[K]) in box order."""
    instances: list[tuple[np.ndarray, np.ndarray]] = []
    for sample in results:
        pred = sample.pred_instances
        keypoints = np.asarray(pred.keypoints, dtype=float)
        scores = np.asarray(pred.keypoint_scores, dtype=float)
        for index in range(keypoints.shape[0]):
            instances.append((keypoints[index], scores[index]))
    return instances


def coco_result_for_instance(
    image_id: int, box: dict[str, Any], coco17: list[list[float | None]]
) -> tuple[dict[str, Any], float, list[float]]:
    """Build a COCO keypoints result row (and instance score) from reduced COCO-17 keypoints."""
    flat: list[float] = []
    visible_scores: list[float] = []
    for x, y, score in coco17:
        if x is None or y is None:
            flat.extend([0.0, 0.0, 0.0])
            continue
        confidence = float(score) if score is not None else 0.0
        flat.extend([float(x), float(y), confidence])
        visible_scores.append(confidence)
    instance_score = float(np.mean(visible_scores)) if visible_scores else 0.0
    x1, y1, x2, y2 = box["xyxy"]
    coco_bbox = [x1, y1, x2 - x1, y2 - y1]
    return (
        {
            "image_id": int(image_id),
            "category_id": 1,
            "keypoints": flat,
            "score": instance_score,
            "bbox": coco_bbox,
        },
        instance_score,
        coco_bbox,
    )


def coco_results_from_predictions(prediction_path: Path) -> list[dict[str, Any]]:
    results = []
    for record in read_jsonl(prediction_path):
        metadata = record.get("metadata", {})
        if "coco_result" in metadata:
            results.append(metadata["coco_result"])
    return results


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
    annotation_file = resolve(dataset["annotation_file"])
    gt_instances = count_gt_person_instances(annotation_file, image_ids)
    predictions = read_jsonl(prediction_path)
    predicted_instances = sum(1 for r in predictions if "coco_result" in r.get("metadata", {}))
    predicted_image_ids = {int(r["image_id"]) for r in predictions}
    latencies = read_jsonl(latency_path)
    end_to_end = [float(r["timing_ms"]["end_to_end"]) for r in latencies if "timing_ms" in r]
    completed = {int(r["image_id"]) for r in latencies}
    coco_metrics = evaluate_coco(annotation_file, coco_results_path, image_ids)
    metrics = {
        **coco_metrics,
        **latency_percentiles(end_to_end),
        "eval_protocol": "topdown_gt_bbox",
        "person_boxes": "coco_gt",
        "evaluated_images": len(selected),
        "completed_images": len(completed),
        "gt_person_instances": gt_instances,
        "predicted_instances": predicted_instances,
        "detection_rate": len(predicted_image_ids) / len(selected) if selected else 0.0,
        "predictions_per_image": predicted_instances / len(selected) if selected else 0.0,
        "batch_size": args.batch_size,
    }
    return {
        "schema_version": "pose_metrics/v1",
        "run_id": args.run_id,
        "model_id": args.model_id,
        "model_name": registry_entry.get("name") or args.model_id,
        "framework": "mmpose",
        "source_skeleton": registry_entry.get("skeleton", "coco_wholebody_133"),
        "dataset_id": args.dataset_id,
        "target_skeleton": dataset.get("target_skeleton", "coco_17"),
        "status": "ok" if len(completed) == len(selected) else "partial",
        "precision": "fp32",
        "backend": "mmpose",
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
    from mmpose.apis import inference_topdown, init_model

    model_config = load_yaml(args.model_config)
    dataset_config = load_yaml(args.dataset_config)
    model = model_config["models"][args.model_id]
    registry_entry = registry_model(args.registry, args.model_id)
    dataset = dataset_config["datasets"][args.dataset_id]
    source_skeleton = registry_entry.get("skeleton", "coco_wholebody_133")

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

    coco = COCO(str(annotation_file))
    selected = selected_images(coco, image_dir, args.start_index, args.limit)
    processed = load_progress(progress_path, prediction_path, args.resume)
    remaining = [image for image in selected if int(image["id"]) not in processed]

    device, device_metadata = device_and_metadata(args.device)
    config_path = resolve(model["config"])
    checkpoint_path = resolve(model["checkpoint"])

    cwd = Path.cwd()
    os.chdir(MMPOSE_DIR)  # MMPose configs resolve their _base_ includes relative to the tree
    try:
        pose_model = init_model(str(config_path), str(checkpoint_path), device=device)
        for image in remaining:
            image_id = int(image["id"])
            boxes = person_boxes_xyxy(coco, image_id)
            start = time.perf_counter()
            prediction_records: list[dict[str, Any]] = []
            if boxes:
                bbox_array = np.array([box["xyxy"] for box in boxes], dtype=float)
                results = inference_topdown(pose_model, image["path"], bbox_array, bbox_format="xyxy")
                instances = instances_from_results(results)
                for box, (keypoints_xy, keypoint_scores) in zip(boxes, instances, strict=False):
                    native = [
                        [float(x), float(y), float(score)]
                        for (x, y), score in zip(keypoints_xy, keypoint_scores, strict=True)
                    ]
                    coco17 = map_keypoints(native, source_skeleton, "coco_17")
                    coco_result, instance_score, coco_bbox = coco_result_for_instance(image_id, box, coco17)
                    prediction_records.append(
                        PredictionRecord(
                            run_id=args.run_id,
                            model_id=args.model_id,
                            dataset_id=args.dataset_id,
                            sample_id=str(image_id),
                            image_id=str(image_id),
                            person_id=f"{image_id}:{box['ann_id']}",
                            source_skeleton=source_skeleton,
                            target_skeletons={"native": native, "coco_17": coco17},
                            bbox_xyxy=[float(v) for v in box["xyxy"]],
                            score=instance_score,
                            timing_ms={},
                            metadata={"file_name": image["file_name"], "coco_result": coco_result},
                        ).to_dict()
                    )
            per_image_ms = (time.perf_counter() - start) * 1000.0
            append_jsonl(prediction_path, prediction_records)
            append_jsonl(
                latency_path,
                [
                    {
                        "schema_version": "pose_latency/v1",
                        "run_id": args.run_id,
                        "model_id": args.model_id,
                        "dataset_id": args.dataset_id,
                        "image_id": str(image_id),
                        "file_name": image["file_name"],
                        "timing_ms": {"end_to_end": per_image_ms},
                    }
                ],
            )
            processed.add(image_id)
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
                    "processed_image_ids": sorted(processed),
                },
            )
            print(f"{args.model_id}: processed {len(processed)}/{len(selected)} images")
    finally:
        os.chdir(cwd)

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
