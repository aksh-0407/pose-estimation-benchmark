"""Shared helpers for COCO-17 keypoint benchmark runners.

These are the pieces common to every full dataset runner (image selection,
resumability, COCO OKS evaluation, latency stats). Model-specific inference lives
in the per-framework `scripts/run_*_coco_benchmark.py` runners that import from here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]


# --- path / config helpers ---------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative, possibly ``~``/env path against the repo root."""
    path = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def display_path(path: str | Path) -> str:
    """Repo-relative string for a path when possible (keeps committed JSON portable)."""
    resolved = resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def registry_model(registry_path: str | Path, model_id: str) -> dict[str, Any]:
    registry = load_yaml(registry_path)
    for model in registry.get("models", []):
        if model.get("id") == model_id:
            return model
    return {"id": model_id, "name": model_id}


# --- dataset selection -------------------------------------------------------

def selected_images(
    coco: COCO, image_dir: Path, start_index: int, limit: int | None
) -> list[dict[str, Any]]:
    """Deterministic, sorted slice of dataset images with absolute ``path`` filled in."""
    images = sorted(coco.dataset.get("images", []), key=lambda item: int(item["id"]))
    if start_index:
        images = images[start_index:]
    if limit is not None:
        images = images[:limit]
    selected = []
    for image in images:
        image = dict(image)
        image["path"] = str(image_dir / image["file_name"])
        selected.append(image)
    return selected


def person_boxes_xyxy(coco: COCO, image_id: int) -> list[dict[str, Any]]:
    """Ground-truth person boxes for an image, as xyxy plus the source annotation id.

    Used by top-down runners (GT-bbox protocol). Crowd regions are skipped.
    """
    ann_ids = coco.getAnnIds(imgIds=[int(image_id)], catIds=[1], iscrowd=False)
    boxes = []
    for ann in coco.loadAnns(ann_ids):
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        boxes.append({"ann_id": int(ann["id"]), "xyxy": [float(x), float(y), float(x + w), float(y + h)]})
    return boxes


def count_gt_person_instances(annotation_file: Path, image_ids: list[int]) -> int:
    coco = COCO(str(annotation_file))
    ann_ids = coco.getAnnIds(imgIds=image_ids, catIds=[1])
    return sum(1 for ann in coco.loadAnns(ann_ids) if ann.get("num_keypoints", 0) > 0)


# --- JSONL / progress IO -----------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_progress(path: Path, prediction_path: Path, resume: bool) -> set[int]:
    if not resume:
        return set()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {int(image_id) for image_id in payload.get("processed_image_ids", [])}
    processed = set()
    for record in read_jsonl(prediction_path):
        if record.get("image_id") is not None:
            processed.add(int(record["image_id"]))
    return processed


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# --- metrics -----------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


_COCO_STAT_KEYS = [
    "coco_oks_ap",
    "coco_oks_ap50",
    "coco_oks_ap75",
    "coco_oks_ap_medium",
    "coco_oks_ap_large",
    "coco_oks_ar",
    "coco_oks_ar50",
    "coco_oks_ar75",
    "coco_oks_ar_medium",
    "coco_oks_ar_large",
]


def evaluate_coco(
    annotation_file: Path, coco_results_path: Path, image_ids: list[int]
) -> dict[str, float | None]:
    """COCO OKS AP/AR over the 17 body keypoints, restricted to ``image_ids``."""
    coco_results = json.loads(coco_results_path.read_text(encoding="utf-8"))
    if not coco_results:
        return {key: None for key in _COCO_STAT_KEYS}
    coco_gt = COCO(str(annotation_file))
    coco_dt = coco_gt.loadRes(str(coco_results_path))
    evaluator = COCOeval(coco_gt, coco_dt, "keypoints")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = [float(value) for value in evaluator.stats.tolist()]
    return dict(zip(_COCO_STAT_KEYS, stats, strict=True))


def latency_percentiles(end_to_end_ms: list[float]) -> dict[str, float | None]:
    p50 = percentile(end_to_end_ms, 50)
    return {
        "latency_p50_ms": p50,
        "latency_p90_ms": percentile(end_to_end_ms, 90),
        "latency_p95_ms": percentile(end_to_end_ms, 95),
        "latency_p99_ms": percentile(end_to_end_ms, 99),
        "latency_mean_ms": float(np.mean(end_to_end_ms)) if end_to_end_ms else None,
        "fps_per_camera": (1000.0 / p50) if p50 else None,
    }


def dataset_block(dataset: dict[str, Any], image_ids: list[int]) -> dict[str, Any]:
    """Portable dataset provenance block: repo-relative paths + annotation hash."""
    from pose_estimation.results_io import file_sha256

    annotation_file = resolve(dataset["annotation_file"])
    return {
        "annotation_file": display_path(annotation_file),
        "annotation_sha256": file_sha256(annotation_file),
        "images": display_path(dataset["images"]),
        "selected_image_ids": image_ids,
    }
