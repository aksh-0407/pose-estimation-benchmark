"""COCO keypoint dataset manifest helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS = ROOT / "configs" / "datasets.yaml"


@dataclass(frozen=True)
class DatasetStatus:
    dataset_id: str
    ready: bool
    image_count: int
    annotation_file: str
    missing: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "ready": self.ready,
            "image_count": self.image_count,
            "annotation_file": self.annotation_file,
            "missing": self.missing,
        }


def load_dataset_config(dataset_id: str, path: str | Path = DEFAULT_DATASETS) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    try:
        return config["datasets"][dataset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset id: {dataset_id}") from exc


class CocoKeypointDataset:
    def __init__(self, dataset_id: str = "coco17_val2017", config_path: str | Path = DEFAULT_DATASETS) -> None:
        self.dataset_id = dataset_id
        self.config = load_dataset_config(dataset_id, config_path)
        self.root = _resolve(self.config["root"])
        self.image_dir = _resolve(self.config["images"])
        self.annotation_file = _resolve(self.config["annotation_file"])

    def status(self) -> DatasetStatus:
        missing = []
        if not self.image_dir.exists():
            missing.append(str(self.image_dir))
        if not self.annotation_file.exists():
            missing.append(str(self.annotation_file))
        image_count = len(list(self.image_dir.glob("*.jpg"))) if self.image_dir.exists() else 0
        expected = int(self.config.get("expected", {}).get("images", 0) or 0)
        if expected and image_count not in {0, expected}:
            missing.append(f"expected {expected} images, found {image_count}")
        return DatasetStatus(
            dataset_id=self.dataset_id,
            ready=not missing and image_count > 0,
            image_count=image_count,
            annotation_file=str(self.annotation_file),
            missing=missing,
        )

    def manifest(self) -> dict[str, Any]:
        status = self.status()
        manifest = status.to_dict()
        manifest.update(
            {
                "name": self.config["name"],
                "split": self.config["split"],
                "target_skeleton": self.config["target_skeleton"],
                "keypoint_mapping": self.config["keypoint_mapping"],
            }
        )
        if self.annotation_file.exists():
            with self.annotation_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            manifest["annotations"] = len(payload.get("annotations", []))
            manifest["images_in_annotation"] = len(payload.get("images", []))
        return manifest


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()

