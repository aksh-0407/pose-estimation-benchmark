#!/usr/bin/env python3
"""Download and validate COCO 2017 val keypoint benchmark assets."""

from __future__ import annotations

import argparse
import json
import ssl
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

COCO_SSL_CONTEXT = ssl._create_unverified_context()

COCO_URLS = {
    "val2017": "https://images.cocodataset.org/zips/val2017.zip",
    "annotations": "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/coco", help="COCO dataset root")
    parser.add_argument("--skip-images", action="store_true", help="Download annotations only")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract existing zips")
    parser.add_argument("--remove-archives", action="store_true", help="Delete zip archives after extraction")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def download(url: str, dest: Path, *, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"{dest}: already downloaded")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"+ download {url} -> {dest}")
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60, context=COCO_SSL_CONTEXT) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            tmp.replace(dest)
            break
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if last_error is not None and not dest.exists():
        raise last_error
    elapsed = time.perf_counter() - started
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"{dest}: {size_mb:.1f} MB in {elapsed:.1f}s")


def extract(zip_path: Path, dest_dir: Path, marker: Path, *, force: bool) -> None:
    if marker.exists() and not force:
        print(f"{marker}: already extracted")
        return
    print(f"+ extract {zip_path} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)


def extract_member(zip_path: Path, member: str, dest_dir: Path, marker: Path, *, force: bool) -> None:
    if marker.exists() and not force:
        print(f"{marker}: already extracted")
        return
    print(f"+ extract {member} from {zip_path} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(member, dest_dir)


def remove_archive(path: Path) -> None:
    if path.exists():
        print(f"+ remove archive {path}")
        path.unlink()


def validate(root: Path) -> dict[str, Any]:
    image_dir = root / "val2017"
    annotation_file = root / "annotations" / "person_keypoints_val2017.json"

    image_count = len(list(image_dir.glob("*.jpg"))) if image_dir.exists() else 0
    if not annotation_file.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_file}")

    with annotation_file.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    person_category = next(
        (category for category in annotations.get("categories", []) if category.get("name") == "person"),
        None,
    )
    if person_category is None:
        raise RuntimeError("COCO person category not found in keypoint annotations")

    keypoints = person_category.get("keypoints", [])
    if len(keypoints) != 17:
        raise RuntimeError(f"Expected 17 COCO keypoints, found {len(keypoints)}")

    annotated_people = sum(1 for row in annotations.get("annotations", []) if row.get("num_keypoints", 0) > 0)
    return {
        "dataset": "coco17_val2017",
        "root": str(root),
        "images": image_count,
        "annotation_file": str(annotation_file),
        "annotations_with_keypoints": annotated_people,
        "keypoints": keypoints,
        "target_skeleton": "coco_17",
        "mapping": "configs/keypoint_mappings.yaml",
    }


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    downloads = root / "downloads"

    annotations_zip = downloads / "annotations_trainval2017.zip"
    download(COCO_URLS["annotations"], annotations_zip, force=args.force)
    extract_member(
        annotations_zip,
        "annotations/person_keypoints_val2017.json",
        root,
        root / "annotations" / "person_keypoints_val2017.json",
        force=args.force,
    )

    if not args.skip_images:
        val_zip = downloads / "val2017.zip"
        download(COCO_URLS["val2017"], val_zip, force=args.force)
        extract(val_zip, root, root / "val2017" / "000000000139.jpg", force=args.force)
        if args.remove_archives:
            remove_archive(val_zip)

    if args.remove_archives:
        remove_archive(annotations_zip)

    manifest = validate(root)
    manifest_path = root / "coco17_val2017_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
