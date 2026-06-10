#!/usr/bin/env python3
"""Generate per-model metadata files in the canonical models/ store."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_estimation.results_io import file_sha256


def main() -> int:
    registry = load_yaml(ROOT / "configs" / "model_registry.yaml")
    envs = load_yaml(ROOT / "configs" / "model_envs.yaml")
    models_root = ROOT / "models"
    models_root.mkdir(exist_ok=True)
    for model in registry["models"]:
        model_id = model["id"]
        model_dir = models_root / model_id
        for child in ["weights", "configs", "checksums"]:
            child_dir = model_dir / child
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / ".gitkeep").touch()
        payload = {
            "schema_version": 1,
            "model": model,
            "environment": envs["models"].get(model_id, {}),
        }
        write_yaml(model_dir / "model.yaml", payload)
        write_readme(model_dir / "README.md", model)
        write_checksums(model_dir, model)
    print(f"Synchronized {len(registry['models'])} model metadata folders under {models_root}")
    return 0


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_readme(path: Path, model: dict[str, Any]) -> None:
    lines = [
        f"# {model.get('name', model['id'])}",
        "",
        f"- ID: `{model['id']}`",
        f"- Family: {model.get('family', '')}",
        f"- Framework: `{model.get('framework', '')}`",
        f"- Skeleton: `{model.get('skeleton', '')}`",
        f"- Role: `{model.get('role', '')}`",
        f"- Checkpoint: `{model.get('checkpoint', '')}`",
        f"- Config: `{model.get('config', '')}`",
        "",
        "## Notes",
        "",
        "This folder is the canonical local model store for metadata, weights, checksums, and local setup notes.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_checksums(model_dir: Path, model: dict[str, Any]) -> None:
    checkpoint = model.get("checkpoint")
    checksums = {}
    if checkpoint:
        path = ROOT / checkpoint
        if path.exists() and path.is_file():
            checksums[str(Path(checkpoint))] = {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
    (model_dir / "checksums" / "sha256.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
