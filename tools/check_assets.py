#!/usr/bin/env python3
"""Report local checkpoint/model asset readiness from configs/model_envs.yaml."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model_envs.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--fail-missing", action="store_true", help="Exit non-zero if required assets are missing")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def selected_models(config: dict[str, Any], requested: list[str]) -> list[str]:
    if requested == ["all"]:
        return list(config["models"])
    unknown = sorted(set(requested) - set(config["models"]))
    if unknown:
        raise SystemExit(f"Unknown model IDs: {', '.join(unknown)}")
    return requested


def describe_path(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "kind": "directory" if exists and path.is_dir() else "file",
        "bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def asset_rows(config: dict[str, Any], model_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for model_id in model_ids:
        model = config["models"][model_id]
        assets = list(model.get("assets", [])) + list(model.get("manual_assets", []))
        if not assets and model.get("checkpoint"):
            assets = [
                {
                    "kind": "checkpoint",
                    "path": model["checkpoint"],
                    "required_for_smoke": True,
                    "description": "checkpoint referenced by model entry",
                }
            ]
        for asset in assets:
            path_info = describe_path(expand(asset["path"]))
            required = bool(asset.get("required_for_smoke"))
            rows.append(
                {
                    "model_id": model_id,
                    "asset_kind": asset.get("kind", "manual"),
                    "required_for_smoke": required,
                    "large": bool(asset.get("large")),
                    "ready": path_info["exists"],
                    "path": path_info["path"],
                    "bytes": path_info["bytes"],
                    "source": asset.get("url") or asset.get("repo_id") or "manual",
                    "description": asset.get("description", ""),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [row for row in rows if row["required_for_smoke"]]
    missing = [row for row in required if not row["ready"]]
    return {
        "schema_version": "pose_asset_status/v1",
        "required_assets": len(required),
        "ready_required_assets": len(required) - len(missing),
        "missing_required_assets": len(missing),
        "models_with_missing_assets": sorted({row["model_id"] for row in missing}),
        "assets": rows,
    }


def print_table(summary: dict[str, Any]) -> None:
    print(
        f"Required assets: {summary['ready_required_assets']}/"
        f"{summary['required_assets']} ready"
    )
    for row in summary["assets"]:
        status = "ready" if row["ready"] else "missing"
        size = "" if row["bytes"] is None else f"{row['bytes'] / (1024 * 1024):.1f} MB"
        required = "required" if row["required_for_smoke"] else "optional"
        print(f"{status:7} {row['model_id']:28} {required:8} {size:>10} {row['path']}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    rows = asset_rows(config, selected_models(config, args.models))
    summary = summarize(rows)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_table(summary)
    return 1 if args.fail_missing and summary["missing_required_assets"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
