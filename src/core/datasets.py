"""Dataset registry + path resolution.

One layout under a single ``DATA_ROOT`` (``--data-root`` / env ``PIPETRACK_DATA`` /
default ``data``), identical on every machine - only the base differs (laptop ``data/``
== L40S ``~/bits-pose-data/``). See ``configs/datasets.yaml`` for the tree and the
per-dataset ``calibration_source`` (both matches share one calibration session, so
``40_full`` borrows ``8_init``'s calibration).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs" / "datasets.yaml"
DEFAULT_DATA_ROOT = "data"
ENV_DATA_ROOT = "PIPETRACK_DATA"


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_data_root(cli_value: str | os.PathLike[str] | None = None) -> Path:
    """The machine's DATA_ROOT: ``--data-root`` > ``$PIPETRACK_DATA`` > ``data``.

    A relative value is resolved against the repo root so ``data`` means the
    in-repo ``data/`` regardless of the current working directory.
    """
    value = cli_value or os.environ.get(ENV_DATA_ROOT) or DEFAULT_DATA_ROOT
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _dataset_entry(dataset: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = registry if registry is not None else load_registry()
    return (registry.get("datasets") or {}).get(dataset, {})


def calibration_source(dataset: str, registry: dict[str, Any] | None = None) -> str:
    """Which dataset owns the calibration ``dataset`` should read.

    ``self`` (or an unregistered name) resolves to the dataset itself, so tmp/test
    datasets and any owner dataset just use their own ``calibration-data/``.
    """
    source = _dataset_entry(dataset, registry).get("calibration_source", dataset)
    return dataset if source in (None, "self") else str(source)


def events_subdir(dataset: str, registry: dict[str, Any] | None = None) -> str:
    return str(_dataset_entry(dataset, registry).get("events_subdir", "events-data"))


# --- roots under DATA_ROOT -------------------------------------------------

def raw_root(data_root: str | os.PathLike[str] | None, dataset: str) -> Path:
    """Footage + calibration-data/ + events-data/ for one dataset."""
    return resolve_data_root(data_root) / "raw" / dataset


def calibration_raw_root(data_root: str | os.PathLike[str] | None, dataset: str) -> Path:
    """Raw root of the dataset whose calibration ``dataset`` uses (borrow-aware)."""
    return resolve_data_root(data_root) / "raw" / calibration_source(dataset)


def derived_root(data_root: str | os.PathLike[str] | None, dataset: str, version: str) -> Path:
    """P1 + stage outputs for one run, e.g. ``derived/8_init/pipetrack_v9``."""
    return resolve_data_root(data_root) / "derived" / dataset / f"pipetrack_v{version}"


def viz_root(data_root: str | os.PathLike[str] | None, dataset: str, version: str) -> Path:
    """Mosaics for one run, e.g. ``viz/8_init/pipetrack_v9``."""
    return resolve_data_root(data_root) / "viz" / dataset / f"pipetrack_v{version}"


# --- resolution from a footage root (what the stage runners hold) ----------

def events_root(footage_root: str | os.PathLike[str]) -> Path:
    """``events-data/`` beside the footage (``<DATA_ROOT>/raw/<dataset>/events-data``)."""
    return Path(footage_root) / "events-data"


def calibration_root_for(footage_root: str | os.PathLike[str]) -> Path:
    """Calibration dataset root for a given footage dataset root.

    ``footage_root`` is ``<DATA_ROOT>/raw/<dataset>``; calibration lives at the
    sibling ``<DATA_ROOT>/raw/<calibration_source(dataset)>`` - so a borrowing
    dataset (40_full) transparently reads 8_init's calibration without threading
    a second root through every stage.
    """
    footage_root = Path(footage_root)
    return footage_root.parent / calibration_source(footage_root.name)
