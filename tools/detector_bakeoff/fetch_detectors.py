#!/usr/bin/env python3
"""Fetch mim-hosted detector presets for the swappable-detector P1 experiment.

Downloads the config + checkpoint for each requested `--detector` preset into
models/<dir>/ (the layout run_phase1's DETECTOR_PRESETS resolver expects). Uses
openmim, which pulls both the .py config and the .pth weight from the mmdet
model zoo. Idempotent: presets that already have a .py + .pth are skipped.

Candidate fix 1 in docs/pipeline/00-inference.md §8 (upgrade the detector).
Pose stays RTMPose-X; only the person detector changes.

    python tools/detector_bakeoff/fetch_detectors.py --detectors rtmdet_l rtmdet_x dino
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from core.inference.run_phase1_rtmpose_inference import DETECTOR_PRESETS  # noqa: E402


def fetch_one(name: str) -> bool:
    preset = DETECTOR_PRESETS.get(name)
    if preset is None:
        print(f"[{name}] unknown preset; choices: {sorted(DETECTOR_PRESETS)}", flush=True)
        return False
    if "mim" not in preset:
        print(f"[{name}] vendored (no download needed): {preset.get('checkpoint')}", flush=True)
        return True
    dest = Path(preset["dir"])
    if sorted(dest.glob("*.py")) and sorted(dest.glob("*.pth")):
        print(f"[{name}] already present in {dest}, skipping", flush=True)
        return True
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] downloading {preset['mim']} -> {dest}", flush=True)
    try:
        from mim import download
        download("mmdet", [preset["mim"]], dest_root=str(dest))
    except Exception as exc:  # noqa: BLE001
        print(f"[{name}] FAILED: {exc}", flush=True)
        return False
    ok = bool(sorted(dest.glob("*.py")) and sorted(dest.glob("*.pth")))
    print(f"[{name}] {'done' if ok else 'INCOMPLETE (missing .py or .pth)'}: "
          f"{[p.name for p in dest.iterdir()]}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mim_presets = [n for n, p in DETECTOR_PRESETS.items() if "mim" in p]
    ap.add_argument("--detectors", nargs="+", default=mim_presets,
                    help=f"Presets to fetch (default: all mim presets {mim_presets})")
    args = ap.parse_args()
    results = {name: fetch_one(name) for name in args.detectors}
    print("\nsummary:", {k: ("ok" if v else "FAIL") for k, v in results.items()}, flush=True)
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
