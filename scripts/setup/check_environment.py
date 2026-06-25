#!/usr/bin/env python3
"""Print local readiness for the cricket pose benchmark environments."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


PACKAGES = [
    "numpy",
    "yaml",
    "cv2",
    "torch",
    "onnxruntime",
    "tensorrt",
    "mmpose",
    "mmcv",
    "mmengine",
    "mmdet",
    "ultralytics",
    "mediapipe",
]

BINARIES = ["nvidia-smi", "nvcc", "gcc", "g++", "cmake", "ffmpeg", "git"]


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE ({exc})"


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print("\nBinaries:")
    for binary in BINARIES:
        path = shutil.which(binary)
        print(f"  {binary}: {path or 'missing'}")

    print("\nPython packages:")
    for package in PACKAGES:
        print(f"  {package}: {bool(importlib.util.find_spec(package))}")

    print("\nGPU:")
    print(run(["nvidia-smi"]))

    if importlib.util.find_spec("onnxruntime"):
        import onnxruntime as ort

        print("\nONNX Runtime:")
        print(f"  version: {ort.__version__}")
        print(f"  providers: {ort.get_available_providers()}")

    if importlib.util.find_spec("torch"):
        import torch

        print("\nPyTorch:")
        print(f"  version: {torch.__version__}")
        print(f"  cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device: {torch.cuda.get_device_name(0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

