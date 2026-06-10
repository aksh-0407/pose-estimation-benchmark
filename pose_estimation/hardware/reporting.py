"""System metadata capture for immutable benchmark runs."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from pose_estimation.results_io import command_output


def collect_hardware_report() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "gpu_query": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
    }


def collect_software_report(workdir: str | Path = ".") -> dict[str, Any]:
    pip_freeze = None
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "git_sha": command_output(["git", "-C", str(workdir), "rev-parse", "HEAD"]),
        "pip_freeze": pip_freeze.splitlines() if pip_freeze else [],
    }

