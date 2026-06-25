#!/usr/bin/env python3
"""Clone and build the official CMU OpenPose runtime used by openpose_body25."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = ROOT / "external" / "openpose"
DEFAULT_BUILD = DEFAULT_REPO / "build"
OPENPOSE_URL = "https://github.com/CMU-Perceptual-Computing-Lab/openpose"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO))
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD))
    parser.add_argument("--url", default=OPENPOSE_URL)
    parser.add_argument("--gpu-mode", default="CPU_ONLY", choices=["CPU_ONLY", "CUDA", "OPENCL"])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--target", default="openpose.bin")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def ensure_source(repo_dir: Path, url: str, dry_run: bool) -> None:
    if not repo_dir.exists():
        if not dry_run:
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", url, str(repo_dir)], dry_run=dry_run)
    run(["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"], dry_run=dry_run)


def apply_openblas_patch(repo_dir: Path, dry_run: bool) -> None:
    cmake_path = repo_dir / "CMakeLists.txt"
    if dry_run:
        print(f"+ patch {cmake_path} with -DBLAS=Open for bundled Caffe")
        return
    text = cmake_path.read_text(encoding="utf-8")
    if "-DBLAS=Open" in text:
        return
    needle = "          -DCPU_ONLY=${CAFFE_CPU_ONLY}\n          -DCMAKE_BUILD_TYPE=Release"
    replacement = "          -DCPU_ONLY=${CAFFE_CPU_ONLY}\n          -DBLAS=Open\n          -DCMAKE_BUILD_TYPE=Release"
    patched = text.replace(needle, replacement)
    if patched == text:
        raise RuntimeError(f"Could not apply OpenBLAS patch to {cmake_path}")
    cmake_path.write_text(patched, encoding="utf-8")


def configure(repo_dir: Path, build_dir: Path, gpu_mode: str, dry_run: bool) -> None:
    run(
        [
            "cmake",
            "-S",
            str(repo_dir),
            "-B",
            str(build_dir),
            f"-DGPU_MODE={gpu_mode}",
            "-DBUILD_PYTHON=OFF",
            "-DBUILD_DOCS=OFF",
            "-DBUILD_EXAMPLES=ON",
            "-DDOWNLOAD_BODY_25_MODEL=OFF",
            "-DDOWNLOAD_BODY_COCO_MODEL=OFF",
            "-DDOWNLOAD_BODY_MPI_MODEL=OFF",
            "-DDOWNLOAD_FACE_MODEL=OFF",
            "-DDOWNLOAD_HAND_MODEL=OFF",
        ],
        dry_run=dry_run,
    )


def build(build_dir: Path, target: str, jobs: int, dry_run: bool) -> None:
    run(["cmake", "--build", str(build_dir), "--target", target, f"-j{jobs}"], dry_run=dry_run)


def main() -> int:
    args = parse_args()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()
    ensure_source(repo_dir, args.url, args.dry_run)
    apply_openblas_patch(repo_dir, args.dry_run)
    configure(repo_dir, build_dir, args.gpu_mode, args.dry_run)
    if not args.skip_build:
        build(build_dir, args.target, args.jobs, args.dry_run)
    executable = build_dir / "examples" / "openpose" / "openpose.bin"
    if not args.dry_run and not executable.exists():
        raise SystemExit(f"OpenPose binary was not created: {executable}")
    print(f"OpenPose binary: {executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
