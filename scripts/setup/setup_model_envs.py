#!/usr/bin/env python3
"""Create per-model Conda environments and fetch benchmark assets."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "model_envs.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--models", nargs="+", default=None, help="Model IDs to set up")
    parser.add_argument("--download-assets", action="store_true", help="Download non-large model assets")
    parser.add_argument("--download-large-assets", action="store_true", help="Also download large HF assets")
    parser.add_argument("--skip-envs", action="store_true", help="Only download assets")
    parser.add_argument("--skip-assets", action="store_true", help="Only create/install environments")
    parser.add_argument("--force-install", action="store_true", help="Re-run install commands even if the stamp exists")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def parse_version(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts)


def query_nvidia_driver_version() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    except Exception:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            return line.split(",", 1)[0].strip()
    return None


def select_torch_channel(index_urls: dict[str, str]) -> tuple[str, str]:
    override = os.environ.get("POSE_BENCHMARK_TORCH_CHANNEL")
    if override:
        channel = override.strip()
        if channel in index_urls:
            return channel, index_urls[channel]
        raise SystemExit(
            f"Unsupported POSE_BENCHMARK_TORCH_CHANNEL={channel!r}; expected one of {', '.join(sorted(index_urls))}"
        )

    driver_version = query_nvidia_driver_version()
    if driver_version is None:
        return "cpu", index_urls["cpu"]

    major = parse_version(driver_version)
    if major >= (560,):
        channel = "cu126"
    elif major >= (550,):
        channel = "cu124"
    elif major >= (530,):
        channel = "cu121"
    else:
        channel = "cpu"

    if channel not in index_urls:
        preferred = parse_version(channel)
        cuda_channels = sorted(
            (name for name in index_urls if name.startswith("cu")),
            key=parse_version,
        )
        compatible = [name for name in cuda_channels if parse_version(name) <= preferred]
        channel = compatible[-1] if compatible else "cpu"
    return channel, index_urls[channel]


def run(command: list[str], *, dry_run: bool, cwd: Path = ROOT) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"+ {printable}")
    if dry_run:
        return
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    subprocess.run(command, cwd=cwd, check=True, env=env)


def validate_profile_paths(profile: dict[str, Any], model_id: str) -> None:
    for command in profile.get("install", {}).get("commands", []):
        for match in re.finditer(r"(?:^|\s)-e\s+([^\s;&|]+)", command):
            path = expand_path(match.group(1))
            if not path.exists():
                raise SystemExit(
                    f"{model_id}: editable install path is missing: {path}\n"
                    "Populate the external dependency first, for example:\n"
                    "  git clone https://github.com/open-mmlab/mmpose.git external/mmpose"
                )


def conda_env_exists(conda: str, env_name: str) -> bool:
    result = subprocess.run(
        [conda, "env", "list"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    return any(line.split()[0] == env_name for line in result.stdout.splitlines() if line and not line.startswith("#"))


def create_env(conda: str, env_name: str, python_version: str, *, dry_run: bool) -> None:
    if not dry_run and conda_env_exists(conda, env_name):
        print(f"{env_name}: Conda env already exists")
        return
    run(
        [conda, "create", "-y", "--override-channels", "-c", "conda-forge", "-n", env_name, f"python={python_version}", "pip"],
        dry_run=dry_run,
    )


def install_profile(conda: str, env_name: str, profile: dict[str, Any], model_id: str, *, dry_run: bool, force: bool) -> None:
    stamp_dir = ROOT / ".model_env_stamps"
    stamp_dir.mkdir(exist_ok=True)
    stamp = stamp_dir / f"{model_id}.stamp"
    if stamp.exists() and not force:
        print(f"{model_id}: install stamp exists ({stamp}); use --force-install to rerun")
        return

    install = profile.get("install", {})
    torch_install = install.get("torch")
    if torch_install:
        index_urls = torch_install.get("index_urls", {})
        if not index_urls:
            raise SystemExit(f"{model_id}: install.torch requires index_urls")
        channel, index_url = select_torch_channel(index_urls)
        print(f"{model_id}: selected torch channel {channel} ({index_url})")
        packages = list(torch_install.get("packages", ["torch", "torchvision"]))
        run(
            [conda, "run", "-n", env_name, "python", "-m", "pip", "install", "--index-url", index_url, *packages],
            dry_run=dry_run,
        )

    conda_install = install.get("conda")
    if conda_install:
        channels = []
        for channel in conda_install.get("channels", []):
            channels.extend(["-c", channel])
        raw_packages = list(conda_install.get("packages", []))
        # Filter out packages that we now install via pip wheels to avoid solver conflicts
        forbidden = ("pytorch", "torchvision", "pytorch-cuda", "mkl", "intel-openmp")
        packages = [p for p in raw_packages if not any(p.startswith(f) for f in forbidden)]
        # Also drop pytorch/nvidia channels if they only served torch packages
        channel_names = [c for c in conda_install.get("channels", [])]
        if packages:
            channel_args: list[str] = []
            for ch in channel_names:
                channel_args.extend(["-c", ch])
            run([conda, "install", "-y", "--override-channels", "-n", env_name, *channel_args, *packages], dry_run=dry_run)
        else:
            print(f"{model_id}: skipping conda install of filtered packages: {raw_packages}")

    for pip_args in install.get("pip", []):
        run([conda, "run", "-n", env_name, "python", "-m", "pip", "install", *shlex.split(pip_args)], dry_run=dry_run)

    for command in install.get("commands", []):
        run([conda, "run", "-n", env_name, "bash", "-lc", command], dry_run=dry_run)

    if not dry_run:
        stamp.write_text("installed\n", encoding="utf-8")


def download_url(url: str, dest: Path, *, dry_run: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"{dest}: already downloaded")
        return
    print(f"+ download {url} -> {dest}")
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp.replace(dest)
            return
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if last_error is not None:
        raise last_error


def download_url_with_fallbacks(asset: dict[str, Any], dest: Path, *, dry_run: bool) -> None:
    urls = [asset["url"], *asset.get("fallback_urls", [])]
    failures = []
    for index, url in enumerate(urls):
        try:
            download_url(url, dest, dry_run=dry_run)
            return
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            if index < len(urls) - 1:
                print(f"{dest}: primary download failed; trying fallback")
    raise RuntimeError("All downloads failed for " + str(dest) + "\n" + "\n".join(failures))


def download_hf(conda: str, env_name: str, asset: dict[str, Any], *, dry_run: bool) -> None:
    dest = expand_path(asset["path"])
    if dest.exists() and (dest.is_dir() or dest.stat().st_size > 0):
        print(f"{dest}: already downloaded")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if asset["kind"] == "hf_repo":
        command = [
            conda,
            "run",
            "-n",
            env_name,
            "hf",
            "download",
            asset["repo_id"],
            "--local-dir",
            str(dest),
        ]
    else:
        command = [
            conda,
            "run",
            "-n",
            env_name,
            "hf",
            "download",
            asset["repo_id"],
            asset["filename"],
            "--local-dir",
            str(dest.parent),
        ]
    run(command, dry_run=dry_run)


def download_assets(conda: str, model_id: str, model: dict[str, Any], *, dry_run: bool, include_large: bool) -> None:
    env_name = model["env_name"]
    for asset in model.get("assets", []):
        if asset.get("large") and not include_large:
            print(f"{model_id}: skipping large asset {asset.get('repo_id') or asset.get('url')}")
            continue
        kind = asset["kind"]
        if kind == "url":
            download_url_with_fallbacks(asset, expand_path(asset["path"]), dry_run=dry_run)
        elif kind in {"hf", "hf_repo"}:
            download_hf(conda, env_name, asset, dry_run=dry_run)
        elif kind == "manual":
            path = expand_path(asset["path"])
            if not path.exists():
                print(f"{model_id}: manual asset needed at {path}: {asset.get('description', 'manual download required')}")
        else:
            raise ValueError(f"Unsupported asset kind for {model_id}: {kind}")

    for asset in model.get("manual_assets", []):
        path = expand_path(asset["path"])
        if not path.exists():
            print(f"{model_id}: manual asset needed at {path}: {asset['description']}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    conda = config.get("defaults", {}).get("conda_executable", "conda")
    if shutil.which(conda) is None:
        raise SystemExit(f"Conda executable not found: {conda}")

    selected = args.models or list(config["models"])
    unknown = sorted(set(selected) - set(config["models"]))
    if unknown:
        raise SystemExit(f"Unknown model IDs: {', '.join(unknown)}")

    for model_id in selected:
        model = config["models"][model_id]
        profile = config["profiles"][model["profile"]]
        print(f"\n== {model_id} -> {model['env_name']} ==")
        if not args.skip_envs:
            validate_profile_paths(profile, model_id)
            create_env(conda, model["env_name"], str(profile["python"]), dry_run=args.dry_run)
            install_profile(
                conda,
                model["env_name"],
                profile,
                model_id,
                dry_run=args.dry_run,
                force=args.force_install,
            )
        if args.download_assets and not args.skip_assets:
            download_assets(
                conda,
                model_id,
                model,
                dry_run=args.dry_run,
                include_large=args.download_large_assets,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
