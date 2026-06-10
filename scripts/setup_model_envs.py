#!/usr/bin/env python3
"""Create per-model Conda environments and fetch benchmark assets."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
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


def run(command: list[str], *, dry_run: bool, cwd: Path = ROOT) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"+ {printable}")
    if dry_run:
        return
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    subprocess.run(command, cwd=cwd, check=True, env=env)


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
    run([conda, "create", "-y", "-n", env_name, f"python={python_version}", "pip"], dry_run=dry_run)


def install_profile(conda: str, env_name: str, profile: dict[str, Any], model_id: str, *, dry_run: bool, force: bool) -> None:
    stamp_dir = ROOT / ".model_env_stamps"
    stamp_dir.mkdir(exist_ok=True)
    stamp = stamp_dir / f"{model_id}.stamp"
    if stamp.exists() and not force:
        print(f"{model_id}: install stamp exists ({stamp}); use --force-install to rerun")
        return

    install = profile.get("install", {})
    conda_install = install.get("conda")
    if conda_install:
        channels = []
        for channel in conda_install.get("channels", []):
            channels.extend(["-c", channel])
        packages = list(conda_install.get("packages", []))
        run([conda, "install", "-y", "-n", env_name, *channels, *packages], dry_run=dry_run)

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
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


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
