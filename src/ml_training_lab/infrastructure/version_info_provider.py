from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path


def version_info(project_root: Path) -> str:
    lines = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"package_version={_distribution_version('gb-training-lab')}",
    ]
    for package in ("numpy", "pandas", "optuna", "catboost", "torch", "pytorch-lightning", "scikit-learn"):
        lines.append(f"{package}={_distribution_version(package)}")
    lines.extend(_git_info(project_root))
    return "\n".join(lines) + "\n"


def _distribution_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_info(project_root: Path) -> list[str]:
    commands = {
        "git_commit": ["git", "rev-parse", "HEAD"],
        "git_branch": ["git", "branch", "--show-current"],
    }
    result: list[str] = []
    for name, command in commands.items():
        result.append(f"{name}={_run_git(command, project_root)}")
    dirty = _run_git(["git", "status", "--porcelain"], project_root)
    result.append(f"git_dirty={bool(dirty)}")
    return result


def _run_git(command: list[str], project_root: Path) -> str:
    try:
        completed = subprocess.run(command, cwd=project_root, check=True, text=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return completed.stdout.strip()
