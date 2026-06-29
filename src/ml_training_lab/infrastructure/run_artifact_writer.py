from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_training_lab.infrastructure.version_info_provider import version_info
from ml_training_lab.shared.json_writer import write_json


def create_run_dir(runs_root: Path, run_id: str) -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(run_dir: Path, request_meta: dict[str, Any], config_snapshot: dict[str, Any], project_root: Path) -> None:
    write_json(run_dir / "request_meta.json", request_meta)
    write_json(run_dir / "config_snapshot.json", config_snapshot)
    (run_dir / "version_info.txt").write_text(version_info(project_root), encoding="utf-8")
