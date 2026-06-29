from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_training_lab.infrastructure.version_info_provider import version_info
from ml_training_lab.shared.json_writer import write_json


def write_tuning_metadata(tune_dir: Path, config_snapshot: dict[str, Any], project_root: Path) -> None:
    write_json(tune_dir / "config_snapshot.json", config_snapshot)
    (tune_dir / "version_info.txt").write_text(version_info(project_root), encoding="utf-8")
