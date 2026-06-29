from __future__ import annotations

import json
from pathlib import Path


MARKER_FILE_NAME = ".last_tune.json"


def last_tune_marker_path(tuning_root: Path) -> Path:
    return tuning_root / MARKER_FILE_NAME


def write_last_tune_id(tuning_root: Path, tune_id: str) -> None:
    _validate_tune_id(tune_id)
    marker_path = last_tune_marker_path(tuning_root)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(tune_id, ensure_ascii=False) + "\n", encoding="utf-8")


def read_last_tune_id(tuning_root: Path) -> str:
    marker_path = last_tune_marker_path(tuning_root)
    if not marker_path.is_file():
        raise FileNotFoundError(f"Latest tuning marker does not exist: {marker_path}. Run tuning first.")
    try:
        value = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Latest tuning marker is not valid JSON: {marker_path}") from error
    if not isinstance(value, str):
        raise ValueError(f"Latest tuning marker must contain a non-empty JSON string: {marker_path}")
    _validate_tune_id(value)
    return value


def _validate_tune_id(tune_id: str) -> None:
    if not tune_id:
        raise ValueError("tune_id cannot be empty.")
    if Path(tune_id).name != tune_id:
        raise ValueError("tune_id must be a folder name, not a path.")
