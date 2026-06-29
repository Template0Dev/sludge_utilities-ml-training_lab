from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml_training_lab.infrastructure.optuna_study_repository import SCHEMA_VERSION
from ml_training_lab.shared.file_hash import file_sha256


def load_summary(path: Path, *, model_type: str, dataset_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Optuna summary does not exist: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported Optuna summary schema: {summary.get('schema_version')}")
    if summary.get("model_type") != model_type:
        raise ValueError(f"Expected a {model_type} summary, got {summary.get('model_type')}")
    actual_hash = file_sha256(dataset_path)
    if summary.get("dataset_sha256") != actual_hash:
        raise ValueError("The tuning summary was produced from a different dataset file.")
    return summary
