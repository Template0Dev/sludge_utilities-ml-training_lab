from __future__ import annotations

from pathlib import Path


def create_tuning_run_dir(tuning_root: Path, tune_id: str) -> Path:
    tune_dir = tuning_root / tune_id
    tune_dir.mkdir(parents=True, exist_ok=False)
    return tune_dir
