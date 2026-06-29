from __future__ import annotations

from pathlib import Path

from ml_training_lab.infrastructure.last_tune_marker import read_last_tune_id


def latest_tuning_summary_path(tuning_root: Path, study_name: str) -> Path:
    tune_id = read_last_tune_id(tuning_root)
    tune_dir = tuning_root / tune_id
    if not tune_dir.is_dir():
        raise FileNotFoundError(f"Latest tuning folder does not exist: {tune_dir}")
    summary_path = tune_dir / f"{study_name}_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Latest tuning summary does not exist: {summary_path}")
    return summary_path
