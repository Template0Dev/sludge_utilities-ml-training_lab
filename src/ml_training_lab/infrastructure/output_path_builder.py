from __future__ import annotations

from pathlib import Path

from ml_training_lab.presentation.common_config import AppOutputConfig


def model_runs_root(project_root: Path, output: AppOutputConfig, model_sub_folder: str) -> Path:
    return project_root.joinpath(
        output.output_base_folder,
        model_sub_folder,
        output.output_runs_prefix_for_saving,
    )


def model_tuning_root(project_root: Path, output: AppOutputConfig, model_sub_folder: str) -> Path:
    return project_root.joinpath(
        output.output_base_folder,
        model_sub_folder,
        output.output_tuning_prefix_for_saving,
    )
