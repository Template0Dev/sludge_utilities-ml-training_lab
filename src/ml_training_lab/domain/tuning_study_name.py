from __future__ import annotations

from ml_training_lab.presentation.common_config import TuningParams


def effective_study_name(tuning_params: TuningParams) -> str:
    if tuning_params.smoke_mode:
        return f"{tuning_params.study_name}_smoke"
    return tuning_params.study_name
