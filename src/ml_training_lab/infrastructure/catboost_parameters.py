from __future__ import annotations

from typing import Any

from ml_training_lab.shared.device_selector import catboost_task_type


def model_parameters(params: dict[str, Any], fixed_hyper_params: dict[str, Any], *, iterations: int) -> dict[str, Any]:
    ignored_search_params = {"pca_components"}
    model_params = {key: value for key, value in params.items() if key not in ignored_search_params}
    fixed_params = {
        key: value
        for key, value in fixed_hyper_params.items()
        if key not in {"iterations", "early_stopping_rounds", "task_type"}
    }
    return {
        **fixed_params,
        **model_params,
        "iterations": iterations,
        "task_type": catboost_task_type(fixed_hyper_params.get("task_type")),
        "allow_writing_files": False,
        "verbose": False,
    }
