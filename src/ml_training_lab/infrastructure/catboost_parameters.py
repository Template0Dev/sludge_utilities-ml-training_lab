from __future__ import annotations

from typing import Any


def model_parameters(params: dict[str, Any], fixed_hyper_params: dict[str, Any], *, iterations: int) -> dict[str, Any]:
    ignored_search_params = {"pca_components"}
    model_params = {key: value for key, value in params.items() if key not in ignored_search_params}
    fixed_params = {
        key: value
        for key, value in fixed_hyper_params.items()
        if key not in {"iterations", "early_stopping_rounds"}
    }
    return {
        **fixed_params,
        **model_params,
        "iterations": iterations,
        "allow_writing_files": False,
        "verbose": False,
    }
