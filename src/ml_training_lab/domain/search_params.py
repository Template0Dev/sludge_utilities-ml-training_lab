from __future__ import annotations

from typing import Any

import optuna


def suggest_params(trial: optuna.Trial, search_params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, definition in search_params.items():
        param_type = definition["type"]
        if param_type == "categorical":
            result[name] = trial.suggest_categorical(name, definition["choices"])
        elif param_type == "float":
            result[name] = trial.suggest_float(
                name,
                float(definition["low"]),
                float(definition["high"]),
                log=bool(definition.get("log", False)),
                step=definition.get("step"),
            )
        elif param_type == "int":
            result[name] = trial.suggest_int(
                name,
                int(definition["low"]),
                int(definition["high"]),
                log=bool(definition.get("log", False)),
                step=int(definition.get("step", 1)),
            )
        else:
            raise ValueError(f"Unsupported search parameter type for {name}: {param_type}")
    return result
