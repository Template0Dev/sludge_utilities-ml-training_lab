from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna


SCHEMA_VERSION = 1


def prepare_study(
    *,
    study_name: str,
    database_path: Path,
    signature: str,
    total_trials: int,
    baseline: dict[str, Any],
    pruner: optuna.pruners.BasePruner,
) -> tuple[optuna.Study, int]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{database_path.resolve()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
        load_if_exists=True,
    )
    stored_signature = study.user_attrs.get("configuration_signature")
    if stored_signature is not None and stored_signature != signature:
        raise ValueError(
            "The existing Optuna study was created with an incompatible dataset, "
            "protocol, or search space. Change the study name or remove its database."
        )
    study.set_user_attr("configuration_signature", signature)
    study.set_user_attr("schema_version", SCHEMA_VERSION)
    if not study.trials:
        study.enqueue_trial(baseline, user_attrs={"is_baseline": True})
    finished_trials = sum(trial.state.is_finished() for trial in study.trials)
    return study, max(0, total_trials - finished_trials)
