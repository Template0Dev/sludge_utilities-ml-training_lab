from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna

from ml_training_lab.domain.tuning_protocol import validation_well_objective_metadata
from ml_training_lab.infrastructure.optuna_study_repository import SCHEMA_VERSION
from ml_training_lab.shared.json_writer import write_json


def export_study(
    *,
    study: optuna.Study,
    output_dir: Path,
    model_type: str,
    dataset_path: Path,
    dataset_hash: str,
    signature: str,
    protocol: dict[str, Any],
    search_params: dict[str, Any],
    fixed_hyper_params: dict[str, Any],
    initial_hyper_params: dict[str, Any],
    config_snapshot: dict[str, Any],
) -> tuple[Path, Path]:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("The study has no completed trial to export.")
    best_trial = study.best_trial
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / f"{study.study_name}_trials.csv"
    study.trials_dataframe(
        attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "params", "user_attrs", "state")
    ).to_csv(trials_path, index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "study_name": study.study_name,
        "configuration_signature": signature,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "objective": validation_well_objective_metadata(protocol),
        "protocol": protocol,
        "search_params": search_params,
        "fixed_hyper_params": fixed_hyper_params,
        "initial_hyper_params": initial_hyper_params,
        "config_snapshot": config_snapshot,
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        },
        "trial_counts": {
            state.name: len([trial for trial in study.trials if trial.state == state])
            for state in optuna.trial.TrialState
        },
    }
    summary_path = output_dir / f"{study.study_name}_summary.json"
    write_json(summary_path, summary)
    return summary_path, trials_path
