from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd


SCHEMA_VERSION = 1


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(json_value(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def macro_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    target_scores = np.mean(np.abs(y_true - y_pred), axis=0)
    return float(np.mean(target_scores)), target_scores.astype(float).tolist()


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


def export_study(
    *,
    study: optuna.Study,
    output_dir: Path,
    model_type: str,
    dataset_path: Path,
    dataset_hash: str,
    signature: str,
    protocol: dict[str, Any],
    search_space: dict[str, Any],
    fixed_parameters: dict[str, Any],
) -> Path:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("The study has no completed trial to export.")
    best_trial = study.best_trial
    output_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe(attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "params", "user_attrs", "state")).to_csv(
        output_dir / f"{study.study_name}_trials.csv", index=False
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "study_name": study.study_name,
        "configuration_signature": signature,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "objective": {
            "name": "equal-well macro MAE",
            "direction": "minimize",
            "aggregation": "mean of per-validation-well target-macro MAE",
        },
        "protocol": protocol,
        "search_space": search_space,
        "fixed_parameters": fixed_parameters,
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        },
        "trial_counts": {state.name: len([trial for trial in study.trials if trial.state == state]) for state in optuna.trial.TrialState},
    }
    summary_path = output_dir / f"{study.study_name}_summary.json"
    summary_path.write_text(json.dumps(json_value(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


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


def originals_for_well(df: pd.DataFrame, well_id: int) -> pd.DataFrame:
    result = df[(df["well_id"] == well_id) & (~df["is_augmented"].astype(bool))].copy()
    if result.empty:
        raise ValueError(f"Well {well_id} has no original validation samples.")
    return result


def assert_tuning_protocol(df: pd.DataFrame, tuning_wells: tuple[int, ...], holdout_well: int) -> None:
    available = set(df["well_id"].unique().tolist())
    required = set(tuning_wells) | {holdout_well}
    if not required.issubset(available):
        raise ValueError(f"Dataset wells {sorted(available)} do not contain required wells {sorted(required)}")
    if holdout_well in tuning_wells:
        raise ValueError("The final holdout well must not participate in tuning.")
