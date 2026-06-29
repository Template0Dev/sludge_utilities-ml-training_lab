from __future__ import annotations

import ast
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from uuid6 import uuid6

from .optuna_support import (
    assert_tuning_protocol,
    config_signature,
    export_study,
    file_sha256,
    load_summary,
    macro_mae,
    originals_for_well,
    prepare_study,
)
from .pipeline_config import load_pipeline_config, optional_wells, resolve_project_path, resolve_training_wells


BASE_FEATURE_COLUMNS = ("interval_start", "interval_end")
TARGET_COLUMNS = ("sandstone_sludge", "siltstone_sludge", "argillite_sludge")

SEARCH_SPACE = {
    "pca_components": {"type": "categorical", "choices": [16, 32, 64, 128]},
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
    "depth": {"type": "int", "low": 4, "high": 10},
    "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 30.0, "log": True},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "rsm": {"type": "float", "low": 0.5, "high": 1.0},
    "random_strength": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
}

BASELINE = {
    "pca_components": 32, "learning_rate": 0.015, "depth": 5, "l2_leaf_reg": 10.0,
    "subsample": 0.8, "rsm": 0.7, "random_strength": 1.0,
}
FIXED_PARAMETERS = {
    "iterations": 5000, "bootstrap_type": "Bernoulli", "loss_function": "MultiRMSE",
    "eval_metric": "MultiRMSE", "early_stopping_rounds": 200, "random_seed": 42,
    "task_type": "CPU",
}


@dataclass(frozen=True)
class GBTuningConfig:
    project_root: Path
    study_name: str = "catboost_target_well_v1"
    total_trials: int = 50
    smoke_mode: bool = False
    include_augmented_records: bool = True
    training_wells: tuple[int, ...] | None = None
    target_well: int = 1

    @classmethod
    def from_file(cls, project_root: Path, path: Path) -> "GBTuningConfig":
        config = load_pipeline_config(path)
        optuna_config = config.get("optuna", {})
        return cls(
            project_root=project_root,
            study_name=optuna_config.get("study_name", cls.study_name),
            total_trials=int(optuna_config.get("total_trials", cls.total_trials)),
            smoke_mode=bool(optuna_config.get("smoke_mode", cls.smoke_mode)),
            include_augmented_records=bool(config.get("include_augmented_records", cls.include_augmented_records)),
            training_wells=optional_wells(config.get("training_wells")),
            target_well=int(config.get("target_well", cls.target_well)),
        )

    @property
    def dataset_path(self) -> Path:
        return self.project_root / "data/meta/processed/gb/metadata_dinov3_embeddings.parquet"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output/gb/optuna"


@dataclass(frozen=True)
class GBFinalConfig:
    project_root: Path
    optuna_summary_path: Path
    include_augmented_records: bool = True
    training_wells: tuple[int, ...] | None = None
    target_well: int = 1
    save_predictions: bool = True

    @classmethod
    def from_file(cls, project_root: Path, path: Path) -> "GBFinalConfig":
        config = load_pipeline_config(path)
        final_config = config.get("final_training", {})
        optuna_summary_path = final_config.get(
            "optuna_summary_path",
            "output/gb/optuna/catboost_target_well_v1_summary.json",
        )
        return cls(
            project_root=project_root,
            optuna_summary_path=resolve_project_path(project_root, optuna_summary_path),
            include_augmented_records=bool(config.get("include_augmented_records", cls.include_augmented_records)),
            training_wells=optional_wells(config.get("training_wells")),
            target_well=int(config.get("target_well", cls.target_well)),
            save_predictions=bool(final_config.get("save_predictions", cls.save_predictions)),
        )


def parse_embedding(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.float32)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df["sludge_dinov3_emb"] = df["sludge_dinov3_emb"].map(parse_embedding)
    return df


def training_records(df: pd.DataFrame, wells: tuple[int, ...], include_augmented_records: bool) -> pd.DataFrame:
    result = df[df["well_id"].isin(wells)]
    if not include_augmented_records:
        result = result[~result["is_augmented"].astype(bool)]
    result = result.copy()
    if result.empty:
        raise ValueError("No training samples remain after applying the augmented-record filter.")
    return result


def fit_fold_features(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, components: int
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    pca = PCA(n_components=components, random_state=42)
    train_embedding = pca.fit_transform(np.stack(train_df["sludge_dinov3_emb"].to_numpy()))
    validation_embedding = pca.transform(np.stack(validation_df["sludge_dinov3_emb"].to_numpy()))
    columns = [f"sludge_emb_pca_{index}" for index in range(components)]
    train_features = pd.concat([
        train_df[list(BASE_FEATURE_COLUMNS)].reset_index(drop=True),
        pd.DataFrame(train_embedding, columns=columns),
    ], axis=1)
    validation_features = pd.concat([
        validation_df[list(BASE_FEATURE_COLUMNS)].reset_index(drop=True),
        pd.DataFrame(validation_embedding, columns=columns),
    ], axis=1)
    return train_features, validation_features, pca


def normalized_predictions(model: CatBoostRegressor, features: pd.DataFrame) -> np.ndarray:
    predictions = np.clip(np.asarray(model.predict(features)), 0.0, None)
    totals = predictions.sum(axis=1, keepdims=True)
    return predictions / np.where(totals == 0.0, 1.0, totals) * 100.0


def model_parameters(params: dict[str, Any], *, iterations: int) -> dict[str, Any]:
    return {
        "iterations": iterations, "learning_rate": params["learning_rate"], "depth": params["depth"],
        "l2_leaf_reg": params["l2_leaf_reg"], "bootstrap_type": "Bernoulli",
        "subsample": params["subsample"], "rsm": params["rsm"], "random_strength": params["random_strength"],
        "loss_function": "MultiRMSE", "eval_metric": "MultiRMSE", "random_seed": 42,
        "task_type": "CPU", "allow_writing_files": False, "verbose": False,
    }


def suggested_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "pca_components": trial.suggest_categorical("pca_components", [16, 32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
    }


def tune_catboost(config: GBTuningConfig) -> Path:
    np.random.seed(42)
    df = load_dataset(config.dataset_path)
    training_wells = resolve_training_wells(
        df, target_well=config.target_well, configured_wells=config.training_wells
    )
    assert_tuning_protocol(df, training_wells, config.target_well)
    iterations = 20 if config.smoke_mode else FIXED_PARAMETERS["iterations"]
    early_stopping = 5 if config.smoke_mode else FIXED_PARAMETERS["early_stopping_rounds"]
    study_name = f"{config.study_name}_smoke" if config.smoke_mode else config.study_name
    dataset_hash = file_sha256(config.dataset_path)
    protocol = {
        "training_wells": list(training_wells), "target_well": config.target_well,
        "include_augmented_records": config.include_augmented_records,
        "validation_strategy": "target_well", "validation_originals_only": True, "pca_fit_inside_fold": True,
        "max_iterations": iterations, "early_stopping_rounds": early_stopping,
    }
    signature = config_signature({
        "dataset_sha256": dataset_hash, "protocol": protocol, "search_space": SEARCH_SPACE,
        "fixed_parameters": FIXED_PARAMETERS, "target_columns": TARGET_COLUMNS,
        "base_feature_columns": BASE_FEATURE_COLUMNS,
    })
    study, remaining = prepare_study(
        study_name=study_name, database_path=config.output_dir / f"{study_name}.db", signature=signature,
        total_trials=1 if config.smoke_mode else config.total_trials, baseline=BASELINE,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggested_params(trial)
        train_df = training_records(df, training_wells, config.include_augmented_records)
        validation_df = originals_for_well(df, config.target_well)
        if config.target_well in train_df["well_id"].unique():
            raise AssertionError("Target leakage detected.")
        x_train, x_validation, _ = fit_fold_features(train_df, validation_df, params["pca_components"])
        y_train = train_df[list(TARGET_COLUMNS)].reset_index(drop=True)
        y_validation = validation_df[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        model = CatBoostRegressor(**model_parameters(params, iterations=iterations))
        model.fit(x_train, y_train, eval_set=(x_validation, y_validation), early_stopping_rounds=early_stopping)
        score, target_scores = macro_mae(y_validation, normalized_predictions(model, x_validation))
        best_iteration = max(1, model.get_best_iteration() + 1)
        trial.report(score, 0)
        if trial.should_prune():
            raise optuna.TrialPruned("Pruned after target-well validation")
        trial.set_user_attr("validation_well", config.target_well)
        trial.set_user_attr("validation_score", score)
        trial.set_user_attr("validation_target_scores", target_scores)
        trial.set_user_attr("fold_scores", [score])
        trial.set_user_attr("fold_target_scores", [target_scores])
        trial.set_user_attr("fold_best_iterations", [best_iteration])
        return score

    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    return export_study(
        study=study, output_dir=config.output_dir, model_type="catboost", dataset_path=config.dataset_path,
        dataset_hash=dataset_hash, signature=signature, protocol=protocol, search_space=SEARCH_SPACE,
        fixed_parameters=FIXED_PARAMETERS,
    )


def detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        target: {
            "mae": float(mean_absolute_error(y_true[:, index], y_pred[:, index])),
            "rmse": float(root_mean_squared_error(y_true[:, index], y_pred[:, index])),
            "r2": float(r2_score(y_true[:, index], y_pred[:, index])),
        }
        for index, target in enumerate(TARGET_COLUMNS)
    }


def train_final_catboost(config: GBFinalConfig) -> tuple[Path, dict[str, Any]]:
    dataset_path = config.project_root / "data/meta/processed/gb/metadata_dinov3_embeddings.parquet"
    summary = load_summary(config.optuna_summary_path, model_type="catboost", dataset_path=dataset_path)
    df = load_dataset(dataset_path)
    training_wells = resolve_training_wells(
        df, target_well=config.target_well, configured_wells=config.training_wells
    )
    protocol = summary.get("protocol", {})
    summary_training_wells = protocol.get("training_wells", protocol.get("tuning_wells"))
    summary_target_well = protocol.get("target_well", protocol.get("holdout_well"))
    if summary_training_wells != list(training_wells) or summary_target_well != config.target_well:
        raise ValueError("The Optuna summary uses a different training/holdout well protocol.")
    if protocol.get("include_augmented_records", True) != config.include_augmented_records:
        raise ValueError("The Optuna summary was produced with a different augmented-record setting.")
    if not protocol.get("pca_fit_inside_fold"):
        raise ValueError("The Optuna summary does not guarantee fold-local PCA fitting.")
    assert_tuning_protocol(df, training_wells, config.target_well)
    params = summary["best_trial"]["params"]
    best_iterations = summary["best_trial"]["user_attrs"].get("fold_best_iterations")
    if not best_iterations or any(iteration < 1 for iteration in best_iterations):
        raise ValueError("The winning trial does not contain valid fold best iterations.")
    final_iterations = max(1, int(np.median(best_iterations)))
    train_df = training_records(df, training_wells, config.include_augmented_records)
    holdout_df = originals_for_well(df, config.target_well)
    x_train, x_holdout, pca = fit_fold_features(train_df, holdout_df, params["pca_components"])
    y_train = train_df[list(TARGET_COLUMNS)].reset_index(drop=True)
    y_holdout = holdout_df[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    model = CatBoostRegressor(**model_parameters(params, iterations=final_iterations))
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    model.fit(x_train, y_train)
    predictions = normalized_predictions(model, x_holdout)
    macro_score, target_mae = macro_mae(y_holdout, predictions)
    training_id = str(uuid6())
    model_dir = config.project_root / "output/gb/models"
    meta_dir = config.project_root / "output/gb/meta"
    predictions_dir = config.project_root / "output/gb/predictions"
    for directory in (model_dir, meta_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"[{training_id}] CatBoost.cbm"
    pca_path = model_dir / f"[{training_id}] SludgePCA.joblib"
    model.save_model(model_path)
    joblib.dump(pca, pca_path)
    metadata = {
        "training_id": training_id, "model_type": "CatBoost", "model_file": model_path.name,
        "pca_file": pca_path.name, "feature_columns": x_train.columns.tolist(),
        "training_started_at": started_at, "training_completed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "hyperparameters": params, "final_iterations": final_iterations,
        "training_wells": list(training_wells), "target_well": config.target_well,
        "final_training_metrics": {"macro_mae": macro_score, "target_mae": dict(zip(TARGET_COLUMNS, target_mae)),
                                   "per_target": detailed_metrics(y_holdout, predictions)},
        "optuna": {"study_name": summary["study_name"], "trial_number": summary["best_trial"]["number"],
                   "objective_value": summary["best_trial"]["value"], "summary_path": str(config.optuna_summary_path)},
    }
    meta_path = meta_dir / f"[{training_id}] CatBoost.txt"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if config.save_predictions:
        prediction_frame = holdout_df[list(TARGET_COLUMNS)].reset_index(drop=True).add_suffix("_true")
        for index, target in enumerate(TARGET_COLUMNS):
            prediction_frame[f"{target}_pred"] = predictions[:, index]
        prediction_frame.to_csv(predictions_dir / f"[{training_id}] CatBoost (Result).csv", index=False)
    return model_path, metadata
