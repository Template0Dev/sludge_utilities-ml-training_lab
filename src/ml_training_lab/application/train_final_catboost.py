from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from ml_training_lab.domain.metrics import detailed_metrics, macro_mae
from ml_training_lab.domain.prediction_normalization import normalize_prediction_rows
from ml_training_lab.domain.record_selection import originals_for_well, training_records
from ml_training_lab.domain.training_wells import resolve_training_wells
from ml_training_lab.domain.tuning_protocol import assert_tuning_protocol
from ml_training_lab.infrastructure.catboost_feature_matrix import fit_fold_features
from ml_training_lab.infrastructure.catboost_parameters import model_parameters
from ml_training_lab.infrastructure.embedding_joiner import load_feature_dataset
from ml_training_lab.infrastructure.optuna_summary_reader import load_summary
from ml_training_lab.infrastructure.output_path_builder import model_runs_root, model_tuning_root
from ml_training_lab.infrastructure.pipeline_config_reader import read_pipeline_config
from ml_training_lab.infrastructure.run_artifact_writer import create_run_dir, write_run_metadata
from ml_training_lab.presentation.model_configs import CatBoostPipelineConfig
from ml_training_lab.presentation.requests import CatBoostRequest
from ml_training_lab.presentation.responses import TrainingResultDto
from ml_training_lab.shared.clock import now_iso
from ml_training_lab.shared.uuid_generator import new_uuid


def train_final_catboost(request: CatBoostRequest) -> TrainingResultDto:
    config = read_pipeline_config(request.config_path, CatBoostPipelineConfig)
    summary_path = _summary_path(request.project_root, config)
    summary = load_summary(summary_path, model_type="catboost", dataset_path=request.base_dataset_path)
    df = load_feature_dataset(
        base_dataset_path=request.base_dataset_path,
        embedding_dataset_path=request.embedding_dataset_path,
        features=config.features,
    )
    training_wells = resolve_training_wells(
        df, target_well=config.target_well, configured_wells=config.training_wells
    )
    _assert_summary_protocol(summary, training_wells, config)
    assert_tuning_protocol(df, training_wells, config.target_well)
    params = summary["best_trial"]["params"]
    best_iterations = summary["best_trial"]["user_attrs"].get("fold_best_iterations")
    if not best_iterations or any(iteration < 1 for iteration in best_iterations):
        raise ValueError("The winning trial does not contain valid fold best iterations.")
    final_iterations = max(1, int(np.median(best_iterations)))
    train_df = training_records(df, training_wells, config.include_augmented_records)
    holdout_df = originals_for_well(df, config.target_well)
    x_train, x_holdout, transformers = fit_fold_features(
        train_df, holdout_df, features=config.features, components=params.get("pca_components", 0)
    )
    y_train = train_df[list(config.target_columns)].reset_index(drop=True)
    y_holdout = holdout_df[list(config.target_columns)].to_numpy(dtype=float)
    model = CatBoostRegressor(**model_parameters(params, config.fixed_hyper_params, iterations=final_iterations))
    started_at = now_iso()
    model.fit(x_train, y_train)
    predictions = normalize_prediction_rows(model.predict(x_holdout))
    macro_score, target_mae = macro_mae(y_holdout, predictions)
    metrics = {
        "macro_mae": macro_score,
        "target_mae": dict(zip(config.target_columns, target_mae, strict=True)),
        "per_target": detailed_metrics(y_holdout, predictions, config.target_columns),
    }
    run_id = new_uuid()
    run_dir = create_run_dir(
        model_runs_root(request.project_root, config.output, config.output.output_gb_sub_folder),
        run_id,
    )
    model_path = run_dir / "model.cbm"
    embeddings_path = run_dir / "embeddings.joblib" if transformers else None
    model.save_model(model_path)
    if embeddings_path is not None:
        joblib.dump(transformers, embeddings_path)
    predictions_path = None
    if config.final_training.save_predictions:
        predictions_path = run_dir / "predictions.csv"
        prediction_frame = holdout_df[list(config.target_columns)].reset_index(drop=True).add_suffix("_true")
        for index, target in enumerate(config.target_columns):
            prediction_frame[f"{target}_pred"] = predictions[:, index]
        prediction_frame.to_csv(predictions_path, index=False)
    request_meta = {
        "training_id": run_id,
        "model_type": "CatBoost",
        "model_file": model_path.name,
        "embeddings_file": embeddings_path.name if embeddings_path else None,
        "predictions_file": predictions_path.name if predictions_path else None,
        "feature_columns": x_train.columns.tolist(),
        "target_columns": config.target_columns,
        "features": config.features,
        "training_started_at": started_at,
        "training_completed_at": now_iso(),
        "hyperparameters": params,
        "final_iterations": final_iterations,
        "training_wells": list(training_wells),
        "target_well": config.target_well,
        "final_training_metrics": metrics,
        "optuna": {
            "study_name": summary["study_name"],
            "trial_number": summary["best_trial"]["number"],
            "objective_value": summary["best_trial"]["value"],
            "summary_path": str(summary_path),
        },
    }
    write_run_metadata(
        run_dir,
        request_meta,
        {"request": request, "config": config},
        request.project_root,
    )
    return TrainingResultDto(
        run_id=run_id,
        run_dir=run_dir,
        model_path=model_path,
        request_meta_path=run_dir / "request_meta.json",
        config_snapshot_path=run_dir / "config_snapshot.json",
        version_info_path=run_dir / "version_info.txt",
        predictions_path=predictions_path,
        embeddings_path=embeddings_path,
        metrics=metrics,
    )


def _assert_summary_protocol(
    summary: dict[str, object],
    training_wells: tuple[int, ...],
    config: CatBoostPipelineConfig,
) -> None:
    protocol = summary.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("The Optuna summary does not contain a valid protocol.")
    summary_training_wells = protocol.get("training_wells", protocol.get("tuning_wells"))
    summary_target_well = protocol.get("target_well", protocol.get("holdout_well"))
    if summary_training_wells != list(training_wells) or summary_target_well != config.target_well:
        raise ValueError("The Optuna summary uses a different training/holdout well protocol.")
    if protocol.get("include_augmented_records", True) != config.include_augmented_records:
        raise ValueError("The Optuna summary was produced with a different augmented-record setting.")
    if not protocol.get("pca_fit_inside_fold"):
        raise ValueError("The Optuna summary does not guarantee fold-local PCA fitting.")


def _summary_path(project_root, config: CatBoostPipelineConfig):
    summary_file_name = Path(config.final_training.optuna_summary_path).name
    return (
        model_tuning_root(project_root, config.output, config.output.output_gb_sub_folder)
        / summary_file_name
    )
