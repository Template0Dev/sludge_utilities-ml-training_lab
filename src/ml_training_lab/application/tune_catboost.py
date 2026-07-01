from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostRegressor

from ml_training_lab.domain.config_signature import config_signature
from ml_training_lab.domain.metrics import macro_mae
from ml_training_lab.domain.prediction_normalization import normalize_prediction_rows
from ml_training_lab.domain.record_selection import originals_for_well, training_records
from ml_training_lab.domain.search_params import suggest_params
from ml_training_lab.domain.training_wells import resolve_dataset_split
from ml_training_lab.domain.tuning_protocol import assert_tuning_protocol
from ml_training_lab.domain.tuning_study_name import effective_study_name
from ml_training_lab.infrastructure.catboost_feature_matrix import fit_fold_features
from ml_training_lab.infrastructure.catboost_parameters import model_parameters
from ml_training_lab.infrastructure.embedding_joiner import load_feature_dataset
from ml_training_lab.infrastructure.last_tune_marker import write_last_tune_id
from ml_training_lab.infrastructure.optuna_study_exporter import export_study
from ml_training_lab.infrastructure.optuna_study_repository import prepare_study
from ml_training_lab.infrastructure.output_path_builder import model_tuning_root
from ml_training_lab.infrastructure.pipeline_config_reader import read_pipeline_config
from ml_training_lab.infrastructure.tuning_artifact_writer import write_tuning_metadata
from ml_training_lab.infrastructure.tuning_run_directory import create_tuning_run_dir
from ml_training_lab.presentation.model_configs import CatBoostPipelineConfig
from ml_training_lab.presentation.requests import CatBoostRequest
from ml_training_lab.presentation.responses import OptunaSummaryDto
from ml_training_lab.shared.file_hash import file_sha256
from ml_training_lab.shared.uuid_generator import new_uuid


def tune_catboost(request: CatBoostRequest) -> OptunaSummaryDto:
    np.random.seed(42)
    config = read_pipeline_config(request.config_path, CatBoostPipelineConfig)
    df = load_feature_dataset(
        base_dataset_path=request.base_dataset_path,
        embedding_dataset_path=request.embedding_dataset_path,
        features=config.features,
    )
    split = resolve_dataset_split(
        df,
        training_wells=config.dataset_params.training_wells,
        validation_well=config.dataset_params.validation_well,
        test_well=config.dataset_params.test_well,
    )
    assert_tuning_protocol(df, split.training_wells, split.validation_well)
    iterations = 20 if config.tuning_params.smoke_mode else int(config.fixed_hyper_params["iterations"])
    early_stopping = 5 if config.tuning_params.smoke_mode else int(config.fixed_hyper_params["early_stopping_rounds"])
    study_name = effective_study_name(config.tuning_params)
    dataset_hash = file_sha256(request.base_dataset_path)
    protocol = {
        "training_wells": list(split.training_wells),
        "validation_well": split.validation_well,
        "include_augmented_records": config.include_augmented_records,
        "validation_strategy": "validation_well",
        "validation_originals_only": True,
        "pca_fit_inside_fold": True,
        "max_iterations": iterations,
        "early_stopping_rounds": early_stopping,
        "target_columns": config.target_columns,
        "features": config.features,
    }
    signature = config_signature({
        "dataset_sha256": dataset_hash,
        "protocol": protocol,
        "search_params": config.search_params,
        "fixed_hyper_params": config.fixed_hyper_params,
        "initial_hyper_params": config.initial_hyper_params,
    })
    tuning_root = model_tuning_root(request.project_root, config.output, config.output.output_gb_sub_folder)
    tune_id = new_uuid()
    tune_dir = create_tuning_run_dir(tuning_root, tune_id)
    database_path = tune_dir / f"{study_name}.db"
    study, remaining = prepare_study(
        study_name=study_name,
        database_path=database_path,
        signature=signature,
        total_trials=1 if config.tuning_params.smoke_mode else config.tuning_params.total_trials,
        baseline=config.initial_hyper_params,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=config.tuning_params.pruner_n_startup_trials),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, config.search_params)
        train_df = training_records(df, split.training_wells, config.include_augmented_records)
        validation_df = originals_for_well(df, split.validation_well)
        if split.validation_well in train_df["well_id"].unique():
            raise AssertionError("Target leakage detected.")
        x_train, x_validation, _ = fit_fold_features(
            train_df, validation_df, features=config.features, components=params.get("pca_components", 0)
        )
        y_train = train_df[list(config.target_columns)].reset_index(drop=True)
        y_validation = validation_df[list(config.target_columns)].to_numpy(dtype=float)
        model = CatBoostRegressor(**model_parameters(params, config.fixed_hyper_params, iterations=iterations))
        model.fit(x_train, y_train, eval_set=(x_validation, y_validation), early_stopping_rounds=early_stopping)
        score, target_scores = macro_mae(y_validation, normalize_prediction_rows(model.predict(x_validation)))
        best_iteration = max(1, model.get_best_iteration() + 1)
        trial.report(score, 0)
        if trial.should_prune():
            raise optuna.TrialPruned("Pruned after validation-well validation")
        trial.set_user_attr("validation_well", split.validation_well)
        trial.set_user_attr("validation_score", score)
        trial.set_user_attr("validation_target_scores", target_scores)
        trial.set_user_attr("fold_scores", [score])
        trial.set_user_attr("fold_target_scores", [target_scores])
        trial.set_user_attr("fold_best_iterations", [best_iteration])
        return score

    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    summary_path, trials_path = export_study(
        study=study,
        output_dir=tune_dir,
        model_type="catboost",
        dataset_path=request.base_dataset_path,
        dataset_hash=dataset_hash,
        signature=signature,
        protocol=protocol,
        search_params=config.search_params,
        fixed_hyper_params=config.fixed_hyper_params,
        initial_hyper_params=config.initial_hyper_params,
        config_snapshot=_config_snapshot(request.config_path, config),
    )
    write_tuning_metadata(tune_dir, _config_snapshot(request.config_path, config), request.project_root)
    write_last_tune_id(tuning_root, tune_id)
    return OptunaSummaryDto(
        tune_id=tune_id,
        tune_dir=tune_dir,
        summary_path=summary_path,
        trials_path=trials_path,
        study_database_path=database_path,
    )


def _config_snapshot(config_path: Path, config: CatBoostPipelineConfig) -> dict[str, object]:
    return {"config_path": config_path, "config": config}
