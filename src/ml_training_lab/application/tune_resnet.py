from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from ml_training_lab.domain.config_signature import config_signature
from ml_training_lab.domain.record_selection import originals_for_well, training_records
from ml_training_lab.domain.search_params import suggest_params
from ml_training_lab.domain.training_wells import resolve_training_wells
from ml_training_lab.domain.tuning_protocol import assert_tuning_protocol
from ml_training_lab.infrastructure.optuna_study_exporter import export_study
from ml_training_lab.infrastructure.optuna_study_repository import prepare_study
from ml_training_lab.infrastructure.output_path_builder import model_tuning_root
from ml_training_lab.infrastructure.pipeline_config_reader import read_pipeline_config
from ml_training_lab.infrastructure.resnet_callbacks import BestMetric, TrialPruning
from ml_training_lab.infrastructure.resnet_loader import loader
from ml_training_lab.infrastructure.resnet_module import SludgeResNet
from ml_training_lab.infrastructure.torch_cleanup import cleanup_model
from ml_training_lab.presentation.model_configs import ResNetPipelineConfig
from ml_training_lab.presentation.requests import ResNetRequest
from ml_training_lab.presentation.responses import OptunaSummaryDto
from ml_training_lab.shared.device_selector import accelerator
from ml_training_lab.shared.file_hash import file_sha256
from ml_training_lab.shared.seed import seed_everything


def tune_resnet(request: ResNetRequest) -> OptunaSummaryDto:
    seed_everything()
    config = read_pipeline_config(request.config_path, ResNetPipelineConfig)
    df = pd.read_parquet(request.dataset_path)
    training_wells = resolve_training_wells(
        df, target_well=config.target_well, configured_wells=config.training_wells
    )
    assert_tuning_protocol(df, training_wells, config.target_well)
    max_epochs = 1 if config.tuning_params.smoke_mode else int(config.fixed_hyper_params["max_epochs"])
    input_size = int(config.fixed_hyper_params["input_size"])
    study_name = (
        f"{config.tuning_params.study_name}_smoke"
        if config.tuning_params.smoke_mode
        else config.tuning_params.study_name
    )
    dataset_hash = file_sha256(request.dataset_path)
    protocol = {
        "training_wells": list(training_wells),
        "target_well": config.target_well,
        "include_augmented_records": config.include_augmented_records,
        "validation_strategy": "target_well",
        "validation_originals_only": True,
        "input_size": input_size,
        "normalization": "ImageNet",
        "max_epochs_per_fold": max_epochs,
        "target_columns": config.target_columns,
    }
    signature = config_signature({
        "dataset_sha256": dataset_hash,
        "protocol": protocol,
        "search_params": config.search_params,
        "fixed_hyper_params": config.fixed_hyper_params,
        "initial_hyper_params": config.initial_hyper_params,
    })
    tuning_root = model_tuning_root(request.project_root, config.output, config.output.output_resnet_sub_folder)
    database_path = tuning_root / f"{study_name}.db"
    study, remaining = prepare_study(
        study_name=study_name,
        database_path=database_path,
        signature=signature,
        total_trials=1 if config.tuning_params.smoke_mode else config.tuning_params.total_trials,
        baseline=config.initial_hyper_params,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.tuning_params.pruner_n_startup_trials,
            n_warmup_steps=config.tuning_params.pruner_n_warmup_steps,
        ),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, config.search_params)
        train_df = training_records(df, training_wells, config.include_augmented_records)
        validation_df = originals_for_well(df, config.target_well)
        if config.target_well in train_df["well_id"].unique():
            raise AssertionError("Target leakage detected.")
        metric = BestMetric()
        model = SludgeResNet(params, config.target_columns)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator(),
            devices=1,
            logger=False,
            enable_checkpointing=False,
            deterministic=False,
            enable_progress_bar=not config.tuning_params.smoke_mode,
            callbacks=[
                metric,
                TrialPruning(trial, 0, max_epochs),
                EarlyStopping("val_mae", mode="min", patience=int(config.fixed_hyper_params["early_stop_patience"])),
            ],
        )
        try:
            trainer.fit(
                model,
                train_dataloaders=loader(
                    train_df,
                    image_root=request.image_root,
                    target_columns=config.target_columns,
                    input_size=input_size,
                    batch_size=params["batch_size"],
                    num_workers=config.data_loader.num_workers,
                    shuffle=True,
                ),
                val_dataloaders=loader(
                    validation_df,
                    image_root=request.image_root,
                    target_columns=config.target_columns,
                    input_size=input_size,
                    batch_size=params["batch_size"],
                    num_workers=config.data_loader.num_workers,
                    shuffle=False,
                ),
            )
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                trial.set_user_attr("failure", "out_of_memory")
                raise optuna.TrialPruned("Device out of memory") from error
            raise
        finally:
            del model, trainer
            cleanup_model()
        trial.set_user_attr("validation_well", config.target_well)
        trial.set_user_attr("validation_score", metric.best_value)
        trial.set_user_attr("fold_scores", [metric.best_value])
        trial.set_user_attr("fold_best_epochs", [metric.best_epoch])
        return metric.best_value

    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    summary_path, trials_path = export_study(
        study=study,
        output_dir=tuning_root,
        model_type="resnet",
        dataset_path=request.dataset_path,
        dataset_hash=dataset_hash,
        signature=signature,
        protocol=protocol,
        search_params=config.search_params,
        fixed_hyper_params=config.fixed_hyper_params,
        initial_hyper_params=config.initial_hyper_params,
        config_snapshot={"request": request, "config": config},
    )
    return OptunaSummaryDto(summary_path=summary_path, trials_path=trials_path, study_database_path=database_path)
