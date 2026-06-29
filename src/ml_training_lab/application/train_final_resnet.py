from __future__ import annotations

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from ml_training_lab.domain.record_selection import originals_for_well, training_records
from ml_training_lab.domain.training_wells import resolve_training_wells
from ml_training_lab.domain.tuning_protocol import assert_tuning_protocol
from ml_training_lab.domain.tuning_study_name import effective_study_name
from ml_training_lab.infrastructure.optuna_summary_reader import load_summary
from ml_training_lab.infrastructure.output_path_builder import model_runs_root, model_tuning_root
from ml_training_lab.infrastructure.pipeline_config_reader import read_pipeline_config
from ml_training_lab.infrastructure.resnet_loader import loader
from ml_training_lab.infrastructure.resnet_module import SludgeResNet
from ml_training_lab.infrastructure.run_artifact_writer import create_run_dir, write_run_metadata
from ml_training_lab.infrastructure.tuning_summary_path import latest_tuning_summary_path
from ml_training_lab.presentation.model_configs import ResNetPipelineConfig
from ml_training_lab.presentation.requests import ResNetRequest
from ml_training_lab.presentation.responses import TrainingResultDto
from ml_training_lab.shared.clock import now_iso
from ml_training_lab.shared.device_selector import accelerator
from ml_training_lab.shared.seed import seed_everything
from ml_training_lab.shared.uuid_generator import new_uuid


def train_final_resnet(request: ResNetRequest) -> TrainingResultDto:
    seed_everything()
    config = read_pipeline_config(request.config_path, ResNetPipelineConfig)
    summary_path = _summary_path(request.project_root, config)
    summary = load_summary(summary_path, model_type="resnet", dataset_path=request.dataset_path)
    df = pd.read_parquet(request.dataset_path)
    training_wells = resolve_training_wells(
        df, target_well=config.target_well, configured_wells=config.training_wells
    )
    _assert_summary_protocol(summary, training_wells, config)
    assert_tuning_protocol(df, training_wells, config.target_well)
    params = summary["best_trial"]["params"]
    best_epochs = summary["best_trial"]["user_attrs"].get("fold_best_epochs")
    if not best_epochs or any(epoch < 1 for epoch in best_epochs):
        raise ValueError("The winning trial does not contain valid fold best epochs.")
    final_epochs = max(1, int(np.median(best_epochs)))
    train_df = training_records(df, training_wells, config.include_augmented_records)
    holdout_df = originals_for_well(df, config.target_well)
    input_size = int(config.fixed_hyper_params["input_size"])
    model = SludgeResNet(params, config.target_columns, scheduler_enabled=False)
    trainer = pl.Trainer(
        max_epochs=final_epochs,
        accelerator=accelerator(),
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    started_at = now_iso()
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
    )
    holdout_loader = loader(
        holdout_df,
        image_root=request.image_root,
        target_columns=config.target_columns,
        input_size=input_size,
        batch_size=params["batch_size"],
        num_workers=config.data_loader.num_workers,
        shuffle=False,
    )
    metrics = trainer.validate(model, dataloaders=holdout_loader, verbose=False)[0]
    predictions = None
    if config.final_training.save_predictions:
        prediction_batches = trainer.predict(model, dataloaders=holdout_loader)
        predictions = torch.cat(prediction_batches).numpy()
    run_id = new_uuid()
    run_dir = create_run_dir(
        model_runs_root(request.project_root, config.output, config.output.output_resnet_sub_folder),
        run_id,
    )
    model_path = run_dir / "model.ckpt"
    trainer.save_checkpoint(model_path)
    predictions_path = None
    if predictions is not None:
        predictions_path = run_dir / "predictions.csv"
        prediction_frame = holdout_df[list(config.target_columns)].reset_index(drop=True).add_suffix("_true")
        for index, target in enumerate(config.target_columns):
            prediction_frame[f"{target}_pred"] = predictions[:, index]
        prediction_frame.to_csv(predictions_path, index=False)
    request_meta = {
        "training_id": run_id,
        "model_type": "ResNet",
        "model_file": model_path.name,
        "predictions_file": predictions_path.name if predictions_path else None,
        "target_columns": config.target_columns,
        "training_started_at": started_at,
        "training_completed_at": now_iso(),
        "hyperparameters": params,
        "final_epochs": final_epochs,
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
    write_run_metadata(run_dir, request_meta, {"request": request, "config": config}, request.project_root)
    return TrainingResultDto(
        run_id=run_id,
        run_dir=run_dir,
        model_path=model_path,
        request_meta_path=run_dir / "request_meta.json",
        config_snapshot_path=run_dir / "config_snapshot.json",
        version_info_path=run_dir / "version_info.txt",
        predictions_path=predictions_path,
        embeddings_path=None,
        metrics=metrics,
    )


def _assert_summary_protocol(
    summary: dict[str, object],
    training_wells: tuple[int, ...],
    config: ResNetPipelineConfig,
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
    if protocol.get("input_size") != int(config.fixed_hyper_params["input_size"]) or protocol.get("normalization") != "ImageNet":
        raise ValueError("The Optuna summary uses incompatible ResNet preprocessing.")


def _summary_path(project_root, config: ResNetPipelineConfig):
    tuning_root = model_tuning_root(project_root, config.output, config.output.output_resnet_sub_folder)
    return latest_tuning_summary_path(tuning_root, effective_study_name(config.tuning_params))
