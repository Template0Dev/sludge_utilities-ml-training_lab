from __future__ import annotations

import datetime as dt
import gc
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# PyTorch 2.12 warns from Lightning's internal compatibility helper. Suppress
# only this dependency warning until Lightning removes its legacy LeafSpec use.
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated,.*",
    category=FutureWarning,
    module=r"pytorch_lightning\.utilities\._pytree",
)

import cv2
import httpx
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch.utils.data import DataLoader, Dataset
from uuid6 import uuid6

from .optuna_support import (
    assert_tuning_protocol,
    config_signature,
    export_study,
    file_sha256,
    load_summary,
    originals_for_well,
    prepare_study,
)
from .pipeline_config import load_pipeline_config, optional_wells, resolve_project_path, resolve_training_wells


TARGET_COLUMNS = (
    "sandstone_sludge", "siltstone_sludge", "argillite_sludge", "radiolarite_sludge",
    "coal_sludge", "limestone_sludge", "clay_sludge", "other_sludge",
)
SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [1, 2, 4]},
    "head_hidden_size": {"type": "categorical", "choices": [256, 512, 1024]},
    "dropout": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1},
    "huber_delta": {"type": "categorical", "choices": [0.5, 1.0, 2.0, 5.0]},
}
BASELINE = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 1,
    "head_hidden_size": 512,
    "dropout": 0.3,
    "huber_delta": 1.0,
}
FIXED_PARAMETERS = {
    "model_name": "resnet50d", "pretrained": True, "optimizer": "AdamW",
    "input_size": 512, "max_epochs": 15, "early_stop_patience": 6,
    "lr_factor": 0.5, "lr_patience": 3, "random_seed": 42,
}
RESNET_MODEL_NAME = "resnet50d"
RESNET_HF_REPOSITORY = "timm/resnet50d.ra2_in1k"
RESNET_HF_WEIGHTS = "model.safetensors"


@dataclass(frozen=True)
class ResNetTuningConfig:
    project_root: Path
    study_name: str = "resnet_target_well_v1"
    total_trials: int = 20
    smoke_mode: bool = False
    include_augmented_records: bool = True
    training_wells: tuple[int, ...] | None = None
    target_well: int = 1
    num_workers: int = 13

    @classmethod
    def from_file(cls, project_root: Path, path: Path) -> "ResNetTuningConfig":
        config = load_pipeline_config(path)
        optuna_config = config.get("optuna", {})
        data_loader_config = config.get("data_loader", {})
        return cls(
            project_root=project_root,
            study_name=optuna_config.get("study_name", cls.study_name),
            total_trials=int(optuna_config.get("total_trials", cls.total_trials)),
            smoke_mode=bool(optuna_config.get("smoke_mode", cls.smoke_mode)),
            include_augmented_records=bool(config.get("include_augmented_records", cls.include_augmented_records)),
            training_wells=optional_wells(config.get("training_wells")),
            target_well=int(config.get("target_well", cls.target_well)),
            num_workers=int(data_loader_config.get("num_workers", cls.num_workers)),
        )

    @property
    def dataset_path(self) -> Path:
        return self.project_root / "data/meta/interim/metadata_augmented.parquet"

    @property
    def image_root(self) -> Path:
        return self.project_root / "data/images"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output/resnet/optuna"


@dataclass(frozen=True)
class ResNetFinalConfig:
    project_root: Path
    optuna_summary_path: Path
    include_augmented_records: bool = True
    target_well: int = 1
    training_wells: tuple[int, ...] | None = None
    num_workers: int = 13
    save_predictions: bool = True

    @classmethod
    def from_file(cls, project_root: Path, path: Path) -> "ResNetFinalConfig":
        config = load_pipeline_config(path)
        final_config = config.get("final_training", {})
        data_loader_config = config.get("data_loader", {})
        optuna_summary_path = final_config.get(
            "optuna_summary_path",
            "output/resnet/optuna/resnet_target_well_v1_summary.json",
        )
        return cls(
            project_root=project_root,
            optuna_summary_path=resolve_project_path(project_root, optuna_summary_path),
            include_augmented_records=bool(config.get("include_augmented_records", cls.include_augmented_records)),
            target_well=int(config.get("target_well", cls.target_well)),
            training_wells=optional_wells(config.get("training_wells")),
            num_workers=int(data_loader_config.get("num_workers", cls.num_workers)),
            save_predictions=bool(final_config.get("save_predictions", cls.save_predictions)),
        )


def seed_everything() -> None:
    random.seed(42)
    np.random.seed(42)
    pl.seed_everything(42, workers=True)


class SludgeImageDataset(Dataset):
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]

    def __init__(self, df: pd.DataFrame, image_root: Path):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image_path = self.image_root / row["sludge_image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image does not exist or cannot be read: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
        image = (image - self.mean) / self.std
        targets = row[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(targets)


def training_records(df: pd.DataFrame, wells: tuple[int, ...], include_augmented_records: bool) -> pd.DataFrame:
    result = df[df["well_id"].isin(wells)]
    if not include_augmented_records:
        result = result[~result["is_augmented"].astype(bool)]
    result = result.copy()
    if result.empty:
        raise ValueError("No training samples remain after applying the augmented-record filter.")
    return result


def create_resnet_backbone() -> nn.Module:
    try:
        return timm.create_model(RESNET_MODEL_NAME, pretrained=True, num_classes=0)
    except httpx.HTTPError as network_error:
        try:
            cached_weights = hf_hub_download(
                repo_id=RESNET_HF_REPOSITORY,
                filename=RESNET_HF_WEIGHTS,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            raise RuntimeError(
                f"Could not download {RESNET_HF_REPOSITORY} and no locally cached "
                f"{RESNET_HF_WEIGHTS} file is available."
            ) from network_error
        return timm.create_model(
            RESNET_MODEL_NAME,
            pretrained=True,
            num_classes=0,
            pretrained_cfg_overlay={"file": cached_weights},
        )


class SludgeResNet(pl.LightningModule):
    def __init__(self, params: dict[str, Any], *, scheduler_enabled: bool = True):
        super().__init__()
        self.save_hyperparameters({"params": params, "scheduler_enabled": scheduler_enabled})
        self.params = params
        self.scheduler_enabled = scheduler_enabled
        self.backbone = create_resnet_backbone()
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, params["head_hidden_size"]),
            nn.LayerNorm(params["head_hidden_size"]), nn.ReLU(),
            nn.Dropout(params["dropout"]), nn.Linear(params["head_hidden_size"], len(TARGET_COLUMNS)),
        )
        self.criterion = nn.HuberLoss(delta=params["huber_delta"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(self.backbone(images)), dim=1) * 100.0

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_index: int) -> torch.Tensor:
        images, targets = batch
        loss = self.criterion(self(images), targets)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_index: int) -> None:
        images, targets = batch
        predictions = self(images)
        self.log("val_loss", self.criterion(predictions, targets), on_epoch=True)
        self.log("val_mae", torch.mean(torch.abs(predictions - targets)), on_epoch=True, prog_bar=True)
        for index, target in enumerate(TARGET_COLUMNS):
            self.log(f"val_mae_{target}", torch.mean(torch.abs(predictions[:, index] - targets[:, index])), on_epoch=True)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_index: int) -> torch.Tensor:
        images, _targets = batch
        return self(images).detach().cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.params["learning_rate"], weight_decay=self.params["weight_decay"]
        )
        if not self.scheduler_enabled:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mae"}}


class BestMetric(Callback):
    def __init__(self) -> None:
        self.best_value = float("inf")
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, _module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val_mae")
        if value is not None and float(value) < self.best_value:
            self.best_value = float(value)
            self.best_epoch = trainer.current_epoch + 1


class TrialPruning(Callback):
    def __init__(self, trial: optuna.Trial, fold_index: int, max_epochs: int) -> None:
        self.trial = trial
        self.fold_index = fold_index
        self.max_epochs = max_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, _module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val_mae")
        epoch = trainer.current_epoch
        if value is None:
            return
        self.trial.report(float(value), self.fold_index * self.max_epochs + epoch)
        if epoch >= 3 and self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned during fold {self.fold_index + 1}, epoch {epoch + 1}")


def accelerator() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def loader(df: pd.DataFrame, config: ResNetTuningConfig | ResNetFinalConfig, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SludgeImageDataset(df, config.project_root / "data/images"), batch_size=batch_size,
        shuffle=shuffle, num_workers=config.num_workers, persistent_workers=config.num_workers > 0,
        drop_last=shuffle and len(df) >= batch_size,
    )


def suggested_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [1, 2, 4]),
        "head_hidden_size": trial.suggest_categorical("head_hidden_size", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "huber_delta": trial.suggest_categorical("huber_delta", [0.5, 1.0, 2.0, 5.0]),
    }


def cleanup_model() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def tune_resnet(config: ResNetTuningConfig) -> Path:
    seed_everything()
    df = pd.read_parquet(config.dataset_path)
    training_wells = resolve_training_wells(
        df, target_well=config.target_well, configured_wells=config.training_wells
    )
    assert_tuning_protocol(df, training_wells, config.target_well)
    max_epochs = 1 if config.smoke_mode else FIXED_PARAMETERS["max_epochs"]
    study_name = f"{config.study_name}_smoke" if config.smoke_mode else config.study_name
    dataset_hash = file_sha256(config.dataset_path)
    protocol = {
        "training_wells": list(training_wells), "target_well": config.target_well,
        "include_augmented_records": config.include_augmented_records,
        "validation_strategy": "target_well", "validation_originals_only": True,
        "input_size": 512, "normalization": "ImageNet",
        "max_epochs_per_fold": max_epochs,
    }
    signature = config_signature({
        "dataset_sha256": dataset_hash, "protocol": protocol, "search_space": SEARCH_SPACE,
        "fixed_parameters": FIXED_PARAMETERS, "target_columns": TARGET_COLUMNS,
    })
    study, remaining = prepare_study(
        study_name=study_name, database_path=config.output_dir / f"{study_name}.db", signature=signature,
        total_trials=1 if config.smoke_mode else config.total_trials, baseline=BASELINE,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggested_params(trial)
        train_df = training_records(df, training_wells, config.include_augmented_records)
        validation_df = originals_for_well(df, config.target_well)
        if config.target_well in train_df["well_id"].unique():
            raise AssertionError("Target leakage detected.")
        metric = BestMetric()
        model = SludgeResNet(params)
        trainer = pl.Trainer(
            max_epochs=max_epochs, accelerator=accelerator(), devices=1, logger=False,
            enable_checkpointing=False, deterministic=False, enable_progress_bar=not config.smoke_mode,
            callbacks=[metric, TrialPruning(trial, 0, max_epochs), EarlyStopping("val_mae", mode="min", patience=6)],
        )
        try:
            trainer.fit(
                model,
                train_dataloaders=loader(train_df, config, params["batch_size"], True),
                val_dataloaders=loader(validation_df, config, params["batch_size"], False),
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
    return export_study(
        study=study, output_dir=config.output_dir, model_type="resnet", dataset_path=config.dataset_path,
        dataset_hash=dataset_hash, signature=signature, protocol=protocol, search_space=SEARCH_SPACE,
        fixed_parameters=FIXED_PARAMETERS,
    )


def train_final_resnet(config: ResNetFinalConfig) -> tuple[Path, dict[str, Any]]:
    seed_everything()
    dataset_path = config.project_root / "data/meta/interim/metadata_augmented.parquet"
    summary = load_summary(config.optuna_summary_path, model_type="resnet", dataset_path=dataset_path)
    protocol = summary.get("protocol", {})
    df = pd.read_parquet(dataset_path)
    training_wells = resolve_training_wells(
        df, target_well=config.target_well, configured_wells=config.training_wells
    )
    summary_training_wells = protocol.get("training_wells", protocol.get("tuning_wells"))
    summary_target_well = protocol.get("target_well", protocol.get("holdout_well"))
    if summary_training_wells != list(training_wells) or summary_target_well != config.target_well:
        raise ValueError("The Optuna summary uses a different training/holdout well protocol.")
    if protocol.get("include_augmented_records", True) != config.include_augmented_records:
        raise ValueError("The Optuna summary was produced with a different augmented-record setting.")
    if protocol.get("input_size") != 512 or protocol.get("normalization") != "ImageNet":
        raise ValueError("The Optuna summary uses incompatible ResNet preprocessing.")
    assert_tuning_protocol(df, training_wells, config.target_well)
    params = summary["best_trial"]["params"]
    best_epochs = summary["best_trial"]["user_attrs"].get("fold_best_epochs")
    if not best_epochs or any(epoch < 1 for epoch in best_epochs):
        raise ValueError("The winning trial does not contain valid fold best epochs.")
    final_epochs = max(1, int(np.median(best_epochs)))
    train_df = training_records(df, training_wells, config.include_augmented_records)
    holdout_df = originals_for_well(df, config.target_well)
    model = SludgeResNet(params, scheduler_enabled=False)
    trainer = pl.Trainer(max_epochs=final_epochs, accelerator=accelerator(), devices=1, logger=False, enable_checkpointing=False)
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    trainer.fit(model, train_dataloaders=loader(train_df, config, params["batch_size"], True))
    holdout_loader = loader(holdout_df, config, params["batch_size"], False)
    metrics = trainer.validate(model, dataloaders=holdout_loader, verbose=False)[0]
    predictions = None
    if config.save_predictions:
        prediction_batches = trainer.predict(model, dataloaders=holdout_loader)
        predictions = torch.cat(prediction_batches).numpy()
    training_id = str(uuid6())
    model_dir = config.project_root / "output/resnet/models"
    meta_dir = config.project_root / "output/resnet/meta"
    predictions_dir = config.project_root / "output/resnet/predictions"
    for directory in (model_dir, meta_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"[{training_id}] ResNet.ckpt"
    trainer.save_checkpoint(model_path)
    metadata = {
        "training_id": training_id, "model_type": "ResNet", "model_file": model_path.name,
        "training_started_at": started_at, "training_completed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "hyperparameters": params, "final_epochs": final_epochs, "training_wells": list(training_wells),
        "target_well": config.target_well, "final_training_metrics": metrics,
        "optuna": {"study_name": summary["study_name"], "trial_number": summary["best_trial"]["number"],
                   "objective_value": summary["best_trial"]["value"], "summary_path": str(config.optuna_summary_path)},
    }
    meta_path = meta_dir / f"[{training_id}] ResNet.txt"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if predictions is not None:
        prediction_frame = holdout_df[list(TARGET_COLUMNS)].reset_index(drop=True).add_suffix("_true")
        for index, target in enumerate(TARGET_COLUMNS):
            prediction_frame[f"{target}_pred"] = predictions[:, index]
        prediction_frame.to_csv(predictions_dir / f"[{training_id}] ResNet (Result).csv", index=False)
    return model_path, metadata
