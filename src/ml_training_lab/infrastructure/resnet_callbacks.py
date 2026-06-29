from __future__ import annotations

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


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
