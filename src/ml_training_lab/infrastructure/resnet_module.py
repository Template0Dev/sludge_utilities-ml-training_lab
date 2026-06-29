from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ml_training_lab.infrastructure.resnet_backbone import create_resnet_backbone


class SludgeResNet(pl.LightningModule):
    def __init__(self, params: dict[str, Any], target_columns: tuple[str, ...], *, scheduler_enabled: bool = True):
        super().__init__()
        self.save_hyperparameters({"params": params, "scheduler_enabled": scheduler_enabled})
        self.params = params
        self.target_columns = target_columns
        self.scheduler_enabled = scheduler_enabled
        self.backbone = create_resnet_backbone()
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, params["head_hidden_size"]),
            nn.LayerNorm(params["head_hidden_size"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["head_hidden_size"], len(target_columns)),
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
        for index, target in enumerate(self.target_columns):
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
