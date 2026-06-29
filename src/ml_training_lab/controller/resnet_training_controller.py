from __future__ import annotations

from ml_training_lab.application.train_final_resnet import train_final_resnet as train_final_resnet_use_case
from ml_training_lab.presentation.requests import ResNetRequest
from ml_training_lab.presentation.responses import TrainingResultDto


def train_final_resnet(request: ResNetRequest) -> TrainingResultDto:
    return train_final_resnet_use_case(request)
