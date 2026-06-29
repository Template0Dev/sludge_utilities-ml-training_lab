from __future__ import annotations

from ml_training_lab.application.tune_resnet import tune_resnet as tune_resnet_use_case
from ml_training_lab.presentation.requests import ResNetRequest
from ml_training_lab.presentation.responses import OptunaSummaryDto


def tune_resnet(request: ResNetRequest) -> OptunaSummaryDto:
    return tune_resnet_use_case(request)
