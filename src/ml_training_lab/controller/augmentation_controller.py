from __future__ import annotations

from ml_training_lab.application.run_augmentation import run_augmentation as run_augmentation_use_case
from ml_training_lab.presentation.requests import AugmentationRequest
from ml_training_lab.presentation.responses import AugmentationResultDto


def run_augmentation(request: AugmentationRequest) -> AugmentationResultDto:
    return run_augmentation_use_case(request)
