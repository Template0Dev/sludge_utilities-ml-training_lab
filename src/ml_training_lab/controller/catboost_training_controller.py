from __future__ import annotations

from ml_training_lab.application.train_final_catboost import train_final_catboost as train_final_catboost_use_case
from ml_training_lab.presentation.requests import CatBoostRequest
from ml_training_lab.presentation.responses import TrainingResultDto


def train_final_catboost(request: CatBoostRequest) -> TrainingResultDto:
    return train_final_catboost_use_case(request)
