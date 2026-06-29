from __future__ import annotations

from ml_training_lab.application.tune_catboost import tune_catboost as tune_catboost_use_case
from ml_training_lab.presentation.requests import CatBoostRequest
from ml_training_lab.presentation.responses import OptunaSummaryDto


def tune_catboost(request: CatBoostRequest) -> OptunaSummaryDto:
    return tune_catboost_use_case(request)
