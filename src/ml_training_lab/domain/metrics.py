from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def macro_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    target_scores = np.mean(np.abs(y_true - y_pred), axis=0)
    return float(np.mean(target_scores)), target_scores.astype(float).tolist()


def detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    return {
        target: {
            "mae": float(mean_absolute_error(y_true[:, index], y_pred[:, index])),
            "rmse": float(root_mean_squared_error(y_true[:, index], y_pred[:, index])),
            "r2": float(r2_score(y_true[:, index], y_pred[:, index])),
        }
        for index, target in enumerate(target_columns)
    }
