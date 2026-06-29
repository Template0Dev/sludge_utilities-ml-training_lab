from __future__ import annotations

import numpy as np


def normalize_prediction_rows(predictions: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(predictions), 0.0, None)
    totals = clipped.sum(axis=1, keepdims=True)
    return clipped / np.where(totals == 0.0, 1.0, totals) * 100.0
