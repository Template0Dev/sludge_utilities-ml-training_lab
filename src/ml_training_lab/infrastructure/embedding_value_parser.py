from __future__ import annotations

import ast
from typing import Any

import numpy as np


def parse_embedding(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.float32)
