from __future__ import annotations

import random

import numpy as np
import pytorch_lightning as pl


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)
