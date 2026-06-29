from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from ml_training_lab.infrastructure.resnet_dataset import SludgeImageDataset


def loader(
    df: pd.DataFrame,
    *,
    image_root: Path,
    target_columns: tuple[str, ...],
    input_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        SludgeImageDataset(df, image_root, target_columns, input_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=shuffle and len(df) >= batch_size,
    )
