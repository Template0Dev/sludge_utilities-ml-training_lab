from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SludgeImageDataset(Dataset):
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]

    def __init__(self, df: pd.DataFrame, image_root: Path, target_columns: tuple[str, ...], input_size: int):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.target_columns = target_columns
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image_path = self.image_root / row["sludge_image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image does not exist or cannot be read: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        image = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
        image = (image - self.mean) / self.std
        targets = row[list(self.target_columns)].to_numpy(dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(targets)
