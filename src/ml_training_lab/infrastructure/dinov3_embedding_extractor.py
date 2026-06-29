from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel


class ImagePathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Path, image_column: str, processor):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.image_column = image_column
        self.processor = processor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(self.image_root / row[self.image_column]).convert("RGB")
        return self.processor(images=image, return_tensors="pt")


def extract_embeddings(
    *,
    df: pd.DataFrame,
    image_root: Path,
    image_column: str,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    dataset = ImagePathDataset(df, image_root, image_column, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader):
            pixel_values = batch["pixel_values"].squeeze(1).to(device)
            outputs = model(pixel_values)
            embeddings.append(outputs.last_hidden_state[:, 0].detach().cpu().numpy())
    return np.vstack(embeddings)
