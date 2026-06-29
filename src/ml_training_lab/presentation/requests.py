from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ConfiguredRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_root: Path
    config_path: Path


class CatBoostRequest(ConfiguredRequest):
    base_dataset_path: Path
    embedding_dataset_path: Path | None = None


class ResNetRequest(ConfiguredRequest):
    dataset_path: Path
    image_root: Path


class AugmentationRequest(BaseModel):
    project_root: Path
    input_path: Path
    output_csv_path: Path
    output_parquet_path: Path
    image_root: Path
    sludge_image_column: str = "sludge_image_path"
    lba_image_column: str = "lba_image_path"
    is_augmented_column: str = "is_augmented"
    sludge_augmented_folder: str = "images-sludge-augmented"
    lba_augmented_folder: str = "images-lba-augmented"
    augmentations_per_record: int = 5
    random_seed: int = 42


class EmbeddingExtractionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_root: Path
    input_path: Path
    output_csv_path: Path
    output_parquet_path: Path
    image_root: Path
    model_name: str
    lba_image_column: str = "lba_image_path"
    sludge_image_column: str = "sludge_image_path"
    lba_embedding_column: str = "lba_dinov3_emb"
    sludge_embedding_column: str = "sludge_dinov3_emb"
    batch_size: int = 8
