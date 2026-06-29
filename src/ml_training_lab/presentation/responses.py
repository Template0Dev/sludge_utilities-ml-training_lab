from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class OptunaSummaryDto(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary_path: Path
    trials_path: Path
    study_database_path: Path


class TrainingResultDto(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    run_dir: Path
    model_path: Path
    request_meta_path: Path
    config_snapshot_path: Path
    version_info_path: Path
    predictions_path: Path | None = None
    embeddings_path: Path | None = None
    metrics: dict[str, Any]


class AugmentationResultDto(BaseModel):
    output_csv_path: Path
    output_parquet_path: Path
    generated_record_count: int
    total_record_count: int


class EmbeddingExtractionResultDto(BaseModel):
    output_csv_path: Path
    output_parquet_path: Path
    row_count: int
