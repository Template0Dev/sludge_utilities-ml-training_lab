from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppOutputConfig(BaseModel):
    output_base_folder: str = "output"
    output_gb_sub_folder: str = "gb"
    output_resnet_sub_folder: str = "resnet"
    output_runs_prefix_for_saving: str = "runs"
    output_tuning_prefix_for_saving: str = "tuning"


class FeatureConfig(BaseModel):
    base_columns: tuple[str, ...] = ("interval_start", "interval_end")
    should_use_sludge_embeddings: bool = True
    should_use_lba_embeddings: bool = False
    sludge_embedding_column: str = "sludge_dinov3_emb"
    lba_embedding_column: str = "lba_dinov3_emb"
    embedding_join_keys: tuple[str, ...] = (
        "well_id",
        "interval_start",
        "interval_end",
        "sludge_image_path",
        "lba_image_path",
        "is_augmented",
    )

    @field_validator("base_columns", "embedding_join_keys")
    @classmethod
    def require_non_empty_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("column list cannot be empty")
        return value


class FinalTrainingConfig(BaseModel):
    optuna_summary_path: str | Path
    save_predictions: bool = True

    @field_validator("optuna_summary_path")
    @classmethod
    def require_summary_file_name(cls, value: str | Path) -> str | Path:
        path = Path(value)
        if path.is_absolute() or path.parent != Path("."):
            raise ValueError("optuna_summary_path must be a file name; output folders come from output config.")
        return value


class TuningParams(BaseModel):
    study_name: str
    total_trials: int
    smoke_mode: bool = False
    pruner_n_startup_trials: int = 5
    pruner_n_warmup_steps: int = 0


class DataLoaderConfig(BaseModel):
    num_workers: int = 13


class PipelineConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_well: int = 1
    training_wells: tuple[int, ...] | None = None
    include_augmented_records: bool = True
    target_columns: tuple[str, ...]
    output: AppOutputConfig = Field(default_factory=AppOutputConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    tuning_params: TuningParams
    search_params: dict[str, dict[str, Any]]
    initial_hyper_params: dict[str, Any]
    fixed_hyper_params: dict[str, Any]
    final_training: FinalTrainingConfig

    @field_validator("training_wells")
    @classmethod
    def validate_training_wells(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is not None and not value:
            raise ValueError("training_wells cannot be empty")
        return value

    @field_validator("target_columns")
    @classmethod
    def validate_target_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("target_columns cannot be empty")
        return value
