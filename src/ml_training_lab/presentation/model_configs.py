from __future__ import annotations

from pydantic import Field

from ml_training_lab.presentation.common_config import DataLoaderConfig, FeatureConfig, PipelineConfig


class CatBoostPipelineConfig(PipelineConfig):
    features: FeatureConfig = Field(default_factory=FeatureConfig)


class ResNetPipelineConfig(PipelineConfig):
    data_loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
