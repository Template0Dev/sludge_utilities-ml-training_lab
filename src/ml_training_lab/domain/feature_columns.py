from __future__ import annotations

from ml_training_lab.presentation.common_config import FeatureConfig


def enabled_embedding_columns(features: FeatureConfig) -> tuple[str, ...]:
    columns: list[str] = []
    if features.should_use_sludge_embeddings:
        columns.append(features.sludge_embedding_column)
    if features.should_use_lba_embeddings:
        columns.append(features.lba_embedding_column)
    return tuple(columns)
