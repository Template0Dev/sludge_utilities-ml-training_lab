from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_training_lab.domain.feature_columns import enabled_embedding_columns
from ml_training_lab.infrastructure.dataframe_storage import read_parquet
from ml_training_lab.infrastructure.embedding_value_parser import parse_embedding
from ml_training_lab.presentation.common_config import FeatureConfig


def load_feature_dataset(
    *,
    base_dataset_path: Path,
    embedding_dataset_path: Path | None,
    features: FeatureConfig,
) -> pd.DataFrame:
    base_df = read_parquet(base_dataset_path).copy()
    embedding_columns = enabled_embedding_columns(features)
    if not embedding_columns:
        return base_df
    if embedding_dataset_path is None:
        raise ValueError("embedding_dataset_path is required when embedding features are enabled.")
    embedding_df = read_parquet(embedding_dataset_path)
    selected_columns = list(features.embedding_join_keys) + list(embedding_columns)
    missing_columns = sorted(set(selected_columns) - set(embedding_df.columns))
    if missing_columns:
        raise ValueError(f"Embedding dataset is missing columns: {missing_columns}")
    duplicate_keys = embedding_df.duplicated(list(features.embedding_join_keys), keep=False)
    if duplicate_keys.any():
        raise ValueError("Embedding dataset contains duplicate rows for configured embedding_join_keys.")
    result = base_df.merge(
        embedding_df[selected_columns],
        on=list(features.embedding_join_keys),
        how="left",
        validate="one_to_one",
    )
    missing_embeddings = [column for column in embedding_columns if result[column].isna().any()]
    if missing_embeddings:
        raise ValueError(f"Enabled embedding columns contain missing values after join: {missing_embeddings}")
    for column in embedding_columns:
        result[column] = result[column].map(parse_embedding)
    return result
