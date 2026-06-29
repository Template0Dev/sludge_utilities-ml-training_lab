from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ml_training_lab.domain.feature_columns import enabled_embedding_columns
from ml_training_lab.presentation.common_config import FeatureConfig


def fit_fold_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    features: FeatureConfig,
    components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_parts = [train_df[list(features.base_columns)].reset_index(drop=True)]
    validation_parts = [validation_df[list(features.base_columns)].reset_index(drop=True)]
    transformers: dict[str, Any] = {}
    for embedding_column in enabled_embedding_columns(features):
        pca = PCA(n_components=components, random_state=42)
        train_embedding = pca.fit_transform(np.stack(train_df[embedding_column].to_numpy()))
        validation_embedding = pca.transform(np.stack(validation_df[embedding_column].to_numpy()))
        prefix = embedding_column.replace("_dinov3_emb", "").replace("_emb", "")
        columns = [f"{prefix}_emb_pca_{index}" for index in range(components)]
        train_parts.append(pd.DataFrame(train_embedding, columns=columns))
        validation_parts.append(pd.DataFrame(validation_embedding, columns=columns))
        transformers[embedding_column] = pca
    return pd.concat(train_parts, axis=1), pd.concat(validation_parts, axis=1), transformers
