from __future__ import annotations

import pandas as pd


def training_records(df: pd.DataFrame, wells: tuple[int, ...], include_augmented_records: bool) -> pd.DataFrame:
    result = df[df["well_id"].isin(wells)]
    if not include_augmented_records:
        result = result[~result["is_augmented"].astype(bool)]
    result = result.copy()
    if result.empty:
        raise ValueError("No training samples remain after applying the augmented-record filter.")
    return result


def originals_for_well(df: pd.DataFrame, well_id: int) -> pd.DataFrame:
    result = df[(df["well_id"] == well_id) & (~df["is_augmented"].astype(bool))].copy()
    if result.empty:
        raise ValueError(f"Well {well_id} has no original validation samples.")
    return result
