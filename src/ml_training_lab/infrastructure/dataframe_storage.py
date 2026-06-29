from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_frame_outputs(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
