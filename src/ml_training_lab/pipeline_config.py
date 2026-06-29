from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_pipeline_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline config does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def optional_wells(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise TypeError("training_wells must be null or a list of well ids.")
    wells = tuple(int(item) for item in value)
    if not wells:
        raise ValueError("training_wells cannot be empty.")
    return wells


def resolve_training_wells(
    df: pd.DataFrame,
    *,
    target_well: int,
    configured_wells: tuple[int, ...] | None,
) -> tuple[int, ...]:
    available_wells = tuple(sorted(int(well) for well in df["well_id"].unique().tolist()))
    if configured_wells is None:
        wells = tuple(well for well in available_wells if well != target_well)
    else:
        wells = configured_wells
    if not wells:
        raise ValueError("No training wells remain after excluding the target well.")
    if target_well in wells:
        raise ValueError("The target well must not participate in training or tuning.")
    missing = sorted(set(wells) - set(available_wells))
    if missing:
        raise ValueError(f"Configured training wells {missing} are not present in the dataset.")
    if target_well not in available_wells:
        raise ValueError(f"Target well {target_well} is not present in the dataset.")
    return wells
