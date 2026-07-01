from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    training_wells: tuple[int, ...]
    validation_well: int
    test_well: int


def resolve_dataset_split(
    df: pd.DataFrame,
    *,
    training_wells: tuple[int, ...],
    validation_well: int,
    test_well: int,
) -> DatasetSplit:
    available_wells = set(int(well) for well in df["well_id"].unique().tolist())
    wells = tuple(int(well) for well in training_wells)
    if not wells:
        raise ValueError("No training wells were configured.")
    if len(set(wells)) != len(wells):
        raise ValueError("Configured training wells cannot contain duplicates.")
    validation_well = int(validation_well)
    test_well = int(test_well)
    if validation_well in wells:
        raise ValueError("The validation well must not participate in training.")
    if test_well in wells:
        raise ValueError("The test well must not participate in training.")
    required = set(wells) | {validation_well, test_well}
    missing = sorted(required - available_wells)
    if missing:
        raise ValueError(f"Configured dataset split wells {missing} are not present in the dataset.")
    return DatasetSplit(training_wells=wells, validation_well=validation_well, test_well=test_well)
