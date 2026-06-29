from __future__ import annotations

import pandas as pd


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
