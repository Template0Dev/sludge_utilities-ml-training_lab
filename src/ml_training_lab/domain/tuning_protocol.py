from __future__ import annotations

import pandas as pd


def assert_tuning_protocol(df: pd.DataFrame, tuning_wells: tuple[int, ...], validation_well: int) -> None:
    available = set(df["well_id"].unique().tolist())
    required = set(tuning_wells) | {validation_well}
    if not required.issubset(available):
        raise ValueError(f"Dataset wells {sorted(available)} do not contain required wells {sorted(required)}")
    if validation_well in tuning_wells:
        raise ValueError("The validation well must not participate in tuning.")


def validation_well_objective_metadata(protocol: dict[str, object]) -> dict[str, str]:
    if protocol.get("validation_strategy") == "validation_well":
        return {
            "name": "validation-well macro MAE",
            "direction": "minimize",
            "aggregation": "target-macro MAE on original samples from the configured validation well",
        }
    return {
        "name": "equal-well macro MAE",
        "direction": "minimize",
        "aggregation": "mean of per-validation-well target-macro MAE",
    }
