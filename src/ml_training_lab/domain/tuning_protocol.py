from __future__ import annotations

import pandas as pd


def assert_tuning_protocol(df: pd.DataFrame, tuning_wells: tuple[int, ...], holdout_well: int) -> None:
    available = set(df["well_id"].unique().tolist())
    required = set(tuning_wells) | {holdout_well}
    if not required.issubset(available):
        raise ValueError(f"Dataset wells {sorted(available)} do not contain required wells {sorted(required)}")
    if holdout_well in tuning_wells:
        raise ValueError("The final holdout well must not participate in tuning.")


def target_well_objective_metadata(protocol: dict[str, object]) -> dict[str, str]:
    if protocol.get("validation_strategy") == "target_well":
        return {
            "name": "target-well macro MAE",
            "direction": "minimize",
            "aggregation": "target-macro MAE on original samples from the configured target well",
        }
    return {
        "name": "equal-well macro MAE",
        "direction": "minimize",
        "aggregation": "mean of per-validation-well target-macro MAE",
    }
