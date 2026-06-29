from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def read_pipeline_config(path: Path, config_type: type[ConfigT]) -> ConfigT:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline config does not exist: {path}")
    return config_type.model_validate_json(path.read_text(encoding="utf-8"))
