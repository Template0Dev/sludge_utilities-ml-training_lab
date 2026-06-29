from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .json_values import json_value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
