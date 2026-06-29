from __future__ import annotations

import hashlib
import json
from typing import Any

from ml_training_lab.shared.json_values import json_value


def config_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(json_value(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
