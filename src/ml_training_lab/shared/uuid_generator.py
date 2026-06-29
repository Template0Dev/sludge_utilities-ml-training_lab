from __future__ import annotations

from uuid6 import uuid6


def new_uuid() -> str:
    return str(uuid6())
