from __future__ import annotations

import torch


def accelerator() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def torch_device() -> torch.device:
    return torch.device(accelerator())
