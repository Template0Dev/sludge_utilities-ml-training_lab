from __future__ import annotations

import torch


def accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def catboost_task_type(configured_task_type: str | None = None) -> str:
    if configured_task_type:
        return configured_task_type
    if torch.cuda.is_available():
        return "GPU"
    return "CPU"


def torch_device() -> torch.device:
    return torch.device(accelerator())
