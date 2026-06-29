from __future__ import annotations

import gc

import torch


def cleanup_model() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
