from __future__ import annotations

import httpx
import timm
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from torch import nn


RESNET_MODEL_NAME = "resnet50d"
RESNET_HF_REPOSITORY = "timm/resnet50d.ra2_in1k"
RESNET_HF_WEIGHTS = "model.safetensors"


def create_resnet_backbone() -> nn.Module:
    try:
        return timm.create_model(RESNET_MODEL_NAME, pretrained=True, num_classes=0)
    except httpx.HTTPError as network_error:
        try:
            cached_weights = hf_hub_download(
                repo_id=RESNET_HF_REPOSITORY,
                filename=RESNET_HF_WEIGHTS,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            raise RuntimeError(
                f"Could not download {RESNET_HF_REPOSITORY} and no locally cached "
                f"{RESNET_HF_WEIGHTS} file is available."
            ) from network_error
        return timm.create_model(
            RESNET_MODEL_NAME,
            pretrained=True,
            num_classes=0,
            pretrained_cfg_overlay={"file": cached_weights},
        )
