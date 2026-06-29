from __future__ import annotations

from ml_training_lab.application.extract_dinov3_embeddings import (
    extract_dinov3_embeddings as extract_dinov3_embeddings_use_case,
)
from ml_training_lab.presentation.requests import EmbeddingExtractionRequest
from ml_training_lab.presentation.responses import EmbeddingExtractionResultDto


def extract_dinov3_embeddings(request: EmbeddingExtractionRequest) -> EmbeddingExtractionResultDto:
    return extract_dinov3_embeddings_use_case(request)
