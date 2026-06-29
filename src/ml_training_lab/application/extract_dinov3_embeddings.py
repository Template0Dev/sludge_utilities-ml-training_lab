from __future__ import annotations

from ml_training_lab.infrastructure.dataframe_storage import read_parquet, write_frame_outputs
from ml_training_lab.infrastructure.dinov3_embedding_extractor import extract_embeddings
from ml_training_lab.presentation.requests import EmbeddingExtractionRequest
from ml_training_lab.presentation.responses import EmbeddingExtractionResultDto
from ml_training_lab.shared.device_selector import torch_device


def extract_dinov3_embeddings(request: EmbeddingExtractionRequest) -> EmbeddingExtractionResultDto:
    df = read_parquet(request.input_path).copy()
    device = torch_device()
    lba_embeddings = extract_embeddings(
        df=df,
        image_root=request.image_root,
        image_column=request.lba_image_column,
        model_name=request.model_name,
        batch_size=request.batch_size,
        device=device,
    )
    sludge_embeddings = extract_embeddings(
        df=df,
        image_root=request.image_root,
        image_column=request.sludge_image_column,
        model_name=request.model_name,
        batch_size=request.batch_size,
        device=device,
    )
    df[request.lba_embedding_column] = list(lba_embeddings)
    df[request.sludge_embedding_column] = list(sludge_embeddings)
    write_frame_outputs(df, request.output_csv_path, request.output_parquet_path)
    return EmbeddingExtractionResultDto(
        output_csv_path=request.output_csv_path,
        output_parquet_path=request.output_parquet_path,
        row_count=len(df),
    )
