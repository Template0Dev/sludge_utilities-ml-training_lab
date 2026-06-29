from __future__ import annotations

from ml_training_lab.infrastructure.dataframe_storage import read_parquet, write_frame_outputs
from ml_training_lab.infrastructure.image_augmentation import enrich_with_augmentations
from ml_training_lab.presentation.requests import AugmentationRequest
from ml_training_lab.presentation.responses import AugmentationResultDto


def run_augmentation(request: AugmentationRequest) -> AugmentationResultDto:
    source_df = read_parquet(request.input_path)
    enriched_df = enrich_with_augmentations(
        source_df,
        image_root=request.image_root,
        sludge_image_column=request.sludge_image_column,
        lba_image_column=request.lba_image_column,
        is_augmented_column=request.is_augmented_column,
        sludge_augmented_folder=request.sludge_augmented_folder,
        lba_augmented_folder=request.lba_augmented_folder,
        augmentations_per_record=request.augmentations_per_record,
        random_seed=request.random_seed,
    )
    write_frame_outputs(enriched_df, request.output_csv_path, request.output_parquet_path)
    return AugmentationResultDto(
        output_csv_path=request.output_csv_path,
        output_parquet_path=request.output_parquet_path,
        generated_record_count=int(enriched_df[request.is_augmented_column].sum()),
        total_record_count=len(enriched_df),
    )
