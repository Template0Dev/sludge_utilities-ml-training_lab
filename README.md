# ML training lab

The project uses a singular PADIC layout under `src/ml_training_lab`:

- `controller/`: notebook-facing API functions.
- `application/`: one use case per file.
- `domain/`: pure training rules, metrics, protocol checks, and feature rules.
- `infrastructure/`: filesystem and library adapters.
- `presentation/`: Pydantic request, response, and config DTOs.
- `shared/`: single-purpose technical helpers.

Notebooks should define local paths and call controller functions. Training and tuning
implementations live in `src`.

## Controller workflow

```python
from ml_training_lab.controller.catboost_tuning_controller import tune_catboost
from ml_training_lab.presentation.requests import CatBoostRequest

request = CatBoostRequest(
    project_root=PROJECT_ROOT,
    config_path=PROJECT_ROOT / "config/gb_training.json",
    base_dataset_path=PROJECT_ROOT / "data/meta/interim/metadata_augmented.parquet",
    embedding_dataset_path=PROJECT_ROOT / "data/meta/processed/gb/metadata_dinov3_embeddings.parquet",
)
result = tune_catboost(request)
```

Equivalent controllers exist for final CatBoost training, ResNet tuning/training,
augmentation, and DINOv3 embedding extraction.

## Configuration

Training configs define target columns, tuning params, search params,
initial hyper-params, fixed hyper-params, and output path parts. CatBoost also
defines feature flags for tabular and embedding inputs; ResNet uses sludge
images directly from the dataset metadata.

GB embedding features are independent:

- `should_use_sludge_embeddings`: joins and uses `sludge_dinov3_emb`.
- `should_use_lba_embeddings`: joins and uses `lba_dinov3_emb`.
- If both are false, only `features.base_columns` are used.

Output roots are built from config parts:

```python
Path(output_base_folder).joinpath(output_gb_sub_folder, output_runs_prefix_for_saving)
Path(output_base_folder).joinpath(output_gb_sub_folder, output_tuning_prefix_for_saving)
```

The same rule applies to ResNet with `output_resnet_sub_folder`.

## Outputs

Final training writes one folder per run:

```text
output/
  gb/
    runs/
      {uuid}/
        request_meta.json
        config_snapshot.json
        version_info.txt
        model.cbm
        embeddings.joblib
        predictions.csv
    tuning/
      .last_tune.json
      {tune_uuid}/
        {study_name}.db
        {study_name}_summary.json
        {study_name}_trials.csv
        config_snapshot.json
        version_info.txt
  resnet/
    runs/
      {uuid}/
        request_meta.json
        config_snapshot.json
        version_info.txt
        model.ckpt
        predictions.csv
    tuning/
      .last_tune.json
      {tune_uuid}/
        {study_name}.db
        {study_name}_summary.json
        {study_name}_trials.csv
        config_snapshot.json
        version_info.txt
```

Optuna tuning files are persistent and intentionally outside final-training run folders.
Final training reads the latest tuning UUID from `.last_tune.json`.
