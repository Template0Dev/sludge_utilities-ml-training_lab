# ML training lab

The training workflow reserves well 1 as the final holdout and uses wells 2–4 for model selection.

## Optuna workflow

1. Run `notebooks/resnet/02-optuna_tuning.ipynb` or `notebooks/gb/03-optuna_tuning.ipynb`.
2. For a quick integration check, set `SMOKE_MODE = True` before running the tuning notebook.
3. Keep the explicit generated summary path in the corresponding final-training notebook.
4. Run `notebooks/resnet/01-model_training.ipynb` or `notebooks/gb/02-catboost_training.ipynb`.

Studies are persisted under `output/<model>/optuna/`. Rerunning a study resumes it only until its configured total trial count is reached. Final-training notebooks reject summaries produced from another dataset or validation protocol.

ResNet tuning uses 512×512 ImageNet-normalized inputs. CatBoost fits PCA independently within every validation fold and saves the final PCA transformer alongside the model.
