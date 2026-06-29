# ML training lab

Training pipeline settings live in `config/gb_training.json` and `config/resnet_training.json`.
The `target_well` is the validation/evaluation well used by Optuna and final reporting.
Set `training_wells` to the wells that may be used for fitting; the current configs use
target well 2 and explicitly exclude well 1 from training.

## Optuna workflow

1. Run `notebooks/resnet/01-optuna_tuning.ipynb` or `notebooks/gb/02-optuna_params_tuning.ipynb`.
2. For a quick integration check, set `optuna.smoke_mode` to `true` in the model config.
3. Keep the explicit generated summary path in the corresponding final-training notebook.
4. Run `notebooks/resnet/02-final_model_training.ipynb` or `notebooks/gb/03-final_catboost_training.ipynb`.

Studies are persisted under `output/<model>/optuna/`. Rerunning a study resumes it only until its configured total trial count is reached. Final-training notebooks reject summaries produced from another dataset or validation protocol.
Optuna trains on the configured training wells and validates on original samples from the configured target well.

ResNet tuning uses 512×512 ImageNet-normalized inputs. CatBoost fits PCA independently within every validation fold and saves the final PCA transformer alongside the model.
