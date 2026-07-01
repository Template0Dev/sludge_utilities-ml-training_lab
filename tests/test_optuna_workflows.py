from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import optuna
import pandas as pd
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_training_lab.application.train_final_catboost import (
    _assert_summary_protocol as assert_catboost_summary_protocol,
)
from ml_training_lab.application.train_final_resnet import _assert_summary_protocol as assert_resnet_summary_protocol
from ml_training_lab.domain.metrics import macro_mae
from ml_training_lab.domain.record_selection import originals_for_well, training_records
from ml_training_lab.domain.training_wells import DatasetSplit, resolve_dataset_split
from ml_training_lab.domain.tuning_protocol import assert_tuning_protocol, validation_well_objective_metadata
from ml_training_lab.infrastructure.catboost_feature_matrix import fit_fold_features
from ml_training_lab.infrastructure.catboost_parameters import model_parameters
from ml_training_lab.infrastructure.embedding_joiner import load_feature_dataset
from ml_training_lab.infrastructure.last_tune_marker import read_last_tune_id, write_last_tune_id
from ml_training_lab.infrastructure.optuna_study_repository import prepare_study
from ml_training_lab.infrastructure.output_path_builder import model_runs_root, model_tuning_root
from ml_training_lab.infrastructure.resnet_backbone import create_resnet_backbone
from ml_training_lab.infrastructure.run_artifact_writer import create_run_dir, write_run_metadata
from ml_training_lab.infrastructure.tuning_run_directory import create_tuning_run_dir
from ml_training_lab.infrastructure.tuning_summary_path import latest_tuning_summary_path
from ml_training_lab.presentation.common_config import AppOutputConfig, FeatureConfig
from ml_training_lab.presentation.model_configs import CatBoostPipelineConfig, ResNetPipelineConfig
from ml_training_lab.shared.device_selector import accelerator, catboost_task_type


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "well_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "is_augmented": [False, True] * 4,
        })

    def test_validation_well_cannot_participate_in_tuning(self) -> None:
        with self.assertRaises(ValueError):
            assert_tuning_protocol(self.frame, (1, 2, 3), 1)

    def test_validation_selects_original_rows_only(self) -> None:
        validation = originals_for_well(self.frame, 2)
        self.assertEqual(len(validation), 1)
        self.assertFalse(validation["is_augmented"].any())

    def test_training_can_exclude_augmented_rows(self) -> None:
        selected = training_records(self.frame, (2, 3), include_augmented_records=False)
        self.assertEqual(len(selected), 2)
        self.assertFalse(selected["is_augmented"].any())

    def test_training_keeps_augmented_rows_by_default(self) -> None:
        selected = training_records(self.frame, (2, 3), include_augmented_records=True)
        self.assertEqual(len(selected), 4)
        self.assertTrue(selected["is_augmented"].any())

    def test_validation_and_test_can_share_well(self) -> None:
        split = resolve_dataset_split(self.frame, training_wells=(1, 2), validation_well=3, test_well=3)
        self.assertEqual(split.training_wells, (1, 2))
        self.assertEqual(split.validation_well, 3)
        self.assertEqual(split.test_well, 3)

    def test_training_wells_cannot_include_validation_well(self) -> None:
        with self.assertRaises(ValueError):
            resolve_dataset_split(self.frame, training_wells=(1, 2), validation_well=2, test_well=3)

    def test_training_wells_cannot_include_test_well(self) -> None:
        with self.assertRaises(ValueError):
            resolve_dataset_split(self.frame, training_wells=(1, 2), validation_well=3, test_well=2)

    def test_split_wells_must_exist_in_dataset(self) -> None:
        with self.assertRaises(ValueError):
            resolve_dataset_split(self.frame, training_wells=(1, 2), validation_well=3, test_well=9)

    def test_macro_mae_averages_targets_equally(self) -> None:
        score, target_scores = macro_mae(np.array([[0.0, 0.0]]), np.array([[2.0, 4.0]]))
        self.assertEqual(target_scores, [2.0, 4.0])
        self.assertEqual(score, 3.0)


class PersistenceTests(unittest.TestCase):
    def test_enqueued_baseline_still_requires_one_optimization_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            study, remaining = prepare_study(
                study_name="test",
                database_path=Path(directory) / "study.db",
                signature="signature",
                total_trials=1,
                baseline={"x": 1.0},
                pruner=optuna.pruners.NopPruner(),
            )
            self.assertEqual(remaining, 1)
            self.assertEqual(study.trials[0].state, optuna.trial.TrialState.WAITING)


class FeatureTests(unittest.TestCase):
    def test_pca_is_fitted_only_from_training_embeddings(self) -> None:
        features = FeatureConfig(
            base_columns=("interval_start", "interval_end"),
            should_use_sludge_embeddings=True,
            should_use_lba_embeddings=False,
            embedding_join_keys=("well_id", "interval_start", "interval_end"),
        )
        train = pd.DataFrame({
            "interval_start": [0.0, 1.0, 2.0],
            "interval_end": [1.0, 2.0, 3.0],
            "sludge_dinov3_emb": [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        })
        validation = pd.DataFrame({
            "interval_start": [3.0],
            "interval_end": [4.0],
            "sludge_dinov3_emb": [np.array([1000.0, 1000.0])],
        })
        _, _, transformers = fit_fold_features(train, validation, features=features, components=1)
        np.testing.assert_allclose(transformers["sludge_dinov3_emb"].mean_, np.array([1 / 3, 1 / 3]))

    def test_embedding_flags_join_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_path, embedding_path = self._write_feature_frames(Path(directory))
            sludge_only = load_feature_dataset(
                base_dataset_path=base_path,
                embedding_dataset_path=embedding_path,
                features=FeatureConfig(
                    should_use_sludge_embeddings=True,
                    should_use_lba_embeddings=False,
                    embedding_join_keys=("well_id", "interval_start", "interval_end"),
                ),
            )
            lba_only = load_feature_dataset(
                base_dataset_path=base_path,
                embedding_dataset_path=embedding_path,
                features=FeatureConfig(
                    should_use_sludge_embeddings=False,
                    should_use_lba_embeddings=True,
                    embedding_join_keys=("well_id", "interval_start", "interval_end"),
                ),
            )
            neither = load_feature_dataset(
                base_dataset_path=base_path,
                embedding_dataset_path=None,
                features=FeatureConfig(
                    should_use_sludge_embeddings=False,
                    should_use_lba_embeddings=False,
                    embedding_join_keys=("well_id", "interval_start", "interval_end"),
                ),
            )
        self.assertIn("sludge_dinov3_emb", sludge_only.columns)
        self.assertNotIn("lba_dinov3_emb", sludge_only.columns)
        self.assertIn("lba_dinov3_emb", lba_only.columns)
        self.assertNotIn("sludge_dinov3_emb", lba_only.columns)
        self.assertNotIn("sludge_dinov3_emb", neither.columns)
        self.assertNotIn("lba_dinov3_emb", neither.columns)

    def test_duplicate_embedding_join_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_path, embedding_path = self._write_feature_frames(Path(directory), duplicate_embeddings=True)
            with self.assertRaises(ValueError):
                load_feature_dataset(
                    base_dataset_path=base_path,
                    embedding_dataset_path=embedding_path,
                    features=FeatureConfig(
                        should_use_sludge_embeddings=True,
                        should_use_lba_embeddings=False,
                        embedding_join_keys=("well_id", "interval_start", "interval_end"),
                    ),
                )

    def test_sludge_embeddings_only_allows_empty_base_columns(self) -> None:
        features = FeatureConfig(
            base_columns=(),
            should_use_sludge_embeddings=True,
            should_use_lba_embeddings=False,
            embedding_join_keys=("well_id", "interval_start", "interval_end"),
        )
        train = pd.DataFrame({
            "well_id": [1, 1, 1],
            "interval_start": [0.0, 1.0, 2.0],
            "interval_end": [1.0, 2.0, 3.0],
            "sludge_dinov3_emb": [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        })
        validation = pd.DataFrame({
            "well_id": [2],
            "interval_start": [3.0],
            "interval_end": [4.0],
            "sludge_dinov3_emb": [np.array([1.0, 1.0])],
        })

        x_train, x_validation, _ = fit_fold_features(train, validation, features=features, components=1)

        self.assertEqual(x_train.columns.tolist(), ["sludge_emb_pca_0"])
        self.assertEqual(x_validation.columns.tolist(), ["sludge_emb_pca_0"])

    def test_feature_config_rejects_no_feature_sources(self) -> None:
        with self.assertRaises(ValidationError):
            FeatureConfig(
                base_columns=(),
                should_use_sludge_embeddings=False,
                should_use_lba_embeddings=False,
                embedding_join_keys=("well_id",),
            )

    def _write_feature_frames(self, directory: Path, duplicate_embeddings: bool = False) -> tuple[Path, Path]:
        base = pd.DataFrame({
            "well_id": [1, 2],
            "interval_start": [10, 20],
            "interval_end": [15, 25],
        })
        embeddings = base.copy()
        embeddings["sludge_dinov3_emb"] = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        embeddings["lba_dinov3_emb"] = [np.array([5.0, 6.0]), np.array([7.0, 8.0])]
        if duplicate_embeddings:
            embeddings = pd.concat([embeddings, embeddings.head(1)], ignore_index=True)
        base_path = directory / "base.parquet"
        embedding_path = directory / "embedding.parquet"
        base.to_parquet(base_path)
        embeddings.to_parquet(embedding_path)
        return base_path, embedding_path


class PipelineConfigTests(unittest.TestCase):
    def test_checked_in_gb_config_loads_feature_settings(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = CatBoostPipelineConfig.model_validate_json(
            (project_root / "config/gb_training.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config.dataset_params.training_wells, (1, 2, 3, 5, 6, 7, 8))
        self.assertEqual(config.dataset_params.validation_well, 4)
        self.assertEqual(config.dataset_params.test_well, 4)
        self.assertFalse(hasattr(config, "target_well"))
        self.assertEqual(config.tuning_params.study_name, "catboost_dataset_split_v1")
        self.assertTrue(config.features.should_use_sludge_embeddings)
        self.assertFalse(config.features.should_use_lba_embeddings)
        self.assertTrue(config.final_training.save_predictions)

    def test_resnet_config_loads_tuning_and_initial_hyper_params(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = ResNetPipelineConfig.model_validate_json(
            (project_root / "config/resnet_training.json").read_text(encoding="utf-8")
        )
        self.assertFalse(hasattr(config, "features"))
        self.assertEqual(config.dataset_params.training_wells, (1, 2, 3, 5, 6, 7, 8))
        self.assertEqual(config.dataset_params.validation_well, 4)
        self.assertEqual(config.dataset_params.test_well, 4)
        self.assertEqual(config.tuning_params.study_name, "resnet_dataset_split_v1")
        self.assertNotIn("batch_size", config.search_params)
        self.assertNotIn("batch_size", config.initial_hyper_params)
        self.assertEqual(config.fixed_hyper_params["batch_size"], 16)
        self.assertEqual(config.fixed_hyper_params["accumulate_grad_batches"], 4)
        self.assertEqual(config.data_loader.num_workers, 13)
        self.assertTrue(config.final_training.save_predictions)

    def test_resnet_config_rejects_gb_feature_settings(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = json.loads((project_root / "config/resnet_training.json").read_text(encoding="utf-8"))
        payload["features"] = {"base_columns": ["interval_start", "interval_end"]}

        with self.assertRaises(ValidationError):
            ResNetPipelineConfig.model_validate(payload)

    def test_validation_well_objective_metadata(self) -> None:
        metadata = validation_well_objective_metadata({"validation_strategy": "validation_well"})
        self.assertEqual(metadata["name"], "validation-well macro MAE")

    def test_old_top_level_split_keys_are_rejected(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = json.loads((project_root / "config/gb_training.json").read_text(encoding="utf-8"))
        payload["target_well"] = 4
        payload["training_wells"] = [1, 2, 3]

        with self.assertRaises(ValidationError):
            CatBoostPipelineConfig.model_validate(payload)

    def test_dataset_params_rejects_training_overlap(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = json.loads((project_root / "config/gb_training.json").read_text(encoding="utf-8"))
        payload["dataset_params"]["training_wells"].append(payload["dataset_params"]["validation_well"])

        with self.assertRaises(ValidationError):
            CatBoostPipelineConfig.model_validate(payload)

    def test_dataset_params_rejects_duplicate_training_wells(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = json.loads((project_root / "config/gb_training.json").read_text(encoding="utf-8"))
        payload["dataset_params"]["training_wells"].append(payload["dataset_params"]["training_wells"][0])

        with self.assertRaises(ValidationError):
            CatBoostPipelineConfig.model_validate(payload)

    def test_final_summary_protocol_uses_validation_well_not_test_well(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        catboost_config = CatBoostPipelineConfig.model_validate_json(
            (project_root / "config/gb_training.json").read_text(encoding="utf-8")
        )
        split = DatasetSplit(
            training_wells=catboost_config.dataset_params.training_wells,
            validation_well=catboost_config.dataset_params.validation_well,
            test_well=catboost_config.dataset_params.test_well,
        )
        summary = {
            "protocol": {
                "training_wells": list(split.training_wells),
                "validation_well": split.validation_well,
                "include_augmented_records": catboost_config.include_augmented_records,
                "pca_fit_inside_fold": True,
            }
        }
        assert_catboost_summary_protocol(summary, split, catboost_config)

    def test_resnet_summary_protocol_uses_validation_well_not_test_well(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        resnet_config = ResNetPipelineConfig.model_validate_json(
            (project_root / "config/resnet_training.json").read_text(encoding="utf-8")
        )
        split = DatasetSplit(
            training_wells=resnet_config.dataset_params.training_wells,
            validation_well=resnet_config.dataset_params.validation_well,
            test_well=resnet_config.dataset_params.test_well,
        )
        summary = {
            "protocol": {
                "training_wells": list(split.training_wells),
                "validation_well": split.validation_well,
                "include_augmented_records": resnet_config.include_augmented_records,
                "input_size": int(resnet_config.fixed_hyper_params["input_size"]),
                "normalization": "ImageNet",
            }
        }
        assert_resnet_summary_protocol(summary, split, resnet_config)

    def test_optuna_summary_path_is_no_longer_accepted(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = json.loads((project_root / "config/gb_training.json").read_text(encoding="utf-8"))
        payload["final_training"]["optuna_summary_path"] = "catboost_dataset_split_v1_summary.json"

        with self.assertRaises(ValidationError):
            CatBoostPipelineConfig.model_validate(payload)


class OutputPathTests(unittest.TestCase):
    def test_output_paths_are_built_from_config_parts(self) -> None:
        output = AppOutputConfig(
            output_base_folder="out",
            output_gb_sub_folder="gradient",
            output_runs_prefix_for_saving="requests",
            output_tuning_prefix_for_saving="studies",
        )
        self.assertEqual(model_runs_root(Path("/project"), output, output.output_gb_sub_folder), Path("/project/out/gradient/requests"))
        self.assertEqual(model_tuning_root(Path("/project"), output, output.output_gb_sub_folder), Path("/project/out/gradient/studies"))

    def test_run_artifact_metadata_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = create_run_dir(Path(directory), "run-id")
            write_run_metadata(run_dir, {"metric": 1}, {"config": "value"}, Path(__file__).resolve().parents[1])
            self.assertTrue((run_dir / "request_meta.json").is_file())
            self.assertTrue((run_dir / "config_snapshot.json").is_file())
            self.assertTrue((run_dir / "version_info.txt").is_file())
            self.assertEqual(json.loads((run_dir / "request_meta.json").read_text())["metric"], 1)

    def test_last_tune_marker_contains_uuid_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tuning_root = Path(directory)
            write_last_tune_id(tuning_root, "tune-uuid")

            self.assertEqual(json.loads((tuning_root / ".last_tune.json").read_text()), "tune-uuid")
            self.assertEqual(read_last_tune_id(tuning_root), "tune-uuid")

    def test_latest_tuning_summary_uses_marker_uuid_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tuning_root = Path(directory)
            tune_dir = create_tuning_run_dir(tuning_root, "tune-uuid")
            summary_path = tune_dir / "study_summary.json"
            summary_path.write_text("{}", encoding="utf-8")
            write_last_tune_id(tuning_root, "tune-uuid")

            self.assertEqual(latest_tuning_summary_path(tuning_root, "study"), summary_path)

    def test_missing_last_tune_marker_requires_tuning_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                latest_tuning_summary_path(Path(directory), "study")


class ResNetBackboneTests(unittest.TestCase):
    @patch("ml_training_lab.infrastructure.resnet_backbone.hf_hub_download")
    @patch("ml_training_lab.infrastructure.resnet_backbone.timm.create_model")
    def test_network_failure_loads_cached_weights(self, create_model, hub_download) -> None:
        backbone = object()
        create_model.side_effect = [httpx.ProxyError("proxy unavailable"), backbone]
        hub_download.return_value = "/cache/model.safetensors"

        self.assertIs(create_resnet_backbone(), backbone)
        hub_download.assert_called_once_with(
            repo_id="timm/resnet50d.ra2_in1k",
            filename="model.safetensors",
            local_files_only=True,
        )
        self.assertEqual(
            create_model.call_args_list[1].kwargs["pretrained_cfg_overlay"],
            {"file": "/cache/model.safetensors"},
        )

    @patch("ml_training_lab.shared.device_selector.torch.backends.mps.is_available", return_value=True)
    @patch("ml_training_lab.shared.device_selector.torch.cuda.is_available", return_value=False)
    def test_mps_is_selected_when_available(self, _cuda_available, _mps_available) -> None:
        self.assertEqual(accelerator(), "mps")

    @patch("ml_training_lab.shared.device_selector.torch.backends.mps.is_available", return_value=True)
    @patch("ml_training_lab.shared.device_selector.torch.cuda.is_available", return_value=True)
    def test_cuda_is_selected_before_mps(self, _cuda_available, _mps_available) -> None:
        self.assertEqual(accelerator(), "cuda")


class CatBoostDeviceTests(unittest.TestCase):
    @patch("ml_training_lab.shared.device_selector.torch.cuda.is_available", return_value=True)
    def test_catboost_uses_gpu_when_cuda_is_available(self, _cuda_available) -> None:
        self.assertEqual(catboost_task_type(), "GPU")

    @patch("ml_training_lab.shared.device_selector.torch.cuda.is_available", return_value=True)
    def test_catboost_configured_task_type_overrides_cuda(self, _cuda_available) -> None:
        self.assertEqual(catboost_task_type("CPU"), "CPU")

    @patch("ml_training_lab.shared.device_selector.torch.cuda.is_available", return_value=True)
    def test_catboost_model_parameters_set_gpu_without_config_override(self, _cuda_available) -> None:
        params = model_parameters(
            {"learning_rate": 0.1, "depth": 4, "pca_components": 16},
            {"iterations": 100, "early_stopping_rounds": 10, "loss_function": "MultiRMSE"},
            iterations=20,
        )

        self.assertEqual(params["task_type"], "GPU")
        self.assertEqual(params["iterations"], 20)
        self.assertNotIn("pca_components", params)


if __name__ == "__main__":
    unittest.main()
