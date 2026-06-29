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

from ml_training_lab.application.train_final_catboost import _summary_path as catboost_summary_path
from ml_training_lab.domain.metrics import macro_mae
from ml_training_lab.domain.record_selection import originals_for_well, training_records
from ml_training_lab.domain.training_wells import resolve_training_wells
from ml_training_lab.domain.tuning_protocol import assert_tuning_protocol, target_well_objective_metadata
from ml_training_lab.infrastructure.catboost_feature_matrix import fit_fold_features
from ml_training_lab.infrastructure.embedding_joiner import load_feature_dataset
from ml_training_lab.infrastructure.optuna_study_repository import prepare_study
from ml_training_lab.infrastructure.output_path_builder import model_runs_root, model_tuning_root
from ml_training_lab.infrastructure.resnet_backbone import create_resnet_backbone
from ml_training_lab.infrastructure.run_artifact_writer import create_run_dir, write_run_metadata
from ml_training_lab.presentation.common_config import AppOutputConfig, FeatureConfig
from ml_training_lab.presentation.model_configs import CatBoostPipelineConfig, ResNetPipelineConfig
from ml_training_lab.shared.device_selector import accelerator


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "well_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "is_augmented": [False, True] * 4,
        })

    def test_holdout_cannot_participate_in_tuning(self) -> None:
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

    def test_training_wells_default_to_all_except_target(self) -> None:
        wells = resolve_training_wells(self.frame, target_well=1, configured_wells=None)
        self.assertEqual(wells, (2, 3, 4))

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
    def test_checked_in_gb_config_uses_target_well_one(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = CatBoostPipelineConfig.model_validate_json(
            (project_root / "config/gb_training.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config.target_well, 1)
        self.assertEqual(config.training_wells, (2, 3, 4))
        self.assertTrue(config.features.should_use_sludge_embeddings)
        self.assertFalse(config.features.should_use_lba_embeddings)
        self.assertEqual(config.final_training.optuna_summary_path, "catboost_target_well_v1_summary.json")

    def test_resnet_config_loads_tuning_and_initial_hyper_params(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = ResNetPipelineConfig.model_validate_json(
            (project_root / "config/resnet_training.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config.tuning_params.study_name, "resnet_target_well_v1")
        self.assertEqual(config.initial_hyper_params["batch_size"], 1)
        self.assertEqual(config.data_loader.num_workers, 13)
        self.assertEqual(config.final_training.optuna_summary_path, "resnet_target_well_v1_summary.json")

    def test_target_well_objective_metadata(self) -> None:
        metadata = target_well_objective_metadata({"validation_strategy": "target_well"})
        self.assertEqual(metadata["name"], "target-well macro MAE")

    def test_optuna_summary_path_must_be_file_name(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = json.loads((project_root / "config/gb_training.json").read_text(encoding="utf-8"))
        payload["final_training"]["optuna_summary_path"] = "output/gb/tuning/summary.json"

        with self.assertRaises(ValidationError):
            CatBoostPipelineConfig.model_validate(payload)

    def test_catboost_summary_path_uses_output_config(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = CatBoostPipelineConfig.model_validate_json(
            (project_root / "config/gb_training.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            catboost_summary_path(Path("/project"), config),
            Path("/project/output/gb/tuning/catboost_target_well_v1_summary.json"),
        )


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
    def test_mps_is_selected_when_available(self, _is_available) -> None:
        self.assertEqual(accelerator(), "mps")


if __name__ == "__main__":
    unittest.main()
