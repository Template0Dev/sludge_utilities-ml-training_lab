from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_training_lab.gb_workflow import GBTuningConfig, fit_fold_features, training_records as gb_training_records
from ml_training_lab.optuna_support import (
    assert_tuning_protocol,
    macro_mae,
    objective_metadata,
    originals_for_well,
    prepare_study,
)
from ml_training_lab.pipeline_config import resolve_training_wells
from ml_training_lab.resnet_workflow import (
    ResNetFinalConfig,
    ResNetTuningConfig,
    accelerator,
    create_resnet_backbone,
    training_records as resnet_training_records,
)


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

    def test_gb_training_can_exclude_augmented_rows(self) -> None:
        training = gb_training_records(self.frame, (2, 3), include_augmented_records=False)
        self.assertEqual(len(training), 2)
        self.assertFalse(training["is_augmented"].any())

    def test_resnet_training_can_exclude_augmented_rows(self) -> None:
        training = resnet_training_records(self.frame, (2, 3), include_augmented_records=False)
        self.assertEqual(len(training), 2)
        self.assertFalse(training["is_augmented"].any())

    def test_training_keeps_augmented_rows_by_default(self) -> None:
        training = gb_training_records(self.frame, (2, 3), include_augmented_records=True)
        self.assertEqual(len(training), 4)
        self.assertTrue(training["is_augmented"].any())

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
                study_name="test", database_path=Path(directory) / "study.db", signature="signature",
                total_trials=1, baseline={"x": 1.0}, pruner=optuna.pruners.NopPruner(),
            )
            self.assertEqual(remaining, 1)
            self.assertEqual(study.trials[0].state, optuna.trial.TrialState.WAITING)


class PCAIsolationTests(unittest.TestCase):
    def test_pca_is_fitted_only_from_training_embeddings(self) -> None:
        train = pd.DataFrame({
            "interval_start": [0.0, 1.0, 2.0], "interval_end": [1.0, 2.0, 3.0],
            "sludge_dinov3_emb": [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        })
        validation = pd.DataFrame({
            "interval_start": [3.0], "interval_end": [4.0],
            "sludge_dinov3_emb": [np.array([1000.0, 1000.0])],
        })
        _, _, pca = fit_fold_features(train, validation, components=1)
        np.testing.assert_allclose(pca.mean_, np.array([1 / 3, 1 / 3]))


class PipelineConfigTests(unittest.TestCase):
    def test_gb_tuning_config_loads_from_file(self) -> None:
        from ml_training_lab.gb_workflow import GBTuningConfig

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps({
                    "target_well": 6,
                    "training_wells": None,
                    "include_augmented_records": False,
                    "optuna": {"study_name": "study", "total_trials": 3, "smoke_mode": True},
                }),
                encoding="utf-8",
            )

            config = GBTuningConfig.from_file(Path("."), config_path)

        self.assertEqual(config.target_well, 6)
        self.assertIsNone(config.training_wells)
        self.assertFalse(config.include_augmented_records)
        self.assertEqual(config.study_name, "study")
        self.assertEqual(config.total_trials, 3)
        self.assertTrue(config.smoke_mode)

    def test_resnet_final_config_resolves_relative_summary_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            config_path = project_root / "config.json"
            config_path.write_text(
                json.dumps({
                    "target_well": 2,
                    "training_wells": [1, 3],
                    "data_loader": {"num_workers": 2},
                    "final_training": {"optuna_summary_path": "output/summary.json", "save_predictions": False},
                }),
                encoding="utf-8",
            )

            config = ResNetFinalConfig.from_file(project_root, config_path)

        self.assertEqual(config.target_well, 2)
        self.assertEqual(config.training_wells, (1, 3))
        self.assertEqual(config.num_workers, 2)
        self.assertFalse(config.save_predictions)
        self.assertEqual(config.optuna_summary_path, project_root / "output/summary.json")

    def test_checked_in_gb_config_excludes_well_one_and_targets_well_two(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = GBTuningConfig.from_file(project_root, project_root / "config/gb_training.json")

        self.assertEqual(config.target_well, 2)
        self.assertEqual(config.training_wells, (3, 4, 5, 6, 7, 8))
        self.assertNotIn(1, config.training_wells)

    def test_target_well_objective_metadata(self) -> None:
        metadata = objective_metadata({"validation_strategy": "target_well"})

        self.assertEqual(metadata["name"], "target-well macro MAE")


class ResNetBackboneTests(unittest.TestCase):
    @patch("ml_training_lab.resnet_workflow.hf_hub_download")
    @patch("ml_training_lab.resnet_workflow.timm.create_model")
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

    def test_configs_default_to_thirteen_data_workers(self) -> None:
        tuning_config = ResNetTuningConfig(project_root=Path("."))
        final_config = ResNetFinalConfig(project_root=Path("."), optuna_summary_path=Path("summary.json"))

        self.assertEqual(tuning_config.num_workers, 13)
        self.assertEqual(final_config.num_workers, 13)
        self.assertTrue(final_config.save_predictions)
        self.assertTrue(tuning_config.include_augmented_records)
        self.assertTrue(final_config.include_augmented_records)
        self.assertIsNone(tuning_config.training_wells)
        self.assertIsNone(final_config.training_wells)

    @patch("ml_training_lab.resnet_workflow.torch.backends.mps.is_available", return_value=True)
    def test_mps_is_selected_when_available(self, _is_available) -> None:
        self.assertEqual(accelerator(), "mps")


if __name__ == "__main__":
    unittest.main()
