from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_training_lab.gb_workflow import fit_fold_features
from ml_training_lab.optuna_support import assert_tuning_protocol, macro_mae, originals_for_well, prepare_study


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


if __name__ == "__main__":
    unittest.main()
