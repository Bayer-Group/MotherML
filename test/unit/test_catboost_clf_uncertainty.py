import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from mother.ml.models.m_catboost import (
    CatboostClassifierMother,
)


class TestCatboostClassifierModels(unittest.TestCase):
    """Test suite for CatBoost classifier models with uncertainty estimation."""

    def setUp(self) -> None:
        """Set up test fixtures for classification tests."""
        # Binary classification data
        X_binary, y_binary = make_classification(
            n_samples=200, n_features=5, n_classes=2, n_informative=3, random_state=42
        )
        self.X_binary = pd.DataFrame(X_binary, columns=[f"feature_{i}" for i in range(X_binary.shape[1])])
        self.y_binary = pd.Series(y_binary, name="target")

        # Multiclass classification data
        X_multiclass, y_multiclass = make_classification(
            n_samples=300, n_features=5, n_classes=3, n_informative=3, random_state=42
        )
        self.X_multiclass = pd.DataFrame(X_multiclass, columns=[f"feature_{i}" for i in range(X_multiclass.shape[1])])
        self.y_multiclass = pd.Series(y_multiclass, name="target")

        # Multi-target binary classification data
        X_multi, y1 = make_classification(n_samples=200, n_features=5, n_classes=2, n_informative=3, random_state=42)
        _, y2 = make_classification(n_samples=200, n_features=5, n_classes=2, n_informative=3, random_state=43)
        self.X_multi_target = pd.DataFrame(X_multi, columns=[f"feature_{i}" for i in range(X_multi.shape[1])])
        self.y_multi_target = pd.DataFrame({"target_1": y1, "target_2": y2})

    # ========== Binary Classification Tests ==========

    def test_CatboostBinaryClassifier_predict(self) -> None:
        """Test binary classification predict method."""
        model = CatboostClassifierMother(
            target_type="single_target", model_type="classification_binary", iterations=50, learning_rate=0.1, verbose=0
        )
        model.fit(self.X_binary, self.y_binary)
        y_pred = model.predict(self.X_binary)
        proba = model.predict_proba(self.X_binary)

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(self.y_binary))
        self.assertEqual(proba.shape, (len(self.y_binary), 2))
        self.assertTrue(all(pred in [0, 1] for pred in y_pred.ravel()))

    def test_CatboostBinaryClassifier_predict_uncertainty(self) -> None:
        """Test binary classification predict_uncertainty method with probabilities."""
        model = CatboostClassifierMother(
            target_type="single_target", model_type="classification_binary", iterations=50, learning_rate=0.1, verbose=0
        )
        model.fit(self.X_binary, self.y_binary)
        y_pred = model.predict(self.X_binary)
        uncertainty_df = model.predict_uncertainty(self.X_binary)

        self.assertIsInstance(uncertainty_df, pd.DataFrame)
        self.assertEqual(len(uncertainty_df), len(self.X_binary))

        # Check standard uncertainty columns
        self.assertIn("knowledge_uncertainty", uncertainty_df.columns)
        self.assertIn("mean_predictions", uncertainty_df.columns)
        self.assertIn("pred", uncertainty_df.columns)
        np.testing.assert_array_equal(uncertainty_df["pred"].to_numpy(), y_pred.ravel())
        self.assertTrue((uncertainty_df["knowledge_uncertainty"] >= 0).all())
        # For classifiers, mean_predictions is intentionally None
        self.assertTrue(uncertainty_df["mean_predictions"].isna().all())

        # Check probability columns (2 classes for binary)
        self.assertIn("proba_0", uncertainty_df.columns)
        self.assertIn("proba_1", uncertainty_df.columns)
        self.assertTrue(np.allclose(uncertainty_df["proba_0"] + uncertainty_df["proba_1"], 1.0))
        self.assertTrue((uncertainty_df["proba_0"] >= 0).all() and (uncertainty_df["proba_1"] >= 0).all())

    def test_CatboostBinaryClassifier_predict_uncertainty_for_opt(self) -> None:
        """Test binary classification uncertainty_for_opt flag."""
        model = CatboostClassifierMother(
            target_type="single_target", model_type="classification_binary", iterations=50, learning_rate=0.1, verbose=0
        )
        model.fit(self.X_binary, self.y_binary)
        result_full = model.predict_uncertainty(self.X_binary, uncertainty_for_opt=False)
        result_opt = model.predict_uncertainty(self.X_binary, uncertainty_for_opt=True)

        self.assertIsInstance(result_opt, pd.DataFrame)
        self.assertIn("knowledge_uncertainty", result_opt.columns)
        self.assertEqual(len(result_opt.columns), 1)
        pd.testing.assert_series_equal(
            result_full["knowledge_uncertainty"], result_opt["knowledge_uncertainty"], check_exact=True
        )

    def test_CatboostBinaryClassifier_predict_uncertainty_single_row_proba(self) -> None:
        """Test single-row predict_proba output is normalized from 1D to 2D."""
        model = CatboostClassifierMother(target_type="single_target", model_type="classification_binary")
        X_single = self.X_binary.iloc[[0]]
        uncertainty_stub = pd.DataFrame(
            {"mean_predictions": [np.nan], "knowledge_uncertainty": [0.1]},
            index=X_single.index,
        )

        with (
            patch("mother.ml.models.m_catboost.utils.get_virtual_prediction", return_value=uncertainty_stub),
            patch.object(model, "predict", return_value=np.array([1])),
            patch.object(model, "predict_proba", return_value=np.array([0.25, 0.75])),
        ):
            uncertainty_df = model.predict_uncertainty(X_single)

        self.assertEqual(list(uncertainty_df[["proba_0", "proba_1"]].iloc[0]), [0.25, 0.75])
        self.assertEqual(uncertainty_df.loc[X_single.index[0], "pred"], 1)

    def test_CatboostBinaryClassifier_predict_uncertainty_invalid_proba_dimensions(self) -> None:
        """Test invalid predict_proba dimensions raise a clear error."""
        model = CatboostClassifierMother(target_type="single_target", model_type="classification_binary")
        uncertainty_stub = pd.DataFrame(
            {
                "mean_predictions": [np.nan] * len(self.X_binary),
                "knowledge_uncertainty": [0.1] * len(self.X_binary),
            },
            index=self.X_binary.index,
        )

        with (
            patch("mother.ml.models.m_catboost.utils.get_virtual_prediction", return_value=uncertainty_stub),
            patch.object(model, "predict", return_value=np.ones(len(self.X_binary))),
            patch.object(model, "predict_proba", return_value=np.zeros((len(self.X_binary), 2, 1))),
        ):
            with self.assertRaisesRegex(ValueError, "1D or 2D"):
                model.predict_uncertainty(self.X_binary)

    def test_CatboostBinaryClassifier_focal_loss(self) -> None:
        """Test binary classification with Focal loss."""
        model = CatboostClassifierMother(
            target_type="single_target",
            model_type="classification_binary",
            iterations=50,
            learning_rate=0.1,
            verbose=0,
            loss_function="Focal:focal_alpha=0.5;focal_gamma=2.0",
        )
        model.fit(self.X_binary, self.y_binary)
        y_pred = model.predict(self.X_binary)

        self.assertEqual(y_pred.shape, self.y_binary.shape)
        self.assertTrue(all(pred in [0, 1] for pred in y_pred))

    # ========== Multiclass Classification Tests ==========

    def test_CatboostMulticlassClassifier_predict(self) -> None:
        """Test multiclass classification predict method."""
        model = CatboostClassifierMother(
            target_type="single_target",
            model_type="classification_multiclass",
            iterations=50,
            learning_rate=0.1,
            verbose=0,
        )
        model.fit(self.X_multiclass, self.y_multiclass)
        y_pred = model.predict(self.X_multiclass)
        proba = model.predict_proba(self.X_multiclass)

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(self.y_multiclass))
        self.assertEqual(proba.shape, (len(self.y_multiclass), 3))

    # ========== Multi-target Classification Tests ==========

    def test_CatboostMultiTargetBinaryClassifier_predict(self) -> None:
        """Test multi-target binary classification predict method."""
        model = CatboostClassifierMother(
            target_type="multi_target", model_type="classification_binary", iterations=50, learning_rate=0.1, verbose=0
        )
        model.fit(self.X_multi_target, self.y_multi_target)
        y_pred = model.predict(self.X_multi_target)

        self.assertEqual(y_pred.shape[0], len(self.X_multi_target))
        self.assertEqual(y_pred.shape[1], 2)  # 2 targets

    def test_CatboostClassifier_get_set_params(self) -> None:
        """Test parameter getting and setting."""
        model = CatboostClassifierMother()

        # Test set_params
        model.set_params(target_type="multi_target", tune_boosting_type=True, model_type="classification_multiclass")
        self.assertEqual(model.target_type, "multi_target")
        self.assertTrue(model.tune_boosting_type)
        self.assertEqual(model.model_type, "classification_multiclass")

        # Test get_params
        params = model.get_params()
        self.assertEqual(params["target_type"], "multi_target")
        self.assertTrue(params["tune_boosting_type"])
        self.assertEqual(params["model_type"], "classification_multiclass")

    def test_CatboostClassifier_default_parameters(self) -> None:
        """Test default parameters."""
        model = CatboostClassifierMother()
        default_params = model.default_parameters()

        self.assertAlmostEqual(default_params["learning_rate"], 0.03)
        self.assertEqual(default_params["bootstrap_type"], "Bayesian")
        self.assertEqual(default_params["random_strength"], 1)
        self.assertEqual(default_params["grow_policy"], "SymmetricTree")
        self.assertEqual(default_params["boosting_type"], "Plain")
        self.assertEqual(default_params["max_depth"], 6)

        # Test with prefix
        prefixed_params = model.default_parameters(prefix="test_")
        self.assertAlmostEqual(prefixed_params["test_learning_rate"], 0.03)
        self.assertEqual(prefixed_params["test_max_depth"], 6)

    def test_CatboostClassifier_serialization(self) -> None:
        """Test model pickling and unpickling."""
        model = CatboostClassifierMother(
            target_type="single_target",
            tune_boosting_type=True,
            model_type="classification_binary",
            iterations=50,
            learning_rate=0.1,
            verbose=0,
        )
        model.fit(self.X_binary, self.y_binary)

        # Get predictions before pickling
        y_pred_original = model.predict(self.X_binary)

        # Pickle and unpickle
        serialized = pickle.dumps(model)
        deserialized_model = pickle.loads(serialized)

        # Test parameters are preserved
        self.assertEqual(deserialized_model.target_type, model.target_type)
        self.assertEqual(deserialized_model.tune_boosting_type, model.tune_boosting_type)
        self.assertEqual(deserialized_model.model_type, model.model_type)

        # Test predictions are the same
        y_pred_deserialized = deserialized_model.predict(self.X_binary)
        np.testing.assert_array_equal(y_pred_original, y_pred_deserialized)

    # ========== Loss-specific Parameter Tests ==========

    def test_CatboostClassifier_suggested_params_loss_logloss(self) -> None:
        """Test loss-specific parameter suggestions for Logloss."""
        import optuna

        model = CatboostClassifierMother(model_type="classification_binary")
        study = optuna.create_study()
        trial = study.ask()

        suggested_params = model.suggested_params_loss(trial, {}, self.y_binary, prefix="")
        self.assertIn("loss_function", suggested_params)
        loss_func = suggested_params["loss_function"]
        self.assertTrue(loss_func in ["Logloss"] or loss_func.startswith("Focal"))

        if loss_func.startswith("Focal"):
            self.assertIn("focal_alpha", loss_func)
            self.assertIn("focal_gamma", loss_func)
            self.assertEqual(suggested_params["auto_class_weights"], "None")

    def test_CatboostClassifier_multiclass_multiclass_incompatible(self) -> None:
        """Test that multiclass with multi_target raises NotImplementedError during suggested_params."""
        import optuna

        model = CatboostClassifierMother(model_type="classification_multiclass", target_type="single_target")
        study = optuna.create_study()
        trial = study.ask()

        # This should work fine for single_target
        suggested_params = model.suggested_params_loss(trial, {}, self.y_multiclass, prefix="")
        self.assertEqual(suggested_params["loss_function"], "MultiClass")
