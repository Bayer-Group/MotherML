import pickle
import unittest
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import clone, is_regressor
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import KFold

import mother.optimization as opt
from mother.ml.models.m_catboost import (
    CatboostGaussianProcessRegressorMother,
    CatboostRegressorMother,
)


class TestCatboostModels(unittest.TestCase):
    def setUp(self) -> None:
        # Create synthetic data using make_blobs
        n_features = 5
        n_samples = 100

        # Generate synthetic data for regression
        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
        self.X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        self.y_regression = pd.Series(y, name="target")

        # Generate multi-target regression data
        self.y_multitarget_regression = pd.DataFrame({"0": y, "1": y})

        # Define quantiles for quantile regression tests
        self.quantiles: List[float] = [0.05, 0.5, 0.95]

    def _test_optim(self, model):
        cv = KFold(5)
        scorer = make_scorer(score_func=root_mean_squared_error, greater_is_better=False)

        # n_threads_optuna is the number of parallel threads for hyperparameter optimization
        # so it should not be higher then the number of cross-validation folds
        tuner = opt.MotherTuner(
            n_threads_optuna=5,
            scorer=scorer,
            n_trials_optuna=1,
            n_startup_trials=1,
            tuning_direction="maximize",
        )

        tuner.optimize(
            model,
            self.X,
            self.y_regression,
            cv,
            hyperparameter_space_function=model.get_hyperparameter_space,
            default_parameters=model.default_parameters(),
        )

    def test_CatboostRegressorMother_single_target_predict_with_uncertainty(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(target_type="single_target")
        model.fit(self.X, self.y_regression)
        result: pd.DataFrame = model.predict_uncertainty(self.X)
        self.assertIsInstance(result, pd.DataFrame)

        self.assertIn("mean_predictions", result.columns)
        self.assertIn("knowledge_uncertainty", result.columns)

        self.assertTrue(np.issubdtype(result["mean_predictions"].dtype, np.number))
        self.assertTrue(np.issubdtype(result["knowledge_uncertainty"].dtype, np.number))

        correlation = result["mean_predictions"].corr(self.y_regression)
        self.assertGreater(correlation, 0.8, "Mean predictions are not sufficiently correlated with actual values.")

        self.assertTrue((result["knowledge_uncertainty"] >= 0).all(), "model uncertainty contains negative values.")

        self._test_optim(model)

    def test_CatboostRegressorMother_single_target_predict_with_uncertainty_for_opt(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(target_type="single_target")
        model.fit(self.X, self.y_regression, verbose=False)
        result: pd.DataFrame = model.predict_uncertainty(self.X, uncertainty_for_opt=True)
        # Get results from predict_with_uncertainty
        result_with_uncertainty: pd.DataFrame = model.predict_uncertainty(self.X)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("knowledge_uncertainty", result.columns)
        self.assertTrue(np.issubdtype(result["knowledge_uncertainty"].dtype, np.number))
        self.assertTrue((result["knowledge_uncertainty"] >= 0).all(), "model uncertainty contains negative values.")
        print(result, result_with_uncertainty)
        # Check if the values of "knowledge_uncertainty" are equal
        pd.testing.assert_series_equal(
            result["knowledge_uncertainty"],
            result_with_uncertainty["knowledge_uncertainty"],
            check_exact=True,
            obj="Model Uncertainty",
        )

    def test_CatboostRegressorMother_multi_target_predict_with_uncertainty(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(target_type="multi_target")
        model.fit(self.X, self.y_multitarget_regression, verbose=False)
        result: pd.DataFrame = model.predict_uncertainty(self.X)
        self.assertIsInstance(result, pd.DataFrame)

        # Dynamically check required columns for each target
        for target in self.y_multitarget_regression.columns:
            self.assertIn(f"{target}_mean_predictions", result.columns)
            self.assertIn(f"{target}_knowledge_uncertainty", result.columns)

        for target in self.y_multitarget_regression.columns:
            self.assertTrue(np.issubdtype(result[f"{target}_mean_predictions"].dtype, np.number))
            self.assertTrue(np.issubdtype(result[f"{target}_knowledge_uncertainty"].dtype, np.number))

        # Check if the shape matches the input
        self.assertEqual(result.shape[0], self.X.shape[0])
        self._test_optim(model)

    def test_CatboostRMSEWithUncertaintyRegression_predict(self):
        model: CatboostRegressorMother = CatboostRegressorMother(
            target_type="single_target", loss_function="RMSEWithUncertainty"
        )
        model.fit(self.X, self.y_regression, verbose=False)
        mean_predictions: np.ndarray = model.predict(self.X)
        self.assertIsInstance(mean_predictions, np.ndarray)
        self.assertEqual(mean_predictions.shape, (self.X.shape[0],))
        self._test_optim(model)

    def test_CatboostRMSEWithUncertaintyRegression_predict_with_uncertainty(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(
            target_type="single_target", loss_function="RMSEWithUncertainty"
        )
        model.fit(self.X, self.y_regression, verbose=False)
        predictions = model.predict_uncertainty(self.X)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertTrue(
            {"knowledge_uncertainty", "data_uncertainty", "total_uncertainty"}.issubset(predictions.columns)
        )
        self.assertEqual(predictions.shape[0], self.X.shape[0])

        self.assertTrue(
            (predictions["knowledge_uncertainty"] >= 0).all(), "model uncertainty contains negative values."
        )
        self.assertTrue((predictions["data_uncertainty"] >= 0).all(), "Data uncertainty contains negative values.")

        self.assertTrue(np.issubdtype(predictions["knowledge_uncertainty"].dtype, np.number))
        self.assertTrue(np.issubdtype(predictions["data_uncertainty"].dtype, np.number))
        self.assertTrue(np.issubdtype(predictions["total_uncertainty"].dtype, np.number))

    def test_CatboostRMSEWithUncertaintyRegression_predict_with_uncertainty_for_opt(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(
            target_type="single_target", loss_function="RMSEWithUncertainty"
        )
        model.fit(self.X, self.y_regression, verbose=False)
        knowledge_uncertainty_df: pd.DataFrame = model.predict_uncertainty(self.X, uncertainty_for_opt=True)
        self.assertIsInstance(knowledge_uncertainty_df, pd.DataFrame)
        self.assertIn("knowledge_uncertainty", knowledge_uncertainty_df.columns)
        self.assertEqual(knowledge_uncertainty_df.shape[0], self.X.shape[0])

        self.assertTrue(np.issubdtype(knowledge_uncertainty_df["knowledge_uncertainty"].dtype, np.number))

        self.assertTrue(
            (knowledge_uncertainty_df["knowledge_uncertainty"] >= 0).all(),
            "model uncertainty contains negative values.",
        )

    def test_CatboostMultiQuantileRegression_predict(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(
            quantiles=self.quantiles.copy(), loss_function=f"MultiQuantile:alpha={', '.join(map(str, self.quantiles))}"
        )
        model.fit(self.X, self.y_regression, verbose=False)
        median: np.ndarray = model.predict(self.X)
        self.assertIsInstance(median, np.ndarray, "Predict function did not return a NumPy array.")
        self.assertEqual(median.shape, (self.X.shape[0],), "Median predictions do not have the correct shape.")

        pred_uncertainty: pd.DataFrame = model.predict_uncertainty(self.X)
        self.assertEqual(pred_uncertainty.shape, (self.X.shape[0], len(self.quantiles)))
        self.assertIsInstance(pred_uncertainty, pd.DataFrame)
        np.testing.assert_array_equal(
            pred_uncertainty["quantile_0.5"].to_numpy(),
            median,
            err_msg="Median predictions do not have the correct value.",
        )
        self._test_optim(model)

    def test_CatboostMultiQuantileRegression_predict_with_uncertainty_for_opt(self) -> None:
        model: CatboostRegressorMother = CatboostRegressorMother(
            quantiles=self.quantiles.copy(),
        )
        model.fit(self.X, self.y_regression, verbose=False)
        result: pd.DataFrame = model.predict_uncertainty(self.X, uncertainty_for_opt=True)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("total_uncertainty", result.columns)
        self.assertTrue(np.issubdtype(result["total_uncertainty"].dtype, np.number))
        self.assertTrue((result["total_uncertainty"] >= 0).all(), "total_uncertainty contains negative values.")

    def test_CatboostGaussianProcessRegressorMother_initialization(self) -> None:
        """Test model initialization and default parameters."""
        model: CatboostGaussianProcessRegressorMother = CatboostGaussianProcessRegressorMother()

        # Test scikit-learn estimator interface
        self.assertTrue(is_regressor(model))

        # Test default parameters
        params = model.get_params()
        self.assertEqual(params["samples"], 10)
        self.assertEqual(params["prior_iterations"], 100)
        self.assertEqual(params["model_type"], "regression")
        self.assertFalse(params["tune_boosting_type"])
        self.assertTrue(params["tune_tree_structure_type"])

    def test_CatboostGaussianProcessRegressorMother_get_set_params(self) -> None:
        """Test parameter getting and setting."""
        model: CatboostGaussianProcessRegressorMother = CatboostGaussianProcessRegressorMother(
            samples=15, learning_rate=0.2, prior_iterations=500, tune_boosting_type=True, tune_tree_structure_type=False
        )
        # Test get_params
        params = model.get_params()
        self.assertEqual(params["samples"], 15)
        self.assertAlmostEqual(params["learning_rate"], 0.2)
        self.assertEqual(params["prior_iterations"], 500)
        self.assertTrue(params["tune_boosting_type"])
        self.assertFalse(params["tune_tree_structure_type"])

        # Test set_params
        model.set_params(samples=20, learning_rate=0.3, prior_iterations=600)
        params = model.get_params()
        self.assertEqual(params["samples"], 20)
        self.assertAlmostEqual(params["learning_rate"], 0.3)
        self.assertEqual(params["prior_iterations"], 600)

        # Test method chaining
        model2 = model.set_params(samples=25)
        self.assertIs(model2, model)
        self.assertEqual(model.get_params()["samples"], 25)

    def test_CatboostGaussianProcessRegressorMother_invalid_model_type(self) -> None:
        """Test that invalid model_type raises ValueError."""
        with self.assertRaises(ValueError):
            CatboostGaussianProcessRegressorMother(model_type="classification")

    def test_CatboostGaussianProcessRegressorMother_fit_predict(self) -> None:
        """Test basic fit and predict functionality."""
        model = CatboostGaussianProcessRegressorMother(prior_iterations=50, samples=5, verbose=False)

        # Test fit returns self
        fit_result = model.fit(self.X, self.y_regression)
        self.assertIs(fit_result, model)

        # Test predict
        predictions = model.predict(self.X)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (len(self.X),))
        self.assertFalse(np.any(np.isnan(predictions)))

        # Test subset prediction
        subset_size = len(self.X) // 2
        subset_pred = model.predict(self.X.iloc[:subset_size])
        self.assertEqual(len(subset_pred), subset_size)

        # Test prediction on modified data
        X_new = self.X * 2
        new_predictions = model.predict(X_new)
        self.assertEqual(len(new_predictions), len(self.X))
        self.assertFalse(np.allclose(predictions, new_predictions))

    def test_CatboostGaussianProcessRegressorMother_predict_uncertainty(self) -> None:
        """Test uncertainty prediction functionality."""
        model = CatboostGaussianProcessRegressorMother(prior_iterations=50, samples=5, verbose=False)
        model.fit(self.X, self.y_regression)

        # Test predict_uncertainty
        results = model.predict_uncertainty(self.X)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("mean_predictions", results.columns)
        self.assertIn("knowledge_uncertainty", results.columns)
        self.assertFalse(np.any(results["knowledge_uncertainty"] < 0))
        self.assertEqual(len(results), len(self.X))

        # Test uncertainty_for_opt=True
        opt_uncertainty = model.predict_uncertainty(self.X, uncertainty_for_opt=True)
        self.assertIsInstance(opt_uncertainty, pd.DataFrame)
        self.assertIn("knowledge_uncertainty", opt_uncertainty.columns)
        self.assertFalse(np.any(np.isnan(opt_uncertainty["knowledge_uncertainty"])))

        # Test that knowledge_uncertainty values are the same
        pd.testing.assert_series_equal(
            results["knowledge_uncertainty"],
            opt_uncertainty["knowledge_uncertainty"],
            check_exact=True,
            obj="Knowledge Uncertainty",
        )

    def test_CatboostGaussianProcessRegressorMother_hyperparameter_optimization(self) -> None:
        """Test hyperparameter optimization using optuna."""
        model = CatboostGaussianProcessRegressorMother(
            prior_iterations=20,  # Reduced for faster testing
            samples=3,
            verbose=False,
        )

        # Test optimization
        self._test_optim(model)

    def test_CatboostGaussianProcessRegressorMother_pickling(self) -> None:
        """Test model serialization and deserialization."""
        model = CatboostGaussianProcessRegressorMother(samples=5, prior_iterations=50, learning_rate=0.1, verbose=False)
        model.fit(self.X, self.y_regression)

        # Get predictions before pickling
        results_before = model.predict_uncertainty(self.X)

        # Test pickling and unpickling
        pickled_model = pickle.dumps(model)
        unpickled_model = pickle.loads(pickled_model)

        # Test that parameters are preserved
        self.assertEqual(unpickled_model.get_params(), model.get_params())

        # Test that predictions are the same
        results_after = unpickled_model.predict_uncertainty(self.X)
        np.testing.assert_array_almost_equal(results_before["mean_predictions"], results_after["mean_predictions"])
        np.testing.assert_array_almost_equal(
            results_before["knowledge_uncertainty"], results_after["knowledge_uncertainty"]
        )

        # Test that unpickled model can be retrained
        unpickled_model.fit(self.X, self.y_regression)
        new_predictions = unpickled_model.predict(self.X)
        self.assertIsInstance(new_predictions, np.ndarray)

    def test_CatboostGaussianProcessRegressorMother_cloning(self) -> None:
        """Test model cloning functionality."""
        model = CatboostGaussianProcessRegressorMother(
            samples=15, prior_iterations=100, learning_rate=0.2, tune_boosting_type=True
        )
        model.fit(self.X, self.y_regression)

        # Test cloning preserves parameters
        original_params = model.get_params()
        cloned_model = clone(model)
        self.assertEqual(cloned_model.get_params(), original_params)

        # Test that cloned model can be modified independently
        cloned_model.set_params(samples=50)
        self.assertEqual(cloned_model.get_params()["samples"], 50)
        self.assertNotEqual(model.get_params()["samples"], 50)

        # Test that cloned model can be fitted independently
        cloned_model.fit(self.X, self.y_regression)
        predictions = cloned_model.predict(self.X)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X))

    def test_CatboostGaussianProcessRegressorMother_state_persistence(self) -> None:
        """Test __getstate__ and __setstate__ methods."""
        model = CatboostGaussianProcessRegressorMother(
            samples=10, prior_iterations=50, tune_boosting_type=True, tune_tree_structure_type=False
        )
        model.fit(self.X, self.y_regression)

        # Test state saving and loading
        state = model.__getstate__()
        new_model = CatboostGaussianProcessRegressorMother()
        new_model.__setstate__(state)

        # Test that state is properly restored
        self.assertEqual(new_model.get_params(), model.get_params())

        # Test that restored model works
        predictions = new_model.predict(self.X)
        self.assertIsInstance(predictions, np.ndarray)
