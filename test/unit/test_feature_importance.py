"""
Test if the feature importance calculation stores the number of input features properly.
And if a feature importance is returned for each feature
"""

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import spearmanr
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, KFold

import mother.pipeline_utils as mother_takes_care
from mother.ml import estimators
from mother.ml.core import PipelineWithHyperparameterRooting
from mother.ml.estimators import (
    MotherBorutaPy,
    MotherCatboostImportance,
    MotherPermutationImportance,
    MotherSelectFromModel,
)
from mother.optimization import MotherTuner
from mother.settings import MotherSettings


@pytest.mark.slow
def test_estimator_feature_importance(cv):
    n_features = 100
    x, y = make_blobs(n_samples=100, centers=5, n_features=n_features)
    y_multitask = pd.DataFrame({"Column1": y, "Column2": y})
    y_multitask.iloc[1, 0] = np.nan
    y_multitask.iloc[5, 1] = np.nan
    y_multitask = y_multitask.to_numpy()

    model_regressor = CatBoostRegressor()
    model_regressor_multitask = CatBoostRegressor(loss_function="MultiRMSEWithMissingValues", thread_count=1)

    importancer_permutation_regression = MotherPermutationImportance(model_regressor, KFold(shuffle=True))
    importancer_permutation_regression_multitask = MotherPermutationImportance(
        model_regressor_multitask, KFold(shuffle=True)
    )
    importancer_catboost_regression = MotherCatboostImportance(model_regressor)
    importancer_catboost_regression_multitask = MotherCatboostImportance(model_regressor_multitask)

    importancer_permutation_regression.fit(x, y)
    importancer_permutation_regression_multitask.fit(x, y_multitask)
    importancer_catboost_regression.fit(x, y)
    importancer_catboost_regression_multitask.fit(x, y_multitask)

    permutation_vs_catboost = spearmanr(
        importancer_permutation_regression.feature_importances_,
        importancer_catboost_regression_multitask.feature_importances_,
    )
    permutation_vs_multitask_permutation = spearmanr(
        importancer_permutation_regression.feature_importances_,
        importancer_permutation_regression_multitask.feature_importances_,
    )
    catboost_vs_multitask_permutation = spearmanr(
        importancer_catboost_regression.feature_importances_,
        importancer_permutation_regression_multitask.feature_importances_,
    )
    multitask_catboost_vs_multitask_permutation = spearmanr(
        importancer_catboost_regression_multitask.feature_importances_,
        importancer_permutation_regression_multitask.feature_importances_,
    )

    importances_regression = importancer_permutation_regression.feature_importances_
    assert importances_regression.__class__ is np.ndarray
    assert len(importances_regression == n_features)
    assert np.invert(np.isnan(importances_regression)).all()
    assert np.std(importances_regression) > 0
    assert np.max(importances_regression) > 0

    assert permutation_vs_catboost[1] <= 0.06
    assert permutation_vs_multitask_permutation[1] <= 0.06
    assert catboost_vs_multitask_permutation[1] <= 0.06
    assert multitask_catboost_vs_multitask_permutation[1] <= 0.06


class TestFeatureImportance:
    def test_feature_importance(self, cv_strategy):
        X, y = datasets.load_diabetes(return_X_y=True)

        # Initialize CatBoostRegressor
        model = CatBoostRegressor(iterations=2, learning_rate=1, max_depth=2)

        # check if it works if a user defined Kfold is being passed
        estimator_permutation = estimators.MotherPermutationImportance(model, cv=cv_strategy)

        estimator_permutation.fit(X, y)

        # check if the number of input features was correctly set
        np.testing.assert_equal(estimator_permutation.n_features_in_, X.shape[1])

        # check if the number of calculated feature importances is correct
        np.testing.assert_equal(estimator_permutation.feature_importances_.shape[0], X.shape[1])

        estimator_catboost = estimators.MotherCatboostImportance(model)

        estimator_catboost.fit(X, y)

        # check if the number of input features was correctly set
        np.testing.assert_equal(estimator_catboost.n_features_in_, X.shape[1])

        # check if the number of calculated feature importances is correct
        np.testing.assert_equal(estimator_catboost.feature_importances_.shape[0], X.shape[1])


@pytest.fixture
def sample_data():
    """
    Generate a sample dataset for testing.
    """
    X, y = make_classification(n_samples=100, n_features=10, random_state=42, n_informative=5)
    # Create DataFrame with proper feature names to avoid sklearn warnings
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y)


@pytest.mark.slow
def test_mother_permutation_importance_percentile(sample_data):
    """
    Test if the percentile parameter in MotherPermutationImportance is respected.
    """
    X, y = sample_data

    # Test with percentile=True
    estimator = CatBoostClassifier(random_seed=42)
    mpi = MotherPermutationImportance(estimator=estimator, percentiles=True)
    mpi.fit(X, y)
    feature_importances = mpi.feature_importances_

    # Ensure feature importances are in the range [-1, 1] (percentiles)
    assert np.all(feature_importances >= -1.0)
    assert np.all(feature_importances <= 1.0)
    assert np.max(feature_importances) == 1.0

    # Test with percentile=False
    mpi = MotherPermutationImportance(estimator=estimator, percentiles=False)
    mpi.fit(X, y)
    feature_importances = mpi.feature_importances_

    # Ensure feature importances are not constrained to [-1, 1] (raw values)
    assert np.max(feature_importances) != 1.0


def test_mother_catboost_importance_percentile_and_scale(sample_data):
    """
    Test if the percentile and scale parameters in MotherCatboostImportance are respected.
    """
    X, y = sample_data

    # Test with scale=True and percentile=False
    estimator = CatBoostClassifier(iterations=10, verbose=0, random_seed=42)
    mci = MotherCatboostImportance(estimator=estimator, scale=True, percentiles=False)
    mci.fit(X, y)
    feature_importances = mci.feature_importances_

    # Ensure feature importances are scaled (robust_scale should center around 0)
    assert np.allclose(np.median(feature_importances), 0, atol=1e-1)

    # Test with percentile=True and scale=False
    mci = MotherCatboostImportance(estimator=estimator, scale=False, percentiles=True)
    mci.fit(X, y)
    feature_importances = mci.feature_importances_

    # Ensure feature importances are in the range [-1, 1] (percentiles)
    assert np.all(feature_importances >= -1.0)
    assert np.all(feature_importances <= 1.0)
    assert np.max(feature_importances) == 1.0

    # Test with both scale=True and percentile=True
    mci = MotherCatboostImportance(estimator=estimator, scale=True, percentiles=True)
    mci.fit(X, y)
    feature_importances = mci.feature_importances_

    # Ensure feature importances are in the range [-1, 1] (percentiles) after scaling
    assert np.all(feature_importances >= -1.0)
    assert np.all(feature_importances <= 1.0)
    assert np.max(feature_importances) == 1.0


@pytest.fixture
def tuner():
    """
    Create a MotherTuner instance for testing.
    """
    return MotherTuner(make_scorer(accuracy, greater_is_better=True), n_trials_optuna=5)


def test_hyperparameter_and_default_evaluation_with_pipeline(sample_data, tuner):
    """
    Test if the hyperparameters and default parameters for MotherSelectFromModel
    are being evaluated during optimization using PipelineWithHyperparameterRooting.
    """
    X, y = sample_data

    # Define a base estimator
    base_estimator = MotherCatboostImportance(
        CatBoostClassifier(iterations=10, learning_rate=0.03, max_depth=6), n_estimators=10
    )

    # Create a MotherSelectFromModel instance
    selector = MotherSelectFromModel(estimator=base_estimator)

    # Define a pipeline with the selector using PipelineWithHyperparameterRooting
    pipeline = PipelineWithHyperparameterRooting(
        [("feature_selection", selector), ("classifier", RandomForestClassifier(random_state=42))]
    )

    # Retrieve default parameters from the pipeline
    default_parameters = pipeline.default_parameters()

    # Optimize the pipeline using MotherTuner
    tuner.optimize(
        estimator=pipeline,
        X=X,
        y=y,
        cross_validation=KFold(n_splits=5, shuffle=True, random_state=42),
        default_parameters=default_parameters,
    )

    # Check if the default parameters have been evaluated
    evaluated_trials = [trial.params for trial in tuner.study.trials]
    assert any(
        all(evaluated.get(key) == value for key, value in default_parameters.items()) for evaluated in evaluated_trials
    ), "Default parameters were not evaluated during the optimization process."

    # Assert if other paramters are also evaluated
    evaluated_trials = [trial.params for trial in tuner.study.trials]
    assert not all(
        all(evaluated.get(key) == value for key, value in default_parameters.items()) for evaluated in evaluated_trials
    ), "Only the default parameters were evaluated in the optimization process."


def test_init_parameter_passing(sample_data):
    """
    Test if parameters passed via the __init__ method of MotherSelectFromModel
    are correctly passed to the superclass and used.
    """
    X, y = sample_data

    # Create a MotherSelectFromModel instance with a custom threshold
    selector = MotherSelectFromModel(estimator=RandomForestClassifier(random_state=42), threshold=0.3)

    # Fit the selector
    selector.fit(X, y)

    # Check if the threshold was correctly passed to the superclass
    assert selector.threshold == 0.3, "The threshold parameter was not correctly passed to the superclass."


def test_mother_select_from_model_clone_preserves_max_features(sample_data):
    """
    Regression test for Issue #39:
    max_features must be preserved by sklearn.base.clone().
    """
    X, y = sample_data

    selector = MotherSelectFromModel(
        estimator=RandomForestClassifier(random_state=42),
        max_features=3,
    )
    selector.fit(X, y)

    cloned_selector = clone(selector)

    assert cloned_selector.get_params()["max_features"] == 3


@pytest.fixture(params=["catboost", "permutation"])
def importance_estimator_fixture(request):
    """
    Fixture to dynamically choose between MotherCatboostImportance and MotherPermutationImportance.
    """

    # Initialize a CatBoostClassifier
    estimator = CatBoostClassifier(iterations=10, verbose=0, random_seed=42)

    if request.param == "catboost":
        return MotherCatboostImportance(estimator=estimator, scale=True, percentiles=False)
    elif request.param == "permutation":
        return MotherPermutationImportance(estimator=estimator, percentiles=False)


class TestFeatureImportanceEstimatorsForTreeDepth:
    def test_depth_inferred_from_base_estimator(self):
        """Test that depth is correctly inferred from the base estimator."""
        # Create base estimators with explicit max_depth
        base_cat_estimator_max_depth = CatBoostClassifier(iterations=10, verbose=0, max_depth=10)

        # Create feature importance estimators
        cat_imp = MotherCatboostImportance(estimator=base_cat_estimator_max_depth)
        perm_imp = MotherPermutationImportance(estimator=base_cat_estimator_max_depth)

        # Verify depth is correctly inferred from base estimator
        assert cat_imp.get_params()["max_depth"] == 10
        # Now directly access params from the perm_imp object
        assert perm_imp.get_params()["max_depth"] == 10

        # Test with depth instead of max_depth
        base_cat_estimator_depth = CatBoostClassifier(iterations=10, verbose=0, max_depth=8)

        cat_imp_depth = MotherCatboostImportance(estimator=base_cat_estimator_depth)
        perm_imp_depth = MotherPermutationImportance(estimator=base_cat_estimator_depth)

        # Verify depth is correctly inferred
        assert cat_imp_depth.get_params()["max_depth"] == 8
        assert perm_imp_depth.get_params()["max_depth"] == 8

    def test_depth_preserved_on_clone(self):
        """Test that inferred depth is preserved when cloning the feature importance estimators."""
        # Create base estimator with explicit depth
        base_cat_estimator = CatBoostClassifier(iterations=10, verbose=0, max_depth=10)

        # Create feature importance estimators
        cat_imp = MotherCatboostImportance(estimator=base_cat_estimator)
        perm_imp = MotherPermutationImportance(estimator=base_cat_estimator)

        # Clone the estimators
        cloned_cat_imp = clone(cat_imp)
        cloned_perm_imp = clone(perm_imp)

        # Verify depth is preserved after cloning
        assert cloned_cat_imp.get_params()["max_depth"] == 10
        assert cloned_perm_imp.get_params()["max_depth"] == 10

    def test_different_base_estimator_depths(self):
        """Test with different types of CatBoost estimators having different depths."""
        # Create different types of CatBoost estimators
        cat_classifier = CatBoostClassifier(iterations=10, verbose=0, max_depth=7)
        cat_regressor = CatBoostRegressor(iterations=10, verbose=0, max_depth=9)

        # Create feature importance estimators
        cat_imp_classifier = MotherCatboostImportance(estimator=cat_classifier)
        cat_imp_regressor = MotherCatboostImportance(estimator=cat_regressor)
        perm_imp_classifier = MotherPermutationImportance(estimator=cat_classifier)
        perm_imp_regressor = MotherPermutationImportance(estimator=cat_regressor)

        # Verify depths are correctly inferred
        assert cat_imp_classifier.get_params()["max_depth"] == 7
        assert cat_imp_regressor.get_params()["max_depth"] == 9
        assert perm_imp_classifier.get_params()["max_depth"] == 7
        assert perm_imp_regressor.get_params()["max_depth"] == 9


def test_user_provided_max_depth_respected():
    """Test that user-provided max_depth parameter is respected and overwrites the estimator's depth."""
    # Create base estimators with one depth value
    base_cat_estimator = CatBoostClassifier(iterations=10, verbose=0, max_depth=6)

    # Create feature importance estimators with a different max_depth value
    # This should override the max_depth in the base estimator
    cat_imp = MotherCatboostImportance(estimator=base_cat_estimator, max_depth=12)
    perm_imp = MotherPermutationImportance(estimator=base_cat_estimator, max_depth=12)

    # Verify that the max_depth parameter has been overridden in the base estimator too
    assert cat_imp.get_params()["max_depth"] == 12
    assert cat_imp.estimator.get_params()["max_depth"] == 12
    assert perm_imp.get_params()["max_depth"] == 12
    assert perm_imp.estimator.get_params()["max_depth"] == 12

    # Test with depth parameter
    base_cat_estimator_depth = CatBoostClassifier(iterations=10, verbose=0, depth=5)

    # Create feature importance estimators with a different max_depth value
    cat_imp_depth = MotherCatboostImportance(estimator=base_cat_estimator_depth, max_depth=10)
    perm_imp_depth = MotherPermutationImportance(estimator=base_cat_estimator_depth, max_depth=10)

    # Verify that max_depth is set correctly
    assert cat_imp_depth.get_params()["max_depth"] == 10
    assert cat_imp_depth.estimator.get_params()["max_depth"] == 10
    assert perm_imp_depth.get_params()["max_depth"] == 10
    assert perm_imp_depth.estimator.get_params()["max_depth"] == 10

    # Test if we're preserving the behavior when no max_depth is provided
    original_depth = CatBoostClassifier(iterations=10, verbose=0, max_depth=7)
    cat_imp_preserve = MotherCatboostImportance(estimator=original_depth)
    perm_imp_preserve = MotherPermutationImportance(estimator=original_depth)

    # Original depth should be maintained
    assert cat_imp_preserve.get_params()["max_depth"] == 7
    assert cat_imp_preserve.estimator.get_params()["max_depth"] == 7
    assert perm_imp_preserve.get_params()["max_depth"] == 7
    assert perm_imp_preserve.estimator.get_params()["max_depth"] == 7


@pytest.fixture
def simple_data():
    """Create very simple synthetic data for quick tests."""
    # Create a small dataset (5 features, 20 samples)
    rng = np.random.default_rng(42)
    X = rng.random((20, 5))
    y = rng.integers(0, 2, 20)

    # Create column names and DataFrame versions
    feature_names = [f"feature_{i}" for i in range(5)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    return X, y, X_df, y_series


@pytest.fixture
def simple_boruta_dataframe():
    """Create a pre-fit MotherBorutaPy instance that simulates DataFrame input."""
    # Use minimal iterations to speed up tests
    cat_est = CatBoostClassifier(iterations=5, verbose=False)
    importance = MotherCatboostImportance(estimator=cat_est, n_estimators=5)

    # Create Boruta with minimal iterations for quick tests
    boruta = MotherBorutaPy(estimator=importance, n_estimators="auto", max_iter=3, verbose=0, random_state=42)

    # Create support_ and ranking_ to avoid actual fitting
    # This mimics a fit model with 5 features, where features 0 and 2 are selected
    boruta.support_ = np.array([True, False, True, False, False])
    boruta.ranking_ = np.array([1, 3, 1, 2, 2])
    boruta.support_weak_ = np.array([False, False, False, True, False])  # Feature 3 is tentative
    # Set feature_names_in_ to simulate DataFrame input - this is sklearn standard behavior
    boruta.feature_names_in_ = np.array(["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"])

    return boruta


@pytest.fixture
def simple_boruta_numpy():
    """Create a pre-fit MotherBorutaPy instance that simulates numpy array input."""
    # Use minimal iterations to speed up tests
    cat_est = CatBoostClassifier(iterations=5, verbose=False)
    importance = MotherCatboostImportance(estimator=cat_est, n_estimators=5)

    # Create Boruta with minimal iterations for quick tests
    boruta = MotherBorutaPy(estimator=importance, n_estimators="auto", max_iter=3, verbose=0, random_state=42)

    # Create support_ and ranking_ to avoid actual fitting
    # This mimics a fit model with 5 features, where features 0 and 2 are selected
    boruta.support_ = np.array([True, False, True, False, False])
    boruta.ranking_ = np.array([1, 3, 1, 2, 2])
    boruta.support_weak_ = np.array([False, False, False, True, False])  # Feature 3 is tentative
    # NO feature_names_in_ set to simulate numpy array input

    return boruta


# Keep the original fixture for backward compatibility, defaulting to DataFrame behavior
@pytest.fixture
def simple_boruta(simple_boruta_dataframe):
    """Create a pre-fit MotherBorutaPy instance for testing (DataFrame behavior by default)."""
    return simple_boruta_dataframe


def test_mother_boruta_transform_handles_all_input_types(simple_data, simple_boruta):
    """Test that transform handles both DataFrame and numpy inputs correctly with scikit-learn's set_output API."""
    X, _, X_df, _ = simple_data
    boruta = simple_boruta

    # Test 1: DataFrame input with default output (returns numpy array by default)
    # Since boruta was "fitted" with DataFrame (has feature_names_in_), test with DataFrame
    result_df = boruta.transform(X_df)
    assert isinstance(result_df, np.ndarray)
    assert result_df.shape == (20, 2)  # 2 selected features

    # Test 2: Test pandas output with DataFrame input
    boruta.set_output(transform="pandas")
    result_df_pandas = boruta.transform(X_df)
    assert isinstance(result_df_pandas, pd.DataFrame)
    assert result_df_pandas.shape == (20, 2)
    # Verify column names are correct (selected features)
    expected_columns = ["feature_0", "feature_2"]  # Based on support_ mask: [True, False, True, False, False]
    assert list(result_df_pandas.columns) == expected_columns

    # Test 3: Set output back to "default" and verify behavior
    boruta.set_output(transform="default")
    result_df_default = boruta.transform(X_df)
    assert isinstance(result_df_default, np.ndarray)
    assert result_df_default.shape == (20, 2)

    # Test 4: Create a new boruta fitted with numpy array for numpy-specific tests
    cat_est = CatBoostClassifier(iterations=5, verbose=False)
    importance = MotherCatboostImportance(estimator=cat_est, n_estimators=5)
    boruta_numpy = MotherBorutaPy(estimator=importance, n_estimators="auto", max_iter=3, verbose=0, random_state=42)

    # Simulate fitting with numpy array (no feature_names_in_)
    boruta_numpy.support_ = np.array([True, False, True, False, False])
    boruta_numpy.ranking_ = np.array([1, 3, 1, 2, 2])
    boruta_numpy.support_weak_ = np.array([False, False, False, True, False])
    # No feature_names_in_ set (simulates numpy array input)

    # Test numpy array input with numpy-fitted boruta
    result_np = boruta_numpy.transform(X)
    assert isinstance(result_np, np.ndarray)
    assert result_np.shape == (20, 2)

    # Test pandas output with numpy-fitted boruta
    boruta_numpy.set_output(transform="pandas")
    result_np_pandas = boruta_numpy.transform(X)
    assert isinstance(result_np_pandas, pd.DataFrame)
    assert result_np_pandas.shape == (20, 2)

    # Test 5: Test include_tentative parameter
    # Create a new boruta with include_tentative=True
    cat_est_tentative = CatBoostClassifier(iterations=5, verbose=False)
    importance_tentative = MotherCatboostImportance(estimator=cat_est_tentative, n_estimators=5)
    boruta_tentative = MotherBorutaPy(
        estimator=importance_tentative,
        n_estimators="auto",
        max_iter=3,
        verbose=0,
        random_state=42,
        include_tentative=True,  # Include tentative features
    )

    # Set the same support arrays as above
    # support_: [True, False, True, False, False] - features 0 and 2 are confirmed
    # support_weak_: [False, False, False, True, False] - feature 3 is tentative
    boruta_tentative.support_ = np.array([True, False, True, False, False])
    boruta_tentative.ranking_ = np.array([1, 3, 1, 2, 2])
    boruta_tentative.support_weak_ = np.array([False, False, False, True, False])
    # Set feature_names_in_ to simulate DataFrame input
    boruta_tentative.feature_names_in_ = np.array(["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"])

    # With include_tentative=True, should select features 0, 2 (confirmed) AND 3 (tentative)
    result_tentative = boruta_tentative.transform(X_df)
    assert isinstance(result_tentative, np.ndarray)
    assert result_tentative.shape == (20, 3)  # 3 selected features (2 confirmed + 1 tentative)

    # Test pandas output with include_tentative=True
    boruta_tentative.set_output(transform="pandas")
    result_tentative_pandas = boruta_tentative.transform(X_df)
    assert isinstance(result_tentative_pandas, pd.DataFrame)
    assert result_tentative_pandas.shape == (20, 3)
    # Verify column names include both confirmed and tentative features
    expected_columns_tentative = ["feature_0", "feature_2", "feature_3"]  # confirmed: 0,2 + tentative: 3
    assert list(result_tentative_pandas.columns) == expected_columns_tentative

    # Test 6: Compare include_tentative=False vs include_tentative=True behavior
    # Create another boruta with include_tentative=False (default)
    boruta_no_tentative = MotherBorutaPy(
        estimator=importance_tentative,
        n_estimators="auto",
        max_iter=3,
        verbose=0,
        random_state=42,
        include_tentative=False,  # Only confirmed features
    )

    # Set the same support arrays
    boruta_no_tentative.support_ = np.array([True, False, True, False, False])
    boruta_no_tentative.ranking_ = np.array([1, 3, 1, 2, 2])
    boruta_no_tentative.support_weak_ = np.array([False, False, False, True, False])
    boruta_no_tentative.feature_names_in_ = np.array(["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"])

    # With include_tentative=False, should only select features 0, 2 (confirmed)
    boruta_no_tentative.set_output(transform="pandas")
    result_no_tentative = boruta_no_tentative.transform(X_df)
    assert isinstance(result_no_tentative, pd.DataFrame)
    assert result_no_tentative.shape == (20, 2)  # Only 2 confirmed features
    expected_columns_no_tentative = ["feature_0", "feature_2"]  # Only confirmed features
    assert list(result_no_tentative.columns) == expected_columns_no_tentative

    # Verify the difference: tentative version has one more feature
    assert result_tentative_pandas.shape[1] == result_no_tentative.shape[1] + 1
    assert "feature_3" in result_tentative_pandas.columns
    assert "feature_3" not in result_no_tentative.columns


def test_boruta_with_permutation_importance_simple():
    """
    Simple test to verify MotherBorutaPy works with MotherPermutationImportance.
    Just runs and checks if there is output.
    """
    # Create a dataset with clear signal for feature selection
    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=3,
        n_redundant=1,
        class_sep=1.2,
        random_state=42,
    )

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Create MotherPermutationImportance with CatBoost
    base_estimator = CatBoostClassifier(iterations=30, verbose=0, random_seed=42)
    perm_importance = MotherPermutationImportance(
        estimator=base_estimator, percentiles=True, cv=KFold(n_splits=3, shuffle=True, random_state=42)
    )

    # Create MotherBorutaPy with PermutationImportance
    boruta = MotherBorutaPy(
        estimator=perm_importance,
        n_estimators="auto",
        max_iter=8,
        verbose=0,
        random_state=42,
        perc=90,
    )

    # Fit and check basic functionality
    boruta.fit(X_df, y)

    # Basic checks: just verify it runs and produces output
    assert hasattr(boruta, "support_"), "Should have support_ attribute after fitting"

    # Transform and check output
    X_transformed = boruta.transform(X_df)
    assert isinstance(X_transformed, np.ndarray), "Transform should return numpy array"
    assert X_transformed.shape[0] == X_df.shape[0], "Should preserve number of samples"
    assert X_transformed.shape[1] <= X_df.shape[1], "Should reduce or maintain feature count"


def test_get_feature_selection_pipeline_with_mother_boruta(sample_data):
    """
    Test if mother.pipeline_utils.get_feature_selection_pipeline works with MotherBorutaPy
    """

    X, y = sample_data

    # Create cross-validation strategy
    cv = GroupKFold(n_splits=5)

    # Configure settings for feature selection with Boruta
    settings: MotherSettings = MotherSettings.create()
    settings.model.feature_selection_flags = [
        "DROP_CONSTANT",
        "DROP_CORRELATED",
        "DROP_DUPLICATES",
        "DROP_UNIMPORTANT",
    ]

    pipeline = mother_takes_care.get_feature_selection_pipeline(
        settings=settings, data=X, use_boruta=True, cv=cv
    ).set_output(transform="pandas")

    pipeline.fit(X, y)

    # After fitting, transform should return a non-empty DataFrame
    X_transformed = pipeline.transform(X)
    assert isinstance(X_transformed, pd.DataFrame), "Transformed output is not a DataFrame"
    assert not X_transformed.empty, "Transformed DataFrame is empty"
    assert X_transformed.shape[0] == X.shape[0], "Number of rows changed after transform"
    assert X_transformed.shape[1] > 0, "No features were selected by the pipeline"


def test_mother_boruta_raises_if_not_fitted(simple_data):
    """
    Test that MotherBorutaPy raises ValueError if transform is called before fitting (support_ not set).
    """
    X, _, X_df, _ = simple_data
    cat_est = CatBoostClassifier(iterations=5, verbose=False)
    importance = MotherCatboostImportance(estimator=cat_est, n_estimators=5)
    boruta = MotherBorutaPy(estimator=importance, n_estimators="auto", max_iter=3, verbose=0, random_state=42)

    # Should raise NotFittedError for numpy input with check_is_fitted's standard error message
    with pytest.raises(ValueError, match="is not fitted yet"):
        boruta.transform(X)

    # Should raise NotFittedError for DataFrame input with check_is_fitted's standard error message
    with pytest.raises(ValueError, match="is not fitted yet"):
        boruta.transform(X_df)

    # Should raise NotFittedError for get_feature_names_out
    with pytest.raises(ValueError, match="is not fitted yet"):
        boruta.get_feature_names_out()


def test_mother_boruta_sklearn_compatibility():
    """Test that MotherBorutaPy behaves like standard sklearn feature selectors."""

    print("Creating a realistic test dataset...")

    # Create a more realistic test dataset with clear signal
    n_samples = 300
    n_features = 10

    # Use make_classification to create a dataset with known informative features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,  # 4 features are actually informative
        n_redundant=2,  # 2 features are redundant (linear combinations)
        n_clusters_per_class=1,
        class_sep=1.5,  # Good separation between classes
        random_state=42,
    )

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    print(f"Dataset shape: {X_df.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Features: {list(X_df.columns)}")

    print("\nTesting MotherBorutaPy with improved parameters...")

    # Create a more powerful estimator for Boruta
    catboost_estimator = CatBoostClassifier(iterations=100, verbose=0, random_seed=42, depth=6, learning_rate=0.1)
    mother_importance = MotherCatboostImportance(estimator=catboost_estimator, percentiles=True, n_estimators=100)

    # Create MotherBorutaPy with better parameters for feature detection
    boruta = MotherBorutaPy(
        estimator=mother_importance,
        n_estimators=50,  # Increase for better detection
        max_iter=100,  # Increase for thorough search
        verbose=1,  # Enable verbose to see progress
        random_state=42,
        alpha=0.05,  # Standard significance level
        perc=90,  # Use 90th percentile instead of 100th
    )

    boruta.fit(X_df, y)

    # Test sklearn SelectFromModel for comparison
    sklearn_selector = SelectFromModel(RandomForestClassifier(random_state=42))
    sklearn_selector.fit(X_df, y)

    # 1. Both should have feature_names_in_
    assert hasattr(boruta, "feature_names_in_"), "MotherBorutaPy should have feature_names_in_"
    assert hasattr(sklearn_selector, "feature_names_in_"), "SelectFromModel should have feature_names_in_"

    np.testing.assert_array_equal(boruta.feature_names_in_, sklearn_selector.feature_names_in_)

    # 2. Both should support get_feature_names_out
    boruta_names = boruta.get_feature_names_out()
    sklearn_names = sklearn_selector.get_feature_names_out()

    assert isinstance(boruta_names, np.ndarray), "get_feature_names_out should return numpy array"
    assert isinstance(sklearn_names, np.ndarray), "get_feature_names_out should return numpy array"

    # 3. Test with numpy array input
    boruta_numpy = MotherBorutaPy(
        estimator=mother_importance, n_estimators="auto", max_iter=5, verbose=0, random_state=42
    )
    sklearn_numpy = SelectFromModel(RandomForestClassifier(random_state=42))

    boruta_numpy.fit(X, y)
    sklearn_numpy.fit(X, y)

    # For numpy input, sklearn convention is to NOT set feature_names_in_
    # MotherBorutaPy should follow this convention to avoid warnings
    assert not hasattr(boruta_numpy, "feature_names_in_") or boruta_numpy.feature_names_in_ is None, (
        "Should not have feature_names_in_ for numpy input (sklearn convention)"
    )

    # sklearn behavior varies by version, so let's just test that MotherBorutaPy follows sklearn convention
    # If sklearn has feature_names_in_, then MotherBorutaPy should too. If not, then neither should have it.
    sklearn_has_feature_names = (
        hasattr(sklearn_numpy, "feature_names_in_") and sklearn_numpy.feature_names_in_ is not None
    )
    boruta_has_feature_names = hasattr(boruta_numpy, "feature_names_in_") and boruta_numpy.feature_names_in_ is not None

    # Both should have the same behavior (either both have it or both don't)
    assert sklearn_has_feature_names == boruta_has_feature_names, (
        f"Inconsistent feature_names_in_ behavior: sklearn={sklearn_has_feature_names}, "
        f"boruta={boruta_has_feature_names}"
    )

    # 4. Test cloning behavior
    boruta_clone = clone(boruta)
    sklearn_clone = clone(sklearn_selector)

    # Cloned estimators should not have feature_names_in_ until fitted
    assert not hasattr(boruta_clone, "feature_names_in_") or boruta_clone.feature_names_in_ is None, (
        "Cloned estimator should not have feature_names_in_"
    )
    assert not hasattr(sklearn_clone, "feature_names_in_") or sklearn_clone.feature_names_in_ is None, (
        "Cloned estimator should not have feature_names_in_"
    )

    # 5. Test transform raises appropriate errors before fitting
    with pytest.raises(ValueError, match="is not fitted yet"):
        boruta_clone.transform(X_df)

    with pytest.raises(ValueError):
        sklearn_clone.transform(X_df)

    # 6. Test set_output behavior (if available)
    try:
        # Test default transform output (should be numpy)
        boruta_transform = boruta.transform(X_df)
        sklearn_transform = sklearn_selector.transform(X_df)

        assert isinstance(boruta_transform, np.ndarray), "Default transform should return numpy array"
        assert isinstance(sklearn_transform, np.ndarray), "Default transform should return numpy array"

        # Test pandas output
        boruta.set_output(transform="pandas")
        sklearn_selector.set_output(transform="pandas")

        boruta_df_transform = boruta.transform(X_df)
        sklearn_df_transform = sklearn_selector.transform(X_df)

        assert isinstance(boruta_df_transform, pd.DataFrame), "set_output('pandas') should return DataFrame"
        assert isinstance(sklearn_df_transform, pd.DataFrame), "set_output('pandas') should return DataFrame"

        # Check that column names are preserved correctly
        if len(boruta_df_transform.columns) > 0:
            assert all(col in feature_names for col in boruta_df_transform.columns), (
                "Column names should be from original features"
            )

    except AttributeError:
        # set_output might not be available in older sklearn versions
        pass
