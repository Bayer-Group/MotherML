import pathlib as pl

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
from optuna import create_study
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer

import mother.ml.utils
from mother import utils
from mother.ml.models.m_catboost import CatboostClassifierMother
from mother.ml.utils import (
    MotherTransformedTargetRegressor,
    OrdinalLabelBinarizer,
    default_loss_function,
    get_tree_depth,
    mean_absolute_error_multi_na,
    signed_percentiles_independent,
)
from mother.optimization import MotherTuner

repo_dir: pl.Path = pl.Path(__file__).parent.parent.parent


@pytest.mark.parametrize(
    "model_type, target_type, expected_loss",
    [
        ("classification_binary", "single_target", "Logloss"),
        ("classification_binary", "multi_target", "MultiLogloss"),
        ("classification_multiclass", "single_target", "MultiClass"),
        ("regression", "single_target", "RMSE"),
        ("regression", "multi_target", "MultiRMSEWithMissingValues"),
        ("ranking", "single_target", "YetiRank"),
        ("ranking", "multi_target", "YetiRank"),
    ],
)
def test_default_loss_function_supported_cases(model_type, target_type, expected_loss):
    assert default_loss_function(model_type=model_type, target_type=target_type) == expected_loss


@pytest.mark.parametrize(
    "model_type, target_type, expected_error_msg",
    [
        ("classification_multiclass", "multi_target", "Loss function not known"),
        ("unknown_model", "single_target", "Loss function not implemented"),
    ],
)
def test_default_loss_function_unsupported_cases(model_type, target_type, expected_error_msg):
    with pytest.raises(NotImplementedError, match=expected_error_msg):
        default_loss_function(model_type=model_type, target_type=target_type)


@pytest.mark.parametrize(
    "vector, expected",
    [
        # Test with a mix of positive, negative, and zero values
        (np.array([-10, -5, 0, 5, 10]), np.array([-1.0, -0.5, 0.0, 0.5, 1.0])),
        # Test with only positive values
        (np.array([0, 5, 10, 15, 20]), np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
        # Test with only negative values
        (np.array([-20, -15, -10, -5, 0]), np.array([-1.0, -0.75, -0.5, -0.25, 0.0])),
        # Test with zeros only
        (np.array([0, 0, 0]), np.array([0.0, 0.0, 0.0])),
        # Test with a single positive value
        (np.array([10]), np.array([1.0])),
        # Test with a single negative value
        (np.array([-10]), np.array([-1.0])),
        # Test with a mix of large and small values
        (np.array([-1000, -10, 0, 10, 1000]), np.array([-1.0, -0.5, 0.0, 0.5, 1.0])),
        # Test with no negative values
        (np.array([0, 1, 2, 3, 4]), np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
        # Test with no positive values
        (np.array([-4, -3, -2, -1, 0]), np.array([-1.0, -0.75, -0.5, -0.25, 0.0])),
    ],
)
def test_signed_percentiles_independent(vector, expected):
    result = signed_percentiles_independent(vector)
    np.testing.assert_almost_equal(result, expected, decimal=6)


def test_calc_max_tree_depth():
    data = pd.DataFrame(index=range(16), columns=range(100))
    min_tree_depth, max_tree_depth = mother.ml.utils.calc_range_tree_depth(data, 2, 16)

    assert min_tree_depth == 2
    assert max_tree_depth == 4

    data = pd.DataFrame(index=range(1000), columns=range(100))
    min_tree_depth, max_tree_depth = mother.ml.utils.calc_range_tree_depth(data, 2, 8)

    assert min_tree_depth == 2
    assert max_tree_depth == 8

    min_tree_depth, max_tree_depth = mother.ml.utils.calc_range_tree_depth(data, 3, 2)

    assert min_tree_depth == 2
    assert max_tree_depth == 2


def test_max_depth_to_max_leaves():
    """
    Test if the transformation from tree depth to max leaves for lossguide
    is correctly performed
    """
    leaves_list = mother.ml.utils.depth_to_leaves_for_lossguide(min_depth=2, max_depth=3)
    min_leaves = leaves_list[0]
    max_leaves = leaves_list[1]
    assert min_leaves == 2
    assert max_leaves == 5

    leaves_list = mother.ml.utils.depth_to_leaves_for_lossguide(min_depth=3, max_depth=4)
    min_leaves = leaves_list[0]
    max_leaves = leaves_list[1]
    assert min_leaves == 3
    assert max_leaves == 9

    min_leaves, max_leaves = mother.ml.utils.depth_to_leaves_for_lossguide(min_depth=4, max_depth=12)

    assert min_leaves == 7
    assert max_leaves == 64


def test_multi_mae_with_na():
    val = np.array([[0, 1, 0], [1, 0, 1]])
    pred = np.array([[0, 1, 0], [1, 0, 1]])

    result = mean_absolute_error_multi_na(val, pred)
    assert result == 0

    val = np.array([[0, 1, 1], [1, 0, 1]])
    pred = np.array([[1, 1, 0], [1, 0, 1]])

    result = mean_absolute_error_multi_na(val, pred)
    assert result == np.mean(np.nanmean(np.abs(val - pred), axis=0))

    val = np.array([[0, np.nan, 1], [1, 0, 1]])
    pred = np.array([[0, 1, 0], [1, 0, 1]])

    result = mean_absolute_error_multi_na(val, pred)
    assert result == np.mean(np.nanmean(np.abs(val - pred), axis=0))


def test_convert_input_dataframe_single_col():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = utils.convert_input(df, col="a")
    assert result.ndim == 1
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_convert_input_series():
    series = pd.Series([1, 2, 3, 4, 5, 6])
    result = utils.convert_input(series)
    assert result.ndim == 1
    expected = np.array([1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(result, expected)


def test_convert_input_flat_list():
    result = utils.convert_input([10, 20, 30])
    assert result.ndim == 1
    np.testing.assert_array_equal(result, np.array([10, 20, 30]))


def test_convert_input_single_column_dataframe():
    df = pd.DataFrame({"a": [7, 8, 9]})
    result = utils.convert_input(df)
    assert result.ndim == 1
    np.testing.assert_array_equal(result, np.array([7, 8, 9]))


def test_numeric_columns_all_numeric():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = utils.get_numeric_columns(df)
    expected = ["a", "b"]
    assert result == expected


def test_numeric_columns_mixed_types():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]})
    result = utils.get_numeric_columns(df)
    expected = ["a", "c"]
    assert result == expected


def test_numeric_columns_no_numeric():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["a", "b", "c"]})
    result = utils.get_numeric_columns(df)
    expected = []
    assert result == expected


def test_numeric_columns_empty_dataframe():
    df = pd.DataFrame()
    result = utils.get_numeric_columns(df)
    expected = []
    assert result == expected


def test_numeric_columns_with_nan():
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
    result = utils.get_numeric_columns(df)
    expected = ["a", "b"]
    assert result == expected


def test_get_categorical_column_names_all_categorical():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": [True, False, True], "c": pd.Categorical(["cat1", "cat2", "cat3"])})
    result = utils.get_categorical_column_names(df)
    expected = ["a", "b", "c"]
    assert result == expected


def test_get_categorical_column_names_mixed_types():
    df = pd.DataFrame(
        {
            "a": ["x", "y", "z"],
            "b": [1, 2, 3],
            "c": [True, False, True],
            "d": pd.Categorical(["cat1", "cat2", "cat3"]),
            "e": [4.0, 5.0, 6.0],
        }
    )
    result = utils.get_categorical_column_names(df)
    expected = ["a", "c", "d"]
    assert result == expected


def test_get_categorical_column_names_no_categorical():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = utils.get_categorical_column_names(df)
    expected = []
    assert result == expected


def test_get_categorical_column_names_empty_dataframe():
    df = pd.DataFrame()
    result = utils.get_categorical_column_names(df)
    expected = []
    assert result == expected


def test_get_categorical_column_names_with_nan():
    df = pd.DataFrame(
        {
            "a": ["x", "y", np.nan],
            "b": [True, False, np.nan],
            "c": pd.Categorical(["cat1", "cat2", np.nan]),
            "d": [1, 2, 3],
        }
    )
    result = utils.get_categorical_column_names(df)
    expected = ["a", "b", "c"]
    assert result == expected


@pytest.mark.parametrize(
    "model_class, params, expected_depth",
    [
        (CatBoostRegressor, {}, 6),  # Default regressor
        (CatBoostRegressor, {"depth": 10}, 10),  # Custom depth regressor
        (CatBoostRegressor, {"max_depth": 7}, 7),  # Custom max_depth regressor
        (CatBoostRegressor, {"depth": 4, "max_depth": 9}, 9),  # Both depth and max_depth regressor
        (CatBoostClassifier, {}, 6),  # Default classifier
        (CatBoostClassifier, {"depth": 8}, 8),  # Custom depth classifier
        (CatBoostClassifier, {"max_depth": 11}, 11),  # Custom max_depth classifier
        (CatBoostClassifier, {"depth": 3, "max_depth": 13}, 13),  # Both depth and max_depth classifier
        (CatBoostRanker, {}, 6),  # Default ranker
        (CatBoostRanker, {"depth": 5}, 5),  # Custom depth ranker
        (CatBoostRanker, {"max_depth": 16}, 16),  # Custom max_depth ranker
    ],
)
def test_get_tree_depth(model_class, params, expected_depth):
    """Test get_tree_depth with various CatBoost models and parameters."""
    model = model_class(**params)
    assert get_tree_depth(model) == expected_depth


class TestOrdinalLabelBinarizer:
    """Test class for OrdinalLabelBinarizer."""

    def test_basic_fit_transform(self):
        """Test basic fit and transform functionality."""

        y_train = [1, 2, 3, 4, 5, 2, 3, 1, 4, 5]

        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        # Check that ordinal classes are correct (background class removed)
        assert np.array_equal(binarizer.ordinal_classes_, [2, 3, 4, 5])
        assert np.array_equal(binarizer.all_classes_, [1, 2, 3, 4, 5])

        # Test transform
        y_transformed = binarizer.transform([3])
        assert y_transformed.shape == (1, 4)
        assert np.array_equal(y_transformed[0], [1, 1, 0, 0])  # >= 2, >= 3, < 4, < 5

    def test_sparse_output(self):
        """Test sparse output functionality."""
        from scipy.sparse import issparse

        y_train = [1, 2, 3, 4, 5]

        binarizer = OrdinalLabelBinarizer(sparse_output=True)
        binarizer.fit(y_train)

        y_transformed = binarizer.transform([3])
        assert issparse(y_transformed)

    def test_inverse_transform_perfect_reconstruction(self):
        """Test inverse transform with perfect binary indicators."""

        y_train = [1, 2, 3, 4, 5, 2, 3, 1, 4, 5]

        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        # Create perfect binary indicators
        y_binary = np.array(
            [
                [0, 0, 0, 0],  # Should be class 1 (no thresholds met)
                [1, 0, 0, 0],  # Should be class 2 (>= 2 only)
                [1, 1, 0, 0],  # Should be class 3 (>= 2, >= 3)
                [1, 1, 1, 0],  # Should be class 4 (>= 2, >= 3, >= 4)
                [1, 1, 1, 1],  # Should be class 5 (all thresholds)
            ]
        )

        y_reconstructed = binarizer.inverse_transform(y_binary)
        expected = np.array([1, 2, 3, 4, 5])

        np.testing.assert_array_equal(y_reconstructed, expected)

    def test_inverse_transform_with_imbalanced_classes(self):
        """Test inverse transform with imbalanced class distribution."""

        # Highly imbalanced: mostly class 1 and 2
        y_train = [1] * 80 + [2] * 15 + [3] * 4 + [4] * 1

        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        # Test with various binary predictions
        y_binary = np.array(
            [
                [0, 0, 0],  # Class 1
                [1, 0, 0],  # Class 2
                [1, 1, 0],  # Class 3
                [1, 1, 1],  # Class 4
            ]
        )

        y_reconstructed = binarizer.inverse_transform(y_binary)
        expected = np.array([1, 2, 3, 4])

        np.testing.assert_array_equal(y_reconstructed, expected)

    def test_inverse_transform_with_probabilities(self):
        """Test inverse transform with probability-like values."""

        y_train = [1, 2, 3, 4, 5] * 20  # Balanced classes

        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        # Test with probability-like values that should be thresholded
        y_proba = np.array(
            [
                [0.1, 0.05, 0.02, 0.01],  # Low probabilities -> likely class 1
                [0.9, 0.7, 0.3, 0.1],  # High probabilities -> likely higher class
            ]
        )

        y_reconstructed = binarizer.inverse_transform(y_proba)

        # Should reconstruct to valid ordinal classes
        assert all(pred in binarizer.all_classes_ for pred in y_reconstructed)

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method."""

        y_train = [1, 2, 3, 4, 5]

        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        feature_names = binarizer.get_feature_names_out()
        expected_names = ["ordinal_2", "ordinal_3", "ordinal_4", "ordinal_5"]

        assert list(feature_names) == expected_names

    def test_single_class_edge_case(self):
        """Test edge case with only one class - should raise ValueError."""

        y_train = [5, 5, 5]

        binarizer = OrdinalLabelBinarizer()

        # Should raise ValueError for single class
        with pytest.raises(ValueError, match="Ordinal encoding requires at least 2 classes"):
            binarizer.fit(y_train)

    def test_empty_input_raises_error(self):
        """Test that empty input raises appropriate error."""

        binarizer = OrdinalLabelBinarizer()

        with pytest.raises(ValueError):
            binarizer.fit([])

    def test_custom_ordinal_scale(self):
        """Test with custom ordinal scale (not 1-5)."""

        # Custom scale: 10, 20, 30, 40
        y_train = [10, 20, 30, 40, 20, 30, 10, 40]

        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        assert np.array_equal(binarizer.ordinal_classes_, [20, 30, 40])
        assert np.array_equal(binarizer.all_classes_, [10, 20, 30, 40])

        # Test transform
        y_transformed = binarizer.transform([25])  # Between 20 and 30
        assert np.array_equal(y_transformed[0], [1, 0, 0])  # >= 20, < 30, < 40

    def test_fit_required_before_transform(self):
        """Test that transform requires fit to be called first."""
        # ...existing code...

        binarizer = OrdinalLabelBinarizer()

        with pytest.raises(NotFittedError):
            binarizer.transform([1, 2, 3])

    def test_fit_required_before_inverse_transform(self):
        """Test that inverse_transform requires fit to be called first."""
        # ...existing code...

        binarizer = OrdinalLabelBinarizer()

        with pytest.raises(NotFittedError):
            binarizer.inverse_transform([[1, 0, 0]])

    @pytest.mark.parametrize("sparse_output", [True, False])
    def test_transform_output_format(self, sparse_output):
        """Test that output format respects sparse_output parameter."""
        from scipy.sparse import issparse

        y_train = [1, 2, 3, 4, 5]

        binarizer = OrdinalLabelBinarizer(sparse_output=sparse_output)
        binarizer.fit(y_train)

        y_transformed = binarizer.transform([3])

        if sparse_output:
            assert issparse(y_transformed)
        else:
            assert not issparse(y_transformed)
            assert isinstance(y_transformed, np.ndarray)

    @pytest.mark.parametrize(
        "ordinal_classes",
        [
            [1, 2, 3],  # Skip [1, 2] as it has special background removal behavior
            [1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5],
        ],
    )
    def test_various_ordinal_scales(self, ordinal_classes):
        """Test binarizer with various ordinal scales."""

        # Create training data with all classes
        y_train = ordinal_classes * 10  # Repeat each class 10 times
        rng = np.random.default_rng(42)  # Use seeded generator for reproducibility
        rng.shuffle(y_train)

        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        # Test that all classes are preserved
        assert set(binarizer.all_classes_) == set(ordinal_classes)

        # Test transform and inverse transform roundtrip for known good cases
        # Skip the background class which gets removed
        test_classes = ordinal_classes[1:]  # Skip background class
        for cls in test_classes:
            y_transformed = binarizer.transform([cls])
            y_reconstructed = binarizer.inverse_transform(y_transformed)
            assert y_reconstructed[0] == cls

    def test_roundtrip_consistency(self):
        """Test that transform -> inverse_transform is consistent."""

        y_train = [1, 2, 3, 4, 5] * 20  # Balanced classes

        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        # Test multiple values
        test_values = [1, 2, 3, 4, 5, 2, 4, 1]

        for val in test_values:
            y_transformed = binarizer.transform([val])
            y_reconstructed = binarizer.inverse_transform(y_transformed)
            assert y_reconstructed[0] == val, f"Failed roundtrip for value {val}"

    def test_catboost_integration_with_multilogloss(self):
        """Test OrdinalLabelBinarizer integration with CatBoost using MultiLogloss."""
        from catboost import CatBoostClassifier

        # Create synthetic data
        rng = np.random.default_rng(42)  # Use modern Generator API
        n_samples = 200
        n_features = 5

        # Create features - some continuous, some categorical
        X = rng.standard_normal((n_samples, n_features))

        # Create ordinal target with multiple classes [1, 2, 3, 4, 5]
        # Higher feature values lead to higher ordinal classes
        feature_sum = X.sum(axis=1)
        y_ordinal = np.digitize(feature_sum, bins=np.percentile(feature_sum, [20, 40, 60, 80])) + 1

        print(f"Original ordinal classes distribution: {np.bincount(y_ordinal)}")

        # Split data
        split_idx = int(0.7 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_ordinal[:split_idx], y_ordinal[split_idx:]

        # Step 1: Transform ordinal labels to multi-target binary
        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        print(f"Ordinal classes after background removal: {binarizer.ordinal_classes_}")
        print(f"All classes stored: {binarizer.all_classes_}")

        # Transform training and test targets
        y_train_binary = binarizer.transform(y_train)

        print(f"Binary target shape: {y_train_binary.shape}")
        print(f"Sample binary targets:\n{y_train_binary[:5]}")

        # Step 2: Train CatBoost with MultiLogloss
        model = CatBoostClassifier(
            loss_function="MultiLogloss", iterations=100, depth=4, learning_rate=0.1, verbose=False, random_seed=42
        )

        # Fit model on binary targets
        model.fit(X_train, y_train_binary)

        # Step 3: Make predictions (get probabilities)
        y_pred_proba = model.predict_proba(X_test)
        print(f"Prediction probabilities shape: {y_pred_proba.shape}")
        print(f"Sample predictions:\n{y_pred_proba[:5]}")

        # Step 4: Inverse transform predictions back to ordinal labels
        y_pred_ordinal = binarizer.inverse_transform(y_pred_proba)

        print(f"Predicted ordinal labels: {y_pred_ordinal[:10]}")
        print(f"True ordinal labels: {y_test[:10].tolist()}")

        # Verify results
        assert len(y_pred_ordinal) == len(y_test), "Prediction length mismatch"
        assert all(pred in binarizer.all_classes_ for pred in y_pred_ordinal), "Invalid predicted classes"

        # Calculate accuracy
        accuracy = np.mean([pred == true for pred, true in zip(y_pred_ordinal, y_test)])
        print(f"Accuracy: {accuracy:.3f}")

        # Should have reasonable accuracy (at least better than random)
        random_accuracy = 1.0 / len(binarizer.all_classes_)
        assert accuracy > random_accuracy, f"Accuracy {accuracy:.3f} should be better than random {random_accuracy:.3f}"

        # Test with hard predictions too
        y_pred_hard = model.predict(X_test)
        y_pred_ordinal_hard = binarizer.inverse_transform(y_pred_hard)

        accuracy_hard = np.mean([pred == true for pred, true in zip(y_pred_ordinal_hard, y_test)])
        print(f"Hard prediction accuracy: {accuracy_hard:.3f}")

        assert accuracy_hard > random_accuracy, "Hard predictions should also be better than random"

        print("✅ CatBoost integration test passed successfully!")

    @pytest.mark.parametrize(
        "regressor_type,base_estimator_factory,test_mother_features",
        [
            (
                "sklearn_standard",
                lambda: CatBoostClassifier(
                    loss_function="MultiLogloss",
                    iterations=100,
                    depth=4,
                    learning_rate=0.1,
                    verbose=False,
                    random_seed=42,
                ),
                False,
            ),
            (
                "mother_enhanced",
                lambda: CatboostClassifierMother(
                    target_type="multi_target",
                    model_type="classification_binary",
                    iterations=100,
                    max_depth=4,  # Use max_depth instead of depth for Mother classes
                    learning_rate=0.1,
                    verbose=False,
                    random_seed=42,
                ),
                True,
            ),
        ],
    )
    def test_transformed_target_regressor_catboost_integration(
        self, regressor_type, base_estimator_factory, test_mother_features
    ):
        """Test OrdinalLabelBinarizer with TransformedTargetRegressor/MotherTransformedTargetRegressor + CatBoost."""

        print(f"\n=== Testing {regressor_type} regressor ===")

        # Create synthetic ordinal data
        rng = np.random.default_rng(42)
        n_samples = 200
        n_features = 5

        # Generate features
        X = rng.standard_normal((n_samples, n_features))

        # Create ordinal target based on feature sum
        feature_sum = X.sum(axis=1)
        y_ordinal = np.digitize(feature_sum, bins=np.percentile(feature_sum, [20, 40, 60, 80])) + 1

        print(f"Original ordinal classes distribution: {np.bincount(y_ordinal)}")

        # Split data
        split_idx = int(0.7 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_ordinal[:split_idx], y_ordinal[split_idx:]

        # Create regressor based on type
        transformer = OrdinalLabelBinarizer()
        base_estimator = base_estimator_factory()

        if test_mother_features:
            # Use MotherTransformedTargetRegressor for enhanced Mother framework integration
            regressor = MotherTransformedTargetRegressor(
                regressor=base_estimator, transformer=transformer, check_inverse=False
            )
            regressor_name = "MotherTransformedTargetRegressor"
        else:
            # Use standard TransformedTargetRegressor
            regressor = TransformedTargetRegressor(
                regressor=base_estimator, transformer=transformer, check_inverse=False
            )
            regressor_name = "TransformedTargetRegressor"

        print(f"Fitting {regressor_name}...")
        regressor.fit(X_train, y_train)

        print("Making predictions...")
        y_pred = regressor.predict(X_test)

        print(f"Predicted ordinal labels (first 10): {y_pred[:10]}")
        print(f"True ordinal labels (first 10): {y_test[:10]}")

        # === Basic functionality tests ===
        assert len(y_pred) == len(y_test), "Prediction length mismatch"

        # Get unique classes from training data
        unique_classes = np.unique(y_train)
        assert all(pred in unique_classes for pred in y_pred), "Invalid predicted classes"

        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy: {accuracy:.3f}")

        # Should have reasonable accuracy (better than random)
        random_accuracy = 1.0 / len(unique_classes)
        assert accuracy > random_accuracy, f"Accuracy {accuracy:.3f} should be better than random {random_accuracy:.3f}"

        # Verify that the transformer was properly fitted
        assert hasattr(regressor.transformer_, "ordinal_classes_"), "Transformer should have ordinal_classes_ attribute"
        assert hasattr(regressor.transformer_, "all_classes_"), "Transformer should have all_classes_ attribute"

        print(f"Transformer ordinal classes: {regressor.transformer_.ordinal_classes_}")
        print(f"Transformer all classes: {regressor.transformer_.all_classes_}")

        # === Mother-specific feature tests (only for MotherTransformedTargetRegressor) ===
        if test_mother_features:
            print("\n=== Testing Mother-specific features ===")

            # Test 1: get_hyperparameter_space() method with Optuna integration
            print("Testing hyperparameter space generation...")
            study = create_study()
            trial = study.ask()

            hyperparameter_space = regressor.get_hyperparameter_space(X_train, y_train, trial, prefix="test_")
            print(f"Hyperparameter space keys: {list(hyperparameter_space.keys())}")

            assert isinstance(hyperparameter_space, dict), "Hyperparameter space should be dict"
            # Should have "regressor__" prefix for delegated parameters
            regressor_params = [k for k in hyperparameter_space.keys() if k.startswith("test_regressor__")]
            assert len(regressor_params) > 0, "Should have regressor-prefixed parameters"

            # Test 2: default_parameters() method
            print("Testing default parameters...")
            default_params = regressor.default_parameters(prefix="default_")
            print(f"Default parameters: {default_params}")

            assert isinstance(default_params, dict), "Default parameters should be dict"
            # Should have "regressor__" prefix for delegated parameters
            default_regressor_params = [k for k in default_params.keys() if k.startswith("default_regressor__")]
            assert len(default_regressor_params) > 0, "Should have default regressor-prefixed parameters"

            # Test 3: Basic sklearn interface methods
            print("Testing basic sklearn interface methods...")

            # Test accessing get_params (should work via inheritance)
            params = regressor.get_params()
            assert isinstance(params, dict), "get_params should return dict"
            assert "regressor" in params, "Should have regressor parameter"
            assert "transformer" in params, "Should have transformer parameter"

            # Test 4: Integration with Mother framework patterns
            print("Testing Mother framework integration...")

            # Verify that regressor was properly fitted and accessible
            assert hasattr(regressor.regressor_, "feature_importances_"), "CatBoost should have feature_importances_"
            feature_importances = regressor.regressor_.feature_importances_
            assert len(feature_importances) == X_train.shape[1], "Feature importances should match feature count"

            # Test 5: Hyperparameter optimization workflow simulation (Mother-style)
            # Use a fast scorer for demonstration
            scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

            # Minimal synthetic data for demonstration (use Generator for reproducibility)
            rng = np.random.default_rng(12345)
            X = rng.standard_normal((30, 3))
            y = rng.integers(1, 5, size=30)

            # Use ordinal labels for demonstration
            binarizer = OrdinalLabelBinarizer()
            binarizer.fit(y)

            # Ensure X and y are pandas DataFrame/Series for CatboostMother compatibility
            X_df = pd.DataFrame(X)
            y_ordinal = pd.Series(y)

            # Create the MotherTransformedTargetRegressor
            mother_reg = MotherTransformedTargetRegressor(
                regressor=CatboostClassifierMother(
                    model_type="classification_binary",
                    target_type="multi_target",
                    iterations=3,
                    verbose=False,
                    random_seed=42,
                ),
                transformer=OrdinalLabelBinarizer(),
                check_inverse=False,
            )

            # Wrap in a PipelineWithHyperparameterRooting for compatibility with MotherTuner.optimize
            from mother.ml import PipelineWithHyperparameterRooting

            mother_pipe = PipelineWithHyperparameterRooting([("reg", mother_reg)])

            # Create the MotherTuner
            mtuner = MotherTuner(scorer=scorer, n_trials_optuna=2, seed=42)

            # Run optimization (on the synthetic data)
            optimized_model = mtuner.optimize(estimator=mother_pipe, X=X_df, y=y_ordinal, cross_validation=cv)

            # Check that the optimized model is fitted and can predict
            y_pred = optimized_model.predict(X)
            assert len(y_pred) == len(y)
            print("Testing hyperparameter optimization workflow...")

            # Only test get_hyperparameter_space if regressor has it
            if hasattr(regressor, "get_hyperparameter_space"):

                def objective(trial):
                    if hasattr(regressor, "get_hyperparameter_space"):
                        suggested_params = regressor.get_hyperparameter_space(X_train, y_train, trial, prefix="opt_")
                        assert len(suggested_params) > 0, "Should get hyperparameter suggestions"
                    return accuracy

                study = create_study(direction="maximize")
                study.optimize(objective, n_trials=2)
                assert len(study.trials) == 2, "Should have completed 2 trials"

            # Test 6: Comparison with standard TransformedTargetRegressor
            print("Comparing with standard TransformedTargetRegressor...")

            # Create equivalent standard regressor using same base estimator
            standard_regressor = TransformedTargetRegressor(
                regressor=base_estimator_factory(),
                transformer=OrdinalLabelBinarizer(),
                check_inverse=False,
            )

            standard_regressor.fit(X_train, y_train)
            y_pred_standard = standard_regressor.predict(X_test)

            accuracy_standard = np.mean(y_pred_standard == y_test)
            print(f"Standard regressor accuracy: {accuracy_standard:.3f}")
            print(f"Mother regressor accuracy: {accuracy:.3f}")

            # Accuracies should be similar (within reasonable tolerance) since same underlying model
            assert abs(accuracy - accuracy_standard) < 0.1, "Accuracies should be similar"

            # But Mother regressor should have additional methods that standard doesn't

            # Only check for absence of methods if they exist
            assert not hasattr(standard_regressor, "get_hyperparameter_space"), (
                "Standard regressor shouldn't have get_hyperparameter_space"
            )
            assert not hasattr(standard_regressor, "default_parameters"), (
                "Standard regressor shouldn't have default_parameters"
            )

        print(f"✅ {regressor_name} integration test passed successfully!")

    def test_strict_inverse_property_comprehensive(self):
        """Test that OrdinalLabelBinarizer satisfies strict inverse property comprehensively.

        This test validates that our transformer is truly strictly inverse using
        sklearn's approach and multiple edge cases.
        """

        # Test with multiple datasets and scales
        test_cases = [
            # Standard ordinal scale
            [1, 2, 3, 4, 5] * 20,
            # Custom scale with gaps
            [10, 20, 30, 40, 50] * 15,
            # Imbalanced distribution
            [1] * 50 + [2] * 30 + [3] * 15 + [4] * 5,
            # Three classes only
            [1, 2, 3] * 20,
            # Large scale
            list(range(1, 11)) * 10,  # Classes 1-10
        ]

        for i, y_data in enumerate(test_cases):
            # Shuffle to remove any order dependencies
            rng = np.random.default_rng(42 + i)
            rng.shuffle(y_data)

            binarizer = OrdinalLabelBinarizer()
            binarizer.fit(y_data)

            # Test 1: Full dataset round-trip (sklearn-style sampling)
            # Use every 10th sample like sklearn does
            step = max(1, len(y_data) // 10)
            y_sample = y_data[::step]

            y_transformed = binarizer.transform(y_sample)
            y_reconstructed = binarizer.inverse_transform(y_transformed)

            # Must be exactly equal for ordinal encoding
            np.testing.assert_array_equal(y_reconstructed, y_sample, err_msg=f"Round-trip failed for test case {i}")

            # Test 2: Individual class reconstruction
            for cls in binarizer.all_classes_:
                y_single = [cls]
                y_trans_single = binarizer.transform(y_single)
                y_recon_single = binarizer.inverse_transform(y_trans_single)
                assert y_recon_single[0] == cls, f"Single class {cls} reconstruction failed"

            # Test 3: Boundary cases with probabilities
            if len(binarizer.ordinal_classes_) > 1:
                # Test with "perfect" binary indicators (should be exactly reconstructible)
                n_ordinal = len(binarizer.ordinal_classes_)

                # Convert to cumulative format expected by our binarizer
                cumulative_indicators = np.zeros((n_ordinal + 1, n_ordinal))
                for j in range(n_ordinal + 1):
                    cumulative_indicators[j, :j] = 1  # Set first j elements to 1

                expected_classes = [binarizer.all_classes_[j] for j in range(len(binarizer.all_classes_))]
                reconstructed_perfect = binarizer.inverse_transform(cumulative_indicators)

                np.testing.assert_array_equal(
                    reconstructed_perfect,
                    expected_classes,
                    err_msg=f"Perfect indicator reconstruction failed for case {i}",
                )

            # Test 4: Verify class distribution preservation
            y_all_transformed = binarizer.transform(y_data)
            y_all_reconstructed = binarizer.inverse_transform(y_all_transformed)

            original_counts = np.bincount(y_data, minlength=max(y_data) + 1)
            reconstructed_counts = np.bincount(y_all_reconstructed, minlength=max(y_data) + 1)

            np.testing.assert_array_equal(
                original_counts, reconstructed_counts, err_msg=f"Class distribution not preserved for case {i}"
            )

        print("✅ Comprehensive strict inverse property test passed!")

    def test_sklearn_compatibility_inverse_check(self):
        """Test compatibility with sklearn's inverse checking mechanism."""
        import warnings

        # ...existing code...

        y_train = [1, 2, 3, 4, 5] * 20

        binarizer = OrdinalLabelBinarizer()
        binarizer.fit(y_train)

        # Test the same way sklearn's _check_inverse_transform does
        def our_transform(y):
            # y comes in as 2D array from sklearn, flatten for our binarizer
            y_flat = y.flatten().astype(int)
            y_binary = binarizer.transform(y_flat)
            return y_binary

        def our_inverse_transform(y_binary):
            # Reconstruct and reshape back to 2D for sklearn
            y_reconstructed = binarizer.inverse_transform(y_binary)
            return y_reconstructed.reshape(-1, 1).astype(float)

        # Create a FunctionTransformer wrapper to test with sklearn's mechanism
        sklearn_wrapper = FunctionTransformer(
            func=our_transform,
            inverse_func=our_inverse_transform,
            check_inverse=True,
            validate=False,  # We handle validation ourselves
        )

        # This should not raise warnings if our inverse is truly strict
        y_array = np.array(y_train).reshape(-1, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # Turn warnings into errors
            sklearn_wrapper.fit(y_array)

        print("✅ sklearn compatibility inverse check passed!")


def test_add_prefix_to_dict_keys_basic():
    from mother.ml.models.utils import add_prefix_to_dict_keys

    d = {"a": 1, "b": 2}
    result = add_prefix_to_dict_keys(d, "pre_")
    assert result == {"pre_a": 1, "pre_b": 2}


def test_add_prefix_to_dict_keys_empty():
    from mother.ml.models.utils import add_prefix_to_dict_keys

    d = {}
    result = add_prefix_to_dict_keys(d, "x_")
    assert result == {}


def test_add_prefix_to_dict_keys_nonstring_keys():
    from mother.ml.models.utils import add_prefix_to_dict_keys

    d = {1: "a", 2: "b"}
    # Should raise TypeError since prefix + key fails if key is not str
    with pytest.raises(TypeError):
        add_prefix_to_dict_keys(d, "num_")


def test_add_prefix_to_dict_keys_empty_prefix():
    from mother.ml.models.utils import add_prefix_to_dict_keys

    d = {"a": 1, "b": 2}
    result = add_prefix_to_dict_keys(d, "")
    assert result == {"a": 1, "b": 2}


def test_add_prefix_to_dict_keys_prefix_special_chars():
    from mother.ml.models.utils import add_prefix_to_dict_keys

    d = {"x": 10}
    result = add_prefix_to_dict_keys(d, "!@#")
    assert result == {"!@#x": 10}
