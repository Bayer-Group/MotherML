import pickle

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from mother.ml.models.m_catboost import (
    CatboostClassifierMother,
)


@pytest.fixture
def synthetic_binary_data():
    """Create synthetic data for binary classification."""
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, n_informative=3, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def synthetic_multiclass_data():
    """Create synthetic data for multiclass classification."""
    X, y = make_classification(n_samples=300, n_features=5, n_classes=3, n_informative=3, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def synthetic_multi_target_data():
    """Create synthetic data for multi-target binary classification."""
    X, y1 = make_classification(n_samples=200, n_features=5, n_classes=2, n_informative=3, random_state=42)
    _, y2 = make_classification(n_samples=200, n_features=5, n_classes=2, n_informative=3, random_state=43)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.DataFrame({"target_1": y1, "target_2": y2})
    return X, y


@pytest.mark.parametrize(
    "data_fixture,model_type,target_type,expected_classes",
    [
        ("synthetic_binary_data", "classification_binary", "single_target", [0, 1]),
        ("synthetic_multiclass_data", "classification_multiclass", "single_target", [0, 1, 2]),
    ],
)
def test_classification_predictions(request, data_fixture, model_type, target_type, expected_classes):
    X, y = request.getfixturevalue(data_fixture)
    model = CatboostClassifierMother(
        target_type=target_type, model_type=model_type, iterations=50, learning_rate=0.1, verbose=0
    )
    model.fit(X, y)
    y_pred = model.predict(X).flatten()
    proba = model.predict_proba(X)
    assert y_pred.shape == y.shape
    assert proba.shape == (len(y), len(expected_classes))
    assert all(pred in expected_classes for pred in y_pred)


def test_focal_loss_binary_classification(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = CatboostClassifierMother(
        target_type="single_target",
        model_type="classification_binary",
        iterations=50,
        learning_rate=0.1,
        verbose=0,
        loss_function="Focal:focal_alpha=0.5;focal_gamma=2.0",
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert all(pred in [0, 1] for pred in y_pred)


@pytest.fixture(params=[5, 10])
def virtual_ensemble_counts(request):
    """Fixture for different virtual ensemble counts."""
    return request.param


@pytest.fixture(params=[1, 2])
def thread_counts(request):
    """Fixture for different thread counts."""
    return request.param


@pytest.fixture(params=[True, False])
def uncertainty_for_opt(request):
    """Fixture for uncertainty for optimization flag."""
    return request.param


@pytest.mark.parametrize("uncertainty_for_opt", [True, False])
def test_uncertainty_estimation(synthetic_binary_data, virtual_ensemble_counts, thread_counts, uncertainty_for_opt):
    X, y = synthetic_binary_data
    model = CatboostClassifierMother(model_type="classification_binary", learning_rate=0.1)
    model.fit(X, y)
    uncertainty_df = model.predict_uncertainty(
        X, n_ensembles=virtual_ensemble_counts, n_threads=thread_counts, uncertainty_for_opt=uncertainty_for_opt
    )
    assert isinstance(uncertainty_df, pd.DataFrame)
    assert len(uncertainty_df) == len(X)
    if uncertainty_for_opt:
        assert len(uncertainty_df.columns) == 1
        assert "knowledge_uncertainty" in uncertainty_df.columns
    else:
        assert {"mean_predictions", "knowledge_uncertainty", "total_uncertainty", "data_uncertainty"}.issubset(
            set(uncertainty_df.columns)
        )


def test_classifier_param_interface():
    model = CatboostClassifierMother()
    model.set_params(target_type="multi_target", tune_boosting_type=True, model_type="classification_multiclass")
    assert model.target_type == "multi_target"
    assert model.tune_boosting_type is True
    assert model.model_type == "classification_multiclass"
    model.set_params(learning_rate=0.1, max_depth=8)
    params = model.get_params()
    assert params["learning_rate"] == pytest.approx(0.1)
    assert params["max_depth"] == 8


def test_classifier_default_parameters():
    model = CatboostClassifierMother()
    default_params = model.default_parameters()
    assert default_params["learning_rate"] == pytest.approx(0.03)
    assert default_params["bootstrap_type"] == "Bayesian"
    assert default_params["random_strength"] == 1
    assert default_params["grow_policy"] == "SymmetricTree"
    assert default_params["boosting_type"] == "Plain"
    assert default_params["max_depth"] == 6
    prefixed_params = model.default_parameters(prefix="test_")
    assert prefixed_params["test_learning_rate"] == pytest.approx(0.03)
    assert prefixed_params["test_max_depth"] == 6


def test_classifier_serialization(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = CatboostClassifierMother(
        target_type="single_target",
        tune_boosting_type=True,
        model_type="classification_binary",
        iterations=50,
        learning_rate=0.1,
        verbose=0,
    )
    model.fit(X, y)
    serialized = pickle.dumps(model)
    deserialized_model = pickle.loads(serialized)
    assert deserialized_model.target_type == model.target_type
    assert deserialized_model.tune_boosting_type == model.tune_boosting_type
    assert deserialized_model.model_type == model.model_type
    y_pred_original = model.predict(X)
    y_pred_deserialized = deserialized_model.predict(X)
    assert (y_pred_original == y_pred_deserialized).all()


def test_suggested_params_loss(synthetic_binary_data):
    _, y = synthetic_binary_data
    model = CatboostClassifierMother(model_type="classification_binary")
    study = optuna.create_study()
    trial = study.ask()
    suggested_params = model.suggested_params_loss(trial, {}, y, prefix="")
    assert suggested_params["loss_function"] in ["Logloss"] or suggested_params["loss_function"].startswith("Focal")
    if suggested_params["loss_function"].startswith("Focal"):
        assert "focal_alpha" in suggested_params["loss_function"]
        assert "focal_gamma" in suggested_params["loss_function"]
        assert suggested_params["auto_class_weights"] == "None"


### CatboostGaussianProcessRegressor


@pytest.mark.parametrize(
    "n_classes, n_targets, model_type, target_type",
    [
        (2, 1, "classification_binary", "single_target"),  # Binary, single-target
        (3, 1, "classification_multiclass", "single_target"),  # Multiclass, single-target
        (2, 2, "classification_binary", "multi_target"),  # Binary, multi-target (simulate)
        (3, 2, "classification_multiclass", "multi_target"),  # Multiclass, multi-target (simulate)
    ],
)
def test_classification_uncertainty_shapes(n_classes, n_targets, model_type, target_type):
    # Generate synthetic data
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42,
    )
    rng = np.random.default_rng(42)
    X = pd.DataFrame(X)
    if n_targets == 1:
        y_df = pd.Series(y)
    else:
        # Simulate multi-target by stacking different random targets
        y_df = pd.DataFrame(
            {f"target_{i}": rng.integers(0, n_classes, size=X.shape[0], dtype=np.int64) for i in range(n_targets)}
        )

    if model_type == "classification_multiclass" and target_type == "multi_target":
        with pytest.raises(NotImplementedError):
            CatboostClassifierMother(model_type=model_type, target_type=target_type)
    else:
        model = CatboostClassifierMother(model_type=model_type, target_type=target_type)
        model.fit(X, y_df)
        uncertainty_df = model.predict_uncertainty(X)

        expected_cols = {"mean_predictions", "knowledge_uncertainty", "data_uncertainty", "total_uncertainty"}
        assert expected_cols.issubset(set(uncertainty_df.columns)), (
            f"Missing columns: {expected_cols - set(uncertainty_df.columns)}"
        )
        assert len(uncertainty_df) == len(X)
        if n_targets > 1:
            for col in expected_cols:
                assert any(col in c for c in uncertainty_df.columns), f"Column {col} missing for multi-target"


@pytest.mark.usefixtures("synthetic_multiclass_data")
def test_uncertainty_warns_for_multciclass(synthetic_multiclass_data, caplog):
    X, y = synthetic_multiclass_data
    model = CatboostClassifierMother(
        model_type="classification_multiclass", iterations=30, learning_rate=0.1, verbose=0
    )
    model.fit(X, y)
    caplog.clear()

    with caplog.at_level("WARNING"):
        _ = model.predict_uncertainty(X, n_ensembles=5, n_threads=1)

    messages = " ".join(r.message for r in caplog.records)
    assert "Uncertainty prediction" in messages and ("MULTICLASS" in messages or "multiclass" in messages)


@pytest.mark.usefixtures("synthetic_multi_target_data")
def test_uncertainty_warns_for_multi_target(synthetic_multi_target_data, caplog):
    X, y = synthetic_multi_target_data
    model = CatboostClassifierMother(
        target_type="multi_target", model_type="classification_binary", iterations=30, learning_rate=0.1, verbose=False
    )
    model.fit(X, y)

    caplog.clear()

    with caplog.at_level("WARNING"):
        _ = model.predict_uncertainty(X, n_ensembles=5, n_threads=1)

    messages = " ".join(r.message for r in caplog.records)
    assert "Uncertainty prediction" in messages and ("MULTICLASS" in messages or "multiclass" in messages)


@pytest.mark.usefixtures("synthetic_multiclass_data")
def test_no_warning_when_uncertainty_false(synthetic_multiclass_data, caplog):
    X, y = synthetic_multiclass_data
    model = CatboostClassifierMother(
        model_type="classification_multiclass", iterations=30, learning_rate=0.1, verbose=0
    )
    model.fit(X, y)
    caplog.clear()

    with caplog.at_level("WARNING"):
        _ = model.predict_uncertainty(X)

    assert any(r.levelname == "WARNING" for r in caplog.records)


@pytest.mark.parametrize("uncertainty_for_opt", [True, False])
def test_uncertainty_output_shape_multiclass(synthetic_multiclass_data, uncertainty_for_opt):
    X, y = synthetic_multiclass_data
    model = CatboostClassifierMother(
        model_type="classification_multiclass", iterations=30, learning_rate=0.1, verbose=0
    )
    model.fit(X, y)
    df = model.predict_uncertainty(X, uncertainty_for_opt=uncertainty_for_opt, n_ensembles=5, n_threads=1)

    assert isinstance(df, pd.DataFrame)
    if uncertainty_for_opt:
        assert list(df.columns) == ["knowledge_uncertainty"]
    else:
        assert {"mean_predictions", "knowledge_uncertainty", "total_uncertainty", "data_uncertainty"} <= set(df.columns)
    assert len(df) == len(X)
