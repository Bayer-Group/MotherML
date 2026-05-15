import itertools
import logging

import numpy as np
import pandas as pd
import pytest
import sklearn.base as skl_base
from pydantic import ValidationError
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from mother import ml
from mother.ml import config as ml_conf
from mother.ml.models.m_catboost import (
    CatboostClassifierMother,
    CatboostRankerMother,
    CatboostRegressorMother,
)


@pytest.fixture(params=ml.get_available_algorithms())
def all_classification_algorithms(request):
    algorithm = request.param
    model = CatboostClassifierMother(target_type="single_target")

    if algorithm == "randomforest":
        from mother.ml.models.m_randomForest import RandomForestClassifierMother

        model = RandomForestClassifierMother()
    elif algorithm == "tabpfn":
        from mother.ml.models.m_tabpfn import TabPFNClassifierMother

        model = TabPFNClassifierMother()
    elif algorithm == "lasso":
        from mother.ml.models.m_lasso import LassoClassifierBinaryMother

        model = LassoClassifierBinaryMother()
    else:
        assert algorithm == "catboost"
    return model


@pytest.fixture(params=ml.get_available_algorithms())
def all_regression_algorithms(request):
    algorithm = request.param
    model = CatboostRegressorMother(target_type="single_target")

    if algorithm == "randomforest":
        from mother.ml.models.m_randomForest import RandomForestRegressorMother

        model = RandomForestRegressorMother()
    elif algorithm == "tabpfn":
        from mother.ml.models.m_tabpfn import TabPFNRegressorMother

        model = TabPFNRegressorMother()
    elif algorithm == "lasso":
        from mother.ml.models.m_lasso import LassoRegressorMother

        model = LassoRegressorMother()
    else:
        assert algorithm == "catboost"
    return model


@pytest.fixture()
def ml_config() -> ml_conf.ModelConfig:
    return ml_conf.ModelConfig(
        categorical_features=["foo", "bar"],
        model_type="regression",
        target_type="single_target",
        feature_selection_type="catboost",
        algorithm="catboost",
        parameters={"iterations": 100, "learning_rate": 0.1},
        feature_selection_flags=[],
    )


def test_clone_works():
    target_type = "multi_target"
    model = CatboostClassifierMother(
        target_type=target_type, model_type="classification_binary", max_depth=2, num_trees=10
    )
    cloned_model = skl_base.clone(model)  # type: ignore
    assert model.target_type == cloned_model.target_type
    assert model.model_type == cloned_model.model_type


@pytest.mark.serial
def test_ml_model(ml_model, modeling_data):
    features, _, target, _, ranking_groups, _ = modeling_data

    if isinstance(ml_model, CatboostRankerMother):
        ml_model.fit(features, target, group_id=ranking_groups)
    else:
        ml_model.fit(features, target)

    predictions = ml_model.predict(features)

    assert len(predictions) == len(target)


@pytest.mark.slow
@pytest.mark.serial
def test_feature_selection(feature_selector, modeling_data):
    features, _, target, _, ranking_groups, _ = modeling_data
    selected_features = feature_selector.fit_transform(features, target)

    assert len(selected_features) == len(target)
    assert selected_features.shape[1] <= features.shape[1]


@pytest.mark.slow
@pytest.mark.serial
def test_pipeline(model_pipeline, modeling_data):
    features, categoric_output, target, categorical_features, ranking_groups, _ = modeling_data

    if isinstance(model_pipeline["model"], CatboostRankerMother):
        model_pipeline.fit(features, target, group_id=ranking_groups)
    else:
        model_pipeline.fit(features, target)
    predictions = model_pipeline.predict(features)

    assert len(predictions) == len(target)


def test_ml_config_raises() -> None:
    with pytest.raises(ValidationError):
        ml_conf.ModelConfig()  # type: ignore


def test_ml_config(ml_config) -> None:
    # test ml config using required fields
    assert ml_config


@pytest.mark.parametrize(
    "algorithm, model_type",
    list(
        itertools.product(
            ml.get_available_algorithms(),
            ["classification_binary", "classification_multiclass", "regression", "ranking"],
        )
    ),
)
def test_model_class_from_config(algorithm, model_type) -> None:
    # Ensure the model class can be instantiated from the config
    if (model_type == "ranking") and (algorithm != "catboost"):
        pytest.skip("ranking is only available in catboost")
    model_class = ml.get_model_class_by_algorithm_and_type(algorithm, model_type)
    assert model_class is not None
    assert issubclass(model_class, ml.AbstractMotherPipeline)  # type: ignore


def test_predict_uncertainty_classification(all_classification_algorithms):
    """
    Test if all available classification algorithms return the same format
    of dataframe with predict_uncertainty()
    """
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    all_classification_algorithms.fit(X_train, y_train)
    pred = all_classification_algorithms.predict_uncertainty(X_test)

    assert all(
        [
            "mean_predictions" in pred.columns,
            "knowledge_uncertainty" in pred.columns,
            "data_uncertainty" in pred.columns,
            "total_uncertainty" in pred.columns,
        ]
    )
    assert pred["total_uncertainty"].notna().all(), "total_uncertainty should be populated for classifiers"


def test_predict_uncertainty_regression(all_regression_algorithms):
    """
    Test if all available regression algorithms return the same format
    of dataframe with predict_uncertainty()
    """
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    all_regression_algorithms.fit(X_train, y_train)
    pred = all_regression_algorithms.predict_uncertainty(X_test)

    required_cols = {
        "pred",
        "mean_predictions",
        "knowledge_uncertainty",
        "data_uncertainty",
        "total_uncertainty",
    }
    missing_cols = required_cols - set(pred.columns)

    assert isinstance(pred, pd.DataFrame)
    assert len(pred) == len(X_test)
    assert not missing_cols, f"Missing required regression uncertainty columns: {sorted(missing_cols)}"


def test_predict_uncertainty_multitarget_with_dataframe():
    """Test multi-target predict_uncertainty always uses standardized target_N names"""
    generator = np.random.default_rng(42)
    X = pd.DataFrame(generator.random((100, 10)), columns=[f"feat_{i}" for i in range(10)])
    # Use custom column names in y - but output should still use target_0, target_1, etc.
    y = pd.DataFrame(generator.random((100, 3)), columns=["activity_A", "activity_B", "activity_C"])

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatboostRegressorMother(target_type="multi_target", iterations=10, verbose=0)
    model.fit(X_train, y_train)

    result = model.predict_uncertainty(X_test)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(X_test)
    expected_cols = {
        "target_0_pred",
        "target_1_pred",
        "target_2_pred",
        "target_0_mean_predictions",
        "target_1_mean_predictions",
        "target_2_mean_predictions",
        "target_0_knowledge_uncertainty",
        "target_1_knowledge_uncertainty",
        "target_2_knowledge_uncertainty",
    }
    assert expected_cols == set(result.columns)
    assert result.shape == (len(X_test), len(expected_cols))


def test_predict_uncertainty_multitarget_preserves_index():
    """Test multi-target predict_uncertainty preserves input DataFrame index"""
    generator = np.random.default_rng(42)
    X = pd.DataFrame(generator.random((100, 10)), index=[f"sample_{i}" for i in range(100)])
    y = pd.DataFrame(generator.random((100, 2)), columns=["custom_A", "custom_B"], index=X.index)

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatboostRegressorMother(target_type="multi_target", iterations=10, verbose=0)
    model.fit(X_train, y_train)

    result = model.predict_uncertainty(X_test)

    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(X_test.index)
    expected_cols = {
        "target_0_pred",
        "target_1_pred",
        "target_0_mean_predictions",
        "target_1_mean_predictions",
        "target_0_knowledge_uncertainty",
        "target_1_knowledge_uncertainty",
    }
    assert expected_cols == set(result.columns)
    assert result.shape[1] == len(expected_cols)


# ---------------------------------------------------------------------------
# Issue #444 — point 1: multi-output fallback in AbstractMotherPipeline
# ---------------------------------------------------------------------------


class _MinimalMultiOutputRegressor(ml.AbstractMotherPipeline):
    """Minimal concrete subclass for testing AbstractMotherPipeline multi-target fallback."""

    _estimator_type = "regressor"

    def fit(self, X, y=None):
        self._n_targets = y.shape[1] if (hasattr(y, "shape") and np.ndim(y) > 1) else 1
        return self

    def predict(self, X):
        rng = np.random.default_rng(0)
        return rng.random((len(X), self._n_targets))

    def get_hyperparameter_space(self, X, y, trial, prefix=""):
        return {}

    def set_params(self, **params):
        return self

    def get_params(self, deep=True):
        return {}


def test_predict_uncertainty_fallback_multitarget_schema():
    """AbstractMotherPipeline fallback produces per-target columns for multi-output predict()."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((50, 5)), columns=[f"f_{i}" for i in range(5)])
    y = pd.DataFrame(rng.random((50, 3)), columns=["a", "b", "c"])

    model = _MinimalMultiOutputRegressor()
    model.fit(X, y)
    result = model.predict_uncertainty(X)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(X)
    expected_cols = {
        "target_0_pred",
        "target_1_pred",
        "target_2_pred",
        "target_0_mean_predictions",
        "target_1_mean_predictions",
        "target_2_mean_predictions",
        "target_0_knowledge_uncertainty",
        "target_1_knowledge_uncertainty",
        "target_2_knowledge_uncertainty",
        "target_0_data_uncertainty",
        "target_1_data_uncertainty",
        "target_2_data_uncertainty",
        "target_0_total_uncertainty",
        "target_1_total_uncertainty",
        "target_2_total_uncertainty",
    }
    assert expected_cols == set(result.columns), f"Unexpected columns: {set(result.columns) ^ expected_cols}"


def test_predict_uncertainty_fallback_multitarget_preserves_index():
    """AbstractMotherPipeline fallback preserves the input DataFrame index for multi-output."""
    rng = np.random.default_rng(7)
    idx = [f"row_{i}" for i in range(30)]
    X = pd.DataFrame(rng.random((30, 4)), index=idx)
    y = pd.DataFrame(rng.random((30, 2)), index=idx)

    model = _MinimalMultiOutputRegressor()
    model.fit(X, y)
    result = model.predict_uncertainty(X)

    assert result.index.tolist() == idx


# ---------------------------------------------------------------------------
# Issue #444 — point 2: uncertainty_for_opt logs a warning for multi-task
# ---------------------------------------------------------------------------


def test_catboost_regressor_uncertainty_for_opt_multitarget_warns(caplog):
    """CatboostRegressorMother logs a warning and returns total_uncertainty for multi-target uncertainty_for_opt."""

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((60, 5)), columns=[f"f_{i}" for i in range(5)])
    y = pd.DataFrame(rng.random((60, 2)), columns=["t0", "t1"])
    X_train, X_test = X.iloc[:50], X.iloc[50:]

    model = CatboostRegressorMother(target_type="multi_target", iterations=5, verbose=0)
    model.fit(X_train, y.iloc[:50])

    with caplog.at_level(logging.WARNING, logger="mother.ml.models.m_catboost"):
        result = model.predict_uncertainty(X_test, uncertainty_for_opt=True)

    assert any("max of per-target" in m for m in caplog.messages), (
        f"Expected a 'max of per-target' warning; got: {caplog.messages}"
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["total_uncertainty"]
    assert len(result) == len(X_test)
