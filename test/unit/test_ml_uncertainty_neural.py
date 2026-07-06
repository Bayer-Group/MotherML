"""
Uncertainty-interface tests for the optional neural algorithms (NODE + MLP heads).

NODE (``node``) and the MLP/Flow heads (``heads``) depend on non-standard optional
dependencies (skorch, torch, zuko). They are excluded from the generic algorithm
sweep in ``test_ml.py`` and tested here separately so that the core suite stays
runnable without the optional ``node`` extra installed (e.g. in ``dist-test``).

These tests mirror ``test_predict_uncertainty_classification`` /
``test_predict_uncertainty_regression`` from ``test_ml.py`` but for the neural
estimators, asserting they return the same ``predict_uncertainty()`` DataFrame
format as the standard Mother estimators.

All tests are marked ``serial`` (PyTorch multiprocessing safety) and ``slow``
(neural-network training).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

# Skip the entire module when the optional NODE/heads dependencies are absent.
pytest.importorskip("skorch")
pytest.importorskip("torch")

from mother.ml.models.m_heads import MLPHeadClassifier, MLPHeadRegressor  # noqa: E402
from mother.ml.models.m_node import NODEClassifier, NODERegressor  # noqa: E402

# - serial: avoid PyTorch multiprocessing issues under pytest-xdist
# - slow: neural-network training is computationally expensive
pytestmark = [pytest.mark.serial, pytest.mark.slow]

REQUIRED_UNCERTAINTY_COLS = {
    "mean_predictions",
    "knowledge_uncertainty",
    "data_uncertainty",
    "total_uncertainty",
}


def _classification_data():
    """Small breast-cancer split as float32 numpy arrays (skorch-friendly)."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _regression_data():
    """Small diabetes split as float32 numpy arrays (skorch-friendly)."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.mark.parametrize(
    "model_factory",
    [
        pytest.param(
            lambda: NODEClassifier(num_trees=16, max_epochs=3, device="cpu", verbose=0),
            id="node",
        ),
        pytest.param(
            lambda: MLPHeadClassifier(max_epochs=5, device="cpu", verbose=0),
            id="mlp",
        ),
    ],
)
def test_predict_uncertainty_classification_neural(model_factory):
    """Neural classifiers return the standard predict_uncertainty() DataFrame format."""
    X_train, X_test, y_train, _ = _classification_data()
    model = model_factory()
    model.fit(X_train, y_train)
    pred = model.predict_uncertainty(X_test)

    assert isinstance(pred, pd.DataFrame)
    assert len(pred) == len(X_test)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing classification uncertainty columns: {sorted(missing_cols)}"
    assert pred["total_uncertainty"].notna().all(), "total_uncertainty should be populated for classifiers"


@pytest.mark.parametrize(
    "model_factory",
    [
        pytest.param(
            lambda: NODERegressor(num_trees=16, max_epochs=3, device="cpu", verbose=0),
            id="node",
        ),
        pytest.param(
            lambda: MLPHeadRegressor(max_epochs=5, device="cpu", verbose=0),
            id="mlp",
        ),
    ],
)
def test_predict_uncertainty_regression_neural(model_factory):
    """Neural regressors return the standard predict_uncertainty() DataFrame format."""
    X_train, X_test, y_train, _ = _regression_data()
    model = model_factory()
    model.fit(X_train, y_train)
    pred = model.predict_uncertainty(X_test)

    assert isinstance(pred, pd.DataFrame)
    assert len(pred) == len(X_test)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing regression uncertainty columns: {sorted(missing_cols)}"
