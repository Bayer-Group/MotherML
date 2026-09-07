"""Uncertainty-interface tests for NODE estimators."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

# Skip the entire module when the optional NODE/heads dependencies are absent.
pytest.importorskip("skorch")
pytest.importorskip("torch")

from mother.ml.models.m_node import NODEClassifier, NODERegressor  # noqa: E402

# - serial: avoid PyTorch multiprocessing issues under pytest-xdist
# - slow: NODE training is computationally expensive
pytestmark = [pytest.mark.serial, pytest.mark.slow]

REQUIRED_UNCERTAINTY_COLS = {
    "pred",
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


def test_predict_uncertainty_classification_node():
    """NODE classifiers return the standard predict_uncertainty() DataFrame format."""
    X_train, X_test, y_train, _ = _classification_data()
    model = NODEClassifier(num_trees=16, max_epochs=3, device="cpu", verbose=0)
    model.fit(X_train, y_train)
    pred = model.predict_uncertainty(X_test)

    assert isinstance(pred, pd.DataFrame)
    assert len(pred) == len(X_test)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing classification uncertainty columns: {sorted(missing_cols)}"
    assert pred["total_uncertainty"].notna().all(), "total_uncertainty should be populated for classifiers"


def test_predict_uncertainty_regression_node():
    """NODE regressors return the standard predict_uncertainty() DataFrame format."""
    X_train, X_test, y_train, _ = _regression_data()
    model = NODERegressor(num_trees=16, max_epochs=3, device="cpu", verbose=0)
    model.fit(X_train, y_train)
    pred = model.predict_uncertainty(X_test)

    assert isinstance(pred, pd.DataFrame)
    assert len(pred) == len(X_test)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing regression uncertainty columns: {sorted(missing_cols)}"


def test_node_flow_uncertainty_columns_present():
    """NODE flow head returns the standard uncertainty columns."""
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.05,
        num_trees=32,
        num_layers=1,
        depth=3,
        max_epochs=6,
        lr=1e-2,
        device="cpu",
        verbose=0,
    )

    reg.fit(X_train, y_train)
    pred = reg.predict_uncertainty(X_test, num_samples=200, num_mc_samples=8)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing regression uncertainty columns: {sorted(missing_cols)}"
    assert pred["data_uncertainty"].notna().all()
    assert pred["total_uncertainty"].notna().all()


def test_node_flow_uncertainty_decomposition_identity():
    """NODE flow head keeps total = data + knowledge uncertainty."""
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.05,
        num_trees=32,
        num_layers=1,
        depth=3,
        max_epochs=6,
        lr=1e-2,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    pred = reg.predict_uncertainty(X_test, num_samples=200, num_mc_samples=8)
    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing regression uncertainty columns: {sorted(missing_cols)}"

    knowledge = pred["knowledge_uncertainty"].to_numpy(dtype=float)
    data = pred["data_uncertainty"].to_numpy(dtype=float)
    total = pred["total_uncertainty"].to_numpy(dtype=float)

    # Epistemic (mutual information) is populated and non-negative; identity holds exactly.
    assert pred["knowledge_uncertainty"].notna().all()
    assert (knowledge >= -1e-6).all()
    np.testing.assert_allclose(total, data + knowledge, atol=1e-5)


def test_node_flow_quantiles_available():
    """NODE flow head returns predictive quantiles."""
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.05,
        num_trees=32,
        num_layers=1,
        depth=3,
        max_epochs=6,
        lr=1e-2,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    q = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9], num_samples=200)
    assert q.shape == (len(X_test), 3)
    # Quantiles are monotonically non-decreasing per row.
    assert (np.diff(q, axis=1) >= -1e-4).all()


def test_predict_uncertainty_warns_when_all_dropouts_zero():
    """NODE emits a warning when MC-dropout uncertainty is requested with dropout=0."""
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="subset",
        input_dropout=0.0,
        tree_dropout=0.0,
        num_trees=16,
        num_layers=1,
        depth=3,
        max_epochs=4,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    with pytest.warns(UserWarning, match="MC-dropout repeats are deterministic"):
        _ = reg.predict_uncertainty(X_test, num_samples=16)


def test_predict_uncertainty_dropout_overrides_are_temporary():
    """Inference-only input/tree dropout overrides enable MC uncertainty safely."""
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="subset",
        input_dropout=0.0,
        tree_dropout=0.0,
        num_trees=16,
        num_layers=1,
        depth=3,
        max_epochs=4,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    result = reg.predict_uncertainty(X_test, num_samples=16, input_dropout=0.1, tree_dropout=0.1)

    assert result["knowledge_uncertainty"].mean() > 0
    assert reg.input_dropout == 0.0
    assert reg.tree_dropout == 0.0
    assert reg.module_.input_dropout == 0.0
    assert reg.module_.tree_dropout == 0.0

    with pytest.raises(ValueError, match="input_dropout must be in"):
        reg.predict_uncertainty(X_test, input_dropout=1.0)


def test_node_flow_balsa_emd_opt_signal_and_total_nan():
    """BALSA-EMD mode exposes epistemic score for optimisation and marks total as NaN."""
    pytest.importorskip("zuko")
    X_train, X_test, y_train, _ = _regression_data()

    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.05,
        num_trees=32,
        num_layers=1,
        depth=3,
        max_epochs=6,
        lr=1e-2,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    pred = reg.predict_uncertainty(
        X_test,
        num_samples=120,
        knowledge_method="balsa_emd",
    )

    missing_cols = REQUIRED_UNCERTAINTY_COLS - set(pred.columns)
    assert not missing_cols, f"Missing regression uncertainty columns: {sorted(missing_cols)}"
    assert pred["knowledge_uncertainty"].notna().all()
    assert pred["total_uncertainty"].isna().all()

    opt_signal = reg.predict_uncertainty(
        X_test,
        num_samples=120,
        knowledge_method="balsa_emd",
        uncertainty_for_opt=True,
    )
    opt_vals = opt_signal.to_numpy(dtype=float)
    pred_vals = pred["knowledge_uncertainty"].to_numpy(dtype=float)
    assert opt_vals.shape == pred_vals.shape
    assert np.isfinite(opt_vals).all()
    assert (opt_vals >= 0).all()
