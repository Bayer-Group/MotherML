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

from mother.ml.models.m_heads import (  # noqa: E402
    FlowHeadRegressor,
    MLPHeadClassifier,
    MLPHeadRegressor,
)
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


def test_flow_head_alone_aleatoric_only():
    """Standalone flow head without an MLP encoder reports aleatoric uncertainty only."""
    zuko = pytest.importorskip("zuko")  # noqa: F841
    X_train, X_test, y_train, _ = _regression_data()

    # Opt out of the default MLP encoder to get a pure flow (flow-alone).
    reg = FlowHeadRegressor(flow_type="NICE", mlp_hidden_dims=None, max_epochs=6, lr=1e-2, device="cpu", verbose=0)
    reg.fit(X_train, y_train)

    assert reg._flow_has_mc_dropout() is False

    pred = reg.predict_uncertainty(X_test, num_samples=200)
    assert isinstance(pred, pd.DataFrame)
    assert len(pred) == len(X_test)
    # No MLP-dropout -> no epistemic (knowledge) uncertainty.
    assert pred["knowledge_uncertainty"].isna().all()
    assert pred["data_uncertainty"].notna().all()
    # total == data when there is only aleatoric uncertainty.
    np.testing.assert_allclose(
        pred["total_uncertainty"].to_numpy(dtype=float),
        pred["data_uncertainty"].to_numpy(dtype=float),
    )


def test_flow_head_default_has_mlp_encoder_with_uncertainty():
    """The standalone flow head defaults to an MLP encoder with dropout (epistemic uncertainty)."""
    zuko = pytest.importorskip("zuko")  # noqa: F841
    X_train, X_test, y_train, _ = _regression_data()

    # No mlp_* args -> default "auto" 2-layer encoder [128, 64] with 0.1 dropout.
    reg = FlowHeadRegressor(flow_type="NICE", max_epochs=6, lr=1e-2, device="cpu", verbose=0)
    assert reg.get_params()["mlp_hidden_dims"] == [128, 64]
    assert reg.get_params()["mlp_dropout"] == 0.1

    reg.fit(X_train, y_train)
    assert reg._flow_has_mc_dropout() is True

    pred = reg.predict_uncertainty(X_test, num_samples=200, num_mc_samples=8)
    # Epistemic uncertainty is available by default.
    assert pred["knowledge_uncertainty"].notna().all()


def test_flow_head_mlp_dropout_uncertainty_decomposition():
    """Flow head with an MLP encoder + dropout exposes the NODE-style BALD decomposition."""
    zuko = pytest.importorskip("zuko")  # noqa: F841
    X_train, X_test, y_train, _ = _regression_data()

    reg = FlowHeadRegressor(
        flow_type="NICE",
        mlp_hidden_dims=[64, 32],
        mlp_dropout=0.1,
        mlp_activation="ReLU",
        mlp_batch_norm=True,
        max_epochs=6,
        lr=1e-2,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_train, y_train)

    assert reg._flow_has_mc_dropout() is True

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

    # Combined-uncertainty helper returns per-pass diagnostics.
    stats = reg.predict_with_combined_uncertainty(X_test, num_mc_samples=8, num_flow_samples=50, return_all=True)
    assert stats["mc_uncertainties"].shape[0] == 8
    assert stats["mc_uncertainties"].shape[1] == len(X_test)


def test_flow_head_quantiles_available():
    """Flow head returns predictive quantiles regardless of the MLP-encoder setting."""
    zuko = pytest.importorskip("zuko")  # noqa: F841
    X_train, X_test, y_train, _ = _regression_data()

    reg = FlowHeadRegressor(flow_type="NICE", max_epochs=6, lr=1e-2, device="cpu", verbose=0)
    reg.fit(X_train, y_train)

    q = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9], num_samples=200)
    assert q.shape == (len(X_test), 3)
    # Quantiles are monotonically non-decreasing per row.
    assert (np.diff(q, axis=1) >= -1e-4).all()


def test_flow_head_clone_preserves_architecture():
    """sklearn.clone() round-trips the full flow architecture, including the MLP encoder."""
    from sklearn.base import clone

    zuko = pytest.importorskip("zuko")  # noqa: F841

    keys = [
        "mlp_hidden_dims",
        "mlp_dropout",
        "mlp_activation",
        "mlp_batch_norm",
        "flow_type",
        "flow_bins",
        "input_dim",
        "output_dim",
    ]

    def snap(est):
        params = est.get_params()
        return {k: params[k] for k in keys}

    # Default "auto" encoder resolves to a concrete list that must survive cloning.
    default = FlowHeadRegressor()
    assert default.get_params()["mlp_hidden_dims"] == [128, 64]
    assert snap(default) == snap(clone(default))

    # Non-default architectures, including batch-norm off and the flow-alone opt-outs.
    for kwargs in (
        dict(
            mlp_hidden_dims=[256, 128, 64],
            mlp_dropout=0.3,
            mlp_activation="GELU",
            mlp_batch_norm=False,
            flow_type="NSF",
            flow_bins=12,
        ),
        dict(mlp_hidden_dims=None, mlp_dropout=0.0),
        dict(mlp_hidden_dims=[], mlp_dropout=0.2),
    ):
        reg = FlowHeadRegressor(**kwargs)
        assert snap(reg) == snap(clone(reg))
        assert snap(reg) == snap(reg.__sklearn_clone__())

    # A cloned (unfitted) copy rebuilds the module with the cloned params.
    X_train, _, y_train, _ = _regression_data()
    reg = FlowHeadRegressor(
        mlp_hidden_dims=[32, 16], mlp_dropout=0.1, mlp_batch_norm=False, max_epochs=3, device="cpu", verbose=0
    )
    cloned = clone(reg)
    cloned.fit(X_train, y_train)
    encoder_types = {m.__class__.__name__ for m in cloned.module_.encoder.modules()}
    assert cloned.module_.mlp_batch_norm is False
    assert "BatchNorm1d" not in encoder_types
    assert "Dropout" in encoder_types
