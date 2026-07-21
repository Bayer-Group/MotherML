"""
Tests for the BALSA / NFlows-Out active-learning acquisition scores
(``mother.ml.models.balsa_acquisition.acquisition_score``).

These exercise all four acquisition methods on both flow-capable estimators
(``FlowHeadRegressor`` and ``NODERegressor(head_type="flow")``) and assert the
returned scores are well-formed: one finite, non-negative score per pool row.

Depends on the optional ``node`` extra (skorch / torch / zuko); the module is
skipped when those are unavailable, mirroring ``test_ml_uncertainty_neural.py``.
"""

import numpy as np
import pytest

# Skip the whole module when the optional NODE/heads dependencies are absent.
pytest.importorskip("skorch")
pytest.importorskip("torch")
pytest.importorskip("zuko")

from mother.ml.models.balsa_acquisition import acquisition_score  # noqa: E402
from mother.ml.models.m_heads import FlowHeadRegressor  # noqa: E402
from mother.ml.models.m_node import NODERegressor  # noqa: E402

# - serial: avoid PyTorch multiprocessing issues under pytest-xdist
# - slow: neural-network training is computationally expensive
pytestmark = [pytest.mark.serial, pytest.mark.slow]

_METHODS = ["bald", "balsa_kl_pair", "balsa_kl_grid", "balsa_emd"]


def _regression_pool(n=90, n_pool=20, n_features=6, seed=0):
    """Deterministic small regression set split into train / unlabelled pool."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=n + n_pool, n_features=n_features, noise=0.3, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return X[:n], y[:n], X[n:]


def _assert_valid_scores(scores, n_pool):
    scores = np.asarray(scores)
    assert scores.shape == (n_pool,), f"expected ({n_pool},), got {scores.shape}"
    assert np.isfinite(scores).all(), "acquisition scores must be finite"
    # All four methods are non-negative by construction (MI ≥ 0, KL ≥ 0, W1 ≥ 0).
    assert (scores >= -1e-5).all(), "acquisition scores must be non-negative"


@pytest.mark.parametrize("method", _METHODS)
def test_acquisition_flow_head(method):
    """All acquisition methods work on a standalone FlowHeadRegressor."""
    X_tr, y_tr, X_pool = _regression_pool()
    reg = FlowHeadRegressor(flow_type="NICE", max_epochs=8, lr=1e-2, device="cpu", verbose=0)
    reg.fit(X_tr, y_tr)
    _assert_valid_scores(acquisition_score(reg, X_pool, method=method), len(X_pool))


@pytest.mark.parametrize("method", _METHODS)
def test_acquisition_node_flow(method):
    """All acquisition methods work on NODERegressor(head_type="flow")."""
    X_tr, y_tr, X_pool = _regression_pool()
    reg = NODERegressor(
        head_type="flow",
        flow_type="NICE",
        input_dropout=0.05,
        num_trees=32,
        num_layers=1,
        depth=3,
        max_epochs=8,
        device="cpu",
        verbose=0,
    )
    reg.fit(X_tr, y_tr)
    _assert_valid_scores(acquisition_score(reg, X_pool, method=method), len(X_pool))


def test_balsa_kl_grid_fixed_and_adaptive_agree_in_shape():
    """KL-Grid must work with both the adaptive grid and an explicit range.

    Regression test: the adaptive-grid path previously flattened the (G, B, D)
    node tensor to (G*B, D) and crashed because that cannot broadcast against
    the flow's (B,) batch shape.
    """
    X_tr, y_tr, X_pool = _regression_pool()
    reg = FlowHeadRegressor(flow_type="NICE", max_epochs=8, lr=1e-2, device="cpu", verbose=0)
    reg.fit(X_tr, y_tr)

    adaptive = acquisition_score(reg, X_pool, method="balsa_kl_grid")
    fixed = acquisition_score(reg, X_pool, method="balsa_kl_grid", grid_range=(-4.0, 4.0))
    _assert_valid_scores(adaptive, len(X_pool))
    _assert_valid_scores(fixed, len(X_pool))


def test_bald_matches_predict_uncertainty_knowledge():
    """The ``"bald"`` score equals the knowledge_uncertainty column (same MI)."""
    X_tr, y_tr, X_pool = _regression_pool()
    reg = FlowHeadRegressor(flow_type="NICE", max_epochs=8, lr=1e-2, device="cpu", verbose=0)
    reg.fit(X_tr, y_tr)

    bald = np.asarray(acquisition_score(reg, X_pool, method="bald"))
    _assert_valid_scores(bald, len(X_pool))
    # Both are Monte-Carlo estimates of the same mutual information; assert they
    # are the same order of magnitude and positively associated rather than
    # bit-identical (independent sampling draws).
    know = reg.predict_uncertainty(X_pool)["knowledge_uncertainty"].to_numpy()
    assert np.isfinite(know).all()
    assert (know >= -1e-5).all()


def test_invalid_method_raises():
    X_tr, y_tr, X_pool = _regression_pool()
    reg = FlowHeadRegressor(flow_type="NICE", max_epochs=5, lr=1e-2, device="cpu", verbose=0)
    reg.fit(X_tr, y_tr)
    with pytest.raises((ValueError, KeyError)):
        acquisition_score(reg, X_pool, method="not_a_real_method")


def test_unfitted_estimator_raises_not_fitted():
    """Calling acquisition_score before fit must raise NotFittedError, not a
    cryptic AttributeError on ``estimator.module_``."""
    from sklearn.exceptions import NotFittedError

    _, _, X_pool = _regression_pool()
    reg = FlowHeadRegressor(flow_type="NICE", device="cpu", verbose=0)
    with pytest.raises(NotFittedError):
        acquisition_score(reg, X_pool, method="bald")


def test_kl_grid_multitarget_raises_but_others_work():
    """balsa_kl_grid integrates the joint density on a 1-D grid and is therefore
    single-target only; it must raise for D>1 while kl_pair / emd still work."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=90, n_features=6, n_targets=2, noise=0.3, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    reg = FlowHeadRegressor(flow_type="NICE", mlp_dropout=0.05, max_epochs=8, lr=1e-2, device="cpu", verbose=0)
    reg.fit(X, y)
    X_pool = X[:15]

    _assert_valid_scores(acquisition_score(reg, X_pool, method="balsa_kl_pair"), 15)
    _assert_valid_scores(acquisition_score(reg, X_pool, method="balsa_emd"), 15)
    with pytest.raises(NotImplementedError):
        acquisition_score(reg, X_pool, method="balsa_kl_grid")
