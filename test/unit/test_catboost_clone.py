"""Tests for __sklearn_clone__ on all CatBoost Mother estimators.

Covers:
- Clone round-trip for mutable constructor params (cat_features, text_features,
  embedding_features) that newer CatBoost versions copy internally, which breaks
  sklearn's default identity-based clone check.
- Clone round-trip without any mutable params.
- Clone preserves metadata routing requests (requires scikit-learn >= 1.3).
- cross_val_score works end-to-end with cat_features (exercises clone inside CV).
"""

import pandas as pd
import pytest
import sklearn
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score

from mother.ml.models.m_catboost import (
    CatboostClassifierMother,
    CatboostRankerMother,
    CatboostRegressorMother,
)

_SKLEARN_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGRESSOR_CLASSES = [CatboostRegressorMother]
_CLASSIFIER_CLASSES = [CatboostClassifierMother]
_ALL_CLASSES = [CatboostRegressorMother, CatboostClassifierMother, CatboostRankerMother]


# ---------------------------------------------------------------------------
# Mutable-param clone tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator_cls", _REGRESSOR_CLASSES)
@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"cat_features": [0, 1]},
        {"embedding_features": [0]},
    ],
)
def test_clone_regressor_with_mutable_params(estimator_cls, extra_kwargs):
    """Cloning a regressor with mutable constructor params must preserve those params."""
    original = estimator_cls(iterations=2, verbose=False, **extra_kwargs)
    cloned = clone(original)

    assert type(cloned) is estimator_cls
    for key, value in extra_kwargs.items():
        assert cloned.get_params(deep=False)[key] == value


@pytest.mark.parametrize("estimator_cls", _CLASSIFIER_CLASSES)
@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"cat_features": [0, 1]},
        {"embedding_features": [0]},
    ],
)
def test_clone_classifier_with_mutable_params(estimator_cls, extra_kwargs):
    """Cloning a classifier with mutable constructor params must preserve those params."""
    original = estimator_cls(iterations=2, verbose=False, **extra_kwargs)
    cloned = clone(original)

    assert type(cloned) is estimator_cls
    for key, value in extra_kwargs.items():
        assert cloned.get_params(deep=False)[key] == value


# ---------------------------------------------------------------------------
# Plain clone (no mutable params)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator_cls", _ALL_CLASSES)
def test_clone_without_mutable_params(estimator_cls):
    """Clone must succeed and produce the correct type even without mutable params."""
    original = estimator_cls(iterations=2, verbose=False)
    cloned = clone(original)
    assert type(cloned) is estimator_cls


# ---------------------------------------------------------------------------
# Metadata routing preservation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _SKLEARN_VERSION < (1, 3),
    reason="metadata routing requires scikit-learn >= 1.3",
)
def test_clone_preserves_metadata_routing_request(preserve_metadata_routing):
    """Clone must carry over set_fit_request state on CatboostRankerMother."""
    with config_context(enable_metadata_routing=True):
        original = CatboostRankerMother(iterations=2, verbose=False)
        original.set_fit_request(group_id=True)

        cloned = clone(original)

        assert hasattr(cloned, "_metadata_request")
        assert repr(original._metadata_request) == repr(cloned._metadata_request)


# ---------------------------------------------------------------------------
# cross_val_score integration (exercises clone inside CV loop)
# ---------------------------------------------------------------------------


def test_cross_val_score_regressor_with_cat_features():
    """cross_val_score must complete without error when cat_features is set."""
    X, y = make_regression(n_samples=90, n_features=5, random_state=0)
    X = pd.DataFrame(X)
    X[0] = (X[0] > 0).astype(int)

    estimator = CatboostRegressorMother(cat_features=[0], iterations=2, verbose=False)
    scores = cross_val_score(estimator, X, y, cv=3)
    assert len(scores) == 3


def test_cross_val_score_classifier_with_cat_features():
    """cross_val_score must complete without error when cat_features is set."""
    X, y = make_classification(n_samples=90, n_features=5, random_state=0)
    X = pd.DataFrame(X)
    X[0] = (X[0] > 0).astype(int)

    estimator = CatboostClassifierMother(cat_features=[0], iterations=2, verbose=False)
    scores = cross_val_score(estimator, X, y, cv=3)
    assert len(scores) == 3
