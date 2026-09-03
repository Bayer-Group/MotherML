"""Tests for the ``tune_bootstrap_level`` flag of the CatBoost Mother estimators.

Covers:
- The bootstrap-level parameters are only suggested when ``tune_bootstrap_level=True``
  and only for the bootstrap type that supports them (``subsample`` for Bernoulli,
  ``bagging_temperature`` for Bayesian, neither for MVS).
- ``CatboostGaussianProcessRegressorMother`` deliberately does not expose the flag,
  because ``sample_gaussian_process`` ignores bootstrap parameters.
"""

import numpy as np
import pandas as pd
import pytest
from optuna.trial import FixedTrial
from sklearn.base import clone

from mother.ml.models.m_catboost import (
    CatboostGaussianProcessRegressorMother,
    CatboostRegressorMother,
)

_SUBSAMPLE = "subsample"
_BAGGING_TEMPERATURE = "bagging_temperature"


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(64, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series(rng.normal(size=64))
    return X, y


def _fixed_trial(bootstrap_type: str) -> FixedTrial:
    return FixedTrial(
        {
            "bootstrap_type": bootstrap_type,
            "learning_rate": 0.05,
            "random_strength": 1.0,
            "subsample": 0.8,
            "bagging_temperature": 1.0,
            "grow_policy": "SymmetricTree",
            "max_depth": 5,
            "loss_function": "RMSE",
        }
    )


@pytest.mark.parametrize(
    "bootstrap_type, expected, not_expected",
    [
        ("Bernoulli", _SUBSAMPLE, _BAGGING_TEMPERATURE),
        ("Bayesian", _BAGGING_TEMPERATURE, _SUBSAMPLE),
    ],
)
def test_bootstrap_level_tuning_adds_matching_parameter(data, bootstrap_type, expected, not_expected):
    X, y = data
    model = CatboostRegressorMother(tune_bootstrap_level=True, tune_tree_structure_type=False)

    params = model.get_hyperparameter_space(X, y, _fixed_trial(bootstrap_type))

    assert params["bootstrap_type"] == bootstrap_type
    assert expected in params
    assert not_expected not in params


def test_bootstrap_level_tuning_skips_mvs(data):
    X, y = data
    model = CatboostRegressorMother(tune_bootstrap_level=True, tune_tree_structure_type=False)

    params = model.get_hyperparameter_space(X, y, _fixed_trial("MVS"))

    assert _SUBSAMPLE not in params
    assert _BAGGING_TEMPERATURE not in params


@pytest.mark.parametrize("bootstrap_type", ["Bernoulli", "Bayesian", "MVS"])
def test_bootstrap_level_tuning_disabled_by_default(data, bootstrap_type):
    X, y = data
    model = CatboostRegressorMother(tune_tree_structure_type=False)

    assert model.tune_bootstrap_level is False

    params = model.get_hyperparameter_space(X, y, _fixed_trial(bootstrap_type))

    assert _SUBSAMPLE not in params
    assert _BAGGING_TEMPERATURE not in params


def test_bootstrap_level_flag_round_trips_through_clone():
    model = CatboostRegressorMother(tune_bootstrap_level=True)

    assert model.get_params()["tune_bootstrap_level"] is True
    assert clone(model).tune_bootstrap_level is True


def test_gaussian_process_does_not_expose_bootstrap_level_tuning(data):
    X, y = data
    model = CatboostGaussianProcessRegressorMother(tune_tree_structure_type=False)

    assert model.tune_bootstrap_level is False
    assert "tune_bootstrap_level" not in model.get_params()

    params = model.get_hyperparameter_space(
        X,
        y,
        FixedTrial(
            {
                "bootstrap_type": "Bernoulli",
                "learning_rate": 0.05,
                "random_strength": 1.0,
                "grow_policy": "SymmetricTree",
                "max_depth": 5,
                "prior_iterations": 100,
                "samples": 10,
                "sigma": 0.1,
                "delta": 0.0,
                "eps": 1e-4,
                "random_score_type": "Gumbel",
            }
        ),
    )

    assert _SUBSAMPLE not in params
    assert _BAGGING_TEMPERATURE not in params
