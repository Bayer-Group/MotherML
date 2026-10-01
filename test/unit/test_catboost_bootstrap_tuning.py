"""Tests for the ``tune_bootstrap_level`` flag of the CatBoost Mother estimators.

Covers:
- The bootstrap-level parameters are only suggested when ``tune_bootstrap_level=True``
    (``subsample`` for Bernoulli, ``bagging_temperature`` for Bayesian).
- MVS sampling levels are intentionally not tuned.
- ``CatboostGaussianProcessRegressorMother`` deliberately does not expose the flag,
  because ``sample_gaussian_process`` ignores bootstrap parameters.
"""

import numpy as np
import pandas as pd
import pytest
from optuna.distributions import FloatDistribution
from optuna.trial import FixedTrial
from sklearn.base import clone

from mother.ml.models.m_catboost import (
    CatboostGaussianProcessRegressorMother,
    CatboostRankerMother,
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
    "bootstrap_type, expected, not_expected, expected_distribution",
    [
        ("Bernoulli", _SUBSAMPLE, _BAGGING_TEMPERATURE, FloatDistribution(0.25, 1.0)),
        ("Bayesian", _BAGGING_TEMPERATURE, _SUBSAMPLE, FloatDistribution(0.01, 10.0, log=True)),
    ],
)
@pytest.mark.parametrize("model_class", [CatboostRegressorMother, CatboostRankerMother])
def test_bootstrap_level_tuning_adds_matching_parameter(
    data, model_class, bootstrap_type, expected, not_expected, expected_distribution
):
    X, y = data
    model = model_class(tune_bootstrap_level=True, tune_tree_structure_type=False, tune_loss_function=False)
    trial = _fixed_trial(bootstrap_type)

    params = model.get_hyperparameter_space(X, y, trial)

    assert params["bootstrap_type"] == bootstrap_type
    assert params[expected] == trial.params[expected]
    assert trial.distributions[expected] == expected_distribution
    assert not_expected not in params
    assert not_expected not in trial.distributions


@pytest.mark.parametrize("model_class", [CatboostRegressorMother, CatboostRankerMother])
def test_bootstrap_level_tuning_skips_mvs(data, model_class):
    X, y = data
    model = model_class(tune_bootstrap_level=True, tune_tree_structure_type=False, tune_loss_function=False)
    trial = _fixed_trial("MVS")

    params = model.get_hyperparameter_space(X, y, trial)

    assert params["bootstrap_type"] == "MVS"
    assert _SUBSAMPLE not in params
    assert _BAGGING_TEMPERATURE not in params
    assert _SUBSAMPLE not in trial.distributions
    assert _BAGGING_TEMPERATURE not in trial.distributions


@pytest.mark.parametrize("bootstrap_type", ["Bernoulli", "Bayesian", "MVS"])
@pytest.mark.parametrize("model_class", [CatboostRegressorMother, CatboostRankerMother])
def test_bootstrap_level_tuning_disabled_by_default(data, model_class, bootstrap_type):
    X, y = data
    model = model_class(tune_tree_structure_type=False, tune_loss_function=False)

    assert model.tune_bootstrap_level is False
    trial = _fixed_trial(bootstrap_type)

    params = model.get_hyperparameter_space(X, y, trial)

    assert _SUBSAMPLE not in params
    assert _BAGGING_TEMPERATURE not in params
    assert _SUBSAMPLE not in trial.distributions
    assert _BAGGING_TEMPERATURE not in trial.distributions


def test_bootstrap_level_flag_round_trips_through_clone():
    model = CatboostRegressorMother(tune_bootstrap_level=True)

    assert model.get_params()["tune_bootstrap_level"] is True
    assert clone(model).tune_bootstrap_level is True


def test_gaussian_process_does_not_expose_bootstrap_level_tuning(data):
    X, y = data
    model = CatboostGaussianProcessRegressorMother()

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
