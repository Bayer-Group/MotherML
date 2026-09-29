import numpy as np
import pandas as pd
import pytest
from scipy.special import rel_entr
from sklearn.base import BaseEstimator

from mother.ml.core import AbstractMotherPipeline
from mother.ml.ensembles.cvEnsemble import CVEnsembleClassifierMother

AGGREGATION_STRATEGIES = ["average"]
UNCERTAINTY_METHODS = ["variance", "disagreement"]


class _FakeClassifier(AbstractMotherPipeline, BaseEstimator):
    """Minimal fitted classifier exposing the ``predict_uncertainty`` contract used by the ensemble."""

    def __init__(self, probas, classes):
        self.probas = np.asarray(probas, dtype=float)
        self.classes_ = np.asarray(classes)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def get_hyperparameter_space(self, X, y, trial, prefix=""):
        return {}

    def predict_uncertainty(self, X, **kwargs):
        probas = self.probas[: len(X)]
        labels = self.classes_[np.argmax(probas, axis=1)]
        columns = {f"proba_{i}": probas[:, i] for i in range(probas.shape[1])}
        return pd.DataFrame({"pred": labels, **columns})


class _ClassifierWithoutClasses(AbstractMotherPipeline, BaseEstimator):
    def __init__(self):
        self.fitted_ = True

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def get_hyperparameter_space(self, X, y, trial, prefix=""):
        return {}

    def predict_uncertainty(self, X, y=None):
        return pd.DataFrame()


class _ClassifierWithoutPredictUncertainty(BaseEstimator):
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def get_hyperparameter_space(self, X, y, trial, prefix=""):
        return {}


class _UnfittedClassifier(AbstractMotherPipeline, BaseEstimator):
    """Estimator without any fitted attribute, so ``check_is_fitted`` fails."""

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def get_hyperparameter_space(self, X, y, trial, prefix=""):
        return {}


@pytest.fixture
def binary_probas():
    first = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    second = np.array([[0.7, 0.3], [0.4, 0.6], [0.3, 0.7]])
    return first, second


@pytest.fixture
def multiclass_probas():
    first = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    second = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]])
    return first, second


@pytest.fixture
def X():
    return pd.DataFrame({"f0": [0.0, 1.0, 2.0], "f1": [3.0, 4.0, 5.0]})


@pytest.fixture
def binary_ensemble(binary_probas):
    first, second = binary_probas
    return CVEnsembleClassifierMother([_FakeClassifier(first, [0, 1]), _FakeClassifier(second, [0, 1])])


@pytest.fixture
def multiclass_ensemble(multiclass_probas):
    first, second = multiclass_probas
    return CVEnsembleClassifierMother([_FakeClassifier(first, [0, 1, 2]), _FakeClassifier(second, [0, 1, 2])])


def _expected_uncertainty(probas: np.ndarray, method: str) -> np.ndarray:
    if method == "variance":
        return np.var(probas, axis=0).mean(axis=1)

    ensemble_probas = np.mean(probas, axis=0)
    return np.sum([np.sum(rel_entr(member, ensemble_probas), axis=1) for member in probas], axis=0)


class TestEnsembleClassifierInitialization:
    def test_initialization(self, binary_ensemble):
        assert binary_ensemble is not None
        assert len(binary_ensemble.estimators) == 2
        np.testing.assert_array_equal(binary_ensemble.classes_, np.array([0, 1]))

    def test_single_estimator_raises(self, binary_probas):
        with pytest.raises(ValueError, match=r"At least two estimators"):
            CVEnsembleClassifierMother([_FakeClassifier(binary_probas[0], [0, 1])])

    def test_empty_estimator_list_raises(self):
        with pytest.raises(ValueError, match=r"At least two estimators"):
            CVEnsembleClassifierMother([])

    def test_unfitted_estimator_raises(self, binary_probas):
        with pytest.raises(ValueError, match=r"is not fitted"):
            CVEnsembleClassifierMother([_FakeClassifier(binary_probas[0], [0, 1]), _UnfittedClassifier()])

    def test_noncallable_predict_uncertainty_raises(self, binary_probas):
        first, second = binary_probas
        with pytest.raises(ValueError, match=r"must have a 'predict_uncertainty' method"):
            CVEnsembleClassifierMother([_ClassifierWithoutPredictUncertainty(), _FakeClassifier(second, [0, 1])])

    def test_missing_class_labels_raise(self, binary_probas):
        first, second = binary_probas
        with pytest.raises(ValueError, match=r"must have a 'classes_' attribute"):
            CVEnsembleClassifierMother([_ClassifierWithoutClasses(), _FakeClassifier(second, [0])])

    def test_inconsistent_class_labels_raise(self, binary_probas):
        first, second = binary_probas
        with pytest.raises(ValueError, match=r"same class labels"):
            CVEnsembleClassifierMother(
                [_FakeClassifier(first, [0, 1]), _FakeClassifier(second, ["active", "inactive"])]
            )


class TestEnsembleClassifierParameters:
    def test_get_params_shallow(self, binary_ensemble):
        params = binary_ensemble.get_params(deep=False)
        assert list(params) == ["estimators"]
        assert params["estimators"] is binary_ensemble.estimators

    def test_get_params_deep(self, binary_ensemble):
        params = binary_ensemble.get_params(deep=True)
        assert set(params) == {"estimators", "estimator_0", "estimator_1"}
        assert params["estimator_0"].keys() == binary_ensemble.estimators[0].get_params().keys()

    def test_set_params_is_noop(self, binary_ensemble):
        estimators_before = list(binary_ensemble.estimators)
        assert binary_ensemble.set_params(estimators=[]) is binary_ensemble
        assert binary_ensemble.set_params() is binary_ensemble
        assert binary_ensemble.estimators == estimators_before

    def test_hyperparameter_space_is_empty(self, binary_ensemble, X):
        assert binary_ensemble.get_hyperparameter_space(X, None, trial=None) == {}
        assert binary_ensemble.get_hyperparameter_space(X, None, trial=None, prefix="model__") == {}

    def test_default_parameters_is_empty(self, binary_ensemble):
        assert binary_ensemble.default_parameters() == {}

    def test_fit_is_a_noop(self, binary_ensemble, X):
        estimators_before = list(binary_ensemble.estimators)
        assert binary_ensemble.fit(X, np.array([0, 1, 1])) is binary_ensemble
        assert binary_ensemble.estimators == estimators_before


class TestEnsembleClassifierPredictions:
    def test_predict_probas_shape(self, binary_ensemble, binary_probas, X):
        probas = binary_ensemble._predict_probas(X)
        assert probas.shape == (2, len(X), 2)
        np.testing.assert_allclose(probas[0], binary_probas[0])
        np.testing.assert_allclose(probas[1], binary_probas[1])

    @pytest.mark.parametrize("strategy", AGGREGATION_STRATEGIES)
    @pytest.mark.parametrize("method", UNCERTAINTY_METHODS)
    def test_binary_predict_uncertainty(self, binary_ensemble, binary_probas, X, strategy, method):
        result = binary_ensemble.predict_uncertainty(X, aggregation_strategy=strategy, uncertainty_method=method)

        assert len(result) == len(X)
        np.testing.assert_array_equal(result["pred"].to_numpy(), np.array([0, 1, 1]))
        np.testing.assert_allclose(
            result["knowledge_uncertainty"].to_numpy(),
            _expected_uncertainty(np.stack(binary_probas, axis=0), method),
        )

    @pytest.mark.parametrize("strategy", AGGREGATION_STRATEGIES)
    @pytest.mark.parametrize("method", UNCERTAINTY_METHODS)
    def test_multiclass_predict_uncertainty(self, multiclass_ensemble, multiclass_probas, X, strategy, method):
        result = multiclass_ensemble.predict_uncertainty(X, aggregation_strategy=strategy, uncertainty_method=method)

        np.testing.assert_array_equal(result["pred"].to_numpy(), np.array([0, 1, 2]))
        np.testing.assert_allclose(
            result["knowledge_uncertainty"].to_numpy(),
            _expected_uncertainty(np.stack(multiclass_probas, axis=0), method),
        )

    @pytest.mark.parametrize("strategy", AGGREGATION_STRATEGIES)
    @pytest.mark.parametrize("method", UNCERTAINTY_METHODS)
    def test_string_class_labels(self, binary_probas, X, strategy, method):
        first, second = binary_probas
        labels = np.array(["active", "inactive"])
        ensemble = CVEnsembleClassifierMother([_FakeClassifier(first, labels), _FakeClassifier(second, labels)])

        result = ensemble.predict_uncertainty(X, aggregation_strategy=strategy, uncertainty_method=method)

        np.testing.assert_array_equal(result["pred"].to_numpy(), np.array(["active", "inactive", "inactive"]))
        np.testing.assert_allclose(
            result["knowledge_uncertainty"].to_numpy(),
            _expected_uncertainty(np.stack(binary_probas, axis=0), method),
        )

    @pytest.mark.parametrize("method", UNCERTAINTY_METHODS)
    def test_identical_members_have_zero_uncertainty(self, binary_probas, X, method):
        first = binary_probas[0]
        ensemble = CVEnsembleClassifierMother([_FakeClassifier(first, [0, 1]), _FakeClassifier(first.copy(), [0, 1])])

        result = ensemble.predict_uncertainty(X, uncertainty_method=method)

        np.testing.assert_allclose(result["knowledge_uncertainty"].to_numpy(), np.zeros(len(X)), atol=1e-12)
        np.testing.assert_array_equal(result["pred"].to_numpy(), np.array([0, 1, 0]))

    def test_predict_proba(self, binary_ensemble, binary_probas, X):
        result = binary_ensemble.predict_proba(X)

        expected = np.mean(np.stack(binary_probas, axis=0), axis=0)
        assert result.shape == expected.shape
        np.testing.assert_allclose(np.sum(result, axis=1), 1)
        np.testing.assert_allclose(result, expected)

    def test_predict(self, binary_ensemble, binary_probas, X):
        result = binary_ensemble.predict(X)

        expected = np.argmax(np.stack(binary_probas, axis=0).mean(axis=0), axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_three_members(self, binary_probas, X):
        first, second = binary_probas
        third = np.array([[0.1, 0.9], [0.1, 0.9], [0.2, 0.8]])
        ensemble = CVEnsembleClassifierMother(
            [
                _FakeClassifier(first, [0, 1]),
                _FakeClassifier(second, [0, 1]),
                _FakeClassifier(third, [0, 1]),
            ]
        )

        result = ensemble.predict_uncertainty(X)
        expected = np.argmax(np.mean(np.stack([first, second, third], axis=0), axis=0), axis=1)

        assert ensemble._predict_probas(X).shape == (3, len(X), 2)
        np.testing.assert_array_equal(result["pred"].to_numpy(), expected)


class TestEnsembleClassifierErrorHandling:
    def test_unknown_aggregation_strategy_raises(self, binary_ensemble, X):
        with pytest.raises(ValueError, match=r"Unknown aggregation strategy"):
            binary_ensemble.predict_uncertainty(X, aggregation_strategy="unknown")

    def test_unknown_uncertainty_method_raises(self, binary_ensemble, X):
        with pytest.raises(ValueError, match=r"Unknown uncertainty estimation method"):
            binary_ensemble.predict_uncertainty(X, uncertainty_method="unknown")

    @pytest.mark.parametrize("shape", [(3, 2), (2, 3, 2, 1)])
    def test_aggregate_predictions_requires_3d_input(self, binary_ensemble, shape):
        with pytest.raises(ValueError, match=r"Unexpected dimension for probas"):
            binary_ensemble._aggregate_predictions(np.zeros(shape))
