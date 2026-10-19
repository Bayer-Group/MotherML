import numpy as np
import pandas as pd
import pytest
import sklearn.base as sk_base
from scipy.special import rel_entr
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from mother.ml.core import AbstractMotherPipeline
from mother.ml.ensembles.cvEnsemble import CVEnsembleClassifierMother

AGGREGATION_STRATEGIES = ["average"]
UNCERTAINTY_METHODS = ["variance", "disagreement", "member_average"]


class _FakeClassifier(AbstractMotherPipeline, BaseEstimator):
    """Minimal fitted classifier exposing the ``predict_uncertainty`` contract used by the ensemble."""

    def __init__(self, probas, uncertainties, classes):
        self.probas = np.asarray(probas, dtype=float)
        self.uncertainties = np.asarray(uncertainties, dtype=float)
        self.classes_ = np.asarray(classes)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {
            "param_1_key": "param_1_value",
            "param_2_key": "param_2_value",
        }

    def set_params(self, **params):
        return self

    def get_hyperparameter_space(self, X, y, trial, prefix=""):
        return {}

    def predict_proba(self, X):
        probas = self.probas[: len(X)]
        columns = {f"proba_{i}": probas[:, i] for i in range(probas.shape[1])}
        return pd.DataFrame(columns)

    def predict_uncertainty(self, X, **kwargs):
        probas = self.probas[: len(X)]
        labels = self.classes_[np.argmax(probas, axis=1)]
        columns = {f"proba_{i}": probas[:, i] for i in range(probas.shape[1])}
        return pd.DataFrame(
            {
                "pred": labels,
                **columns,
                "prefix_knowledge_uncertainty": self.uncertainties[: len(X)],
            }
        )


class _ClassifierWithoutKnowledgeUncertaintyValues(AbstractMotherPipeline, BaseEstimator):
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

    def predict_proba(self, X):
        probas = self.probas[: len(X)]
        columns = {f"proba_{i}": probas[:, i] for i in range(probas.shape[1])}
        return pd.DataFrame(columns)

    def predict_uncertainty(self, X, **kwargs):
        probas = self.probas[: len(X)]
        labels = self.classes_[np.argmax(probas, axis=1)]
        columns = {f"proba_{i}": probas[:, i] for i in range(probas.shape[1])}
        return pd.DataFrame({"pred": labels, **columns, "prefix_knowledge_uncertainty": np.nan})


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
def knowledge_uncertainties():
    first = np.array([0.1, 0.3, 0.5])
    second = np.array([0.3, 0.5, 0.7])
    return first, second


@pytest.fixture
def X():
    return pd.DataFrame({"f0": [0.0, 1.0, 2.0], "f1": [3.0, 4.0, 5.0]})


@pytest.fixture
def binary_ensemble(binary_probas, knowledge_uncertainties):
    first_proba, second_proba = binary_probas
    first_unc, second_unc = knowledge_uncertainties
    return CVEnsembleClassifierMother(
        [
            _FakeClassifier(first_proba, first_unc, classes=[0, 1]),
            _FakeClassifier(second_proba, second_unc, classes=[0, 1]),
        ]
    )


@pytest.fixture
def multiclass_ensemble(multiclass_probas, knowledge_uncertainties):
    first_proba, second_proba = multiclass_probas
    first_unc, second_unc = knowledge_uncertainties
    return CVEnsembleClassifierMother(
        [
            _FakeClassifier(first_proba, first_unc, classes=[0, 1, 2]),
            _FakeClassifier(second_proba, second_unc, classes=[0, 1, 2]),
        ]
    )


def _expected_uncertainty(probas: np.ndarray, knowledge_uncertainties: np.ndarray, method: str) -> np.ndarray:
    if method == "variance":
        return np.var(probas, axis=0).mean(axis=1)
    if method == "member_average":
        return np.mean(knowledge_uncertainties, axis=0)

    # method "disagreement" using KL divergence
    ensemble_probas = np.mean(probas, axis=0)
    return np.sum([np.sum(rel_entr(member, ensemble_probas), axis=1) for member in probas], axis=0)


class TestEnsembleClassifierInitialization:
    def test_initialization(self, binary_ensemble):
        assert binary_ensemble is not None
        assert len(binary_ensemble.estimators) == 2

    def test_single_estimator_raises(self, binary_probas, knowledge_uncertainties):
        with pytest.raises(ValueError, match=r"At least two estimators"):
            CVEnsembleClassifierMother([_FakeClassifier(binary_probas[0], knowledge_uncertainties[0], classes=[0, 1])])

    def test_empty_estimator_list_raises(self):
        with pytest.raises(ValueError, match=r"At least two estimators"):
            CVEnsembleClassifierMother([])

    def test_unfitted_estimator_raises(self, binary_probas, knowledge_uncertainties):
        with pytest.raises(ValueError, match=r"is not fitted"):
            CVEnsembleClassifierMother(
                [
                    _FakeClassifier(binary_probas[0], knowledge_uncertainties[0], classes=[0, 1]),
                    _UnfittedClassifier(),
                ]
            )

    def test_noncallable_predict_uncertainty_raises(self, binary_probas, knowledge_uncertainties):
        first_proba, second_proba = binary_probas
        first_unc, second_unc = knowledge_uncertainties
        with pytest.raises(ValueError, match=r"must have a 'predict_uncertainty' method"):
            CVEnsembleClassifierMother(
                [
                    _ClassifierWithoutPredictUncertainty(),
                    _FakeClassifier(second_proba, second_unc, classes=[0, 1]),
                ]
            )

    def test_missing_class_labels_raise(self, binary_probas, knowledge_uncertainties):
        first_proba, second_proba = binary_probas
        first_unc, second_unc = knowledge_uncertainties
        with pytest.raises(ValueError, match=r"must have a 'classes_' attribute"):
            CVEnsembleClassifierMother(
                [
                    _ClassifierWithoutClasses(),
                    _FakeClassifier(second_proba, second_unc, classes=[0]),
                ]
            )

    def test_inconsistent_class_labels_raise(self, binary_probas, knowledge_uncertainties):
        first_proba, second_proba = binary_probas
        first_unc, second_unc = knowledge_uncertainties
        with pytest.raises(ValueError, match=r"same class labels"):
            CVEnsembleClassifierMother(
                [
                    _FakeClassifier(first_proba, first_unc, classes=[0, 1]),
                    _FakeClassifier(second_proba, second_unc, classes=["active", "inactive"]),
                ]
            )


class TestSklearnCompatibility:
    def test_cloning_preserves_fitted_state(self, binary_ensemble, X):
        clone = sk_base.clone(binary_ensemble)
        assert clone is not binary_ensemble
        assert clone.estimators is not binary_ensemble.estimators
        for original, cloned in zip(binary_ensemble.estimators, clone.estimators):
            assert cloned is not original
            assert original.get_params() == cloned.get_params()
            np.testing.assert_array_equal(cloned.classes_, original.classes_)

        expected_proba = binary_ensemble.predict_proba(X)
        expected_pred = binary_ensemble.predict(X)
        np.testing.assert_array_equal(clone.predict_proba(X), expected_proba)
        np.testing.assert_array_equal(clone.predict(X), expected_pred)

    def test_classifer_recognition(self, binary_ensemble):
        assert sk_base.is_classifier(binary_ensemble)

    def test_only_single_output(self, binary_ensemble):
        tags = sk_base.get_tags(binary_ensemble)
        assert tags.target_tags.multi_output is False

    def test_fitted_state(self, binary_ensemble):
        assert hasattr(binary_ensemble, "classes_")
        np.testing.assert_array_equal(binary_ensemble.classes_, np.array([0, 1]))
        np.testing.assert_array_equal(binary_ensemble.classes_, binary_ensemble.estimators[0].classes_)

    def test_pipeline_use(self, binary_ensemble):
        pipeline = Pipeline([("model", binary_ensemble)])
        assert sk_base.is_classifier(pipeline)
        pipeline.fit(np.array([[0], [1], [2]]), np.array([0, 1, 1]))


class TestEnsembleClassifierParameters:
    def test_get_params_shallow(self, binary_ensemble):
        params = binary_ensemble.get_params(deep=False)
        assert list(params) == ["estimators"]
        assert params["estimators"] is binary_ensemble.estimators

    def test_get_params_deep(self, binary_ensemble):
        params = binary_ensemble.get_params(deep=True)
        first_estimator_keys = binary_ensemble.estimators[0].get_params().keys()
        assert len(first_estimator_keys) > 0
        assert set(params) >= {
            "estimators",
            "estimator_0__param_1_key",
            "estimator_0__param_2_key",
        }

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
        probas = binary_ensemble._predict_probas_and_uncertainties(X)
        assert probas.shape == (2, len(X), 2)
        np.testing.assert_allclose(probas[0], binary_probas[0])
        np.testing.assert_allclose(probas[1], binary_probas[1])

    def test_predict_uncertainty_result(self, binary_ensemble, X):
        result = binary_ensemble.predict_uncertainty(X)
        assert {
            "pred",
            "mean_predictions",
            "knowledge_uncertainty",
            "data_uncertainty",
            "total_uncertainty",
        }.issubset(result.columns)
        assert len(result) == len(X)

    def test_entropy_kl_div_method_agreement(self, binary_ensemble, X):
        entropy_result = binary_ensemble.predict_uncertainty(X, uncertainty_method="entropy")
        disagreement_result = binary_ensemble.predict_uncertainty(X, uncertainty_method="disagreement")
        assert "knowledge_uncertainty" in entropy_result.columns
        assert "knowledge_uncertainty" in disagreement_result.columns
        np.testing.assert_allclose(
            entropy_result["knowledge_uncertainty"].to_numpy(),
            disagreement_result["knowledge_uncertainty"].to_numpy() / len(binary_ensemble.estimators),
        )

    @pytest.mark.parametrize("strategy", AGGREGATION_STRATEGIES)
    @pytest.mark.parametrize("method", UNCERTAINTY_METHODS)
    def test_binary_predict_uncertainty(
        self,
        binary_ensemble,
        binary_probas,
        knowledge_uncertainties,
        X,
        strategy,
        method,
    ):
        result = binary_ensemble.predict_uncertainty(X, aggregation_strategy=strategy, uncertainty_method=method)

        assert len(result) == len(X)
        np.testing.assert_array_equal(result["pred"].to_numpy(), np.array([0, 1, 1]))
        np.testing.assert_allclose(
            result["knowledge_uncertainty"].to_numpy(),
            _expected_uncertainty(
                np.stack(binary_probas, axis=0),
                np.stack(knowledge_uncertainties, axis=0),
                method,
            ),
        )

    @pytest.mark.parametrize("strategy", AGGREGATION_STRATEGIES)
    @pytest.mark.parametrize("method", UNCERTAINTY_METHODS)
    def test_multiclass_predict_uncertainty(
        self,
        multiclass_ensemble,
        multiclass_probas,
        knowledge_uncertainties,
        X,
        strategy,
        method,
    ):
        result = multiclass_ensemble.predict_uncertainty(X, aggregation_strategy=strategy, uncertainty_method=method)

        np.testing.assert_array_equal(result["pred"].to_numpy(), np.array([0, 1, 2]))
        np.testing.assert_allclose(
            result["knowledge_uncertainty"].to_numpy(),
            _expected_uncertainty(
                np.stack(multiclass_probas, axis=0),
                np.stack(knowledge_uncertainties, axis=0),
                method,
            ),
        )

    @pytest.mark.parametrize("strategy", AGGREGATION_STRATEGIES)
    @pytest.mark.parametrize("method", UNCERTAINTY_METHODS)
    def test_string_class_labels(self, binary_probas, knowledge_uncertainties, X, strategy, method):
        first_proba, second_proba = binary_probas
        first_unc, second_unc = knowledge_uncertainties
        labels = np.array(["active", "inactive"])
        ensemble = CVEnsembleClassifierMother(
            [
                _FakeClassifier(first_proba, first_unc, classes=labels),
                _FakeClassifier(second_proba, second_unc, classes=labels),
            ]
        )

        result = ensemble.predict_uncertainty(X, aggregation_strategy=strategy, uncertainty_method=method)

        np.testing.assert_array_equal(result["pred"].to_numpy(), np.array(["active", "inactive", "inactive"]))

    @pytest.mark.parametrize("method", UNCERTAINTY_METHODS)
    def test_identical_members_have_zero_uncertainty(self, binary_probas, knowledge_uncertainties, X, method):
        if method == "member_average":
            pytest.skip(
                "Skipping test. For identical members, the 'member_average' method returns non-zero uncertainty."
            )

        first_proba = binary_probas[0]
        first_unc = knowledge_uncertainties[0]
        ensemble = CVEnsembleClassifierMother(
            [
                _FakeClassifier(first_proba, first_unc, classes=[0, 1]),
                _FakeClassifier(first_proba.copy(), first_unc.copy(), classes=[0, 1]),
            ]
        )

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

    def test_three_members(self, binary_probas, knowledge_uncertainties, X):
        first, second = binary_probas
        third = np.array([[0.1, 0.9], [0.1, 0.9], [0.2, 0.8]])
        first_unc, second_unc = knowledge_uncertainties
        third_unc = np.array([0.9, 0.8, 0.2])
        ensemble = CVEnsembleClassifierMother(
            [
                _FakeClassifier(first, first_unc, classes=[0, 1]),
                _FakeClassifier(second, second_unc, classes=[0, 1]),
                _FakeClassifier(third, third_unc, classes=[0, 1]),
            ]
        )

        result = ensemble.predict_uncertainty(X)
        expected = np.argmax(np.mean(np.stack([first, second, third], axis=0), axis=0), axis=1)

        assert ensemble._predict_probas_and_uncertainties(X).shape == (3, len(X), 2)
        np.testing.assert_array_equal(result["pred"].to_numpy(), expected)


class TestEnsembleClassifierErrorHandling:
    def test_empty_uncertainty_raises(self, binary_probas, knowledge_uncertainties, X):
        first, second = binary_probas
        first_unc, second_unc = knowledge_uncertainties
        ensemble = CVEnsembleClassifierMother(
            [
                _ClassifierWithoutKnowledgeUncertaintyValues(first, classes=[0, 1]),
                _FakeClassifier(second, second_unc, classes=[0, 1]),
            ]
        )
        with pytest.raises(ValueError, match=r"No valid member uncertainties available"):
            ensemble.predict_uncertainty(X, uncertainty_method="member_average")

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
