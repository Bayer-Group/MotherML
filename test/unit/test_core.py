from unittest.mock import MagicMock

import pytest
from optuna.trial import Trial
from sklearn.base import BaseEstimator, TransformerMixin

from mother.ml import core


class DummyStep(core.AbstractMotherPipeline, BaseEstimator, TransformerMixin):
    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = ""):
        return {f"{prefix}param": 1}

    def set_params(self, **params):
        self._params = params
        return self

    def default_parameters(self, prefix: str = "") -> dict:
        return {f"{prefix}param": 1}

    def get_params(self, deep=True):
        return getattr(self, "_params", {})


def test_dummy_step_hyperparameter_space():
    step = DummyStep()
    trial = MagicMock(spec=Trial)
    result = step.get_hyperparameter_space(None, None, trial)
    assert "param" in result


def test_dummy_step_set_get_params():
    step = DummyStep()
    step.set_params(a=1, b=2)
    params = step.get_params()
    assert params == {"a": 1, "b": 2}


def test_default_parameters_returns_non_empty_dict():
    step = DummyStep()
    assert step.default_parameters(prefix="pref__") == {"pref__param": 1}


def test_get_all_params_warns_and_returns_empty(caplog):
    step = DummyStep()
    with caplog.at_level("WARNING"):
        result = step.get_all_params()
    assert result == {}
    assert "Could not find get_all_params" in caplog.text


class DummyPipeline(core.PipelineWithHyperparameterRooting):
    def __init__(self):
        steps = [("step1", DummyStep()), ("step2", DummyStep())]
        super().__init__(steps)


def test_pipeline_with_hyperparameter_rooting():
    pipe = DummyPipeline()
    trial = MagicMock(spec=Trial)
    space = pipe.get_hyperparameter_space(None, None, trial)
    assert "step1__param" in space and "step2__param" in space


def test_pipeline_default_parameters():
    pipe = DummyPipeline()
    params = pipe.default_parameters()
    assert "step1__param" in params and "step2__param" in params


class DummyColumnTransformer(core.ColumnTransformerWithHyperparameterRooting):
    def __init__(self):
        transformers = [("trans1", DummyStep(), [0]), ("trans2", DummyStep(), [1])]
        super().__init__(transformers)


def test_column_transformer_with_hyperparameter_rooting():
    ct = DummyColumnTransformer()
    trial = MagicMock(spec=Trial)
    space = ct.get_hyperparameter_space(None, None, trial)
    assert "trans1__param" in space and "trans2__param" in space


def test_column_transformer_default_parameters():
    ct = DummyColumnTransformer()
    params = ct.default_parameters()
    assert "trans1__param" in params and "trans2__param" in params


class DummyFeatureUnion(core.FeatureUnionWithHyperparameterRooting):
    def __init__(self):
        transformer_list = [("f1", DummyStep()), ("f2", DummyStep())]
        super().__init__(transformer_list)


def test_feature_union_with_hyperparameter_rooting():
    fu = DummyFeatureUnion()
    trial = MagicMock(spec=Trial)
    space = fu.get_hyperparameter_space(None, None, trial)
    assert "f1__param" in space and "f2__param" in space


def test_feature_union_default_parameters():
    fu = DummyFeatureUnion()
    params = fu.default_parameters()
    assert "f1__param" in params and "f2__param" in params


def test_abstract_methods_raise():
    class Incomplete(core.AbstractMotherPipeline):
        def get_hyperparameter_space(self, X, y, trial, prefix=""):
            raise NotImplementedError()

        def set_params(self, **params):
            raise NotImplementedError()

        def get_params(self, deep=True):
            raise NotImplementedError()

    obj = Incomplete()
    with pytest.raises(NotImplementedError):
        obj.get_hyperparameter_space(None, None, MagicMock())
    with pytest.raises(NotImplementedError):
        obj.set_params()
    with pytest.raises(NotImplementedError):
        obj.get_params()
