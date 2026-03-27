import pytest

import mother.ml as ml


def test_get_available_algorithms_returns_list():
    algos = ml.get_available_algorithms()
    assert isinstance(algos, list)
    # Should be lowercase strings
    assert all(isinstance(a, str) for a in algos)


@pytest.mark.serial
def test_algo_is_supported_true_false():
    assert ml.algo_is_supported("catboost")
    assert not ml.algo_is_supported("baz")


@pytest.mark.serial
def test_get_model_class_keyerror():
    with pytest.raises(KeyError):
        ml.get_model_class("NotExist")


@pytest.mark.serial
def test_get_model_class_by_algorithm_keyerror():
    assert len(ml.get_model_class_by_algorithm("notfound")) == 0


@pytest.mark.serial
def test_describe_model_keyerror():
    with pytest.raises(KeyError):
        ml.describe_model("NotExist")


def test_get_model_class_by_algorithm_and_type_catboost():
    from mother.ml.models.m_catboost import (
        CatboostClassifierMother,
        CatboostRegressorMother,
    )

    assert ml.get_model_class_by_algorithm_and_type("catboost", "classification_binary") is CatboostClassifierMother
    assert ml.get_model_class_by_algorithm_and_type("catboost", "classification_multiclass") is CatboostClassifierMother
    assert ml.get_model_class_by_algorithm_and_type("catboost", "regression") is CatboostRegressorMother
    # Unknown type
    with pytest.raises(ValueError):
        ml.get_model_class_by_algorithm_and_type("catboost", "classification_unknown")
    # Unknown algorithm
    with pytest.raises(ValueError):
        ml.get_model_class_by_algorithm_and_type("unknown", "classification_binary")


def test_get_model_class_by_algorithm_and_type_invalid_type():
    with pytest.raises(ValueError):
        ml.get_model_class_by_algorithm_and_type("catboost", "foo_bar")
