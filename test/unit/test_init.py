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


@pytest.mark.parametrize(
    "module_name, expected_algorithm",
    [
        # Existing modules keep their current (correct) names
        ("m_catboost", "catboost"),
        ("m_lasso", "lasso"),
        ("m_randomForest", "randomforest"),
        ("m_tabicl", "tabicl"),
        ("m_tabpfn", "tabpfn"),
        # Regression test for #68: algorithm names starting with 'm' were
        # mangled by str.lstrip, which strips a *set* of characters instead
        # of a prefix (e.g. 'm_mlp' became 'lp')
        ("m_mlp", "mlp"),
        ("m_mars", "mars"),
        ("m_mlr", "mlr"),
        ("m_m_mixture", "m_mixture"),
    ],
)
def test_algorithm_from_module_name(module_name, expected_algorithm):
    from mother.ml import _algorithm_from_module_name

    assert _algorithm_from_module_name(module_name) == expected_algorithm


def test_available_algorithms_are_well_formed_module_names():
    from pathlib import Path

    import mother.ml.models as models_pkg

    from mother.ml import _algorithm_from_module_name

    models_dir = Path(models_pkg.__file__).parent
    expected = {_algorithm_from_module_name(f.stem) for f in models_dir.glob("m_*.py")}
    registered = set(ml.get_available_algorithms())

    # Every registered algorithm token is non-empty and correctly derived
    # from an existing 'm_*.py' module filename (no mangled names like 'lp')
    assert registered
    assert all(algorithm for algorithm in registered)
    assert registered <= expected
    assert "catboost" in registered