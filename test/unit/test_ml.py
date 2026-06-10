import itertools

import pytest
import sklearn.base as skl_base
from pydantic import ValidationError
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from mother import ml
from mother.ml import config as ml_conf
from mother.ml.models.m_catboost import (
    CatboostClassifierMother,
    CatboostRankerMother,
    CatboostRegressorMother,
)


@pytest.fixture(params=ml.get_available_algorithms())
def all_classification_algorithms(request):
    algorithm = request.param
    model = CatboostClassifierMother(target_type="single_target")

    if algorithm == "randomforest":
        from mother.ml.models.m_randomForest import RandomForestClassifierMother

        model = RandomForestClassifierMother()
    elif algorithm == "tabpfn":
        from mother.ml.models.m_tabpfn import TabPFNClassifierMother

        model = TabPFNClassifierMother()
    elif algorithm == "tabicl":
        from mother.ml.models.m_tabicl import TabICLClassifierMother

        model = TabICLClassifierMother()
    elif algorithm == "lasso":
        from mother.ml.models.m_lasso import LassoClassifierBinaryMother

        model = LassoClassifierBinaryMother()
    else:
        assert algorithm == "catboost"
    return model


@pytest.fixture(params=ml.get_available_algorithms())
def all_regression_algorithms(request):
    algorithm = request.param
    model = CatboostRegressorMother(target_type="single_target")

    if algorithm == "randomforest":
        from mother.ml.models.m_randomForest import RandomForestRegressorMother

        model = RandomForestRegressorMother()
    elif algorithm == "tabpfn":
        from mother.ml.models.m_tabpfn import TabPFNRegressorMother

        model = TabPFNRegressorMother()
    elif algorithm == "tabicl":
        from mother.ml.models.m_tabicl import TabICLRegressorMother

        model = TabICLRegressorMother()
    elif algorithm == "lasso":
        from mother.ml.models.m_lasso import LassoRegressorMother

        model = LassoRegressorMother()
    else:
        assert algorithm == "catboost"
    return model


@pytest.fixture()
def ml_config() -> ml_conf.ModelConfig:
    return ml_conf.ModelConfig(
        categorical_features=["foo", "bar"],
        model_type="regression",
        target_type="single_target",
        feature_selection_type="catboost",
        algorithm="catboost",
        parameters={"iterations": 100, "learning_rate": 0.1},
        feature_selection_flags=[],
    )


def test_clone_works():
    target_type = "multi_target"
    model = CatboostClassifierMother(
        target_type=target_type, model_type="classification_binary", max_depth=2, cat_features=[], num_trees=10
    )
    cloned_model = skl_base.clone(model)  # type: ignore
    assert model.target_type == cloned_model.target_type
    assert model.model_type == cloned_model.model_type


@pytest.mark.serial
def test_ml_model(ml_model, modeling_data):
    features, _, target, _, ranking_groups, _ = modeling_data

    if isinstance(ml_model, CatboostRankerMother):
        ml_model.fit(features, target, group_id=ranking_groups)
    else:
        ml_model.fit(features, target)

    predictions = ml_model.predict(features)

    assert len(predictions) == len(target)


@pytest.mark.serial
def test_feature_selection(feature_selector, modeling_data):
    features, _, target, _, ranking_groups, _ = modeling_data
    selected_features = feature_selector.fit_transform(features, target)

    assert len(selected_features) == len(target)
    assert selected_features.shape[1] <= features.shape[1]


@pytest.mark.serial
def test_pipeline(model_pipeline, modeling_data):
    features, categoric_output, target, categorical_features, ranking_groups, _ = modeling_data

    if isinstance(model_pipeline["model"], CatboostRankerMother):
        model_pipeline.fit(features, target, group_id=ranking_groups)
    else:
        model_pipeline.fit(features, target)
    predictions = model_pipeline.predict(features)

    assert len(predictions) == len(target)


def test_ml_config_raises() -> None:
    with pytest.raises(ValidationError):
        ml_conf.ModelConfig()  # type: ignore


def test_ml_config(ml_config) -> None:
    # test ml config using required fields
    assert ml_config


@pytest.mark.parametrize(
    "algorithm, model_type",
    list(
        itertools.product(
            ml.get_available_algorithms(),
            ["classification_binary", "classification_multiclass", "regression", "ranking"],
        )
    ),
)
def test_model_class_from_config(algorithm, model_type) -> None:
    # Ensure the model class can be instantiated from the config
    if (model_type == "ranking") and (algorithm != "catboost"):
        pytest.skip("ranking is only available in catboost")
    model_class = ml.get_model_class_by_algorithm_and_type(algorithm, model_type)
    assert model_class is not None
    assert issubclass(model_class, ml.AbstractMotherPipeline)  # type: ignore


def test_predict_uncertainty_classification(all_classification_algorithms):
    """
    Test if all available classification algorithms return the same format
    of dataframe with predict_uncertainty()
    """
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    all_classification_algorithms.fit(X_train, y_train)
    pred = all_classification_algorithms.predict_uncertainty(X_test)

    assert all(
        [
            "mean_predictions" in pred.columns,
            "knowledge_uncertainty" in pred.columns,
            "data_uncertainty" in pred.columns,
            "total_uncertainty" in pred.columns,
        ]
    )


def test_predict_uncertainty_regression(all_regression_algorithms):
    """
    Test if all available regression algorithms return the same format
    of dataframe with predict_uncertainty()
    """
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    all_regression_algorithms.fit(X_train, y_train)
    pred = all_regression_algorithms.predict_uncertainty(X_test)

    assert all(
        [
            "mean_predictions" in pred.columns,
            "knowledge_uncertainty" in pred.columns,
            "data_uncertainty" in pred.columns,
            "total_uncertainty" in pred.columns,
        ]
    )
