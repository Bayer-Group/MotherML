import logging
import typing

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import sklearn.feature_selection as skl_feature_sel
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection
from rdkit import Chem
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mother import ml
from mother.ml import estimators as mother_estimators
from mother.ml.estimators import MotherBorutaPy
from mother.pipeline_utils import (
    get_feature_generation_pipeline,
    get_feature_selection_pipeline,
    get_groups,
    get_importance_selector,
    get_model,
    get_preprocessing_pipeline,
    get_ranking_pipeline,
    report_feature_selection,
)
from mother.settings import MotherSettings


def test_get_groups_tanimoto_grouping(settings: MotherSettings) -> None:
    assert settings.cv is not None
    molecule_col: str = "Molecule"
    settings.cv.parameters["similarity_threshold"] = 0.9

    mol_data = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    mol_data[molecule_col] = mol_data["smiles"].apply(Chem.MolFromSmiles)

    result = get_groups(settings, mol_data[molecule_col])

    assert isinstance(result, pd.DataFrame)
    assert len(result[result.columns[0]].value_counts()) == len(mol_data)


def test_get_feature_selection_pipeline_no_feature_selection(settings: MotherSettings) -> None:
    settings.model.feature_selection_flags = []
    settings.pipeline.remainder = "passthrough"
    settings.pipeline.verbose_feature_names_out = False

    data: pd.DataFrame = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    transformer: ColumnTransformer = get_feature_selection_pipeline(settings=settings, data=data)
    transformer.set_output(transform=settings.pipeline.transform)
    result: pd.DataFrame = transformer.fit_transform(data)
    pdt.assert_frame_equal(data, result)


def test_get_feature_selection_pipeline_impute_categorical(settings: MotherSettings) -> None:
    settings.model.feature_selection_flags = ["IMPUTE_CATEGORICAL"]
    settings.model.categorical_features = ["A"]
    settings.pipeline.remainder = "passthrough"

    data: pd.DataFrame = pd.DataFrame({"A": ["cat", "dog", None], "B": [3, 4, 5]})
    assert data[settings.model.categorical_features].isnull().any().any()

    transformer: ColumnTransformer = get_feature_selection_pipeline(settings=settings, data=data)
    assert transformer is not None
    transformer.set_output(transform=settings.pipeline.transform)
    result: pd.DataFrame = transformer.fit_transform(data)
    assert result[settings.model.categorical_features].notnull().any().any()
    assert any("categorical_encoder" in step for step in transformer.named_transformers_)


def test_get_feature_selection_pipeline_drop_duplicates(settings: MotherSettings) -> None:
    settings.model.feature_selection_flags = ["DROP_DUPLICATES"]
    settings.model.categorical_features = []
    settings.pipeline.remainder = "passthrough"

    data: pd.DataFrame = pd.DataFrame({"A": [1, 0], "B": [1, 0]})

    transformer: ColumnTransformer = get_feature_selection_pipeline(settings=settings, data=data)
    assert transformer is not None
    transformer.set_output(transform=settings.pipeline.transform)
    result: pd.DataFrame = transformer.fit_transform(data)  # type: ignore
    assert any("feature_selector" in step for step in transformer.named_transformers_)
    for trans in transformer.named_transformers_:
        assert any("duplicate_selector" in name for name, sel in transformer.named_transformers_[trans].steps)
    assert len(result.columns) == 1


def test_get_feature_selection_pipeline_drop_constant(settings: MotherSettings) -> None:
    settings.model.feature_selection_flags = ["DROP_CONSTANT"]
    settings.model.feature_selection_type = "catboost"
    settings.pipeline.remainder = "passthrough"
    data: pd.DataFrame = pd.DataFrame({"A": [1, 1], "B": [3, 3], "remaining_feature": [1, 0]})

    transformer: ColumnTransformer = get_feature_selection_pipeline(settings=settings, data=data)
    assert transformer is not None
    transformer.set_output(transform=settings.pipeline.transform)
    result: pd.DataFrame = transformer.fit_transform(data)  # type: ignore
    for trans in transformer.named_transformers_:
        assert any("constant_selector" in name for name, sel in transformer.named_transformers_[trans].steps)
    assert len(result.columns) == 1
    assert "remaining_feature" in result.columns


def test_get_feature_selection_pipeline_categorical_features_mismatch(settings: MotherSettings) -> None:
    settings.model.feature_selection_flags = ["DROP_CONSTANT"]
    settings.pipeline.remainder = "passthrough"
    settings.model.categorical_features = ["A"]

    data: pd.DataFrame = pd.DataFrame({"A": ["cat", "dog"], "B": [3, 4], "C": ["foo", "bar"]})

    with pytest.raises(
        ValueError, match="Categorical features are not matching the provided categorical features list"
    ):
        get_feature_selection_pipeline(settings=settings, data=data)


@pytest.fixture
def pearson_correlation_callable():
    """
    Fixture that returns a callable for Pearson correlation,
    which only returns the correlation score.
    """

    def pearson_correlation_only(x, y):
        return pearsonr(x, y)[0]

    return pearson_correlation_only


@pytest.fixture(params=["default", "string", "callable"])
def correlation_type_fixture(request):
    """
    Fixture that returns three different strings for requests.
    """
    return request.param


@pytest.mark.parametrize(
    "algorithm, model_type, parameters",
    [
        ("catboost", "regression", None),
        ("catboost", "classification_multiclass", None),
        ("lasso", "regression", {"alpha": 0.1}),
        ("lasso", "classification_binary", {"C": 0.1}),
        ("lasso", "classification_multiclass", {"C": 0.1}),
        ("randomforest", "regression", {"n_estimators": 100}),
        ("randomforest", "classification_multiclass", {"n_estimators": 100}),
        ("randomforest", "classification_binary", {"n_estimators": 100}),
    ],
)
def test_get_model(
    settings: MotherSettings,
    algorithm: str,
    model_type: str,
    parameters: typing.Optional[dict],
) -> None:
    settings.model.algorithm = algorithm
    settings.model.model_type = model_type  # type: ignore
    if parameters is not None:
        settings.model.parameters = parameters
    model = get_model(settings)
    assert model is not None
    assert isinstance(model, ml.AbstractMotherPipeline)


def test_get_model_raises_error(settings: MotherSettings) -> None:
    settings.model.algorithm = "unsupported_algo"
    settings.model.model_type = "regression"  # type: ignore
    with pytest.raises(ValueError, match="Unsupported algorithm: unsupported_algo"):
        get_model(settings)


def test_get_model_raises_invalid_parameters(settings: MotherSettings) -> None:
    settings.model.algorithm = "lasso"
    settings.model.model_type = "regression"  # type: ignore
    settings.model.parameters = {"invalid_param": "value"}  # Invalid parameter for Lasso
    with pytest.raises(ValueError, match="Invalid parameters for"):
        get_model(settings)


def test_get_feature_selection_pipeline_drop_correlated(
    settings: MotherSettings, pearson_correlation_callable, correlation_type_fixture, caplog
) -> None:
    caplog.set_level(logging.INFO)

    settings.model.feature_selection_flags = ["DROP_CORRELATED"]
    settings.pipeline.remainder = "passthrough"
    settings.model.categorical_features = ["categorical_A"]
    data: pd.DataFrame = pd.DataFrame(
        {
            "constant_A": [1, 1],
            "constant_B": [3, 3],
            "correlated_A": [1, 2],
            "correlated_B": [1, 2],
            "remaining_feature": [1, 9],
            "dupl_A": [1, 0],
            "dupl_B": [1, 0],
            "categorical_A": ["cat", None],
        }
    )

    if correlation_type_fixture == "default":
        transformer: ColumnTransformer = get_feature_selection_pipeline(settings=settings, data=data)
        assert "The default correlation method is used" in caplog.text
    elif correlation_type_fixture == "string":
        transformer: ColumnTransformer = get_feature_selection_pipeline(settings=settings, data=data, method="spearman")
        assert "A string was passed as the correlation method" in caplog.text
    else:
        transformer: ColumnTransformer = get_feature_selection_pipeline(
            settings=settings, data=data, method=pearson_correlation_callable
        )
        assert "A callable was passed as the correlation method" in caplog.text
    assert transformer is not None
    transformer.set_output(transform=settings.pipeline.transform)
    result: pd.DataFrame = transformer.fit_transform(data)  # type: ignore
    assert len(result.columns) == 4


def test_get_feature_selection_pipeline_multiple(settings: MotherSettings) -> None:
    settings.model.feature_selection_flags = [
        "DROP_CONSTANT",
        "DROP_CORRELATED",
        "DROP_DUPLICATES",
        "IMPUTE_CATEGORICAL",
    ]
    settings.pipeline.remainder = "passthrough"
    settings.model.categorical_features = ["categorical_A"]
    data: pd.DataFrame = pd.DataFrame(
        {
            "constant_A": [1, 1],
            "constant_B": [3, 3],
            "correlated_A": [1, 2],
            "correlated_B": [1, 2],
            "remaining_feature": [1, 9],
            "dupl_A": [1, 0],
            "dupl_B": [1, 0],
            "categorical_A": ["cat", None],
        }
    )
    transformer: ColumnTransformer = get_feature_selection_pipeline(settings=settings, data=data)
    assert transformer is not None
    transformer.set_output(transform=settings.pipeline.transform)
    result: pd.DataFrame = transformer.fit_transform(data)  # type: ignore
    assert len(result.columns) == 2


def test_report_feature_selection_logs_info(caplog, settings: MotherSettings):
    caplog.set_level(logging.INFO)

    data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": [9, 10, 11, 12]})

    transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                ["A", "B"],
            )
        ]
    )

    transformer.fit(data)
    report_feature_selection(transformer, data)

    assert "Analyzing feature selection pipeline" in caplog.text
    assert "Analyzing num transformer" in caplog.text
    assert "Nested pipeline found" in caplog.text
    assert "Pipeline step: imputer" in caplog.text
    assert "Pipeline step: scaler" in caplog.text


def test_report_feature_selection_with_correlated_features(caplog, settings: MotherSettings):
    caplog.set_level(logging.INFO)

    data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 2, 3, 4], "C": [9, 10, 11, 12]})

    transformer = ColumnTransformer(
        transformers=[("correlation_selector", SmartCorrelatedSelection(threshold=0.9), ["A", "B"])]
    )

    transformer.fit(data)
    report_feature_selection(transformer, data)

    assert "Analyzing feature selection pipeline" in caplog.text
    assert "Analyzing correlation_selector transformer" in caplog.text
    assert "Correlated features: " in caplog.text


def test_report_feature_selection_with_features_to_drop(caplog, settings: MotherSettings):
    caplog.set_level(logging.INFO)

    data: pd.DataFrame = pd.DataFrame({"A": [1, 1, 1, 1], "B": [2, 2, 2, 2], "C": [3, 4, 5, 6]})

    transformer = ColumnTransformer(
        transformers=[("constant_selector", DropConstantFeatures(), ["A", "B", "C"])],
        **settings.pipeline.model_dump(include={"remainder", "n_jobs", "verbose_feature_names_out"}),
    )

    transformer.fit(data)
    report_feature_selection(transformer, data)

    assert "Analyzing feature selection pipeline" in caplog.text
    assert "Analyzing constant_selector transformer" in caplog.text
    assert "Features to drop: " in caplog.text


@pytest.fixture(params=[True, False])
def use_boruta(request):
    """
    Fixture that provides True and False values for testing the use of Boruta.
    """
    return request.param


def test_get_importance_selector_catboost(
    settings: MotherSettings, features: pd.DataFrame, data: pd.DataFrame, use_boruta
) -> None:
    settings.model.algorithm = "catboost"
    settings.model.model_type = "regression"
    settings.model.feature_selection_type = "catboost"
    settings.model.feature_selection_threshold = 0.01
    settings.model.feature_selection_max_features = 10

    if use_boruta:
        selector_boruta: MotherBorutaPy = get_importance_selector(
            model_settings=settings.model.model_dump(), cv=None, use_boruta=use_boruta
        )
        assert isinstance(selector_boruta, MotherBorutaPy)
    else:
        selector: skl_feature_sel.SelectFromModel = get_importance_selector(
            model_settings=settings.model.model_dump(), cv=None, use_boruta=use_boruta
        )
        assert selector.threshold == 0.01
        assert selector.max_features == 10
        assert selector is not None
        assert isinstance(selector.estimator, mother_estimators.MotherCatboostImportance)
        # selector.fit_transform(features, data[settings.target_columns])


def test_get_importance_selector_permutation(settings: MotherSettings, features: pd.DataFrame, cv, use_boruta) -> None:
    settings.model.algorithm = "catboost"
    settings.model.model_type = "classification_binary"
    settings.model.feature_selection_type = "permutation"
    settings.model.feature_selection_threshold = 0.01
    settings.model.feature_selection_max_features = 10

    if use_boruta:
        selector_boruta: MotherBorutaPy = get_importance_selector(
            model_settings=settings.model.model_dump(), cv=None, use_boruta=use_boruta
        )
        assert isinstance(selector_boruta, MotherBorutaPy)
    else:
        selector: skl_feature_sel.SelectFromModel = get_importance_selector(
            model_settings=settings.model.model_dump(), cv=cv, use_boruta=use_boruta
        )
        assert selector is not None
        assert isinstance(selector.estimator, mother_estimators.MotherPermutationImportance)
        assert selector.threshold == 0.01
        assert selector.max_features == 10


def test_get_importance_selector_unsupported_algorithm(settings: MotherSettings, features: pd.DataFrame) -> None:
    settings.model.algorithm = "unsupported_algo"
    with pytest.raises(ValueError, match="Unsupported algorithm: unsupported_algo"):
        get_importance_selector(model_settings=settings.model.model_dump(), cv=None)


@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.parametrize("cv_cross_validation", [False, True], indirect=True, ids=["no_cv", "with_cv"])
def test_get_ranking_pipeline(modeling_data, cv_cross_validation, preserve_metadata_routing):
    features, output_type, target, categorical_features, ranking_groups, cv_groups = modeling_data
    if "ranking" not in output_type:
        pytest.skip("Skipping ranking pipeline test as output_type does not include 'ranking'.")

    pipeline, tuner = get_ranking_pipeline(
        categorical_features=categorical_features,
        k_scorer=3,
        tuner_kwargs={
            "n_trials_optuna": 3,
        },
    )

    # test fitting
    pipeline["model"].fit(features, target, group_id=ranking_groups)
    pipeline.fit(features, target, group_id=ranking_groups)

    # test prediction
    _ = pipeline.predict(features)

    # test hyperparameter optimization
    _ = tuner.optimize(
        pipeline,
        X=features,
        y=target,
        cross_validation=cv_cross_validation,
        groups=cv_groups if isinstance(cv_cross_validation, GroupKFold) else None,
        ranking_groups=ranking_groups,
    )


def test_component_compatibility(
    caplog,
    settings: MotherSettings,
    data: pd.DataFrame,
) -> None:
    caplog.set_level(logging.INFO)
    # Use the preprocessing pipeline as documented
    prep_pipeline = get_preprocessing_pipeline(settings=settings)
    prep_pipeline.set_output(transform=settings.pipeline.transform)
    feature_pipeline = get_feature_generation_pipeline(settings=settings)
    feature_pipeline.set_output(transform=settings.pipeline.transform)

    # Transform
    prep_result: pd.DataFrame = prep_pipeline.fit_transform(data[["smiles"]])
    logging.info(f"Preprocessed shape: {prep_result.shape}")
    logging.info(f"Preprocessed sample: {prep_result}")
    assert prep_result.shape[1] == 1  # Still one column after preprocessing
    assert not prep_result.isna().any().any()  # Check for NaNs in the preprocessed result
    features: pd.DataFrame = feature_pipeline.fit_transform(prep_result)
    logging.info(f"NaN count: {np.isnan(features).sum()}")
    assert not features.isna().any().any()  # No NaNs after feature generation
