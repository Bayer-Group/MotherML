import logging
import typing

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
    get_cv_folds,
    get_feature_generation_pipeline,
    get_feature_importance,
    get_feature_selection_pipeline,
    get_groups,
    get_importance_selector,
    get_model,
    get_preprocessing_pipeline,
    get_preprocessing_steps,
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
    # Invalid parameter for Lasso
    settings.model.parameters = {"invalid_param": "value"}
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

    # Metadata routing must be enabled for ranking fit and optimize calls.
    # get_ranking_pipeline() no longer sets this permanently; the
    # preserve_metadata_routing fixture restores the original value on teardown.
    import sklearn as _sklearn

    _sklearn.set_config(enable_metadata_routing=True)

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


def test_get_cv_folds_groups(settings: MotherSettings) -> None:
    """Test get_cv_folds with GROUPS cv_type"""
    from mother import cv as cv_module

    settings.cv.cv_type = cv_module.CVtype.GROUPS
    settings.cv.n_splits = 3
    settings.model.model_type = "regression"

    group_data = pd.DataFrame({"group": [1, 1, 2, 2, 3, 3]})

    cv = get_cv_folds(settings, group_data)
    assert isinstance(cv, GroupKFold)
    assert cv.n_splits == 3


def test_get_cv_folds_tanimoto_grouping(settings: MotherSettings) -> None:
    """Test get_cv_folds with TANIMOTO_GROUPING cv_type"""
    from mother import cv as cv_module

    settings.cv.cv_type = cv_module.CVtype.TANIMOTO_GROUPING
    settings.cv.n_splits = 2
    settings.model.model_type = "classification_binary"

    group_data = pd.DataFrame({"group": [1, 1, 2, 2]})

    cv = get_cv_folds(settings, group_data)
    from sklearn.model_selection import StratifiedGroupKFold

    assert isinstance(cv, StratifiedGroupKFold)
    assert cv.n_splits == 2


def test_get_cv_folds_time_series(settings: MotherSettings) -> None:
    """Test get_cv_folds with TIME_SERIES cv_type"""
    from sklearn.model_selection import TimeSeriesSplit

    from mother import cv as cv_module

    settings.cv.cv_type = cv_module.CVtype.TIME_SERIES
    settings.cv.n_splits = 4

    cv = get_cv_folds(settings)
    assert isinstance(cv, TimeSeriesSplit)
    assert cv.n_splits == 4


def test_get_cv_folds_missing_group_data(settings: MotherSettings) -> None:
    """Test get_cv_folds raises error when group data is required but not provided"""
    from mother import cv as cv_module

    settings.cv.cv_type = cv_module.CVtype.GROUPS
    with pytest.raises(ValueError, match="Group data is required for group-based cross-validation"):
        get_cv_folds(settings, group_data=None)


def test_get_cv_folds_with_nan_groups(settings: MotherSettings) -> None:
    """Test get_cv_folds raises error when group data contains NaN"""
    from mother import cv as cv_module

    settings.cv.cv_type = cv_module.CVtype.GROUPS
    group_data = pd.DataFrame({"group": [1, 2, None, 4]})

    with pytest.raises(ValueError, match="Group column contains missing values"):
        get_cv_folds(settings, group_data)


def test_get_preprocessing_steps(settings: MotherSettings) -> None:
    """Test get_preprocessing_steps returns correct steps"""
    steps = get_preprocessing_steps(settings, molecule_col="mol_col", smiles_col="smi_col")

    assert len(steps) == 2
    assert steps[0][0] == "smiles_standardizer"
    assert steps[1][0] == "smiles_to_mol"

    from mother.preprocessing.core import (
        SmilesToMolTransformer,
        StandardizerTransformer,
    )

    assert isinstance(steps[0][1], StandardizerTransformer)
    assert isinstance(steps[1][1], SmilesToMolTransformer)


def test_get_preprocessing_pipeline(settings: MotherSettings) -> None:
    """Test get_preprocessing_pipeline returns a valid pipeline"""
    pipeline = get_preprocessing_pipeline(settings)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert "smiles_standardizer" in pipeline.named_steps
    assert "smiles_to_mol" in pipeline.named_steps


def test_get_feature_generation_pipeline_with_maccs(settings: MotherSettings) -> None:
    """Test get_feature_generation_pipeline with MACCS fingerprints"""
    settings.feature_generation.maccs = True
    settings.feature_generation.chemical_descriptors = None
    settings.feature_generation.fingerprints = []

    pipeline = get_feature_generation_pipeline(settings)

    assert isinstance(pipeline, ml.FeatureUnionWithHyperparameterRooting)
    assert len(pipeline.transformer_list) == 1
    assert pipeline.transformer_list[0][0] == "Maccs"


def test_get_feature_generation_pipeline_with_descriptors(settings: MotherSettings) -> None:
    """Test get_feature_generation_pipeline with chemical descriptors"""
    from mother.feature_generation.config import ChemicalDescriptorsParams

    settings.feature_generation.maccs = False
    settings.feature_generation.chemical_descriptors = ChemicalDescriptorsParams()
    settings.feature_generation.fingerprints = []

    pipeline = get_feature_generation_pipeline(settings)

    assert isinstance(pipeline, ml.FeatureUnionWithHyperparameterRooting)
    assert len(pipeline.transformer_list) == 1
    assert pipeline.transformer_list[0][0] == "Desc"


def test_get_feature_generation_pipeline_with_multiple_fingerprints(settings: MotherSettings) -> None:
    """Test get_feature_generation_pipeline with multiple fingerprint types"""
    settings.feature_generation.maccs = True
    settings.feature_generation.chemical_descriptors = None
    settings.feature_generation.fingerprints = [
        {"AtomPairFP": {"nBits": 2048}},
        {"MorganFP": {"radius": 2, "nBits": 2048}},
    ]

    pipeline = get_feature_generation_pipeline(settings)

    assert isinstance(pipeline, ml.FeatureUnionWithHyperparameterRooting)
    assert len(pipeline.transformer_list) == 3  # Maccs + AtomPair + Morgan
    transformer_names = [t[0] for t in pipeline.transformer_list]
    assert "Maccs" in transformer_names
    assert "AtomPairFP" in transformer_names
    assert "MorganFP" in transformer_names


def test_get_feature_importance_no_importance_attribute() -> None:
    """Test get_feature_importance raises error when model doesn't have feature importance"""
    from sklearn.linear_model import LinearRegression

    X = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    y = pd.Series([1, 2, 3])

    model = LinearRegression()
    pipeline = Pipeline([("model", model)], memory=None)
    pipeline.fit(X, y)

    with pytest.raises(ValueError, match="Model does not have feature importance"):
        get_feature_importance(pipeline, model_step_name="model")
