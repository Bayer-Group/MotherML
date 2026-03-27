import pickle
from typing import Type

import pandas as pd
import pandas.testing as pdt
import pytest
from rdkit import Chem
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion

import mother.feature_generation as fg
from mother import ml
from mother.feature_generation import config as fg_config
from mother.feature_generation.config import FeatureGenerationConfig, FingerprintParams
from mother.ml.core import PipelineWithHyperparameterRooting
from mother.settings import MotherSettings


def test_config_from_yaml(settings: MotherSettings) -> None:
    assert settings is not None


def test_preprocessing_pipeline(preprocessor, data, settings: MotherSettings):
    structure_data = data[settings.input.structure_col]
    mol_data = preprocessor.fit_transform(structure_data)
    assert not mol_data.empty
    assert mol_data["molecule"].apply(lambda x: isinstance(x, Chem.rdchem.Mol)).all()


def test_feature_generation_pipeline(feature_generator, mol_data, settings: MotherSettings):
    features = feature_generator.fit_transform(mol_data["molecule"])
    assert not features.empty


@pytest.fixture(params=["regressor", "classifier_binary", "classifier_multiclass"])
def base_model(request):
    # default to CatboostRegressorMother
    # target type set to non default value to test pickling
    target_type: str = "multi_target"
    model = ml.CatboostRegressorMother(target_type=target_type)
    if request.param == "classifier_binary":
        model = ml.CatboostClassifierMother(target_type=target_type, model_type="classification_binary")
    elif request.param == "classifier_multiclass":
        model = ml.CatboostClassifierMother(target_type="single_target", model_type="classification_multiclass")
    # Check if the model is a valid model
    assert isinstance(model, (ml.CatboostRegressorMother, ml.CatboostClassifierMother)), (
        "Model should be a valid Catboost model"
    )
    return model


def test_pickle_works_for_model(base_model):
    # Attempt to pickle and unpickle the model
    try:
        pickled_model = pickle.dumps(base_model)
        assert pickled_model
        unpickled_model = pickle.loads(pickled_model)
        assert unpickled_model
    except Exception as e:
        pytest.fail(f"Model could not be pickled: {e}")

    assert base_model.target_type == unpickled_model.target_type, "Target type should be the same after pickling"


def test_model_target_type_attribute():
    model = ml.CatboostRegressorMother()
    assert hasattr(model, "target_type"), "CatboostRegressorMother should have target_type attribute"
    pickled_model = pickle.dumps(model)
    assert pickled_model
    unpickled_model = pickle.loads(pickled_model)
    assert unpickled_model
    assert hasattr(unpickled_model, "target_type"), "CatboostRegressorMother should have target_type attribute"


def test_model_training(model, features, data, settings: MotherSettings):
    model.fit(features, data[settings.input.target_columns])
    targets_pred = model.predict(features)
    assert len(targets_pred) == len(data[settings.input.target_columns])


def test_cross_validation(model, features, data, groups, cv, settings: MotherSettings):
    cross_val_scores = cross_validate(model, features, data[settings.input.target_columns], groups=groups, cv=cv)
    assert "test_score" in cross_val_scores


def test_hyperparameter_optimization(tuner, model, features, data, groups, cv, settings: MotherSettings):
    tuner.optimize(
        estimator=model,
        X=features,
        y=data[settings.input.target_columns],
        cross_validation=cv,
        groups=groups.to_numpy() if groups is not None else None,
    )
    assert model.get_params() != model.default_parameters()


def test_inference(model, features, data, settings: MotherSettings):
    model.fit(features, data[settings.input.target_columns])
    predictions = model.predict(features)
    assert len(predictions) == len(data[settings.input.target_columns])
    # TODO, set_output does not work yet
    predictions_df: pd.DataFrame = pd.DataFrame(predictions)
    assert not any(predictions_df.isnull())
    assert not any(predictions_df.isna())
    assert predictions_df.dtypes[0] == "float64"
    assert not all(predictions_df == 0)


@pytest.mark.parametrize(
    "pipeline_step,input_data,column_name",
    [
        ("preprocessor", "data", "smiles"),
        ("feature_generator", "mol_data", "molecule"),
        # ("model",""),
    ],
)
def test_pipeline_pickling(pipeline_step, input_data, column_name, request):
    step = request.getfixturevalue(pipeline_step)
    data = request.getfixturevalue(input_data)
    # Attempt to pickle and unpickle each part of the pipeline
    try:
        pickled_pipeline = pickle.dumps(step)
        assert pickled_pipeline
        unpickled_pipeline = pickle.loads(pickled_pipeline)
        assert unpickled_pipeline
        result: pd.DataFrame = unpickled_pipeline.fit_transform(data[column_name])
        pickled_pipeline = pickle.dumps(unpickled_pipeline)
        assert pickled_pipeline
        fitted_pipeline = pickle.loads(pickled_pipeline)
        result2: pd.DataFrame = fitted_pipeline.transform(data[column_name])
    except Exception as e:
        pytest.fail(f"PipelineConfig could not be pickled: {e}")
    if pipeline_step != "preprocessor":
        # preprocessor returns rdkit molecules that differ in stored location
        pdt.assert_frame_equal(result, result2)


def test_pickled_model_raises_unexpectedly(mol_data, settings: MotherSettings):
    # AtomPairFP does not return identical fingerprint...
    settings.feature_generation.fingerprints = []
    settings.feature_generation.fingerprints.append({"AtomPairFP": {"includeChirality": True, "maxDistance": 4}})
    fg_conf: FeatureGenerationConfig = settings.feature_generation
    transformer_list = []
    for fp in fg_conf.fingerprints:
        fp_type: str = next(iter(fp.keys()))
        params_class: Type[FingerprintParams] = fg_config.get_params_for_fp_type(fp_type)
        params: FingerprintParams = params_class(**fp[fp_type])
        transformer_list.append((fp_type, fg.FingerprintsGeneric(fp_type=fp_type, parameters=params.model_dump())))
    feature_generator = FeatureUnion(transformer_list=transformer_list).set_output(transform="pandas")
    result1: pd.DataFrame = feature_generator.fit_transform(mol_data["molecule"])  # type: ignore
    pickled_model = pickle.dumps(feature_generator)
    assert pickled_model
    unpickled_model = pickle.loads(pickled_model)
    assert unpickled_model
    result2: pd.DataFrame = unpickled_model.transform(mol_data["molecule"])
    with pytest.raises(AssertionError):
        pdt.assert_frame_equal(result1, result2)


@pytest.fixture(params=["catboost_regressor_single", "catboost_regressor_multi", "catboost_classifier"])
def model_with_uncertainty(request):
    """Create models that support predict_uncertainty."""
    if request.param == "catboost_regressor_single":
        return ml.CatboostRegressorMother(target_type="single_target")
    elif request.param == "catboost_regressor_multi":
        return ml.CatboostRegressorMother(target_type="multi_target")
    elif request.param == "catboost_classifier":
        return ml.CatboostClassifierMother(target_type="single_target", model_type="classification_multiclass")


@pytest.fixture
def model_without_uncertainty():
    """Create a model that does NOT support predict_uncertainty."""
    from mother.ml.models import m_lasso

    return m_lasso.LassoRegressorMother()


@pytest.fixture
def pipeline_with_uncertainty(model_with_uncertainty, features, data, settings: MotherSettings):
    """Create and fit a pipeline with a model that supports predict_uncertainty."""

    pipeline = PipelineWithHyperparameterRooting(
        steps=[
            ("ml_model", model_with_uncertainty),
        ]
    )
    pipeline.fit(features, data[settings.input.target_columns])
    return pipeline


@pytest.fixture
def pipeline_without_uncertainty(model_without_uncertainty, features, data, settings: MotherSettings):
    """Create and fit a pipeline with a model that does NOT support predict_uncertainty."""

    pipeline = PipelineWithHyperparameterRooting(
        steps=[
            ("ml_model", model_without_uncertainty),
        ]
    )
    pipeline.fit(features, data[settings.input.target_columns])
    return pipeline


class TestPipelineWithHyperparameterRootingPredictUncertainty:
    """Test suite for predict_uncertainty method with parameter routing."""

    def test_predict_uncertainty_without_parameters(self, pipeline_with_uncertainty, features, model_with_uncertainty):
        """Test predict_uncertainty works without any parameters."""
        result = pipeline_with_uncertainty.predict_uncertainty(features)
        assert result is not None
        assert isinstance(result, pd.DataFrame), "single_target should return DataFrame"
        assert len(result) == len(features)

    def test_predict_uncertainty_with_valid_parameters(
        self, pipeline_with_uncertainty, features, model_with_uncertainty
    ):
        """Test predict_uncertainty with valid step__param naming convention."""
        result = pipeline_with_uncertainty.predict_uncertainty(features, ml_model__n_ensembles=5)
        assert result is not None

        assert isinstance(result, pd.DataFrame), "single_target should return DataFrame"
        assert len(result) == len(features)

    def test_predict_uncertainty_with_invalid_parameter_no_separator(self, pipeline_with_uncertainty, features):
        """Test that parameter without __ separator raises ValueError."""
        with pytest.raises(ValueError, match="unexpected keyword argument"):
            pipeline_with_uncertainty.predict_uncertainty(features, n_ensembles=5)

    def test_predict_uncertainty_with_invalid_step_name(self, pipeline_with_uncertainty, features):
        """Test that parameter with wrong step name raises ValueError."""
        with pytest.raises(ValueError, match="unexpected keyword argument"):
            pipeline_with_uncertainty.predict_uncertainty(features, wrong_step__param=5)

    def test_predict_uncertainty_with_multiple_parameters(self, pipeline_with_uncertainty, features):
        """Test predict_uncertainty with multiple valid parameters to same step."""
        result = pipeline_with_uncertainty.predict_uncertainty(
            features,
            ml_model__n_ensembles=5,
        )
        assert result is not None
        assert isinstance(result, pd.DataFrame), "single_target should return DataFrame"
        assert len(result) == len(features)

    def test_predict_uncertainty_mixed_valid_invalid_parameters(self, pipeline_with_uncertainty, features):
        """Test that mixing valid and invalid parameters raises error."""
        with pytest.raises(ValueError, match="unexpected keyword argument"):
            pipeline_with_uncertainty.predict_uncertainty(
                features,
                ml_model__n_ensembles=5,
                invalid_param=10,
            )

    def test_predict_uncertainty_validates_result_structure(
        self, pipeline_with_uncertainty, features, model_with_uncertainty
    ):
        """Test that predict_uncertainty returns expected uncertainty columns."""
        result = pipeline_with_uncertainty.predict_uncertainty(features, ml_model__n_ensembles=5)
        assert result is not None

        expected_cols = ["knowledge_uncertainty", "data_uncertainty", "total_uncertainty"]
        if model_with_uncertainty.target_type == "single_target":
            assert isinstance(result, pd.DataFrame), "single_target should return DataFrame"
            assert all(col in result.columns for col in expected_cols), "Missing uncertainty columns"
            assert len(result) == len(features)
