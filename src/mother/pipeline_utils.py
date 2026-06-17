import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

import catboost
import numpy.typing as npt
import pandas as pd
import sklearn.feature_selection as skl_feature_sel
import sklearn.model_selection as skl_model_sel
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

from mother import cv as cv_module
from mother import errors
from mother import feature_generation as fg
from mother import ml
from mother import preprocessing as prep
from mother import utils as mother_utils
from mother.feature_generation import config as fg_config
from mother.ml import (
    CatboostRankerMother,
    PipelineWithHyperparameterRooting,
    avg_ndcg_score,
    properties,
    utils,
)
from mother.ml.estimators import MotherBorutaPy, MotherSelectFromModel
from mother.optimization import MotherTuner
from mother.settings import MotherSettings

module_logger: logging.Logger = logging.getLogger(__name__)


def get_groups(
    settings: MotherSettings,
    mol_data: Iterable[Any],
) -> pd.DataFrame:
    """
    Generate groups for cross-validation based on the provided settings and molecular data.

    Parameters:
        settings (MotherSettings): Configuration settings for the mother pipeline, including cross-validation settings.
        mol_data (Iterable[Any]): Iterable containing molecular data to be grouped.

    Returns:
        Any | pd.DataFrame: The grouped data as a DataFrame or any other format depending on the grouping engine used.

    Raises:
        ValueError: If CV settings are missing or if the wrong configuration is provided for Time Series usage.

    Notes:
        - If the cross-validation type is TANIMOTO_GROUPING, Tanimoto grouping is used.
        - If the cross-validation type is TIME_SERIES, Time Series grouping is used, and the data set should be sorted.
        - If a group column is provided, it will be used for grouping.
    """
    groups_engine: BaseEstimator = cv_module.DefaultGrouping()
    if not settings.cv:
        raise ValueError("CV settings missing")
    cv_conf: cv_module.GenericCVModel = settings.cv.get_cv_settings()
    if settings.cv.cv_type == cv_module.CVtype.TANIMOTO_GROUPING:
        groups_engine = cv_module.TanimotoGroupingFromMols(**cv_conf.model_dump())
    elif settings.cv.cv_type == cv_module.CVtype.TIME_SERIES:
        if not isinstance(cv_conf, cv_module.TimeSeriesParams):
            raise ValueError("Wrong configuration for Time Series usage")
        groups_engine = cv_module.TimeSeriesGrouping(cv_conf.datetime_fmt)  # type: ignore
        module_logger.info(
            "Time Series split does not require groups to be generated. Please ensure that your data set is sorted."
        )
    else:
        module_logger.info("Using provided group column")
    groups_engine = groups_engine.set_output(transform=settings.pipeline.transform)
    return groups_engine.fit_transform(mol_data)  # type: ignore


def get_cv_folds(
    settings: MotherSettings, group_data: Optional[pd.DataFrame] = None, **kwargs: Any
) -> skl_model_sel.BaseCrossValidator:
    """
    Generate a cross-validator based on the provided settings.

    Parameters:
    settings (MotherSettings): Configuration settings for cross-validation.
    group_data (Optional[pd.DataFrame]): DataFrame containing group information for group-based cross-validation.
                                         Required if cv_type is GROUPS or TANIMOTO_GROUPING.
    **kwargs (Any): Additional keyword arguments to pass to the cross-validator.

    Returns:
    skl_model_sel.BaseCrossValidator: An instance of a scikit-learn cross-validator.

    Raises:
    ValueError: If cross-validation settings are missing, group data is required but not provided,
                group data contains missing values, or an unsupported cross-validation type is specified.
    """
    if settings.cv is None:
        raise ValueError("Cross-validation settings are missing")
    if settings.cv.cv_type in [
        cv_module.CVtype.GROUPS,
        cv_module.CVtype.TANIMOTO_GROUPING,
    ]:
        if group_data is None:
            raise ValueError("Group data is required for group-based cross-validation")
        if any(group_data.isna().any()):
            raise ValueError("Group column contains missing values")
        if settings.model.model_type == "regression":
            module_logger.info(
                "Creating GroupKFold cross-validator using %d splits",
                settings.cv.n_splits,
            )
            return skl_model_sel.GroupKFold(**settings.cv.model_dump(exclude={"cv_type", "parameters"}), **kwargs)
        module_logger.info(
            "Creating StratifiedGroupKFold cross-validator using %d splits",
            settings.cv.n_splits,
        )
        return skl_model_sel.StratifiedGroupKFold(**settings.cv.model_dump(exclude={"cv_type", "parameters"}), **kwargs)
    if settings.cv.cv_type == cv_module.CVtype.TIME_SERIES:
        module_logger.info(
            "Creating TimeSeriesSplit cross-validator using %d splits",
            settings.cv.n_splits,
        )
        return skl_model_sel.TimeSeriesSplit(n_splits=settings.cv.n_splits, **kwargs)
    raise ValueError(f"Unsupported cross-validation type: {settings.cv.cv_type}")


def get_preprocessing_steps(settings: MotherSettings, **kwargs) -> List[Tuple]:
    """
    Generate a list of preprocessing steps based on the provided settings.

    Args:
        settings (MotherSettings): An instance of MotherSettings containing the preprocessing configuration.

    Returns:
        List[Tuple]: A list of tuples where each tuple contains the name of the preprocessing step
            and the corresponding transformer object.
    """
    prep_config: prep.PreprocessingConfig = settings.preprocessing
    return [
        (
            "smiles_standardizer",
            prep.StandardizerTransformer(flags=prep_config.flags),
        ),
        (
            "smiles_to_mol",
            prep.SmilesToMolTransformer(
                molecule_col=kwargs.get("molecule_col", "Molecule"),
                smiles_col=kwargs.get("smiles_col", None),
            ),
        ),
    ]


def get_preprocessing_pipeline(settings: MotherSettings) -> Pipeline:
    """
    Creates a preprocessing pipeline based on the provided settings.

    Args:
        settings (MotherSettings): The settings object containing preprocessing configuration.

    Returns:
        Pipeline: A scikit-learn Pipeline object with the specified preprocessing steps.
    """
    return Pipeline(
        steps=get_preprocessing_steps(settings),
        memory=settings.pipeline.memory,
        verbose=settings.pipeline.verbose,
    )


def get_feature_generation_pipeline(
    settings: MotherSettings,
) -> ml.FeatureUnionWithHyperparameterRooting:
    """
    Constructs a feature generation pipeline based on the provided settings.

    Args:
        settings (MotherSettings): Configuration settings for feature generation.

    Returns:
        FeatureUnionWithHyperparameterRooting: A scikit-learn FeatureUnion object that combines
            multiple feature extraction methods that enables hyperparameter tuning.

    The pipeline may include the following transformers based on the settings:
        - MaccsFingerprints: Generates MACCS fingerprints if `maccs` is enabled in the settings.
        - ChemicalDescriptors: Generates chemical descriptors if `chemical_descriptors` is enabled in the settings.
        - FingerprintsGeneric: Generates various types of fingerprints as specified in the settings.

    Each transformer is configured with parameters specified in the settings.
    """
    fg_conf: fg_config.FeatureGenerationConfig = settings.feature_generation
    module_logger.info("Creating feature generation pipeline")

    transformer_list: List[Tuple] = []
    if fg_conf.maccs:
        module_logger.debug("Adding MACCS fingerprints to the feature generation pipeline")
        maccs_params: fg_config.MaccsFingerprintsParams = fg_config.MaccsFingerprintsParams()
        transformer_list.append(("Maccs", fg.MaccsFingerprints(**maccs_params.model_dump())))
    if fg_conf.chemical_descriptors:
        module_logger.debug("Adding chemical descriptors to the feature generation pipeline")
        physchem_params: fg_config.ChemicalDescriptorsParams = fg_config.ChemicalDescriptorsParams(
            **fg_conf.chemical_descriptors.model_dump()
        )
        transformer_list.append(("Desc", fg.ChemicalDescriptors(**physchem_params.model_dump())))
    for fp in fg_conf.fingerprints:
        module_logger.debug(f"Adding {fp} fingerprints to the feature generation pipeline")
        fp_type: str = next(iter(fp.keys()))
        params_class: Type[fg_config.FingerprintParams] = fg_config.get_params_for_fp_type(fp_type)
        params: fg_config.FingerprintParams = params_class(**fp[fp_type])
        transformer_list.append(
            (
                fp_type,
                fg.FingerprintsGeneric(fp_type=fp_type, parameters=params.model_dump(), use_counts=fg_conf.use_counts),
            )
        )
    module_logger.info("Feature generation pipeline created")
    module_logger.debug("Feature generation pipeline steps: %s", transformer_list)
    return ml.FeatureUnionWithHyperparameterRooting(transformer_list=transformer_list)


def get_importance_selector(
    model_settings: Dict,
    cv: Optional[skl_model_sel.BaseCrossValidator],
    use_boruta: bool = False,
) -> Union[skl_feature_sel.SelectFromModel, MotherBorutaPy]:
    """
    Creates and returns a feature selector based on the provided settings and cross-validator.

    Args:
        settings (Dict): The settings dictionary containing model configuration.
        cv (skl_model_sel.BaseCrossValidator): The cross-validator to be used for feature selection.
        use_boruta (bool): Whether to use the Boruta feature selection algorithm.

    Returns:
        BaseEstimator: An instance of a feature selector configured according to the provided settings.

    Raises:
        ValueError: If the algorithm specified in the model settings is not supported.
    """
    selector_model: Union[catboost.CatBoostClassifier, catboost.CatBoostRegressor]
    importance_selector: Union[skl_feature_sel.SelectFromModel, MotherBorutaPy]
    if not ml.algo_is_supported(algorithm=model_settings["algorithm"]):
        raise ValueError(f"Unsupported algorithm: {model_settings['algorithm']}")

    from mother.ml import estimators

    # use default settings for permutation importance
    params = {
        "loss_function": utils.default_loss_function(model_settings["model_type"], model_settings["target_type"]),
        "allow_const_label": True,
        "silent": True,
    }
    module_logger.debug(
        "Using default Catboost parameters for feature selection: %s",
        json.dumps(params, indent=4),
    )
    if "regression" in model_settings["model_type"]:
        selector_model = catboost.CatBoostRegressor(**params)  # type: ignore
    else:
        selector_model = catboost.CatBoostClassifier(**params)  # type: ignore
    if model_settings["feature_selection_type"] == "catboost":
        module_logger.info("Setting up catboost importance feature selection")

        estimator = estimators.MotherCatboostImportance(selector_model)
        if use_boruta:
            importance_selector = MotherBorutaPy(estimator, n_estimators="auto", random_state=42)
        else:
            importance_selector = MotherSelectFromModel(
                estimator=estimator,
                threshold=model_settings.get("feature_selection_threshold", None),
                max_features=model_settings.get("feature_selection_max_features", None),
            )
    else:
        if model_settings["feature_selection_type"] != "permutation":
            raise errors.ConfigurationError("Unsupported feature selection type provided. {}")
        module_logger.info("Setting up permutation importance feature selection")
        estimator = estimators.MotherPermutationImportance(selector_model, cv=cv)
        if use_boruta:
            importance_selector = MotherBorutaPy(estimator, n_estimators="auto", random_state=42)
        else:
            importance_selector = MotherSelectFromModel(
                estimator=estimator,
                threshold=model_settings.get("feature_selection_threshold", None),
                max_features=model_settings.get("feature_selection_max_features", None),
            )

    return importance_selector


@overload
def get_feature_selection_pipeline(
    settings: MotherSettings,
    *,
    data: Optional[pd.DataFrame] = None,
    cv: Optional[skl_model_sel.BaseCrossValidator] = None,
) -> ml.ColumnTransformerWithHyperparameterRooting: ...


@overload
def get_feature_selection_pipeline(
    settings: Dict,
    *,
    pipeline_settings: Dict = {},
    data: Optional[pd.DataFrame] = None,
    cv: Optional[skl_model_sel.BaseCrossValidator] = None,
    **kwargs,
) -> ml.ColumnTransformerWithHyperparameterRooting: ...


def get_feature_selection_pipeline(
    settings: Union[Dict, MotherSettings],
    *,
    pipeline_settings: Optional[Dict] = None,
    data: Optional[pd.DataFrame] = None,
    cv: Optional[skl_model_sel.BaseCrossValidator] = None,
    **kwargs,
) -> ml.ColumnTransformerWithHyperparameterRooting:
    model_settings: Dict = settings if isinstance(settings, dict) else settings.model.model_dump()
    pipeline_settings: Dict = (
        settings.pipeline.model_dump() if isinstance(settings, MotherSettings) else pipeline_settings
    )
    if pipeline_settings is None:
        pipeline_settings = {}
    module_logger.info("Creating feature selection pipeline for numeric columns.")
    transformer_list: Sequence[
        Tuple[
            str,
            Union[BaseEstimator, ml.FeatureUnionWithHyperparameterRooting],
            Union[str, int, List[int], List[str], Callable],
        ]
    ] = []
    numeric_columns_list: Sequence[Tuple[str, Union[TransformerMixin, ml.PipelineWithHyperparameterRooting]]] = []

    if data is not None:
        module_logger.info("Determine categorical and numerical features. Categorical features are skipped")
        numerical_ix: List[str] = mother_utils.get_numeric_columns(data=data)
        categorical_ix: List[str] = mother_utils.get_categorical_column_names(data=data)
        if set(categorical_ix).difference(set(model_settings["categorical_features"])):
            raise ValueError("Categorical features are not matching the provided categorical features list")
        module_logger.debug("Numerical features: %s", numerical_ix)

    from mother.ml import config as feature_config

    flags: properties.FeatureSelectionFlags = feature_config.getFlagFromStrings(
        model_settings["feature_selection_flags"]
    )
    if flags & properties.FeatureSelectionFlags.NONE:
        module_logger.debug("No feature selection is applied")
        pipeline_settings.pop("transform", None)
        return ml.ColumnTransformerWithHyperparameterRooting(
            transformers=transformer_list,
            **pipeline_settings,
        )
    if flags & properties.FeatureSelectionFlags.IMPUTE_CATEGORICAL:
        if len(model_settings["categorical_features"]) == 0:
            raise errors.ConfigurationError("Categorical features are missing")
        from feature_engine.imputation import CategoricalImputer

        module_logger.debug("Adding CategoricalImputer to the pipeline")
        transformer_list.append(
            (
                "categorical_encoder",
                CategoricalImputer(imputation_method="missing"),
                model_settings["categorical_features"],
            )
        )
    if flags & properties.FeatureSelectionFlags.DROP_DUPLICATES:
        from feature_engine.selection import DropDuplicateFeatures

        module_logger.debug("Adding DropDuplicateFeatures transformer to the pipeline")

        numeric_columns_list.append(("duplicate_selector", DropDuplicateFeatures()))
    if flags & properties.FeatureSelectionFlags.DROP_CONSTANT:
        from feature_engine.selection import DropConstantFeatures

        module_logger.debug("Adding DropConstantFeatures transformer to the pipeline")
        numeric_columns_list.append(("constant_selector", DropConstantFeatures(missing_values="ignore")))

    if flags & properties.FeatureSelectionFlags.DROP_CORRELATED:
        from feature_engine.selection import SmartCorrelatedSelection

        selection_method: str = kwargs.get("selection_method", "variance")
        method_string: str = "pearson"
        method: Union[Callable, str] = kwargs.get("method", method_string)

        if callable(method):
            module_logger.info("A callable was passed as the correlation method")
        elif isinstance(method, str) and method != method_string:
            module_logger.info("A string was passed as the correlation method")
        else:
            module_logger.info("The default correlation method is used")

        module_logger.debug(
            "Adding DropCorrelated feature selection to the pipeline. Selection method: %s, Correlation method: %s",
            selection_method,
            method if isinstance(method, str) else method.__name__,
        )

        numeric_columns_list.append(
            (
                "correlation_selector",
                SmartCorrelatedSelection(
                    threshold=model_settings["correlation_threshold"],
                    selection_method=selection_method,
                    missing_values=kwargs.get("missing_values", "ignore"),
                    method=method,
                ),
            )
        )

    if flags & properties.FeatureSelectionFlags.DROP_UNIMPORTANT:
        module_logger.debug("Adding DropUnimportant feature selection to the pipeline")

        use_boruta: bool = kwargs.get("use_boruta", False)
        module_logger.info("Boruta usage: %s", str(use_boruta))
        numeric_columns_list.append(
            (
                "importance_selector",
                get_importance_selector(model_settings, cv, use_boruta=use_boruta),
            )
        )
    if len(numeric_columns_list) != 0:
        pipeline_settings.pop("transform", None)
        transformer_list.append(
            (
                "feature_selector",
                ml.PipelineWithHyperparameterRooting(
                    steps=numeric_columns_list,
                    verbose=pipeline_settings.get("verbose", False),
                    memory=pipeline_settings.get("memory", None),
                ),
                mother_utils.get_numeric_columns,
            )
        )
    assert pipeline_settings is not None
    for key in ["transform", "memory"]:
        pipeline_settings.pop(key, None)  # Use pop with default value to avoid KeyError if key is not found

    return ml.ColumnTransformerWithHyperparameterRooting(
        transformer_list,
        **pipeline_settings,
    )


def report_feature_selection(feature_selection: ColumnTransformer, data: Optional[pd.DataFrame] = None) -> None:
    """
    Logs detailed information about the feature selection process within a ColumnTransformer.

    Parameters:
        feature_selection (ColumnTransformer): The ColumnTransformer object containing the feature selection pipeline.
        data (Optional[pd.DataFrame]): Optional DataFrame containing the input data. If provided, standard deviations of
            correlated feature sets will be logged.

    Returns:
    None
    """

    def report(feature_selector: Any) -> None:
        if hasattr(feature_selector, "correlated_feature_dict_"):
            module_logger.info("Correlated features: %s", feature_selector.correlated_feature_dict_)
            if data is not None:
                for feature_set in feature_selector.correlated_feature_sets_:
                    try:
                        module_logger.info(
                            f"""Standard deviation for features:
{data[list(feature_set)].std().sort_values(ascending=False)}"""
                        )
                    except KeyError:
                        module_logger.info(
                            "Feature set not found in input data. Did you provide the correct input data?"
                        )
                    module_logger.info("")
        if hasattr(feature_selector, "features_to_drop_"):
            to_drop = feature_selector.features_to_drop_
            if len(to_drop) != 0:
                module_logger.info("Features to drop: %s", feature_selector.features_to_drop_)
            else:
                module_logger.info("No features to drop")
        # elif hasattr(feature_selector, "get_feature_names_out"):
        # module_logger.info("Feature names: %s", feature_selector.get_feature_names_out())

    module_logger.info("Analyzing feature selection pipeline")
    for transformer in feature_selection.named_transformers_:
        module_logger.info("Analyzing %s transformer", transformer)
        report(feature_selection.named_transformers_[transformer])
        if isinstance(feature_selection.named_transformers_[transformer], Pipeline):
            module_logger.info("Nested pipeline found")
            for name, sel in feature_selection.named_transformers_[transformer].steps:
                module_logger.info(f"Pipeline step: {name}")
                report(sel)


@overload
def get_model(settings: MotherSettings) -> ml.AbstractMotherPipeline: ...


@overload
def get_model(settings: Dict) -> ml.AbstractMotherPipeline: ...


def get_model(settings: Union[MotherSettings, Dict]) -> ml.AbstractMotherPipeline:
    """
    Creates and returns an instance of an appropriate machine learning model based on the provided settings.

    Args:
        settings (MotherSettings): Configuration settings for the model, including algorithm type and parameters.

    Returns:
        ml.AbstractMotherPipeline: An instance of a machine learning model pipeline.

    Raises:
        ValueError: If the algorithm specified in the settings is not supported.
        ValueError: If the Catboost parameters are missing when the algorithm is Catboost.
        ValueError: If the classification type specified in the settings is not supported.
        ValueError: If the target type specified in the settings is not supported.
    """
    model_settings: Dict = settings if isinstance(settings, dict) else settings.model.model_dump()
    if not ml.algo_is_supported(model_settings["algorithm"]):
        raise ValueError(f"Unsupported algorithm: {model_settings['algorithm']}")

    estimator: ml.AbstractMotherPipeline
    # extract the following into a separate function and use copilot to generate it...
    if model_settings["algorithm"] == "catboost":
        model_class = ml.get_model_class_by_algorithm_and_type(
            algorithm=model_settings["algorithm"],
            model_type=model_settings["model_type"],
        )
        estimator = model_class(target_type=model_settings["target_type"], **model_settings["parameters"])
    else:
        try:
            estimator = ml.get_model_class_by_algorithm_and_type(
                algorithm=model_settings["algorithm"],
                model_type=model_settings["model_type"],
            )(
                **model_settings["parameters"],
            )
        except TypeError as e:
            if "__init__() got an unexpected keyword argument" in str(e):
                # This is a common error when the model class does not accept the provided parameters.
                raise ValueError(
                    f"Invalid parameters for {model_settings['algorithm']} model: {model_settings['parameters']}. "
                    "Please check the model settings and parameters."
                ) from e
            raise ValueError(
                f"Unsupported algorithm: {model_settings['algorithm']}. Please check the model settings and parameters."
            ) from e
    return estimator


def get_feature_importance(
    pipeline: Pipeline,
    model_step_name: str,
    lowest_importance: float = 1.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Extracts feature importance from a pipeline as data frame. Importances are stored in the column
        'Importances'

    Args:
        pipeline (Pipeline): The scikit-learn pipeline object.
        model_step_name (str): The name of the model step in the pipeline.

    Returns:
        pd.DataFrame: A DataFrame containing feature importance values.
    """
    if model_step_name not in pipeline.named_steps:
        raise ValueError(f"Model step '{model_step_name}' not found in the pipeline")
    model = pipeline.named_steps[model_step_name]
    feature_importances: Optional[Any] = None
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    if hasattr(model, "get_feature_importance"):
        feature_importances = model.get_feature_importance(prettified=kwargs.get("prettified", True))

    def make_pretty(styler):
        styler.set_caption("Feature Importance")
        styler.format(precision=kwargs.get("precision", 3), thousands=".", decimal=",")
        styler.background_gradient(axis=None, vmin=1, vmax=5, cmap="YlGnBu")
        return styler

    feature_importance_data: pd.DataFrame
    if feature_importances is None:
        raise ValueError("Model does not have feature importance")
    if isinstance(feature_importances, pd.DataFrame):
        feature_importance_data = feature_importances
    else:
        feature_importance_data = pd.DataFrame(feature_importances, columns=["Importances"])

    return feature_importance_data[feature_importance_data["Importances"] >= lowest_importance].style.pipe(make_pretty)


@overload
def mother_cv(
    estimator: Union[ml.PipelineWithHyperparameterRooting, ml.AbstractMotherPipeline],
    *,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    cv: skl_model_sel.BaseCrossValidator,
    groups: Optional[pd.DataFrame] = None,
    prediction_prefix: str = "pred_",
    return_estimators: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def mother_cv(
    estimator: Union[ml.PipelineWithHyperparameterRooting, ml.AbstractMotherPipeline],
    *,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    cv: skl_model_sel.BaseCrossValidator,
    groups: Optional[pd.DataFrame] = None,
    prediction_prefix: str = "pred_",
    return_estimators: Literal[True],
) -> tuple[pd.DataFrame, dict[str, Any]]: ...


@overload
def mother_cv(
    estimator: Union[ml.PipelineWithHyperparameterRooting, ml.AbstractMotherPipeline],
    *,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    cv: skl_model_sel.BaseCrossValidator,
    inner_cv: skl_model_sel.BaseCrossValidator,
    tuner: MotherTuner,
    hyperparameter_space_function: Optional[Callable] = None,
    default_parameters: Optional[dict] = None,
    groups: Optional[pd.DataFrame] = None,
    prediction_prefix: str = "pred_",
    return_estimators: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def mother_cv(
    estimator: Union[ml.PipelineWithHyperparameterRooting, ml.AbstractMotherPipeline],
    *,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    cv: skl_model_sel.BaseCrossValidator,
    inner_cv: skl_model_sel.BaseCrossValidator,
    tuner: MotherTuner,
    hyperparameter_space_function: Optional[Callable] = None,
    default_parameters: Optional[dict] = None,
    groups: Optional[pd.DataFrame] = None,
    prediction_prefix: str = "pred_",
    return_estimators: Literal[True],
) -> tuple[pd.DataFrame, dict[str, Any]]: ...


def mother_cv(
    estimator: Union[ml.PipelineWithHyperparameterRooting, ml.AbstractMotherPipeline],
    *,
    cv: skl_model_sel.BaseCrossValidator,
    inner_cv: Optional[skl_model_sel.BaseCrossValidator] = None,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    groups: Optional[pd.DataFrame] = None,
    tuner: Optional[MotherTuner] = None,
    hyperparameter_space_function: Optional[Callable] = None,
    default_parameters: Optional[dict] = None,
    prediction_prefix: str = "pred_",
    return_estimators: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict[str, Any]]]:
    """
    Runs nested cross validation if a tuner is provided, otherwise
    runs standard cross validation

    Parameters
    ----------
    estimator
        Must be a PipelineWithHyperparameterRooting or AbstractMotherPipeline.
        sklearn.pipeline.Pipeline is not supported. If you want to use a sklearn pipeline,
        please wrap it in a PipelineWithHyperparameterRooting
        and provide the hyperparameter space function.
    inner_cv
        the cross validation generator used during the inner cv loop,
        the folds are generated automatically
    cv
        cross validation generator for the outer cv loop
    X
        a data frame that contains the data to be processed by the
        estimator (pipeline)
    y
        the data labels used for training the supervised models
    groups
        the groups are used for grouped/blocked cv in the inner cv
    tuner
        the tuner to be used to optimize the estimators hyperparameters
        if none is passed no tuning will be performed
    hyperparameter_space_function
        the hyperparameter function to be passed to the tuner
    default_parameters
        the default parameters to be passed to the tuner
    prediction_prefix
        the prefix added to the column names of the model predictions
    return_estimators:
        If True, return a tuple (performance_data, estimators_dict) containing
        fitted/optimized estimators for each fold. If False, return only the DataFrame.
    Returns
    -------
    If return_estimators=False: A dataframe containing the results of cross-validation.
    If return_estimators=True: A tuple (dataframe, estimators_dict) where estimators_dict contains:
        - "estimators": list of fitted estimators for each fold
        - "prediction_prefix": the prefix used for prediction columns
        - "target_columns": names of target columns

    The dataframe contains:
    For AbstractMotherPipeline and PipelineWithHyperparameterRooting: mean predictions
    and uncertainty estimates if the model supports it.

    """

    # Validate estimator type early
    if not isinstance(estimator, (ml.PipelineWithHyperparameterRooting, ml.AbstractMotherPipeline)):
        raise ValueError("Estimator must be a PipelineWithHyperparameterRooting or AbstractMotherPipeline")

    prediction_columns: List[str] = mother_utils.get_names(data=y)

    module_logger.info("Starting cross validation...")
    performance_data_list: List[pd.DataFrame] = []
    fold_estimators: List[Any] = []
    intermediate_performance_data: pd.DataFrame

    # make sure group is given as np.ndarray
    np_groups: npt.ArrayLike | None = groups.to_numpy().ravel() if groups is not None else None
    if groups is not None and not groups.index.equals(X.index):
        raise ValueError("groups must have the same index as X")

    for iteration, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=np_groups)):
        module_logger.debug(f"Running validation for split {iteration}")
        cv_groups: pd.DataFrame | None = None
        cv_groups_test: pd.DataFrame | None = None
        if groups is not None:
            cv_groups = groups.iloc[train_idx]
            cv_groups_test = groups.iloc[test_idx]
        if tuner is not None:
            module_logger.debug("Starting hyperparameter optimization and training in CV")
            val_estimator = tuner.optimize(
                estimator=estimator,
                X=X.iloc[train_idx],
                y=y.iloc[train_idx],
                cross_validation=inner_cv,
                hyperparameter_space_function=mother_utils.get_hyperparameter_space_function(
                    func=hyperparameter_space_function, estimator=estimator
                ),
                default_parameters=mother_utils.get_default_parameters(params=default_parameters, estimator=estimator),
                groups=cv_groups.to_numpy() if cv_groups is not None else None,
            )
        else:
            module_logger.debug("Start estimator training in CV")
            val_estimator = estimator.fit(X=X.iloc[train_idx], y=mother_utils.convert_input(y.iloc[train_idx]))

        if return_estimators:
            fold_estimators.append(val_estimator)

        module_logger.debug("The target values are being predicted")

        intermediate_performance_data: pd.DataFrame = val_estimator.predict_uncertainty(X.iloc[test_idx, :])

        if not isinstance(intermediate_performance_data, pd.DataFrame):
            # model returns non-pd.Data Frame
            intermediate_performance_data = pd.DataFrame(
                data=intermediate_performance_data, columns=prediction_columns, index=X.iloc[test_idx].index
            )

        elif "pred" in intermediate_performance_data.columns:
            intermediate_performance_data.rename(columns={"pred": prediction_columns[0]}, inplace=True)

        elif intermediate_performance_data.shape[1] == len(prediction_columns):
            intermediate_performance_data.columns = prediction_columns
        else:
            module_logger.debug(
                "Keeping predict_uncertainty output columns as returned: got %s columns for %s targets.",
                intermediate_performance_data.shape[1],
                len(prediction_columns),
            )

        module_logger.debug("The prefix is being added to the column names for the prediction columns")
        intermediate_performance_data = intermediate_performance_data.add_prefix(prediction_prefix)

        if is_classifier(val_estimator):
            existing_proba_columns = [str(col) for col in intermediate_performance_data.columns if "proba_" in str(col)]

            if existing_proba_columns:
                module_logger.debug(
                    "Skipping predict_proba generation because proba columns already exist: %s",
                    existing_proba_columns,
                )
            else:
                module_logger.debug("The model is a classification model so class probabilities will be predicted")
                prediction_columns_proba: List[str] = prediction_columns
                if len(prediction_columns) > 1:
                    module_logger.debug(
                        "For multitarget binary classification the prediction_column names will be used"
                    )
                else:
                    module_logger.debug(
                        "For single target classification the class labels are used as prediction columns"
                    )
                    prediction_columns_proba = val_estimator.classes_

                test_predicted_proba = pd.DataFrame(
                    data=val_estimator.predict_proba(X.iloc[test_idx, :]),
                    index=X.iloc[test_idx].index,
                    columns=prediction_columns_proba,
                )

                module_logger.debug("The prefix is being added to the column names for the proba columns")
                proba_prefix = prediction_prefix.rstrip("_") + "_proba_"
                test_predicted_proba = test_predicted_proba.add_prefix(proba_prefix)

                module_logger.debug(
                    "The probabilities are added to the predictions in addition to the class predictions"
                )
                intermediate_performance_data = intermediate_performance_data.merge(
                    test_predicted_proba,
                    left_index=True,
                    right_index=True,
                )

        module_logger.debug("Add the original target values to the predictions")
        intermediate_performance_data = pd.concat(
            [
                intermediate_performance_data,
                y.iloc[test_idx],
            ],
            axis=1,
            verify_integrity=False,
        )

        assert all(intermediate_performance_data.index == X.iloc[test_idx].index)
        module_logger.debug("Add the cv metainformation to the predictions")
        intermediate_performance_data["cv_group"] = (
            cv_groups_test.values.ravel() if cv_groups_test is not None else None
        )  # Flatten to avoid multi-index issues

        intermediate_performance_data["iteration"] = iteration
        intermediate_performance_data["test_index"] = test_idx

        performance_data_list.append(intermediate_performance_data)
        module_logger.debug("Validation for split %d completed", iteration)

    performance_data: pd.DataFrame = pd.concat(performance_data_list, verify_integrity=True).convert_dtypes()
    performance_data.columns = performance_data.columns.astype(str)
    performance_data.dropna(axis=1, how="all", inplace=True)
    if "cv_group" in performance_data.columns:
        performance_data = performance_data.astype({"cv_group": int, "iteration": int, "test_index": int})

    module_logger.debug("Sort the data frame by the original index")
    performance_data = performance_data.sort_values("test_index")

    module_logger.info("Cross validation completed")

    if return_estimators:
        estimator_output: dict[str, Any] = {
            "estimators": fold_estimators,
            "prediction_prefix": prediction_prefix,
            "target_columns": prediction_columns,
        }
        module_logger.info("Returning performance_data with estimators as tuple")
        return performance_data, estimator_output

    return performance_data


def get_training_pipeline(settings: MotherSettings) -> NoReturn:
    raise NotImplementedError("Training pipeline loading is not implemented yet")


def get_inference_pipeline(settings: MotherSettings) -> NoReturn:
    raise NotImplementedError("Inference pipeline loading is not implemented yet")


def get_ranking_model(target_type: properties.TargetType, categorical_features, **kwargs) -> CatboostRankerMother:
    # only single target supported for ranking currently
    assert target_type == "single_target", "Only single target ranking is supported currently"

    ranking_model: CatboostRankerMother = CatboostRankerMother(
        target_type=target_type, cat_features=categorical_features, **kwargs
    ).set_fit_request(group_id="group_id")

    return ranking_model


def get_ranking_scorer(k: int) -> Callable:
    def score_func(y, y_pred, group_id):
        return avg_ndcg_score(y, y_pred, group_id, k=k)

    return make_scorer(score_func, greater_is_better=True).set_score_request(group_id="group_id")


def get_ranking_pipeline(
    categorical_features,
    k_scorer: int,
    target_type: properties.TargetType = "single_target",
    model_kwargs: dict | None = None,
    tuner_kwargs: dict | None = None,
) -> Tuple[Pipeline, MotherTuner]:
    """
    building the model pipeline for feature selection and ranking model
    - will set the group_id fit_request for the ranking model
    - will configure the ranking scorer with the provided k
    - will use hyperparameter routing for the model, and the scorer

    Parameters
    ----------
    target_type (properties.TargetType): the target type, only single_target is supported for ranking
    categorical_features (list): list of categorical feature names
    k_scorer (int): the k value to be used for the ranking scorer (NDCG@k)
    model_kwargs (dict): additional keyword arguments to be passed to the ranking model
    tuner_kwargs (dict): additional keyword arguments to be passed to the tuner

    Returns
    -------
    modeling_pipeline (Pipeline): the modeling pipeline
    tuner (MotherTuner): the tuner for hyperparameter optimization, including the NDCG-scorer

    """
    ranking_model = get_ranking_model(
        target_type=target_type,
        categorical_features=categorical_features,
        **(model_kwargs if model_kwargs else {}),
    )

    scorer = get_ranking_scorer(k=k_scorer)

    # default settings for feature selection
    model_settings = {
        "feature_selection_flags": [
            "DROP_CORRELATED",
            "DROP_CONSTANT",
            "DROP_DUPLICATES",
            "DROP_UNIMPORTANT",
        ],
        "categorical_features": categorical_features,
        "feature_selection_threshold": 0,
        "correlation_threshold": 0.9,
        "algorithm": "catboost",
        "feature_selection_type": "permutation",
        "model_type": "regression",
        "target_type": "single_target",
    }
    pipeline_settings = {
        "remainder": "drop" if len(categorical_features) == 0 else "passthrough",
        "verbose_feature_names_out": False,
    }

    feature_selection_pipeline = get_feature_selection_pipeline(
        settings=model_settings,
        pipeline_settings=pipeline_settings,
        cv=None,  # use default cv for feature selection
    ).set_output(transform="pandas")

    ranking_pipeline = PipelineWithHyperparameterRooting(
        steps=[
            ("feature_selection", feature_selection_pipeline),
            ("model", ranking_model),
        ]
    )

    tuner = MotherTuner(scorer=scorer, **(tuner_kwargs if tuner_kwargs else {}))

    return ranking_pipeline, tuner
