import typing

import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mother.ml import properties


class ModelConfig(BaseSettings):
    """
    MLConfig is a configuration class for machine learning settings.

    Attributes:
        categorical_features (List[str]): Column names of categorical features.
        type (properties.ModelType): Model type. One of the available model types in properties.ModelType.
        target_type (properties.TargetType): ML target type. One of the available target types in properties.TargetType.
        feature_selection_type (properties.FeatureSelectionType): Feature selection type.
            One of the available feature selection types in properties.FeatureSelectionType.
        name (str): Name of the model (supported by mother).
        parameters (Dict[str, Any]): Model specific parameters. For example, everything to configure catboost.
    """

    categorical_features: typing.List[str] = Field(default=[], description="Column names of categorical features")
    model_type: properties.ModelType = Field(description=f"Model type. One of {properties.ModelType}")
    target_type: typing.Annotated[
        properties.TargetType, Field(description=f"ML target type. One of {properties.TargetType}")
    ]
    feature_selection_type: typing.Annotated[
        properties.FeatureSelectionType,
        Field(description=f"Feature selection type. One of {properties.FeatureSelectionType}"),
    ]
    correlation_threshold: float = Field(default=0.9, description="Correlation threshold for feature selection")
    feature_selection_flags: typing.Annotated[
        properties.FeatureSelectionFlagList, Field(default=[], description="Feature selection flags")
    ]
    algorithm: str = Field(description="Model architecture (supported by mother)", examples=["catboost", "lasso"])
    parameters: typing.Dict[str, typing.Any] = Field(
        default={}, description="Model specific parameters. For example, everything to configure catboost"
    )
    feature_selection_threshold: typing.Optional[float] = Field(
        default=None,
        description=" Features whose absolute importance value is greater or equal are kept while others are discarded",
    )
    feature_selection_max_features: typing.Optional[int] = Field(
        default=None, description="Feature importance selection max features to return"
    )
    model_config = SettingsConfigDict(
        env_prefix="mother_ml_",
        case_sensitive=False,
        use_enum_values=False,
        extra="ignore",
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
        arbitrary_types_allowed=True,
    )


def getFlagFromStrings(flags: properties.FeatureSelectionFlagList) -> properties.FeatureSelectionFlags:
    """
    Converts a list of user-defined flags into a combined StandardizationFlag.

    Args:
        values (List[Flag]): A list of user-defined flags.

    Returns:
        StandardizationFlag: A combined StandardizationFlag based on the input flags.
    """
    _flag_conversion_map: typing.Dict[str, int] = {str(flag): flag for flag in properties.FeatureSelectionFlags}
    # fix for py 3.11 and above
    _flag_conversion_map[str(properties.FeatureSelectionFlags.NONE)] = properties.FeatureSelectionFlags.NONE
    flag: properties.FeatureSelectionFlags = properties.FeatureSelectionFlags.NONE
    for value in flags:
        try:
            flag |= properties.FeatureSelectionFlags(_flag_conversion_map[str(value)])
        except KeyError:
            raise ValueError(f"Unknown flag {value}")
    return flag
