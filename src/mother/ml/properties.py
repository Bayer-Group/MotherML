import typing
from enum import IntFlag, auto, unique
from typing import Annotated

from pydantic import Field

ModelType = typing.Literal["classification_binary", "classification_multiclass", "regression", "ranking"]

TargetType = typing.Literal["single_target", "multi_target"]

FeatureSelectionType = typing.Literal["catboost", "permutation"]


@unique
class FeatureSelectionFlags(IntFlag):
    """
    FeatureSelectionFlags is a flag class for feature selection types.
    """

    NONE = 0
    IMPUTE_CATEGORICAL = auto()  # Impute missing values in categorical features
    DROP_DUPLICATES = auto()  # Drop duplicate features
    DROP_CONSTANT = auto()  # Drop constant features
    DROP_CORRELATED = auto()  # Drop correlated features
    DROP_UNIMPORTANT = auto()  # Drop unimportant features

    def __str__(self) -> str:
        return self.name if self.name else ""

    def __rep__(self) -> typing.Optional[str]:
        return self.name


PositiveInt = Annotated[int, Field(gt=0)]

FeatureSelectionFlagLiteral = Annotated[
    typing.Literal[
        FeatureSelectionFlags.NONE.__rep__(),
        FeatureSelectionFlags.DROP_CONSTANT.__rep__(),
        FeatureSelectionFlags.DROP_DUPLICATES.__rep__(),
        FeatureSelectionFlags.IMPUTE_CATEGORICAL.__rep__(),
        FeatureSelectionFlags.DROP_CORRELATED.__rep__(),
        FeatureSelectionFlags.DROP_UNIMPORTANT.__rep__(),
    ],
    Field(description="Feature selection flags"),
]

FeatureSelectionFlagList = typing.List[FeatureSelectionFlagLiteral]
