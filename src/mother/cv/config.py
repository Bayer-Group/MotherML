import logging
from enum import Enum
from typing import Any, Dict, Optional

import dotenv
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

module_logger: logging.Logger = logging.Logger(__name__)


class CVtype(str, Enum):
    TANIMOTO_GROUPING = "tanimoto_grouping"
    TIME_SERIES = "time_series"
    GROUPS = "groups"


class GenericCVModel(BaseModel):
    model_config = SettingsConfigDict(
        env_prefix="mother_cv_",
        case_sensitive=False,
        use_enum_values=False,
        extra="forbid",
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
        arbitrary_types_allowed=True,
    )


class TanimotoSimilarityParams(GenericCVModel):
    similarity_threshold: float = Field(default=0.3, description="Threshold for Tanimoto similarity")


class TimeSeriesParams(GenericCVModel):
    datetime_fmt: str = Field(default="%Y-%m-%d", description="Datetime format to convert string to datetime object")
    max_train_size: Optional[int] = Field(default=None, description="Maximum training set size")
    test_size: Optional[int] = Field(default=None, description="Used to limit the size of the test set.")
    gap: int = Field(
        default=0, description="Number of samples to exclude from the end of each train set before the test set."
    )


class CVSettings(BaseModel):
    cv_type: CVtype = Field(
        default=CVtype.TANIMOTO_GROUPING, description="Define supported types to perform cross validation."
    )
    n_splits: int = Field(default=5, description="Number of splits for cross-validation")
    parameters: Dict[str, Any] = {"similarity_threshold": 0.3}

    def get_cv_settings(self) -> GenericCVModel:
        if self.cv_type == CVtype.TANIMOTO_GROUPING:
            return TanimotoSimilarityParams(**self.parameters)
        elif self.cv_type == CVtype.TIME_SERIES:
            return TimeSeriesParams(**self.parameters)
        elif self.cv_type == CVtype.GROUPS:
            # provided group data
            return GenericCVModel(**self.parameters)
        else:
            raise ValueError(f"Provided cv type '{self.cv_type}' is not supported")

    model_config = SettingsConfigDict(
        env_prefix="mother_cv_",
        case_sensitive=False,
        use_enum_values=False,
        extra="ignore",
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
        arbitrary_types_allowed=True,
    )
