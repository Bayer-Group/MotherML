import logging
import pathlib as pl
import time
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_list_like
from sklearn.pipeline import Pipeline

from mother import ml

log: logging.Logger = logging.getLogger(__name__)


def setup_logging(level: typing.Union[int, str], folder: typing.Optional[pl.Path], **kwargs) -> logging.Logger:
    """
    _summary_

    Args:
        level (logging._Level): log level to be used
        log_folder (typing.Optional[Path]): path to the log folder

    Returns:
        logging.Logger: configured logger for mother
    """
    logger = logging.getLogger("mother")

    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s | %(name)30s | %(levelname)8s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if folder:
        if not folder.exists():
            folder.mkdir(
                parents=kwargs.get("parents", True),
                exist_ok=kwargs.get("exist_ok", True),
            )
        file_handler = logging.FileHandler(
            folder.joinpath(f"mother_run_{time.asctime().replace(' ', '_').replace(':', '-')}.log").as_posix()
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def convert_input(
    input: typing.Iterable, col: typing.Optional[typing.Union[str, typing.List[str]]] = None
) -> npt.NDArray:
    res: npt.NDArray
    if isinstance(input, pd.DataFrame):
        if col is None:
            res = input.to_numpy()
        else:
            res = input[col].to_numpy()
        if len(input.columns) == 1:
            res = res.reshape(-1)
    else:
        assert is_list_like(input)
        res = np.array(input)
    return res


def get_names(data: typing.Union[pd.DataFrame, pd.Series]) -> typing.List[str]:
    """
    Returns the column names if the input is a DataFrame or the name if the input is a Series.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        The input data which can be a DataFrame or a Series.

    Returns
    -------
    Union[List[str], Optional[str]]
        The column names if the input is a DataFrame or the name if the input is a Series.
    """
    if isinstance(data, pd.DataFrame):
        return data.columns.tolist()
    elif isinstance(data, pd.Series):
        return [data.name]  # type: ignore
    else:
        raise TypeError("Input must be a pandas DataFrame or Series")


def get_numeric_columns(data: pd.DataFrame) -> typing.List[str]:
    """
    Returns a list of column names in the given DataFrame that have numeric data types.

    Parameters:
    data (pd.DataFrame): The DataFrame to inspect for numeric columns.

    Returns:
    typing.List[str]: A list of column names that have data types 'int64' or 'float64'.
    """
    return data.select_dtypes(include=["int64", "float64"]).columns.tolist()


def get_categorical_column_names(data: pd.DataFrame) -> typing.List[str]:
    """
    Get the names of categorical columns in a DataFrame.

    This function identifies columns in the provided DataFrame that have
    data types typically associated with categorical data, such as 'object',
    'bool', and 'category', and returns their names.

    Parameters:
    data (pd.DataFrame): The DataFrame from which to extract categorical column names.

    Returns:
    typing.List[str]: A list of column names that are of categorical data types.
    """
    return data.select_dtypes(include=["object", "bool", "category"]).columns.tolist()


def get_default_parameters(
    params: typing.Optional[typing.Dict],
    estimator: typing.Union[Pipeline, ml.PipelineWithHyperparameterRooting, ml.AbstractMotherPipeline],
) -> dict:
    if params is None:
        if hasattr(estimator, "default_parameters"):
            return estimator.default_parameters()  # type: ignore
        return {}
    return params


def get_hyperparameter_space_function(
    func: typing.Optional[typing.Callable],
    estimator: typing.Union[Pipeline, ml.PipelineWithHyperparameterRooting, ml.AbstractMotherPipeline],
) -> typing.Callable:
    if func is None:
        if hasattr(estimator, "get_hyperparameter_space"):
            hyperspace_func = getattr(estimator, "get_hyperparameter_space")
            if callable(hyperspace_func):
                return hyperspace_func
            else:
                raise ValueError("Hyperparameter space function is not callable")
        raise ValueError("Hyperparameter space function is missing")
    return func
