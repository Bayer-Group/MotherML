import json
import logging
import typing

import numpy as np
import pandas as pd

module_logger: logging.Logger = logging.getLogger(__name__)


def dump_best_trial_parameters(best_parameters: dict[str, typing.Any]) -> None:
    try:
        print(best_parameters)
        module_logger.info(
            "Final pipeline parameters:\n %s",
            json.dumps(best_parameters, indent=4),
        )
    except Exception as e:
        module_logger.error("Error logging final pipeline parameters: %s", e)


def y_toArray(y: typing.Union[pd.DataFrame, pd.Series]) -> np.ndarray:
    """
    Converts a pandas DataFrame or Series to a NumPy array.

    This function ensures that the input `y` is a pandas DataFrame or Series and is not empty.
    It then converts the input to a NumPy array. If the resulting array is
    one-dimensional, it reshapes it to ensure compatibility with downstream processes.
    If the reshaped array has only one column, it flattens it to a one-dimensional array.

    Args:
        y (Union[pd.DataFrame, pd.Series]): The input pandas DataFrame or Series to be converted.

    Returns:
        np.ndarray: The converted NumPy array.

    Raises:
        ValueError: If `y` is not a pandas DataFrame or Series, or if it is empty.
    """
    if not isinstance(y, (pd.DataFrame, pd.Series)):
        raise ValueError("y should be a pandas DataFrame or Series")
    if y.empty:
        raise ValueError("y should not be empty")

    y_array: np.ndarray = y.to_numpy()
    if y_array.ndim == 1:
        y_array = y_array.reshape((-1, 1))
    if y_array.shape[1] == 1:
        y_array = y_array[:, 0]
    return y_array
