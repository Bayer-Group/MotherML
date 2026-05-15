import logging
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from optuna.trial import Trial
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from mother.ml.core import AbstractMotherPipeline
from mother.ml.models import utils

module_logger: logging.Logger = logging.getLogger(__name__)
DEFAULT_QUANTILES: list[float] = [0.25, 0.5, 0.75]


class _RandomForestMotherBase(AbstractMotherPipeline):
    """Base class for RandomForest models with common state management methods."""

    _extra_params: dict

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_saved_params_snapshot"] = self.get_params(deep=False)
        return state

    def __setstate__(self, state: dict) -> None:
        params = state.pop("_saved_params_snapshot", None)
        self.__dict__.update(state)

        if params:
            try:
                self.set_params(**params)
            except Exception:
                pass


class RandomForestClassifierMother(RandomForestClassifier, _RandomForestMotherBase):
    """
    A RandomForest classifier pipeline for the MOTHER framework, integrating hyperparameter optimization
    via Optuna and providing default parameter management. Inherits from both scikit-learn's
    RandomForestClassifier and the AbstractMotherPipeline for seamless integration with the MOTHER
    machine learning workflow.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        class_weight: Optional[Union[dict, list, Literal["balanced", "balanced_subsample"]]] = "balanced_subsample",
        **kwargs: Any,
    ):
        """
        Initializes the RandomForestClassifierMother with optional keyword arguments. See
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        for available parameters.


        Args:
            class_weight: {"balanced", "balanced_subsample"} or dict or list of dicts, or None, optional
                We use the same typing as sklearn's RandomForestClassifier.
            **kwargs: Additional keyword arguments to pass to the RandomForestClassifier constructor.
        """
        valid_class_weights = ["balanced", "balanced_subsample", None]  # Add more valid options if needed
        if isinstance(class_weight, str) and class_weight not in valid_class_weights:
            raise ValueError("Invalid class weight")

        self._extra_params = dict(kwargs)
        super().__init__(n_estimators=n_estimators, class_weight=class_weight, **kwargs)

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Defines the hyperparameter search space for RandomForestClassifier using Optuna.

        Args:
            X: Feature matrix for training data.
            y: Target vector for training data.
            trial (Trial): Optuna trial object for suggesting hyperparameters.
            prefix (str, optional): Prefix to add to hyperparameter names. Defaults to "".

        Returns:
            dict: Dictionary of hyperparameter names (with prefix) and their suggested values.
        """
        return utils.add_prefix_to_dict_keys(
            {
                "criterion": trial.suggest_categorical(
                    name=prefix + "criterion", choices=["gini", "entropy", "log_loss"]
                ),
                "max_features": trial.suggest_categorical(name=prefix + "max_features", choices=["sqrt", "log2"]),
                "min_samples_leaf": trial.suggest_int(name=prefix + "min_samples_leaf", low=1, high=5, step=1),
            },
            prefix=prefix,
        )

    def default_parameters(self, prefix: str = "", **kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Returns the default hyperparameters for the RandomForestClassifier.

        Args:
            prefix (str, optional): Prefix to add to hyperparameter names. Defaults to "".

        Returns:
            dict: Dictionary of default hyperparameter names (with prefix) and their values.
        """
        _defaults: dict[str, Any] = {
            "criterion": "gini",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
        }
        if kwargs:
            module_logger.warning(
                "Default parameters for RandomForestRegressorMother are being overridden by provided kwargs."
            )
            _defaults.update(kwargs)
        return utils.add_prefix_to_dict_keys(
            _defaults,
            prefix=prefix,
        )

    def get_params(self, deep: bool = True) -> dict:
        """
        Return all constructor parameters, including those passed via **kwargs
        """
        try:
            params = super().get_params(deep=deep)
        except Exception:
            params = {}

        params.update(
            {
                "n_estimators": getattr(self, "n_estimators", 500),
                "class_weight": getattr(self, "class_weight", "balanced_subsample"),
            }
        )

        params.update(getattr(self, "_extra_params", {}))
        return params

    def set_params(self, **params) -> "RandomForestClassifierMother":
        if not params:
            return self

        # Store all params that aren't in the base constructor signature as extras
        extras = {}
        for k, v in params.items():
            setattr(self, k, v)
            if k not in {"n_estimators", "class_weight"}:
                extras[k] = v

        # Update _extra_params
        if hasattr(self, "_extra_params"):
            self._extra_params.update(extras)
        else:
            self._extra_params = dict(extras)

        # Try to set params on parent class
        try:
            super().set_params(**params)
        except Exception:
            pass

        return self


class RandomForestRegressorMother(RandomForestQuantileRegressor, _RandomForestMotherBase):
    """
    A RandomForest regression pipeline for the MOTHER framework, integrating hyperparameter optimization
    via Optuna, providing default parameter management and uncertainty quantification methods.
    This class inherits from both quantile-forest's RandomForestQuantileRegressor and the
    AbstractMotherPipeline for seamless integration with the MOTHER machine learning workflow.
    """

    def __init__(self, n_estimators: int = 500, min_samples_leaf: int = 5, **kwargs: Any):
        """
        Initializes the RandomForestRegressorMother with optional keyword arguments.
        See https://zillow.github.io/quantile-forest/generated/quantile_forest.RandomForestQuantileRegressor.html
        for available parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to the RandomForestRegressor constructor.
        """

        self._extra_params = dict(kwargs)

        super().__init__(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, default_quantiles="mean", **kwargs
        )

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Defines the hyperparameter search space for RandomForestRegressor using Optuna.

        Args:
            X: Feature matrix for training data.
            y: Target vector for training data.
            trial (Trial): Optuna trial object for suggesting hyperparameters.
            prefix (str, optional): Prefix to add to hyperparameter names. Defaults to "".

        Returns:
            dict: Dictionary of hyperparameter names (with prefix) and their suggested values.
        """
        suggested_params: dict = {
            "max_features": trial.suggest_categorical(name=prefix + "max_features", choices=["sqrt", "log2"]),
            "min_samples_leaf": trial.suggest_int(name=prefix + "min_samples_leaf", low=1, high=5, step=1),
        }

        choices = ["squared_error", "absolute_error", "friedman_mse", "poisson"]
        if np.any(y < 0):
            module_logger.debug("Target is not strictly non-negative, removing 'poisson' from criterion choices")
            choices.remove("poisson")

        suggested_params["criterion"] = trial.suggest_categorical(name=prefix + "criterion", choices=choices)
        return utils.add_prefix_to_dict_keys(suggested_params, prefix=prefix)

    def default_parameters(self, prefix: str = "", **kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Returns the default hyperparameters for the RandomForestRegressor.

        Args:
            prefix (str, optional): Prefix to add to hyperparameter names. Defaults to "".

        Returns:
            dict: Dictionary of default hyperparameter names (with prefix) and their values.
        """
        _defaults: dict[str, Any] = {
            "criterion": "squared_error",
            "max_features": "sqrt",
            "min_samples_leaf": 5,
        }
        if kwargs:
            module_logger.warning(
                "Default parameters for RandomForestRegressorMother are being overridden by provided kwargs."
            )
            _defaults.update(kwargs)
        return utils.add_prefix_to_dict_keys(
            _defaults,
            prefix=prefix,
        )

    @staticmethod
    def _ensure_2d_array(values: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> np.ndarray:
        """Return input as a 2D numpy array (n_samples, n_targets)."""
        array = np.asarray(values)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        return array

    def _build_prediction_output(
        self,
        index: Optional[pd.Index],
        output_sections: dict[str, Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]]],
    ) -> pd.DataFrame:
        """Build a standardized prediction DataFrame from named output sections.

        Each section (for example ``pred``, ``mean_predictions`` or ``total_uncertainty``)
        is normalized to a 2D array and validated to have the same shape. The final output
        keeps a consistent column schema:
        - single-target: ``<section_name>``
        - multi-target: ``target_<idx>_<section_name>``
        """
        section_arrays: dict[str, np.ndarray] = {}
        n_samples: Optional[int] = None
        n_targets: Optional[int] = None

        for section_name, section_values in output_sections.items():
            if section_values is None:
                continue

            section_array = self._ensure_2d_array(section_values)
            if n_samples is None or n_targets is None:
                n_samples, n_targets = section_array.shape
            elif section_array.shape != (n_samples, n_targets):
                raise ValueError(
                    f"Section '{section_name}' has shape {section_array.shape}, expected {(n_samples, n_targets)}."
                )

            section_arrays[section_name] = section_array

        if n_samples is None or n_targets is None:
            raise ValueError("At least one prediction section with values is required to build the output.")

        section_frames = []
        for section_name, section_values in output_sections.items():
            if section_values is None:
                section_array = np.full((n_samples, n_targets), None, dtype=object)
            else:
                section_array = section_arrays[section_name]

            columns = [section_name] if n_targets == 1 else [f"target_{idx}_{section_name}" for idx in range(n_targets)]
            section_frames.append(pd.DataFrame(section_array, index=index, columns=columns))

        return pd.concat(section_frames, axis=1)

    def predict_uncertainty(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_quantiles: bool = False,
        quantiles: list[float] = DEFAULT_QUANTILES,
        uncertainty_for_opt: bool = False,
        **kwargs,
    ) -> Union[
        pd.DataFrame,
        dict[str, pd.DataFrame],
        tuple[
            Union[pd.DataFrame, dict[str, pd.DataFrame]],
            Union[pd.DataFrame, dict[str, pd.DataFrame]],
        ],
    ]:
        """
        Predict the target value and estimate the uncertainty (optional) for RandomForestQuantileRegressor.
        The knowledge uncertainty estimation uses interquartile range of each sample.

        Parameters:
        - X: pd.DataFrame
            Input data for prediction.
        - return_quantiles: bool (default: False)
            If True, return quantile values.
        - uncertainty: bool (default: False)
            Whether estimate the uncertainty or not
        - quantiles : list = [0.25, 0.5, 0.75]
            List of quantiles to calculate the uncertainty. All values must be in the range [0, 1].
        - uncertainty_for_opt: bool (default: False)
            Whether to returned a standardized DataFrame for optimization,
            containing only estimated total uncertainty.

        Returns:
            Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame | dict[str, pd.DataFrame]]]:
                - If `uncertainty_for_opt=True`: A DataFrame with exactly one column,
                  `'total_uncertainty'`.
                - If `return_quantiles=False`: A DataFrame with columns:
                    - 'mean_predictions': The mean predictions for each sample (mean of quantiles).
                    - 'knowledge_uncertainty' : None.
                    - 'data_uncertainty' : None.
                    - 'total_uncertainty': The uncertainty quantified for each sample (interquartile range).
                - If `return_quantiles=True`: A tuple containing:
                    - The DataFrame described above.
                    - A quantile table (`pd.DataFrame` for single-target or
                      `dict[str, pd.DataFrame]` for multi-target).

        Raises:
            ValueError: If quantiles list is empty or contains values outside [0, 1].
        """

        check_is_fitted(self)

        # Avoid mutating caller/default list while ensuring required quantiles are present.
        quantiles = list(quantiles)

        # Validate quantiles
        if not quantiles:
            raise ValueError("Quantiles list cannot be empty.")

        invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
        if len(invalid_quantiles) > 0:
            raise ValueError(f"All quantiles must be in the range [0, 1]. Invalid values: {invalid_quantiles}")

        # Ensure DEFAULT_QUANTILES are included for IQR calculation
        for q in DEFAULT_QUANTILES:
            if q not in quantiles:
                quantiles.append(q)
        quantiles.sort()

        # Get index for DataFrame creation
        index = X.index if isinstance(X, pd.DataFrame) else None

        # Get quantile predictions
        module_logger.info(f"Using provided quantile {quantiles} for prediction")
        quantile_predictions = super().predict(X, quantiles=quantiles)

        # Use the same prediction path as user-facing predict() to keep outputs aligned.
        point_predictions = self.predict(X)

        # Check if preditions has a multitask output (n_samples, n_targets, n_quantiles
        if quantile_predictions.ndim == 3 and quantile_predictions.shape[1] > 1:
            _, n_targets, _ = quantile_predictions.shape

            total_uncertainty = np.empty((quantile_predictions.shape[0], n_targets))
            per_target_quantiles = {}

            for task_idx in range(n_targets):
                target_index = f"target_{task_idx}"
                column_names: list[str] = [f"quantile_{q}" for q in quantiles]
                target_quantiles_df = pd.DataFrame(
                    quantile_predictions[:, task_idx, :],
                    index=index,
                    columns=column_names,  # type: ignore[arg-type]
                )

                per_target_quantiles[target_index] = target_quantiles_df
                total_uncertainty[:, task_idx] = (
                    target_quantiles_df[f"quantile_{0.75}"] - target_quantiles_df[f"quantile_{0.25}"]
                )

            uncertainty_output = self._build_prediction_output(
                index=index,
                output_sections={
                    "pred": point_predictions,
                    "mean_predictions": point_predictions,
                    "total_uncertainty": total_uncertainty,
                },
            )

            if uncertainty_for_opt:
                module_logger.warning(
                    "uncertainty_for_opt=True for multi-target regression: "
                    "returning max of per-target total_uncertainty as total_uncertainty."
                )
                return pd.DataFrame(
                    {"total_uncertainty": total_uncertainty.max(axis=1)},
                    index=index,
                )

            if return_quantiles:
                return uncertainty_output, per_target_quantiles

            return uncertainty_output
        else:
            if quantile_predictions.shape[1] != len(quantiles):
                module_logger.warning("Shape mismatch between predictions and quantiles. Check model configuration!")

            total_uncertainty = (
                quantile_predictions[:, quantiles.index(0.75)] - quantile_predictions[:, quantiles.index(0.25)]
            )

            uncertainty_output = self._build_prediction_output(
                index=index,
                output_sections={
                    "pred": point_predictions,
                    "mean_predictions": None,
                    "knowledge_uncertainty": None,
                    "data_uncertainty": None,
                    "total_uncertainty": total_uncertainty,
                },
            )

            if return_quantiles:
                quantile_table = pd.DataFrame(
                    quantile_predictions,
                    index=index,
                    columns=[f"quantile_{q}" for q in quantiles],
                )
                return uncertainty_output, quantile_table
            if uncertainty_for_opt:
                return uncertainty_output.loc[:, ["total_uncertainty"]]
            return uncertainty_output

    def get_params(self, deep: bool = True) -> dict:
        """
        Return all constructor parameters, including those passed via **kwargs
        """
        try:
            params = super().get_params(deep=deep)
        except Exception:
            params = {}

        params.update(
            {
                "n_estimators": getattr(self, "n_estimators", 500),
                "min_samples_leaf": getattr(self, "min_samples_leaf", 5),
            }
        )

        params.update(getattr(self, "_extra_params", {}))
        return params

    def set_params(self, **params) -> "RandomForestRegressorMother":
        if not params:
            return self

        # Store all params that aren't in the base constructor signature as extras
        extras = {}
        for k, v in params.items():
            setattr(self, k, v)
            if k not in {"n_estimators", "min_samples_leaf"}:
                extras[k] = v

        # Update _extra_params
        if hasattr(self, "_extra_params"):
            self._extra_params.update(extras)
        else:
            self._extra_params = dict(extras)

        # Try to set params on parent class
        try:
            super().set_params(**params)
        except Exception:
            pass

        return self
