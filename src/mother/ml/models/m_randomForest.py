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

        super().__init__(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, **kwargs)

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
        tuple[Union[pd.DataFrame, dict[str, pd.DataFrame]], Union[pd.DataFrame, dict[str, pd.DataFrame]]],
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
            Union[pd.DataFrame, tuple[pd.DataFrame, np.array]]:
                - If `return_quantiles=False`: A DataFrame with columns:
                    - 'mean_predictions': The mean predictions for each sample (mean of quantiles).
                    - 'knowledge_uncertainty' : None.
                    - 'data_uncertainty' : None.
                    - 'total_uncertainty': The uncertainty quantified for each sample (interquartile range).
                - If `return_quantiles=True`: A tuple containing:
                    - The DataFrame described above.
                    - np.array of quantile values whose shape is (# samples, # quantiles).

        Raises:
            ValueError: If quantiles list is empty or contains values outside [0, 1].
        """

        check_is_fitted(self)

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

        module_logger.info(f"Using provided quantile {quantiles} for prediction")
        predictions = super().predict(X, quantiles=quantiles)

        # Check if preditions has a multitask output (n_samples, n_targets, n_quantiles
        if predictions.ndim == 3 and predictions.shape[1] > 1:
            _, n_targets, _ = predictions.shape

            output = {}
            quantile_outputs = {}
            uncertainty_matrix = {}

            for task_idx in range(n_targets):
                target_name = f"target_{task_idx}"
                column_names: list[str] = [f"quantile_{q}" for q in quantiles]
                pred_df = pd.DataFrame(
                    predictions[:, task_idx, :],
                    index=index,
                    columns=column_names,  # type: ignore[arg-type]
                )

                q50_col = f"quantile_{0.5}" if 0.5 in quantiles else pred_df.mean(axis=1)
                total_uncertainty = pred_df[f"quantile_{0.75}"] - pred_df[f"quantile_{0.25}"]
                uncertainty_matrix[target_name] = total_uncertainty

                task_output = pd.DataFrame(
                    {
                        "mean_predictions": pred_df[q50_col] if isinstance(q50_col, str) else q50_col,
                        "knowledge_uncertainty": None,
                        "data_uncertainty": None,
                        "total_uncertainty": total_uncertainty,
                    },
                    index=index,
                )

                output[target_name] = task_output
                quantile_outputs[target_name] = pred_df

            if uncertainty_for_opt:
                return pd.DataFrame(uncertainty_matrix, index=index)

            if return_quantiles:
                return output, quantile_outputs

            return output
        else:
            if predictions.shape[1] != len(quantiles):
                module_logger.warning("Shape mismatch between predictions and quantiles. Check model configuration!")

            column_names: list[str] = [f"quantile_{q}" for q in quantiles]
            pred_df = pd.DataFrame(predictions, index=index, columns=column_names)  # type: ignore[arg-type]
            total_uncertainty = pred_df[f"quantile_{0.75}"] - pred_df[f"quantile_{0.25}"]
            q50_col = f"quantile_{0.5}" if 0.5 in quantiles else pred_df.mean(axis=1)

            output = pd.DataFrame(
                {
                    "mean_predictions": pred_df[q50_col] if isinstance(q50_col, str) else q50_col,
                    "knowledge_uncertainty": None,
                    "data_uncertainty": None,
                    "total_uncertainty": total_uncertainty,
                },
                index=index,
            )

            if return_quantiles:
                return output, pred_df
            if uncertainty_for_opt:
                return output.loc[:, ["total_uncertainty"]]
            return output

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
