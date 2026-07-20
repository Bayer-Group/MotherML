"""
CatBoost Model Wrappers for the Mother Framework

This module provides custom wrappers for CatBoost models (regression, classification, and Gaussian process regression)
with advanced hyperparameter tuning and uncertainty estimation capabilities, designed for seamless integration with the
Mother machine learning framework.

Key Features:
- Unified interfaces for CatBoostRegressor, CatBoostClassifier,
  CatBoost Gaussian Process Regressor, and CatBoost Ranker.
- Dynamic Optuna hyperparameter search spaces, including loss-specific
  and tree structure parameters.
- Consistent uncertainty estimation and active learning support via
  standardized prediction methods.
- Support for multi-quantile regression, RMSEWithUncertainty, Focal loss,
  and other advanced CatBoost features.
- Handles model serialization, parameter management, and compatibility
  with scikit-learn and Mother conventions.

Classes:
- _CatboostHyperParams: Utility for defining and managing CatBoost hyperparameter spaces.
- CatboostRegressorMother: Extended CatBoostRegressor with uncertainty estimation and quantile support.
- CatboostGaussianProcessRegressorMother: CatBoost-based Gaussian Process regressor for epistemic uncertainty.
- CatboostClassifierMother: Unified classifier for binary and multiclass tasks with loss-specific tuning.
- CatboostRankerMother: Extended CatBoostRanker with rank prediction, uncertainty estimation, and Optuna tuning support.

All classes provide methods for Optuna-based hyperparameter optimization, uncertainty-aware prediction, and
Mother framework compatibility.
"""

import copy
import logging
from functools import wraps
from typing import Any, Callable, Optional, Union, cast

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
from optuna.trial import Trial
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y

import mother.ml.properties as props
from mother.ml import utils
from mother.ml.core import AbstractMotherPipeline
from mother.ml.models import utils as models_utils

module_logger = logging.getLogger(__name__)
DEFAULT_QUANTILES: list[float] = [0.25, 0.5, 0.75]


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert an array of scores to 1-based integer ranks (rank 1 = highest score)."""
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def scores_matrix_to_ranks(score_matrix: np.ndarray) -> np.ndarray:
    """Convert score matrix (n_samples, n_ensembles) to per-ensemble 1-based ranks."""
    arr = np.asarray(score_matrix)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D score_matrix, got {arr.ndim}D.")
    return np.column_stack([scores_to_ranks(arr[:, ensemble_idx]) for ensemble_idx in range(arr.shape[1])]).astype(
        float
    )


def _validate_ranking_group_id(group_id: np.ndarray, n_samples: int) -> np.ndarray:
    """Validate and normalise a group ID array to 1-D, aligned with X rows."""
    group_arr = np.asarray(group_id).reshape(-1)
    if group_arr.shape[0] != n_samples:
        raise ValueError(f"group_id length must match number of rows in X ({n_samples}), got {group_arr.shape[0]}.")
    return group_arr


def _iter_ranking_group_indices(group_id: np.ndarray) -> list[np.ndarray]:
    """Return stable, input-order index arrays (positional) for each unique group."""
    return [np.flatnonzero(group_id == g) for g in pd.unique(group_id)]


def ensure_metadata_routing(func: Callable) -> Callable:
    """
    Decorator to ensure metadata routing is enabled before executing a function.

    This decorator checks if sklearn's metadata routing is enabled and activates it
    if necessary. It's particularly useful for initializing ranking models that require
    metadata routing for passing additional parameters like group_id.
class _CatboostModelMotherBase(AbstractMotherPipeline):
    def __sklearn_clone__(self):
        """Custom clone that uses content equality instead of identity.

        Newer CatBoost versions internally copy mutable constructor params
        (cat_features, embedding_features, text_features, etc.), which breaks
        sklearn.base.clone's identity check (``param1 is param2``).
        This override reconstructs the estimator from its params directly,
        bypassing that check.

        It also preserves any metadata routing requests (e.g.
        ``set_fit_request(group_id=...)``), which are stored on the instance
        and would otherwise be lost when constructing a fresh object.
        """
        klass: type = self.__class__
        params: dict = self.get_params(deep=False)
        new_params: dict = {k: copy.deepcopy(v) for k, v in params.items()}
        new_obj = klass(**new_params)

        # Verify that mutable params were copied with equal content.
        # Use explicit type-dispatch to avoid ambiguous truth-value errors from
        # numpy arrays and pandas objects when using a plain `!=` comparison.
        cloned_params = new_obj.get_params(deep=False)
        for key, original_value in params.items():
            cloned_value = cloned_params[key]
            try:
                if isinstance(original_value, np.ndarray) or isinstance(cloned_value, np.ndarray):
                    # equal_nan=True: NaN values at the same position count as equal
                    equal = np.array_equal(original_value, cloned_value, equal_nan=True)
                elif isinstance(original_value, (pd.Series, pd.DataFrame)) and isinstance(
                    cloned_value, type(original_value)
                ):
                    equal = original_value.equals(cloned_value)
                else:
                    equal = bool(original_value == cloned_value)
                    if not equal:
                        # NaN != NaN by IEEE 754; x != x is True only for NaN
                        try:
                            equal = original_value != original_value and cloned_value != cloned_value
                        except Exception:
                            pass
            except Exception:
                # Last resort: compare reprs (handles custom types with no __eq__)
                equal = repr(original_value) == repr(cloned_value)
            if not equal:
                raise RuntimeError(
                    f"Parameter '{key}' was not correctly cloned: original={original_value!r}, cloned={cloned_value!r}"
                )

        # Preserve metadata routing requests across clone
        if hasattr(self, "_metadata_request"):
            new_obj._metadata_request = copy.deepcopy(self._metadata_request)
            # MetadataRequest does not implement __eq__, so compare via repr
            if repr(self._metadata_request) != repr(new_obj._metadata_request):
                raise RuntimeError(
                    "Metadata routing requests were not correctly cloned: "
                    f"original={self._metadata_request!r}, "
                    f"cloned={new_obj._metadata_request!r}"
                )

        return new_obj


class _CatboostHyperParams(AbstractMotherPipeline):
    """
    A utility class for managing and defining hyperparameter spaces for CatBoost models.

    This class provides methods to define hyperparameter spaces for CatBoost models (e.g., CatBoostClassifier,
    CatBoostRegressor) and to post-process the suggested hyperparameters. It supports dynamic hyperparameter
    tuning using Optuna and integrates seamlessly with the Mother framework.

    Attributes
    ----------
    tune_boosting_type : bool
        Whether to include the "boosting_type" parameter in the hyperparameter space for tuning.
    tune_tree_structure_type : bool
        Whether to include the "grow_policy" parameter in the hyperparameter space for tuning.
        If False Symmetric Trees are used which allows the use of i.e. object importance or
        monotonic constraints.
    tune_loss_function : bool
        Whether to include the loss function in the hyperparameter search space.
        If False, the loss function set on the model at construction time is kept fixed
        throughout tuning and ``suggested_params_loss`` is not called.

    Methods
    -------
    get_hyperparameter_space(X, y, trial, prefix=None) -> dict
        Defines the hyperparameter search space for CatBoost models based on the input data and trial.
    suggested_params_loss(trial, suggested_params, y, prefix) -> dict
        Adds loss-specific hyperparameters to the suggested parameters based on the target type.
    """

    def __init__(
        self,
        tune_boosting_type: bool = False,
        tune_tree_structure_type: bool = True,
        tune_loss_function: bool = True,
    ):
        """
        Initialize the _CatboostHyperParams.

        Args:
            tune_boosting_type : bool, optional
                Whether to include the "boosting_type" parameter in the hyperparameter space for tuning.
            tune_tree_structure_type : bool, optional
                Whether to include the "grow_policy" parameter in the hyperparameter space for tuning.
                If False Symmetric Trees are used which allows the use of i.e. object importance or
                monotonic constraints.
            tune_loss_function : bool, optional
                Whether to include the loss function in the hyperparameter search space.
                If False, the loss function set on the model at construction time is kept fixed
                throughout tuning and ``suggested_params_loss`` is not called. Defaults to ``True``.
        """
        self.tune_boosting_type = tune_boosting_type
        self.tune_tree_structure_type = tune_tree_structure_type
        self.tune_loss_function = tune_loss_function

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        min_depth, max_depth = utils.calc_range_tree_depth(X)

        suggested_params = {
            prefix + "bootstrap_type": trial.suggest_categorical(
                prefix + "bootstrap_type", ("Bayesian", "MVS", "Bernoulli")
            ),
            prefix + "learning_rate": trial.suggest_float(prefix + "learning_rate", 0.000001, 0.5, log=True),
            prefix + "random_strength": trial.suggest_float(prefix + "random_strength", 0, 2, log=False),
        }

        if self.tune_tree_structure_type:
            suggested_params[prefix + "grow_policy"] = trial.suggest_categorical(
                prefix + "grow_policy", ("SymmetricTree", "Depthwise", "Lossguide")
            )
        else:
            suggested_params[prefix + "grow_policy"] = "SymmetricTree"

        if suggested_params[prefix + "grow_policy"] != "SymmetricTree":
            suggested_params[prefix + "boosting_type"] = "Plain"
        elif self.tune_boosting_type:
            suggested_params[prefix + "boosting_type"] = trial.suggest_categorical(
                prefix + "boosting_type", ("Plain", "Ordered")
            )

        if suggested_params[prefix + "grow_policy"] == "Lossguide":
            suggested_params[prefix + "max_depth"] = max_depth
            (
                min_number_of_leaves,
                max_number_of_leaves,
            ) = utils.depth_to_leaves_for_lossguide(min_depth, max_depth)
            suggested_params[prefix + "max_leaves"] = trial.suggest_int(
                prefix + "max_leaves", min_number_of_leaves, max_number_of_leaves
            )
        else:
            suggested_params[prefix + "max_depth"] = trial.suggest_int(prefix + "max_depth", min_depth, max_depth)

        if self.tune_loss_function:
            suggested_params = self.suggested_params_loss(trial, suggested_params, y, prefix)

        module_logger.info(f"Suggested parameters in trial {trial.number}: {suggested_params}")

        return suggested_params

    def suggested_params_loss(self, trial: Trial, suggested_params: dict, y: pd.DataFrame, prefix: str) -> dict:
        """
        Adds loss-specific hyperparameters to the suggested parameters based on the target type.

        Args:
            trial : optuna.trial.Trial
                Optuna trial object.
            suggested_params : dict
                Current suggested parameters.
            y : pd.DataFrame
                Target data.
            prefix : str
                Parameter prefix.

        Returns:
            dict: Updated suggested parameters.
        """
        return suggested_params


class CatboostRegressorMother(CatBoostRegressor, _CatboostModelMotherBase, _CatboostHyperParams):
    """
    A custom implementation of CatBoostRegressor with extended functionality for hyperparameter tuning.

    This class extends the CatBoostRegressor and integrates with the Mother framework to provide
    dynamic hyperparameter tuning using Optuna. It supports loss-specific hyperparameter suggestions
    and post-processing for regression tasks.

    Methods
    -------
    default_parameters(prefix: str = "") -> dict
        Returns the default parameters for the CatBoostRegressor.
    get_params(deep=True) -> dict
        Returns the current parameters for the CatBoostRegressor.
    set_params(**params) -> self
        Sets the given parameters with new values.
    suggested_params_loss(trial, suggested_params, y, prefix) -> dict
        Adds loss-specific hyperparameters to the suggested parameters based on the target type.
    predict_uncertainty(X, n_ensembles=10, n_threads=1, uncertainty_for_opt=False) -> np.ndarray or pd.DataFrame
        Estimates target values and uncertainty.
    """

    def __init__(
        self,
        target_type: props.TargetType = "single_target",
        tune_tree_structure_type: bool = True,
        tune_boosting_type: bool = False,
        tune_loss_function: bool = True,
        quantiles: list[float] | None = None,
        data_uncertainty: bool = False,
        model_type: props.ModelType = "regression",
        **kwargs,
    ):
        """
        Initialize the CatboostRegressorMother.

        Args:
            target_type : str, optional
                Target type ("single_target" or "multi_target").
            tune_tree_structure_type : bool, optional
                Whether to include the "grow_policy" parameter in hyperparameter tuning.
            tune_boosting_type : bool, optional
                Whether to tune boosting_type.
            tune_loss_function : bool, optional
                Whether to include the loss function in the hyperparameter search space.
                If ``False``, the loss function is fixed at construction time and
                ``suggested_params_loss`` is not called during Optuna tuning.
                Defaults to ``True``.
            quantiles : list[float] or None, optional
                Quantiles for multi-quantile regression.
            data_uncertainty : bool, optional
                Use RMSEWithUncertainty loss to calculate both
                data and model uncertainties.
            model_type : str, optional
                Model type (should be "regression").
            **kwargs
                Additional CatBoostRegressor parameters.
        """
        # Initialize hyperparameter tuning configuration
        _CatboostHyperParams.__init__(self, tune_boosting_type, tune_tree_structure_type, tune_loss_function)

        # set the correct model_type
        if model_type != "regression":
            # model_type is implemented for the consistent design with the classifier
            raise ValueError("model_type for CatboostRegressorMother must be 'regression'.")
        self.model_type = model_type

        self.target_type = target_type
        self.data_uncertainty = data_uncertainty

        if quantiles is not None:
            if self.data_uncertainty:
                module_logger.error("data_uncertainty cannot be True when quantiles are given. It'll be reset")
                self.data_uncertainty = False
            # multi-quantile regression
            # Validate quantiles
            if not all(0 < q < 1 for q in quantiles):
                raise ValueError("Quantiles must be a list of floats between 0 and 1.")
            # Store original quantiles for sklearn clone compatibility (get_params/set_params round-trip).
            # Store the original quantiles exactly as passed (for cloning compatibility).
            self.quantiles = quantiles
            # Build extended list: ensure 0.5 for median and DEFAULT_QUANTILES (0.25, 0.75) for IQR.
            quantiles_processed = list(quantiles)

            for q in DEFAULT_QUANTILES:
                if q not in quantiles_processed:
                    quantiles_processed.append(q)
            quantiles_processed.sort()

            module_logger.info(
                "Using quantile regression with mandatory default quantiles %s and user quantiles %s; "
                "final quantiles used by the model: %s",
                DEFAULT_QUANTILES,
                self.quantiles,
                quantiles_processed,
            )

            self._quantiles_processed = quantiles_processed

            if "loss_function" not in kwargs:
                kwargs["loss_function"] = f"MultiQuantile:alpha={', '.join(map(str, quantiles_processed))}"
        else:
            self.quantiles = None
            self._quantiles_processed = None

        if data_uncertainty and quantiles is None:
            kwargs["loss_function"] = "RMSEWithUncertainty"
        elif "loss_function" not in list(kwargs):
            # A specific loss not given. Use the default loss function
            module_logger.warning("No loss_function specified. Using default loss function based on target type.")
            kwargs["loss_function"] = utils.default_loss_function(self.model_type, self.target_type)

        # handle posterior_sampling for uncertainty
        if "posterior_sampling" not in kwargs.keys():
            kwargs["posterior_sampling"] = True

        for key, val in self.default_parameters().items():
            if key not in list(kwargs):
                kwargs[key] = val

        CatBoostRegressor.__init__(self, **kwargs)

    def __getstate__(self):
        state = super(CatBoostRegressor, self).__getstate__()
        state.update(
            {
                "target_type": self.target_type,
                "tune_boosting_type": self.tune_boosting_type,
                "tune_loss_function": self.tune_loss_function,
                "model_type": self.model_type,
                "quantiles": self.quantiles,
                "_quantiles_processed": getattr(self, "_quantiles_processed", None),
                "data_uncertainty": self.data_uncertainty,
                "tune_tree_structure_type": self.tune_tree_structure_type,
            }
        )
        return state

    def __setstate__(self, state):
        self.target_type = state.pop("target_type", "single_target")
        self.tune_boosting_type = state.pop("tune_boosting_type", False)
        self.tune_loss_function = state.pop("tune_loss_function")
        self.model_type = state.pop("model_type", "regression")
        self.quantiles = state.pop("quantiles", None)
        self._quantiles_processed = state.pop("_quantiles_processed", None)
        self.data_uncertainty = state.pop("data_uncertainty", False)
        self.tune_tree_structure_type = state.pop("tune_tree_structure_type", True)

        super(CatBoostRegressor, self).__setstate__(state)

    def get_params(self, deep=True):
        """
        Returns the current parameters for the CatBoostRegressor.

        Args:
            deep : bool, optional
                Whether to return parameters of subobjects.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = super().get_params(deep=deep)
        params.update(
            {
                "quantiles": self.quantiles,
                "target_type": self.target_type,
                "tune_boosting_type": self.tune_boosting_type,
                "tune_loss_function": self.tune_loss_function,
                "model_type": self.model_type,
                "data_uncertainty": self.data_uncertainty,
                "tune_tree_structure_type": self.tune_tree_structure_type,
            }
        )

        return params

    def set_params(self, **params):
        """
        Sets the given parameters with new values.

        Args:
            **params
                Parameters to set.

        Returns:
            self
        """
        our_params = [
            "target_type",
            "tune_boosting_type",
            "tune_loss_function",
            "model_type",
            "quantiles",
            "data_uncertainty",
            "tune_tree_structure_type",
        ]

        for param in our_params:
            if param in params:
                setattr(self, param, params[param])

                params.pop(param)

        return super().set_params(**params)

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Returns the default recommended parameters for the CatBoostRegressor.

        Args:
            prefix : str, optional
                Optional prefix for parameter names.

        Returns:
            dict: Default parameters.
        """
        return models_utils.add_prefix_to_dict_keys(
            {
                "learning_rate": 0.03,
                "bootstrap_type": "Bayesian",
                "random_strength": 1,
                "grow_policy": "SymmetricTree",
                "boosting_type": "Plain",
                "max_depth": 6,
                "loss_function": "RMSE" if self.target_type != "multi_target" else "MultiRMSEWithMissingValues",
            },
            prefix=prefix,
        )

    def suggested_params_loss(self, trial: Trial, suggested_params: dict, y: pd.DataFrame, prefix: str) -> dict:
        """
        Adds loss-specific hyperparameters to the suggested parameters based on the target type.

        Args:
            trial : optuna.trial.Trial
                Optuna trial object.
            suggested_params : dict
                Current suggested parameters.
            y : pd.DataFrame
                Target data.
            prefix : str
                Parameter prefix.

        Returns:
            dict: Updated suggested parameters.
        """
        # loss function other than the three -> don't optimize the type of loss
        losses_for_optim = ["RMSE", "MAE", "LogCosh"]

        if (self.target_type == "single_target") and (self._init_params["loss_function"] in losses_for_optim):
            if np.all(y >= 0):
                module_logger.debug(
                    "Appending Tweedie loss function to hyperparameter tuning since all values are positive"
                )
                losses_for_optim.append("Tweedie")

            suggested_loss_function = trial.suggest_categorical(prefix + "loss_function", losses_for_optim)

            if suggested_loss_function == "Tweedie":
                variance_power = trial.suggest_float(prefix + "Tweedie:variance_power", 1.000001, 1.999999)
                suggested_loss_function = "Tweedie:variance_power=" + str(variance_power)

            suggested_params[prefix + "loss_function"] = suggested_loss_function

        return suggested_params

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predicts target values.

        Args:
            X : pd.DataFrame
                Input data.

        Returns:
            np.ndarray : Predictions
        """

        if self._init_params["loss_function"] == "RMSEWithUncertainty":
            return super().predict(data=X, **kwargs)[:, 0]  # first column
        elif self.quantiles is not None:
            # for quantile regression - return the median prediction
            return super().predict(data=X, **kwargs)[:, self._quantiles_processed.index(0.5)]
        else:
            return super().predict(data=X, **kwargs)

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        n_ensembles: int = 10,
        n_threads: int = 1,
        uncertainty_for_opt: bool = False,
        return_quantiles: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        """
        Estimate targets and uncertainty for regression.

        Args:
            X : pd.DataFrame
                Input data.
            n_ensembles : int, optional
                Number of ensembles.
            n_threads : int, optional
                Number of threads.
            uncertainty_for_opt : bool, optional
                If True, return only uncertainty for optimization.
            return_quantiles : bool, optional
                If True and using quantile regression, return tuple (output_df, quantile_array).

        Returns:
            pd.DataFrame | tuple[pd.DataFrame, np.ndarray]: Predictions with uncertainty.
                Returns a tuple (uncertainty_df, quantile_array) only when
                return_quantiles=True and quantile regression is used; otherwise returns pd.DataFrame.
        """
        quantile_array = None
        # Get predictions from the model's predict method
        model_predictions = self.predict(X)

        if self.quantiles is not None:
            # Use the model's predict method to get multi-quantile predictions
            # return samples-by-quantiles prediction
            quantile_array = super().predict(X)
            module_logger.debug("Shape of predictions: %s", quantile_array.shape)
            if quantile_array.shape[1] != len(self._quantiles_processed):
                logging.warning("Mismatch between predictions and quantiles. Check model configuration.")

            # For quantile-based regression, the interquartile range (IQR) represents
            # the total uncertainty. In this implementation quantile regression
            # does not provide a natural decomposition into knowledge (epistemic) and
            # data (aleatoric) uncertainty components. Therefore, these fields are set
            # to None. The 'total_uncertainty' column contains the IQR (Q3 - Q1)
            # which is the primary uncertainty measure.
            # For detailed uncertainty decomposition, consider using RMSEWithUncertainty
            # loss or ensemble-based approaches (e.g., virtual ensembles with staged_predict).
            uncertainty_df = pd.DataFrame(
                {
                    "pred": model_predictions,
                    "mean_predictions": None,
                    "knowledge_uncertainty": None,
                    "data_uncertainty": None,
                    "total_uncertainty": quantile_array[:, self._quantiles_processed.index(0.75)]
                    - quantile_array[:, self._quantiles_processed.index(0.25)],
                },
                index=X.index,
            )

        elif self.target_type == "single_target":
            uncertainty_df = utils.get_virtual_prediction(
                X=X,
                model=self,
                virtual_ensembles_count=n_ensembles,
                thread_count=n_threads,
            )
            # set the index of data frame
            uncertainty_df.index = X.index
            # Insert one column with model predictions
            uncertainty_df.insert(0, "pred", model_predictions)

        elif self.target_type == "multi_target":
            module_logger.info("Using custom knowledge uncertainty prediction due to multi target regression")
            eval_period = max(1, self.tree_count_ // n_ensembles)
            staged_generator = self.staged_predict(
                X,
                ntree_start=0,
                ntree_end=self.tree_count_,
                eval_period=eval_period,
                thread_count=n_threads,
                verbose=True,
            )

            staged_predictions = list(staged_generator)  # get all results from the generator
            if len(staged_predictions) != n_ensembles:
                module_logger.warning(f"Generated {len(staged_predictions)} ensembles, expected {n_ensembles}")

            # Validate the structure of staged_predictions
            if not staged_predictions or not isinstance(staged_predictions[0], np.ndarray):
                raise ValueError(
                    "Unexpected structure in staged_predictions. Ensure the model is trained on multi-target data."
                )

            # calculate uncertainty and mean predictions from multiple tree results
            knowledge_uncertainty: pd.DataFrame = pd.DataFrame(
                np.array(staged_predictions).std(axis=0, ddof=1), index=X.index
            )

            mean_predictions: pd.DataFrame = pd.DataFrame(np.array(staged_predictions).mean(axis=0), index=X.index)

            # Set the index to match the input DataFrame
            # knowledge_uncertainty.index = X.index
            # mean_predictions.index = X.index

            # Assign column names and reorder: all predictions first, then means, then uncertainties
            n_targets = staged_predictions[0].shape[1] if len(staged_predictions[0].shape) > 1 else 1
            mean_predictions.columns = [f"target_{i}_mean_predictions" for i in range(n_targets)]
            knowledge_uncertainty.columns = [f"target_{i}_knowledge_uncertainty" for i in range(n_targets)]
            target_indices = [str(i) for i in range(n_targets)]

            module_logger.info(f"Shape of mean_predictions: {mean_predictions.shape}")
            module_logger.info(f"Number of targets: {n_targets}")

            if len(target_indices) != mean_predictions.shape[1]:
                raise ValueError(
                    f"Length mismatch: target_indices has {len(target_indices)} elements, "
                    f"but mean_predictions has {mean_predictions.shape[1]} columns."
                )

            if isinstance(model_predictions, np.ndarray) and model_predictions.ndim == 1:
                model_predictions = model_predictions.reshape(-1, 1)
            y_pred_df = pd.DataFrame(model_predictions, index=X.index)
            y_pred_df.columns = [f"target_{idx}_pred" for idx in target_indices]

            # Keep the multi-target output schema aligned with single-target output.
            # Data and total uncertainty are currently unavailable in this path.
            data_uncertainty = pd.DataFrame(
                np.nan,
                index=X.index,
                columns=[f"target_{idx}_data_uncertainty" for idx in target_indices],
            )
            total_uncertainty = pd.DataFrame(
                np.nan,
                index=X.index,
                columns=[f"target_{idx}_total_uncertainty" for idx in target_indices],
            )

            # Concat in order: predictions, then means, then uncertainties
            uncertainty_df = pd.concat(
                [
                    y_pred_df,
                    mean_predictions,
                    knowledge_uncertainty,
                    data_uncertainty,
                    total_uncertainty,
                ],
                axis=1,
            )
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")

        if uncertainty_for_opt:
            if self.target_type == "multi_target":
                module_logger.warning(
                    "uncertainty_for_opt=True for multi-target regression: "
                    "returning max of per-target knowledge_uncertainty as total_uncertainty."
                )
                ku_cols = [c for c in uncertainty_df.columns if c.endswith("_knowledge_uncertainty")]
                return pd.DataFrame(
                    {"total_uncertainty": uncertainty_df[ku_cols].astype(float).max(axis=1)},
                    index=X.index,
                )
            elif self.quantiles is not None:
                return pd.DataFrame(
                    {"total_uncertainty": uncertainty_df["total_uncertainty"]},
                    index=X.index,
                )
            else:
                return pd.DataFrame(
                    {"knowledge_uncertainty": uncertainty_df["knowledge_uncertainty"]},
                    index=X.index,
                )

        # Check if should return quantile array along with standard output
        if return_quantiles and quantile_array is not None:
            return uncertainty_df, quantile_array

        return uncertainty_df


class CatboostGaussianProcessRegressorMother(CatBoostRegressor, _CatboostModelMotherBase, _CatboostHyperParams):
    """
    Scikit-learn-compatible CatBoost Gaussian Process Regressor for Uncertainty Estimation.

    This estimator uses CatBoost's `sample_gaussian_process` method to perform
    Gaussian Process regression using an ensemble of CatBoost models, as described in
    "Gradient Boosting Performs Gaussian Process Inference" (https://arxiv.org/abs/2206.05608).
    It provides mean predictions and epistemic (knowledge) uncertainty estimates, and
    integrates with the Mother framework for hyperparameter optimization and uncertainty-aware workflows.

    Attributes
    ----------
    models_ : list
        List of CatBoost models representing posterior samples (the ensemble).
    params : dict
        Dictionary of parameters used for sampling and fitting.

    Methods
    -------
    fit(X, y)
        Fit the ensemble of CatBoost models using Gaussian Process sampling.
    predict(X)
        Predict mean.
    predict_uncertainty(X, uncertainty_for_opt=False)
        Estimate mean and uncertainty.
    get_params(deep=True)
        Get estimator parameters (scikit-learn API).
    set_params(**params)
        Set estimator parameters (scikit-learn API).
    get_hyperparameter_space(X, y, trial, prefix="")
        Defines the Optuna hyperparameter search space for this estimator.
    __getstate__()
        Support for pickling.
    __setstate__(state)
        Support for unpickling.
    """

    def __init__(
        self,
        samples: int = 10,
        prior_iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        sigma: float = 0.1,
        delta: float = 0,
        random_strength: float = 0.1,
        random_score_type: str = "Gumbel",
        eps: float = 1e-4,
        tune_boosting_type: bool = False,
        tune_tree_structure_type: bool = True,
        verbose: bool = False,
        model_type: str = "regression",
        target_type: props.TargetType = "single_target",
        **kwargs,
    ):
        """
        Initialize CatboostGaussianProcessRegressorMother.

        Args:
            samples : int, default=10
                Number of posterior samples (ensemble size).
            prior_iterations : int, default=100
                Number of boosting iterations for prior.
            learning_rate : float, default=0.1
                Learning rate for boosting.
            max_depth : int, default=6
                Tree depth.
            sigma : float, default=0.1
                Kernel variance parameter.
            delta : float, default=0
                Noise variance parameter.
            random_strength : float, default=0.1
                Randomness strength for splits.
            random_score_type : str, default="Gumbel"
                Type of random score for splits.
            eps : float, default=1e-4
                Numerical stability parameter.
            tune_boosting_type : bool, default=False
                Whether to include boosting type in hyperparameter optimization.
            tune_tree_structure_type : bool, default=True
                Whether to include the "grow_policy" parameter in hyperparameter tuning.
            verbose : bool, default=False
                Whether to print training logs during fitting.
            model_type : str, default="regression"
                The type of model. For compatibility with the Mother framework.
            target_type : str, default="single_target"
                Target type (must be "single_target").
            **kwargs : dict
                Additional parameters for CatBoost's `sample_gaussian_process` method.
        """
        # Initialize hyperparameter tuning configuration
        _CatboostHyperParams.__init__(self, tune_boosting_type, tune_tree_structure_type)

        # Check for 'model_type'
        if model_type != "regression":
            raise ValueError("model_type for CatboostGaussianProcessRegressorMother must be 'regression'.")

        if target_type != "single_target":
            raise ValueError(
                "target_type for CatboostGaussianProcessRegressorMother must be 'single_target'. "
                "Multi-target regression is not supported."
            )

        self.model_type = model_type
        self.target_type = target_type

        # Store GP-specific parameters as instance attributes
        self.samples = samples
        self.prior_iterations = prior_iterations
        self.sigma = sigma
        self.delta = delta
        self.eps = eps

        # Build parameters for CatBoostRegressor initialization
        catboost_params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_strength": random_strength,
            "random_score_type": random_score_type,
            "verbose": verbose,
        }

        # Add any additional kwargs
        catboost_params.update(kwargs)

        # Add default parameters that aren't already set
        for key, val in self.default_parameters().items():
            if key not in catboost_params:
                catboost_params[key] = val

        # Initialize CatBoostRegressor properly to ensure _init_params is set
        CatBoostRegressor.__init__(self, **catboost_params)

        # Store parameters for GP functionality (used in fit method)
        self.gp_params = {
            "samples": samples,
            "prior_iterations": prior_iterations,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "sigma": sigma,
            "delta": delta,
            "random_strength": random_strength,
            "random_score_type": random_score_type,
            "eps": eps,
            "verbose": verbose,
        }

        # Initialize empty models list
        self.models_ = []

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): Whether to return parameters of subobjects.

        Returns:
            dict: Parameter names mapped to their values.
        """
        # Get base CatBoost parameters
        base_params = super().get_params(deep=deep)

        # Add our custom parameters
        custom_params = {
            "model_type": self.model_type,
            "target_type": self.target_type,
            "tune_boosting_type": self.tune_boosting_type,
            "tune_tree_structure_type": self.tune_tree_structure_type,
            "samples": self.samples,
            "prior_iterations": self.prior_iterations,
            "sigma": self.sigma,
            "delta": self.delta,
            "eps": self.eps,
        }

        # Merge parameters
        base_params.update(custom_params)
        return base_params

    def set_params(self, **params):
        """
        Set parameters for this estimator.

        Args:
            **params: Parameters to set.

        Returns:
            self
        """
        # Handle our custom parameters
        custom_param_names = {
            "model_type",
            "target_type",
            "tune_boosting_type",
            "tune_tree_structure_type",
            "samples",
            "prior_iterations",
            "sigma",
            "delta",
            "eps",
        }

        # Update custom parameters and remove them from params dict
        params_to_remove = []
        for key, value in params.items():
            if key in custom_param_names:
                setattr(self, key, value)
                # Also update gp_params if it exists there
                if hasattr(self, "gp_params") and key in self.gp_params:
                    self.gp_params[key] = value
                params_to_remove.append(key)

        # Remove handled parameters
        for key in params_to_remove:
            params.pop(key, None)

        # Let parent handle remaining parameters
        if params:
            super().set_params(**params)

        return self

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Return default hyperparameters for CatboostGaussianProcessRegressorMother.

        Args:
            prefix : str
                Optional prefix for parameter names.

        Returns:
            dict: Default parameters.
        """
        # Use the same defaults as CatboostRegressorMother for consistency
        base_defaults = {
            "learning_rate": 0.03,
            "bootstrap_type": "Bayesian",
            "random_strength": 1,
            "grow_policy": "SymmetricTree",
            "boosting_type": "Plain",
            "max_depth": 6,
            "loss_function": "RMSE",
        }

        return models_utils.add_prefix_to_dict_keys(base_defaults, prefix=prefix)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the CatboostGaussianProcessRegressorMother.

        Args:
            X : pd.DataFrame
                Training data.
            y : pd.Series
                Target values.

        Returns:
            self
        """
        if self.target_type != "single_target":
            raise ValueError(
                "target_type for CatboostGaussianProcessRegressorMother must be 'single_target'. "
                "Multi-target regression is not supported."
            )

        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True, dtype=None)
        posterior_iterations = 1000 - self.prior_iterations

        module_logger.warning(
            f"Parameter prior_iterations set to: {self.prior_iterations}, "
            f"automatically updating posterior_iterations to: {posterior_iterations}."
        )

        self.models_ = catboost.core.sample_gaussian_process(
            X=X,
            y=y,
            samples=self.samples,
            posterior_iterations=posterior_iterations,
            prior_iterations=self.prior_iterations,
            learning_rate=self.gp_params["learning_rate"],
            depth=self.gp_params["max_depth"],
            sigma=self.sigma,
            delta=self.delta,
            random_strength=self.gp_params["random_strength"],
            random_score_type=self.gp_params["random_score_type"],
            eps=self.eps,
            verbose=self.gp_params["verbose"],
        )
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict regression targets.

        Args:
            X : pd.DataFrame
                Input data.
            **kwargs
                Additional keyword arguments passed to CatBoostRegressor.predict().
                Examples: prediction_type, ntree_start, ntree_end, thread_count, etc.

        Returns:
            np.ndarray: Mean predictions averaged over all ensemble models.
        """
        if not hasattr(self, "models_") or not self.models_:
            module_logger.error("Prediction requested before model was fitted. Call 'fit' before 'predict'.")
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'predict'.")

        predictions = np.array([model.predict(X, **kwargs) for model in self.models_])
        mean_predictions = predictions.mean(axis=0)
        return mean_predictions

    def predict_uncertainty(self, X: pd.DataFrame, uncertainty_for_opt: bool = False, **kwargs) -> pd.DataFrame:
        """
        Estimate knowledge uncertainty for regression.

        Args:
            X : pd.DataFrame
                Input data.
            uncertainty_for_opt : bool
                If True, return only uncertainty for optimization.
            **kwargs
                Additional keyword arguments passed to CatBoostRegressor.predict().
                Examples: prediction_type, ntree_start, ntree_end, thread_count, etc.

        Returns:
            pd.DataFrame: Predictions with uncertainty (knowledge_uncertainty computed as std across models).
        """
        if not hasattr(self, "models_") or not self.models_:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'predict_uncertainty'.")

        predictions = np.array([model.predict(X, **kwargs) for model in self.models_])

        mean_predictions = predictions.mean(axis=0)
        knowledge_uncertainty = predictions.std(axis=0)

        results = pd.DataFrame(
            {
                "pred": mean_predictions,
                "mean_predictions": None,
                "knowledge_uncertainty": knowledge_uncertainty,
                "data_uncertainty": None,
                "total_uncertainty": None,
            },
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )

        if uncertainty_for_opt:
            return pd.DataFrame(
                {"knowledge_uncertainty": results["knowledge_uncertainty"]},
                index=X.index if isinstance(X, pd.DataFrame) else None,
            )

        return results

    def get_hyperparameter_space(self, X, y, trial, prefix: str = "") -> dict:
        """
        Define the Optuna hyperparameter search space for CatBoost Gaussian Process regression.

        Args:
            X : pd.DataFrame
                Feature data.
            y : pd.Series
                Target data.
            trial : optuna.trial.Trial
                Optuna trial object.
            prefix : str
                Optional prefix for parameter names.

        Returns:
            dict: Suggested hyperparameters for the current trial.
        """
        # Get the base hyperparameter space from parent class
        base_params = super().get_hyperparameter_space(X, y, trial, prefix)

        # Add GP-specific parameters
        gp_params = {
            "prior_iterations": trial.suggest_int(prefix + "prior_iterations", 50, 500),
            "samples": trial.suggest_int(prefix + "samples", 5, 50),
            "sigma": trial.suggest_float(prefix + "sigma", 1e-3, 0.5, log=True),
            "delta": trial.suggest_float(prefix + "delta", 0.0, 0.1),
            "eps": trial.suggest_float(prefix + "eps", 1e-6, 1e-3, log=True),
            "random_score_type": trial.suggest_categorical(
                prefix + "random_score_type", ["Gumbel", "NormalWithModelSizeDecrease"]
            ),
        }

        # Merge with base parameters
        base_params.update(gp_params)

        module_logger.info(f"Suggested parameters in trial {trial.number}: {base_params}")
        return base_params

    def __getstate__(self):
        """Get state for pickling."""
        # Get state from CatBoostRegressor
        state = super().__getstate__()

        # Add our custom attributes
        state.update(
            {
                "model_type": self.model_type,
                "target_type": self.target_type,
                "tune_boosting_type": self.tune_boosting_type,
                "tune_tree_structure_type": self.tune_tree_structure_type,
                "samples": self.samples,
                "prior_iterations": self.prior_iterations,
                "sigma": self.sigma,
                "delta": self.delta,
                "eps": self.eps,
                "gp_params": getattr(self, "gp_params", {}),
                "models_": getattr(self, "models_", []),
            }
        )
        return state

    def __setstate__(self, state):
        """Set state for unpickling."""
        # Extract our custom attributes
        self.model_type = state.pop("model_type", "regression")
        self.target_type = state.pop("target_type", "single_target")
        self.tune_boosting_type = state.pop("tune_boosting_type", False)
        self.tune_tree_structure_type = state.pop("tune_tree_structure_type", True)
        self.samples = state.pop("samples", 10)
        self.prior_iterations = state.pop("prior_iterations", 100)
        self.sigma = state.pop("sigma", 0.1)
        self.delta = state.pop("delta", 0)
        self.eps = state.pop("eps", 1e-4)
        self.gp_params = state.pop("gp_params", {})
        self.models_ = state.pop("models_", [])

        # Let parent handle the rest
        super().__setstate__(state)


class CatboostClassifierMother(CatBoostClassifier, _CatboostModelMotherBase, _CatboostHyperParams):
    """
    Unified CatBoost classifier for binary and multiclass classification with Optuna hyperparameter tuning,
    uncertainty estimation, and active learning support, designed for integration with the Mother framework.

    This class extends CatBoostClassifier and provides:
      - Dynamic hyperparameter search spaces for Optuna, including loss-specific and tree parameters.
      - Automatic handling of boosting type (Plain/Ordered) and grow policy.
      - Loss-specific parameter tuning (e.g., Focal loss alpha/gamma for binary classification).
      - Consistent uncertainty estimation interface for both standard and active learning workflows.
      - Support for both single-target and multi-target classification.

    Attributes
    ----------
    model_type : str
        Classification mode ("classification_binary" or "classification_multiclass").
    target_type : props.TargetType
        Target variable type.
    tune_boosting_type : bool
        Whether boosting type tuning is enabled.

    Methods
    -------
    default_parameters(prefix: str = "") -> dict
        Returns default hyperparameters for CatBoostClassifier.
    get_params(deep=True) -> dict
        Returns all estimator parameters, including custom Mother framework options.
    set_params(**params) -> self
        Sets estimator parameters, handling both CatBoost and Mother-specific options.
    suggested_params_loss(trial, suggested_params, y, prefix) -> dict
        Adds loss-specific hyperparameters (e.g., Focal loss alpha/gamma for binary).
    predict_uncertainty(X, uncertainty_for_opt=False, n_ensembles=10, n_threads=1) -> np.ndarray or pd.DataFrame
        Predicts class labels and estimates uncertainty.

    Notes
    -----
    - For single_target binary classification, supports both "Logloss" and "Focal" loss (with alpha/gamma tuning).
    - For multi target binary classification, uses standard MultiLogloss.
    - For multiclass, uses standard MultiClass loss (no loss-specific tuning provided).
    """

    def __init__(
        self,
        target_type: props.TargetType = "single_target",
        tune_boosting_type: bool = False,
        model_type: props.ModelType = "classification_binary",
        tune_tree_structure_type: bool = True,
        tune_loss_function: bool = True,
        **kwargs,
    ):
        """
        Initialize CatboostClassifierMother.

        Args:
            target_type (str): Target type ("single_target" or "multi_target").
            tune_boosting_type (bool): Whether to tune boosting_type.
            model_type (str): Model type ("classification_binary" or "classification_multiclass").
            tune_tree_structure_type (bool): Whether to include the "grow_policy" parameter in hyperparameter tuning.
            tune_loss_function (bool): Whether to include the loss function in the hyperparameter search space.
                If ``False``, the loss function is fixed at construction time and
                ``suggested_params_loss`` is not called during Optuna tuning. Defaults to ``True``.
            **kwargs: Additional CatBoostClassifier parameters.
        """

        # Initialize hyperparameter tuning configuration
        _CatboostHyperParams.__init__(self, tune_boosting_type, tune_tree_structure_type, tune_loss_function)

        self.model_type = model_type
        self.target_type = target_type

        if "loss_function" not in kwargs:
            kwargs["loss_function"] = utils.default_loss_function(
                model_type=self.model_type, target_type=self.target_type
            )

        # Set auto_class_weights
        if (
            self.model_type == "classification_binary"
            and self.target_type == "single_target"
            and kwargs["loss_function"] == "Logloss"
        ):
            if "auto_class_weights" not in kwargs:
                kwargs["auto_class_weights"] = "Balanced"

        elif self.model_type == "classification_multiclass" and self.target_type == "single_target":
            if "auto_class_weights" not in list(kwargs):
                kwargs["auto_class_weights"] = "Balanced"

        if "posterior_sampling" not in kwargs.keys():
            kwargs["posterior_sampling"] = True

        # Initialize with default parameters
        for key, val in self.default_parameters().items():
            if key not in list(kwargs):
                kwargs[key] = val

        super().__init__(**kwargs)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator, including custom Mother parameters.

        Args:
            deep (bool): Whether to return parameters of subobjects.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = super().get_params(deep=deep)
        params.update(
            {
                "target_type": self.target_type,
                "tune_boosting_type": self.tune_boosting_type,
                "tune_loss_function": self.tune_loss_function,
                "model_type": self.model_type,
                "tune_tree_structure_type": self.tune_tree_structure_type,
            }
        )
        return params

    def set_params(self, **params):
        """
        Set parameters for this estimator, including custom Mother parameters.

        Args:
            **params: Parameters to set.

        Returns:
            self
        """
        our_params = [
            "target_type",
            "tune_boosting_type",
            "tune_loss_function",
            "model_type",
            "tune_tree_structure_type",
        ]

        for param in our_params:
            if param in params:
                setattr(self, param, params[param])
                params.pop(param, None)

        return super().set_params(**params)

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Return default hyperparameters for CatBoostClassifier.

        Args:
            prefix : str, optional
                Optional prefix for parameter names.

        Returns:
            dict: Default parameters.
        """
        loss_function = utils.default_loss_function(model_type=self.model_type, target_type=self.target_type)
        return models_utils.add_prefix_to_dict_keys(
            {
                "learning_rate": 0.03,
                "bootstrap_type": "Bayesian",
                "random_strength": 1,
                "grow_policy": "SymmetricTree",
                "boosting_type": "Plain",
                "max_depth": 6,
                "loss_function": loss_function,
            },
            prefix=prefix,
        )

    def suggested_params_loss(self, trial: Trial, suggested_params: dict, y: pd.DataFrame, prefix: str) -> dict:
        """
        Add classification loss-specific hyperparameters to the suggested parameters.

        Args:
            trial : optuna.trial.Trial
                Optuna trial object.
            suggested_params : dict
                Current suggested parameters.
            y : pd.DataFrame
                Target data.
            prefix : str
                Parameter prefix.

        Returns:
            dict: Updated suggested parameters.
        """

        if self.model_type == "classification_binary":
            if self.target_type == "multi_target":
                return suggested_params

            losses = ("Logloss", "Focal")
            suggested_loss_function = trial.suggest_categorical(prefix + "loss_function", losses)

            if suggested_loss_function == "Focal":
                suggested_params[prefix + "auto_class_weights"] = "None"
                focal_alpha = trial.suggest_float(prefix + "alpha", 0.000001, 0.999999)
                focal_gamma = trial.suggest_float(prefix + "gamma", 0.000001, 7)
                suggested_params[prefix + "loss_function"] = (
                    "Focal:focal_alpha=" + str(focal_alpha) + ";focal_gamma=" + str(focal_gamma)
                )
            else:
                suggested_params[prefix + "loss_function"] = suggested_loss_function

        elif self.model_type == "classification_multiclass":
            if self.target_type == "multi_target":
                raise NotImplementedError(
                    "target_type=multi_target cannot be used with model_type==classification_multiclass."
                )
            else:
                suggested_params[prefix + "loss_function"] = "MultiClass"

        return suggested_params

    def __getstate__(self):
        state = super(CatBoostClassifier, self).__getstate__()
        state.update(
            {
                "target_type": self.target_type,
                "tune_boosting_type": self.tune_boosting_type,
                "tune_loss_function": self.tune_loss_function,
                "model_type": self.model_type,
                "tune_tree_structure_type": self.tune_tree_structure_type,
            }
        )
        return state

    def __setstate__(self, state):
        self.target_type = state.pop("target_type", "single_target")
        self.tune_boosting_type = state.pop("tune_boosting_type", False)
        self.tune_loss_function = state.pop("tune_loss_function")
        self.model_type = state.pop("model_type", "classification_binary")
        self.tune_tree_structure_type = state.pop("tune_tree_structure_type", True)
        super(CatBoostClassifier, self).__setstate__(state)

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        n_ensembles: int = 10,
        n_threads: int = 1,
        uncertainty_for_opt=False,
    ) -> pd.DataFrame:
        """
        Predicts class labels and estimates uncertainty for classification.

        Args:
            X : pd.DataFrame
                Input data.
            n_ensembles : int, optional
                Number of ensembles.
            n_threads : int, optional
                Number of threads.
            uncertainty_for_opt : bool, optional
                If True, return only uncertainty for optimization.

        Returns:
            pd.DataFrame: Predictions with uncertainty and probabilities,
            including the standard model prediction in column "pred".
        """
        if str(self.target_type) == "multi_target" or str(self.model_type) == "classification_multiclass":
            module_logger.warning(
                "Uncertainty prediction for MULTICLASS or MULTI-TARGET is not yet supported. "
                "Please check CatBoost docs for compatibility."
            )

        uncertainty_df = utils.get_virtual_prediction(
            X, model=self, virtual_ensembles_count=n_ensembles, thread_count=n_threads
        )

        # Supports only single
        # Keep standard model predictions separate from virtual-ensemble mean_predictions.
        model_predictions = np.asarray(self.predict(X)).flatten()
        pred_df = pd.DataFrame({"pred": model_predictions}, index=X.index)

        # Get model predictions probabilities
        model_predictions_proba = np.asarray(self.predict_proba(X))

        # Create DataFrame with probabilities
        if model_predictions_proba.ndim == 1:
            model_predictions_proba = model_predictions_proba.reshape(1, -1)

        if model_predictions_proba.ndim != 2:
            raise ValueError(
                "Expected predict_proba output to be a 1D or 2D array, "
                f"got array with {model_predictions_proba.ndim} dimensions"
            )

        if model_predictions_proba.shape[0] != len(X.index):
            raise ValueError(
                "Number of probability predictions does not match number of input rows: "
                f"{model_predictions_proba.shape[0]} vs {len(X.index)} rows"
            )

        n_classes = model_predictions_proba.shape[1]
        prediction_columns_proba = [f"proba_{i}" for i in range(n_classes)]
        proba_df = pd.DataFrame(
            data=model_predictions_proba,
            index=X.index,
            columns=prediction_columns_proba,
        )

        # Keep a stable output order: standard prediction, probabilities, uncertainty terms.
        uncertainty_df = pd.concat([pred_df, proba_df, uncertainty_df], axis=1)

        if uncertainty_for_opt:
            return pd.DataFrame(
                {"knowledge_uncertainty": uncertainty_df["knowledge_uncertainty"]},
                index=X.index,
            )

        return uncertainty_df


class CatboostRankerMother(CatBoostRanker, _CatboostModelMotherBase, _CatboostHyperParams, BaseEstimator):
    """
    A custom implementation of CatBoostRanker with extended functionality for hyperparameter tuning.

    This class extends the CatBoostRanker and integrates with the Mother framework to provide
    dynamic hyperparameter tuning using Optuna. Callers are responsible for enabling sklearn's
    metadata routing (``sklearn.set_config(enable_metadata_routing=True)``) before calling
    ``fit()`` or ``tuner.optimize()``, as it is required for passing ``group_id`` during training.

    Attributes
    ----------
    model_type : str
        The type of model, set to "ranking".
    target_type : props.TargetType
        The type of target variable, typically "single_target" for ranking.
    tune_boosting_type : bool
        Whether to include the "boosting_type" parameter in the hyperparameter space for tuning.
    tune_tree_structure_type : bool
        Whether to include the "grow_policy" parameter in the hyperparameter space for tuning.
    tune_pairwise_type : bool
        Whether to include Pairwise loss functions (``YetiRankPairwise``,
        ``PairLogitPairwise``) in the hyperparameter space for tuning.
        Pairwise losses require ``grow_policy="SymmetricTree"`` and
        ``boosting_type="Plain"``; therefore pairwise tuning is only possible
        when both ``tune_tree_structure_type=False`` and
        ``tune_boosting_type=False``.

        If an incompatible combination is passed at construction time (e.g.
        ``tune_pairwise_type=True`` together with
        ``tune_tree_structure_type=True``), a warning is emitted and
        ``tune_pairwise_type`` is automatically set to ``False``.
    top : Optional[int]
        Maximum number of top samples to consider for ranking metrics (e.g., NDCG@k).
        Only works with YetiRank loss function. Default is 0 (disabled).
    max_pairs : Optional[int]
        Maximum number of pairs to generate for PairLogit losses. This can significantly
        reduce computation time for large ranking tasks. Only applies to PairLogit and
        PairLogitPairwise loss functions. Default is ``None`` (no limit). Also used during
        hyperparameter tuning to set an upper limit on the number of pairs when evaluating
        PairLogit losses.

    Methods
    -------
    default_parameters(prefix: str = "") -> dict
        Returns the default parameters for the CatBoostRanker.

    Notes
    -----
    - The ``__init__`` method is decorated with ``@ensure_metadata_routing`` to automatically
      enable sklearn's metadata routing configuration, which is required for ranking models
      to accept ``group_id`` parameters during training.
    - Passing an explicit Pairwise ``loss_function`` (e.g. ``"YetiRankPairwise"``) together
      with an incompatible ``grow_policy`` or ``boosting_type`` raises a ``ValueError``
      immediately at construction time.
    """

    def __init__(
        self,
        target_type: props.TargetType = "single_target",
        tune_pairwise_type: bool = False,
        tune_boosting_type: bool = False,
        tune_tree_structure_type: bool = True,
        tune_loss_function: bool = True,
        model_type: props.ModelType = "ranking",
        top: Optional[int] = 0,
        max_pairs: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize CatboostRankerMother.

        Validates compatibility between pairwise tuning flags and tree/boosting
        configuration at construction time:

        - If ``tune_pairwise_type=True`` is combined with
          ``tune_tree_structure_type=True`` or ``tune_boosting_type=True``, a
          warning is logged and ``tune_pairwise_type`` is set to ``False``
          (pairwise losses require fixed ``SymmetricTree`` + ``Plain`` and
          cannot coexist with dynamic Optuna categoricals for those params).
        - If an explicit Pairwise ``loss_function`` (e.g.
          ``"YetiRankPairwise"``) is passed together with an incompatible
          ``grow_policy`` or ``boosting_type``, a ``ValueError`` is raised.

        Args:
            target_type : props.TargetType, optional
                Target type (``"single_target"`` for ranking).
            tune_pairwise_type : bool, optional
                Whether to include Pairwise loss functions in the
                hyperparameter search space.  Requires both
                ``tune_tree_structure_type=False`` and
                ``tune_boosting_type=False``; otherwise automatically
                disabled with a warning.
            tune_boosting_type : bool, optional
                Whether to tune ``boosting_type``.
            tune_tree_structure_type : bool, optional
                Whether to include the ``grow_policy`` parameter in
                hyperparameter tuning.
            tune_loss_function : bool, optional
                Whether to include the loss function in the hyperparameter
                search space.  If ``False``, the loss function is fixed at
                construction time and ``suggested_params_loss`` is not called
                during Optuna tuning.  Defaults to ``True``.
            top : Optional[int], optional
                Maximum number of top samples for NDCG calculation.
                Only works with YetiRank loss function.
            max_pairs : Optional[int], optional
                Maximum number of pairs to generate for PairLogit losses.
            **kwargs
                Additional CatBoostRanker parameters.

        Raises:
            ValueError
                If an explicit Pairwise ``loss_function`` is passed with an
                incompatible ``grow_policy`` (not ``SymmetricTree``) or
                ``boosting_type`` (not ``Plain``).
        """
        # Initialize hyperparameter tuning configuration
        _CatboostHyperParams.__init__(self, tune_boosting_type, tune_tree_structure_type, tune_loss_function)

        self.model_type: props.ModelType = "ranking"
        self.target_type: props.TargetType = target_type
        self.tune_pairwise_type: bool = tune_pairwise_type
        self.top: Optional[int] = top
        self.max_pairs: Optional[int] = max_pairs

        # --- Validate pairwise tuning compatibility ---
        # Pairwise losses (YetiRankPairwise, PairLogitPairwise) require SymmetricTree + Plain
        # boosting.  When tune_tree_structure_type or tune_boosting_type is True the tree
        # structure / boosting type varies across Optuna trials, so we cannot guarantee
        # compatibility.  Adding pairwise losses to a dynamic categorical list would also
        # break Optuna (it requires a fixed set of choices per parameter name).
        if self.tune_pairwise_type:
            conflicts: list[str] = []
            if self.tune_tree_structure_type:
                conflicts.append("tune_tree_structure_type=True")
            if self.tune_boosting_type:
                conflicts.append("tune_boosting_type=True")

            if conflicts:
                module_logger.warning(
                    "tune_pairwise_type=True is incompatible with %s. "
                    "Pairwise losses require SymmetricTree grow_policy and Plain boosting_type, "
                    "but these settings vary across Optuna trials when their tuning flags are "
                    "enabled.  Pairwise losses will be excluded from the search space. "
                    "To enable pairwise tuning, set %s.",
                    " and ".join(conflicts),
                    " and ".join(c.replace("True", "False") for c in conflicts),
                )
                self.tune_pairwise_type = False

        # Validate explicit pairwise loss_function against grow_policy / boosting_type
        explicit_loss = kwargs.get("loss_function", "")
        if "Pairwise" in str(explicit_loss):
            explicit_grow = kwargs.get("grow_policy", "SymmetricTree")
            explicit_boost = kwargs.get("boosting_type", "Plain")
            incompatible: list[str] = []
            if explicit_grow != "SymmetricTree":
                incompatible.append(f"grow_policy='{explicit_grow}' (must be 'SymmetricTree')")
            if explicit_boost != "Plain":
                incompatible.append(f"boosting_type='{explicit_boost}' (must be 'Plain')")
            if incompatible:
                raise ValueError(
                    f"Pairwise loss '{explicit_loss}' requires SymmetricTree grow_policy "
                    f"and Plain boosting_type, but got {', '.join(incompatible)}."
                )
            # A fixed Pairwise loss also requires fixed tree structure and boosting type
            # during Optuna tuning; disable the flags if they would vary across trials.
            tuning_conflicts: list[str] = []
            if self.tune_tree_structure_type:
                tuning_conflicts.append("tune_tree_structure_type")
            if self.tune_boosting_type:
                tuning_conflicts.append("tune_boosting_type")
            if tuning_conflicts:
                module_logger.warning(
                    "Explicit Pairwise loss '%s' requires fixed SymmetricTree + Plain boosting, "
                    "but %s are True. Disabling them to prevent incompatible Optuna trials.",
                    explicit_loss,
                    " and ".join(tuning_conflicts),
                )
                self.tune_tree_structure_type = False
                self.tune_boosting_type = False

        if "loss_function" not in list(kwargs):
            if self.top is not None and self.top > 0:
                kwargs["loss_function"] = f"YetiRank:mode=NDCG;top={self.top}"
            else:
                kwargs["loss_function"] = "YetiRank:mode=Classic"

        elif (
            "PairLogit" in kwargs["loss_function"]
            and "max_pairs" not in kwargs["loss_function"]
            and self.max_pairs is not None
        ):
            sep = ";" if ":" in kwargs["loss_function"] else ":"
            kwargs["loss_function"] += f"{sep}max_pairs={self.max_pairs}"

        # Always enable posterior_sampling for uncertainty estimation
        if "posterior_sampling" not in kwargs:
            kwargs["posterior_sampling"] = True

        # Apply defaults, excluding building block parameters that are only for Optuna
        # These parameters (base_loss, mode, dcg_denominator, dcg_type) are combined into loss_function
        tuning_building_blocks = ("base_loss", "mode", "dcg_denominator", "dcg_type")
        for key, val in self.default_parameters().items():
            if key not in list(kwargs) and key not in tuning_building_blocks:
                kwargs[key] = val

        CatBoostRanker.__init__(self, **kwargs)

    def get_params(self, deep: bool = True) -> dict:
        """
        Override get_params to include custom parameters like target_type.
        """
        params = super().get_params(deep=deep)
        params.update(
            {
                "target_type": self.target_type,
                "model_type": self.model_type,
                "tune_pairwise_type": self.tune_pairwise_type,
                "tune_boosting_type": self.tune_boosting_type,
                "tune_tree_structure_type": self.tune_tree_structure_type,
                "tune_loss_function": self.tune_loss_function,
                "top": self.top,
                "max_pairs": self.max_pairs,
            }
        )
        return params

    def set_params(self, **params) -> "CatboostRankerMother":
        """
        Override set_params to handle custom parameters like target_type.
        """
        # model_type must always be 'ranking' — discard any attempt to change it.
        params.pop("model_type", None)
        self.model_type = "ranking"

        for param in [
            "target_type",
            "tune_pairwise_type",
            "tune_boosting_type",
            "tune_tree_structure_type",
            "tune_loss_function",
            "top",
            "max_pairs",
        ]:
            if param in params:
                setattr(self, param, params.pop(param))

        if self.tune_pairwise_type and (self.tune_tree_structure_type or self.tune_boosting_type):
            module_logger.warning(
                "tune_pairwise_type=True is incompatible with tune_tree_structure_type=True or "
                "tune_boosting_type=True; disabling tune_pairwise_type."
            )
            self.tune_pairwise_type = False

        return super().set_params(**params)

    def __getstate__(self) -> dict:
        """
        Custom getstate to handle pickling of the model.
        This ensures all necessary attributes are included when the object is pickled.

        Returns:
            dict: A dictionary containing the state of the object.
        """
        state = super().__getstate__()
        state.update(
            {
                "target_type": self.target_type,
                "model_type": self.model_type,
                "tune_pairwise_type": self.tune_pairwise_type,
                "tune_boosting_type": self.tune_boosting_type,
                "tune_tree_structure_type": self.tune_tree_structure_type,
                "tune_loss_function": self.tune_loss_function,
                "top": self.top,
                "max_pairs": self.max_pairs,
            }
        )
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Custom setstate to handle unpickling of the model.
        This ensures all attributes are properly restored when the object is unpickled.

        Args:
            state (dict): A dictionary containing the state of the object.
        """
        self.target_type = state.pop("target_type", "single_target")
        self.model_type = state.pop("model_type", "ranking")
        state.pop("posterior_sampling", None)  # legacy field, no longer stored
        self.tune_pairwise_type = state.pop("tune_pairwise_type", False)
        self.tune_boosting_type = state.pop("tune_boosting_type", False)
        self.tune_tree_structure_type = state.pop("tune_tree_structure_type", True)
        self.tune_loss_function = state.pop("tune_loss_function", True)
        self.top = state.pop("top", 0)
        self.max_pairs = state.pop("max_pairs", None)
        super().__setstate__(state)

    def predict(
        self,
        X: pd.DataFrame,
        ntree_start: int = 0,
        ntree_end: int = 0,
        thread_count: int = -1,
        verbose: Optional[bool] = None,
        use_ranks: bool = False,
        normalize_by_group_size: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict scores or ranks for a single query group.

        When ``use_ranks`` is False (default) the method returns raw scores as produced
        by CatBoost.  When ``use_ranks`` is True the method returns 1-based integer ranks
        where rank 1 corresponds to the highest score (descending order).  The output
        order matches the input row order.

        For datasets containing multiple query groups, use the module-level helper
        :func:`ranker_predict_for_groups` which calls this method per group.

        Note on Ranking Convention
        ---------------------------
        CatBoost assigns higher scores to better-ranked samples, so rank 1 is given to
        the sample with the highest score (descending order). This matches the standard
        "rank 1 = best" interpretation used throughout the Mother framework.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Features for the samples (n_samples, n_features).
        ntree_start, ntree_end, thread_count, verbose : passed to CatBoost ``predict``.
        use_ranks : bool
            If True, return 1-based integer ranks (rank 1 = highest score).
        normalize_by_group_size : bool
            If True and ``use_ranks`` is True, divide ranks by ``len(X)``.
            Has no effect when ``use_ranks`` is False.
        **kwargs
            Additional keyword arguments (currently unused; accepted for API consistency).

        Returns
        -------
        np.ndarray
            If ``use_ranks`` is False: array of raw scores (floats) in the same order as ``X``.
            If ``use_ranks`` is True: array of integer ranks (1-based) corresponding to rows
            in ``X``, optionally normalised by group size.
        """
        preds = super().predict(
            X, ntree_start=ntree_start, ntree_end=ntree_end, thread_count=thread_count, verbose=verbose, **kwargs
        )

        if not use_ranks:
            return preds

        rank_values = scores_to_ranks(preds).astype(float)
        if normalize_by_group_size:
            return np.round(rank_values / len(X), 4)
        return rank_values.astype(int)

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        n_ensembles: int = 10,
        n_threads: int = 1,
        use_ranks: bool = False,
        uncertainty_for_opt: bool = False,
        normalize_by_group_size: bool = False,
        return_quantiles: bool = False,
    ) -> pd.DataFrame:
        """
        Estimate rank uncertainty using virtual ensembles.

        Uses CatBoost's ``virtual_ensembles_predict`` through the shared Mother utility.
        Operates on a single ranking group; for multi-group datasets use the module-level
        helper :func:`ranker_predict_uncertainty_for_groups`.

        Args:
            X : pd.DataFrame
                Feature matrix (n_samples, n_features) for a single ranking group.
            n_ensembles : int, optional
                Number of virtual ensembles used by CatBoost for uncertainty estimation.
                Must be >= 1.
            n_threads : int, optional
                Number of threads passed to ``virtual_ensembles_predict``.
            use_ranks : bool, optional
                If True, convert virtual-ensemble scores to per-ensemble ranks and
                compute ``mean_predictions`` as the mean of those per-ensemble ranks,
                with ``knowledge_uncertainty`` as their standard deviation.
                If False (default), quantities stay on the raw score scale.
            uncertainty_for_opt : bool, optional
                If True, return only ``knowledge_uncertainty`` for optimisation.
            normalize_by_group_size : bool, optional
                If True, divide ``knowledge_uncertainty`` and ``mean_predictions``
                by ``len(X)``.  Default is ``False``.
            return_quantiles : bool, optional
                If True, append score-quantile columns derived from
                ``DEFAULT_QUANTILES`` (defaults: ``score_q25``, ``score_q50``,
                ``score_q75``), computed across virtual ensembles on raw scores
                when ``use_ranks=False`` and on ensemble ranks when
                ``use_ranks=True``.
                Ignored when ``uncertainty_for_opt=True``.  Default is ``False``.

        Returns:
            pd.DataFrame with the same index as ``X`` and columns:
                - ``pred``: full-model prediction (raw score or 1-based rank).
                - ``mean_predictions``: mean of per-ensemble scores (``use_ranks=False``)
                  or mean of per-ensemble ranks (``use_ranks=True``).
                - ``knowledge_uncertainty``: std of per-ensemble scores or ranks;
                  optionally normalised by group size.
                - ``data_uncertainty``: ``None`` (not available for ranking).
                - ``total_uncertainty``: ``None`` (not available for ranking).
                                - Quantile columns from ``DEFAULT_QUANTILES`` *(only when*
                                    ``return_quantiles=True`` *; defaults: ``score_q25``,
                                    ``score_q50``, ``score_q75``)*: empirical quantiles of score
                                    or rank distributions.
        """
        if n_ensembles < 1:
            raise ValueError(f"n_ensembles must be >= 1, got {n_ensembles}.")

        result_index = X.index

        quantile_levels = sorted({float(q) for q in DEFAULT_QUANTILES})
        quantile_columns = [f"score_q{int(q * 100):02d}" for q in quantile_levels]

        if uncertainty_for_opt:
            result_df = pd.DataFrame(index=result_index, columns=["knowledge_uncertainty"])
        else:
            base_cols = ["pred", "mean_predictions", "knowledge_uncertainty", "data_uncertainty", "total_uncertainty"]
            if return_quantiles:
                base_cols += quantile_columns
            result_df = pd.DataFrame(index=result_index, columns=base_cols)

        uncertainty_df, raw_scores = cast(
            tuple[pd.DataFrame, np.ndarray],
            utils.get_virtual_prediction(
                X=X,
                model=cast(Any, self),
                virtual_ensembles_count=n_ensembles,
                thread_count=n_threads,
            ),
        )

        mean_scores = np.asarray(uncertainty_df["mean_predictions"], dtype=float)

        if use_ranks:
            raw_scores_arr = np.asarray(raw_scores, dtype=float)
            if raw_scores_arr.ndim != 2:
                raise ValueError(f"Expected 2D raw_scores, got {raw_scores_arr.ndim}D.")

            # Reuse the matrix helper, which internally applies scores_to_ranks per ensemble column.
            rank_ensembles = scores_matrix_to_ranks(raw_scores_arr)
            mean_predictions = rank_ensembles.mean(axis=1)
            ddof = 1 if rank_ensembles.shape[1] > 1 else 0
            knowledge_uncertainty = np.nan_to_num(rank_ensembles.std(axis=1, ddof=ddof), nan=0.0)
            quantile_source = rank_ensembles
        else:
            mean_predictions = mean_scores
            knowledge_uncertainty = np.asarray(uncertainty_df["knowledge_uncertainty"], dtype=float)
            quantile_source = np.asarray(raw_scores, dtype=float)

        if quantile_source.ndim != 2:
            raise ValueError(f"Expected 2D quantile_source, got {quantile_source.ndim}D.")

        if normalize_by_group_size:
            n = len(X)
            knowledge_uncertainty = np.round(knowledge_uncertainty / n, 4)
            mean_predictions = np.round(mean_predictions / n, 4)

        if uncertainty_for_opt:
            result_df["knowledge_uncertainty"] = knowledge_uncertainty
        else:
            result_df["pred"] = self.predict(
                X,
                use_ranks=use_ranks,
                normalize_by_group_size=normalize_by_group_size,
            )
            result_df["mean_predictions"] = mean_predictions
            result_df["knowledge_uncertainty"] = knowledge_uncertainty
            result_df["data_uncertainty"] = None
            result_df["total_uncertainty"] = None

            if return_quantiles:
                quantile_values = np.percentile(quantile_source, [q * 100 for q in quantile_levels], axis=1)
                for col_name, col_values in zip(quantile_columns, quantile_values):
                    result_df[col_name] = col_values

        # Keep numeric columns numeric after incremental assignment through loc.
        for col in result_df.columns:
            if col not in {"data_uncertainty", "total_uncertainty"}:
                result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

        return result_df

    def suggested_params_loss(
        self,
        trial: Trial,
        suggested_params: dict[str, Any],
        y: Union[pd.DataFrame, pd.Series],
        prefix: str,
    ) -> dict[str, Any]:
        """
        Add ranking-loss-specific hyperparameters to the suggested parameters.

        Builds a composite ``loss_function`` string from individually suggested
        building-block parameters (``base_loss``, ``mode``, ``dcg_denominator``,
        ``dcg_type``) and removes the building blocks afterwards so that only
        the final ``loss_function`` is passed to CatBoost.

        Pairwise losses (``YetiRankPairwise``, ``PairLogitPairwise``) are only
        included in the Optuna search space when ``tune_pairwise_type=True``
        **and** the current trial's ``grow_policy`` / ``boosting_type`` are
        compatible (``SymmetricTree`` + ``Plain``).  The ``__init__`` method
        already guarantees ``tune_pairwise_type=False`` when the tuning flags
        would conflict, so the runtime check here only inspects the actual
        trial values.

        Args:
            trial : optuna.trial.Trial
                Optuna trial object.
            suggested_params : dict[str, Any]
                Current suggested parameters (mutated in place).
            y : Union[pd.DataFrame, pd.Series]
                Target data.
            prefix : str
                Parameter prefix.

        Returns:
            dict[str, Any]: Updated suggested parameters.
        """
        if self.target_type == "multi_target":
            return suggested_params

        # Check if target is binary (only contains 0 and 1)
        is_binary: bool = bool(np.array_equal(np.unique(y), [0, 1]))

        # Check if the current trial's tree structure and boosting type are
        # compatible with Pairwise losses (require SymmetricTree + Plain).
        # Note: __init__ already guarantees tune_pairwise_type=False when
        # tune_tree_structure_type or tune_boosting_type are True, so we only
        # need to verify the actual values chosen for this trial.
        grow_policy: Optional[str] = suggested_params.get(prefix + "grow_policy")
        boosting_type: str = suggested_params.get(prefix + "boosting_type", "Plain")
        can_use_pairwise: bool = grow_policy == "SymmetricTree" and boosting_type == "Plain"

        # Build list of possible loss functions
        has_top: bool = self.top is not None and self.top > 0

        # When top is specified, only use YetiRank (top parameter only works with YetiRank)
        suggested_base_loss: str
        if has_top:
            if self.tune_pairwise_type and can_use_pairwise:
                suggested_base_loss = trial.suggest_categorical(prefix + "base_loss", ["YetiRank", "YetiRankPairwise"])
            else:
                suggested_base_loss = "YetiRank"
        else:
            base_losses: list[str] = ["YetiRank", "PairLogit"]

            # Add QuerySoftMax if the target y is binary, otherwise use QueryRMSE
            if is_binary:
                base_losses.append("QuerySoftMax")
            else:
                base_losses.append("QueryRMSE")

            # Only add Pairwise losses when pairwise tuning is enabled and the
            # current trial's tree/boosting settings are compatible
            if self.tune_pairwise_type and can_use_pairwise:
                base_losses.extend(["YetiRankPairwise", "PairLogitPairwise"])

            suggested_base_loss = trial.suggest_categorical(prefix + "base_loss", base_losses)

        if suggested_base_loss in ["YetiRank", "YetiRankPairwise"]:
            # Suggest mode (NDCG or Classic)
            suggested_mode: str
            if has_top:
                if is_binary:
                    suggested_mode = trial.suggest_categorical(prefix + "mode", ["NDCG", "MAP"])
                else:
                    suggested_mode = "NDCG"
            else:
                if is_binary:
                    suggested_mode = trial.suggest_categorical(prefix + "mode", ["NDCG", "Classic", "MAP"])
                else:
                    suggested_mode = trial.suggest_categorical(prefix + "mode", ["NDCG", "Classic"])

            # Build loss function string
            suggested_loss_function: str = f"{suggested_base_loss}:mode={suggested_mode}"

            # Add NDCG-specific parameters
            if suggested_mode == "NDCG":
                suggested_dcg_denominator: str = trial.suggest_categorical(
                    prefix + "dcg_denominator", ["LogPosition", "Position"]
                )
                suggested_loss_function += f";dcg_denominator={suggested_dcg_denominator}"

                suggested_dcg_type: str = trial.suggest_categorical(prefix + "dcg_type", ["Base", "Exp"])
                suggested_loss_function += f";dcg_type={suggested_dcg_type}"

            if suggested_mode != "Classic" and has_top:
                suggested_loss_function += f";top={self.top}"

            suggested_params[prefix + "loss_function"] = suggested_loss_function

        else:
            # Non-YetiRank losses (QueryRMSE, QuerySoftMax, PairLogit, PairLogitPairwise)
            loss_function: str = suggested_base_loss

            # Add max_pairs parameter for PairLogit losses if specified
            if "PairLogit" in suggested_base_loss and self.max_pairs is not None and self.max_pairs > 0:
                loss_function += f":max_pairs={self.max_pairs}"

            suggested_params[prefix + "loss_function"] = loss_function

        # Remove building block parameters — only loss_function should be passed to CatBoost
        # (similar to how Focal loss removes alpha/gamma after building the loss string)
        for param in ["base_loss", "mode", "dcg_denominator", "dcg_type"]:
            suggested_params.pop(prefix + param, None)

        return suggested_params

    def default_parameters(self, prefix: str = "") -> dict[str, Any]:
        """
        Returns the default recommended parameters for the CatBoostRanker.

        The returned dictionary includes Optuna building-block keys (``base_loss``,
        ``mode``, ``dcg_denominator``, ``dcg_type``) that are used by the tuning
        logic to compose the composite ``loss_function`` string.  These building-block
        keys are **not** passed directly to CatBoost — they are consumed and removed
        by :meth:`suggested_params_loss`.

        When ``top`` is set, the default mode is ``NDCG`` with
        ``dcg_denominator=Position`` and ``dcg_type=Base``; otherwise the default
        mode is ``Classic``.

        Args:
            prefix : str, optional
                Optional prefix for parameter names (default: ``""``).

        Returns:
            dict[str, Any]: Default parameters for the ranker.
        """
        defaults: dict[str, Any] = {
            prefix + "learning_rate": 0.03,
            prefix + "bootstrap_type": "MVS",
            prefix + "random_strength": 1,
            prefix + "grow_policy": "SymmetricTree",
            prefix + "boosting_type": "Plain",
            prefix + "max_depth": 6,
            prefix + "base_loss": "YetiRank",
        }

        if self.top is not None and self.top > 0:
            defaults[prefix + "mode"] = "NDCG"
            defaults[prefix + "dcg_denominator"] = "Position"
            defaults[prefix + "dcg_type"] = "Base"
        else:
            defaults[prefix + "mode"] = "Classic"

        return defaults


def ranker_predict_for_groups(
    model: "CatboostRankerMother",
    X: pd.DataFrame,
    group_id: np.ndarray,
    use_ranks: bool = False,
    normalize_by_group_size: bool = False,
) -> np.ndarray:
    """Predict scores or ranks for a dataset containing multiple ranking groups.

    Calls :meth:`CatboostRankerMother.predict` independently for each group so
    that when ``use_ranks=True`` ranks are assigned within each group rather than
    globally across the whole dataset.

    Parameters
    ----------
    model : CatboostRankerMother
        A fitted ranker.
    X : pd.DataFrame
        Feature matrix (n_samples, n_features).
    group_id : np.ndarray
        Group IDs aligned with ``X`` rows (one entry per row).
    use_ranks : bool
        Forwarded to :meth:`~CatboostRankerMother.predict`.
    normalize_by_group_size : bool
        Forwarded to :meth:`~CatboostRankerMother.predict`.

    Returns
    -------
    np.ndarray
        Predictions in the original row order of ``X``.
    """
    group_arr = _validate_ranking_group_id(group_id, len(X))
    result = np.empty(len(X), dtype=float)

    for idx in _iter_ranking_group_indices(group_arr):
        X_group = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
        result[idx] = model.predict(
            X_group,
            use_ranks=use_ranks,
            normalize_by_group_size=normalize_by_group_size,
        )

    return result


def ranker_predict_uncertainty_for_groups(
    model: "CatboostRankerMother",
    X: pd.DataFrame,
    group_id: np.ndarray,
    **kwargs,
) -> pd.DataFrame:
    """Estimate uncertainty for a dataset containing multiple ranking groups.

    Calls :meth:`CatboostRankerMother.predict_uncertainty` independently for
    each group so that rank conversion and normalisation are applied within
    group boundaries.

    Parameters
    ----------
    model : CatboostRankerMother
        A fitted ranker.
    X : pd.DataFrame
        Feature matrix (n_samples, n_features).
    group_id : np.ndarray
        Group IDs aligned with ``X`` rows.
    **kwargs
        Forwarded to :meth:`~CatboostRankerMother.predict_uncertainty`.

    Returns
    -------
    pd.DataFrame
        Concatenated uncertainty DataFrame in the original row order of ``X``,
        with the same columns as :meth:`~CatboostRankerMother.predict_uncertainty`.
    """
    group_arr = _validate_ranking_group_id(group_id, len(X))
    frames: list[pd.DataFrame] = []

    for idx in _iter_ranking_group_indices(group_arr):
        X_group = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
        frames.append(model.predict_uncertainty(X_group, **kwargs))

    return pd.concat(frames).loc[X.index]
