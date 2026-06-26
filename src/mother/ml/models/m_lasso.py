import logging
from typing import Literal, Mapping, Optional, Union

import numpy as np
import pandas as pd
from optuna.trial import Trial
from sklearn.linear_model import (
    ARDRegression,
    Lasso,
    LogisticRegression,
    MultiTaskLasso,
)

from mother.ml.core import AbstractMotherPipeline
from mother.ml.models import utils

module_logger: logging.Logger = logging.getLogger(__name__)


class LassoRegressorMother(Lasso, AbstractMotherPipeline):
    """
    MOTHER class for a LASSO regression including hyperparameter optimization
    """

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Define the hyperparameter search space for Lasso regression.

        Parameters:
            X: array-like
                Feature matrix.
            y: array-like
                Target vector.
            trial: optuna.trial.Trial
                Optuna trial object for suggesting hyperparameters.
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing hyperparameter names and their suggested values.
        """
        return utils.add_prefix_to_dict_keys(
            {"alpha": trial.suggest_float(prefix + "alpha", 1e-6, 1e1, log=True)},
            prefix=prefix,
        )

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Return the default hyperparameters for the Lasso model.

        Parameters:
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing default hyperparameter values.
        """
        return utils.add_prefix_to_dict_keys({"alpha": 1e-3}, prefix=prefix)

    def set_params(self, **params):
        """
        Set the parameters of the Lasso model.

        Parameters:
        **params: Keyword arguments for the parameters to set.
        """
        return super().set_params(**params)

    def get_params(self, deep=True) -> dict:
        return super().get_params(deep=deep)


class LassoClassifierBinaryMother(LogisticRegression, AbstractMotherPipeline):
    """
    MOTHER class for a LASSO classification including hyperparameter optimization
    """

    def __init__(
        self,
        penalty: Literal["l1"] = "l1",  # Lasso uses L1 penalty
        *,
        dual: bool = False,
        tol: float = 0.0001,
        C: float = 1,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Optional[Union[Mapping, str]] = "balanced",
        random_state: int = 42,
        solver: str = "liblinear",  # 'liblinear' can be used for L1 penalty and is less complex
        max_iter: int = 3000,
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__(
            penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,  # type: ignore
            max_iter=max_iter,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
        )

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Define the hyperparameter search space for Lasso classification.

        Parameters:
            X: array-like
                Feature matrix.
            y: array-like
                Target vector.
            trial: optuna.trial.Trial
                Optuna trial object for suggesting hyperparameters.
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing hyperparameter names and their suggested values.
        """
        return utils.add_prefix_to_dict_keys(
            {
                "C": trial.suggest_float(prefix + "C", 1e-6, 1e1, log=True),
            },
            prefix=prefix,
        )

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Return the default hyperparameters for the Lasso model.

        Parameters:
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing default hyperparameter values.
        """
        return utils.add_prefix_to_dict_keys({"C": 1e0}, prefix=prefix)

    def set_params(self, **params):
        """
        Set the parameters of the Lasso model.

        Parameters:
        **params: Keyword arguments for the parameters to set.
        """
        return super().set_params(**params)

    def get_params(self, deep=True) -> dict:
        return super().get_params(deep=deep)


class LassoClassifierMulticlassMother(LassoClassifierBinaryMother):
    """
    MOTHER class for a LASSO classification with multiclass support.
    Inherits from LassoClassifierBinaryMother.
    """

    def __init__(self, **kwargs):
        module_logger.warning(
            """LassoClassifierMother selected. 'Saga' is used as solver.
            Scale input features beforehand to improve convergence."""
        )
        if "solver" in kwargs:
            module_logger.warning(
                "LassoClassifierMulticlassMother selected. 'Saga' is used as solver for multiclass problems."
            )
            # Use 'saga' solver for multiclass support
            # 'saga' supports L1 penalty and is suitable for large datasets
        kwargs["solver"] = "saga"
        super().__init__(**kwargs)


class MultiTaskLassoMother(MultiTaskLasso, AbstractMotherPipeline):
    """
    MOTHER class for a MultiTask LASSO regression including hyperparameter optimization.

    Wraps sklearn.linear_model.MultiTaskLasso to fit multiple regression targets
    simultaneously with a shared L1 sparsity pattern. Features are selected or
    discarded jointly across all targets, which is appropriate when targets share
    a common set of relevant features.

    Use this model when y has shape (n_samples, n_targets). For single-target
    regression use LassoRegressorMother instead.
    """

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Define the hyperparameter search space for MultiTask Lasso regression.

        Parameters:
            X: array-like
                Feature matrix.
            y: array-like
                Target matrix (n_samples, n_targets).
            trial: optuna.trial.Trial
                Optuna trial object for suggesting hyperparameters.
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing hyperparameter names and their suggested values.
        """
        return utils.add_prefix_to_dict_keys(
            {"alpha": trial.suggest_float(prefix + "alpha", 1e-6, 1e1, log=True)},
            prefix=prefix,
        )

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Return the default hyperparameters for the MultiTask Lasso model.

        Parameters:
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing default hyperparameter values.
        """
        return utils.add_prefix_to_dict_keys({"alpha": 1e-3}, prefix=prefix)

    def set_params(self, **params):
        return super().set_params(**params)

    def get_params(self, deep=True) -> dict:
        return super().get_params(deep=deep)


class ARDRegressionMother(ARDRegression, AbstractMotherPipeline):
    """
    MOTHER class for ARD (Automatic Relevance Determination) regression including
    hyperparameter optimization and native uncertainty estimation.

    ARDRegression is a Bayesian linear model that places individual sparsity-inducing
    priors over each feature weight. It produces a posterior distribution over the weights,
    yielding calibrated predictive uncertainty (posterior predictive standard deviation)
    without requiring conformal post-processing or ensembling.

    The regularisation strength is inferred per-feature from the data, making this
    model effective on high-dimensional datasets where only a subset of features are
    relevant.

    predict_uncertainty() follows the standard Mother uncertainty interface and returns
    the posterior predictive standard deviation as knowledge_uncertainty.
    """

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Define the hyperparameter search space for ARD regression.

        The Bayesian priors are parameterised by four hyperparameters that control the
        shape of the Gamma prior over the noise precision (alpha) and the weight
        precision (lambda). Tuning these allows the model to adapt the degree of
        sparsity and noise tolerance to the dataset.

        Parameters:
            X: array-like
                Feature matrix.
            y: array-like
                Target vector.
            trial: optuna.trial.Trial
                Optuna trial object for suggesting hyperparameters.
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing hyperparameter names and their suggested values.
        """
        return utils.add_prefix_to_dict_keys(
            {
                "alpha_1": trial.suggest_float(prefix + "alpha_1", 1e-7, 1e-4, log=True),
                "alpha_2": trial.suggest_float(prefix + "alpha_2", 1e-7, 1e-4, log=True),
                "lambda_1": trial.suggest_float(prefix + "lambda_1", 1e-7, 1e-4, log=True),
                "lambda_2": trial.suggest_float(prefix + "lambda_2", 1e-7, 1e-4, log=True),
            },
            prefix=prefix,
        )

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Return the default hyperparameters for the ARD regression model.

        Parameters:
            prefix: str, optional
                Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary containing default hyperparameter values.
        """
        return utils.add_prefix_to_dict_keys(
            {
                "alpha_1": 1e-6,
                "alpha_2": 1e-6,
                "lambda_1": 1e-6,
                "lambda_2": 1e-6,
            },
            prefix=prefix,
        )

    def set_params(self, **params):
        return super().set_params(**params)

    def get_params(self, deep=True) -> dict:
        return super().get_params(deep=deep)

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        uncertainty_for_opt: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict target values and estimate uncertainty using the ARD posterior.

        ARDRegression natively provides a predictive standard deviation via
        return_std=True, which reflects the posterior predictive uncertainty over
        the weights (epistemic / knowledge uncertainty). No ensembling or
        post-hoc calibration is required.

        Parameters:
            X: pd.DataFrame
                Input features.
            uncertainty_for_opt: bool, optional
                If True, return only the knowledge_uncertainty column for use
                in uncertainty-aware optimisation workflows.

        Returns:
            pd.DataFrame with columns:
                - 'pred': point predictions (posterior mean)
                - 'mean_predictions': posterior mean (same as pred)
                - 'knowledge_uncertainty': posterior predictive standard deviation
                - 'data_uncertainty': None (not available from ARD)
                - 'total_uncertainty': same as knowledge_uncertainty
        """
        mean_preds, std_preds = super().predict(X, return_std=True)

        index = X.index if isinstance(X, pd.DataFrame) else None

        result = pd.DataFrame(
            {
                "pred": mean_preds,
                "mean_predictions": mean_preds,
                "knowledge_uncertainty": std_preds,
                "data_uncertainty": np.nan,
                "total_uncertainty": std_preds,
            },
            index=index,
        )

        if uncertainty_for_opt:
            return pd.DataFrame({"knowledge_uncertainty": result["knowledge_uncertainty"]}, index=index)

        return result
