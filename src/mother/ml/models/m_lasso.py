import logging
from typing import Mapping, Optional, Union

from optuna.trial import Trial
from sklearn.linear_model import Lasso, LogisticRegression

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
        *,
        dual: bool = False,
        tol: float = 0.0001,
        C: float = 1,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Optional[Union[Mapping, str]] = "balanced",
        random_state: int = 42,
        solver: str = "liblinear",  # liblinear supports pure L1 (l1_ratio=1)
        max_iter: int = 3000,
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        # l1_ratio=1: pure L1 regularisation via the sklearn ≥1.8 API.
        # penalty='l1' was deprecated in sklearn 1.8 and will be removed in 1.10;
        # l1_ratio=1 is the replacement (analogous to LogisticRegressionCV).
        super().__init__(
            l1_ratio=1,
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
