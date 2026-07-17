import gc
import json
import logging
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps

import numpy as np
import optuna
import optuna.logging
import pandas as pd
import sklearn.base as skl_base
import sklearn.metrics as skl_metrics
import sklearn.model_selection as skl_model_sel
from optuna.study import Study, StudyDirection
from optuna.terminator import TerminatorCallback, report_cross_validation_scores
from sklearn import get_config as skl_get_config
from sklearn.pipeline import Pipeline

from mother import utils as mother_utils
from mother.optimization import utils

module_logger = logging.getLogger(__name__)
torch_available: bool = True

try:
    import torch  # noqa
except ImportError:
    torch_available = False


def handle_metadata_routing(func: typing.Callable) -> typing.Callable:
    """
    Decorator to handle metadata routing configuration for optimization methods.

    This decorator manages the routing of groups and ranking_groups parameters based on
    sklearn's metadata routing configuration. It automatically determines whether to pass
    groups as cross-validation arguments or as fit kwargs, and ensures ranking groups
    are properly routed when metadata routing is enabled.

    The decorator expects the wrapped function to have the following parameters:
    - groups: Optional[np.ndarray]
    - ranking_groups: Optional[np.ndarray]
    - fit_kwargs: Optional[dict]

    It will modify fit_kwargs in place and return a tuple (fit_kwargs, groups_as_cross_val_args)
    that can be used by the wrapped function.

    Parameters
    ----------
    func : Callable
        The function to be decorated (typically an optimize method)

    Returns
    -------
    Callable
        The wrapped function with metadata routing handled
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract relevant parameters
        groups = kwargs.get("groups")
        ranking_groups = kwargs.get("ranking_groups")
        fit_kwargs = kwargs.get("fit_kwargs")

        if fit_kwargs is None:
            fit_kwargs = {}
            kwargs["fit_kwargs"] = fit_kwargs

        # Check metadata routing configuration
        use_metadata_routing: bool = bool(skl_get_config().get("enable_metadata_routing", False))

        # Determine how to pass groups
        groups_as_cross_val_args: bool = True
        if use_metadata_routing and groups is not None:
            groups_as_cross_val_args = False
            fit_kwargs["groups"] = groups

        # Handle ranking groups
        if ranking_groups is not None:
            if not use_metadata_routing:
                raise AssertionError(
                    "To use ranking groups please enable metadata routing in sklearn. "
                    "Call sklearn.set_config(enable_metadata_routing=True)"
                )
            fit_kwargs["group_id"] = ranking_groups

        # Store the flag for use in the function
        kwargs["_groups_as_cross_val_args"] = groups_as_cross_val_args

        return func(*args, **kwargs)

    return wrapper


@dataclass
class ObjectiveContext:
    """
    This is a dataclass to pass all arguments of `def optimize()` to
    a customized `def objective()`.
    """

    get_hyper_space: typing.Callable
    estimator: Pipeline
    X: pd.DataFrame
    y: typing.Union[pd.DataFrame, pd.Series]
    cross_validation: skl_model_sel.BaseCrossValidator
    fit_kwargs: dict
    groups_as_cross_val_args: bool
    groups: typing.Optional[np.ndarray]
    extras: dict[str, typing.Any] = field(default_factory=dict)


class AbstractMotherTuner(ABC):
    def __init__(
        self,
        sampler: typing.Optional[optuna.samplers.BaseSampler] = None,
        early_stopping_optuna: bool = False,
        tuning_direction: typing.Union[StudyDirection, str] = StudyDirection.MAXIMIZE,
        n_trials_optuna: int = 100,
        n_threads_optuna: int = 1,
        n_startup_trials: int = 12,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.n_trials_optuna: int = n_trials_optuna
        self.n_startup_trials: int = n_startup_trials
        self.n_threads_optuna: int = n_threads_optuna
        self.early_stopping_optuna: bool = early_stopping_optuna
        self.tuning_direction: typing.Union[StudyDirection, str] = tuning_direction
        if sampler is None:
            module_logger.debug("Setting up default sampler TPE")
            self.sampler = optuna.samplers.TPESampler(
                multivariate=kwargs.get("multivariate", True),
                group=True,
                constant_liar=True,
                seed=seed,
                n_startup_trials=n_startup_trials,
            )
        else:
            self.sampler = sampler

        self.study: typing.Optional[Study] | None = None

    def get_callbacks(self):
        """
        Prepares and returns a list of callbacks for early stopping in Optuna optimization.

        If early stopping with Optuna is enabled and PyTorch is available, this method
        will return a list containing a TerminatorCallback instance. If PyTorch is not
        available, it will log a warning and return None.

        Returns:
            typing.Optional[typing.List[TerminatorCallback]]: A list of TerminatorCallback
            instances if early stopping is enabled and PyTorch is available, otherwise None.
        """
        callbacks: typing.Optional[typing.List[TerminatorCallback]] = None
        if self.early_stopping_optuna:
            if not torch_available:
                module_logger.warning("Torch not installed, early optuna termination will not be available")
                module_logger.warning(
                    """Torch is required for early termination of optuna optimization.
                    To enable early termination please install torch:
                    pip install mother[torch] or uv add mother[torch]"""
                )
            else:
                callbacks = [TerminatorCallback()]
        return callbacks

    @abstractmethod
    def objective(self, trial: optuna.trial.Trial, context: ObjectiveContext) -> float:
        """
        Placeholder to implement a customized objective function
        This function is the func argument of optuna.study.optimize
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
        """
        raise NotImplementedError

    @abstractmethod
    def call_optimize(self, context: ObjectiveContext) -> None:
        """
        Placeholder to implement a customized function to
        1. additional processing on data/model
        2. call the Study().optimize() function with self.objective()
        """

        raise NotImplementedError

    @handle_metadata_routing
    def optimize(
        self,
        estimator: Pipeline,
        X: pd.DataFrame,
        y: typing.Union[pd.DataFrame, pd.Series],
        cross_validation: skl_model_sel.BaseCrossValidator,
        hyperparameter_space_function: typing.Optional[typing.Callable] = None,
        default_parameters: typing.Optional[dict] = None,
        groups: typing.Optional[np.ndarray] = None,
        ranking_groups: typing.Optional[np.ndarray] = None,
        fit_kwargs: typing.Optional[dict] = None,
        _groups_as_cross_val_args: bool = True,
        **kwargs,
    ) -> Pipeline:
        """
        Takes an estimator as input and optimizes the hyperparameters according to
        passed hyperparameter suggestion functions. Then returns a tuned and fitted
        object.

        Parameters
        ----------
        X: data to be used for training the model
        y: the target values used for training
        hyperparameter_space_function: an optuna compatible function that generates a hyperparameter dict
            defaults to None. If None, the function will be derived from the estimator
        default_parameters: a dict that defines default parameters that are evaluated before tuning
            If None, the function will be derived from the estimator.
        groups: groups used to generate grouped cross validation splits
        ranking_groups: groups, passed as "group_id" to the estimator, used for ranking models
        fit_kwargs: additional keyword arguments passed to the fit method of the estimator
        _groups_as_cross_val_args: internal parameter set by the decorator to control how groups are passed

        Returns
        -------
        a fitted pipeline object

        """
        get_hyper_space: typing.Callable = mother_utils.get_hyperparameter_space_function(
            func=hyperparameter_space_function, estimator=estimator
        )

        # The decorator ensures fit_kwargs is never None
        assert fit_kwargs is not None
        groups_as_cross_val_args = _groups_as_cross_val_args

        # set ObjectiveContext
        obj_context = ObjectiveContext(
            get_hyper_space=get_hyper_space,
            estimator=estimator,
            X=X,
            y=y,
            cross_validation=cross_validation,
            fit_kwargs=fit_kwargs,
            groups_as_cross_val_args=groups_as_cross_val_args,
            groups=groups,
            extras=kwargs,
        )

        module_logger.info(
            "Setting up Optuna to optimize hyperparameters with direction: %s",
            self.tuning_direction,
        )
        optuna.logging.set_verbosity(module_logger.getEffectiveLevel())
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

        self.study = optuna.create_study(sampler=self.sampler, direction=self.tuning_direction)

        module_logger.info("Running hyperparameter optimization with %d trials", self.n_trials_optuna)

        default_parameters = mother_utils.get_default_parameters(
            params=default_parameters,
            estimator=estimator,
        )
        if default_parameters != {}:
            module_logger.debug(
                "Enqueuing default parameters as first trial: \n %s",
                json.dumps(default_parameters, indent=4),
            )
            self.study.enqueue_trial(default_parameters)

        # call optuna study optimize
        self.call_optimize(obj_context)

        if default_parameters != {}:
            module_logger.info("Check if the default parameters have been evaluated in the study")
            evaluated = self.study.trials[0].params
            enqueued = default_parameters
            assert all(evaluated[k] == enqueued[k] for k in evaluated.keys() & enqueued.keys())

        module_logger.info("Hyperparameter optimization completed, getting best parameters from frozen trial")
        module_logger.info("Best trial number: %d", self.study.best_trial.number)
        best_parameters: dict[str, typing.Any] = get_hyper_space(trial=self.study.best_trial, X=X, y=y)

        # make a clone to keep an untuned original version that can be tuned again
        pipeline = skl_base.clone(estimator)
        pipeline.set_params(**best_parameters)
        module_logger.info("Setting optuna optimization result")
        utils.dump_best_trial_parameters(best_parameters)

        for parameter, value in best_parameters.items():
            module_logger.info(f"{parameter: >25}: {value}")

        module_logger.info("Running training")
        if not groups_as_cross_val_args:
            fit_kwargs.pop("groups")
        pipeline.fit(X, utils.y_toArray(y), **fit_kwargs)
        module_logger.info("Training completed")

        return pipeline


class MotherTuner(AbstractMotherTuner):
    """MotherTuner is a class that facilitates hyperparameter tuning using Optuna.

    Attributes:
        scorer (typing.Callable): Scoring function or string identifier for scoring.
        n_trials_optuna (int): Number of trials for Optuna optimization.
        n_startup_trials (int): Number of startup trials for Optuna.
        n_threads_optuna (int): Number of threads for Optuna optimization.
        early_stopping_optuna (bool): Flag to enable early stopping in Optuna.
        tuning_direction (StudyDirection or string): Direction of optimization (maximize or minimize).
        sampler (optuna.samplers.BaseSampler): Sampler for Optuna trials.
        study (typing.Optional[Study]): Optuna study object.
        **kwargs (Any): additional arguments for the scorer

    Methods:
        __init__(self, scorer, sampler=None, early_stopping_optuna=False, tuning_direction=StudyDirection.MAXIMIZE,
            n_trials_optuna=100, n_threads_optuna=1, n_startup_trials=12, seed=42):
            Initializes the MotherTuner with the given parameters.

        get_callbacks(self):

        optimize(self, estimator, X, y, cross_validation, groups=None, direction="maximize")
             -> Pipeline:
    """

    def __init__(self, scorer: typing.Union[typing.Callable, str], **kwargs):

        self.scorer: typing.Callable = skl_metrics.get_scorer(scorer)
        super().__init__(**kwargs)

    def objective(self, trial: optuna.trial.Trial, context: ObjectiveContext) -> float:
        suggested_params_to_train: dict = context.get_hyper_space(trial=trial, X=context.X, y=context.y)
        module_logger.debug("Cloning pipeline")
        pipeline_cv: Pipeline = skl_base.clone(context.estimator)
        pipeline_cv.set_params(**suggested_params_to_train)

        cross_val_kwargs = {}
        if context.groups_as_cross_val_args:
            cross_val_kwargs["groups"] = context.groups

        module_logger.debug("Perform cross validation scoring")
        cv_score = skl_model_sel.cross_val_score(
            estimator=pipeline_cv,
            X=context.X,
            y=utils.y_toArray(context.y),
            cv=context.cross_validation,
            scoring=self.scorer,
            n_jobs=np.min([self.n_threads_optuna, context.cross_validation.get_n_splits()]),
            pre_dispatch=np.min([self.n_threads_optuna, context.cross_validation.get_n_splits()]),
            error_score="raise",  # only for debugging
            params=context.fit_kwargs,
            **cross_val_kwargs,
        )

        gc.collect()
        module_logger.info(f"Trial {trial.number}, cv score: {cv_score}")
        cv_score_not_na: np.ndarray = cv_score[~np.isnan(cv_score)]
        report_cross_validation_scores(trial, list(cv_score_not_na))
        mean_cv_score: float = cv_score_not_na.mean()
        return mean_cv_score

    def call_optimize(self, context: ObjectiveContext) -> None:
        """Call study.optimize() funtion.

        Can be customised in the case of data processing needed for cross validation / optimisation

        Args:
            context (ObjectiveContext): optimize() function calls this function
            with a ObjectiveContext object containing all necessary arguments.
                self.call_optimize(obj_context)
        """
        self.study.optimize(
            lambda trial: self.objective(trial, context=context),
            n_trials=self.n_trials_optuna,
            gc_after_trial=True,
            callbacks=self.get_callbacks(),
        )
