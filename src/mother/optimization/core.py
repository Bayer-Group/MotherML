import gc
import json
import logging
import typing
from functools import wraps

import numpy as np
import optuna
import optuna.logging
import pandas as pd
import sklearn
import sklearn.base as skl_base
import sklearn.metrics as skl_metrics
import sklearn.model_selection as skl_model_sel
from optuna.study import Study, StudyDirection
from optuna.terminator import TerminatorCallback, report_cross_validation_scores
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
    Decorator that routes ``ranking_groups`` / ``groups`` to the right arguments
    of ``cross_val_score`` depending on whether the call is a ranking call.

    **Non-ranking** (``ranking_groups`` is None)
        If sklearn's global ``enable_metadata_routing`` is *off* (default),
        CV-split groups are passed as a direct ``groups=`` kwarg to
        ``cross_val_score``.  If routing is *on* (set by a prior ranking call),
        sklearn 1.5+ rejects ``groups=`` as a direct kwarg, so the decorator
        redirects groups into ``fit_kwargs`` (forwarded as ``params=``) instead.

    **Ranking** (``ranking_groups`` is not None)
        ``group_id`` (query groups) must reach *both* the ranker's ``fit()``
        *and* the NDCG scorer's ``score()`` call.  sklearn only routes kwargs to
        the scorer when ``enable_metadata_routing=True`` is set globally.
        Therefore, for ranking:
        - both ``groups`` (CV splitter) and ``group_id`` (ranker + scorer) are
          injected into ``fit_kwargs`` and forwarded via ``params=``; and
        - ``groups=`` is *not* passed as a direct kwarg.
        The decorator raises ``RuntimeError`` early if routing is not enabled,
        rather than silently producing incorrect ranking metrics.

    The decorator also sets ``_groups_as_cross_val_args`` so ``optimize()``
    knows which path was taken.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        groups = kwargs.get("groups")
        ranking_groups = kwargs.get("ranking_groups")
        fit_kwargs = kwargs.get("fit_kwargs")

        # Shallow-copy so the decorator never mutates a caller-provided dict.
        # Injected keys (group_id, groups) must not leak into subsequent calls
        # that reuse the same fit_kwargs object.
        fit_kwargs = dict(fit_kwargs) if fit_kwargs is not None else {}
        kwargs["fit_kwargs"] = fit_kwargs

        routing_on: bool = sklearn.get_config()["enable_metadata_routing"]

        if ranking_groups is not None:
            # Guard: routing must be enabled or group_id will never reach the
            # NDCG scorer, producing silently wrong ranking metrics.
            if not routing_on:
                raise RuntimeError(
                    "Ranking optimization requires sklearn metadata routing to be enabled. "
                    "Call sklearn.set_config(enable_metadata_routing=True) before optimize()."
                )
            # Ranking: query groups must reach the ranker's fit() AND the
            # NDCG scorer's score() — both via params= with routing enabled.
            fit_kwargs["group_id"] = ranking_groups
            if groups is not None:
                fit_kwargs["groups"] = groups
            kwargs["_groups_as_cross_val_args"] = False
        else:
            if routing_on:
                # sklearn 1.5+ raises ValueError when groups= is passed directly
                # to cross_val_score while enable_metadata_routing=True.  Route
                # via params= (fit_kwargs) instead so the CV splitter picks it up.
                if groups is not None:
                    fit_kwargs["groups"] = groups
                kwargs["_groups_as_cross_val_args"] = False
            else:
                # Classic path: pass groups directly to cross_val_score.
                kwargs["_groups_as_cross_val_args"] = True

        return func(*args, **kwargs)

    return wrapper


class MotherTuner:
    """MotherTuner is a class that facilitates hyperparameter tuning using Optuna.

    Attributes:
        n_trials_optuna (int): Number of trials for Optuna optimization.
        n_startup_trials (int): Number of startup trials for Optuna.
        n_threads_optuna (int): Number of threads for Optuna optimization.
        early_stopping_optuna (bool): Flag to enable early stopping in Optuna.
        tuning_direction (StudyDirection or string): Direction of optimization (maximize or minimize).
        scorer (typing.Callable): Scoring function or string identifier for scoring.
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

    def __init__(
        self,
        scorer: typing.Union[typing.Callable, str],
        sampler: typing.Optional[optuna.samplers.BaseSampler] = None,
        early_stopping_optuna: bool = False,
        tuning_direction: typing.Union[StudyDirection, str] = StudyDirection.MAXIMIZE,
        n_trials_optuna: int = 100,
        n_threads_optuna: int = 1,
        n_startup_trials: int = 12,
        seed: int = 42,
        **kwargs,
    ):
        self.n_trials_optuna: int = n_trials_optuna
        self.n_startup_trials: int = n_startup_trials
        self.n_threads_optuna: int = n_threads_optuna
        self.early_stopping_optuna: bool = early_stopping_optuna
        self.tuning_direction: typing.Union[StudyDirection, str] = tuning_direction
        self.scorer: typing.Callable = skl_metrics.get_scorer(scorer)
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

        self.study: typing.Optional[Study] = None

    def get_callbacks(self, cross_validation: typing.Optional[skl_model_sel.BaseCrossValidator] = None):
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
                if cross_validation is not None and cross_validation.get_n_splits() < 2:
                    module_logger.warning(
                        "Optuna early termination requires at least 2 CV splits; disabling callback for hold-out setup"
                    )
                    return None
                callbacks = [TerminatorCallback()]
        return callbacks

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

        def objective(trial: optuna.trial.Trial) -> float:
            suggested_params_to_train: dict = get_hyper_space(trial=trial, X=X, y=y)
            module_logger.debug("Cloning pipeline")
            pipeline_cv: Pipeline = skl_base.clone(estimator)
            pipeline_cv.set_params(**suggested_params_to_train)

            cross_val_kwargs = {}
            if groups_as_cross_val_args:
                cross_val_kwargs["groups"] = groups

            module_logger.debug("Perform cross validation scoring")
            cv_score = skl_model_sel.cross_val_score(
                estimator=pipeline_cv,
                X=X,
                y=utils.y_toArray(y),
                cv=cross_validation,
                scoring=self.scorer,
                n_jobs=np.min([self.n_threads_optuna, cross_validation.get_n_splits()]),
                pre_dispatch=np.min([self.n_threads_optuna, cross_validation.get_n_splits()]),
                error_score="raise",  # only for debugging
                params=fit_kwargs,
                **cross_val_kwargs,
            )

            gc.collect()
            module_logger.info(f"Trial {trial.number}, cv score: {cv_score}")
            cv_score_not_na: np.ndarray = cv_score[~np.isnan(cv_score)]
            if len(cv_score_not_na) > 1:
                report_cross_validation_scores(trial, list(cv_score_not_na))
            mean_cv_score: float = cv_score_not_na.mean()
            return mean_cv_score

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
        self.study.optimize(
            objective,
            n_trials=self.n_trials_optuna,
            gc_after_trial=True,
            callbacks=self.get_callbacks(cross_validation=cross_validation),
        )

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
            # Remove the "groups" key that was injected by the decorator for
            # metadata-routing-based CV splits; it must not be forwarded to
            # the final pipeline.fit() call. Use pop-with-default so this is
            # safe even when groups was None and was therefore never injected.
            fit_kwargs.pop("groups", None)
        pipeline.fit(X, utils.y_toArray(y), **fit_kwargs)
        module_logger.info("Training completed")

        return pipeline
