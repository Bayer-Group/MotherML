import logging
import typing

import catboost
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.base as skl_base
import sklearn.model_selection as skl_model_sel
from boruta import BorutaPy
from optuna.trial import Trial
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel, SelectorMixin
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import robust_scale
from sklearn.utils.validation import check_is_fitted

from mother import ml, utils
from mother.ml.models.m_catboost import CatBoostClassifier, CatBoostRegressor
from mother.ml.utils import get_tree_depth, signed_percentiles_independent

module_logger = logging.getLogger(__name__)


class _AbstractImportanceClass:
    pass


class MotherPermutationImportance(BaseEstimator, _AbstractImportanceClass):
    """
    A custom estimator for computing feature importances using permutation importance with cross-validation.

    This class calculates feature importances by permuting the values of each feature and measuring the
    impact on the model's performance. It supports cross-validation to ensure robust importance estimates
    and can optionally scale the importances into percentiles to allow for usability with MotherSelectFromModel.

    Parameters
    ----------
    estimator : typing.Union[CatBoostClassifier, CatBoostRegressor]
        The base estimator used to compute feature importances. Must support the `get_feature_importance` method.
    cv : typing.Optional[skl_model_sel.BaseCrossValidator], default=None
        The cross-validation strategy to use. If None, a default `RepeatedKFold` with 3 splits and 3 repeats
        is used.
    cols : typing.Optional[typing.Union[str, typing.List[str]]], default=None
        Column names or indices to use for feature selection. If None, all columns are used.
    percentiles : bool, default=True
        Whether to scale the computed feature importances into percentiles.
    n_estimators : typing.Optional[int], default=None
        The number of estimators to use in the CatBoost model.
    random_state: int, default=42
        The random seed to use for the multitarget permutation importance
    max_depth : int, default=None
        The maximum depth of the trees in the CatBoost model. If None, it is determined from the estimator.
        Please keep in mind that the CatBoost model will be trained with this depth. Also please keep in mind that
        for CatBoost only either max_depth or depth can be set, not both. In mother the max_depth is used.

    Attributes
    ----------
    feature_importances_ : typing.Optional[np.ndarray]
        The computed feature importances after fitting. Initialized as None.
    n_features_in_ : typing.Optional[int]
        The number of features seen during fit. Initialized as None.

    Methods
    -------
    _more_tags() -> typing.Dict[str, bool]
        Returns tags used for scikit-learn data validation.
    fit(X: typing.Union[typing.Iterable, pd.DataFrame], y: typing.Union[typing.Iterable, pd.DataFrame]) -> self
        Computes the `feature_importances_` attribute and optionally fits the base estimator.
    _cv_scores_importances(X: typing.Union[typing.Iterable, pd.DataFrame],
     y: typing.Union[typing.Iterable, pd.DataFrame])
        -> np.ndarray
        Internal method to compute feature importances using cross-validation and the default cross
        validation

    Example
    -------
    >>> from sklearn.datasets import make_classification
    >>> from catboost import CatBoostClassifier
    >>> from mother.ml.estimators import MotherPermutationImportance
    >>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    >>> estimator = CatBoostClassifier(iterations=10, verbose=0)
    >>> mpi = MotherPermutationImportance(estimator=estimator)
    >>> mpi.fit(X, y)
    >>> print(mpi.feature_importances_)
    """

    def __init__(
        self,
        estimator: typing.Union[CatBoostClassifier, CatBoostRegressor],
        cv: typing.Optional[skl_model_sel.BaseCrossValidator] = None,
        cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
        percentiles: bool = True,
        n_estimators: typing.Optional[int] = None,
        random_state: typing.Union[int, np.random.RandomState] = 42,
        max_depth: typing.Optional[int] = None,
    ):
        self.estimator: typing.Union[CatBoostClassifier, CatBoostRegressor] = estimator
        self.cv: skl_model_sel.BaseCrossValidator
        self.cols: typing.Optional[typing.Union[str, typing.List[str]]] = cols
        self.percentiles = percentiles
        self.n_estimators: typing.Optional[int] = n_estimators
        self.estimator: typing.Union[CatBoostClassifier, CatBoostRegressor] = estimator
        self.random_state: typing.Union[int, np.random.RandomState] = random_state
        if max_depth is None:
            self.max_depth: int = get_tree_depth(estimator)
        else:
            self.max_depth: int = max_depth
            self.estimator.set_params(max_depth=self.max_depth)

        if cv is None:
            module_logger.warning(
                "No cv strategy was passed, using the RepeatedKfold (n_repeats=3, n_splits=3, random_state=42)"
            )
            self.cv = RepeatedKFold(n_repeats=3, n_splits=3, random_state=self.random_state)
        else:
            self.cv = cv
        self.feature_importances_: typing.Optional[np.ndarray] = None
        self.n_features_in_: typing.Optional[int] = None

    def _more_tags(self) -> typing.Dict[str, bool]:
        """Tags used for scikit-learn data validation."""
        return {"allow_nan": True, "no_validation": True}

    def fit(self, X: typing.Union[typing.Iterable, pd.DataFrame], y: typing.Union[typing.Iterable, pd.DataFrame]):
        """Compute ``feature_importances_`` attribute and optionally
        fit the base estimator. If the number of threads is not explicitely
        defined in the base estimator only 1 core will be used.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).


        Returns
        -------
        self : object
            Returns self.
        """
        # X = check_array(X, force_all_finite="allow-nan")
        input_features: npt.NDArray = utils.convert_input(X, col=self.cols)
        input_target: npt.NDArray = utils.convert_input(y, col=self.cols)
        self.feature_importances_ = self._cv_scores_importances(input_features, input_target)
        self.n_features_in_ = input_features.shape[1] if np.ndim(input_features) > 1 else 1

        module_logger.info("Initialized permutation feature importance estimator")

        return self

    def _cv_scores_importances(
        self, X: typing.Union[typing.Iterable, pd.DataFrame], y: typing.Union[typing.Iterable, pd.DataFrame]
    ):
        assert self.cv is not None
        thread_count: typing.Optional[int] = self.estimator.get_params().get("thread_count")
        if thread_count is None or thread_count == 0:
            thread_count = 1
        cv: skl_model_sel.BaseCrossValidator = skl_model_sel.check_cv(self.cv, y)
        feature_importances_list: list = []
        module_logger.info("Start permutation feature importance calculation for selection")
        module_logger.debug(f"The number of cores used for feature importance calculation is {str(thread_count)}")
        for train, test in cv.split(X, y):
            est: typing.Union[CatBoostClassifier, CatBoostRegressor] = skl_base.clone(self.estimator)
            if self.n_estimators is not None:
                module_logger.info("User defined number of estimators was passed so it will be used for the model")
                est: typing.Union[CatBoostClassifier, CatBoostRegressor] = est.set_params(iterations=self.n_estimators)
            est: typing.Union[CatBoostClassifier, CatBoostRegressor] = est.fit(X[train], y[train])

            if y.ndim == 1:
                _importances = est.get_feature_importance(
                    data=catboost.Pool(X[test], y[test]),
                    type=catboost.EFstrType.LossFunctionChange,
                    thread_count=thread_count,
                )
            else:
                _importances_list_for_repeats: typing.List[np.ndarray] = [
                    permutation_importance(
                        est,
                        X[test],
                        y[test],
                        random_state=random_state,
                        max_samples=np.min([1000, len(test)]),
                        n_jobs=thread_count,
                        n_repeats=1,
                    ).importances_mean
                    for random_state in range(self.random_state, self.random_state + 6)
                ]
                _importances = np.nanmean(_importances_list_for_repeats, axis=0)
            feature_importances_list.append(np.array(_importances))

        feature_importances: np.ndarray = np.nanmean(feature_importances_list, axis=0)

        if self.percentiles:
            feature_importances = signed_percentiles_independent(feature_importances)
            module_logger.info("Feature importances have been turned into percentiles")

        module_logger.info("Finished permutation feature importance calculation for selection")
        return feature_importances


class MotherCatboostImportance(BaseEstimator, _AbstractImportanceClass):
    """
    MotherCatboostImportance is a custom estimator that computes feature importances using CatBoost models.

    estimator : typing.Union[CatBoostClassifier, CatBoostRegressor]
        The CatBoost estimator to be used for computing feature importances.
        The importances ave to be percentiles, so for the estimator set percentiles=True
    n_estimators : int, default=1000
        The number of estimators to use in the CatBoost model.
    random_state: int, default=42
        Used when training the provided CatBoost model
    max_depth : int, default=None
        The maximum depth of the trees in the CatBoost model. If None, it is determined from the estimator.
        Please keep in mind that the CatBoost model will be trained with this depth. Also please keep in mind that
        for CatBoost only either max_depth or depth can be set, not both. In mother the max_depth is used.

    Attributes
    feature_importances_ : typing.Optional[np.ndarray]
        The computed feature importances. Initialized as None.

    n_features_in_ : typing.Optional[int]
        The number of features seen during fit. Initialized as None.

    Methods
    _more_tags()
        Returns tags used for scikit-learn data validation.

    fit(X: typing.Union[typing.Iterable, pd.DataFrame], y: typing.Union[typing.Iterable, pd.DataFrame])
        Compute `feature_importances_` attribute and optionally fit the base estimator.

    _catboost_importances(X: np.ndarray, y: np.ndarray)
        Internal method to compute feature importances using CatBoost.
    """

    def __init__(
        self,
        estimator: typing.Union[CatBoostClassifier, CatBoostRegressor],
        percentiles: bool = True,
        scale: bool = False,
        n_estimators: typing.Optional[int] = None,
        random_state: typing.Union[int, np.random.RandomState] = 42,
        max_depth: typing.Optional[int] = None,
    ):
        self.estimator = estimator
        self.feature_importances_: typing.Optional[np.ndarray] = None
        self.n_features_in_: typing.Optional[int] = None
        self.cols: typing.Optional[typing.List[str]] = None
        self.percentiles: bool = percentiles
        self.scale: bool = scale
        self.n_estimators: typing.Optional[int] = n_estimators
        self.estimator: typing.Union[CatBoostClassifier, CatBoostRegressor] = estimator
        self.random_state: typing.Union[int, np.random.RandomState] = random_state
        if max_depth is None:
            self.max_depth: int = get_tree_depth(estimator)
        else:
            self.max_depth: int = max_depth
            self.estimator.set_params(max_depth=self.max_depth)

    def _more_tags(self) -> typing.Dict[str, bool]:
        """Tags used for scikit-learn data validation."""
        return {"allow_nan": True, "no_validation": True}

    def fit(self, X: typing.Union[typing.Iterable, pd.DataFrame], y: typing.Union[typing.Iterable, pd.DataFrame]):
        #
        """Compute ``feature_importances_`` attribute and optionally
        fit the base estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        Returns
        -------
        self : object
            Returns self.
        """

        input_features: npt.NDArray = utils.convert_input(X, col=self.cols)
        input_target: npt.NDArray = utils.convert_input(y, col=self.cols)
        results: np.ndarray = self._catboost_importances(input_features, input_target)

        self.feature_importances_ = results
        self.n_features_in_ = input_features.shape[1] if np.ndim(input_features) > 1 else 1

        module_logger.info("Initialized catboost feature importance estimator")

        return self

    def _catboost_importances(
        self, X: typing.Union[typing.Iterable, pd.DataFrame], y: typing.Union[typing.Iterable, pd.DataFrame]
    ):
        thread_count: typing.Optional[int] = self.estimator.get_params().get("thread_count")
        if thread_count is None or thread_count == 0:
            thread_count = 1

        est = skl_base.clone(self.estimator)

        # Set random_seed (CatBoost's preferred parameter) using our random_state value
        if isinstance(self.random_state, np.random.RandomState):
            random_seed = self.random_state.randint(1, 42)
        else:
            random_seed = self.random_state
        est = est.set_params(random_seed=random_seed)

        if self.n_estimators is not None:
            module_logger.info("User defined number of estimators was passed so it will be used for the model")
            est: typing.Union[CatBoostClassifier, CatBoostRegressor] = est.set_params(iterations=self.n_estimators)

        est: typing.Union[CatBoostClassifier, CatBoostRegressor] = est.fit(X, y)

        module_logger.debug(f"The number of cores used for feature importance calculation is {str(thread_count)}")
        _importances: typing.List[float] = est.get_feature_importance(
            type=catboost.EFstrType.PredictionValuesChange,
            thread_count=thread_count,
        )
        module_logger.info("Finished catboost feature importance calculation")

        feature_importances: np.ndarray = np.array(_importances)

        if self.scale:
            feature_importances = robust_scale(feature_importances)
            module_logger.info("Feature importances have been scaled (robust_scale)")

        if self.percentiles:
            feature_importances = signed_percentiles_independent(feature_importances)
            module_logger.info("Feature importances have been turned into percentiles")

        return feature_importances


class MotherSelectFromModel(SelectFromModel, ml.AbstractMotherPipeline):
    """
    This class inherits from the SelectFromModel class in the scikit-learn library and extends it by adding
    hyperparameter optimization capabilities using Optuna. The parameter tuning assumes that the feature importances
    from MotherCatboostImportance or MotherPermutationImportance have been turned into percentiles.
    """

    def __init__(
        self,
        estimator: typing.Union[MotherCatboostImportance, MotherPermutationImportance],
        threshold=np.finfo(np.float64).tiny,
        max_features: typing.Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize MotherSelectFromModel.

        Parameters
        ----------
        estimator : MotherCatboostImportance or MotherPermutationImportance
            The base estimator from which the feature importances are obtained.
        threshold : str, float, or None, optional
            The threshold value to use for feature selection. Passed to the superclass.
        max_features : int, optional
            The maximum number of features to select. Passed to the superclass.
        kwargs : dict
            Additional arguments passed to the superclass.
        """
        # Pass the threshold and other arguments to the superclass
        super().__init__(estimator=estimator, threshold=threshold, max_features=max_features, **kwargs)

    def default_parameters(self, prefix: str = "") -> dict:
        return {
            prefix + "threshold": np.finfo(np.float64).tiny,
        }

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        The range here assumes that the feature impotances have benn turned into percentiles.
        """
        suggested_params = {prefix + "threshold": trial.suggest_float(prefix + "threshold", 0, 1, step=0.25)}

        return suggested_params


class MotherBorutaPy(SelectorMixin, BorutaPy):
    """
    A DataFrame-compatible extension of BorutaPy with scikit-learn set_output support.

    BorutaPy is a powerful feature selection method, but its original implementation
    does not natively support the scikit-learn set_output API,
    which are increasingly standard in modern machine learning workflows. This subclass
    addresses these limitations by:

    - Automatically detecting pandas DataFrame inputs and preserving column names and indices
        in the output, making feature selection results easier to interpret and integrate into
        pandas-based pipelines.
    - Implementing the set_output method from scikit-learn, allowing users to control the
        output format ('default' for numpy arrays or 'pandas' for DataFrames) of transform and
        fit_transform methods, thus improving compatibility with scikit-learn's ecosystem.
    - Ensuring seamless conversion between numpy arrays and DataFrames as needed, so that
        downstream code can rely on consistent and expected data types.

    This wrapper is especially useful for users who work with pandas DataFrames and require
    feature selection results that retain DataFrame structure, or who wish to leverage
    scikit-learn's output configuration for better pipeline integration.

        Parameters
    ----------
    estimator : object
        A supervised learning estimator with a feature_importances_ attribute or
        a callable that returns feature importances when passed X and y.
        The estimator must be an instance of Mother importance estimator
        (e.g., MotherCatboostImportance or MotherPermutationImportance).

    n_estimators : int or 'auto', default='auto'
        Number of estimators to build. If 'auto', this is determined automatically
        based on the size of the dataset.

    perc : int, default=100
        Percentile of the empirical distribution of importances to be used as threshold
        for feature selection.

    alpha : float, default=0.05
        Level of significance for statistical tests.

    two_step : bool, default=True
        If True, perform two-step feature selection: tentative -> confirmatory.
        If False, perform one-step feature selection: all features vs the shadow ones.

    max_iter : int, default=100
        Maximum number of iterations to perform.

    verbose : int, default=0
        Controls verbosity of the output:
        - 0: no output
        - 1: displays iteration number
        - 2: displays iteration number and feature status information

    random_state : int, RandomState or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState, the random number generator is the RandomState instance used
        by np.random.

    include_tentative : bool, default=False
        If True, both confirmed and tentatively selected features are included in the
        transformed output. If False, only confirmed features are included.

    """

    def __init__(
        self,
        estimator,
        n_estimators: typing.Union[str, int] = "auto",  # Can be int or str "auto"
        perc: int = 100,
        alpha: float = 0.05,
        two_step: bool = True,
        max_iter: int = 100,
        verbose: int = 0,
        random_state: typing.Union[int, np.random.RandomState] = 42,
        include_tentative: bool = False,
    ):
        # Initialize the parent class with all explicit parameters
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            perc=perc,
            alpha=alpha,
            two_step=two_step,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
        )
        self.include_tentative = include_tentative

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        This method overrides BorutaPy's fit_transform to avoid issues with
        method signature mismatches when using SelectorMixin's transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_selected_features)
            The transformed data with selected features.
        """

        return self.fit(X, y).transform(X)

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected.
        Implements SelectorMixin requirement.

        If include_tentative is True, both confirmed and tentative features are included.
        Otherwise, only confirmed features are included.
        """
        check_is_fitted(self, ["support_", "support_weak_"])

        if self.include_tentative:
            return np.logical_or(self.support_, self.support_weak_)
        else:
            return self.support_

    def fit(self, X: typing.Union[pd.DataFrame, np.ndarray], y: typing.Union[pd.Series, np.ndarray]):
        """
        Fit the MotherBorutaPy model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
        """
        # For scikit-learn compatibility, only store feature_names_in_ when input has feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
        # For numpy arrays, sklearn convention is to NOT set feature_names_in_
        # This prevents warnings about mismatched feature names

        return super().fit(X, y)
