"""
Purposes
--------
This module provides Mother-compatible wrappers for TabICL models and exposes
Optuna-ready hyperparameter handling through the MotherML API.

More information on the following repository: https://github.com/soda-inria/tabicl

Hyperparameters Tuned by Optuna
-------------------------------
The shared tuning logic in _TabICLHyperParams.get_hyperparameter_space suggests:

- n_estimators
    - Type: int
    - Range: [1, 12]
    - Meaning: number of ensemble estimators

- softmax_temperature (only if present in model init params, typically classifier)
    - Type: float
    - Range: [0.5, 2.0]
    - Meaning: softmax temperature for probability calibration

- average_logits (only if present in model init params, typically classifier)
    - Type: categorical
    - Values: [True, False]
    - Meaning: average logits before softmax vs average probabilities after softmax

- outlier_threshold (only if present in model init params, typically regressor)
    - Type: float
    - Range: [2.0, 8.0]
    - Meaning: clipping threshold used to handle outlier context examples
"""

import logging
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from optuna.trial import Trial
from six import iteritems
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from tabicl import TabICLClassifier, TabICLRegressor

from mother.ml.core import AbstractMotherPipeline
from mother.ml.models import utils

module_logger = logging.getLogger(__name__)

# Default quantile for regression uncertainty estimation (interquartile range)
DEFAULT_QUANTILES: list[float] = [0.25, 0.5, 0.75]


class _TabICLHyperParams(AbstractMotherPipeline):
    """Shared Mother-style parameter management mixin for TabICL estimators.

    Provides a unified implementation of `get_params`, `set_params`,
    `get_hyperparameter_space` and input validation that is inherited by
    both :class:`TabICLClassifierMother` and :class:`TabICLRegressorMother`.

    This class is not intended to be instantiated directly.  It must be combined
    with a concrete ``TabICLClassifier`` or ``TabICLRegressor`` via multiple
    inheritance, and ``_store_initial_params`` must be called at the end of the
    child ``__init__`` once the parent TabICL constructor has run.

    Attributes
    ----------
    _init_params : dict
        Snapshot of all constructor parameters taken after the parent TabICL
        ``__init__`` has executed.  Used as the single source of truth by
        `get_params`, `set_params`, and the conditional logic in
        `get_hyperparameter_space`.
    _is_fitted : bool
        Guard flag set to ``True`` in `fit`.  Prevents parameter
        modification after the model has been trained.
    """

    def __init__(self):
        self._init_params: dict = {}
        self._is_fitted: bool = False

    def _store_initial_params(self) -> None:
        """Snapshot the current instance attributes into ``_init_params``.

        Must be called at the **end** of the child class ``__init__``, after
        ``super().__init__(**kwargs)`` has run, so that all hyperparameters
        assigned by the parent TabICL constructor (via ``self.xxx = ...``)
        are captured.
        """
        non_optimised_params = {"_init_params", "_is_fitted"}
        self._init_params = {key: value for key, value in self.__dict__.items() if key not in non_optimised_params}

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """Suggest hyperparameters for an Optuna trial.

        The set of suggested parameters depends on which keys are present in
        ``_init_params`` (populated from the concrete child class defaults):

        - ``n_estimators`` is always suggested (range [1, 12]).
        - ``softmax_temperature`` is suggested only for classifiers.
        - ``average_logits`` is suggested only for classifiers, and is forced
          to ``False`` when ``n_estimators == 1`` (averaging is meaningless
          with a single estimator).
        - ``outlier_threshold`` is suggested only for regressors.

        All parameter names passed to Optuna are prefixed with ``prefix`` to
        avoid name collisions when multiple models share the same trial.
        The returned dict keys are also prefixed for use with
        `set_params`.

        Parameters
        ----------
        X : array-like
            Training features (not used directly, required by the interface).
        y : array-like
            Training targets (not used directly, required by the interface).
        trial : optuna.trial.Trial
            Current Optuna trial object used to sample hyperparameter values.
        prefix : str, default ""
            String prepended to every parameter name, both in Optuna and in
            the returned dict (e.g. ``"clf__"`` for pipeline compatibility).

        Returns
        -------
        dict
            Mapping of prefixed parameter names to sampled values.
        """
        suggested_params: dict[str, object] = {
            "n_estimators": trial.suggest_int(prefix + "n_estimators", 1, 12, log=False),
        }

        if "softmax_temperature" in self._init_params:
            suggested_params["softmax_temperature"] = trial.suggest_float(
                prefix + "softmax_temperature", 0.5, 2.0, log=False
            )

        if "average_logits" in self._init_params:
            if suggested_params["n_estimators"] == 1:
                suggested_params["average_logits"] = False
            else:
                suggested_params["average_logits"] = trial.suggest_categorical(prefix + "average_logits", (True, False))

        if "outlier_threshold" in self._init_params:
            suggested_params["outlier_threshold"] = trial.suggest_float(
                prefix + "outlier_threshold", 2.0, 8.0, log=False
            )

        module_logger.info(
            "Suggested TabICL parameters in trial %s: %s",
            trial.number,
            suggested_params,
        )
        return utils.add_prefix_to_dict_keys(suggested_params, prefix=prefix)

    def get_params(self, deep=True) -> dict:
        """Return the hyperparameters stored in ``_init_params``.

        Overrides the default sklearn MRO resolution so that the Mother
        parameter store is always used instead of ``TabICLClassifier``'s
        own ``get_params``.

        Parameters
        ----------
        deep : bool, default True
            Ignored; kept for sklearn API compatibility.

        Returns
        -------
        dict
            Current hyperparameter names and values.
        """
        return self._init_params

    def set_params(self, **params):
        """Update hyperparameter values before the model is fitted.

        Updates both ``_init_params`` (the Mother store) and the
        corresponding instance attributes so that TabICL uses the new
        values on the next `fit` call.

        Modification is blocked after fitting (``_is_fitted=True``) and
        an error is logged instead of raising, to keep pipeline behaviour
        predictable.

        Parameters
        ----------
        **params
            Keyword arguments mapping parameter names to new values.
            Unknown keys (not in ``_init_params`` and not an instance
            attribute) are silently ignored.

        Returns
        -------
        self
        """
        if self._is_fitted:
            module_logger.error("The model is already fitted. You cannot change the parameters in the fitted model.")
            return self

        for key, value in params.items():
            if key in self._init_params:
                self._init_params[key] = value

            if hasattr(self, key):
                setattr(self, key, value)

        return self

    @staticmethod
    def _check_input_type(X, y: Optional[Union[np.ndarray, pd.Series]]) -> None:
        """Validate types and shapes of ``X`` and ``y`` before fitting.

        Raises a :exc:`TypeError` or :exc:`ValueError` early so that
        errors are surfaced before TabICL runs its own internal checks.
        NaN values are allowed because TabICL handles missing data natively.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray or pd.Series of shape (n_samples,)
            Target vector.

        Raises
        ------
        TypeError
            If ``X`` or ``y`` is not array-like.
        ValueError
            If ``X`` or ``y`` is empty.
        """
        if not hasattr(X, "shape") or (X.shape is None):
            raise TypeError(f"X must be array-like, not {type(X)}.")
        if y is None or (not hasattr(y, "shape")) or (y.shape is None):
            raise TypeError(f"y must be np.ndarray or pd.Series, not {type(y)}")
        if (X.shape[0] == 0) or (y.shape[0] == 0):
            raise ValueError("X and y must not be empty.")

        # Allow nan values since TabICL can handle them
        # Ensure_2d is False to allow 1D input for regression targets
        check_X_y(X, y, ensure_2d=False, force_all_finite="allow-nan")


# =========================================================== Classifier model


class TabICLClassifierMother(TabICLClassifier, _TabICLHyperParams):
    """Mother-compatible wrapper around :class:`tabicl.TabICLClassifier`.

    Combines TabICL's in-context-learning classifier with the MotherML
    hyperparameter management API (Optuna-ready, sklearn-compatible).

    Parameters
    ----------
    n_estimators : int, default 8
        Number of ensemble estimators.  Higher values improve stability at
        the cost of inference time.
    softmax_temperature : float, default 0.9
        Temperature applied to logits before the softmax.  Values below 1.0
        sharpen the distribution; values above 1.0 soften it.
    average_logits : bool, default True
        If ``True``, logits are averaged across estimators before the final
        softmax.  Automatically set to ``False`` when ``n_estimators == 1``.
    allow_auto_download : bool, default True
        Whether to automatically download the model checkpoint if not found
        locally.
    checkpoint_version : str, default "tabicl-classifier-v2-20260212.ckpt"
        Identifier of the pre-trained checkpoint to load.
    kv_cache : bool, default False
        Whether to use key-value caching during inference for faster
        repeated predictions on the same context.
    **kwargs
        Any additional keyword arguments are forwarded to
        :class:`tabicl.TabICLClassifier`.

    Examples
    --------
    Basic binary classification:

    >>> import numpy as np
    >>> from mother.ml.models.m_tabicl import TabICLClassifierMother
    >>> X_train = np.random.rand(50, 8)
    >>> y_train = (X_train[:, 0] > 0.5).astype(int)
    >>> X_test = np.random.rand(10, 8)
    >>> clf = TabICLClassifierMother()
    >>> clf.fit(X_train, y_train)
    TabICLClassifierMother(...)
    >>> y_pred = clf.predict(X_test)
    >>> y_proba = clf.predict_proba(X_test)  # shape (10, 2)

    Using inside a Mother pipeline with Optuna optimisation:

    >>> from mother.ml.models.m_tabicl import TabICLClassifierMother
    >>> clf = TabICLClassifierMother(n_estimators=8, softmax_temperature=0.9)
    >>> # clf.get_hyperparameter_space(X, y, trial) returns an Optuna-ready dict
    """

    def __init__(self, **kwargs):
        """Initialise the classifier with Mother defaults merged with user kwargs.

        Initialisation order:

        1. ``_TabICLHyperParams.__init__`` sets ``_init_params={}`` and
           ``_is_fitted=False``.
        2. `default_parameters` values are injected into ``kwargs`` only
           for keys not already supplied by the caller.
        3. ``super().__init__(**kwargs)`` runs ``TabICLClassifier.__init__``,
           which assigns every parameter as an instance attribute
           (``self.n_estimators = n_estimators``, etc.).
        4. `_store_initial_params` snapshots ``self.__dict__`` into
           ``_init_params`` for use by `get_params` and
           `get_hyperparameter_space`.
        """
        _TabICLHyperParams.__init__(self)

        # Set default parameters if not provided in kwargs, keeping user choices
        for key, value in self.default_parameters().items():
            kwargs.setdefault(key, value)

        super().__init__(**kwargs)
        self._store_initial_params()

    def get_params(self, deep=True) -> dict:
        """Return the classifier's hyperparameters from the Mother store.

        Explicitly overrides ``TabICLClassifier.get_params`` (first in MRO)
        to guarantee the Mother ``_init_params`` dict is always returned.

        Parameters
        ----------
        deep : bool, default True
            Ignored; kept for sklearn API compatibility.

        Returns
        -------
        dict
            Current hyperparameter names and values.
        """
        return self._init_params

    def set_params(self, **params):
        """Update classifier hyperparameters before fitting.

        Mirrors the pattern of `get_params`: explicitly overrides
        ``TabICLClassifier.set_params`` so that both the Mother store
        (``_init_params``) and the underlying ``TabICLClassifier`` instance
        attributes are updated in sync.

        Parameters
        ----------
        **params
            Keyword arguments mapping parameter names to new values.

        Returns
        -------
        self
        """
        if self._is_fitted:
            module_logger.error("The model is already fitted. You cannot change the parameters in the fitted model.")
            return self

        for key, value in iteritems(params):
            if key in self._init_params.keys():
                self._init_params[key] = value

        # super class: TabICLClassifier to update the parameters
        super().set_params(**params)
        return self

    def default_parameters(self, prefix: str = "") -> dict:
        """Return the default hyperparameter values for this classifier.

        Used during ``__init__`` to fill in any parameters not explicitly
        provided by the caller, and by the MotherML tuning infrastructure
        to know which parameters exist for this model. Default parameters
        seleted based on TabICl github's repository.

        Parameters
        ----------
        prefix : str, default ""
            Optional prefix prepended to every key (e.g. ``"clf__"`` for
            pipeline use).

        Returns
        -------
        dict
            Default hyperparameter names (optionally prefixed) and values.
        """
        return utils.add_prefix_to_dict_keys(
            {
                "n_estimators": 8,
                "softmax_temperature": 0.9,
                "average_logits": True,
                "allow_auto_download": True,
                "kv_cache": False,
            },
            prefix=prefix,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "TabICLClassifierMother":
        """Fit the TabICL classifier on labelled data.

        Validates inputs, then delegates to
        `tabicl.TabICLClassifier.fit`.  Sets ``_is_fitted=True`` to
        block further parameter modification via `set_params`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.  Lists are accepted and automatically converted
            to a column vector with a warning.
        y : np.ndarray of shape (n_samples,)
            Class labels (integer or string).  Lists are accepted and
            automatically converted.

        Returns
        -------
        self
            The fitted classifier instance.

        Raises
        ------
        TypeError
            If ``X`` or ``y`` is not array-like.
        ValueError
            If ``X`` or ``y`` is empty.
        """
        if isinstance(X, list):
            X = np.array(X).reshape(-1, 1)
            module_logger.warning(
                "X is given as list type. It is converted into np.array with shape %s.",
                X.shape,
            )
        if isinstance(y, list):
            y = np.array(y)
            module_logger.warning(
                "y is given as list type. It is converted into np.array with shape %s.",
                y.shape,
            )

        self._check_input_type(X, y)

        self._is_fitted = True
        # Fit the TabICLClassifier on the data
        super().fit(X, y)
        return self


# =========================================================== Regressor Model
class TabICLRegressorMother(TabICLRegressor, _TabICLHyperParams):
    """Mother-compatible wrapper around :class:`tabicl.TabICLRegressor`.

    Combines TabICL's in-context-learning regressor with MotherML's
    hyperparameter and pipeline API.

    Parameters
    ----------
    n_estimators : int, default 8
        Number of ensemble estimators. Higher values typically improve
        prediction stability at the cost of runtime.
    outlier_threshold : float, default 4.0
        Clipping threshold used by TabICL to reduce sensitivity to extreme
        context examples.
    allow_auto_download : bool, default True
        Whether to automatically download model checkpoints if needed.
    kv_cache : bool, default False
        Whether to enable key-value cache acceleration for repeated inference.
    **kwargs
        Any additional keyword arguments accepted by
        :class:`tabicl.TabICLRegressor`.

    Examples
    --------
    Basic regression:

    >>> import numpy as np
    >>> from mother.ml.models.m_tabicl import TabICLRegressorMother
    >>> X_train = np.random.rand(60, 6)
    >>> y_train = X_train[:, 0] * 2.5 - X_train[:, 1] + np.random.normal(0, 0.05, 60)
    >>> X_test = np.random.rand(8, 6)
    >>> reg = TabICLRegressorMother()
    >>> reg.fit(X_train, y_train)
    TabICLRegressorMother(...)
    >>> y_pred = reg.predict(X_test)

    Regression with uncertainty estimates:

    >>> uncertainty_df = reg.predict_uncertainty(X_test)
    >>> list(uncertainty_df.columns)
    ['mean_predictions', 'knowledge_uncertainty', 'data_uncertainty', 'total_uncertainty']
    """

    def __init__(self, **kwargs):
        # Initialize empty hyperparameter dictionary
        _TabICLHyperParams.__init__(self)

        # Update the kwargs with hyperparameters with the defaults parameters
        for key, value in self.default_parameters().items():
            kwargs.setdefault(key, value)

        # Set up the TabICLRegressor original class with the hyperparameters
        super().__init__(**kwargs)

        # Copy the hyperparameters in the child class (for optuna optimization)
        self._store_initial_params()

    def get_params(self, deep=True) -> dict:
        """Return the regressor's hyperparameters from the Mother store.

        Parameters
        ----------
        deep : bool, default True
            Ignored; kept for sklearn API compatibility.

        Returns
        -------
        dict
            Current hyperparameter names and values.
        """
        return _TabICLHyperParams.get_params(self, deep=deep)

    def set_params(self, **params):
        """Update regressor hyperparameters before fitting.

        Parameters
        ----------
        **params
            Keyword arguments mapping parameter names to new values.

        Returns
        -------
        self
        """
        if self._is_fitted:
            module_logger.error("The model is already fitted. You cannot change the parameters in the fitted model.")
            return self

        # Update the mother wrapper class
        for key, value in iteritems(params):
            if key in self._init_params.keys():
                self._init_params[key] = value

        # Update the original tabicl regressor class
        super().set_params(**params)

        return self

    def default_parameters(self, prefix: str = "") -> dict:
        """Return default hyperparameter values for the regressor.

        Parameters
        ----------
        prefix : str, default ""
            Optional prefix prepended to each key (for pipeline usage).

        Returns
        -------
        dict
            Default hyperparameter names (optionally prefixed) and values.
        """
        return utils.add_prefix_to_dict_keys(
            {
                "n_estimators": 8,
                "outlier_threshold": 4.0,
                "allow_auto_download": True,
                "kv_cache": False,
            },
            prefix=prefix,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "TabICLRegressorMother":
        """Fit the TabICL regressor on labeled data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix. Lists are accepted and converted to arrays.
        y : np.ndarray of shape (n_samples,)
            Continuous target values. Lists are accepted and converted.

        Returns
        -------
        self
            The fitted regressor instance.
        """
        if isinstance(X, list):
            X = np.array(X).reshape(-1, 1)
            module_logger.warning(
                "X is given as list type. It is converted into np.array with shape %s.",
                X.shape,
            )
        if isinstance(y, list):
            y = np.array(y)
            module_logger.warning(
                "y is given as list type. It is converted into np.array with shape %s.",
                y.shape,
            )

        self._check_input_type(X, y)

        self._is_fitted = True

        # fit the original tabicl regressor class on the data
        super().fit(X, y)
        return self

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        return_quantiles: bool = False,
        quantiles: list = DEFAULT_QUANTILES,
        uncertainty_for_opt: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, pd.Series, tuple[pd.DataFrame, np.ndarray]]:
        """
        Predict the target values and estimate uncertainty of given input.
        The uncertainty is measured by interquartile range for each sample.

        Args:
            X : ArrayLike
                input to predict and estimate the uncertainty.
            quantiles : list = [.25, .5, .75]
                list of quantiles to calculate the uncertainty.
            return_quantiles : bool
                If True, return quantile values (default is False).

        Returns:
            Union[pd.DataFrame, tuple[pd.DataFrame, np.array]]:
                - If `return_quantiles=False`: A DataFrame with columns:
                    - 'mean_predictions': The mean predictions for each sample (mean of quantiles).
                    - 'knowledge_uncertainty': None,
                    currently not supported for this method (included just for compatibility)
                    - 'data_uncertainty': None,
                    currently not supported for this method (included just for compatibility)
                    - 'total_uncertainty': The uncertainty quantified for each sample (interquartile range).
                - If `return_quantiles=True`: A tuple containing:
                    - The DataFrame described above.
                    - np.array of quantile values whose shape is (# samples, # quantiles).
        """
        check_is_fitted(self)

        # Update the quantiles list with default quantiles
        for q in DEFAULT_QUANTILES:
            if q not in quantiles:
                quantiles.append(q)
        quantiles.sort()

        pred_res: Union[np.ndarray, dict] = self.predict(
            np.array(X), output_type="quantiles", alphas=quantiles, **kwargs
        )

        if not isinstance(pred_res, np.ndarray):
            raise TypeError(
                "Expected TabICLRegressor.predict with output_type='quantiles' to return an array, "
                f"got {type(pred_res)}."
            )

        output: pd.DataFrame = pd.DataFrame(
            {
                "mean_predictions": pred_res.mean(axis=1).tolist(),
                "knowledge_uncertainty": None,  # Not available for this model
                "data_uncertainty": None,  # Not available for this model
                "total_uncertainty": (pred_res[:, quantiles.index(0.75)] - pred_res[:, quantiles.index(0.25)]).tolist(),
            },
        )

        # Apply the correct index if a dataframe is given as input
        if isinstance(X, pd.DataFrame):
            output.index = X.index

        # If return_quantiles is True, also return the quantiles values as a numpy array
        if uncertainty_for_opt:
            return output.loc[:, "total_uncertainty"]
        if return_quantiles:
            return output, pred_res

        return output


# TODO: Test and confirm the good functioning of this class
class TabICLEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Transformer that extracts TabICL row-interaction representations as embeddings.

    Hooks into the ``row_interactor`` output of the underlying TabICL model to
    capture per-row embedding vectors.  These vectors encode both within-row
    feature interactions and the distributional context of the full dataset.

    For training data, k-fold cross-validation is used to generate out-of-fold
    embeddings and avoid data leakage.  For new data, a single model fitted on
    all available training samples generates the embeddings.

    Parameters
    ----------
    model_type : {'classification', 'regression'}, default='classification'
        Whether to use a :class:`tabicl.TabICLClassifier` or
        :class:`tabicl.TabICLRegressor` as the underlying model.
        Ignored when a pre-fitted *model* is provided.
    n_folds : int, default=5
        Number of folds for cross-validation when generating training embeddings.
    use_kfold : bool, default=True
        Whether to use k-fold cross-validation for training embeddings.
        If ``False``, a single model is fitted on all data and its representations
        for the training set are stored (note: this introduces data leakage for
        the training embeddings).
    random_state : int or None, default=42
        Random seed for reproducibility of k-fold splitting and TabICL ensemble.
    embedding_column_name : str, default='tabiclembedding'
        Name or prefix for the output embedding columns.
    return_separate_columns : bool, default=True
        If ``True``, each embedding dimension is returned as a separate column
        (e.g. ``tabiclembedding_0``, ``tabiclembedding_1``, …).
        If ``False``, each row's embedding vector is stored as a single object
        in one column.
    model : TabICLClassifier or TabICLRegressor or None, default=None
        A pre-fitted TabICL estimator.  When provided the model is used as-is
        and k-fold fitting is skipped.  Cannot be combined with
        ``use_kfold=True``.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`tabicl.TabICLClassifier` or :class:`tabicl.TabICLRegressor`
        when creating new estimators.

    Attributes
    ----------
    model : TabICLClassifier or TabICLRegressor
        The fitted estimator used for representing new data.
    train_embeddings_ : ndarray of shape (n_samples, embedding_dim)
        Out-of-fold (or full-data, when ``use_kfold=False``) embeddings for
        the training samples.
    input_features_ : list of str or None
        Column names seen during ``fit``.  ``None`` when ``X`` was a NumPy array.
    train_index_ : Index or None
        Row index from the training ``DataFrame``.  ``None`` when ``X`` was a
        NumPy array.

    Examples
    --------
    Classification embeddings with out-of-fold training vectors:

    >>> import pandas as pd
    >>> from mother.ml.models.m_tabicl import TabICLEmbeddingTransformer
    >>> X = pd.DataFrame({"f1": [0.1, 0.2, 0.9, 1.0], "f2": [1.0, 0.9, 0.2, 0.1]})
    >>> y = pd.Series([0, 0, 1, 1])
    >>> emb = TabICLEmbeddingTransformer(model_type="classification", n_folds=2)
    >>> X_emb_train = emb.fit_transform(X, y)
    >>> X_emb_test = emb.transform(X.iloc[:2])

    Regression embeddings as one vector column:

    >>> emb_reg = TabICLEmbeddingTransformer(
    ...     model_type="regression",
    ...     return_separate_columns=False,
    ...     n_folds=2,
    ... )
    >>> y_reg = pd.Series([1.2, 1.0, 2.5, 2.8])
    >>> X_emb = emb_reg.fit_transform(X, y_reg)
    >>> X_emb.columns.tolist()
    ['tabiclembedding']
    """

    def __init__(
        self,
        model_type: Literal["classification", "regression"] = "classification",
        n_folds: int = 5,
        use_kfold: bool = True,
        random_state: Optional[int] = None,
        embedding_column_name: str = "tabiclembedding",
        return_separate_columns: bool = True,
        model: Optional[Union[TabICLClassifier, TabICLRegressor]] = None,
        **kwargs: Any,
    ) -> None:
        self.model_type = model_type
        self.n_folds = n_folds
        self.use_kfold = use_kfold
        self.random_state = random_state
        self.embedding_column_name = embedding_column_name
        self.return_separate_columns = return_separate_columns
        self.kwargs: Dict[str, Any] = kwargs
        self.model = model
        self.pre_fitted: bool = model is not None

        # Populated during fit
        self.input_features_: Optional[List[str]] = None
        self.train_embeddings_: Optional[np.ndarray] = None
        self.train_index_: Optional[pd.Index] = None
        self._embedding_dim: Optional[int] = None

        if self.pre_fitted and self.use_kfold:
            raise ValueError(
                "Cannot use k-fold fitting when a pre-fitted model is already given. "
                "Set either use_kfold=False or model=None."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_estimator(self) -> Union[TabICLClassifier, TabICLRegressor]:
        """Instantiate a new TabICL estimator with the configured parameters."""
        kwargs = dict(self.kwargs)

        if self.random_state is not None:
            kwargs.setdefault("random_state", self.random_state)

        if self.model_type == "classification":
            return TabICLClassifier(**kwargs)

        elif self.model_type == "regression":
            return TabICLRegressor(**kwargs)

        else:
            raise ValueError(f"Invalid model_type '{self.model_type}'. Use 'classification' or 'regression'.")

    def _extract_representations(
        self,
        fitted_estimator: Union[TabICLClassifier, TabICLRegressor],
        X_test: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Extract per-row representations from the row-interaction transformer.

        Registers a forward hook on ``fitted_estimator.model_.row_interactor``
        to capture its output tensor during a single ``predict_proba`` / ``predict``
        call.

        Two cases arise depending on whether KV caching is active:

        **Why the hook is placed on row_interactor:**

        TabICL's ``row_interactor`` is a *within-row* transformer: it uses learnable
        CLS tokens to aggregate the column embeddings of each row independently
        (the row axis ``T`` is a batch dimension — rows do not attend to each other
        here).  The training context is baked in *before* this step, by
        ``col_embedder``, which uses set-attention over all feature columns with
        access to the training distribution.  By the time ``row_interactor`` runs,
        the per-column embeddings already encode the dataset context, making the
        output of ``row_interactor`` the cleanest representation of each sample that
        is still upstream of the cross-row ICL predictor head.

        There is **no public API** in TabICL for extracting these representations
        (unlike TabPFN's ``get_embeddings()``).  A PyTorch forward hook is therefore
        the correct and minimal approach — it requires no modification of TabICL
        internals and is removed immediately after the forward pass.

        Two shapes arise at the hook depending on KV caching:

        * **No KV cache** (default): all rows are concatenated and passed jointly —
          ``[X_train | X_test]`` — so the hook captures tensors of shape
          ``(B, train_size + test_size, repr_dim)``.  Only the test-sample slice
          ``[:, train_size:, :]`` is kept.

        * **KV cache active** (``kv_cache="kv"`` or ``kv_cache="repr"``): training
          data is handled via pre-computed cached projections, so
          ``row_interactor`` is only called on the test samples and captures
          ``(B, test_size, repr_dim)``.  The training context is still present
          because ``col_embedder`` attended to the cached training projections
          before producing the inputs to ``row_interactor``.

        Parameters
        ----------
        fitted_estimator : TabICLClassifier or TabICLRegressor
            A fully fitted TabICL estimator (``fit`` already called).
        X_test : array-like of shape (n_test_samples, n_features)
            Samples whose representations should be extracted.

        Returns
        -------
        ndarray of shape (n_samples, repr_dim)
            Per-row representation vectors averaged over all ensemble members.
        """
        representations_list: List[np.ndarray] = []

        def _hook(module, input, output):  # noqa: ANN001

            # Get the ouput (tensor), detach from calculation graph, move the tensor on the CPU to convert it to numpy
            # array and append it to the representations list
            representations_list.append(output.detach().cpu().float().numpy())

        hook = fitted_estimator.model_.row_interactor.register_forward_hook(_hook)

        # Force a forward pass to trigger the hook and populate representations_list.
        try:
            if self.model_type == "classification":
                if not isinstance(fitted_estimator, TabICLClassifier):
                    raise TypeError(
                        f"model_type='classification' requires a fitted TabICLClassifier, got {type(fitted_estimator)}."
                    )
                fitted_estimator.predict_proba(self._to_array(X_test))
            else:
                if not isinstance(fitted_estimator, TabICLRegressor):
                    raise TypeError(
                        f"model_type='regression' requires a fitted TabICLRegressor, got {type(fitted_estimator)}."
                    )
                fitted_estimator.predict(self._to_array(X_test))
        # Remove the hook even if prediction fails to avoid side effects.
        finally:
            hook.remove()

        n_test: int = X_test.shape[0]

        # Get the number of training samples if KV cache is not used (train and test samples passed together to the row
        # interactor)
        train_size: int = fitted_estimator.n_samples_in_

        # Each entry in representations_list has shape (batch_B, T (number of lines in row_interactor), repr_dim).
        all_repr = np.concatenate(representations_list, axis=0)  # (total_estimators, T, repr_dim)
        T: int = all_repr.shape[1]

        if T == n_test:
            # KV cache was active: row_interactor processed only test samples.
            test_repr = all_repr
        else:
            # No cache: row_interactor processed [X_train | X_test] jointly.
            # Slice off the train rows (T == train_size + n_test).
            test_repr = all_repr[:, train_size:, :]  # (total_estimators, n_test, repr_dim)

        # Average over all ensemble members to obtain a single vector per sample (n_estimators parameters in the model
        # class).
        return test_repr.mean(axis=0)  # (n_test, repr_dim)

    def _to_array(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return X.values if isinstance(X, pd.DataFrame) else X

    def _to_array_y(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        return np.array(y) if isinstance(y, pd.Series) else y

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "TabICLEmbeddingTransformer":
        """Fit the transformer and compute training embeddings.

        Uses k-fold cross-validation (when ``use_kfold=True``) to compute
        out-of-fold embeddings for the training data, avoiding data leakage.
        A final model is fitted on all training data to enable embedding of
        new test samples via :meth:`transform`.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like or Series of shape (n_samples,)
            Target values.  Required by TabICL for in-context learning.
        groups : array-like of shape (n_samples,) or None, default=None
            Group labels used for group-aware k-fold splitting.  When provided,
            classification uses :class:`~sklearn.model_selection.StratifiedGroupKFold`
            and regression uses :class:`~sklearn.model_selection.GroupKFold`.

        Returns
        -------
        self : TabICLEmbeddingTransformer
        """
        if y is None:
            raise ValueError("TabICL requires target values (y) for fitting.")

        is_df: bool = isinstance(X, pd.DataFrame)
        self.train_index_ = X.index if is_df else None
        self.input_features_ = X.columns.tolist() if is_df else None

        X_arr: np.ndarray = self._to_array(X)
        y_arr: np.ndarray = self._to_array_y(y)
        groups_arr: Optional[np.ndarray] = np.array(groups) if isinstance(groups, pd.Series) else groups

        if self.pre_fitted:
            if self.model is None:
                raise RuntimeError("Internal error: pre_fitted=True but model is None.")

            # Use the supplied model directly — no fitting required.
            module_logger.info(
                "A pre-fitted model was provided. Extracting training representations without refitting."
            )

            self.train_embeddings_ = self._extract_representations(self.model, X)

        else:
            # Train the model using kfold to avoid data leakage in the embeddings representation
            if self.use_kfold and X_arr.shape[0] >= self.n_folds:
                module_logger.info("Fitting TabICL with %d-fold cross-validation.", self.n_folds)

                fold_iterator: Iterator[Tuple[np.ndarray, np.ndarray]] = self._build_fold_iterator(
                    X_arr, y_arr, groups_arr
                )

                # List to hold (original_index, embedding) pairs for each validation fold, which will be concatenated
                # and sorted
                embedding_chunks: List[Tuple[int, np.ndarray]] = []

                for train_idx, val_idx in fold_iterator:
                    # Generate the estimator for this fold and fit it on the training split
                    fold_est = self._make_estimator()
                    X_train_fold = X.iloc[train_idx] if is_df else X_arr[train_idx]
                    X_val_fold = X.iloc[val_idx] if is_df else X_arr[val_idx]
                    fold_est.fit(np.array(X_train_fold), y_arr[train_idx])

                    # Extract the embeddings on the validation split to avoid data leakage
                    val_repr: np.ndarray = self._extract_representations(fold_est, np.array(X_val_fold))

                    # Save the original indices and corresponding embeddings for this fold
                    for orig_idx, emb in zip(val_idx, val_repr):
                        embedding_chunks.append((int(orig_idx), emb))

                # Restore original sample order and concatenate the embeddings in a single matrix
                embedding_chunks.sort(key=lambda chunk_tuple: chunk_tuple[0])
                self.train_embeddings_ = np.stack([emb for _, emb in embedding_chunks], axis=0)

                # Train the main model on *all* data for use during transform().
                module_logger.info("Fitting main TabICL model on full training data.")
                self.model = self._make_estimator()
                self.model.fit(np.array(X), y_arr)

            else:
                if self.use_kfold:
                    module_logger.warning(
                        "Number of samples (%d) is less than n_folds (%d). "
                        "Falling back to a single model without k-fold.",
                        X_arr.shape[0],
                        self.n_folds,
                    )
                self.model = self._make_estimator()
                self.model.fit(np.array(X), y_arr)

                # No OOF available — use self-context embeddings (data leakage risk).
                self.train_embeddings_ = self._extract_representations(self.model, np.array(X))

        self._embedding_dim = self.train_embeddings_.shape[1]
        return self

    def _build_fold_iterator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return a k-fold iterator appropriate for the task and group settings."""
        if self.model_type == "classification":
            if groups is not None:
                n_splits = min(self.n_folds, len(np.unique(groups)))
                kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                return kf.split(X, y, groups=groups)
            else:
                kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                return kf.split(X, y)
        else:
            if groups is not None:
                n_splits = min(self.n_folds, len(np.unique(groups)))
                kf = GroupKFold(n_splits=n_splits)
                return kf.split(X, y, groups=groups)
            else:
                kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                return kf.split(X)

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> pd.DataFrame:
        """Transform samples into TabICL row-interaction representations.

        For training samples (identified by index), cached out-of-fold embeddings
        are returned directly.  For new samples, the fitted main model generates
        representations via a single forward pass.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Samples to embed.

        Returns
        -------
        pd.DataFrame
            Embeddings.  Shape depends on ``return_separate_columns``:

            * ``True``  → ``(n_samples, embedding_dim)`` with columns
              ``tabiclembedding_0``, …, ``tabiclembedding_{d-1}``.
            * ``False`` → ``(n_samples, 1)`` with a single column
              ``tabiclembedding`` whose values are 1-D arrays.
        """
        check_is_fitted(self, "model")
        if self.model is None:
            raise RuntimeError("Internal error: model is None after fitting.")

        is_df: bool = isinstance(X, pd.DataFrame)
        index = X.index if is_df else None

        if self.input_features_ is not None and is_df:
            missing = set(self.input_features_) - set(X.columns)
            if missing:
                raise ValueError(f"Features {missing} seen during training are missing from X.")
            X = X[self.input_features_]

        embeddings = self._extract_representations(self.model, X)
        return self._to_dataframe(embeddings, index)

    def fit_transform(  # type: ignore[override]
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Fit and return embeddings for the training data.

        Delegates to :meth:`fit` and then returns the cached
        ``train_embeddings_`` computed during fitting (out-of-fold when
        ``use_kfold=True``).

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
        y : array-like or Series of shape (n_samples,)
        groups : array-like of shape (n_samples,) or None, default=None

        Returns
        -------
        pd.DataFrame
            Training embeddings (out-of-fold when ``use_kfold=True``).
        """
        self.fit(X, y, groups)

        if self.train_embeddings_ is None:
            raise RuntimeError("train_embeddings_ was not computed during fit.")

        return self._to_dataframe(self.train_embeddings_, self.train_index_)

    def get_feature_names_out(self) -> np.ndarray:
        """Return output feature names.

        Returns
        -------
        ndarray of str
            Column names matching the output of :meth:`transform`.
        """
        check_is_fitted(self, "model")
        if self._embedding_dim is None:
            raise ValueError("Transformer has not been fitted yet.")

        if self.return_separate_columns:
            return np.array([f"{self.embedding_column_name}_{i}" for i in range(self._embedding_dim)])
        return np.array([self.embedding_column_name])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_dataframe(
        self,
        embeddings: np.ndarray,
        index: Optional[pd.Index],
    ) -> pd.DataFrame:
        if self.return_separate_columns:
            cols: List[str] = [f"{self.embedding_column_name}_{i}" for i in range(embeddings.shape[1])]
            if index is not None:
                return pd.DataFrame(embeddings, columns=cols, index=index)
            return pd.DataFrame(embeddings, columns=cols)
        else:
            data = {self.embedding_column_name: list(embeddings)}
            if index is not None:
                return pd.DataFrame(data, index=index)
            return pd.DataFrame(data)
