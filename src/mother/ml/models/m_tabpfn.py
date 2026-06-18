"""
### **Purpose**
    These classes extend TabPFN models with advanced
    hyperparameter tuning capabilities, integrating
    seamlessly with the Mother framework and Optuna.


### **Hyper-paramters Tuned by Optuna**

Please find the details on https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/regressor.py#L198

1. **`n_estimators`**:
    - Type: int
    - Values: `[1, 8]`
    - Purpose: the number of estimators in the ensemble.


2. **`softmax_temperature`**:
    - Type: float
    - Values: `[0.5, 2.0]`
    - Purpose: the temperature value for the soft-max function which introduces
               a randomness/confidence in the soft-max converted probabilities.

3. **`average_before_softmax`**:
    - Type: bool
    - Values: `(True, False)`
    - Purpose: Whether to average the predictions of the estimators before applying
               the softmax function. (applied only if n_estimators > 1). This is only
               applied when predicting during a post-processing.
"""

import logging
import random
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import torch
from optuna.trial import Trial
from six import iteritems
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.regressor import FullOutputDict

from mother.ml.core import AbstractMotherPipeline
from mother.ml.models import utils

module_logger = logging.getLogger(__name__)

# sklearn >= 1.8 is the minimum required version; `ensure_all_finite` is always available.
_ALLOW_NAN_KWARG: dict = {"ensure_all_finite": "allow-nan"}
DEFAULT_QUANTILES: list[float] = [0.25, 0.5, 0.75]


class _TabPFNHyperParams(AbstractMotherPipeline):
    """
    A utility class for managing and defining hyperparameter spaces for TabPFN models.

    This class provides methods to define hyperparameter spaces for TabPFN models (e.g., TabPFNClassifier,
    TabPFNRegressor). It supports dynamic hyperparameter
    tuning using Optuna and integrates seamlessly with the Mother framework.

    Attributes
    ----------
    _is_fitted : bool
        Indicate whether the model is already fitted or not.
    _init_params : dict
        A dictionary of parameters only used for the TabPFN training


    Methods
    -------
    get_hyperparameter_space(X, y, trial, prefix=None) -> dict
        Defines the hyperparameter space for TabPFN models based on the input data and trial.
    _check_input_shape(X_shape) -> None
        automatically turns on the 'ignore_pretraining_limits' option when the given data
        exceeds the limit of # feattures or # samples in TabPFN.
    _check_input_type(self, X: ArrayLike, y: ArrayLike) -> None:
        check whether the input X and y have the right data type and format.
    """

    def __init__(self):
        # dictionary to save all tunable parameters
        self._init_params: dict = {}

        # To prevent the parameter update
        self._is_fitted: bool = False

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Defines the Optuna hyperparameter search space for TabPFN models based on the input data and trial.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        y : pd.Series or pd.DataFrame
            Target data.
        trial : optuna.trial.Trial
            Optuna trial object.
        prefix : str, optional
            Optional prefix for parameter names.

        Returns
        -------
        dict
            Suggested hyperparameters for the current trial.
        """
        suggested_params: dict = {
            "softmax_temperature": trial.suggest_float(prefix + "softmax_temperature", 0.5, 2.0, log=False)
        }

        # Condition for the average_before_softmax
        if self.n_estimators == 1:
            suggested_params["average_before_softmax"] = False
        else:
            suggested_params["average_before_softmax"] = trial.suggest_categorical(
                prefix + "average_before_softmax", (True, False)
            )

        module_logger.info(f"Suggested parameters in trial {trial.number}: {suggested_params}")

        return utils.add_prefix_to_dict_keys(suggested_params, prefix=prefix)

    def _check_input_shape(self, X_shape: Union[list, tuple]) -> None:
        if not self._init_params["ignore_pretraining_limits"]:
            if X_shape[1] > 500:
                module_logger.warning(
                    f"The input has {X_shape[1]} features, whereas TabPFN encourages users to have # features <= 500"
                )
                self.set_params(ignore_pretraining_limits=True)

            if X_shape[0] > 10000:
                module_logger.warning(
                    f"The input has {X_shape[0]} samples, whereas TabPFN encourages users to have # samples <= 10000"
                )
                self.set_params(ignore_pretraining_limits=True)

            if self._init_params["ignore_pretraining_limits"]:
                module_logger.warning("ignore_pretraining_limits option is automatically turned on")

    def _check_input_type(self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]]) -> None:
        if not hasattr(X, "shape") or (X.shape is None):
            raise TypeError(f"X must be pd.DataFrame not {type(X)}.")
        elif not hasattr(y, "shape") or (y.shape is None):
            raise TypeError(f"y must be np.ndarray or pd.Series not {type(y)}")
        elif (X.shape[0] == 0) or (y.shape[0] == 0):
            raise ValueError("X and y must not be empty.")

        # ensure_2d is off because of the 1D case
        check_array(X, ensure_2d=False, **_ALLOW_NAN_KWARG)
        check_array(y, ensure_2d=False, dtype=None, **_ALLOW_NAN_KWARG)


class TabPFNRegressorMother(TabPFNRegressor, _TabPFNHyperParams):
    """
     A custom implementation of TabPFNRegressor with extended functionality for hyperparameter tuning.

    This class extends the TabPFNRegressor and integrates with the Mother framework to provide
    dynamic hyperparameter tuning using Optuna.

    Methods
    -------
    default_parameters(prefix: str = "") -> dict
        Returns the default parameters for the TabPFNRegressor.
    get_params() -> dict
        Returns the current parameters for the TabPFNRegressor.
    set_params(**params) -> None
        Sets the given parameters with new values (applicable only when the model is not fitted).

    Example
    -------
    >>> from optuna.trial import Trial
    >>> from mother.ml.m_tabpfn import TabPFNRegressorMother
    >>> import pandas as pd
    >>> import numpy as np
    >>> trial = Trial(...)
    >>> X = pd.DataFrame(np.random.rand(100, 10))
    >>> y = pd.Series(np.random.rand(100))
    >>> regressor = TabPFNRegressorMother()
    >>> hyperparams = regressor.get_hyperparameter_space(X, y, trial)
    >>> print(hyperparams)
    """

    def __init__(self, **kwargs):
        _TabPFNHyperParams.__init__(self)

        # Default to commercially-licensed V2 model weights.
        # Following the official recommendation:
        #   TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        if "model_path" not in kwargs:
            kwargs["model_path"] = TabPFNRegressor.create_default_for_version(ModelVersion.V2).model_path

        for key, val in self.default_parameters().items():
            if key not in list(kwargs):
                kwargs[key] = val
        super().__init__(**kwargs)

        # set up the initial parameters for optuna optimisation
        non_optimised_params: list[str] = ["_init_params", "_is_fitted"]
        for k, v in self.__dict__.items():
            if k not in non_optimised_params:
                self._init_params[k] = v

    def default_parameters(self, prefix: str = "") -> dict:
        return utils.add_prefix_to_dict_keys({"softmax_temperature": 1}, prefix=prefix)

    def get_params(self, deep=True) -> dict:
        return self._init_params

    def set_params(self, **params) -> "TabPFNRegressorMother":
        if self._is_fitted:
            module_logger.error("The model is already fitted. You cannot change the parameters in the fitted model.")
            return self

        for key, value in iteritems(params):
            if key in self._init_params.keys():
                self._init_params[key] = value

        # super class
        super().set_params(**params)

        return self

    def fit(self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]]) -> "TabPFNRegressorMother":
        """
        Fit the model to the given data.

        Args:
            X : ArrayLike
                input to fit.
            y : ArrayLike
                target value to fit.
        """
        if isinstance(X, list):
            X = np.array(X).reshape(-1, 1)
            module_logger.warning(f"X is given as list type. It is convereted into np.array with a shape of {X.shape}.")
        if isinstance(y, list):
            y = np.array(y)
            module_logger.warning(f"y is given as list type. It is convereted into np.array with a shape of {y.shape}.")

        self._check_input_type(X, y)
        self._check_input_shape(X.shape)

        # fit the TabPFNRegressor model
        self._is_fitted = True
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

        for q in DEFAULT_QUANTILES:
            if q not in quantiles:
                quantiles.append(q)
        quantiles.sort()

        pred_res: FullOutputDict = super().predict(X=X, quantiles=quantiles, output_type="full")
        pred_res["quantiles"] = np.array(pred_res["quantiles"]).T

        output: pd.DataFrame = pd.DataFrame(
            {
                "pred": pred_res["mean"],
                "mean_predictions": None,
                "knowledge_uncertainty": None,
                "data_uncertainty": None,
                "total_uncertainty": pred_res["quantiles"][:, quantiles.index(0.75)]
                - pred_res["quantiles"][:, quantiles.index(0.25)],
            },
            index=X.index,
        )

        if return_quantiles:
            return output, pred_res["quantiles"]
        elif uncertainty_for_opt:
            return output.loc[:, "total_uncertainty"]
        else:
            return output


class TabPFNClassifierMother(TabPFNClassifier, _TabPFNHyperParams):
    """
     A custom implementation of TabPFNClassifier with extended functionality for hyperparameter tuning.

    This class extends the TabPFNClassifier and integrates with the Mother framework to provide
    dynamic hyperparameter tuning using Optuna.

    Attributes
    ----------

    Methods
    -------
    default_parameters(prefix: str = "") -> dict
        Returns the default parameters for the TabPFNClassifier.

    get_params() -> dict
        Returns the current parameters for the TabPFNClassifier.

    set_params(**params) -> None
        Sets the given parameters with new values (applicable only when the model is not fitted).

    Example
    -------
    >>> from optuna.trial import Trial
    >>> from mother.ml.m_tabpfn import TabPFNClassifierrMother
    >>> import pandas as pd
    >>> import numpy as np
    >>> trial = Trial(...)
    >>> X = pd.DataFrame(np.random.rand(100, 10))
    >>> y = pd.Series(np.random.rand(100))
    >>> classifier = TabPFNClassifierMother()
    >>> hyperparams = classifier.get_hyperparameter_space(X, y, trial)
    >>> print(hyperparams)
    """

    def __init__(self, **kwargs):
        _TabPFNHyperParams.__init__(self)

        # Default to commercially-licensed V2 model weights.
        # Following the official recommendation:
        #   TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        if "model_path" not in kwargs:
            kwargs["model_path"] = TabPFNClassifier.create_default_for_version(ModelVersion.V2).model_path

        for key, val in self.default_parameters().items():
            if key not in list(kwargs):
                kwargs[key] = val

        # set balance_probabilities=True by defalut
        kwargs["balance_probabilities"] = True

        super().__init__(**kwargs)

        # set up the initial parameters for optuna optimisation
        non_optimised_params: list[str] = ["_init_params", "_is_fitted"]
        for k, v in self.__dict__.items():
            if k not in non_optimised_params:
                self._init_params[k] = v

    def default_parameters(self, prefix: str = "") -> dict:
        return utils.add_prefix_to_dict_keys(
            {
                "softmax_temperature": 1,
                "ignore_pretraining_limits": False,
                "balance_probabilities": True,
            },
            prefix=prefix,
        )

    def get_params(self, deep=True) -> dict:
        return self._init_params

    def set_params(self, **params) -> "TabPFNClassifierMother":
        if self._is_fitted:
            module_logger.error("The model is already fitted. You cannot change the parameters in the fitted model.")
            return self

        for key, value in iteritems(params):
            if key in self._init_params.keys():
                self._init_params[key] = value

        # super class
        super().set_params(**params)

        return self

    def fit(self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]]) -> "TabPFNClassifierMother":
        """
        Fit the model to the given data.

        Args:
            X : ArrayLike
                input to fit.
            y : ArrayLike
                target value to fit.
        """
        if isinstance(X, list):
            X = np.array(X).reshape(-1, 1)
            module_logger.warning(f"X is given as list type. It is converted  into np.array with a shape of {X.shape}.")
        if isinstance(y, list):
            y = np.array(y)
            module_logger.warning(f"y is given as list type. It is converted  into np.array with a shape of {y.shape}.")

        self._check_input_type(X, y)
        self._check_input_shape(X.shape)

        # fit the TabPFNClassifier model
        self._is_fitted = True
        super().fit(X, y)

        return self


class TabPFNEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts TabPFN embeddings using a k-fold approach for training data.

    Uses k-fold cross-validation to generate out-of-fold embeddings for training data to avoid data leakage.
    Uses a single model trained on all data to generate embeddings for new data.

    Parameters
    ----------
    task : {'classification', 'regression'}, default='classification'
        The type of task to perform. This is ignored when 'model' is given.
    device : str, default='cpu'
        Device to run the TabPFN model on ('cpu' or 'cuda').
    n_folds : int, default=5
        Number of folds for cross-validation when generating training embeddings.
    use_kfold : bool, default=True
        Whether to use k-fold strategy for training embeddings.
        If False, falls back to standard fitting.
    random_state : int, default=None
        Random state for k-fold splitting.
    embedding_column_name : str, default='tabpfnembedding'
        Name of the column containing the embedding vectors in the output DataFrame.
        When return_separate_columns=True, used as the prefix for column names.
    return_separate_columns : bool, default=False
        If True, returns each embedding dimension as a separate column.
        If False, returns embeddings as vectors in a single column.
    model : TabPFNClassifierMother or TabPFNRegressorMother, default=None
        A pre-fitted TabPFN model instance. If provided, this model will be used instead
        of fitting a new one, and the k-fold scheme will be skipped for training data.
    ignore_pretraining_limits : bool, default=True
        When True, bypasses TabPFN's restriction on the number of features (default 500).
        Set to False to enforce the pretraining limits.
    **kwargs
        Additional parameters passed to TabPFNClassifierMother or TabPFNRegressorMother.
    """

    def __init__(
        self,
        model_type: Literal["classification", "regression"] = "classification",
        device: str = "cpu",
        n_folds: int = 5,
        use_kfold: bool = True,
        random_state: Optional[int] = None,
        embedding_column_name: str = "tabpfnembedding",
        return_separate_columns: bool = True,
        model: Optional[Union[TabPFNClassifierMother, TabPFNRegressorMother]] = None,
        ignore_pretraining_limits: bool = True,
        **kwargs: Any,
    ):
        self.model_type: Literal["classification", "regression"] = model_type
        self.device: str = device
        self.n_folds: int = n_folds
        self.use_kfold: bool = use_kfold
        self.random_state: Optional[int] = random_state
        self.embedding_column_name: str = embedding_column_name

        # TabPFNMother model and model parameters
        self.return_separate_columns: bool = return_separate_columns
        self.ignore_pretraining_limits: bool = ignore_pretraining_limits
        self.kwargs: Dict[str, Any] = kwargs
        self.model: Optional[Union[TabPFNClassifierMother, TabPFNRegressorMother]] = model
        # otherwise, the model will be fitted every time new data is given
        self.pre_fitted = self.model is not None

        # For embeddings
        self.input_features_: Optional[List[str]] = None
        self.train_embeddings_: Optional[np.ndarray] = None
        self.train_index_: Optional[pd.Index] = None
        self._embedding_dim: Optional[int] = None

        if self.pre_fitted and self.use_kfold:
            raise ValueError(
                """Cannot use k-fold fitting when a pre-trained model is already given.
                Please set either use_kfold=False or model=None."""
            )

        # set random state
        if self.random_state is not None:
            self._set_random_state()

    def _set_random_state(self) -> None:
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _get_best_embeddings(
        self,
        embeddings: np.ndarray,
        y: Optional[Union[np.ndarray, pd.Series]],
    ) -> Tuple[np.ndarray, np.intp]:
        """
        Get the best ensemble model and return embeddings from only the selected model
        """
        if len(embeddings.shape) != 3:
            raise ValueError(
                "The input embedding must have three dimensions (n_estimators, n_samples, dim_embedding_space)"
            )

        # evaluate each embedding space
        ensemble_scores: List[float]
        if self.model_type == "classification":
            # calculate the shilouette score to find the best embedding space
            ensemble_scores = [
                silhouette_score(embeddings[i, :, :], y, random_state=self.random_state)
                for i in range(embeddings.shape[0])
            ]
        else:
            # calculate the out-of-bag (R2) score for regression
            regressor = RandomForestRegressor(oob_score=True, random_state=self.random_state)
            ensemble_scores = list()
            for i in range(embeddings.shape[0]):
                cloned_regressor = clone(regressor)
                cloned_regressor.fit(embeddings[i, :, :], y)
                ensemble_scores.append(np.mean(cloned_regressor.oob_score_))

        best_idx: np.intp = np.argmax(np.array(ensemble_scores))
        return embeddings[best_idx, :, :], best_idx

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None,
        only_best_embeddings: bool = False,
    ):
        """
        Fit the TabPFN model to the training data and generate TabPFN embeddings.

        Uses k-fold cross-validation to generate out-of-fold embeddings for training data,
        or fits a single model if a pre-fitted model is provided or use_kfold=False.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like or Series of shape (n_samples,), default=None
            Target values.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples. Used for group-based cross-validation. When groups are given, classification
            models perform StratifiedGroupKFold whereas regression
            models perform GroupKFold.
        only_best_embeddings : bool, default=False
            If True, selects the best embedding space from multiple ensembles.

        Returns
        -------
        self : object
            Returns self.
        """
        # Store feature names if input is DataFrame
        is_dataframe: bool = isinstance(X, pd.DataFrame)
        self.train_index_ = X.index if is_dataframe else None

        if is_dataframe:
            self.input_features_ = X.columns.tolist()
            X_array: np.ndarray = X.values
        else:
            self.input_features_ = None
            X_array: np.ndarray = X

        if y is None:
            raise ValueError("TabPFN requires target values for fitting when no prefitted_model is provided.")

        # Convert y to numpy array if it's a pandas Series
        y_array: np.ndarray = y.values if isinstance(y, pd.Series) else y

        # Convert groups to numpy array if it's a pandas Series
        if groups is not None and isinstance(groups, pd.Series):
            groups_array: np.ndarray = groups.values
        else:
            groups_array: Optional[np.ndarray] = groups

        # If we were provided a prefitted model, use it directly
        if self.pre_fitted:
            module_logger.info(
                "A pre-fitted model has been given. The new data will not be used for fitting the model."
            )
            self.train_embeddings_ = self.model.get_embeddings(X_array)
            self._embedding_dim = self.train_embeddings_.shape[1]
        else:
            # Otherwise, follow the original fitting process

            # Select appropriate TabPFN model based on task
            if self.model_type == "classification":
                model_class: Type[Union[TabPFNClassifierMother, TabPFNRegressorMother]] = TabPFNClassifierMother
            elif self.model_type == "regression":
                model_class: Type[Union[TabPFNClassifierMother, TabPFNRegressorMother]] = TabPFNRegressorMother
            else:
                raise ValueError(f"Invalid task: {self.model_type}. Use 'classification' or 'regression'.")

            # Generate k-fold embeddings for training data if requested
            if self.use_kfold and (X_array.shape[0] >= self.n_folds):
                logging.info(f"Run K-fold to fit the model and obtain the embeddings with K={self.n_folds}")

                fold_iterator: Iterator[Tuple[np.ndarray, np.ndarray]]
                # Select CV strategy based on task and whether groups are provided
                if self.model_type == "classification":
                    if groups_array is not None:
                        # Use StratifiedGroupKFold for classification with groups
                        n_splits: int = min(self.n_folds, len(np.unique(groups_array)))
                        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                        fold_iterator = kf.split(X_array, y_array, groups=groups_array)
                    else:
                        # Use StratifiedKFold for classification without groups
                        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                        fold_iterator = kf.split(X_array, y_array)
                else:  # regression
                    if groups_array is not None:
                        # Use GroupKFold for regression with groups
                        n_splits: int = min(self.n_folds, len(np.unique(groups_array)))
                        kf = GroupKFold(n_splits=n_splits)
                        fold_iterator = kf.split(X_array, y_array, groups=groups_array)
                    else:
                        # Use regular KFold for regression without groups
                        n_splits = self.n_folds
                        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                        fold_iterator: Iterator[Tuple[np.ndarray, np.ndarray]] = kf.split(X_array)

                # initialise the array to save embeddings
                embedding_chunks: List[Tuple[int, np.ndarray]] = []

                for train_idx, val_idx in fold_iterator:
                    # Train TabPFN model on training fold
                    fold_model: Union[TabPFNClassifierMother, TabPFNRegressorMother] = model_class(
                        device=self.device,
                        ignore_pretraining_limits=self.ignore_pretraining_limits,
                        **self.kwargs,
                    )
                    fold_model.fit(X_array[train_idx], y_array[train_idx])

                    # Generate embeddings for validation fold
                    val_embeddings: np.ndarray = fold_model.get_embeddings(X_array[val_idx])

                    # Store embeddings with their original indices
                    for i, sample_idx in zip(val_idx, range(val_embeddings.shape[1])):
                        embedding_chunks.append((int(i), val_embeddings[:, sample_idx, :]))

                # Sort by original indices and extract embeddings
                embedding_chunks.sort(key=lambda x: x[0])
                self.train_embeddings_ = np.concatenate(
                    [np.expand_dims(emb, axis=1) for _, emb in embedding_chunks], axis=1
                )

            else:
                # Train main model on all data for transform of new data
                self.model = model_class(
                    device=self.device,
                    ignore_pretraining_limits=self.ignore_pretraining_limits,
                    **self.kwargs,
                )
                self.model.fit(X_array, y_array)

                if self.use_kfold:
                    module_logger.warning(
                        f"Number of samples ({X_array.shape[0]}) is less than n_folds ({self.n_folds}). "
                        "Using main model embeddings for training data without K-Fold."
                    )

                # Store main model embeddings for training data
                self.train_embeddings_ = self.model.get_embeddings(X_array)

        # Collapse feature vectors from different estimators (3D -> 2D) (avg)
        # So each sample gets 1D feature vecture
        if len(self.train_embeddings_.shape) == 3:
            if only_best_embeddings:
                self.train_embeddings_, self.best_estimator_idx = self._get_best_embeddings(
                    self.train_embeddings_, y_array
                )
            else:
                self.train_embeddings_ = np.concatenate(self.train_embeddings_, axis=1)  # concat along the sample axis
                self.best_estimator_idx = None

        self._embedding_dim = self.train_embeddings_.shape[1]
        return self

    def transform(self, X: pd.DataFrame, only_best_embeddings: bool = False) -> pd.DataFrame:
        """
        Transform new data into TabPFN embeddings using the fitted model.

        For training data, returns stored out-of-fold embeddings.
        For new data, generates embeddings using the model trained on all training data.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data to transform.
        only_best_embeddings : bool, default=False
            If True, uses only the best embedding space if available.

        Returns
        -------
        X_transformed : DataFrame
            DataFrame containing TabPFN embeddings for each sample.
        """
        if self.model is None:
            raise ValueError("Transformer hasn't been fitted. Call 'fit' first or provide a pre-fitted model.")

        index = X.index
        if self.input_features_ is not None:
            # Ensure the DataFrame has the same columns as the training data
            if not all(feature in X.columns for feature in self.input_features_):
                missing_features: set = set(self.input_features_) - set(X.columns)
                raise ValueError(f"Features {missing_features} used for training are missing")
            # Use the same column order as during training
            X_array: np.ndarray = X[self.input_features_].values
        else:
            X_array: np.ndarray = X.values

        # Get embeddings for new data using the main model
        embeddings = self.model.get_embeddings(X_array)
        # collapse the additional column caused by estimators (avg)
        if len(embeddings.shape) == 3:
            if only_best_embeddings:
                if self.best_estimator_idx is None:
                    raise AttributeError(
                        "best_estimator_idx is None."
                        "It seems to be the model was not fitted with only_best_embeddings=True"
                    )
                embeddings = embeddings[self.best_estimator_idx, :, :]
            else:
                embeddings = np.concatenate(embeddings, axis=1)

        # Create output DataFrame based on return_separate_columns setting
        if self.return_separate_columns:
            # Create DataFrame with separate columns
            column_names: list[str] = [f"{self.embedding_column_name}_{i}" for i in range(embeddings.shape[1])]
            if index is not None:
                df: pd.DataFrame = pd.DataFrame(embeddings, columns=column_names, index=index)
            else:
                df = pd.DataFrame(embeddings, columns=column_names)
        else:
            # Create a DataFrame with a single column containing the embedding vectors
            if index is not None:
                df = pd.DataFrame({self.embedding_column_name: list(embeddings)}, index=index)
            else:
                df = pd.DataFrame({self.embedding_column_name: list(embeddings)})

        return df

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None,
        only_best_embeddings: bool = False,
    ) -> pd.DataFrame:
        """
        Fit the transformer and transform the training data in one step.

        Efficiently computes and returns TabPFN embeddings for the training data.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like or Series of shape (n_samples,), default=None
            Target values.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples. Used for group-based
            cross-validation. When groups are given, classification models
            perform StratifiedGroupKFold whereas regression models perform
            GroupKFold.
        only_best_embeddings : bool, default=False
            If True, selects the best embedding space from multiple ensembles.

        Returns
        -------
        X_transformed : DataFrame
            DataFrame containing TabPFN embeddings for each sample.
        """

        self.fit(X, y, groups, only_best_embeddings)

        # Create output DataFrame based on return_separate_columns setting
        if self.return_separate_columns:
            column_names: list[str] = [
                f"{self.embedding_column_name}_{i}" for i in range(self.train_embeddings_.shape[1])
            ]
            # Create DataFrame with separate columns
            if self.train_index_ is not None:
                df: pd.DataFrame = pd.DataFrame(self.train_embeddings_, columns=column_names, index=self.train_index_)
            else:
                df = pd.DataFrame(self.train_embeddings_, columns=column_names)
        else:
            # Create a DataFrame with a single column containing the embedding vectors
            if self.train_index_ is not None:
                df = pd.DataFrame(
                    {self.embedding_column_name: list(self.train_embeddings_)},
                    index=self.train_index_,
                )
            else:
                df = pd.DataFrame({self.embedding_column_name: list(self.train_embeddings_)})
        return df

    def get_feature_names_out(self) -> np.ndarray:
        """
        Get output feature names for the TabPFN embeddings.

        Returns
        -------
        feature_names_out : ndarray of str
            Names of the output embedding features.
        """
        if self.model is None:
            raise ValueError("Transformer hasn't been fitted. Call 'fit' first or provide a pre-fitted model.")

        if self.return_separate_columns:
            # Return separate column names for each embedding dimension
            feature_names: List[str] = [
                f"{self.embedding_column_name}_{i}" for i in range(cast(int, self._embedding_dim))
            ]

            return np.array(feature_names)
        else:
            # Return a single column name
            return np.array([self.embedding_column_name])
