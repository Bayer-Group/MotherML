import logging
import typing
from typing import Any, Optional

import numpy as np
import pandas as pd
from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor
from optuna.trial import Trial
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor

from mother.ml import AbstractMotherPipeline

try:
    from sklearn.utils._fit_context import _fit_context
except ImportError:
    # For older sklearn versions
    def _fit_context(prefer_skip_nested_validation=True):
        def decorator(func):
            return func

        return decorator


from sklearn.metrics import ndcg_score
from sklearn.utils.validation import check_is_fitted

import mother.errors as errors
import mother.ml.properties as props

module_logger: logging.Logger = logging.getLogger(__name__)


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert scores to 1-based ranks, with rank 1 assigned to the highest score."""
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def scores_matrix_to_ranks(score_matrix: np.ndarray) -> np.ndarray:
    """Convert score matrix columns to per-ensemble 1-based ranks."""
    arr = np.asarray(score_matrix)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D score_matrix, got {arr.ndim}D.")
    return np.column_stack([scores_to_ranks(arr[:, ensemble_idx]) for ensemble_idx in range(arr.shape[1])]).astype(
        float
    )


def default_loss_function(
    model_type: props.ModelType = "classification_binary",
    target_type: props.TargetType = "single_target",
) -> str:
    if model_type == "classification_binary":
        if target_type == "single_target":
            module_logger.debug("Using 'Logloss' as loss function.")
            return "Logloss"

        elif target_type == "multi_target":
            module_logger.debug("Using 'MultiLogloss' as loss function.")
            return "MultiLogloss"

    elif model_type == "classification_multiclass":
        if target_type == "single_target":
            module_logger.debug("Using 'MultiClass' as loss function.")
            return "MultiClass"

        elif target_type == "multi_target":
            raise NotImplementedError("Loss function not known for 'multi_target' and 'classification_multiclass'")

    elif model_type == "regression":
        if target_type == "single_target":
            module_logger.info("Using 'RMSE' as loss function.")
            return "RMSE"

        elif target_type == "multi_target":
            module_logger.info("Using 'MultiRMSEWithMissingValues' as loss function.")
            return "MultiRMSEWithMissingValues"

    elif model_type == "ranking":
        return "YetiRank"

    raise NotImplementedError("Loss function not implemented for given model and target type combination")


def signed_percentiles_independent(
    vector: typing.Union[np.ndarray, typing.List],
) -> np.ndarray:
    """
    Transform a vector into percentiles independently for negative and positive values,
    while keeping zeros as zeros.

    # Example vector
    vector = np.array([-10, -5, 0, 5, 10])

    # Transform the vector into percentiles
    percentile_vector = signed_percentiles_independent(vector)

    print("Original vector:", vector)
    print("Transformed percentiles:", percentile_vector)

    Original vector: [-10  -5   0   5  10]
    Transformed percentiles: [-1.  -0.5  0.   0.5  1. ]

    Parameters
    ----------
    vector : np.ndarray
        Input vector with both negative and positive values.

    Returns
    -------
    np.ndarray
        A vector of percentiles with the same signs as the original vector, with zeros unchanged.
    """
    # Ensure the input is a NumPy array
    vector_array: np.ndarray = np.array(vector, dtype=float)

    # Handle zeros explicitly
    is_zero: np.ndarray = vector_array == 0

    # Separate positive and negative values
    positive_mask: np.ndarray = vector_array > 0
    negative_mask: np.ndarray = vector_array < 0

    # Compute percentiles for positive and negative values independently
    positive_percentiles: np.ndarray = np.zeros_like(vector_array)
    negative_percentiles: np.ndarray = np.zeros_like(vector_array)

    if np.any(positive_mask):
        positive_ranks: np.ndarray = rankdata(vector_array[positive_mask], method="average")
        positive_percentiles[positive_mask] = positive_ranks / len(vector_array[positive_mask])

    if np.any(negative_mask):
        negative_ranks: np.ndarray = rankdata(-vector_array[negative_mask], method="average")
        negative_percentiles[negative_mask] = -negative_ranks / len(vector_array[negative_mask])

    # Combine positive and negative percentiles, keeping zeros as zeros
    result: np.ndarray = positive_percentiles + negative_percentiles
    result[is_zero] = 0

    return result


def calc_range_tree_depth(x: pd.DataFrame, min_depth: int = 1, max_depth: int = 16) -> typing.Tuple[int, int]:
    """
    Get the tree depth range for a given data frame.
    If the passed max_depth is <= the min depth then the min_depth is set to max_depth.
    ----------
    x : pd.DataFrame
        The input DataFrame from which the maximum tree depth will be calculated.
    min_depth: int
        The minimum depth for a tree
    max_depth: int
        The maximum depth for a tree
    Returns
    -------
    int
        The maximum tree depth
    Example
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({"A": [1, 2, 3], "B": ["cat", "dog", "cat"], "C": [0.1, 0.2, 0.3]})
    >>> calc_max_tree_depth(data, min_depth=2, max_depth=16)
    (2, 2)
    """
    module_logger.info("Calculating tree depth range based on the number of rows in the data frame")
    calculated_min_depth: int = max_depth
    calculated_max_depth: int = max_depth
    if max_depth > min_depth:
        calculated_max_depth = range(min_depth, max_depth + 1)[
            np.argmin([np.abs(2**i - x.shape[0]) for i in range(min_depth, max_depth + 1)])
        ]
        calculated_min_depth = min_depth
    module_logger.debug(f"Calculated tree depth range: [{calculated_min_depth}, {calculated_max_depth}]")
    return calculated_min_depth, calculated_max_depth


def depth_to_leaves_for_lossguide(min_depth: int, max_depth: int) -> typing.Tuple[int, int]:
    """
    Function to transform a max_depth range to a max_leaves range for lossguided tree tuning:
    -----------
    Args:
        min_depth (int): minimum tree depth
        max_depth (int) : maximum tree depth
    Example:
        min_leaves, max_leaves  = depth_to_leaves_for_lossguide(5,10)
    Returns:
        (min_leaves, max_leaves): min_leaves and max_leaves
    """
    if min_depth > max_depth:
        raise errors.ConfigurationError("min_depth should be less than max_depth")

    min_number_of_leaves: int = np.max([2, (2**min_depth) / 2 - 1])
    max_number_of_leaves: int = np.min([64, (2**max_depth) / 2 + 1])
    module_logger.debug(f"Transformed depth range to leaves range: {min_number_of_leaves}, {max_number_of_leaves}")
    return (min_number_of_leaves, max_number_of_leaves)


def mean_absolute_error_multi_na(
    true: typing.Union[pd.DataFrame, np.ndarray],
    pred: typing.Union[pd.DataFrame, np.ndarray],
) -> float:
    """
    Calculates the mean absolute error (MAE) between true and predicted values for multi-output regression,
    handling missing values (NaNs) by skipping them in the computation.

    Parameters
    ----------
    true : array-like or DataFrame
        Ground truth (correct) target values. Can be a pandas DataFrame, numpy array, or similar structure.
    pred : array-like or DataFrame
        Estimated target values. Must have the same shape as `true`.

    Returns
    -------
    float
        The mean absolute error averaged across all outputs, ignoring NaN values.
    """
    true_df: pd.DataFrame = pd.DataFrame(true)
    pred_df: pd.DataFrame = pd.DataFrame(pred)

    diff: pd.DataFrame = true_df - pred_df
    diff = diff.abs()
    mean_diff: pd.Series = diff.mean(axis=0, skipna=True)

    return mean_diff.mean(skipna=True)


def get_tree_depth(model: CatBoost) -> int:
    """
    Get the tree depth from a CatBoost model, checking for 'max_depth' first,
    then 'depth', defaulting to 6 if neither is available.

    Args:
        model: A CatBoost model (CatBoostRegressor, CatBoostClassifier, or CatBoostRanker)

    Returns:
        int: The tree depth value
    """
    params: dict = model.get_params()
    return params.get("max_depth", params.get("depth", 6))


class OrdinalLabelBinarizer(BaseEstimator, TransformerMixin):
    """Custom label binarizer for multi-label classification.

    This transformer can convert ordinal classes into a multitask binary indicator format
    and vice versa.

    Parameters
    ----------
    classes : array-like, optional
        Indicates the labels that will be used for binarization. If not provided,
        classes will be inferred from the data during fitting.
    sparse_output : bool, default=False
        Set to True to return sparse matrices from transform method.

    Attributes
    ----------
    ordinal_classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.
    """

    def __init__(self, sparse_output: bool = False) -> None:
        self.sparse_output = sparse_output

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y: typing.Union[np.ndarray, typing.List]) -> "OrdinalLabelBinarizer":
        """Fit the label sets binarizer, storing :term:`ordinal_classes_`.

        Parameters
        ----------
        y : array-like
            A set of ordinal labels for each sample.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        y_array: np.ndarray = np.asarray(y)

        # Handle case where TransformedTargetRegressor passes 2D array
        if y_array.ndim > 1:
            y_array = y_array.ravel()

        self.ordinal_classes_: np.ndarray = np.array(sorted(set(y_array)))

        # the lowest ordinal class is considered background so best remove
        self.all_classes_: np.ndarray = self.ordinal_classes_.copy()
        if self.ordinal_classes_.shape[0] > 1:
            self.ordinal_classes_ = self.ordinal_classes_[1:]
        else:
            # For single class, we can't do ordinal encoding
            raise ValueError("Ordinal encoding requires at least 2 classes, but only 1 class found in the input data")

        # Ensure we have a valid ordinal_classes_ attribute for check_is_fitted
        if self.ordinal_classes_.shape[0] == 0:
            raise ValueError("No classes found in the input data")

        return self

    def transform(self, y: typing.Union[np.ndarray, typing.List]) -> typing.Union[np.ndarray, csr_matrix]:
        """Transform the given label sets.

        Parameters
        ----------
        y : array-like
            A set of ordinal labels for each sample.

        Returns
        -------
        y_transformed
        """

        check_is_fitted(self)

        y_array: np.ndarray = np.asarray(y)

        # Handle case where TransformedTargetRegressor passes 2D array
        if y_array.ndim > 1:
            y_array = y_array.ravel()

        thresholds: np.ndarray = np.asarray(self.ordinal_classes_)
        y_transformed: np.ndarray = (y_array[:, None] >= thresholds).astype(int)

        if self.sparse_output:
            y_transformed_sparse: csr_matrix = csr_matrix(y_transformed)
            return y_transformed_sparse
        else:
            y_transformed_dense: np.ndarray = np.array(y_transformed, dtype=int)
            return y_transformed_dense

    def inverse_transform(self, yt: typing.Union[np.ndarray, csr_matrix]) -> np.ndarray:
        """Transform the given indicator matrix into label sets.

        Parameters
        ----------
        yt : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            A matrix containing only 1s ands 0s.

        Returns
        -------
        y_original : list of floats
            The reconstructed ordinal values for each sample.
        """
        check_is_fitted(self)

        yt_processed: typing.Union[np.ndarray, csr_matrix] = yt
        # Handle sparse matrices using issparse for sklearn-style robustness
        if issparse(yt_processed):
            yt_processed = yt_processed.toarray()
        yt_array: np.ndarray = np.asarray(yt_processed)

        yt_binary: np.ndarray = (yt_array > 0.5).astype(int)

        y_original: typing.List = [self.all_classes_[sum(i)] for i in yt_binary]
        return np.array(y_original)

    def get_feature_names_out(self, input_features: typing.Optional[typing.List[str]] = None) -> np.ndarray:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of shape (n_features_out,), dtype=str
            Transformed feature names. Returns the ordinal classes as strings.
        """
        check_is_fitted(self)
        feature_names: typing.List[str] = [f"ordinal_{cls}" for cls in self.ordinal_classes_]
        return np.array(feature_names, dtype=str)


class MotherTransformedTargetRegressor(TransformedTargetRegressor, AbstractMotherPipeline):
    """
    Enhanced TransformedTargetRegressor that automatically delegates
    method calls to the underlying regressor model.

    This allows direct access to methods from the underlying regressor
    without needing to go through regressor_ attribute.
    """

    def __init__(
        self,
        regressor: typing.Union[BaseEstimator, AbstractMotherPipeline],
        transformer: Optional[TransformerMixin] = None,
        func: Optional[Any] = None,
        inverse_func: Optional[Any] = None,
        check_inverse: bool = False,
    ) -> None:
        super().__init__(
            regressor=regressor,
            transformer=transformer,
            func=func,
            inverse_func=inverse_func,
            check_inverse=check_inverse,
        )

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Get hyperparameter space from the underlying regressor.
        """
        base_regressor = self.regressor
        if hasattr(base_regressor, "get_hyperparameter_space"):
            return base_regressor.get_hyperparameter_space(X, y, trial, prefix + "regressor__")
        else:
            raise AttributeError(
                f"Underlying regressor {type(base_regressor).__name__} does not have get_hyperparameter_space method"
            )

    def default_parameters(self, prefix: str = "") -> dict:
        """
        Get default parameters from the underlying regressor, passing the prefix argument through.
        """
        base_regressor = self.regressor
        if hasattr(base_regressor, "default_parameters"):
            return base_regressor.default_parameters(prefix + "regressor__")
        else:
            raise AttributeError(
                f"Underlying regressor {type(base_regressor).__name__} does not have default_parameters method"
            )


@typing.overload
def get_virtual_prediction(
    X: pd.DataFrame,
    model: CatBoostRanker,
    virtual_ensembles_count: int = 10,
    thread_count: int = 1,
) -> typing.Tuple[pd.DataFrame, np.ndarray]: ...


@typing.overload
def get_virtual_prediction(
    X: pd.DataFrame,
    model: typing.Union[CatBoostRegressor, CatBoostClassifier],
    virtual_ensembles_count: int = 10,
    thread_count: int = 1,
) -> pd.DataFrame: ...


def get_virtual_prediction(
    X: pd.DataFrame,
    model: typing.Union[
        CatBoostRegressor,
        CatBoostClassifier,
        CatBoostRanker,
    ],
    virtual_ensembles_count: int = 10,
    thread_count: int = 1,
) -> typing.Union[pd.DataFrame, typing.Tuple[pd.DataFrame, np.ndarray]]:
    """
    Generates virtual ensemble predictions using CatBoost's built-in uncertainty prediction.

    Please, find the details about uncertainty calculation for different
    Catboost models here: https://catboost.ai/docs/en/references/uncertainty

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the features for prediction.
    model : typing.Union[CatBoostRegressor, CatBoostClassifier, CatBoostRanker]
        Trained CatBoost model (regressor, classifier, or ranker).
    virtual_ensembles_count : int, optional
        Number of virtual ensembles to use for uncertainty estimation. Default is 10.
    thread_count : int, optional
        Number of threads. Use -1 to use all available threads. Default is 1.
    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, np.ndarray]
        For ``CatBoostRegressor`` and ``CatBoostClassifier``: a DataFrame with columns
        ``mean_predictions``, ``knowledge_uncertainty``, ``data_uncertainty``, ``total_uncertainty``.

        For ``CatBoostRanker``: a tuple ``(uncertainty_df, raw_scores)`` where
        ``uncertainty_df`` has the same columns as above and ``raw_scores`` is a float
        array of shape ``(n_samples, virtual_ensembles_count)`` with the raw score from
        each virtual ensemble.  Callers can apply ``scores_to_ranks`` column-wise to
        ``raw_scores`` to obtain per-ensemble rank distributions.

    Notes:
    ---------
    Regression:
        CatBoost returns predictive uncertainties as variances. We convert them to standard deviations
        to improve interpretability. For regression models not using 'RMSEWithUncertainty' loss,
        only mean_predictions and knowledge_uncertainty will have values.
    Classification:
        Uncertainties are entropy-based as defined by CatBoost and are returned without transformation.
    Ranking:
        Returns mean raw ranking scores and epistemic uncertainty from virtual ensembles.
        Knowledge uncertainty is the standard deviation of the virtual-ensemble scores.
    """

    if (
        not isinstance(virtual_ensembles_count, (int, np.integer))
        or isinstance(virtual_ensembles_count, bool)
        or virtual_ensembles_count < 1
    ):
        raise ValueError(f"virtual_ensembles_count must be a positive integer, got {virtual_ensembles_count}.")
    virtual_ensembles_count = int(virtual_ensembles_count)

    if not isinstance(model, (CatBoostRegressor, CatBoostClassifier, CatBoostRanker)):
        raise ValueError("model must be an instance of CatBoostRegressor, CatBoostClassifier, or CatBoostRanker.")

    module_logger.info("Using catboost's builtin uncertainty prediction")

    if isinstance(model, CatBoostRanker):
        # For rankers, VirtEnsembles gives one score column per virtual ensemble
        # (shape: n_samples × virtual_ensembles_count), which lets us compute
        # mean and std directly without a separate TotalUncertainty call.
        raw_scores = np.asarray(
            model.virtual_ensembles_predict(
                X,
                prediction_type="VirtEnsembles",
                ntree_end=0,
                virtual_ensembles_count=virtual_ensembles_count,
                thread_count=thread_count,
                verbose=None,
            )
        )

        # CatBoost may return either (n_samples, n_ensembles) or
        # (n_samples, n_ensembles, 1). Handle both shapes safely.
        if raw_scores.ndim == 3 and raw_scores.shape[-1] == 1:
            raw_scores = raw_scores[..., 0]
        elif raw_scores.ndim != 2:
            raise ValueError(
                "Unexpected shape returned by CatBoostRanker.virtual_ensembles_predict("
                f"prediction_type='VirtEnsembles'): {raw_scores.shape}. "
                "Expected (n_samples, n_ensembles) or (n_samples, n_ensembles, 1)."
            )

        ddof = 1 if virtual_ensembles_count > 1 else 0
        knowledge_uncertainty = np.nan_to_num(raw_scores.std(axis=1, ddof=ddof), nan=0.0)

        uncertainty_df = pd.DataFrame(
            {
                "mean_predictions": raw_scores.mean(axis=1),
                "knowledge_uncertainty": knowledge_uncertainty,
                "data_uncertainty": None,
                "total_uncertainty": None,
            },
            index=X.index,
        )

        return uncertainty_df, raw_scores

    virtual_prediction = model.virtual_ensembles_predict(
        X,
        prediction_type="TotalUncertainty",
        ntree_end=0,
        virtual_ensembles_count=virtual_ensembles_count,
        thread_count=thread_count,
        verbose=None,
    )

    if isinstance(model, CatBoostRegressor):
        if model._init_params["loss_function"] == "RMSEWithUncertainty":
            """
            For RMSEWithUncertainty, the total uncertainty is calculated
            by summing up the knowledge and data uncertainty
            """
            return pd.DataFrame(
                {
                    "mean_predictions": virtual_prediction[:, 0],
                    "knowledge_uncertainty": np.sqrt(virtual_prediction[:, 1]),
                    "data_uncertainty": np.sqrt(virtual_prediction[:, 2]),
                    "total_uncertainty": np.sqrt(virtual_prediction[:, 1:3].sum(axis=1)),
                },
                index=X.index,
            )
        else:
            return pd.DataFrame(
                {
                    "mean_predictions": virtual_prediction[:, 0],
                    "knowledge_uncertainty": np.sqrt(virtual_prediction[:, 1]),
                    "data_uncertainty": None,
                    "total_uncertainty": None,
                },
                index=X.index,
            )
    elif isinstance(model, CatBoostClassifier):
        # only for binary classification
        data_uncertainty = virtual_prediction[:, 0]
        total_uncertainty = virtual_prediction[:, 1]
        knowledge_uncertainty = total_uncertainty - data_uncertainty

        return pd.DataFrame(
            {
                "mean_predictions": None,
                "knowledge_uncertainty": knowledge_uncertainty,
                "data_uncertainty": data_uncertainty,
                "total_uncertainty": total_uncertainty,
            },
            index=X.index,
        )

    else:
        raise ValueError("The model must inherit CatBoostClassifier, CatBoostRegressor, or CatBoostRanker")


def single_group_rank_pred(
    y_pred: list | pd.DataFrame | np.ndarray,
    y: list | pd.DataFrame | np.ndarray,
    as_frame: bool = False,
) -> tuple | pd.DataFrame:
    """Returns the true and predicted ranks for a dataframe
    Args:
        y_pred (list | pd.DataFrame | np.ndarray): predicted values
        y (list | pd.DataFrame | np.ndarray): true values
        as_frame(bool): if True returns a dataframe table instead of 2 single arrays

    Returns:
        tuple: numpy array of true rankings, numpy array of predicted rankings or a dataFrame
    """
    if not isinstance(y, pd.DataFrame):
        y = pd.Series(y)
    true_ranking = y.sort_values(ascending=False).index.tolist()  # type: ignore
    data = {
        "entry index": y.index,
        "target value": y,
        "predicted score": y_pred,
    }
    result = pd.DataFrame(data)
    true_ranking_dict = {
        entry: true_ranking.index(entry) if entry in true_ranking else -1 for entry in data["entry index"]
    }

    result = result.sort_values(by=["predicted score"], ascending=False)
    if isinstance(y_pred, list):
        result["predicted rank"] = range(len(y_pred))
    else:
        result["predicted rank"] = range(y_pred.shape[0])
    result["true rank"] = result["entry index"].map(true_ranking_dict)
    if as_frame:
        return result
    else:
        true_ranks = result["true rank"].to_numpy()
        pred_ranks = result["predicted rank"].to_numpy()
        return true_ranks, pred_ranks


def avg_ndcg_score(
    y: list | pd.DataFrame | np.ndarray,
    y_pred: list | pd.DataFrame | np.ndarray,
    groups: list | np.ndarray,
    k: int,
    verbose: bool = False,
) -> float:
    """calculates the average ndcg score of a model's prediction across all groups
       in a dataframe
    Args:
        y (list | pd.DataFrame | np.ndarray): true target values
        y_pred (list | pd.DataFrame | np.ndarray): predicted values
        groups (list): group indices for each entry
        k (int): number of elements to consider for ndcg calculation. Consistent across all groups
        verbose (bool, optional): If True prints the true ranks of each group as well as the ndcg
        score of each group. Defaults to False.

    Returns:
        float: single val average ndcg score across all groups
    """
    group_dict = {}
    ndcg_list = []
    if verbose:
        print(f"Group codes: {groups}")
    for true_val, pred_val, group_index in zip(y, y_pred, groups):
        if group_index not in group_dict:
            group_dict[group_index] = ([], [])
        group_dict[group_index][0].append(true_val)
        group_dict[group_index][1].append(pred_val)
    if verbose:
        print(f"dictionary of groups: {group_dict}")
    for true_list, preds_list in group_dict.values():
        true_ranks, pred_ranks = single_group_rank_pred(preds_list, true_list)
        if verbose:
            print(true_ranks)
        ndcg_list.append(ndcg_score([true_ranks], [pred_ranks], k=k))  # type: ignore
    if verbose:
        print(f"List of every group ndcg score: {ndcg_list}")
    return np.average(ndcg_list)


def topk_rank_disagreement(
    rank_ensembles: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute per-item probability of disagreement about top-k membership.

    For each sample, let ``p`` be the fraction of ensemble members that place
    it in the top-k positions. This returns ``2 * p * (1 - p)``, the probability
    that two independently selected ensemble members disagree about membership.
    A value of 0.0 means all ensembles agree, whether the item is in or out of
    the top-k; larger values indicate less stable membership.

    Parameters
    ----------
    rank_ensembles : np.ndarray, shape (n_samples, n_ensembles)
        Rank matrix where each column contains 1-based ranks from one virtual ensemble.
    k : int
        Number of top positions to consider.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Pairwise disagreement probability for each item's top-k membership
        (range [0, 0.5]).
    """
    arr = np.asarray(rank_ensembles)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D rank_ensembles, got {arr.ndim}D.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if k > arr.shape[0]:
        raise ValueError(f"k must be <= the number of items ({arr.shape[0]}), got {k}.")
    if not np.isfinite(arr).all():
        raise ValueError("rank_ensembles must contain only finite values.")
    if (arr < 1).any() or (arr > arr.shape[0]).any():
        raise ValueError(f"rank_ensembles must be 1-based values in [1, {arr.shape[0]}].")
    in_topk = arr <= k
    membership_probability = in_topk.mean(axis=1)
    return 2.0 * membership_probability * (1.0 - membership_probability)


def topk_score_variance(
    score_ensembles: np.ndarray,
    k: int,
    reference_ranks: typing.Optional[np.ndarray] = None,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute score variance for items in the top-k of a reference ranking.

    Identifies items that are in the top-k of a reference ranking (or the
    ensemble-mean ranking if not provided) and returns their score variance
    across ensemble members.

    Parameters
    ----------
    score_ensembles : np.ndarray, shape (n_samples, n_ensembles)
        Raw score matrix from virtual ensembles.
    k : int
        Number of top positions to select.
    reference_ranks : np.ndarray, shape (n_samples,), optional
        1-based ranks to determine which items are top-k. If None, ranks
        are derived from the mean of ``score_ensembles``.

    Returns
    -------
    topk_mask : np.ndarray, shape (n_samples,), dtype bool
        Boolean mask indicating items in the top-k.
    variances : np.ndarray, shape (n_samples,)
        Score variance across ensembles (0.0 for items outside top-k).
    """
    arr = np.asarray(score_ensembles, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D score_ensembles, got {arr.ndim}D.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if k > arr.shape[0]:
        raise ValueError(f"k must be <= the number of items ({arr.shape[0]}), got {k}.")
    if not np.isfinite(arr).all():
        raise ValueError("score_ensembles must contain only finite values.")

    if reference_ranks is None:
        mean_scores = arr.mean(axis=1)
        reference_ranks = scores_to_ranks(mean_scores)
    reference_ranks = np.asarray(reference_ranks).reshape(-1)
    if len(reference_ranks) != arr.shape[0]:
        raise ValueError("reference_ranks must have one entry per score_ensembles row.")
    if not np.isfinite(reference_ranks).all():
        raise ValueError("reference_ranks must contain only finite values.")
    if (reference_ranks < 1).any() or (reference_ranks > arr.shape[0]).any():
        raise ValueError(f"reference_ranks must be 1-based values in [1, {arr.shape[0]}].")

    topk_mask = reference_ranks <= k
    variances = np.zeros(arr.shape[0])
    if topk_mask.any():
        ddof = 1 if arr.shape[1] > 1 else 0
        variances[topk_mask] = arr[topk_mask].var(axis=1, ddof=ddof)
    return topk_mask, variances


def groupwise_topk_analysis(
    uncertainty_df: pd.DataFrame,
    score_ensembles: np.ndarray,
    group_ids: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """Perform groupwise top-k uncertainty analysis across ranking groups.

    For each ranking group, computes:
    - ``topk_disagreement_prob``: fraction of ensembles disagreeing about each item's top-k membership
    - ``topk_score_var``: score variance for items in the top-k (0 otherwise)
    - ``topk_member``: whether the item is in the consensus top-k (based on mean rank)

    Parameters
    ----------
    uncertainty_df : pd.DataFrame
        Output from ``predict_uncertainty`` containing ``mean_predictions`` and
        ``knowledge_uncertainty`` columns.
    score_ensembles : np.ndarray, shape (n_samples, n_ensembles)
        Raw score matrix from virtual ensembles (obtained from ``get_virtual_prediction``).
    group_ids : np.ndarray
        Group IDs aligned with rows.
    k : int
        Number of top positions to analyse.

    Returns
    -------
    pd.DataFrame
        Copy of ``uncertainty_df`` with added columns ``topk_disagreement_prob`` (the
        disagreement probability),
        ``topk_score_var``, and ``topk_member``.
    """
    result = uncertainty_df.copy()
    result["topk_disagreement_prob"] = 0.0
    result["topk_score_var"] = 0.0
    result["topk_member"] = False

    group_arr = np.asarray(group_ids).reshape(-1)
    score_arr = np.asarray(score_ensembles, dtype=float)
    if score_arr.ndim != 2:
        raise ValueError(f"Expected 2D score_ensembles, got {score_arr.ndim}D.")
    if len(uncertainty_df) != len(group_arr) or score_arr.shape[0] != len(group_arr):
        raise ValueError("uncertainty_df, score_ensembles, and group_ids must have the same number of rows.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if not np.isfinite(score_arr).all():
        raise ValueError("score_ensembles must contain only finite values.")

    for g in pd.unique(group_arr):
        idx = np.flatnonzero(group_arr == g)
        if k > len(idx):
            raise ValueError(f"k must be <= the number of items in every group; group {g!r} has {len(idx)} items.")
        group_scores = score_arr[idx]
        group_ranks = scores_matrix_to_ranks(group_scores)

        result.iloc[idx, result.columns.get_loc("topk_disagreement_prob")] = topk_rank_disagreement(group_ranks, k)

        # Consensus ordering is the mean of the per-ensemble ranks; negated because
        # rank 1 is best while scores_to_ranks expects higher-is-better.
        mean_ranks = group_ranks.mean(axis=1)
        ref_ranks = scores_to_ranks(-mean_ranks)
        _, var = topk_score_variance(group_scores, k, reference_ranks=ref_ranks)
        result.iloc[idx, result.columns.get_loc("topk_score_var")] = var
        result.iloc[idx, result.columns.get_loc("topk_member")] = ref_ranks <= k

    return result
