import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import sklearn.base as skl_base
from sklearn import set_config as skl_set_config
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold

import mother.ml.models.m_catboost as m_catboost
from mother.ml import utils
from mother.ml.core import AbstractMotherPipeline
from mother.ml.models.m_catboost import CatboostRankerMother
from mother.pipeline_utils import mother_cv

pytestmark = pytest.mark.usefixtures("preserve_metadata_routing")


# ---------------------------------------------------------------------------
# scores_to_ranks unit tests (no model needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scores, expected",
    [
        # Descending: higher score -> better (lower) rank
        (np.array([0.5, 0.9, 0.1, 0.7]), np.array([3, 1, 4, 2])),
        (np.array([10, 5, 20, 15]), np.array([3, 4, 1, 2])),
        # Identical scores - returns identical ranks
        (np.array([1.0, 1.0, 1.0]), np.array([1, 1, 1])),
        # Single value
        (np.array([5.0]), np.array([1])),
        # Negative values: least negative = highest = rank 1
        (np.array([-1.0, -5.0, -3.0]), np.array([1, 3, 2])),
        # Mixed positive/negative
        (np.array([1.0, -1.0, 0.0, 2.0]), np.array([2, 4, 3, 1])),
        # Zeros with one positive and one negative
        (np.array([0.0, 1.0, -1.0, 0.0]), np.array([2, 1, 3, 2])),
    ],
)
def test_scores_to_ranks(scores, expected):
    """scores_to_ranks uses descending order: highest score -> rank 1."""
    result = utils.scores_to_ranks(scores)
    np.testing.assert_array_equal(result, expected)


def test_scores_to_ranks_preserves_input_order():
    """Output positions correspond to input positions (not sorted positions)."""
    scores = np.array([0.3, 0.7, 0.1, 0.9, 0.5])
    ranks = utils.scores_to_ranks(scores)

    assert len(ranks) == len(scores)
    assert set(ranks) == set(range(1, len(scores) + 1))

    # 0.9 -> 1, 0.7 -> 2, 0.5 -> 3, 0.3 -> 4, 0.1 -> 5
    np.testing.assert_array_equal(ranks, np.array([4, 2, 5, 1, 3]))


def test_scores_matrix_to_ranks_matches_columnwise_scores_to_ranks():
    score_matrix = np.array(
        [
            [0.4, 0.1, 0.3],
            [0.9, 0.7, 0.8],
            [0.1, 0.3, 0.2],
            [0.6, 0.9, 0.4],
        ]
    )

    expected = np.column_stack(
        [utils.scores_to_ranks(score_matrix[:, i]) for i in range(score_matrix.shape[1])]
    ).astype(float)
    got = utils.scores_matrix_to_ranks(score_matrix)
    np.testing.assert_array_equal(got, expected)


def test_scores_matrix_to_ranks_rejects_non_2d_input():
    with pytest.raises(ValueError, match="Expected 2D score_matrix"):
        _ = utils.scores_matrix_to_ranks(np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# Helpers shared by the model tests
# ---------------------------------------------------------------------------


def _make_ranker_data(n_samples: int = 100, n_features: int = 5, n_groups: int = 10, seed: int = 42):
    """Return (X, y, group_ids) suitable for fitting CatboostRankerMother."""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=seed)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")

    group_ids = np.repeat(np.arange(n_groups), int(np.ceil(n_samples / n_groups)))[:n_samples]
    # Sort by group so CatBoost receives contiguous groups
    order = np.argsort(group_ids, kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    group_ids = group_ids[order]
    return X, y, group_ids


def _fit_ranker(num_trees: int = 20) -> tuple:
    skl_set_config(enable_metadata_routing=True)
    X, y, groups = _make_ranker_data()
    model = CatboostRankerMother(target_type="single_target", num_trees=num_trees).set_fit_request(group_id="group_id")
    model.fit(X, y, group_id=groups, verbose=False)
    return model, X, y, groups


@pytest.fixture
def fitted_ranker_data():
    model, X, y, groups = _fit_ranker()
    mask = groups == 0
    return {
        "model": model,
        "X": X,
        "y": y,
        "groups": groups,
        "X_group": X[mask],
    }


@pytest.fixture
def mock_ranker_uncertainty_inputs():
    mock_X = pd.DataFrame(
        {
            "feature_0": [0.1, 0.2, 0.3, 0.4],
            "feature_1": [1.0, 0.5, -0.2, 0.0],
        },
        index=["a", "b", "c", "d"],
    )
    mock_helper_output = pd.DataFrame(
        {
            "mean_predictions": [0.4, 0.9, 0.1, 0.6],
            "knowledge_uncertainty": [0.25, 0.5, 0.1, 0.75],
            "data_uncertainty": [None, None, None, None],
            "total_uncertainty": [None, None, None, None],
        },
        index=mock_X.index,
    )
    return {
        "mock_X": mock_X,
        "mock_helper_output": mock_helper_output,
    }


# ---------------------------------------------------------------------------
# CatboostRankerMother.predict tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_returns_scores_by_default(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    scores = model.predict(X_group)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(X_group)
    assert np.issubdtype(scores.dtype, np.floating)


@pytest.mark.slow
def test_predict_ranks_returns_1_based_integers(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    ranks = model.predict(X_group, use_ranks=True)
    assert isinstance(ranks, np.ndarray)
    assert len(ranks) == len(X_group)
    n = len(X_group)
    # scores_to_ranks uses dense ranking, so tied scores (which tree models can
    # produce) share a rank instead of consuming every integer up to n -- only
    # the 1..n range is guaranteed, not the full set.
    assert set(ranks) <= set(range(1, n + 1))
    assert ranks.min() >= 1
    assert ranks.max() <= n


@pytest.mark.slow
def test_predict_ranks_highest_score_gets_rank_1(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    scores = model.predict(X_group)
    ranks = model.predict(X_group, use_ranks=True)
    best_idx = int(np.argmax(scores))
    assert ranks[best_idx] == 1


@pytest.mark.slow
def test_predict_normalize_by_group_size(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    n = len(X_group)
    ranks_norm = model.predict(X_group, use_ranks=True, normalize_by_group_size=True)
    ranks_raw = model.predict(X_group, use_ranks=True)
    assert (ranks_norm > 0).all()
    assert (ranks_norm <= 1).all()
    np.testing.assert_array_almost_equal(ranks_norm, np.round(ranks_raw / n, 4))


@pytest.mark.slow
def test_predict_normalize_no_effect_without_ranks(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    scores_plain = model.predict(X_group)
    scores_norm = model.predict(X_group, normalize_by_group_size=True)
    np.testing.assert_array_equal(scores_plain, scores_norm)


@pytest.mark.slow
def test_predict_groupwise_ranks_for_multiple_groups(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X = fitted_ranker_data["X"]
    groups = fitted_ranker_data["groups"]
    mask = np.isin(groups, [0, 1])
    X_multi = X[mask]
    groups_multi = groups[mask]

    scores = model.predict(X_multi)
    ranks = m_catboost.ranker_predict_for_groups(model, X_multi, groups_multi, use_ranks=True)

    expected = np.empty(len(scores), dtype=float)
    for group in np.unique(groups_multi):
        idx = np.flatnonzero(groups_multi == group)
        expected[idx] = utils.scores_to_ranks(scores[idx])

    np.testing.assert_array_equal(ranks, expected)


# ---------------------------------------------------------------------------
# CatboostRankerMother.predict_uncertainty tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_output_schema(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group)
    assert isinstance(result, pd.DataFrame)
    for col in (
        "pred",
        "mean_predictions",
        "knowledge_uncertainty",
        "data_uncertainty",
        "total_uncertainty",
    ):
        assert col in result.columns
    assert len(result) == len(X_group)


@pytest.mark.slow
def test_pred_column_matches_predict_scores_when_use_ranks_false(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group)
    expected_pred = model.predict(X_group, use_ranks=False)
    np.testing.assert_array_equal(result["pred"].values, expected_pred)


@pytest.mark.slow
def test_pred_column_matches_predict_ranks_when_use_ranks_true(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group, use_ranks=True)
    expected_pred = model.predict(X_group, use_ranks=True)
    np.testing.assert_array_equal(result["pred"].values, expected_pred)


@pytest.mark.slow
def test_knowledge_uncertainty_non_negative(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group)
    assert (result["knowledge_uncertainty"] >= 0).all()


@pytest.mark.slow
def test_total_uncertainty_is_none(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group)
    assert result["total_uncertainty"].isna().all()


@pytest.mark.slow
def test_data_uncertainty_is_none(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group)
    assert result["data_uncertainty"].isna().all()


@pytest.mark.slow
def test_uncertainty_for_opt_returns_only_knowledge_uncertainty_on_fitted_model(
    fitted_ranker_data,
):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result_opt = model.predict_uncertainty(X_group, uncertainty_for_opt=True)
    result_full = model.predict_uncertainty(X_group)
    assert list(result_opt.columns) == ["knowledge_uncertainty"]
    pd.testing.assert_series_equal(result_opt["knowledge_uncertainty"], result_full["knowledge_uncertainty"])


@pytest.mark.slow
def test_normalize_by_group_size_scales_uncertainty(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    n = len(X_group)
    result_norm = model.predict_uncertainty(X_group, normalize_by_group_size=True)
    result_raw = model.predict_uncertainty(X_group)
    np.testing.assert_array_almost_equal(
        result_norm["knowledge_uncertainty"].values,
        np.round(result_raw["knowledge_uncertainty"].values / n, 4),
    )
    np.testing.assert_array_almost_equal(
        result_norm["mean_predictions"].values,
        np.round(result_raw["mean_predictions"].values / n, 4),
    )


def test_normalize_by_group_size_scales_quantiles(mock_ranker_uncertainty_inputs):
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    mock_helper_output = mock_ranker_uncertainty_inputs["mock_helper_output"]

    raw_scores = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [1.0, 1.0, 1.0, 1.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=float,
    )

    with patch(
        "mother.ml.models.m_catboost.utils.get_virtual_prediction",
        return_value=(mock_helper_output, raw_scores),
    ):
        with patch.object(model, "predict", return_value=np.array([0.2, 0.9, 0.1, 0.6])):
            result_raw = model.predict_uncertainty(mock_X, return_quantiles=True)
            result_norm = model.predict_uncertainty(mock_X, normalize_by_group_size=True, return_quantiles=True)

    n = len(mock_X)
    for col in ("score_q25", "score_q50", "score_q75"):
        np.testing.assert_allclose(
            result_norm[col].to_numpy(),
            np.round(result_raw[col].to_numpy() / n, 4),
        )


@pytest.mark.slow
def test_normalize_uncertainty_in_0_1_range(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group, use_ranks=True, normalize_by_group_size=True)
    assert (result["knowledge_uncertainty"] >= 0).all()
    assert (result["knowledge_uncertainty"] <= 1).all()


@pytest.mark.slow
def test_index_preserved(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group)
    pd.testing.assert_index_equal(result.index, X_group.index)


@pytest.mark.slow
def test_n_ensembles_one_has_finite_knowledge_uncertainty(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group, n_ensembles=1)
    assert np.isfinite(result["knowledge_uncertainty"].to_numpy()).all()


def test_virtual_ensemble_helper_is_called_with_forwarded_parameters(
    mock_ranker_uncertainty_inputs,
):
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    mock_helper_output = mock_ranker_uncertainty_inputs["mock_helper_output"]

    dummy_raw = np.zeros((4, 7))
    with patch(
        "mother.ml.models.m_catboost.utils.get_virtual_prediction",
        return_value=(mock_helper_output, dummy_raw),
    ) as mocked:
        with patch.object(model, "predict", return_value=np.array([0.1, 0.2, 0.3, 0.4])):
            _ = model.predict_uncertainty(mock_X, n_ensembles=7, n_threads=3)

    mocked.assert_called_once()
    _, called_kwargs = mocked.call_args
    pd.testing.assert_frame_equal(called_kwargs["X"], mock_X)
    assert called_kwargs["model"] is model
    assert called_kwargs["virtual_ensembles_count"] == 7
    assert called_kwargs["thread_count"] == 3


def test_virtual_ensemble_mean_scores_remain_scores_by_default(
    mock_ranker_uncertainty_inputs,
):
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    mock_helper_output = mock_ranker_uncertainty_inputs["mock_helper_output"]

    dummy_raw = np.zeros((4, 10))
    with patch(
        "mother.ml.models.m_catboost.utils.get_virtual_prediction",
        return_value=(mock_helper_output, dummy_raw),
    ):
        with patch.object(model, "predict", return_value=np.array([0.2, 0.9, 0.1, 0.6])):
            result = model.predict_uncertainty(mock_X)

    np.testing.assert_allclose(result["mean_predictions"].to_numpy(), np.array([0.4, 0.9, 0.1, 0.6]))
    np.testing.assert_array_equal(result["pred"].to_numpy(), np.array([0.2, 0.9, 0.1, 0.6]))
    np.testing.assert_allclose(result["knowledge_uncertainty"].to_numpy(), np.array([0.25, 0.5, 0.1, 0.75]))
    assert result["total_uncertainty"].isna().all()
    assert result["data_uncertainty"].isna().all()


def test_use_ranks_converts_mean_and_uncertainty_to_rank_scale(
    mock_ranker_uncertainty_inputs,
):
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]

    raw_scores = np.array(
        [
            [0.4, 0.1, 0.3],
            [0.9, 0.7, 0.8],
            [0.1, 0.3, 0.2],
            [0.6, 0.9, 0.4],
        ]
    )

    mock_helper_output = pd.DataFrame(
        {
            "mean_predictions": raw_scores.mean(axis=1),
            "knowledge_uncertainty": [9.0, 9.0, 9.0, 9.0],
            "data_uncertainty": [None, None, None, None],
            "total_uncertainty": [None, None, None, None],
        },
        index=mock_X.index,
    )

    with patch(
        "mother.ml.models.m_catboost.utils.get_virtual_prediction",
        return_value=(mock_helper_output, raw_scores),
    ):
        with patch.object(model, "predict", return_value=np.array([2, 1, 4, 3])):
            result = model.predict_uncertainty(mock_X, use_ranks=True)

    rank_ensembles = np.array(
        [
            [3, 4, 3],
            [1, 2, 1],
            [4, 3, 4],
            [2, 1, 2],
        ],
        dtype=float,
    )
    expected_mean_rank = rank_ensembles.mean(axis=1)
    np.testing.assert_allclose(result["mean_predictions"].to_numpy(), expected_mean_rank)

    expected_rank_uncertainty = rank_ensembles.std(axis=1, ddof=1)
    np.testing.assert_allclose(result["knowledge_uncertainty"].to_numpy(), expected_rank_uncertainty)


@pytest.mark.parametrize("use_ranks", [False, True], ids=["score_mode", "rank_mode"])
def test_predict_uncertainty_combined_new_parameters_ranker_calls_mother_cv(use_ranks):
    """Exercise combined ranker uncertainty kwargs through mother_cv forwarding."""

    class DummyRankEstimator(skl_base.BaseEstimator, AbstractMotherPipeline):
        def __init__(self):
            self.kwarg_calls = []

        def get_hyperparameter_space(self, X, y, trial, prefix: str = ""):
            return {}

        def fit(self, X, y):
            return self

        def predict_uncertainty(
            self,
            X,
            n_ensembles: int = 10,
            n_threads: int = 1,
            use_ranks: bool = False,
            uncertainty_for_opt: bool = False,
            normalize_by_group_size: bool = False,
            return_quantiles: bool = False,
        ):
            self.kwarg_calls.append(
                {
                    "n_ensembles": n_ensembles,
                    "n_threads": n_threads,
                    "use_ranks": use_ranks,
                    "uncertainty_for_opt": uncertainty_for_opt,
                    "normalize_by_group_size": normalize_by_group_size,
                    "return_quantiles": return_quantiles,
                }
            )

            n = len(X)
            if uncertainty_for_opt:
                return pd.DataFrame({"knowledge_uncertainty": np.zeros(n)}, index=X.index)

            pred = np.arange(1, n + 1, dtype=float) if use_ranks else np.linspace(0.1, 0.9, num=n)
            result = pd.DataFrame(
                {
                    "pred": pred,
                    "mean_predictions": pred,
                    "knowledge_uncertainty": np.zeros(n, dtype=float),
                    "data_uncertainty": [None] * n,
                    "total_uncertainty": [None] * n,
                },
                index=X.index,
            )
            if return_quantiles:
                result["score_q25"] = pred
                result["score_q50"] = pred
                result["score_q75"] = pred
            return result

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.DataFrame({"target": np.arange(12, dtype=float)})
    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    estimator = DummyRankEstimator()

    result = mother_cv(
        estimator,
        cv=cv,
        X=X,
        y=y,
        n_ensembles=4,
        n_threads=2,
        use_ranks=use_ranks,
        normalize_by_group_size=True,
        return_quantiles=True,
        uncertainty_for_opt=False,
    )

    assert len(estimator.kwarg_calls) == cv.get_n_splits()
    for call in estimator.kwarg_calls:
        assert call["n_ensembles"] == 4
        assert call["n_threads"] == 2
        assert call["use_ranks"] is use_ranks
        assert call["normalize_by_group_size"] is True
        assert call["return_quantiles"] is True
        assert call["uncertainty_for_opt"] is False

    expected_prefixed_cols = {
        "pred_target",
        "pred_mean_predictions",
        "pred_knowledge_uncertainty",
        "pred_score_q25",
        "pred_score_q50",
        "pred_score_q75",
    }
    assert expected_prefixed_cols.issubset(set(result.columns))
    # mother_cv drops columns that are all NaN.
    assert "pred_data_uncertainty" not in result.columns
    assert "pred_total_uncertainty" not in result.columns
    if use_ranks:
        assert np.allclose(
            result["pred_target"].to_numpy(),
            result["pred_target"].to_numpy().astype(int),
        )
    else:
        assert not np.allclose(
            result["pred_target"].to_numpy(),
            result["pred_target"].to_numpy().astype(int),
        )


def test_uncertainty_for_opt_returns_only_knowledge_uncertainty(
    mock_ranker_uncertainty_inputs,
):
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    mock_helper_output = mock_ranker_uncertainty_inputs["mock_helper_output"]

    dummy_raw = np.zeros((4, 10))
    with patch(
        "mother.ml.models.m_catboost.utils.get_virtual_prediction",
        return_value=(mock_helper_output, dummy_raw),
    ):
        with patch.object(model, "predict", return_value=np.array([0.2, 0.9, 0.1, 0.6])):
            result = model.predict_uncertainty(mock_X, uncertainty_for_opt=True)

    assert list(result.columns) == ["knowledge_uncertainty"]
    np.testing.assert_allclose(result["knowledge_uncertainty"].to_numpy(), np.array([0.25, 0.5, 0.1, 0.75]))


def test_invalid_n_ensembles_raises(mock_ranker_uncertainty_inputs):
    model = CatboostRankerMother()
    with pytest.raises(ValueError):
        model.predict_uncertainty(mock_ranker_uncertainty_inputs["mock_X"], n_ensembles=0)


def test_predict_uncertainty_rejects_unknown_kwargs(mock_ranker_uncertainty_inputs):
    model = CatboostRankerMother()
    with pytest.raises(TypeError):
        model.predict_uncertainty(mock_ranker_uncertainty_inputs["mock_X"], foo="bar")


def test_predict_uncertainty_use_ranks_groupwise_via_helper(
    mock_ranker_uncertainty_inputs,
):
    """ranker_predict_uncertainty_for_groups calls predict_uncertainty per group."""
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    group_ids = np.array([0, 0, 1, 1])

    group0_result = pd.DataFrame(
        {
            "pred": [1, 2],
            "mean_predictions": [1.0, 2.0],
            "knowledge_uncertainty": [0.1, 0.2],
            "data_uncertainty": [None, None],
            "total_uncertainty": [None, None],
        },
        index=["a", "b"],
    )
    group1_result = pd.DataFrame(
        {
            "pred": [2, 1],
            "mean_predictions": [2.0, 1.0],
            "knowledge_uncertainty": [0.3, 0.4],
            "data_uncertainty": [None, None],
            "total_uncertainty": [None, None],
        },
        index=["c", "d"],
    )

    call_results = [group0_result, group1_result]
    with patch.object(model, "predict_uncertainty", side_effect=call_results) as mock_pu:
        result = m_catboost.ranker_predict_uncertainty_for_groups(model, mock_X, group_ids, use_ranks=True)

    assert mock_pu.call_count == 2
    pd.testing.assert_index_equal(result.index, mock_X.index)
    np.testing.assert_allclose(result["knowledge_uncertainty"].to_numpy(), [0.1, 0.2, 0.3, 0.4])


def test_ranker_predict_uncertainty_for_groups_matches_per_group_manual(
    mock_ranker_uncertainty_inputs,
):
    """ranker_predict_uncertainty_for_groups result equals manual per-group call."""
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    group_ids = np.array([0, 0, 1, 1])

    mock_out = pd.DataFrame(
        {
            "pred": [1.0, 2.0],
            "mean_predictions": [0.4, 0.9],
            "knowledge_uncertainty": [0.25, 0.5],
            "data_uncertainty": [None, None],
            "total_uncertainty": [None, None],
        },
    )

    def _per_group_side_effect(X_group, **kw):
        out = mock_out.copy()
        out.index = X_group.index
        return out

    with patch.object(model, "predict_uncertainty", side_effect=_per_group_side_effect):
        from_helper = m_catboost.ranker_predict_uncertainty_for_groups(model, mock_X, group_ids, use_ranks=True)

    expected = pd.concat(
        [
            _per_group_side_effect(mock_X.iloc[[0, 1]]),
            _per_group_side_effect(mock_X.iloc[[2, 3]]),
        ]
    ).loc[mock_X.index]

    pd.testing.assert_frame_equal(from_helper, expected)


def test_ranker_predict_uncertainty_for_groups_preserves_duplicate_index_order(
    mock_ranker_uncertainty_inputs,
):
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"].copy()
    mock_X.index = ["duplicate", "duplicate", "other", "other"]
    group_ids = np.array([0, 0, 1, 1])

    def _per_group_side_effect(X_group, **kw):
        return pd.DataFrame({"value": X_group.iloc[:, 0].to_numpy()})

    with patch.object(model, "predict_uncertainty", side_effect=_per_group_side_effect):
        result = m_catboost.ranker_predict_uncertainty_for_groups(model, mock_X, group_ids, use_ranks=True)

    expected = pd.DataFrame({"value": mock_X.iloc[:, 0].to_numpy()}, index=mock_X.index)
    pd.testing.assert_frame_equal(result, expected)


def test_ranker_predict_for_groups_rejects_missing_group_id(mock_ranker_uncertainty_inputs):
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    group_ids = np.array([0, np.nan, 1, 1])

    with pytest.raises(ValueError, match="group_id must not contain missing values"):
        m_catboost.ranker_predict_for_groups(model, mock_X, group_ids)


def test_ranker_predict_for_groups_score_mode_calls_predict_once(mock_ranker_uncertainty_inputs):
    """Raw scores don't depend on group boundaries, so use_ranks=False must call
    predict() once for the whole dataset instead of once per group."""
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    group_ids = np.array([0, 0, 1, 1])
    expected_scores = np.array([0.4, 0.9, 0.1, 0.6])

    with patch.object(model, "predict", return_value=expected_scores) as mock_predict:
        result = m_catboost.ranker_predict_for_groups(model, mock_X, group_ids, use_ranks=False)

    mock_predict.assert_called_once()
    call_args, call_kwargs = mock_predict.call_args
    pd.testing.assert_frame_equal(call_args[0], mock_X)
    assert call_kwargs["use_ranks"] is False
    np.testing.assert_array_equal(result, expected_scores)


def test_ranker_predict_uncertainty_for_groups_score_mode_calls_predict_uncertainty_once(
    mock_ranker_uncertainty_inputs,
):
    """Score-mode uncertainty (use_ranks=False, normalize_by_group_size=False) doesn't
    depend on group boundaries, so it must call predict_uncertainty() once for the
    whole dataset instead of once per group."""
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    group_ids = np.array([0, 0, 1, 1])
    expected = pd.DataFrame(
        {
            "pred": [1.0, 2.0, 3.0, 4.0],
            "mean_predictions": [1.0, 2.0, 3.0, 4.0],
            "knowledge_uncertainty": [0.1, 0.2, 0.3, 0.4],
            "data_uncertainty": [None, None, None, None],
            "total_uncertainty": [None, None, None, None],
        },
        index=mock_X.index,
    )

    with patch.object(model, "predict_uncertainty", return_value=expected) as mock_pu:
        result = m_catboost.ranker_predict_uncertainty_for_groups(model, mock_X, group_ids)

    mock_pu.assert_called_once()
    call_args, _ = mock_pu.call_args
    pd.testing.assert_frame_equal(call_args[0], mock_X)
    pd.testing.assert_frame_equal(result, expected)


def test_ranker_predict_uncertainty_for_groups_use_ranks_still_loops_per_group(mock_ranker_uncertainty_inputs):
    """use_ranks=True must still call predict_uncertainty per group, since rank
    conversion depends on group boundaries."""
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    group_ids = np.array([0, 0, 1, 1])

    def _per_group_side_effect(X_group, **kw):
        return pd.DataFrame({"value": X_group.iloc[:, 0].to_numpy()}, index=X_group.index)

    with patch.object(model, "predict_uncertainty", side_effect=_per_group_side_effect) as mock_pu:
        m_catboost.ranker_predict_uncertainty_for_groups(model, mock_X, group_ids, use_ranks=True)

    assert mock_pu.call_count == 2


def test_ranker_predict_uncertainty_for_groups_normalize_by_group_size_still_loops_per_group(
    mock_ranker_uncertainty_inputs,
):
    """normalize_by_group_size=True must still call predict_uncertainty per group,
    since the normalization divisor depends on group size."""
    model = CatboostRankerMother()
    mock_X = mock_ranker_uncertainty_inputs["mock_X"]
    group_ids = np.array([0, 0, 1, 1])

    def _per_group_side_effect(X_group, **kw):
        return pd.DataFrame({"value": X_group.iloc[:, 0].to_numpy()}, index=X_group.index)

    with patch.object(model, "predict_uncertainty", side_effect=_per_group_side_effect) as mock_pu:
        m_catboost.ranker_predict_uncertainty_for_groups(model, mock_X, group_ids, normalize_by_group_size=True)

    assert mock_pu.call_count == 2


# ---------------------------------------------------------------------------
# CatboostRankerMother.predict_uncertainty return_raw tests
# ---------------------------------------------------------------------------
# Test return_raw parameter
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_return_raw_integration(fitted_ranker_data):
    """Test return_raw parameter returns correct output for topk analysis."""
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]

    # Test default behavior
    result_default = model.predict_uncertainty(X_group, n_ensembles=10)
    assert isinstance(result_default, pd.DataFrame)

    # Test return_raw=True
    uncertainty_df, raw_scores = model.predict_uncertainty(X_group, n_ensembles=10, return_raw=True)

    assert isinstance(uncertainty_df, pd.DataFrame)
    assert isinstance(raw_scores, np.ndarray)
    assert raw_scores.shape == (len(X_group), 10)

    # Verify mean_predictions matches raw scores mean
    np.testing.assert_allclose(
        uncertainty_df["mean_predictions"].values,
        raw_scores.mean(axis=1),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# CatboostRankerMother get_params / set_params / clone / pickle tests
# ---------------------------------------------------------------------------


def test_get_params_contains_all_custom_keys():
    model = CatboostRankerMother()
    params = model.get_params()
    for key in (
        "target_type",
        "model_type",
        "tune_pairwise_type",
        "tune_boosting_type",
        "tune_tree_structure_type",
        "tune_loss_function",
        "top",
        "max_pairs",
    ):
        assert key in params, f"'{key}' missing from get_params()"


def test_get_params_default_values():
    model = CatboostRankerMother()
    params = model.get_params()
    assert params["model_type"] == "ranking"
    assert params["target_type"] == "single_target"
    assert not params["tune_pairwise_type"]
    assert not params["tune_boosting_type"]
    assert params["tune_tree_structure_type"]
    assert params["tune_loss_function"]
    assert params["top"] == 0
    assert params["max_pairs"] is None


def test_init_normalizes_pairlogit_parameter_separator_for_max_pairs():
    model = CatboostRankerMother(loss_function="PairLogit;foo=1", max_pairs=125)

    assert model.get_params()["loss_function"] == "PairLogit:foo=1;max_pairs=125"


def test_init_normalizes_pairlogit_parameter_separator_without_max_pairs():
    """Separator normalization must not be gated on appending a max_pairs suffix."""
    model = CatboostRankerMother(loss_function="PairLogit;foo=1")

    assert model.get_params()["loss_function"] == "PairLogit:foo=1"

    model = CatboostRankerMother(loss_function="PairLogit;max_pairs=75")

    assert model.get_params()["loss_function"] == "PairLogit:max_pairs=75"


def test_init_resolves_none_loss_function_to_wrapper_default():
    """An explicit loss_function=None must be treated the same as omitting it: `None`
    has no string to bake `top`/`max_pairs` into, so passing it straight through to
    CatBoost would silently drop any requested cutoff/pair cap. It must resolve to the
    wrapper's own default loss instead, exactly as if loss_function had not been given."""
    model = CatboostRankerMother(loss_function=None)

    assert model.get_params()["loss_function"] == "YetiRank:mode=Classic"

    model_with_top = CatboostRankerMother(loss_function=None, top=5)

    assert model_with_top.get_params()["loss_function"] == "YetiRank:mode=NDCG;top=5"
    assert model_with_top.top == 5


def test_init_syncs_top_attribute_from_explicit_loss_function_string():
    """If `top` is only defined inside an explicit loss_function string (not via the
    dedicated `top` parameter), the `top` attribute must still reflect that effective
    value -- get_params() must never report a stale/default top that disagrees with
    the loss actually in effect."""
    model = CatboostRankerMother(loss_function="YetiRank:mode=NDCG;top=7")

    assert model.top == 7
    assert model.get_params()["top"] == 7


def test_init_syncs_max_pairs_attribute_from_explicit_loss_function_string():
    model = CatboostRankerMother(loss_function="PairLogit:max_pairs=42")

    assert model.max_pairs == 42
    assert model.get_params()["max_pairs"] == 42


def test_set_params_syncs_top_attribute_from_explicit_loss_function_string():
    model = CatboostRankerMother()

    model.set_params(loss_function="YetiRank:mode=NDCG;top=9")

    assert model.top == 9


def test_set_params_syncs_max_pairs_attribute_from_explicit_loss_function_string():
    model = CatboostRankerMother()

    model.set_params(loss_function="PairLogit:max_pairs=17")

    assert model.max_pairs == 17


def test_set_params_resets_top_when_new_loss_function_omits_it():
    """set_params(loss_function=...) without touching `top` must reset the `top`
    attribute to the disabled default when the new string doesn't define `top` either --
    otherwise get_params()/clone() would keep reporting a stale top that has nothing to
    do with the loss actually in effect (and would conflict when spliced back in)."""
    model = CatboostRankerMother(loss_function="YetiRank:mode=NDCG;top=5")
    assert model.top == 5

    model.set_params(loss_function="YetiRank:mode=Classic")

    assert model.top == 0
    assert model.get_params()["loss_function"] == "YetiRank:mode=Classic"


def test_set_params_resets_max_pairs_when_new_loss_function_omits_it():
    model = CatboostRankerMother(loss_function="PairLogit:max_pairs=17")
    assert model.max_pairs == 17

    model.set_params(loss_function="PairLogit")

    assert model.max_pairs is None
    assert model.get_params()["loss_function"] == "PairLogit"


def test_set_params_resolves_none_loss_function_to_current_loss():
    """An explicit set_params(loss_function=None) must be treated the same as not
    passing loss_function at all: `None` has no string to bake `top`/`max_pairs` into,
    so passing it straight through to CatBoost would silently drop the requested top."""
    model = CatboostRankerMother()

    model.set_params(loss_function=None, top=5)

    assert model.top == 5
    assert model.get_params()["loss_function"] == "YetiRank:mode=NDCG;top=5"


def test_set_params_rejects_conflicting_max_pairs_baked_into_new_loss_function():
    """Even when `max_pairs` isn't part of *this* set_params call, a previously
    configured meaningful `self.max_pairs` must not silently be overridden by a
    different value baked into a newly-supplied loss_function string -- `top`/
    `max_pairs` must still be defined in exactly one place."""
    model = CatboostRankerMother()
    model.set_params(max_pairs=50)

    with pytest.raises(ValueError, match="max_pairs"):
        model.set_params(loss_function="PairLogit:max_pairs=999")

    # The rejected call must not have left the object partially updated.
    assert model.max_pairs == 50


def test_set_params_rejects_conflicting_top_baked_into_new_loss_function():
    model = CatboostRankerMother(top=5)

    with pytest.raises(ValueError, match="top"):
        model.set_params(loss_function="YetiRank:mode=NDCG;top=9")

    assert model.top == 5


def test_set_params_conflict_leaves_top_and_max_pairs_unchanged():
    """A rejected set_params(loss_function=..., top=...) call must not leave `self.top`
    partially updated to the rejected candidate value -- validation must run before any
    attribute is committed, not after."""
    model = CatboostRankerMother(loss_function="YetiRank:mode=NDCG;top=5")

    with pytest.raises(ValueError, match="top"):
        model.set_params(loss_function="YetiRank:mode=NDCG;top=5", top=3)

    assert model.top == 5
    assert model.get_params()["loss_function"] == "YetiRank:mode=NDCG;top=5"


def test_init_does_not_append_zero_max_pairs():
    model = CatboostRankerMother(loss_function="PairLogit", max_pairs=0)

    assert model.get_params()["loss_function"] == "PairLogit"


@pytest.mark.parametrize("max_pairs", [-1, 2.5, True])
def test_init_rejects_invalid_max_pairs_type_or_value(max_pairs):
    with pytest.raises((TypeError, ValueError)):
        CatboostRankerMother(max_pairs=max_pairs)


@pytest.mark.parametrize("top", [-1, 2.5, True, "5"])
def test_init_rejects_invalid_top_type_or_value(top):
    with pytest.raises((TypeError, ValueError)):
        CatboostRankerMother(top=top)


@pytest.mark.parametrize("top", [-1, 2.5, True, "5"])
def test_set_params_rejects_invalid_top_type_or_value(top):
    model = CatboostRankerMother()
    with pytest.raises((TypeError, ValueError)):
        model.set_params(top=top)


def test_set_params_updates_attributes():
    model = CatboostRankerMother()
    model.set_params(
        tune_boosting_type=True,
        tune_loss_function=False,
        tune_pairwise_type=False,
        top=10,
        max_pairs=100,
    )
    assert model.tune_boosting_type
    assert not model.tune_loss_function
    assert model.top == 10
    assert model.max_pairs == 100


def test_set_params_reflected_in_get_params():
    model = CatboostRankerMother()
    model.set_params(tune_loss_function=False, top=5)
    params = model.get_params()
    assert not params["tune_loss_function"]
    assert params["top"] == 5


def test_set_params_updates_supported_loss_parameters_only():
    model = CatboostRankerMother(max_pairs=100)

    model.set_params(loss_function="YetiRank:mode=Classic")
    assert model.get_params()["loss_function"] == "YetiRank:mode=Classic"

    model.set_params(loss_function="PairLogit")
    assert model.get_params()["loss_function"] == "PairLogit"

    model.set_params(loss_function="PairLogit:max_pairs=50")
    assert model.get_params()["loss_function"] == "PairLogit:max_pairs=50"

    model.set_params(max_pairs=200)
    assert model.get_params()["loss_function"] == "PairLogit:max_pairs=200"

    model.set_params(loss_function="QueryRMSE", max_pairs=300)
    assert model.get_params()["loss_function"] == "QueryRMSE"

    model.set_params(loss_function="PairLogit:max_pairs=75", max_pairs=None)
    assert model.get_params()["loss_function"] == "PairLogit:max_pairs=75"

    model.set_params(loss_function="PairLogit;foo=1", max_pairs=125)
    assert model.get_params()["loss_function"] == "PairLogit:foo=1;max_pairs=125"

    model = CatboostRankerMother(loss_function="PairLogit:max_pairs=75", max_pairs=75)
    model.set_params(max_pairs=None)
    assert model.get_params()["loss_function"] == "PairLogit"


def test_set_params_normalizes_separator_when_removing_max_pairs_with_trailing_param():
    """When max_pairs=... is the first parameter followed by another one (e.g.
    'PairLogit:max_pairs=75;foo=1'), setting max_pairs to an invalid value (None)
    removes it but must not leave a stray ';' right after the loss name."""
    model = CatboostRankerMother(loss_function="PairLogit:max_pairs=75;foo=1", max_pairs=75)

    model.set_params(max_pairs=None)

    assert model.get_params()["loss_function"] == "PairLogit:foo=1"


def test_set_params_normalizes_explicit_pairlogit_loss_without_other_changes():
    """An explicit loss_function must be normalized even when top/max_pairs are untouched."""
    model = CatboostRankerMother()

    model.set_params(loss_function="PairLogit;max_pairs=75")

    assert model.get_params()["loss_function"] == "PairLogit:max_pairs=75"


def test_set_params_removes_top_with_colon_separator():
    """Top removal must handle 'YetiRank:top=5' (colon), not just the semicolon form."""
    model = CatboostRankerMother(loss_function="YetiRank:top=5", top=5)

    model.set_params(top=10)

    loss_function = model.get_params()["loss_function"]
    assert loss_function == "YetiRank:mode=NDCG;top=10"


def test_set_params_removes_top_with_colon_separator_and_trailing_param():
    """When top=... is the first parameter and another parameter follows it
    (e.g. 'YetiRank:top=5;mode=NDCG'), removing it must not leave a stray ';'
    right after the loss name -- that separator has to be normalized back to ':'.
    """
    model = CatboostRankerMother(loss_function="YetiRank:top=5;mode=NDCG", top=5)

    model.set_params(top=10)

    loss_function = model.get_params()["loss_function"]
    assert loss_function == "YetiRank:mode=NDCG;top=10"


@pytest.mark.slow
def test_set_params_top_on_yetirank_pairwise_produces_fittable_loss():
    """top is documented for, and works with, YetiRankPairwise (not just plain YetiRank):
    https://catboost.ai/docs/en/concepts/loss-functions-ranking lists `top` under both
    YetiRank and YetiRankPairwise (any mode except Classic). Confirm set_params(top=...)
    on a YetiRankPairwise model produces a loss string CatBoost actually accepts.
    """
    skl_set_config(enable_metadata_routing=True)
    X, y, groups = _make_ranker_data(n_samples=40, n_groups=8)
    model = CatboostRankerMother(
        loss_function="YetiRankPairwise:mode=Classic",
        num_trees=5,
    ).set_fit_request(group_id="group_id")

    model.set_params(top=3)

    assert model.get_params()["loss_function"] == "YetiRankPairwise:mode=NDCG;top=3"
    model.fit(X, y, group_id=groups, verbose=False)  # raises if CatBoost rejects the loss string


@pytest.mark.slow
def test_max_pairs_on_pairlogit_produces_fittable_loss():
    """max_pairs is documented for PairLogit and PairLogitPairwise:
    https://catboost.ai/docs/en/concepts/loss-functions-ranking lists it under both.
    Confirm max_pairs=... actually fits with CatBoost for both loss names."""
    skl_set_config(enable_metadata_routing=True)
    X, y, groups = _make_ranker_data(n_samples=40, n_groups=8)

    model = CatboostRankerMother(loss_function="PairLogit", max_pairs=10, num_trees=5).set_fit_request(
        group_id="group_id"
    )
    assert model.get_params()["loss_function"] == "PairLogit:max_pairs=10"
    model.fit(X, y, group_id=groups, verbose=False)  # raises if CatBoost rejects the loss string

    model = CatboostRankerMother(loss_function="PairLogitPairwise", max_pairs=10, num_trees=5).set_fit_request(
        group_id="group_id"
    )
    assert model.get_params()["loss_function"] == "PairLogitPairwise:max_pairs=10"
    model.fit(X, y, group_id=groups, verbose=False)  # raises if CatBoost rejects the loss string


def test_set_params_top_zero_preserves_ndcg_mode_and_parameters():
    model = CatboostRankerMother(loss_function="YetiRank:mode=NDCG;dcg_denominator=Position;dcg_type=Base", top=5)

    model.set_params(top=0)

    loss_function = model.get_params()["loss_function"]
    assert loss_function == "YetiRank:mode=NDCG;dcg_denominator=Position;dcg_type=Base"


def test_set_params_top_zero_does_not_revert_to_classic_mode():
    """Clearing `top` never reverts `mode` back to Classic automatically -- `top` only
    ever switches mode *to* NDCG (Classic has no concept of a cutoff), never back."""
    model = CatboostRankerMother(loss_function="YetiRank:mode=Classic", top=5)

    model.set_params(top=0)

    assert model.get_params()["loss_function"] == "YetiRank:mode=NDCG"


def test_init_adds_ndcg_mode_when_top_embedded_without_mode():
    """'YetiRank:top=5' with no 'mode=' defaults to CatBoost's Classic mode, where
    'top' has no effect. Constructing with such a string must make the effective
    mode explicit instead of silently leaving an ambiguous/ineffective cutoff."""
    model = CatboostRankerMother(loss_function="YetiRank:top=5", top=5)
    assert model.get_params()["loss_function"] == "YetiRank:top=5;mode=NDCG"

    model = CatboostRankerMother(loss_function="YetiRankPairwise:top=5", top=5)
    assert model.get_params()["loss_function"] == "YetiRankPairwise:top=5;mode=NDCG"


def test_init_rejects_top_defined_in_both_places():
    with pytest.raises(ValueError, match="'top=' is already present"):
        CatboostRankerMother(loss_function="YetiRank:mode=NDCG;top=5", top=3)


def test_init_rejects_top_with_classic_mode():
    """'top' has no effect in Classic mode; an explicit loss_function combining
    both must raise instead of being silently accepted."""
    with pytest.raises(ValueError, match="mode=Classic"):
        CatboostRankerMother(loss_function="YetiRank:mode=Classic;top=5")


def test_set_params_rejects_top_with_classic_mode():
    model = CatboostRankerMother()
    with pytest.raises(ValueError, match="mode=Classic"):
        model.set_params(loss_function="YetiRank:mode=Classic;top=5")


def test_init_rejects_max_pairs_defined_in_both_places():
    with pytest.raises(ValueError, match="'max_pairs=' is already present"):
        CatboostRankerMother(loss_function="PairLogit:max_pairs=50", max_pairs=25)


def test_init_allows_matching_top_defined_in_both_places():
    """Re-supplying the *same* value in both places (e.g. via a get_params() round-trip)
    is not treated as a conflict."""
    model = CatboostRankerMother(loss_function="YetiRank:mode=NDCG;top=5", top=5)

    assert model.get_params()["loss_function"] == "YetiRank:mode=NDCG;top=5"


def test_set_params_rejects_top_defined_in_both_places():
    model = CatboostRankerMother()
    with pytest.raises(ValueError, match="'top=' is already present"):
        model.set_params(loss_function="YetiRank:mode=NDCG;top=5", top=3)


def test_set_params_rejects_max_pairs_defined_in_both_places():
    model = CatboostRankerMother()
    with pytest.raises(ValueError, match="'max_pairs=' is already present"):
        model.set_params(loss_function="PairLogit:max_pairs=50", max_pairs=25)


def test_model_type_must_be_ranking():
    with pytest.raises(ValueError, match="model_type for CatboostRankerMother must be 'ranking'"):
        CatboostRankerMother(model_type="regression")

    model = CatboostRankerMother(model_type="ranking")
    with pytest.raises(ValueError, match="model_type for CatboostRankerMother must be 'ranking'"):
        model.set_params(model_type="regression")


def test_set_params_pairwise_guard_disables_incompatible_combination():
    """set_params must re-apply pairwise incompatibility guard after updates."""
    model = CatboostRankerMother(
        tune_pairwise_type=True,
        tune_tree_structure_type=False,
        tune_boosting_type=False,
    )
    assert model.tune_pairwise_type
    model.set_params(tune_tree_structure_type=True)
    assert not model.tune_pairwise_type


def test_set_params_rejects_invalid_pairwise_loss_configuration():
    model = CatboostRankerMother()
    with pytest.raises(ValueError, match="requires SymmetricTree grow_policy and Plain boosting_type"):
        model.set_params(loss_function="YetiRankPairwise", grow_policy="Lossguide")


def test_set_params_pairwise_loss_disables_incompatible_tuning_flags():
    model = CatboostRankerMother(
        tune_tree_structure_type=True,
        tune_boosting_type=True,
    )

    model.set_params(loss_function="YetiRankPairwise")

    assert not model.tune_tree_structure_type
    assert not model.tune_boosting_type


def test_suggested_params_loss_excludes_pairwise_for_fixed_ordered_boosting():
    class RecordingTrial:
        number = 0

        def __init__(self):
            self.choices = {}

        def suggest_categorical(self, name, choices):
            self.choices[name] = choices
            return choices[0]

        def suggest_float(self, name, low, high, log=False):
            return low

    model = CatboostRankerMother(
        boosting_type="Ordered",
        tune_pairwise_type=True,
        tune_tree_structure_type=False,
        tune_boosting_type=False,
    )
    trial = RecordingTrial()

    model.suggested_params_loss(
        trial,
        {"grow_policy": "SymmetricTree"},
        pd.Series([0.0, 1.0]),
        prefix="",
    )

    assert trial.choices["base_loss"] == ["YetiRank", "PairLogit", "QuerySoftMax"]


def test_suggested_params_loss_with_top_allows_pairwise_when_enabled():
    """top works with both YetiRank and YetiRankPairwise (verified against CatBoost directly:
    YetiRankPairwise:mode=NDCG;top=N fits without error), so pairwise tuning must stay available."""

    class RecordingTrial:
        number = 0

        def __init__(self):
            self.choices = {}

        def suggest_categorical(self, name, choices):
            self.choices[name] = choices
            return choices[0]

    model = CatboostRankerMother(
        top=5,
        tune_pairwise_type=True,
        tune_tree_structure_type=False,
        tune_boosting_type=False,
    )
    trial = RecordingTrial()

    model.suggested_params_loss(
        trial,
        {"grow_policy": "SymmetricTree", "boosting_type": "Plain"},
        pd.Series([0.0, 1.0]),
        prefix="",
    )

    assert trial.choices["base_loss"] == ["YetiRank", "YetiRankPairwise"]


def test_suggested_params_loss_with_top_uses_only_yetirank_when_pairwise_disabled():
    """Without pairwise tuning enabled (or incompatible tree/boosting settings), the
    has_top branch falls back to plain YetiRank without offering a base_loss choice."""

    class RecordingTrial:
        number = 0

        def __init__(self):
            self.choices = {}

        def suggest_categorical(self, name, choices):
            self.choices[name] = choices
            return choices[0]

    model = CatboostRankerMother(top=5, tune_pairwise_type=False)
    trial = RecordingTrial()

    suggested = model.suggested_params_loss(
        trial,
        {"grow_policy": "SymmetricTree", "boosting_type": "Plain"},
        pd.Series([0.0, 1.0]),
        prefix="",
    )

    assert "base_loss" not in trial.choices
    assert suggested["loss_function"].startswith("YetiRank:")


def test_suggested_params_loss_does_not_apply_max_pairs_to_non_pairlogit_losses():
    """max_pairs is only documented for the PairLogit family (PairLogit, PairLogitPairwise);
    CatBoost has no such parameter for YetiRank/QueryRMSE/QuerySoftMax, so tuning must never
    splice a `max_pairs=...` suffix onto those losses."""

    class SelectingTrial:
        number = 0

        def __init__(self, selection):
            self.selection = selection
            self.choices = {}

        def suggest_categorical(self, name, choices):
            self.choices[name] = choices
            return self.selection.get(name, choices[0])

    model = CatboostRankerMother(max_pairs=50, tune_tree_structure_type=False, tune_boosting_type=False)
    trial = SelectingTrial({"base_loss": "QueryRMSE"})

    suggested = model.suggested_params_loss(trial, {}, pd.Series([1.0, 2.0, 3.0]), prefix="")

    assert suggested["loss_function"] == "QueryRMSE"
    # max_pairs must still be explicitly re-asserted (not silently dropped) so that
    # set_params() sees it as "touched" and doesn't clear the configured cap.
    assert suggested["max_pairs"] == 50


def test_suggested_params_loss_preserves_max_pairs_across_non_pairlogit_trial():
    """Regression test: a trial that picks a non-PairLogit loss must not permanently wipe
    a previously-configured max_pairs, or every later PairLogit trial would be tuned
    without it. set_params() clears max_pairs when it's omitted from a call that also
    supplies an explicit loss_function not defining it -- suggested_params_loss must
    therefore keep re-asserting max_pairs every trial regardless of the loss chosen."""

    class SelectingTrial:
        number = 0

        def __init__(self, selection):
            self.selection = selection
            self.choices = {}

        def suggest_categorical(self, name, choices):
            self.choices[name] = choices
            return self.selection.get(name, choices[0])

    model = CatboostRankerMother(max_pairs=50, tune_tree_structure_type=False, tune_boosting_type=False)
    y = pd.Series([1.0, 2.0, 3.0])

    # Trial 1: a non-PairLogit loss is picked -- must not clear self.max_pairs.
    trial_1 = SelectingTrial({"base_loss": "QueryRMSE"})
    suggested_1 = model.suggested_params_loss(trial_1, {}, y, prefix="")
    model.set_params(**suggested_1)
    assert model.max_pairs == 50

    # Trial 2: back to PairLogit -- the configured cap must still be applied.
    trial_2 = SelectingTrial({"base_loss": "PairLogit"})
    suggested_2 = model.suggested_params_loss(trial_2, {}, y, prefix="")
    assert suggested_2["loss_function"] == "PairLogit:max_pairs=50"


def test_set_params_leaves_state_unchanged_when_parent_set_params_raises():
    """If super().set_params() (CatBoost/sklearn) rejects the update, self.top/self.max_pairs
    and the tuning flags must be left exactly as they were before the call -- the staged
    values must only be committed once the parent update has actually succeeded."""
    model = CatboostRankerMother(top=5, tune_tree_structure_type=False, tune_boosting_type=False)

    with patch.object(m_catboost.CatBoostRanker, "set_params", side_effect=ValueError("boom")):
        with pytest.raises(ValueError, match="boom"):
            model.set_params(top=9, tune_tree_structure_type=True)

    assert model.top == 5
    assert model.tune_tree_structure_type is False


def test_sklearn_clone_preserves_params():
    skl_set_config(enable_metadata_routing=True)
    model = CatboostRankerMother(
        tune_boosting_type=True,
        tune_loss_function=False,
        top=5,
        max_pairs=50,
    )
    cloned = skl_base.clone(model)
    assert cloned.tune_boosting_type == model.tune_boosting_type
    assert cloned.tune_loss_function == model.tune_loss_function
    assert cloned.top == model.top
    assert cloned.max_pairs == model.max_pairs
    assert cloned.model_type == "ranking"


@pytest.mark.slow
def test_pickle_roundtrip_preserves_params_and_predictions():
    skl_set_config(enable_metadata_routing=True)
    model, X, _, groups = _fit_ranker()
    mask = groups == 0
    X_group = X[mask]

    pred_before = model.predict(X_group)

    serialized = pickle.dumps(model)
    restored = pickle.loads(serialized)

    assert restored.model_type == model.model_type
    assert restored.target_type == model.target_type
    assert restored.tune_boosting_type == model.tune_boosting_type
    assert restored.tune_loss_function == model.tune_loss_function
    assert restored.tune_tree_structure_type == model.tune_tree_structure_type
    assert restored.get_params()["posterior_sampling"] == model.get_params()["posterior_sampling"]

    pred_after = restored.predict(X_group)
    np.testing.assert_array_almost_equal(pred_before, pred_after)
