import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import sklearn.base as skl_base
from sklearn import set_config as skl_set_config
from sklearn.datasets import make_regression

import mother.ml.models.m_catboost as m_catboost
from mother.ml.models.m_catboost import CatboostRankerMother

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
        # Identical scores - stable sort preserves input order
        (np.array([1.0, 1.0, 1.0]), np.array([1, 2, 3])),
        # Single value
        (np.array([5.0]), np.array([1])),
        # Negative values: least negative = highest = rank 1
        (np.array([-1.0, -5.0, -3.0]), np.array([1, 3, 2])),
        # Mixed positive/negative
        (np.array([1.0, -1.0, 0.0, 2.0]), np.array([2, 4, 3, 1])),
        # Zeros with one positive and one negative
        (np.array([0.0, 1.0, -1.0, 0.0]), np.array([2, 1, 4, 3])),
    ],
)
def test_scores_to_ranks(scores, expected):
    """scores_to_ranks uses descending order: highest score -> rank 1."""
    result = m_catboost.scores_to_ranks(scores)
    np.testing.assert_array_equal(result, expected)


def test_scores_to_ranks_preserves_input_order():
    """Output positions correspond to input positions (not sorted positions)."""
    scores = np.array([0.3, 0.7, 0.1, 0.9, 0.5])
    ranks = m_catboost.scores_to_ranks(scores)

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
        [m_catboost.scores_to_ranks(score_matrix[:, i]) for i in range(score_matrix.shape[1])]
    ).astype(float)
    got = m_catboost.scores_matrix_to_ranks(score_matrix)
    np.testing.assert_array_equal(got, expected)


def test_scores_matrix_to_ranks_rejects_non_2d_input():
    with pytest.raises(ValueError, match="Expected 2D score_matrix"):
        _ = m_catboost.scores_matrix_to_ranks(np.array([0.1, 0.2, 0.3]))


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
    assert set(ranks) == set(range(1, n + 1))


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
        expected[idx] = m_catboost.scores_to_ranks(scores[idx])

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
    for col in ("pred", "mean_predictions", "knowledge_uncertainty", "data_uncertainty", "total_uncertainty"):
        assert col in result.columns
    assert len(result) == len(X_group)


@pytest.mark.slow
def test_pred_column_matches_predict_ranks(fitted_ranker_data):
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
def test_uncertainty_for_opt_returns_only_knowledge_uncertainty_on_fitted_model(fitted_ranker_data):
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


@pytest.mark.slow
def test_normalize_uncertainty_in_0_1_range(fitted_ranker_data):
    model = fitted_ranker_data["model"]
    X_group = fitted_ranker_data["X_group"]
    result = model.predict_uncertainty(X_group, normalize_by_group_size=True)
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


def test_virtual_ensemble_helper_is_called_with_forwarded_parameters(mock_ranker_uncertainty_inputs):
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


def test_virtual_ensemble_mean_scores_remain_scores_by_default(mock_ranker_uncertainty_inputs):
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


def test_use_ranks_converts_mean_and_uncertainty_to_rank_scale(mock_ranker_uncertainty_inputs):
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


def test_uncertainty_for_opt_returns_only_knowledge_uncertainty(mock_ranker_uncertainty_inputs):
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


def test_predict_uncertainty_use_ranks_groupwise_via_helper(mock_ranker_uncertainty_inputs):
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


def test_ranker_predict_uncertainty_for_groups_matches_per_group_manual(mock_ranker_uncertainty_inputs):
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
        from_helper = m_catboost.ranker_predict_uncertainty_for_groups(model, mock_X, group_ids)

    expected = pd.concat(
        [
            _per_group_side_effect(mock_X.iloc[[0, 1]]),
            _per_group_side_effect(mock_X.iloc[[2, 3]]),
        ]
    ).loc[mock_X.index]

    pd.testing.assert_frame_equal(from_helper, expected)


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


def test_model_type_always_ranking():
    """model_type must stay 'ranking' regardless of what set_params receives."""
    model = CatboostRankerMother(model_type="ranking")
    assert model.model_type == "ranking"
    model.set_params(model_type="regression")
    assert model.model_type == "ranking"


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

    pred_after = restored.predict(X_group)
    np.testing.assert_array_almost_equal(pred_before, pred_after)
