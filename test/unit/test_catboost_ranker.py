import pickle
import unittest

import numpy as np
import pandas as pd
import pytest
import sklearn.base as skl_base
from sklearn import set_config as skl_set_config
from sklearn.datasets import make_regression

from mother.ml.models.m_catboost import CatboostRankerMother, scores_to_ranks

# ---------------------------------------------------------------------------
# scores_to_ranks unit tests (no model needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scores, expected",
    [
        # Descending: higher score → better (lower) rank
        (np.array([0.5, 0.9, 0.1, 0.7]), np.array([3, 1, 4, 2])),
        (np.array([10, 5, 20, 15]), np.array([3, 4, 1, 2])),
        # Identical scores — stable sort preserves input order
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
    """scores_to_ranks uses descending order: highest score → rank 1."""
    result = scores_to_ranks(scores)
    np.testing.assert_array_equal(result, expected)


def test_scores_to_ranks_preserves_input_order():
    """Output positions correspond to input positions (not sorted positions)."""
    scores = np.array([0.3, 0.7, 0.1, 0.9, 0.5])
    ranks = scores_to_ranks(scores)

    assert len(ranks) == len(scores)
    assert set(ranks) == set(range(1, len(scores) + 1))

    # 0.9 → 1, 0.7 → 2, 0.5 → 3, 0.3 → 4, 0.1 → 5
    np.testing.assert_array_equal(ranks, np.array([4, 2, 5, 1, 3]))


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


# ---------------------------------------------------------------------------
# CatboostRankerMother.predict tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.usefixtures("preserve_metadata_routing")
class TestCatboostRankerPredict(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y, self.groups = _fit_ranker()
        # Use a single group for predict tests
        mask = self.groups == 0
        self.X_group = self.X[mask]

    def test_predict_returns_scores_by_default(self):
        scores = self.model.predict(self.X_group)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.X_group))
        # Raw scores are floats
        self.assertTrue(np.issubdtype(scores.dtype, np.floating))

    def test_predict_ranks_returns_1_based_integers(self):
        ranks = self.model.predict(self.X_group, ranks=True)
        self.assertIsInstance(ranks, np.ndarray)
        self.assertEqual(len(ranks), len(self.X_group))
        n = len(self.X_group)
        self.assertEqual(set(ranks), set(range(1, n + 1)))

    def test_predict_ranks_highest_score_gets_rank_1(self):
        scores = self.model.predict(self.X_group)
        ranks = self.model.predict(self.X_group, ranks=True)
        best_idx = int(np.argmax(scores))
        self.assertEqual(ranks[best_idx], 1)

    def test_predict_normalize_by_group_size(self):
        n = len(self.X_group)
        ranks_norm = self.model.predict(self.X_group, ranks=True, normalize_by_group_size=True)
        ranks_raw = self.model.predict(self.X_group, ranks=True)
        self.assertTrue((ranks_norm > 0).all())
        self.assertTrue((ranks_norm <= 1).all())
        np.testing.assert_array_almost_equal(ranks_norm, np.round(ranks_raw / n, 4))

    def test_predict_normalize_no_effect_without_ranks(self):
        scores_plain = self.model.predict(self.X_group)
        scores_norm = self.model.predict(self.X_group, normalize_by_group_size=True)
        np.testing.assert_array_equal(scores_plain, scores_norm)


# ---------------------------------------------------------------------------
# CatboostRankerMother.predict_uncertainty tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.usefixtures("preserve_metadata_routing")
class TestCatboostRankerPredictUncertainty(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y, self.groups = _fit_ranker()
        mask = self.groups == 0
        self.X_group = self.X[mask]

    def test_output_schema(self):
        result = self.model.predict_uncertainty(self.X_group)
        self.assertIsInstance(result, pd.DataFrame)
        for col in ("pred", "mean_predictions", "knowledge_uncertainty", "data_uncertainty", "total_uncertainty"):
            self.assertIn(col, result.columns)
        self.assertEqual(len(result), len(self.X_group))

    def test_pred_column_matches_predict_ranks(self):
        result = self.model.predict_uncertainty(self.X_group)
        expected_pred = self.model.predict(self.X_group, ranks=True)
        np.testing.assert_array_equal(result["pred"].values, expected_pred)

    def test_knowledge_uncertainty_non_negative(self):
        result = self.model.predict_uncertainty(self.X_group)
        self.assertTrue((result["knowledge_uncertainty"] >= 0).all())

    def test_total_uncertainty_equals_knowledge_uncertainty(self):
        result = self.model.predict_uncertainty(self.X_group)
        pd.testing.assert_series_equal(
            result["knowledge_uncertainty"],
            result["total_uncertainty"],
            check_names=False,
        )

    def test_data_uncertainty_is_none(self):
        result = self.model.predict_uncertainty(self.X_group)
        self.assertTrue(result["data_uncertainty"].isna().all())

    def test_uncertainty_for_opt_returns_only_knowledge_uncertainty(self):
        result_opt = self.model.predict_uncertainty(self.X_group, uncertainty_for_opt=True)
        result_full = self.model.predict_uncertainty(self.X_group)
        self.assertListEqual(list(result_opt.columns), ["knowledge_uncertainty"])
        pd.testing.assert_series_equal(
            result_opt["knowledge_uncertainty"],
            result_full["knowledge_uncertainty"],
        )

    def test_normalize_by_group_size_scales_uncertainty(self):
        n = len(self.X_group)
        result_norm = self.model.predict_uncertainty(self.X_group, normalize_by_group_size=True)
        result_raw = self.model.predict_uncertainty(self.X_group)
        np.testing.assert_array_almost_equal(
            result_norm["knowledge_uncertainty"].values,
            np.round(result_raw["knowledge_uncertainty"].values / n, 4),
        )
        np.testing.assert_array_almost_equal(
            result_norm["mean_predictions"].values,
            np.round(result_raw["mean_predictions"].values / n, 4),
        )

    def test_normalize_uncertainty_in_0_1_range(self):
        result = self.model.predict_uncertainty(self.X_group, normalize_by_group_size=True)
        self.assertTrue((result["knowledge_uncertainty"] >= 0).all())
        self.assertTrue((result["knowledge_uncertainty"] <= 1).all())

    def test_index_preserved(self):
        result = self.model.predict_uncertainty(self.X_group)
        pd.testing.assert_index_equal(result.index, self.X_group.index)


# ---------------------------------------------------------------------------
# CatboostRankerMother get_params / set_params / clone / pickle tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("preserve_metadata_routing")
class TestCatboostRankerGetSetParams(unittest.TestCase):
    def test_get_params_contains_all_custom_keys(self):
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
            self.assertIn(key, params, msg=f"'{key}' missing from get_params()")

    def test_get_params_default_values(self):
        model = CatboostRankerMother()
        params = model.get_params()
        self.assertEqual(params["model_type"], "ranking")
        self.assertEqual(params["target_type"], "single_target")
        self.assertFalse(params["tune_pairwise_type"])
        self.assertFalse(params["tune_boosting_type"])
        self.assertTrue(params["tune_tree_structure_type"])
        self.assertTrue(params["tune_loss_function"])
        self.assertEqual(params["top"], 0)
        self.assertIsNone(params["max_pairs"])

    def test_set_params_updates_attributes(self):
        model = CatboostRankerMother()
        model.set_params(
            tune_boosting_type=True,
            tune_loss_function=False,
            tune_pairwise_type=False,
            top=10,
            max_pairs=100,
        )
        self.assertTrue(model.tune_boosting_type)
        self.assertFalse(model.tune_loss_function)
        self.assertEqual(model.top, 10)
        self.assertEqual(model.max_pairs, 100)

    def test_set_params_reflected_in_get_params(self):
        model = CatboostRankerMother()
        model.set_params(tune_loss_function=False, top=5)
        params = model.get_params()
        self.assertFalse(params["tune_loss_function"])
        self.assertEqual(params["top"], 5)

    def test_model_type_always_ranking(self):
        """model_type must stay 'ranking' regardless of what set_params receives."""
        model = CatboostRankerMother(model_type="ranking")
        self.assertEqual(model.model_type, "ranking")
        # set_params must not allow changing model_type away from 'ranking'
        model.set_params(model_type="regression")
        self.assertEqual(model.model_type, "ranking")

    def test_sklearn_clone_preserves_params(self):
        skl_set_config(enable_metadata_routing=True)
        model = CatboostRankerMother(
            tune_boosting_type=True,
            tune_loss_function=False,
            top=5,
            max_pairs=50,
        )
        cloned = skl_base.clone(model)
        self.assertEqual(cloned.tune_boosting_type, model.tune_boosting_type)
        self.assertEqual(cloned.tune_loss_function, model.tune_loss_function)
        self.assertEqual(cloned.top, model.top)
        self.assertEqual(cloned.max_pairs, model.max_pairs)
        self.assertEqual(cloned.model_type, "ranking")

    @pytest.mark.slow
    def test_pickle_roundtrip_preserves_params_and_predictions(self):
        skl_set_config(enable_metadata_routing=True)
        model, X, _, groups = _fit_ranker()
        mask = groups == 0
        X_group = X[mask]

        pred_before = model.predict(X_group)

        serialized = pickle.dumps(model)
        restored = pickle.loads(serialized)

        # Custom params preserved
        self.assertEqual(restored.model_type, model.model_type)
        self.assertEqual(restored.target_type, model.target_type)
        self.assertEqual(restored.tune_boosting_type, model.tune_boosting_type)
        self.assertEqual(restored.tune_loss_function, model.tune_loss_function)
        self.assertEqual(restored.tune_tree_structure_type, model.tune_tree_structure_type)

        # Predictions unchanged
        pred_after = restored.predict(X_group)
        np.testing.assert_array_almost_equal(pred_before, pred_after)
