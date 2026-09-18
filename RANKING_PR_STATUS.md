# PR #45 — Learning to Rank Tuning and Uncertainty

This branch (`learningToRankTuningAndUncertainty`) adds ranking support to Mother.
`main` has no ranking model at all today: CatBoost has a dedicated ranking model
(`CatBoostRanker`), but Mother had no wrapper for it, no way to tune its
ranking-specific settings, no way to predict per group, and no way to estimate how
confident it is in its rankings. This PR adds all of that, plus a handful of
unrelated correctness fixes found along the way.

## 1. New model: `CatboostRankerMother`

File: `src/mother/ml/models/m_catboost.py`

A subclass of CatBoost's `CatBoostRanker` that plugs into Mother's existing
pipeline/tuning conventions, the same way `CatboostRegressorMother` and
`CatboostClassifierMother` already do. It adds:

- **Automatic loss-function selection during tuning**: Optuna tries different
  ranking loss functions (`YetiRank`, `YetiRankPairwise`, `PairLogit`,
  `PairLogitPairwise`, `QueryRMSE`, `QuerySoftMax`) and keeps whichever works best
  on the data.
- **`top` parameter**: an optional cutoff (e.g. "only the top 5 matter"), valid for
  `YetiRank`/`YetiRankPairwise` in any mode except `Classic`.
- **`max_pairs` parameter**: for the `PairLogit` family, caps how many item pairs
  CatBoost generates per group, keeping training fast on large groups.
- **`top`/`max_pairs` must be defined in exactly one place** — either baked
  directly into an explicit `loss_function` string (e.g.
  `"YetiRank:mode=NDCG;top=5"`) or passed as the dedicated `top=`/`max_pairs=`
  argument, but not conflicting values in both. Re-supplying the same
  already-resolved value in both places is fine (this is what makes
  `sklearn.clone()`/`get_params()`/`set_params()` round-trips work); a genuine
  conflict raises a clear `ValueError` instead of guessing which one should win.
  Whichever way `top`/`max_pairs` was set, the attribute and `get_params()` always
  reflect the value actually in effect.
- **Pairwise-loss safety checks**: `YetiRankPairwise` and `PairLogitPairwise`
  require `grow_policy="SymmetricTree"` and `boosting_type="Plain"`; requesting a
  Pairwise loss with incompatible tree settings raises immediately instead of
  failing deep inside CatBoost later.
- **`top=` embedded in an explicit loss string now requires an explicit mode**:
  `loss_function="YetiRank:top=5"` has no `mode=`, which defaults to CatBoost's
  `Classic` mode, where `top` has no effect. Previously this string was accepted
  unchanged, leaving an ambiguous/ineffective cutoff; it now gets `mode=NDCG`
  appended automatically, for both `YetiRank` and `YetiRankPairwise`.

## 2. Predicting per group, not globally

File: `src/mother/ml/models/m_catboost.py`

Ranks only make sense *within* a group (e.g. one experiment batch), not across
unrelated groups, so:

- `ranker_predict_for_groups(model, X, group_id, ...)` splits `X` by `group_id`,
  predicts per group, and reassembles results in the original row order. Raw-score
  mode (`use_ranks=False`) instead makes a single vectorized `model.predict(X)`
  call over the whole dataset, since scores don't depend on group boundaries —
  only rank mode needs the per-group loop.
- `ranker_predict_uncertainty_for_groups(model, X, group_id, ...)` does the same
  for uncertainty estimation (section 3), with the same vectorized fast path when
  `use_ranks=False` and `normalize_by_group_size=False`.
- Both validate `group_id` up front: it must match `X` in length and must not
  contain missing values.

## 3. Measuring uncertainty with virtual ensembles

Files: `src/mother/ml/utils.py`, `src/mother/ml/models/m_catboost.py`

CatBoost can generate several "virtual" versions of one trained model; comparing
their predictions tells us how stable/confident a ranking is. New utilities turn
that into concrete numbers:

- **`scores_to_ranks`/`scores_matrix_to_ranks`**: convert raw scores into 1-based
  ranks (rank 1 = best). Items with identical scores get the identical rank
  (`[1.0, 1.0, 1.0] -> [1, 1, 1]`) rather than an arbitrary tie-break — the model
  genuinely can't distinguish them, so it shouldn't be made to look like it can.
- **`topk_rank_disagreement`**: for each item, the probability that two randomly
  chosen ensemble members would disagree about whether it belongs in the top `k`.
- **`topk_score_variance`**: for items in the top `k`, how much their raw scores
  vary across ensemble members.
- **`groupwise_topk_analysis`**: combines the above per group into a consensus
  ranking (each item's mean rank across ensemble members) plus a `topk_member`
  flag and a `topk_score_var` column. A genuine tie in the consensus ranking can
  flag more than `k` members in a group — documented and tested behavior,
  consistent with the tie handling above rather than a special-cased tie-break.

## 4. `mother_cv` support for ranking

File: `src/mother/pipeline_utils.py`

`mother_cv` now forwards extra `**kwargs` to each estimator's
`predict_uncertainty()` call (needed so ranking can pass `group_id`), and
explicitly rejects estimators whose `predict_uncertainty()` returns a tuple (e.g.
the ranker's `return_raw=True` mode), raising a clear `TypeError` instead of
failing confusingly later. The docstring now also spells out that ranking
query/group metadata must *not* be forwarded through `**kwargs` here: this loop
only slices `X`/`y`/`groups` per fold, so a full-dataset group array would
either be rejected by `CatboostRankerMother.predict_uncertainty` or mismatched
in length for a custom estimator — group-aware ranking uncertainty must go
through `ranker_predict_uncertainty_for_groups` instead.

## 5. Other fixes bundled into this branch

- **`avg_ndcg_score` (`src/mother/ml/utils.py`) was scoring rankings backwards.**
  It converted true/predicted values to ranks (where, by this codebase's
  convention, rank `0` = best) and fed those ranks into sklearn's `ndcg_score`,
  which expects the opposite convention (larger = more relevant). A perfect
  ranking could score as if it were completely reversed. Fixed by passing the
  original values straight through instead of converting to ranks first; the
  function now has its first test coverage.
- **`CatboostGaussianProcessRegressorMother`**: `set_params` now keeps
  `self.gp_params` (what `fit()` actually reads `learning_rate`/`max_depth`/etc.
  from) in sync for every key it contains, not just a small custom subset —
  previously an Optuna trial calling `set_params(learning_rate=...)` silently had
  no effect on training. `__init__` now rejects unsupported
  `tune_boosting_type`/`tune_tree_structure_type`/`tune_loss_function` kwargs the
  same way `set_params` already did, instead of forwarding them into CatBoost
  where they'd surface as a confusing failure at `fit()` time. `set_params` is
  also now atomic: a call that raises (e.g. an invalid CatBoost parameter) no
  longer leaves the object with a partially-applied update.
- **TabPFN `BFloat16` embedding crash**: pre-fitted TabPFN models that default to
  bfloat16 autocast could crash `get_embeddings()` on CPU
  (`TypeError: Got unsupported ScalarType BFloat16`). `inference_precision` is now
  forced to `float32` only for the duration of each embedding call, then restored
  afterward, so the crash is avoided without permanently mutating a caller-owned
  model.
- **`CatboostRegressorMother`'s constructor argument order preserved**: the new
  `tune_loss_function` parameter is appended after the existing parameters instead
  of before them, so existing positional construction isn't silently shifted.
- **CI**: the `build-docs` job is now gated to `main` only (matching
  `deploy-docs`), so it no longer runs — and fails on a Pages-artifact permission
  error — on every feature-branch push.
- **Docs**: added a "Ranking with `CatboostRankerMother`" section to
  `docs/mother/models.md` with runnable examples, and added the ranking tutorial
  notebook to `docs/examples.md`.

## 6. Tests

`test/unit/test_catboost_ranker.py` (new) covers the ranker model, its parameter
handling, and the groupwise helper functions. `test/unit/test_utils.py`,
`test/unit/test_mother_cv.py`, and `test/unit/test_catboost_reg_uncertainty.py`
were extended with regression tests for the fixes above.

Two tests were also tightened based on review feedback:

- `test_set_params_top_zero_preserves_classic_mode_without_ndcg_parameters` was
  renamed to `test_set_params_top_zero_does_not_revert_to_classic_mode` — the
  old name claimed Classic mode was preserved, but the test's own setup and
  assertion show `mode` switching to and staying at `NDCG`.
- `test_stability_differences_across_groups`'s `min_prob_g0 < 0.5` assertion was
  mathematically unreachable as a real check (with `n_ensembles=15` the maximum
  possible disagreement probability is `112/225 ≈ 0.498`, so it always passed
  regardless of actual stability). The fixture's "near-tie" group now has two
  items with near-duplicate features but different targets competing for the
  `k=2` cutoff, giving the model genuine, unresolvable ambiguity there, and the
  assertion now compares the two groups' mean disagreement directly instead of
  against a fixed threshold.

## 7. How this was verified

```
uv run pytest test/unit/test_utils.py test/unit/test_catboost_ranker.py -m "not slow"
# 180 passed, 23 deselected (slow)
```

The TabPFN fix needs the `tabpfn` extra installed (`uv sync --extra tabpfn`):

```
uv run pytest test/unit/test_tabpfn.py -q
# 34 passed
```

The docs update was verified with:

```
uv run poe docs-python-fences
# 27 block(s) executed, 1 block(s) skipped, 0 failures
```

Before merging, run the full non-slow suite once more (`uv run poe test` /
`uv run poe coverage`), and the full `test-slow` suite (`uv run poe test-slow`),
to make sure nothing outside these files regressed.
