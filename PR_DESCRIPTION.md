# Dependency updates & CatBoost sklearn-clone fix

## Summary

This PR updates three key dependencies to their current versions, fixes a latent bug in the CatBoost sklearn-clone integration, enforces commercially-licensed TabPFN model weights, and adjusts the supported Python range accordingly.

---

## Changes

### `scikit-learn` — 1.5.0 → `<1.9`

- Upper bound raised from `<=1.5.0` to `<1.9` (i.e., 1.5–1.8.x).
- sklearn 1.8.0 requires **Python ≥ 3.11**, which drove the Python version change below.
- sklearn 1.9.0 is excluded: `quantile-forest` uses the removed private symbol `sklearn.tree._tree.DTYPE`; will be revisited once `quantile-forest` publishes a compatible release (the fix is already merged on their GitHub).
- **`umap-learn` updated to `>=0.5.12,<0.6`** — the 0.5.12 release (April 2026) is the first to support sklearn 1.8's revised `__sklearn_tags__` API.
- **`pipeline_utils.py`** — replaced deprecated `val_estimator._estimator_type == "classifier"` check with `sklearn.base.is_classifier(val_estimator)`. In sklearn 1.8, `Pipeline` no longer exposes `_estimator_type` as an attribute; the old check silently returned `False`, causing `_proba` columns to be missing from multi-target classification CV output.

### `catboost` — ≥1.2.6 → ≥1.2.9

- Version range updated to `>=1.2.9,<=1.2.10`.
- CatBoost 1.2.9 explicitly adds Python 3.14 support, fixes the `setuptools.Distribution.dry_run` removal, and adds `__sklearn_tags__` for sklearn ≥ 1.8 compatibility.
- Newer CatBoost versions internally copy mutable constructor params (`cat_features`, `text_features`, `embedding_features`), which broke `sklearn.base.clone`'s identity check. The fix below targets exactly this behaviour.

### `tabpfn` — 2.1.0 → 8.0.8

- Updated to current release.
- TabPFN 3 (the new default in 8.x) is released under a **non-commercial license**. The code now explicitly enforces the commercially-licensed V2 weights by following the [official recommendation](https://github.com/PriorLabs/TabPFN):
  ```python
  TabPFNRegressor.create_default_for_version(ModelVersion.V2)
  TabPFNClassifier.create_default_for_version(ModelVersion.V2)
  ```
  Users can still override `model_path` to select a different version.

---

### CatBoost `__sklearn_clone__` fix (`m_catboost.py`)

**Problem:** `sklearn.base.clone` uses an identity check (`param1 is param2`) to verify that constructor parameters were not mutated. Newer CatBoost copies mutable params internally, causing this check to fail and breaking cross-validation with `cat_features` etc.

**Fix:**
- Introduced `_CatbooostModelMotherBase` as a proper mixin (removed the spurious `catboost.CatBoost` base — it is a mixin, not a model) with a custom `__sklearn_clone__` that reconstructs the estimator from `get_params()` directly and preserves any metadata-routing requests (`_metadata_request`).
- All four concrete classes now inherit from this mixin so the fix is active everywhere:

  | Class | Inheritance (before) | Inheritance (after) |
  |---|---|---|
  | `CatboostRegressorMother` | `CatBoostRegressor, _CatboostHyperParams` | `CatBoostRegressor, _CatbooostModelMotherBase, _CatboostHyperParams` |
  | `CatboostGaussianProcessRegressorMother` | `CatBoostRegressor, _CatboostHyperParams` | `CatBoostRegressor, _CatbooostModelMotherBase, _CatboostHyperParams` |
  | `CatboostClassifierMother` | `CatBoostClassifier, _CatboostHyperParams, AbstractMotherPipeline` | `CatBoostClassifier, _CatbooostModelMotherBase, _CatboostHyperParams` |
  | `CatboostRankerMother` | `CatBoostRanker, _CatboostHyperParams, BaseEstimator` | `CatBoostRanker, _CatbooostModelMotherBase, _CatboostHyperParams, BaseEstimator` |

- Added the missing `import copy` (the method used `copy.deepcopy` but the module was never imported).

---

### New tests — `test/unit/test_catboost_clone.py`

10 new tests covering the clone fix:

| Test | What it checks |
|---|---|
| `test_clone_regressor_with_mutable_params` | `cat_features` / `embedding_features` survive clone on regressor |
| `test_clone_classifier_with_mutable_params` | Same for classifier |
| `test_clone_without_mutable_params` | Plain clone works for all three classes |
| `test_clone_preserves_metadata_routing_request` | `set_fit_request` state is preserved on `CatboostRankerMother` (skipped on sklearn < 1.3) |
| `test_cross_val_score_regressor_with_cat_features` | `cross_val_score` completes end-to-end with `cat_features` |
| `test_cross_val_score_classifier_with_cat_features` | Same for classifier |

---

### Python version support

| Version | Before | After |
|---|---|---|
| 3.10 | ✅ | ❌ dropped — sklearn 1.8 requires ≥ 3.11 |
| 3.11 | ✅ | ✅ |
| 3.12 | ✅ | ✅ |
| 3.13 | ✅ | ✅ |
| 3.14 | ❌ | ✅ added — enabled by catboost 1.2.9 (Python 3.14 support + setuptools `dry_run` fix) |

CI matrix updated: 3.10 removed, 3.14 added.

---

## Test results

All existing tests pass. New clone tests: **10/10 ✅**. TabPFN tests: **29/29 ✅**.
