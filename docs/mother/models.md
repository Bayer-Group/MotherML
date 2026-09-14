# Using different Models

Although mother is build around catboost it basically supports other models from the ML community (like all sklearn estimators). For example, `RandomForest` is already supported. Furthermore, [own models](#providing-your-own-model) can be provided.

## Currently supported models

Mother discovers available model wrappers from `src/mother/ml/models/m_*.py`.
At the moment, the built-in algorithm groups are:

- `catboost`
- `randomforest`
- `lasso`
- `tabpfn`

You can always verify what is available in your environment:

```python
from mother import ml

print(ml.get_available_algorithms())
print(ml.get_supported_models())
```

### Built-in model classes

| Algorithm key | Main model classes |
|---|---|
| `catboost` | `CatboostRegressorMother`, `CatboostGaussianProcessRegressorMother`, `CatboostClassifierMother`, `CatboostRankerMother` |
| `randomforest` | `RandomForestRegressorMother`, `RandomForestClassifierMother` |
| `lasso` | `LassoRegressorMother`, `LassoClassifierBinaryMother`, `LassoClassifierMulticlassMother` |
| `tabpfn` | `TabPFNRegressorMother`, `TabPFNClassifierMother` |

### Easy usage patterns

The easiest way is to ask Mother for the model class by algorithm and task type, then instantiate it.

```python
from mother import ml

# Regressor (catboost)
reg_cls = ml.get_model_class_by_algorithm_and_type("catboost", "regression")
reg = reg_cls()

# Classifier (random forest)
clf_cls = ml.get_model_class_by_algorithm_and_type("randomForest", "classification")
clf = clf_cls()
```

Use an explicit subtype when needed:

```python
from mother import ml

# Lasso multiclass classifier
lasso_multi_cls = ml.get_model_class_by_algorithm_and_type(
    "lasso", "classification_multiclass"
)
lasso_multi = lasso_multi_cls()
```

Or retrieve all classes for one algorithm and pick one:

```python
from mother import ml

all_catboost_models = ml.get_model_class_by_algorithm("catboost")
print([m.__name__ for m in all_catboost_models])
```

## Ranking with `CatboostRankerMother`

`CatboostRankerMother` wraps CatBoost's learning-to-rank (`CatBoostRanker`) for tasks where the goal is to
*order* items within a group (e.g. rank candidates within one experiment batch) rather than predict an
isolated value per item. Every ranking row must carry a `group_id` identifying which group it belongs to;
ranks are only ever compared within a group, never across groups.

```python
import numpy as np
import pandas as pd
import sklearn

from mother.ml.models.m_catboost import CatboostRankerMother

sklearn.set_config(enable_metadata_routing=True)

rng = np.random.default_rng(0)
X_features = pd.DataFrame(rng.random((12, 3)), columns=["f0", "f1", "f2"])
y = pd.Series(rng.random(12))
groups = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

ranker = CatboostRankerMother(logging_level="Silent", num_trees=200).set_fit_request(group_id="group_id")
ranker.fit(X=X_features, y=y, group_id=groups)
```

By default CatBoost uses the `YetiRank` loss. Other ranking losses (`YetiRankPairwise`, `PairLogit`,
`PairLogitPairwise`, `QueryRMSE`, `QuerySoftMax`) can be set via `loss_function`, or left to Optuna to choose
automatically during hyperparameter tuning. Two convenience parameters are folded into the `loss_function`
string automatically:

- `top`: restricts `NDCG`/`MAP`-style losses to the top-`k` items per group (any mode except `Classic`).
- `max_pairs`: caps how many item pairs the `PairLogit` family samples per group, to keep training fast on
  large groups.

### Predicting and estimating uncertainty per group

Because ranks and rank uncertainty only make sense within a group, use the module-level helpers instead of
calling `predict`/`predict_uncertainty` directly on a mixed-group `X`:

```python
from mother.ml.models.m_catboost import (
    ranker_predict_for_groups,
    ranker_predict_uncertainty_for_groups,
)

ranks = ranker_predict_for_groups(ranker, X_features, group_id=groups, use_ranks=True)
uncertainty = ranker_predict_uncertainty_for_groups(ranker, X_features, group_id=groups, use_ranks=True)
```

`predict_uncertainty` estimates ranking confidence using CatBoost's virtual ensembles: several slightly
different versions of the trained model are compared, and how much they disagree on an item's score/rank
indicates how stable that item's ranking is. `mother.ml.utils.groupwise_topk_analysis` builds on this to flag
which top-`k` selections are unstable across groups.

See the [ranking tutorial notebook](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/05_advanced/04_ranking_model.ipynb)
for a full worked example, including tuning, groupwise uncertainty, and out-of-fold validation.

!!! tip

    If you are unsure about exact class names or capabilities, use:

    ```python
    from mother import ml
    print(ml.describe_model("RandomForestClassifierMother"))
    ```

## Prediction and Uncertainty Interface

Starting with the 1.0.1 release line, model wrappers expose a more consistent prediction interface:

- `predict(...)` returns aligned outputs across model backends.
- `predict_uncertainty(...)` provides uncertainty outputs for models that support it.

For regression models with uncertainty support, output columns follow a common naming pattern:

- `prediction`
- `uncertainty_data`
- `uncertainty_knowledge`
- `uncertainty_total`

Depending on model capabilities, one or more uncertainty columns can be present. The returned DataFrame keeps index alignment with the input rows to make downstream merging and analysis robust.

!!! note

    `mother_cv` now has improved return typing and estimator-return behavior to better support workflows that inspect trained estimators after CV.

!!! tip

    === "Getting a list of provided algorithms"
        ```python exec="on" source="tabbed-left"
        from mother import ml
        print(ml.get_available_algorithms())
        ```

    === "Getting a list of oob models"
        ```python exec="on" source="tabbed-left"
        from mother import ml
        print(ml.get_supported_models())
        ```


    === "Getting information on a model class"
        ```python exec="on" source="tabbed-left"
        from mother import ml
        print(ml.describe_model("RandomForestClassifierMother"))
        ```

## Using Lasso with Hyperparameter Tuning

### Providing your own Model

To provide your own model and make this step as easy as possible, we provide the `AbstractMotherPipelineClass`.

::: mother.ml.core.AbstractMotherPipeline

Your own model just has to inherit from that class and implement the required functions that provide the hyperparameters you want to tune. For example, see the implementation of the Lasso model. Since lasso basically has one parameter to be tuned, the implementation is fairly easy.

~~~python title="Lasso with Hyperparameter Tuning"
--8<-- "src/mother/ml/models/m_lasso.py"
~~~

## Registering your model using MotherModelRegistry

To register your own model and use it easily within the mother framework you can register your model with the available decorator.

::: mother.ml

 See the following example how to implement your custom RandomForest Classifier.

!!! example
    ```python
    from mother import ml
    from sklearn.ensemble import RandomForestClassifier

    @ml.register_model("custom_rf")
    class CustomRandomForestMother(RandomForestClassifier, ml.AbstractMotherPipeline):
        def get_hyperparameter_space(self, X, y, trial, prefix=""):
            return {
                f"{prefix}n_estimators": trial.suggest_int("n_estimators", 10, 100),
                f"{prefix}max_depth": trial.suggest_int("max_depth", 3, 10),
            }

        def default_parameters(self, prefix=""):
            return {f"{prefix}n_estimators": 50, f"{prefix}max_depth": 5}

    print(ml.get_model_class_by_algorithm("custom_rf"))
    ```
