# Using different Models

Although mother is build around catboost it basically supports other models from the ML community (like all sklearn estimators). For example, `RandomForest` is already supported. Furthermore, [own models](#providing-your-own-model) can be provided.

## Currently supported models

Mother discovers available model wrappers from `src/mother/ml/models/m_*.py`.
At the moment, the built-in algorithm groups are:

- `catboost`
- `randomforest`
- `lasso`
- `tabpfn`
- `tabicl`
- `tabm` (optional `tabm` extra)

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
| `tabicl` | `TabICLRegressorMother`, `TabICLClassifierMother` , `TabICLEmbeddingTransformer` (utility extracting TabICL embeddings for dowstream models)|
| `tabm` | `TabMRegressorMother`, `TabMClassifierMother` |

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

## TabM with skorch

For a runnable tour of the supported functionality, see the
[TabM integration notebook](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/05_advanced/06_tabm_skorch_integration.ipynb).

Install the optional dependencies with `pip install 'mother-ml[tabm]'` or
`uv sync --extra tabm`. TabM uses PyTorch and skorch; installing pytabkit is
not required. The neural architecture is adapted from
[pytabkit's TabM implementation](https://github.com/dholzmueller/pytabkit/blob/main/pytabkit/models/nn_models/tabm.py),
based on [TabM](https://github.com/yandex-research/tabm).

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from mother.ml.models.m_tabm import TabMRegressorMother, TabMClassifierMother

X, y = make_regression(
    n_samples=120, n_features=6, noise=5.0, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42,
)
demo_parameters = dict(
    n_blocks=1, d_block=32, k=4, max_epochs=5, batch_size=32,
)
regressor = TabMRegressorMother(
    **demo_parameters,
)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
assert predictions.shape == (len(X_test),)

class_labels = np.where(y_train >= np.median(y_train), "high", "low")
classifier = TabMClassifierMother(**demo_parameters)
classifier.fit(X_train, class_labels)
probabilities = classifier.predict_proba(X_test)
assert probabilities.shape == (len(X_test), 2)
np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
```

This self-contained smoke example uses a deliberately small training budget,
not a recommended configuration for model-quality evaluation.

The supported tasks are finite single-target regression, binary classification,
and multiclass classification. Both pandas DataFrames and dense NumPy arrays
are accepted. For DataFrames, column names and order must match at prediction
time. Sparse inputs, multilabel classification, multi-target regression,
missing-target learning, and flow heads are not supported in this initial integration.

### Categorical inputs and preprocessing

Declare categorical columns explicitly with `cat_features=["category"]` for a
DataFrame or positional indices for an array. This includes integer-valued
categories. Columns not declared categorical must be finite numeric values.
Impute missing numeric and categorical values upstream.

The estimator fits its numeric scaler, category encoder, and regression-target
scaler on the data supplied to each `fit()` call. Prediction reuses those fitted
transforms; unseen categories use the network's all-zero unknown-category
encoding. Numeric scaling is enabled by default and can be disabled with
`scale_features=False`. Regression targets are standardized internally and
predictions are converted back to the original target scale.

Numeric feature scaling and regression-target scaling use float64 precision
before converting the prepared training arrays to float32 for the network.
This preserves small differences around large offsets during standardization.
Regression predictions, including the prediction columns in uncertainty
outputs, are restored to original units in float64. With feature scaling
disabled, numeric inputs are still converted to float32 for the network, so
small differences around large offsets may be lost.

When combining TabM with upstream feature selection, preserve its categorical
columns and their names yourself; declaring `cat_features` in TabM does not
configure a separate ColumnTransformer. For configuration-driven construction,
use `model.algorithm="tabm"` and put TabM constructor options, including
`cat_features`, in `model.parameters`.

### Optional numerical embeddings

Numerical embeddings are disabled by default (`num_emb_type="none"`), preserving
the ordinary numeric-input workflow. An embedding changes how each numeric
value is presented to the neural network. Instead of giving the network one
number directly, it gives it a small set of learned signals that can make
useful patterns easier to recognize. The embedding does not create new
measurements; it changes the representation of the values you already have.

Choose an embedding based on the kinds of patterns you expect:

- `"linear"` learns a simple, feature-specific rescaling and offset. This is a
  straightforward way to give each numeric feature its own learned input
  representation.
- `"linear_relu"` adds a threshold-like step to the linear version. This can
  help the model treat values below and above a learned point differently.
- `"periodic"` represents a value with repeating wave-like signals. This can
  help when a feature has a repeating pattern, such as an hour of day or a
  seasonal cycle. For genuinely cyclic features, make sure their numeric
  encoding reflects that cycle (for example, hour-of-day values wrap around).
- `"pwl"` splits the observed range into intervals and learns how the value
  should be represented in each interval. This can help when a feature's effect
  changes across its range, such as a stronger effect at low values and a
  flatter effect at high values.

These are options to try, not guarantees that one representation will work
better. If you are unsure, keep embeddings disabled or use
`num_emb_type="auto"` with MotherTuner to compare the four embedding types.

For example, PWL uses the same configuration as
[pytabkit's TabM adapter](https://github.com/dholzmueller/pytabkit/blob/main/pytabkit/models/alg_interfaces/tabm_interface.py):

```python
embedded_regressor = TabMRegressorMother(
    **demo_parameters,
    num_emb_type="pwl",
    d_embedding=16,
    num_emb_n_bins=48,
)
embedded_regressor.fit(X_train, y_train)
assert embedded_regressor.predict(X_test).shape == (len(X_test),)
```

The same options work for `TabMClassifierMother` and can be supplied through
`model.parameters` in MotherSettings. PWL uses a piecewise-linear embedding
with quantile-based intervals, fitted from training data only. Its internal
implementation uses version `"B"` without an extra ReLU. `d_embedding`
defaults to 16 for each embedding type; for PWL, `num_emb_n_bins` defaults to
48. The effective number of intervals is capped by the training sample count,
and repeated quantile boundaries are removed, so some features may end up with
fewer intervals.

PWL bins are fitted only on the numeric training data, after optional numeric
scaling, on every `fit()` call (and therefore separately within each CV fold).
Prediction reuses the fitted bins; serialization preserves them. Each numeric
feature is represented with `d_embedding` values inside the model, increasing
the backbone's input size and computational cost without adding new measured
information.
Categorical encoding is unchanged; categorical-only inputs do not use numerical
embeddings. PWL training requires at least three rows and at least two distinct
values per numeric column. Remove constant numeric columns upstream or leave
PWL embeddings disabled.

By default, `n_blocks=None` follows pytabkit's automatic-depth heuristic: use
two blocks when PWL embeddings are active on numeric features, otherwise three.
Linear, linear/ReLU, periodic, and categorical-only inputs therefore use three
blocks. This is a fixed rule, not a search for the best depth. Set
`n_blocks` to a positive integer to use that depth during ordinary fitting;
the actual fitted depth is stored in `n_blocks_` without changing the constructor
parameter. MotherTuner continues to search depths from one to four, overriding
either `None` or an explicit depth, with three blocks in its default first trial.
This changes the untuned PWL default from three blocks to two; specify
`n_blocks=3` to retain the previous depth.

With a fixed embedding type, MotherTuner tunes its settings: `d_embedding`
(integers 4–64, logarithmic sampling) for all types; `num_emb_n_bins` (integers
2–128, logarithmic sampling) for PWL; and `num_emb_n_frequencies` (integers
8–128, logarithmic sampling), `num_emb_frequency_init_scale` (0.001–0.1,
logarithmic sampling), and `num_emb_activation` for periodic embeddings. The
periodic defaults are 48 frequencies, scale 0.01, and activation enabled.

Set `num_emb_type="auto"` to let MotherTuner choose among `"linear"`,
`"linear_relu"`, `"periodic"`, and `"pwl"`, then tune that type's settings. The
default first trial uses PWL with width 16 and 48 bins. `"auto"` is for tuning;
direct `.fit()` requires a specific embedding type. Embeddings remain opt-in:
`num_emb_type="none"` does not search or enable them. Rooted pipelines
automatically prefix these parameters like the other model hyperparameters.

### Training, tuning, and predictions

- `arch_type="tabm"` is the default. Ensemble members share the main network
  but have their own small adjustments throughout its layers. With
  `"tabm-mini"`, members have their own input adjustments and output heads,
  while sharing the intervening network. Mini has fewer member-specific
  parameters, not fewer members; `k` controls ensemble size for both variants.
- Set `arch_type="auto"` to let MotherTuner compare `"tabm"` and
  `"tabm-mini"`. The first default trial uses `"tabm"`. Explicit choices stay
  fixed during tuning. Like embedding `"auto"`, architecture `"auto"` is for
  tuning; direct `.fit()` requires a specific variant. Architecture and
  embedding selection can be tuned together.
- Each member is trained against the target separately. Regression predictions
  are averaged; classification applies softmax to each member before averaging
  probabilities. Labels are decoded using `classes_`.
- Training uses skorch mini-batches and AdamW, retaining upstream zero-weight-decay
  parameter groups. `max_epochs` is a fixed budget: this initial version has no
  internal validation split or early stopping.
- CPU is the default; use `device="cuda"` for an available CUDA device.
  `random_state` seeds training while restoring the caller's PyTorch CPU and
  selected CUDA generator state afterward. GPU execution is not guaranteed to
  be bitwise deterministic.
- Mother discovers the wrappers under `algorithm="tabm"`. They support sklearn
  cloning, serialization, pipelines, cross-validation, and MotherTuner.
  The initial tuning space covers learning rate, weight decay, depth, width,
  dropout, and ensemble size.
- `predict_uncertainty()` reports regression member spread and the classification
  entropy decomposition described below.

### Regression knowledge uncertainty from TabM members

For member predictions \(f_k(x)\) in the original target units, regression returns
their mean in both `pred` and `mean_predictions`, and their population standard
deviation in `knowledge_uncertainty`:

\[
\sigma_{\mathrm{knowledge}}(x) =
\sqrt{\frac{1}{K}\sum_{k=1}^{K}(f_k(x)-\bar f(x))^2}.
\]

This uses `ddof=0`, so a single member or identical predictions produce zero
spread. The target scaler is accounted for: uncertainty is in the same units as
the predictions, not standardized units. This is member disagreement, **not**
the standard error of the mean (there is no division by \(\sqrt K\)), a calibrated
prediction interval, or a measure of observed-label noise. Shared parameters
can reduce member diversity; low spread does not guarantee a correct prediction.

`data_uncertainty` and `total_uncertainty` remain unset because this adapter
does not estimate observation noise. The returned DataFrame preserves the input
index; `uncertainty_for_opt=True` returns only `knowledge_uncertainty`.

### Classification uncertainty from TabM members

For each input, let \(p_k(c)\) be member \(k\)'s class probabilities,
\(\bar p = K^{-1}\sum_k p_k\), and \(H(p)=-\sum_c p(c)\log p(c)\).
TabM classification returns:

| Output | Definition | Interpretation |
|---|---|---|
| `total_uncertainty` | \(H(\bar p)\) | Entropy of the averaged prediction |
| `data_uncertainty` | \(K^{-1}\sum_k H(p_k)\) | Average uncertainty within members |
| `knowledge_uncertainty` | \(H(\bar p)-K^{-1}\sum_k H(p_k)\) | Disagreement between members |

These values are in **nats**, use all fitted members, and satisfy
`total_uncertainty = data_uncertainty + knowledge_uncertainty` up to numerical
precision. The knowledge term is the generalized Jensen-Shannon divergence.
Confident members that disagree produce high knowledge uncertainty; members
that all predict the same uncertain distribution produce data uncertainty but
no knowledge uncertainty. With one member, knowledge uncertainty is zero.

This entropy-based interpretation is analogous to CatBoost's virtual-ensemble
classification uncertainty. Independence is not required for the decomposition:
TabM members share parameters and CatBoost virtual ensembles share trees.
Dependence can reduce diversity, however, so disagreement is an epistemic proxy,
not a calibrated Bayesian posterior estimate. Expected member entropy is a
data-uncertainty proxy, not a guaranteed isolation of irreducible label noise.
Low disagreement does not prove that a prediction is correct.

The returned DataFrame preserves the input index and contains `pred`, `proba_i`
in `classes_` order, unset `mean_predictions`, and the three uncertainty columns.
Both binary and multiclass classification are supported. No target labels are
needed; this uses predicted probabilities, not observed-label likelihood or
negative log-likelihood. `uncertainty_for_opt=True` returns only the
`knowledge_uncertainty` column for acquisition/optimization workflows.

Non-shared training batches are not exposed by this adapter. Numerical
embeddings are opt-in as described above; linear/ReLU and periodic embedding
variants are not exposed by the Mother wrapper.

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
for a full worked example, including groupwise uncertainty and out-of-fold validation.

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
