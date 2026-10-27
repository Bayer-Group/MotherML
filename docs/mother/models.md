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

The estimators support single- and multitarget regression, single-target binary
or multiclass classification, and multitarget binary classification. Missing
target labels are supported for both regression and classification. Both pandas
DataFrames and dense NumPy arrays are accepted. For DataFrames, feature column
names and order must match at prediction time. Sparse inputs, multitarget
multiclass classification, mixed regression/classification tasks in one
estimator, and flow heads are not supported.

### Multitask learning and missing labels

Regression and classification remain separate estimators. Use
`target_type="multi_target"` and a two-dimensional target matrix or DataFrame
to train multiple tasks together. The tasks share one neural backbone, rather
than fitting a separate network per target. Regression uses one output per task;
multitarget binary classification uses one sigmoid output per task.
A target vector or Series is also accepted as a one-column target matrix,
including when CV or tuning flattens a one-column DataFrame. Predictions retain
their two-dimensional multitarget shape, and probabilities remain a list with
one array per task.

```text
                  Shared TabM backbone
                          |
                +---------+---------+
                |                   |
             Task 0              Task 1
                |                   |
           Observed label?     Observed label?
                |                   |
         Yes: include loss   No: mask this loss
```

For classification, each task column must contain exactly two observed classes.
Labels are encoded independently per task; different tasks may use different
label sets. Single-target classification still supports more than two classes.

Like the CatBoost estimators, both TabM classes accept `model_type`
(`"regression"`, `"classification_binary"`, or `"classification_multiclass"`).
The default `None` infers the task from the class and labels; an explicit
`"classification_binary"` rejects labels with more than two classes.
`get_model` passes `model.model_type` from MotherSettings automatically.

```python
from mother.ml.models.m_tabm import TabMRegressorMother, TabMClassifierMother

# Reuse X_train, X_test, y_train, and demo_parameters from the example above.
# Target matrices have shape (n_samples, n_tasks). Missing entries may be NaN.
regression_targets = np.column_stack([y_train, -2.0 * y_train])
binary_targets = np.column_stack([
    (y_train > np.median(y_train)).astype(float),
    (X_train[:, 0] > np.median(X_train[:, 0])).astype(float),
])
regression_targets[::4, 0] = np.nan
regression_targets[1::4, 1] = np.nan
binary_targets[::4, 0] = np.nan
binary_targets[1::4, 1] = np.nan

regressor = TabMRegressorMother(**demo_parameters, target_type="multi_target")
regressor.fit(X_train, regression_targets)
regression_predictions = regressor.predict(X_test)  # (n_test, n_tasks), float64
assert regression_predictions.shape == (len(X_test), 2)
assert np.isfinite(regression_predictions).all()

classifier = TabMClassifierMother(**demo_parameters, target_type="multi_target")
classifier.fit(X_train, binary_targets)
classification_predictions = classifier.predict(X_test)  # (n_test, n_tasks)
task_probabilities = classifier.predict_proba(X_test)
# List of (n_test, 2) probability arrays, one per task.
# classifier.classes_[task] gives the label order for that array.
assert classification_predictions.shape == (len(X_test), 2)
assert len(task_probabilities) == 2
for probabilities in task_probabilities:
    assert probabilities.shape == (len(X_test), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
```

In MotherSettings, set `model.target_type="multi_target"` and use
`model.model_type="regression"` or `"classification_binary"`. Constructor
parameters such as network width remain in `model.parameters`.

Missing labels (`NaN`, `None`, or pandas missing values) are masked, not imputed.
For each mini-batch, the loss first averages over the observed labels and
ensemble members within each task, then averages over tasks that have an
observed label in that batch. Tasks without labels in a batch contribute no
loss. This gives each present task equal weight rather than giving tasks with
more observed labels a larger weight.

Rows with no observed target labels are excluded from training with a warning,
before fitting feature preprocessing or PWL bins. Every regression task needs
at least one observed finite training value; each classification task needs both
classes in the training fold. Infinities are rejected. Numeric and categorical
**input features must still be nonmissing** and should be imputed upstream.

Regression target scaling uses observed values separately for each task and
keeps the missing-label mask intact. This also prevents a target with large
numeric units from dominating solely because of its scale.

Multitarget uncertainty outputs use `target_0_`, `target_1_`, etc., regardless of
the original target column names:

| Output | Multitarget regression | Multitarget binary classification |
|---|---|---|
| `target_i_pred` | Mean member prediction in task units | Decoded class label |
| `target_i_mean_predictions` | Same as `target_i_pred` | Unset |
| `target_i_proba_0`, `target_i_proba_1` | Not returned | Mean class probabilities in task class order |
| `target_i_knowledge_uncertainty` | Member standard deviation in task units | Member disagreement in nats |
| `target_i_data_uncertainty` | Unset | Mean member entropy in nats |
| `target_i_total_uncertainty` | Unset | Entropy of mean probabilities in nats |

`uncertainty_for_opt=True` returns only the per-task knowledge-uncertainty
columns; it does not collapse tasks into one acquisition value. Single-target
prediction shapes and unprefixed uncertainty column names are unchanged.

The estimators' `.score()` methods mask missing evaluation labels. Regression
returns the mean per-task R-squared and requires at least two observed labels
per evaluated task. Classification returns mean per-task accuracy, not
all-tasks-correct subset accuracy, and requires observed labels in each
evaluated task. Evaluation classes absent from training count as incorrect
predictions rather than raising a label-encoding error.
With complete single-target labels, these match the usual
sklearn scoring definitions.

For missing-label CV, use an appropriate splitter such as `KFold`, ensuring
each training fold contains enough observations/classes for every task.
Standard stratification cannot consume arbitrary missing multitask labels.
External sklearn scorers such as `"neg_mean_squared_error"` do not become
missing-label aware automatically. To use the estimator's masked scoring in
MotherTuner, pass a scorer callable:

```python
from mother.optimization.core import MotherTuner

def observed_target_score(estimator, X, y):
    return estimator.score(X, y)

tuner = MotherTuner(scorer=observed_target_score)
```

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
`num_emb_type="auto"` with MotherTuner to compare no embedding with the four
embedding types.

For example, PWL uses the same configuration as
[pytabkit's TabM adapter](https://github.com/dholzmueller/pytabkit/blob/main/pytabkit/models/alg_interfaces/tabm_interface.py):

```python
from mother.ml.models.m_tabm import TabMRegressorMother

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

Set `num_emb_type="auto"` to let MotherTuner choose among `"none"` (no
embedding), `"linear"`, `"linear_relu"`, `"periodic"`, and `"pwl"`, then tune
that type's settings. The
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
- `loss_function=None` (default) means you have not chosen a loss. Direct
  `.fit()` then trains with squared error for regression and cross-entropy for
  classification, and MotherTuner tunes the loss (see below). The alternatives
  mirror the losses MotherTuner searches for CatBoost. Regressors also accept
  `"mae"`, `"logcosh"`, `"huber"` (`huber_delta` > 0 in target standard
  deviations, default 1.0), and `"tweedie"` (`tweedie_variance_power` in (1, 2),
  default 1.5). Tweedie needs nonnegative targets with a positive value per
  task, such as skewed counts or amounts. Targets are then rescaled without
  centering, and member outputs are log means that are exponentiated before
  averaging. Classifiers also accept `"focal"` (`focal_alpha` in (0, 1),
  default 0.5, and positive `focal_gamma`, default 2.0) for binary tasks, which
  down-weights well-classified samples.
- Loss tuning follows CatBoost. The shared `get_hyperparameter_space` calls
    `suggested_params_loss`, which `TabMRegressorMother` and
    `TabMClassifierMother` each implement. What MotherTuner tunes depends on
    `loss_function` and `tune_loss_function`:

    | `loss_function` | `tune_loss_function=True` (default) | `tune_loss_function=False` |
    |---|---|---|
    | `None` (default) | Loss type **and** its parameters are tuned | Default loss (MSE / cross-entropy), nothing tuned |
    | Explicit, e.g. `"huber"` | Loss type fixed, only its own parameters tuned | Loss and its parameters fixed at the given values |
    | Custom callable | Used as given, never tuned | Used as given, never tuned |

    So setting a loss explicitly picks the loss but still tunes what it needs,
    and `tune_loss_function=False` pins a parameter such as `huber_delta`.
    When the type is tuned, each trial first picks a loss —
    `"mse"`, `"logcosh"`, `"huber"`, or `"tweedie"` for regression, and
    `"cross_entropy"` or `"focal"` for classification — and then tunes only that
    loss's parameters. MSE, MAE, log-cosh and cross-entropy have no parameters.
    MAE is not searched because its constant-size gradients
    train neural networks poorly; Huber with a small delta covers that robust
    regime. The Tweedie variance power uses CatBoost's range, 1.000001 to
    1.999999. Ranges for the other losses suit neural networks: `huber_delta`
    0.1 to 3 (log scale), `focal_alpha` 0.05 to 0.95, and `focal_gamma` 0.5 to 5.
    Tweedie is offered only for valid nonnegative targets and focal only when
    every task is binary (multiclass always uses cross-entropy). When tuning
    selects focal loss, class weighting is turned off. `default_parameters()`
    always includes `loss_function`, so the first default trial uses the default
    loss.
- `loss_function` also accepts your own callable `loss(predictions, targets)`.
  A custom loss is used as given and is not tuned. It must return one loss per
  target element, like PyTorch losses with `reduction="none"`; Mother then
  masks missing labels, applies class weights, and averages over members and
  tasks. Regression and multitarget classification pass
  `(rows, members, tasks)` outputs with float targets (standardized for
  regression); single-target classification passes `(rows * members, classes)`
  logits with integer class targets. Custom losses get no class weighting by
  default. Use a module-level function or `functools.partial` so fitted models
  can be pickled, for example
  `TabMRegressorMother(loss_function=partial(functional.huber_loss, reduction="none", delta=0.5))`
  with `from torch.nn import functional`.
- `TabMClassifierMother` accepts CatBoost's `auto_class_weights` (regressors
  do not offer it): `"Balanced"` weights each
  class by the largest class count divided by its count, `"SqrtBalanced"` uses
  the square root of that ratio, and `None` or `"None"` disables weighting.
  As with Mother's CatBoost classifier, omitting the option (the `"default"`
  value) uses `"Balanced"` for single-target cross-entropy and no weighting
  for focal loss or multitarget models. When
  tuning selects focal loss, class weighting is turned off, as for CatBoost.
  Explicit multitarget weights are computed per task. Weighting shifts
  predicted probabilities toward rarer classes.
- Each member is trained against the target separately. Regression predictions
  are averaged; single-target classification applies softmax to each member,
  while multitarget binary classification applies a sigmoid per task.
  Probabilities are averaged across members before decoding labels using
  `classes_`.
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

For single-target classification, the returned DataFrame preserves the input
index and contains `pred`, `proba_i` in `classes_` order, unset
`mean_predictions`, and the three uncertainty columns. Both binary and multiclass
classification are supported. Multitarget binary classification uses the
per-task columns documented above. No target labels are
needed; this uses predicted probabilities, not observed-label likelihood or
negative log-likelihood. `uncertainty_for_opt=True` returns only the
`knowledge_uncertainty` column for acquisition/optimization workflows.

Non-shared training batches are not exposed by this adapter. Numerical
embeddings remain opt-in as described above.

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
