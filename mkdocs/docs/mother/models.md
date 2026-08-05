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
{%
    include "../../../src/mother/ml/models/m_lasso.py"
%}
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
