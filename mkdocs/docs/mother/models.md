# Using different Models

Although mother is build around catboost it basically supports other models from the ML community (like all sklearn estimators). For example, `RandomForest` is already supported. Furthermore, [own models](#providing-your-own-model) can be provided.

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
