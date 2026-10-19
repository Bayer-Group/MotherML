# Using mother in your ML project

Here, I want to give an overview on how mother can be used in a project scope. As mentioned earlier, multiple examples can be found in the [examples folder](https://github.com/Bayer-Group/MotherML/tree/main/examples/notebooks). The beauty of the current mother implementation is, that every step can be performed individually (in case you have the appropriate input for the step).
To achieve that, the code of each step is put into separate submodules. Examples for the most important submodules will be given below. A comprehensive overview can be found in the examples.

To faciliatate configuration of the ML project a `MotherSettings` class is created. In the following usage of that settings class (that is based on a simple yaml file) is shown. Project specific configuration files can be loaded as well. The example below shows the python code in the 'Source' tab and the result(print) in the 'Result' tab.

!!! example

    === "Complete Training or Inference Pipeline"

        ~~~python exec="on" source="tabbed-left"
        import typing
        from mother import pipeline_utils as mother_takes_care
        from mother import ml
        from mother.settings import MotherSettings
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn import set_config
        import sklearn.model_selection as skl_model_sel
        mother_settings: MotherSettings = MotherSettings.create()
        model_steps: typing.Sequence[
            typing.Tuple[str, typing.Union[Pipeline, FeatureUnion, ml.AbstractMotherPipeline, typing.Any]]
        ] = []
        model_steps.append(("preprocessor", mother_takes_care.get_preprocessing_pipeline(settings=mother_settings)))
        model_steps.append(("feature_generator", mother_takes_care.get_feature_generation_pipeline(settings=mother_settings)))
        model_steps.append(("feature_selector", mother_takes_care.get_feature_selection_pipeline(settings=mother_settings, cv=skl_model_sel.GroupKFold(n_splits=5))))
        model_steps.append(("model", mother_takes_care.get_model(settings=mother_settings)))
        training_pipeline: Pipeline = Pipeline(steps=model_steps)
        training_pipeline.set_output(transform="pandas")
        print(training_pipeline)
        # training_pipeline.fit(X_train, y_train)
        #inference_pipeline = training_pipeline
        # inference_pipeline.predict(["CCC"])
        ~~~

    === "Configuration"

        ~~~python exec="on" source="tabbed-left"
        from mother.settings import MotherSettings
        # generate some default settings
        my_settings: MotherSettings = MotherSettings.create()
        # settings override example
        my_settings.preprocessing.flags = ["STANDARDIZE", "NEUTRALIZE", "FLATTEN_STEREOCHEMISTRY"]
        print(my_settings)
        ~~~

    === "Preprocessing"

        ~~~python exec="on" source="tabbed-left"
        from mother.preprocessing import PreprocessingConfig, SmilesToMolTransformer, StandardizerTransformer
        from sklearn import pipeline as sklearn_pipeline
        from mother.settings import MotherSettings
        my_settings: MotherSettings = MotherSettings.create()
        preprocessor: sklearn_pipeline.Pipeline = sklearn_pipeline.Pipeline(
            [
                (
                    "smiles_standardizer",
                    StandardizerTransformer(**my_settings.preprocessing.model_dump()),
                ),
                ("smiles_to_mol", SmilesToMolTransformer()),
                # Add other column transformations here if needed
            ],
            memory=None,
        ).set_output(transform="pandas")
        print(preprocessor)
        ~~~

    === "Feature Generation"

        ~~~python exec="on" source="tabbed-left"
        from mother.feature_generation import ChemicalDescriptors, MaccsFingerprints, MorganFingerprints
        from mother.settings import MotherSettings
        my_settings: MotherSettings = MotherSettings.create()
        from sklearn import pipeline as sklearn_pipeline
        feature_generator = sklearn_pipeline.FeatureUnion(transformer_list=[
            ("maccs", MaccsFingerprints()),
            ("morgan", MorganFingerprints(my_settings.feature_generation.fingerprints[0]["MorganFP"])),
            ("desc", ChemicalDescriptors(**my_settings.feature_generation.chemical_descriptors.model_dump())),
            ],
        ).set_output(transform="pandas")
        print(feature_generator)
        ~~~

    === "Feature Selection and Model Training"

        ~~~python exec="on" source="tabbed-left"
        from sklearn import pipeline as sklearn_pipeline
        import sklearn.model_selection as skl_model_sel
        import mother.pipeline_utils as mother_takes_care
        from mother import ml
        from mother.settings import MotherSettings
        my_settings: MotherSettings = MotherSettings.create()
        model = sklearn_pipeline.FeatureUnion(transformer_list=[
        (
            "feature_selector",
            mother_takes_care.get_feature_selection_pipeline(
                settings=my_settings,
                cv=skl_model_sel.GroupKFold(n_splits=5),
            ).set_output(transform="pandas"),
        ),
        ("ml_model", ml.CatboostRegressorMother(target_type=my_settings.model.target_type, logging_level="Silent")),
        ])
        print(model)
        ~~~

## MotherSettings class

The mother settings class can be used as convenience to store all project relevant settings in one place. To jump start your project a yaml file containing some defaults (examples) is provided. The MotherSettings class is based on [pydantic models](https://docs.pydantic.dev/latest/concepts/models/). This design choice was made to validate the input directly. Thus, avoiding a lot of if/else cases in the source code to handle user errors by wrong configuration. Thus, using the MotherSettings class is highly recommended and avoids a lot of configuration isues. Nevertheless, every transformer and a complete training pipeline can be created with this class.

::: mother.settings.MotherSettings

To dump the default configuration as yaml file and to be able to modify it to your needs you can use the provided function described above. In the same way, the created yaml file can be loaded again (see above).

### Configuration Subsections

In the following every highlighted subsection represents the settings for a different transformer. A specialty is the `model` section which contains also the information for feature selection which can be used to create a more complex feature selection pipeline. For example, `pipeline` contains all the required parameters to configure a scikit learn pipeline. Thus, parameters can be set once and reused at different stages of your pipeline.

### Native CatBoost feature types

Declare native feature columns in the `model` section using the same list-of-column-names interface:

```yaml
model:
  algorithm: catboost
  categorical_features: ["category"]
  text_features: ["description"]
  embedding_features: ["embedding"]
```

The example YAML template uses `["cat_col"]`, `["text_col"]`, and
`["embedding_col"]`. Replace these placeholders with your own columns, or use
`[]` for feature types you do not have. When constructing `ModelConfig` directly,
omitted lists default to `[]`. Column names must refer to the data reaching the
estimator, not columns discarded by an earlier preprocessing or feature-generation step.

| Setting | Value in each DataFrame cell | CatBoost parameter |
|---|---|---|
| `categorical_features` | Category label, such as a string or integer | `cat_features` |
| `text_features` | Raw text string | `text_features` |
| `embedding_features` | Fixed-length numeric vector, such as a list or NumPy array | `embedding_features` |

An embedding column contains **one vector per row**. It is not a string containing
a serialized vector or a list of separate scalar feature columns. Already expanded
numeric descriptors or fingerprint dimensions can remain ordinary numeric features.
Mother does not automatically generate embeddings or convert raw text into vectors;
CatBoost processes the declared native inputs.

```python
from mother.settings import MotherSettings
from mother.pipeline_utils import get_model

native_settings = MotherSettings.create()
native_settings.model.algorithm = "catboost"
native_settings.model.categorical_features = ["category"]
native_settings.model.text_features = ["description"]
native_settings.model.embedding_features = ["embedding"]
native_settings.pipeline.remainder = "passthrough"
native_settings.pipeline.verbose_feature_names_out = False
native_model = get_model(native_settings)
assert native_model.get_params()["text_features"] == ["description"]
assert native_model.get_params()["embedding_features"] == ["embedding"]
```

The feature-selection builder applies numeric selection to numeric columns;
string categories, text, and vector-valued embedding columns pass through the
remainder. Use `remainder="passthrough"`, pandas output
(`.set_output(transform="pandas")`), and `verbose_feature_names_out=False` to retain
those columns and their original names for CatBoost. `IMPUTE_CATEGORICAL` still
applies only to categorical columns; supply valid, nonmissing text and embedding
values yourself.

Direct CatBoost declarations under `model.parameters` remain supported:
`cat_features`, `text_features`, and `embedding_features`. Use column names rather
than positional indices when feature selection can reorder or remove columns.
Explicit declarations in `model.parameters` take precedence over the corresponding
top-level lists. Keep the top-level lists consistent with the columns passed
through feature selection. These settings are used for CatBoost; they do not
add native text or embedding support to other models.
CatBoost's own restrictions for the chosen task, loss, and processing options
still apply.

~~~yaml hl_lines="8 15 18 31 36 45" title="mother/data/mother_config.yaml"
--8<-- "src/mother/data/mother_config.yaml"
~~~
