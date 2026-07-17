# Mother-ML

A ML framework that takes care.

Mother is a machine-learning framework for predicting properties from chemical molecules. The major features are:

- :microscope: **SMILES** preprocessing
- :floppy_disk: Generating of **feature vectors** from molecules
- :chart_with_upwards_trend: Grouping and cross-validation, based on chemical similarity
- :computer: Model Training: Standard catboost models, and feature selection methods
- :bicyclist: Training, cross-validation, and hyperparameter optimization of machine-learning models
- :cyclone: Handling Gene expression data from transcriptomics experiments including different normalisation techniques
- :sparkles: <s> Explainability analysis with *SHAP* </s> (Currently not supported, will be added in a later release)
- :blue_car: <s> Generative chemistry </s> (Currently not supported)


Mother provides methods for each of these steps in the form of sklearn transformer objects. By that, all methods are designed to be easily accessible and usable in a modular way. The methods can be combined to ML workflows with [sklearn pipelines, column transformers, and feature unions](https://scikit-learn.org/dev/modules/compose.html).

All methods can be used as sklearn `transformer` or `estimator`. Combination with other methods, or own methods and models (e.g. using mother preprocessing with other model) is therefore straightforward. To be as compatible as possible, every transformer can be constructed using a dictionary containing the required parameters. However, to provide some convenience to the users, a settings class [MotherSettings](../../src/mother/settings.py). This class can be used to store all relevant settings for your ML project.

## Usage

A basic example can be found in the [example regression notebook](/examples/notebooks/example_regression.ipynb). Other
examples are in the [examples folder](/examples/notebooks).

### :microscope: SMILES preprocessing and mol-object generation

SMILES preprocessing is done with the `StandardizerTransformer` class. The class is used to preprocess SMILES strings to construct a pipeline from SMILES to rdkit mol-objects with:

```python
import pandas as pd
from sklearn import pipeline as sklearn_pipeline
from mother.preprocessing.core import SmilesToMolTransformer, StandardizerTransformer

structure_data = pd.DataFrame(
    {
        "smiles": [
            "CCO",
            "CCN",
            "c1ccccc1",
            "CC(=O)O",
            "CC(C)O",
            "CCCC",
        ]
    }
)

preprocessor: sklearn_pipeline.Pipeline = sklearn_pipeline.Pipeline(
    [
        (
            "smiles_standardizer",
            StandardizerTransformer(
                flags=["STANDARDIZE", "DESALT", "NEUTRALIZE"],
                smiles_col="smiles",
            ),
        ),
        ("smiles_to_mol", SmilesToMolTransformer(molecule_col="Molecule")),
        # Add other column transformations here if needed
    ],
    memory=None,
).set_output(transform="pandas")

mol_data: pd.DataFrame = preprocessor.fit_transform(structure_data)

```

Customize by changing the `flags` attribute.

### :floppy_disk: Feature Generation

Mother provides three types of feature generators: `MaccsFingerprints`, `MorganFingerprints`, and `ChemicalDescriptors`:

```python
from sklearn import pipeline as sklearn_pipeline
from mother.feature_generation.core import (
    ChemicalDescriptors,
    MaccsFingerprints,
    MorganFingerprints,
)

feature_generator = sklearn_pipeline.FeatureUnion(
    transformer_list=[
        ("maccs", MaccsFingerprints()),
        ("morgan", MorganFingerprints()),
        ("desc", ChemicalDescriptors()),
    ],
).set_output(transform="pandas")

features: pd.DataFrame = feature_generator.fit_transform(mol_data["Molecule"])

```

The `FeatureUnion` class is used to combine the feature generators. Each feature generator can be configured.

### :chart_with_upwards_trend: Grouping and Cross-Validation

For cross-validation, or test-set selection based on chemical similarity, mother provides a transformer-class for
generating groups (`TanimotoGroupingFromMols`):

```python
import mother.cv as cv_module

groups_engine = cv_module.TanimotoGroupingFromMols(similarity_threshold=0.3)

groups: pd.DataFrame = groups_engine.set_output(transform="pandas").fit_transform(mol_data["Molecule"])

```

These groups can be used, e.g. in the `GroupKFold` class from the `sklearn.model_selection` module:

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=3)
```

### :computer: Model Training

The standard model setup of Mother consists of a `feature selection`, and a classification- or regression
model. Both are based on `Catboost`. The standard setup for a regression task would be:

```python
import mother.pipeline_utils as mother_takes_care
from mother import ml

model_settings = {
    "feature_selection_flags": ["DROP_CORRELATED", "DROP_CONSTANT", "DROP_DUPLICATES"],
    "correlation_threshold": 0.9,
    "categorical_features": [],
    "feature_selection_type": "catboost",
    "model_type": "regression",
    "target_type": "single_target",
}
pipeline_settings = {
    "remainder": "drop",
    "verbose_feature_names_out": False,
}

model = ml.PipelineWithHyperparameterRooting(
    [
        (
            "feature_selector",
            mother_takes_care.get_feature_selection_pipeline(
                settings=model_settings,
                pipeline_settings=pipeline_settings,
                data=features,
                cv=cv,
            ).set_output(transform="pandas"),
        ),
        (
            "ml_model",
            ml.CatboostRegressorMother(
                target_type="single_target",
                logging_level="Silent",
                random_seed=42,
                iterations=10,
            ),
        ),
    ]
)

targets = pd.Series([0.2, 0.4, 0.7, 1.1, 1.5, 2.0], name="target")
model.fit(features, targets)
```

Here, we use the extended sklearn pipeline `PipelineWithHyperparameterRooting` for some additional methods for hyperparameter
tuning.

Without feature selection, this is simplified:

```python
model = ml.CatboostRegressorMother(target_type="single_target", logging_level="Silent")
```

Any other sklearn model, or own model can be used instead of `CatboostRegressorMother`. An example, on how a custom
preprocessing step is added to the model, can be found in the
[example notebook on custom preprocessing](/examples/notebooks/example_custom_preprocessing.ipynb).

### Cross-validation

Having used any sklearn `pipeline`, or sklearn `estimator` or `transformer` classes, we can use the sklearn
methods for e.g. cross-validation (`cross_validate`):

```python
from sklearn.model_selection import cross_validate

cross_validate(model, features, targets, groups=groups.values.ravel(), cv=cv, n_jobs=1)
```

A more convenient method is provided by mother. Using this methods gives you additional output considering CV and groups.

```python
import mother.pipeline_utils as mother_takes_care

mother_takes_care.mother_cv(
    estimator=model,
    X=features,
    y=targets,
    groups=groups,
    cv=cv,
)
```

### :bicyclist: Hyperparameter Optimization

The Mother object `MotherTuner` uses optuna to optimize hyperparameters:

```python
import mother.optimization as opt

tuner = opt.MotherTuner(
    scorer="r2",
    n_trials_optuna=2,
    n_threads_optuna=1,
)

model_tuned = tuner.optimize(
    model,
    features,
    targets,
    cv,
    groups=groups.values.ravel(),
)
```

The function `model.get_hyperparameter_space` returns the hyperparameter space for the model. For the default
catboost model, and the `PipelineWithHyperparameterRooting` class, this is already implemented.

For examples, on how to customize the hyperparameter optimization, or define hyperparameters for your own
models, see the [example notebook](/examples/notebooks/custom_hyperparameter_optimization.ipynb).

### :cyclone: Handling Gene expression data from transcriptomics experiments including different normalisation techniques

The RNA processing pipeline is implemented in the RNA class, which incorporates various preprocessing steps tailored for RNA sequencing data. All RNA code can be found in the rna.py file.

The pipeline includes normalization, feature selection, and discretization, utilizing the power of the scikit-learn framework. The normalization methods available are "Scanpy," "UQ," "CUF," and "CPM.". You can customise the pipeline to your needs, or try different normalisation
methods and bin sizes in hyper-parameter tuning. The pipeline can be fitted and re-applied to avoid data-leakage during the normalisation.

Here’s how to set up and use the RNA processing pipeline:

```python
from mother.ml.rna import RNA
import numpy as np
import pandas as pd

rna_model = RNA(
    n_features=3,  # Number of features (=genes) to keep for the prediction.
    n_bins=20,  # Number of bins to use for the discretisation of the target variable.
    normalisation_method="UQ",  # Which normalisation to use
)

rng = np.random.default_rng(42)
rna_data_train = pd.DataFrame(
    rng.integers(0, 200, size=(20, 8)),
    columns=[f"gene_{i}" for i in range(8)],
)
rna_data_test = pd.DataFrame(
    rng.integers(0, 200, size=(5, 8)),
    columns=rna_data_train.columns,
)
y_train = (rna_data_train["gene_0"] > 100).astype(int).rename("class")

# Fit the pipeline to your RNA sequencing data
transformed_train_data: pd.DataFrame = rna_model.fit_transform(rna_data_train, y_train)
transformed_test_data: pd.DataFrame = rna_model.transform(rna_data_test)

```

A complete walkthrough of the RNA functionality is found in the [example notebook](/examples/notebooks/example_rna_preprocessing.ipynb).

## Install

uv add mother-ml

### Optional Features and Extras

To keep the package size small, some dependencies are added as optional extras. These extras provide additional functionality for specific use cases:

| Extra | Description | Key Packages | Notes |
|-------|-------------|--------------|-------|
| `all` | All optional features | All packages below | Installs everything |
| `report` | Visualization and reporting tools | plotly, kaleido | For generating plots and reports |
| `rna` | RNA sequence analysis | rnalib | RNA-specific preprocessing |
| `torch` | PyTorch neural network support | torch, pytorch-tabular | **Adds ~3GB to environment size!** |
| `tabpfn` | TabPFN model support | tabpfn | Prior-fitted networks for tabular data |
| `clustering` | Chemical compound clustering | mol2vec, cluster-my-molecules | For molecular clustering analysis |

#### Installation Examples

**Using pip:**

```bash
# Install with report generation support
pip install 'mother[report]'

# Install with PyTorch support (adds ~3GB!)
pip install 'mother[torch]'

# Install multiple extras
pip install 'mother[report,torch,tabpfn]'
```

**Using uv:**

```bash
# Install with specific extras
uv add mother --extra report --extra torch
```

> **Note:** There is also a different `mother` package on PyPI. Be sure to install `mother-ml`.

## Acknowledgements

Thank you to the following contributors:

- Thomas Wolf
- Lukas Hebing
- Kai Sommer

and all the others.
