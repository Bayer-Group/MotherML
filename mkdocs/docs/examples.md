# Mother tutorial

Here, you can find `.ipynb` files explaining how to effectively use Mother for chemical data procesing and model training:


| Category         | Tutorial Title                           | Description                                                                                                       | Link                                      |
|------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| Data Processing   | example_feature_selection.ipynb         | Use a default feature selection pipeline in Mother.                                                              | [View Tutorial](https://github.com/bayer-int/mother-ml/blob/master/examples/notebooks/example_feature_selection.ipynb)         |
| Data Processing   | example_custom_preprocessing.ipynb     | Design a custom preprocessing pipeline accordingly with Mother.                                                  | [View Tutorial](https://github.com/bayer-int/mother-ml/blob/master/examples/notebooks/example_custom_preprocessing.ipynb)       |
| Model Training    | example_regression.ipynb               | Predict a continuous target value from molecular structure using CatboostRegressor implemented in Mother. This tutorial includes group k-fold cross-validation. | [View Tutorial](https://github.com/bayer-int/mother-ml/blob/master/examples/notebooks/example_regression.ipynb)                 |
| Model Training    | Classification (coming soon)           | Tutorial for future classification tasks.                                                                         | [View Tutorial](https://example.com/classification_coming_soon)               |
| Optimisation      | example_pipeline_with_settings.ipynb    | Use MotherSettings to create a pipeline with Mother, including data preprocessing, feature generation and selection, and model training. | [View Tutorial](https://github.com/bayer-int/mother-ml/blob/master/examples/notebooks/example_pipeline_with_settings.ipynb)     |
| Optimisation      | example_hyperparameter_optimization.ipynb | Customize hyperparameter optimization with Mother, following a baseline design of Optuna.                        | [View Tutorial](https://github.com/bayer-int/mother-ml/blob/master/examples/notebooks/custom_hyperparameter_optimization.ipynb) |


Use [MotherSettings](https://github.com/bayer-int/mother-ml/blob/master/mother/settings.py) to create a pipeline with Mother.

The pipeline can include all: data preprocessing, feature generation and selection, and model training.
  - `example_hyperparameter_optimization.ipynb`

Customise a hyperparameter optimisation with Mother.

This follows a baseline design of [Optuna](https://github.com/optuna/optuna).
