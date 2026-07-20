# Mother tutorial

Here, you can find `.ipynb` files explaining how to effectively use Mother for chemical data procesing and model training:


| Category         | Tutorial Title                           | Description                                                                                                       | Link                                      |
|------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| Data Processing   | 01_feature_selection.ipynb         | Use a default feature selection pipeline in Mother.                                                              | [View Tutorial](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/04_feature_engineering/01_feature_selection.ipynb)         |
| Data Processing   | 03_custom_preprocessing.ipynb     | Design a custom preprocessing pipeline accordingly with Mother.                                                  | [View Tutorial](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/01_basics/03_custom_preprocessing.ipynb)       |
| Model Training    | 01_basic_regression.ipynb               | Predict a continuous target value from molecular structure using CatboostRegressor implemented in Mother. This tutorial includes group k-fold cross-validation. | [View Tutorial](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/02_regression/01_basic_regression.ipynb)                 |
| Model Training    | 06_predict_predict_uncertainty_interface.ipynb | Demonstrates the unified `predict` and `predict_uncertainty` interface, including uncertainty outputs and model-specific behavior. | [View Tutorial](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/01_basics/06_predict_predict_uncertainty_interface.ipynb) |
| Model Training    | 01_lasso_classification.ipynb           | Classify molecules using a Lasso-based model in Mother.                                                          | [View Tutorial](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/03_classification/01_lasso_classification.ipynb)               |
| Optimisation      | 04_pipeline_with_settings.ipynb    | Use MotherSettings to create a pipeline with Mother, including data preprocessing, feature generation and selection, and model training. | [View Tutorial](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/01_basics/04_pipeline_with_settings.ipynb)     |
| Optimisation      | 01_custom_hyperparameter_optimization.ipynb | Customize hyperparameter optimization with Mother, following a baseline design of Optuna.                        | [View Tutorial](https://github.com/Bayer-Group/MotherML/blob/main/examples/notebooks/05_advanced/01_custom_hyperparameter_optimization.ipynb) |


Use [MotherSettings](https://github.com/Bayer-Group/MotherML/blob/main/src/mother/settings.py) to create a pipeline with Mother.

The pipeline can include all: data preprocessing, feature generation and selection, and model training.

Customise a hyperparameter optimisation with Mother.

This follows a baseline design of [Optuna](https://github.com/optuna/optuna).
