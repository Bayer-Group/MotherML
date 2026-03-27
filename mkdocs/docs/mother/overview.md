# Overview of the Mother Framework

The Mother framework is a comprehensive machine learning (ML) framework designed to streamline the process of model training, inference, and configuration management. It leverages modern Python practices and libraries to provide a robust platform for data scientists working on various ML projects. Below is an overview of the key functionalities and how data scientists can utilize the Mother framework to enhance their ML workflows.

## Key Features

- **Scikit learn like**: To be as compatible as possible to standard ML frameworks and to reduce the learning curve, every step in the mother pipeline is implemented as [transformer](https://scikit-learn.org/stable/api/sklearn.base.html).

- **Flexible Configuration Management**: Utilizes [pydantic](https://docs.pydantic.dev/latest/) for settings validation and organization. However, all transformers can be set up using plain dictionaries.

## Module structure

A ML project usually contains multiple steps like:

- preprocessing
- feature generation
- feature selection and
- model training

Thus, every step of this can be part of a pipeline. To reflect this structure in the mother package, every step is a submodule of mother. This is done to provide a clear structure to users and developers as well.
