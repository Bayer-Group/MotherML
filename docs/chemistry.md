# Introduction to Chemistry for Machine Learning

Welcome to the beginner's guide to chemistry for machine learning! This page is designed to help you understand the basic chemical concepts and terminologies you'll encounter while working with the **Mother** framework. Whether you're a data scientist new to chemistry or just need a refresher, this guide will get you started.

---

## What is Chemistry?

Chemistry is the study of matter, its properties, and how it interacts with other matter and energy. In the context of machine learning, we often work with chemical data to predict properties, classify molecules, or design new compounds.

---

## Key Concepts in Chemistry

### 1. **Molecules and SMILES**
- **Molecules** are groups of atoms bonded together. They are the building blocks of matter.
- **SMILES (Simplified Molecular Input Line Entry System)** is a text-based representation of molecules. For example:
  - Water: `O`
  - Ethanol: `CCO`
  - Benzene: `c1ccccc1`

The **Mother** framework uses SMILES strings as input to preprocess and generate molecular features.

---

### 2. **Atoms and Bonds**
- **Atoms** are the smallest units of matter, such as Carbon (C), Hydrogen (H), Oxygen (O), and Nitrogen (N).
- **Bonds** are connections between atoms:
  - **Single bonds**: `-`
  - **Double bonds**: `=`
  - **Triple bonds**: `#`

Understanding bonds is crucial for interpreting molecular structures.

---

### 3. **Descriptors**
Descriptors are numerical values that represent chemical properties of molecules. These are used as features in machine learning models. Examples include:

- **Molecular weight**: The sum of the atomic weights of all atoms in a molecule.
- **LogP**: A measure of a molecule's hydrophobicity (how it interacts with water).
- **Number of hydrogen bond donors/acceptors**: Important for biological activity.

The **Mother** framework calculates these descriptors using tools like RDKit.

---

### 4. **Chemical Fingerprints**
Chemical fingerprints are binary vectors that encode the presence or absence of specific substructures in a molecule. Common types include:

- **Morgan fingerprints**: Circular fingerprints that capture molecular substructures.
- **MACCS keys**: A predefined set of 166 structural keys.

These fingerprints are used for tasks like similarity searches and feature generation.

---

## How Chemistry Fits into Machine Learning

In the **Mother** framework, chemistry concepts are integrated into the machine learning pipeline:

1. **Preprocessing**: Convert SMILES strings into molecular objects
  ([Read more about molecule standardization](standardization.md)).

2. **Feature Generation**: Extract descriptors and fingerprints
  ([Read more about feature generation](feature_generation.md)).
3. **Model Training**: Use these features to train machine learning models for tasks like property prediction or classification.

---

## Additional Useful Resources

| Tutorial Title                               | Author        | Description                                                                      | Link                                                                                       |
|----------------------------------------------|---------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Cheminformatics Tutorials                    | Pat Walters   | A beginner-friendly guide covering essential topics in cheminformatics.        | [View Tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials)     |
| Talktorials on Computer-Aided Drug Design    | VolkamerLab   | Covers topics related to ligand-based and structure-based compound discovery.    | [View Talktorials](https://projects.volkamerlab.org/teachopencadd/talktorials.html)      |

---
## Example: Predicting Solubility

Let’s walk through a simple example:

1. **Input**: A dataset of molecules represented as SMILES strings.
2. **Preprocessing**: Standardize the SMILES and convert them into molecular objects.
3. **Feature Generation**: Calculate descriptors like molecular weight and LogP.
4. **Model Training**: Train a regression model to predict solubility.

```python
import pandas as pd
from sklearn import pipeline as sklearn_pipeline
from mother.preprocessing import SmilesToMolTransformer, StandardizerTransformer
from mother.feature_generation import ChemicalDescriptors
from mother.ml import CatboostRegressorMother

smiles_data = pd.DataFrame(
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
target = pd.Series([0.2, 0.4, 0.7, 1.1, 1.5, 2.0], name="solubility")

# Preprocessing pipeline
preprocessor = sklearn_pipeline.Pipeline([
  (
    "standardizer",
    StandardizerTransformer(
      flags=["STANDARDIZE", "DESALT", "NEUTRALIZE"],
      smiles_col="smiles",
    ),
  ),
  ("smiles_to_mol", SmilesToMolTransformer(molecule_col="Molecule"))
]).set_output(transform="pandas")

# Feature generation
feature_generator = sklearn_pipeline.FeatureUnion([
    ("descriptors", ChemicalDescriptors(descriptor_list=["MolWt", "MolLogP"]))
])

regression_model = sklearn_pipeline.Pipeline([
  (
    "regressor",
    CatboostRegressorMother(
      target_type="single_target",
      logging_level="Silent",
      random_seed=42,
      iterations=10,
    ),
  ),
    ]
)

# Example workflow
mol_data = preprocessor.fit_transform(smiles_data)
features = feature_generator.fit_transform(mol_data["Molecule"])

# fit the model to data
regression_model.fit(features, target)
targets_pred = regression_model.predict(features)
```

<div align="right">
  <a href="../standardization">Next: Molecule Standardisation &rarr;</a>
</div>
