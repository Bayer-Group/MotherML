import numpy as np
import pandas as pd
import pytest
import sklearn.base as skl_base
from rdkit import Chem
from sklearn import exceptions
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion

from mother.feature_generation import config as fg_config
from mother.feature_generation.core import (
    ChemicalDescriptors,
    MaccsFingerprints,
    MorganFingerprints,
)


@pytest.fixture(
    params=[
        "ChemicalDescriptors",
        "MorganFingerprints",
        "MaccsFingerprints",
    ]
)
def transformer(request):
    if request.param == "ChemicalDescriptors":
        return ChemicalDescriptors, fg_config.ChemicalDescriptorsParams
    elif request.param == "MorganFingerprints":
        return MorganFingerprints, fg_config.MorganFingerprintsParams
    elif request.param == "MaccsFingerprints":
        return MaccsFingerprints, fg_config.MaccsFingerprintsParams


class TestTransformers:
    def test_transformers(self, transformer, mols):
        transformer, config = transformer

        # initialize with default parameters
        conf: fg_config.FingerprintParams = config()
        fg = transformer(**conf.model_dump())

        # fit and transfirm
        fg.fit()
        features = fg.transform(mols)
        assert isinstance(features, np.ndarray)

        # set output to pandas
        fg.set_output(transform="pandas")
        features = fg.fit_transform(mols)
        for col in features.columns:
            assert features[col].notna().all
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == len(mols)

    def test_set_params(self, mols):
        fg = MorganFingerprints(radius=2)
        features_radius_2 = fg.fit_transform(mols)

        fg.set_params(radius=3)
        features_radius_3 = fg.fit_transform(mols)
        assert not np.array_equal(features_radius_2, features_radius_3)

        fg.set_params(fpSize=64)
        features_fpSize_64 = fg.fit_transform(mols)
        assert features_fpSize_64.shape[1] == 64

    def test_invalid_molecule(self, transformer, invalid_compounds):
        transformer, config = transformer

        # initialize with default parameters
        conf: fg_config.FingerprintParams = config()
        fg = transformer(**conf.model_dump())

        # fit and transfosrm
        fg.fit()
        features = fg.transform(invalid_compounds)
        assert np.isnan(features[-1]).all()


@pytest.fixture
def feature_generator():
    return FeatureUnion(
        [
            ("maccs", MaccsFingerprints(**fg_config.MaccsFingerprintsParams().model_dump())),
            ("morgan", MorganFingerprints(**fg_config.MorganFingerprintsParams().model_dump())),
            ("desc", ChemicalDescriptors(**fg_config.ChemicalDescriptorsParams().model_dump())),
        ]
    )


class TestFeatureGenerator:
    def test_assert_is_fitted(self, feature_generator, mols):
        # assert is fitted
        with pytest.raises(exceptions.NotFittedError):
            feature_generator.transform(mols)

    def test_training_and_format(self, feature_generator, mols):
        feature_generator.fit(mols)
        features = feature_generator.transform(mols)
        assert isinstance(features, np.ndarray)

        feature_generator.set_output(transform="pandas")
        features = feature_generator.fit_transform(mols)
        assert isinstance(features, pd.DataFrame)


def test_clone_works():
    model = ChemicalDescriptors(descriptor_list=["MolWt", "MolLogP"])
    cloned_model = skl_base.clone(model)
    assert model.descriptor_list == cloned_model.descriptor_list


def test_chemical_descriptors_column_transformer():
    """
    Test if the ChemicalDescriptors class is compatible with ColumnTransformer.
    """
    # Initialize the ChemicalDescriptors instance
    descriptor_transformer = ChemicalDescriptors(
        descriptor_list=["MolWt", "MolLogP"]  # Example subset of descriptors
    )

    # Create a ColumnTransformer
    column_transformer = ColumnTransformer(
        transformers=[
            ("chemical_features", descriptor_transformer, "Molecule"),
        ],
        remainder="drop",
    )

    # Example data as a pandas DataFrame
    data = pd.DataFrame(
        {
            "Molecule": [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")],
        }
    )

    # Fit and transform the data
    transformed_features = column_transformer.fit_transform(data)

    # Check if the output is a valid numpy array
    assert transformed_features is not None, "Transformed features are None"
    assert transformed_features.shape[1] == len(descriptor_transformer.descriptor_list), (
        "Number of output features does not match the number of descriptors"
    )


def test_check_descriptor_list():
    """
    Test the _check_descriptor_list method of the ChemicalDescriptors class.
    """
    # Case 1: descriptor_list is None
    descriptor_transformer = ChemicalDescriptors()
    with pytest.raises(NotFittedError, match="This ChemicalDescriptors instance is not fitted yet."):
        descriptor_transformer._check_descriptor_list()

    # Case 2: descriptor_list is an empty list
    descriptor_transformer = ChemicalDescriptors(descriptor_list=[])
    with pytest.raises(NotFittedError, match="This ChemicalDescriptors instance is not fitted yet."):
        descriptor_transformer._check_descriptor_list()

    # Case 3: descriptor_list is properly set
    descriptor_transformer = ChemicalDescriptors(descriptor_list=["MolWt", "MolLogP"])
    try:
        descriptor_transformer._check_descriptor_list()  # Should not raise an error
    except NotFittedError:
        pytest.fail("_check_descriptor_list raised NotFittedError unexpectedly.")

    print("All tests for _check_descriptor_list passed.")
