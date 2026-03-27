import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from mother.chem import calculate_descriptor, maccs


def test_maccs_fingerprint():
    # Test with a simple molecule
    mol = Chem.MolFromSmiles("CCO")
    fingerprint = maccs(mol)

    assert isinstance(fingerprint, np.ndarray)
    assert fingerprint.shape == (167,)
    assert np.sum(fingerprint) > 0


def test_calculate_descriptor():
    # Test with a simple molecule and a known descriptor function
    mol = Chem.MolFromSmiles("CCO")
    result = calculate_descriptor(Descriptors.MolWt, mol)

    assert isinstance(result, (np.int32, np.float32, float, int))
    assert result == 46.069

    # Test with a molecule that causes an infinite value
    mol = Chem.MolFromSmiles("C1=CC=CC=C1C(=O)O")
    result = calculate_descriptor(lambda x: float("inf"), mol)

    assert np.isnan(result)

    # Test with a molecule that causes a very large value
