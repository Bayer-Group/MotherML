import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFG
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mother.feature_generation.fp_gen import FingerprintFactory
from mother.feature_generation.fp_gnn_gen import CheMeleonFingerprintTransformer


def test_initialization() -> None:
    # Test valid initialization
    params = {"n_bits": 1024, "countSimulation": True}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    assert factory.fp_type == "MorganFP"
    assert factory.parameters["fpSize"] == 1024
    assert factory.parameters["countSimulation"] is True

    # Test renaming of n_bits to fpSize
    assert "n_bits" not in factory.parameters
    assert "fpSize" in factory.parameters

    # Test default values
    params = {}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    assert factory.parameters["fpSize"] == 2048
    assert factory.parameters["countSimulation"] is False

    # Test invalid fingerprint type
    with pytest.raises(ValueError, match="Unknown fingerprint type: InvalidFP"):
        FingerprintFactory(fp_type="InvalidFP", parameters=params)


@pytest.mark.parametrize("fp_type", ["MorganFP", "AtomPairFP", "RDKitFP", "TopologicalTorsionFP"])
def test_get_fingerprint_generator(fp_type) -> None:
    params = {"fpSize": 1024, "countSimulation": True}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    generator = factory.get_fingerprint_generator()
    assert isinstance(generator, rdFG.FingerprintGenerator64)


def test_get_fingerprint_generator_raises() -> None:
    with pytest.raises(
        ValueError,
    ):
        params = {"fpSize": 1024, "countSimulation": True}
        FingerprintFactory(fp_type="InvalidFP", parameters=params)


def test_fpSize_property() -> None:
    params = {"fpSize": 1024}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    assert factory.fpSize == 1024


def test_countSimulation_property() -> None:
    params = {"countSimulation": True}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    assert factory.countSimulation is True


def test_is_supported() -> None:
    assert FingerprintFactory.is_supported("MorganFP") is True
    assert FingerprintFactory.is_supported("InvalidFP") is False


def test_chemeleon_transform_flattens_single_column_object_array() -> None:
    molecule = Chem.MolFromSmiles("CCO")
    transformer = CheMeleonFingerprintTransformer(
        output_dim=2,
        embedder=lambda smiles: np.ones((len(smiles), 2), dtype=np.float32),
    ).fit([molecule])

    result = transformer.transform(np.array([[molecule]], dtype=object))

    assert result.shape == (1, 2)
    assert np.all(result == 1.0)


@pytest.mark.parametrize(
    "input_factory",
    [list, np.asarray, lambda values: np.asarray(values).reshape(-1, 1), pd.Series, pd.DataFrame, iter],
    ids=["list", "array", "column_array", "series", "dataframe", "generator"],
)
def test_chemeleon_transform_preserves_molecule_rows(input_factory) -> None:
    molecules = [Chem.MolFromSmiles(smiles) for smiles in ["CCO", "CC", "C"]]
    transformer = CheMeleonFingerprintTransformer(
        output_dim=2,
        embedder=lambda smiles_batch: np.asarray([[len(smiles), 1] for smiles in smiles_batch], dtype=np.float32),
    )

    result = transformer.fit_transform(input_factory(molecules))

    np.testing.assert_array_equal(result, [[3, 1], [2, 1], [1, 1]])
    assert result.dtype == np.float32
    assert transformer.get_feature_names_out() == ["CheMeleonGNNFP_0", "CheMeleonGNNFP_1"]


def test_chemeleon_batching_preserves_invalid_rows() -> None:
    batches = []

    def embedder(smiles_batch):
        batches.append(list(smiles_batch))
        return np.full((len(smiles_batch), 2), len(batches), dtype=np.float32)

    molecules = [Chem.MolFromSmiles("CCO"), None, Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("C")]
    transformer = CheMeleonFingerprintTransformer(output_dim=2, batch_size=1, embedder=embedder)

    result = transformer.fit_transform(molecules)

    assert batches == [["CCO"], ["CC"], ["C"]]
    np.testing.assert_array_equal(result[[0, 2, 3]], [[1, 1], [2, 2], [3, 3]])
    assert np.isnan(result[1]).all()
    assert transformer.transform([]).shape == (0, 2)
    assert np.isnan(transformer.transform([None, "invalid"])).all()
    assert len(batches) == 3


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 2), "Expected 2 embedding rows"),
        ((3, 2), "Expected 2 embedding rows"),
        ((2, 3), "Expected embedding size 2"),
        ((2,), "must return a 2D array"),
    ],
)
def test_chemeleon_rejects_malformed_embedder_output(shape, message) -> None:
    molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC")]
    transformer = CheMeleonFingerprintTransformer(
        output_dim=2, embedder=lambda smiles_batch: np.ones(shape, dtype=np.float32)
    )

    with pytest.raises(ValueError, match=message):
        transformer.fit_transform(molecules)


def test_chemeleon_sklearn_contract() -> None:
    molecule = Chem.MolFromSmiles("CCO")
    transformer = CheMeleonFingerprintTransformer(
        output_dim=2, embedder=lambda smiles_batch: np.ones((len(smiles_batch), 2), dtype=np.float32)
    )

    with pytest.raises(NotFittedError):
        transformer.transform([molecule])
    fitted = transformer.fit([molecule])
    assert fitted is transformer
    with pytest.raises(NotFittedError):
        clone(transformer).transform([molecule])
    with pytest.raises(ValueError, match="single-column table"):
        transformer.transform(np.asarray([[molecule, molecule]], dtype=object))


if __name__ == "__main__":
    pytest.main()
