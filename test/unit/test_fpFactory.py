import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFG
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mother.feature_generation.fp_gen import FingerprintFactory
from mother.feature_generation.fp_gnn_gen import (
    CheMeleonFingerprintFactory,
    CheMeleonFingerprintTransformer,
)


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


@pytest.mark.parametrize("return_separate_columns", [True, False])
def test_chemeleon_factory_output_layout(return_separate_columns) -> None:
    molecules = pd.Series(
        [Chem.MolFromSmiles("CCO"), None, Chem.MolFromSmiles("CC")],
        index=pd.Index([12, 3, 8], name="molecule_id"),
    )
    factory = CheMeleonFingerprintFactory(
        output_dim=2,
        batch_size=1,
        embedder=lambda smiles_batch: np.asarray([[len(smiles), 1] for smiles in smiles_batch], dtype=np.float32),
        embedding_column_name="chemeleon_embedding",
        return_separate_columns=return_separate_columns,
    )
    transformer = factory.get_fingerprint_generator()

    result = transformer.fit_transform(molecules)

    expected = [[3, 1], [np.nan, np.nan], [2, 1]]
    if return_separate_columns:
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)
        assert transformer.get_feature_names_out() == ["chemeleon_embedding_0", "chemeleon_embedding_1"]
    else:
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)
        pd.testing.assert_index_equal(result.index, molecules.index)
        assert result.columns.tolist() == transformer.get_feature_names_out() == ["chemeleon_embedding"]
        np.testing.assert_array_equal(np.stack(result["chemeleon_embedding"]), expected)
    assert transformer.get_output_dimension() == 2
    cloned = clone(transformer)
    assert cloned.embedding_column_name == "chemeleon_embedding"
    assert cloned.return_separate_columns is return_separate_columns
    with pytest.raises(NotFittedError):
        cloned.transform(molecules)


@pytest.mark.parametrize("invalid_values", [[], [None, "invalid"]], ids=["empty", "all_invalid"])
@pytest.mark.parametrize("pandas_input", [False, True], ids=["iterator", "dataframe"])
def test_chemeleon_single_column_empty_or_invalid_rows(invalid_values, pandas_input) -> None:
    def embedder(smiles_batch):
        pytest.fail("The embedder must not be called without valid molecules.")

    index = pd.Index(range(10, 10 + len(invalid_values)), name="molecule_id")
    molecules = pd.DataFrame({"molecule": invalid_values}, index=index) if pandas_input else iter(invalid_values)
    transformer = CheMeleonFingerprintTransformer(output_dim=2, embedder=embedder, return_separate_columns=False)

    result = transformer.fit_transform(molecules)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(invalid_values), 1)
    assert result.columns.tolist() == transformer.get_feature_names_out() == ["CheMeleonGNNFP"]
    pd.testing.assert_index_equal(result.index, index if pandas_input else pd.RangeIndex(len(invalid_values)))
    for vector in result.iloc[:, 0]:
        assert vector.shape == (2,)
        assert vector.dtype == np.float32
        assert np.isnan(vector).all()


@pytest.mark.parametrize("return_separate_columns", [True, False])
def test_chemeleon_sklearn_pandas_output(return_separate_columns) -> None:
    molecules = pd.DataFrame({"molecule": [Chem.MolFromSmiles("CCO")]}, index=pd.Index([7], name="molecule_id"))
    transformer = CheMeleonFingerprintTransformer(
        output_dim=2,
        embedder=lambda smiles_batch: np.ones((len(smiles_batch), 2), dtype=np.float32),
        embedding_column_name="custom_embedding",
        return_separate_columns=return_separate_columns,
    ).set_output(transform="pandas")

    result = transformer.fit_transform(molecules)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 2 if return_separate_columns else 1)
    assert result.columns.tolist() == transformer.get_feature_names_out()
    pd.testing.assert_index_equal(result.index, molecules.index)
    vector = result.iloc[0].to_numpy() if return_separate_columns else result.iloc[0, 0]
    np.testing.assert_array_equal(vector, [1, 1])


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
