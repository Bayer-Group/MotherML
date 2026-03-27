import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from rdkit import Chem
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from mother.preprocessing.core import SmilesToMolTransformer, StandardizerTransformer
from mother.preprocessing.standardizer import StandardizationFlag, Standardizer, utils

GLOBAL_FLAG: typing.List[str] = ["STANDARDIZE", "NEUTRALIZE", "DESALT"]


def standardize(
    mol_list: typing.Sequence[Chem.rdchem.Mol | str],
    flags: StandardizationFlag = utils.get_standardization_flag_from_strings(GLOBAL_FLAG),
) -> list[Chem.rdchem.Mol | str]:
    standardizer = Standardizer(flag=flags, as_smiles=True)
    return [standardizer.standardize(mol) for mol in mol_list]


@pytest.fixture
def std_transformer() -> StandardizerTransformer:
    return StandardizerTransformer(flags=GLOBAL_FLAG)


@pytest.fixture
def smiles_transformer() -> SmilesToMolTransformer:
    return SmilesToMolTransformer()


_SMILES_LIST = [
    "O=C(O)c1ccccc1",
    "O=C([O-])c1ccccc1",
    "O=C([O-])c1ccccc1.[Na+]",
    "O=C(O[Na])c1ccccc1",
    "C[N+](C)C.O=C([O-])c1ccccc1",
    "CN=C=O",
    "CCc1ccc2c3[nH]c4ccccc4c3cc[n+]2c1.[Cl-]",
    None,
    "c1",  # invalid SMILES
    "COc1cc2c(c3oc(=O)c4c(c13)CCC4=O)[C@@H]1C=CO[C@@H]1O2",
]
_CANONICAL_SMILES_LIST = standardize(mol_list=_SMILES_LIST)

_CONTAINER_CREATORS = [
    lambda x: x,
    lambda x: np.array(x),
    lambda x: pd.Series(x),
    # lambda x: np.array(x).reshape(-1, 1),
]


@pytest.fixture
def smiles_list() -> list[str]:
    return _SMILES_LIST.copy()


@pytest.fixture
def smiles_frame() -> pd.DataFrame:
    return pd.DataFrame(_SMILES_LIST, columns=["SMILES"])


@pytest.fixture(params=[container(_SMILES_LIST) for container in _CONTAINER_CREATORS])
def smiles_container(
    request,
):
    return request.param.copy()


@pytest.fixture(params=[container(_CANONICAL_SMILES_LIST) for container in _CONTAINER_CREATORS])
def std_smiles_container(
    request,
):
    return request.param.copy()


@pytest.fixture
def invalid(invalid_smiles, missing_smiles):
    return [invalid_smiles, missing_smiles]


class TestStdTransformer:
    def test_standardizer(
        self, std_transformer: StandardizerTransformer, smiles_container: npt.NDArray, transform_result: str
    ) -> None:
        std_transformer.set_output(transform=transform_result)  # type: ignore
        result = std_transformer.fit_transform(smiles_container)

        if isinstance(result, pd.DataFrame):
            result_smiles = result.iloc[:, 0].tolist()
        else:
            result_smiles = result.flatten()
        assert len(result_smiles) == len(smiles_container)
        assert all([a == b for a, b in zip(_CANONICAL_SMILES_LIST, result_smiles)])

    def test_as_transformer(self, std_transformer: StandardizerTransformer, smiles_frame: pd.DataFrame) -> None:
        # ColumnTransformer to apply transformations to specific columns
        preprocessor: ColumnTransformer = ColumnTransformer(
            transformers=[
                ("smiles_standardizer", std_transformer, "SMILES"),
                # Add other column transformations here if needed
            ],
            remainder="passthrough",  # Specify what to do with columns not explicitly selected
        ).set_output(transform="pandas")  # type: ignore

        preprocessor = preprocessor.fit(smiles_frame)
        preprocessor.fit_transform(smiles_frame)
        smiles_results: pd.DataFrame = preprocessor.transform(smiles_frame)  # type: ignore

        result: pd.DataFrame = std_transformer.fit_transform(smiles_frame.loc[:, "SMILES"])  # type: ignore
        if isinstance(smiles_frame, pd.DataFrame):
            expected_smiles = result.iloc[:, 0].tolist()
            smi_result = smiles_results.iloc[:, 0].tolist()
        else:
            expected_smiles = result
            smi_result = smiles_results.flatten()
        assert len(result) == len(smiles_results)
        assert all([a == b for a, b in zip(expected_smiles, smi_result)])

        # pd_test.assert_frame_equal(result, smiles_results,check_names=False)

    def test_raises(self, std_transformer: StandardizerTransformer, smiles_list) -> None:
        with pytest.raises(NotFittedError):
            std_transformer.transform(smiles_list)

    def test_fit_raises_with_more_than_one_feature(self) -> None:
        std_transformer = StandardizerTransformer(flags=GLOBAL_FLAG)
        multi_feature_input = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "OTHER": ["foo", "bar"],
            }
        )

        with pytest.raises(ValueError, match="Expected input with 1 feature"):
            std_transformer.fit(multi_feature_input)

    def test_invalid_smiles(self, invalid_smiles) -> None:
        std_transformer: StandardizerTransformer = StandardizerTransformer(flags=GLOBAL_FLAG, error="raise")
        with pytest.raises(ValueError):
            res = std_transformer.fit_transform(invalid_smiles)
            print(res)


class TestMolFromSmilesTransformer:
    def test_transformer(
        self, smiles_transformer: SmilesToMolTransformer, std_smiles_container: npt.NDArray, transform_result: str
    ) -> None:
        smiles_transformer.set_output(transform=transform_result)  # type: ignore
        result = smiles_transformer.fit_transform(std_smiles_container)

        if isinstance(result, pd.DataFrame):
            result_mols = result.iloc[:, 0].tolist()
        else:
            result_mols = result.flatten()
        assert len(result_mols) == len(std_smiles_container)
        result_smiles = [Chem.MolToSmiles(mol) if mol is not None else None for mol in result_mols]
        assert all([a == b for a, b in zip(_CANONICAL_SMILES_LIST, result_smiles)])

    def test_fit_raises_with_more_than_one_feature(self) -> None:
        smiles_transformer = SmilesToMolTransformer()
        multi_feature_input = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "OTHER": ["foo", "bar"],
            }
        )

        with pytest.raises(ValueError, match="Expected input with 1 feature"):
            smiles_transformer.fit(multi_feature_input)


class TestPipeline:
    def test_smiles_preprocessor(
        self,
        std_transformer: StandardizerTransformer,
        smiles_transformer: SmilesToMolTransformer,
        smiles_frame: pd.DataFrame,
    ) -> None:
        preprocessor: Pipeline = Pipeline(
            steps=[
                (
                    "smiles_standardizer",
                    std_transformer,
                ),
                ("smiles_to_mol", smiles_transformer),
                # Add other column transformations here if needed
            ],
            memory=None,
        ).set_output(transform="pandas")  # type: ignore
        preprocessor = preprocessor.fit(smiles_frame)
        preprocessor.fit_transform(smiles_frame)
        result_mols: pd.DataFrame = preprocessor.transform(smiles_frame)  # type: ignore
        result_smiles = [
            Chem.MolToSmiles(mol) if isinstance(mol, Chem.rdchem.Mol) else None
            for mol in result_mols.iloc[:, 0].tolist()
        ]
        assert all([a == b for a, b in zip(_CANONICAL_SMILES_LIST, result_smiles)])
