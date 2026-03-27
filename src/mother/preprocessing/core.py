import logging
import typing
from abc import ABC
from typing import Iterable, List, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from mother import chem as m_chem
from mother.preprocessing import utils
from mother.preprocessing.standardizer import Standardizer
from mother.utils import convert_input

module_logger: logging.Logger = logging.getLogger(__name__)


class StandardizerTransformer(ABC, BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(
        self,
        flags: List[str],
        smiles_col: typing.Optional[str] = None,
        error: str = "ignore",
    ) -> None:
        self.flags: List[str] = flags
        self.smiles_col: typing.Optional[str] = smiles_col
        self.error: str = error  #'raise'

    @overload
    def fit(
        self, X: npt.NDArray, y: typing.Optional[typing.Union[pd.Series, np.ndarray]] = None
    ) -> "StandardizerTransformer": ...
    @overload
    def fit(
        self, X: pd.DataFrame, y: typing.Optional[typing.Union[pd.Series, np.ndarray]] = None
    ) -> "StandardizerTransformer": ...

    def fit(
        self,
        X: typing.Union[npt.NDArray, pd.DataFrame],
        y: typing.Optional[typing.Union[pd.Series, np.ndarray]] = None,
    ) -> "StandardizerTransformer":
        """Included for scikit-learn compatibility

        Also sets the column prefix for use by the transform method with dataframe output.
        """
        smiles_lst: npt.NDArray = convert_input(X, self.smiles_col)
        self.n_features_in_ = smiles_lst.shape[1] if np.ndim(smiles_lst) > 1 else 1
        if not self.n_features_in_ == 1:
            raise ValueError(
                f"Expected input with 1 feature (SMILES), but got {self.n_features_in_} features. "
                f"Please specify the correct column with 'smiles_col' or check your input data."
            )
        return self

    def _transform(
        self, smiles_list: npt.NDArray, y: typing.Optional[typing.Union[pd.Series, np.ndarray]] = None
    ) -> npt.ArrayLike:
        block = BlockLogs()  # Block all RDkit logging
        module_logger.info(f"Standardizing SMILES with flags: {self.flags}")
        # initialize standardizer on transform since it can not be pickled properly
        standardizer: Standardizer = Standardizer(
            flag=utils.get_standardization_flag_from_strings(self.flags), as_smiles=True
        )
        arr = []
        for index, smi in enumerate(smiles_list):
            m: typing.Optional[Chem.rdchem.Mol] = None
            try:
                with m_chem.RaiseRDKitErrors():
                    m = standardizer.standardize(smi)
            except m_chem.RDKitException:
                module_logger.error(
                    "Catched exception from RDKit during standardization of SMILES with INDEX: %s", index
                )
            if m is not None:
                arr.append(m)
            else:
                module_logger.error("MOTHER failed to standardize input structure with INDEX: %s", index)
                if self.error == "ignore":
                    arr.append(None)
                    continue
                raise ValueError(
                    f"""Issue with parsing SMILES with index {index}.
                    You probably should check your dataset first or use 'ignore'
                    for error handling"""
                )
        del block  # Release logging block to previous state
        return np.array(arr).reshape(-1, 1)

    def get_feature_names_out(self, *args, **kwargs) -> np.ndarray:
        assert self.n_features_in_ == 1
        return np.array([f"STD_{self.smiles_col}" if self.smiles_col else "STD_SMILES"], dtype=str)

    def transform(self, X: Iterable, y: typing.Optional[typing.Union[pd.Series, np.ndarray]] = None):
        """Standardizes SMILES strings in X according to the specified flags"""
        check_is_fitted(self, "n_features_in_")
        return self._transform(convert_input(X, self.smiles_col))


class SmilesToMolTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(
        self,
        smiles_col: typing.Optional[str] = None,
        molecule_col: typing.Optional[str] = "Molecule",
    ):
        self.smiles_col: typing.Optional[str] = smiles_col
        self.molecule_col: typing.Optional[str] = molecule_col
        self.error: str = "ignore"  #'raise' #TODO, implement error handling

    def fit(self, X: Iterable, _: typing.Optional[typing.Union[pd.Series, np.ndarray]] = None):
        smiles_lst: npt.NDArray = convert_input(X, col=self.smiles_col)
        self.n_features_in_ = smiles_lst.shape[1] if np.ndim(smiles_lst) > 1 else 1
        if not self.n_features_in_ == 1:
            raise ValueError(
                f"Expected input with 1 feature (SMILES), but got {self.n_features_in_} features. "
                f"Please specify the correct column with 'smiles_col' or check your input data."
            )
        return self

    def transform(self, X: Iterable, y: typing.Optional[typing.Union[pd.Series, np.ndarray]] = None):
        """Converts SMILES into RDKit mols

        Parameters
        ----------
        X : list-like
            A list of RDKit parsable strings

        Returns
        -------
        List
            List of RDKit mol objects

        Raises
        ------
        ValueError
            Raises ValueError if a SMILES string is unparsable by RDKit
        """
        check_is_fitted(self, "n_features_in_")
        return self._transform(convert_input(X, col=self.smiles_col))

    def _transform(self, smiles_list: npt.NDArray) -> npt.NDArray[typing.Any]:
        mol_valid = []
        smiles_input: npt.NDArray = smiles_list.ravel() if np.ndim(smiles_list) > 1 else smiles_list
        mask_valid: npt.NDArray = np.array([isinstance(smi, str) for smi in smiles_input])
        mol_out: npt.NDArray = np.full((len(smiles_input), 1), None, dtype=Chem.rdchem.Mol)
        mol: Chem.rdchem.Mol | None = None
        for smiles in smiles_input[mask_valid]:
            try:
                with m_chem.RaiseRDKitErrors():
                    mol = Chem.MolFromSmiles(smiles)
            except m_chem.RDKitException:
                module_logger.exception(f"RDKit failed to parse SMILES '{smiles}'")
            if mol is not None:
                mol_valid.append(mol)
            else:
                if self.error == "ignore":
                    mol_valid.append(None)
                    continue
                raise ValueError(
                    f"""Issue with parsing SMILES {smiles}.
                    You probably should use the standardizer on your dataset first"""
                )
        mol_out[mask_valid, :] = np.array(mol_valid, dtype=Chem.rdchem.Mol).reshape(-1, 1)  # Reshape to 2D array
        return mol_out

    def get_feature_names_out(self, *args, **kwargs) -> np.ndarray:
        return np.array([self.molecule_col if self.molecule_col else "Molecule"], dtype=str)
