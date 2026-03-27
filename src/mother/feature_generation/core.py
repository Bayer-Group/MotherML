import logging
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator as rdFG
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from mother import chem
from mother.feature_generation.fp_gen import FingerprintFactory

module_logger: logging.Logger = logging.getLogger(__name__)


class _TransformOnlyValidMols:
    """helper class for transforming only valid rdkit-mols, and putting np.nan in the rest"""

    @abstractmethod
    def _transform_molecules(self, valid_compounds: Iterable) -> np.ndarray:
        pass

    def transform(self, compounds: Iterable) -> np.ndarray:
        """helper function for passing only valid rdkit-mols to the transformation function"""
        valid_compounds: ArrayLike = np.array(compounds).reshape(-1)
        mask_valid: ArrayLike = np.array([isinstance(compound, chem.Chem.Mol) for compound in valid_compounds])

        y = np.full((len(valid_compounds), self.get_output_dimension()), np.nan)
        y_valid = self._transform_molecules(valid_compounds[mask_valid])
        if len(y_valid) > 0:
            y[mask_valid, :] = y_valid
        return y

    @abstractmethod
    def get_output_dimension(self) -> int:
        raise NotImplementedError


class MaccsFingerprints(BaseEstimator, TransformerMixin, _TransformOnlyValidMols):
    """calculates MACCS fingerprints from rdkit-mol objects"""

    def get_output_dimension(self) -> int:
        return 167

    def fit(self, *args, **kwargs) -> BaseEstimator:
        self.is_fitted_ = True
        return self

    def _transform_molecules(self, valid_compounds: Iterable) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        module_logger.info("Calculating MACCS fingerprints")
        return np.array([chem.maccs(compound) for compound in valid_compounds])

    def get_feature_names_out(self, *args, **kwargs) -> List[str]:
        return [f"maccs_{i}" for i in range(self.get_output_dimension())]


class FingerprintsGeneric(BaseEstimator, TransformerMixin, _TransformOnlyValidMols):
    """calculates fingerprints from rdkit-mol objects
    the type and configuration of the fingerprint is determined by the parameters
    """

    def __init__(self, fp_type: str, parameters: dict, use_counts: bool = False) -> None:
        self.fp_type: str = fp_type
        self.parameters: dict = parameters
        self.use_counts: bool = use_counts

    def fit(self, *args, **kwargs) -> BaseEstimator:
        self.is_fitted_ = True
        return self

    def _transform_molecules(self, valid_compounds: Iterable) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        module_logger.info(
            "Calculating %s fingerprints using parameters: %s",
            self.fp_type,
            self.parameters,
        )
        generator: rdFG.FingerprintGenerator64 = FingerprintFactory(
            self.fp_type, self.parameters
        ).get_fingerprint_generator()
        fingerprints: np.ndarray = np.empty((len(list(valid_compounds)), self.parameters["fpSize"]), dtype=np.int64)
        for idx, compound in enumerate(valid_compounds):
            try:
                if self.use_counts:
                    fingerprint = generator.GetCountFingerprintAsNumPy(compound)
                else:
                    fingerprint = generator.GetFingerprintAsNumPy(compound)
                fingerprints[idx] = fingerprint
            except ValueError as e:
                if str(e) == "Bad Conformer Id":
                    compound = self.handle_bad_conformer(compound, e)
                    if self.use_counts:
                        fingerprint = generator.GetCountFingerprintAsNumPy(compound)
                    else:
                        fingerprint = generator.GetFingerprintAsNumPy(compound)
                    fingerprints[idx] = fingerprint
                else:
                    raise
        return fingerprints

    def get_feature_names_out(self, *args, **kwargs) -> List[str]:
        return [f"{self.fp_type}_{i}" for i in range(self.get_output_dimension())]

    def get_output_dimension(self) -> int:
        return self.parameters["fpSize"]

    def handle_bad_conformer(self, compound, error) -> Chem.rdchem.Mol:
        module_logger.warning(f"Handling bad conformer for compound: {compound}, error: {error}")
        module_logger.debug("AtomPairFP requires a valid conformer, embedding one now")
        compound = Chem.AddHs(compound)
        AllChem.EmbedMolecule(compound)  # type: ignore
        return compound


class MorganFingerprints(FingerprintsGeneric):
    def __init__(
        self, radius: int = 2, fpSize=1024, include_chirality: bool = False, use_counts: bool = False, **kwargs
    ) -> None:
        super().__init__(
            "MorganFP",
            {
                "radius": radius,
                "fpSize": fpSize,
                "includeChirality": include_chirality,
            },
            use_counts=use_counts,
        )

        self.radius = radius
        self.fpSize = fpSize
        self.include_chirality = include_chirality

    def fit(self, *args, **kwargs) -> BaseEstimator:
        self.parameters = {
            "radius": self.radius,
            "fpSize": self.fpSize,
            "includeChirality": self.include_chirality,
        }
        return super().fit(*args, **kwargs)


class ChemicalDescriptors(BaseEstimator, TransformerMixin, _TransformOnlyValidMols):
    """
    Calculates chemical descriptors from RDKit molecule objects.

    Parameters
    ----------
    omit_prefixes : Tuple[str, ...], default=()
        List of prefix strings to omit descriptor methods.
    descriptor_prefix : str, default=""
        Prefix for the descriptor names.
    descriptor_list : Optional[List[str]], default=None
        List of descriptor names to calculate. If None or an empty list,
        all available descriptors are used.
    """

    def __init__(
        self,
        omit_prefixes: Tuple[str, ...] = (),
        descriptor_prefix: str = "",
        descriptor_list: Optional[List[str]] = None,
    ):
        self.omit_prefixes: Tuple[str, ...] = omit_prefixes
        self.descriptor_prefix: str = descriptor_prefix
        self.descriptor_list: Optional[List[str]] = descriptor_list

    def _check_descriptor_list(self):
        """
        Internal method to check if the 'descriptor_list' attribute is set and not None
        or an empty list.

        Raises
        ------
        NotFittedError
            If the 'descriptor_list' attribute is None or not set.
        """
        if not self.descriptor_list:
            raise NotFittedError(
                "This ChemicalDescriptors instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def fit(self, *args, **kwargs):
        """
        Fit the transformer.

        Returns
        -------
        self : object
            Returns self.
        """
        # Ensure descriptor_list is resolved during fit
        if not self.descriptor_list:
            self.descriptor_list = [
                descriptor
                for descriptor, _ in Descriptors.descList
                if not any(descriptor.startswith(prefix) for prefix in self.omit_prefixes)
            ]
        return self

    def get_output_dimension(self) -> int:
        """
        Returns the number of descriptors to calculate.
        """
        self._check_descriptor_list()
        return len(self.descriptor_list)

    def _transform_molecules(self, valid_compounds: Iterable) -> np.ndarray:
        """
        Internal method to calculate descriptors for valid RDKit molecules.

        Parameters
        ----------
        valid_compounds : Iterable
            Iterable of valid RDKit molecule objects.

        Returns
        -------
        np.ndarray
            Array of calculated descriptors.
        """
        self._check_descriptor_list()
        phys_chem_properties = np.full((len(list(valid_compounds)), self.get_output_dimension()), np.nan)

        # Resolve descriptor functions dynamically as only strings were stored
        descriptor_functions = {
            descriptor: func for descriptor, func in Descriptors.descList if descriptor in self.descriptor_list
        }

        for mol_idx, mol_ in enumerate(valid_compounds):
            for i, descriptor in enumerate(self.descriptor_list):
                func = descriptor_functions[descriptor]
                try:
                    module_logger.debug("Using function %s", descriptor)
                    phys_chem_properties[mol_idx, i] = chem.calculate_descriptor(func, mol_)
                except chem.RDKitException as exc:
                    chem.handle_rdkit_exception(
                        col_id=mol_idx,
                        name=f"mother.preprocessing.add_descriptors -> {descriptor}",
                        exception=exc,
                    )
                    phys_chem_properties[mol_idx, i] = np.nan

        return phys_chem_properties

    def get_feature_names_out(self, *args, **kwargs) -> List[str]:
        """
        Returns the names of the output features.

        Returns
        -------
        List[str]
            List of output feature names.
        """
        self._check_descriptor_list()
        return [f"{self.descriptor_prefix}{desc}" for desc in self.descriptor_list]
