from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import SaltRemover

# contains C++ reimplementation of standardization procedure
from rdkit.Chem.MolStandardize import rdMolStandardize

from mother.preprocessing import utils
from mother.preprocessing.utils import StandardizationFlag

# creating logger
log = logging.getLogger(__name__)

# creating rdkit logger
rdkit_logger, logger_sio = utils.create_rdkit_logger()

data_dir: Path = Path(__file__).parent.joinpath("data")
rm_pattern: str = r"(Removed .*)\n"


# pylint: disable=too-many-public-methods
class Standardizer:  # type: ignore
    """Configurable standardization pipeline interface to facilitate usage
    and to provide some useful defaults.

    Uses the rdkit.Chem.MolStandardize.Standardizer.

    Notes:
        This class wraps RDKit MolStandardize components and allows fine‑grained control
        over standardization steps via `StandardizationFlag`.

        Extra RDKit configuration can be passed via `**kwargs` in ``__init__``. Supported
        keys are forwarded to ``rdMolStandardize.CleanupParameters``:

        - doCanonical (bool): Whether to canonicalize during cleanup. Default: True.
        - largestFragmentChooserUseAtomCount (bool): Use atom count when choosing the
          largest fragment. Default: True.
        - tautomerRemoveSp3Stereo (bool): Remove sp3 stereo in tautomer canonicalization.
          Default: False.
        - tautomerRemoveBondStereo (bool): Remove bond stereo in tautomer canonicalization.
          Default: False.
        - tautomerRemoveIsotopicHs (bool): Remove isotopic Hs in tautomer canonicalization.
          Default: False.

        These kwargs are applied only at construction time and stored in internal
        RDKit parameter objects.
    """

    def __init__(
        self,
        flag: StandardizationFlag = StandardizationFlag.STANDARDIZE,
        max_restarts: int = 200,
        max_tautomers: int = 100,
        max_transforms: int = 1000,
        prefer_organic: bool = True,
        as_smiles: bool = False,
        **kwargs,
    ) -> None:
        """Create a Standardizer instance.

        Args:
            flag (StandardizationFlag, optional): Standardization pipeline flags.
            max_restarts (int, optional): Max restart count for normalization.
            max_tautomers (int, optional): Max tautomers to enumerate.
            max_transforms (int, optional): Max transforms during tautomer handling.
            prefer_organic (bool, optional): Prefer organic fragments when choosing
                the largest fragment.
            as_smiles (bool, optional): Return SMILES if True, else RDKit Mol.

        Keyword Args:
            doCanonical (bool): See class notes.
            largestFragmentChooserUseAtomCount (bool): See class notes.
            tautomerRemoveSp3Stereo (bool): See class notes.
            tautomerRemoveBondStereo (bool): See class notes.
            tautomerRemoveIsotopicHs (bool): See class notes.
        """
        args: list[Any] = []
        super().__init__(*args, **kwargs)
        # Tell the RDKit's C++ backend to log to use the python logger:
        rdBase.LogToPythonLogger()
        self.flag: StandardizationFlag = flag
        self.logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._as_smiles: bool = as_smiles

        # init some rdkit objects to be used later
        # https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html
        self._params = rdMolStandardize.CleanupParameters()
        self._params.doCanonical = kwargs.get("doCanonical", True)
        self._params.maxRestarts = max_restarts
        self._params.maxTautomers = max_tautomers
        self._params.maxTransforms = max_transforms
        self._params.preferOrganic = prefer_organic
        self._params.largestFragmentChooserUseAtomCount = kwargs.get("largestFragmentChooserUseAtomCount", True)
        self._params.tautomerRemoveSp3Stereo = kwargs.get("tautomerRemoveSp3Stereo", False)
        self._params.tautomerRemoveBondStereo = kwargs.get("tautomerRemoveBondStereo", False)
        self._params.tautomerRemoveIsotopicHs = kwargs.get("tautomerRemoveIsotopicHs", False)

        self.normalizer = rdMolStandardize.Normalizer(self._params.normalizationsFile, self._params.maxRestarts)
        self.uncharger = rdMolStandardize.Uncharger()
        self.reionizer = rdMolStandardize.Reionizer()
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
        self.disconnector = rdMolStandardize.MetalDisconnector()
        self.largest_fragment = rdMolStandardize.LargestFragmentChooser(self._params)
        self.fragment_remover = rdMolStandardize.FragmentRemover()
        self.salt_remover = SaltRemover.SaltRemover(defnFilename=data_dir.joinpath("Salts.txt"))
        self.molecule_fixer = rdMolStandardize.Normalizer(
            data_dir.joinpath("kekulization_fixes.txt").as_posix(),
            self._params.maxRestarts,
        )

    def set_flag(self, flag: StandardizationFlag | None) -> Standardizer:
        """Override initialized flags to enable reuse of standardizer object.

        Args:
            flag (StandardizationFlag): the flag(s) to be set

        Returns:
            Standardizer: the object itself
        """
        if flag:
            self.flag = flag
        return self

    def init(self, args) -> None:
        if len(args) > 0 and args[0] is not None and isinstance(args[0], StandardizationFlag):
            self.set_flag(args[0])
        if len(args) > 1 and args[1] is not None and isinstance(args[1], bool):
            self.as_smiles(args[1])

    def as_smiles(self, as_smiles: bool = True) -> Standardizer:
        """Define the standardizer result type, SMILES if true, rdkit mol otherwise

        Args:
            as_smiles (bool, optional): Export result as SMILES if true. Defaults to True.
        Returns:
            Standardizer: the object itself
        """
        self._as_smiles = as_smiles
        return self

    def disable_flag(self, flag: StandardizationFlag) -> Standardizer:
        """Disable a specific flag. Facilitates reusage of the standardizer pipeline

        Args:
            flag (StandardizationFlag): the flag to be disabled

        Returns:
            Standardizer: the object itself
        """
        self.flag &= ~flag
        return self

    def enable_all_flags(self) -> Standardizer:
        """Activate all flags to standardize the molecule

        Returns:
            Standardizer: the object itself
        """
        self.flag = StandardizationFlag.ALL
        return self

    def neutralize(self) -> Standardizer:
        """Activate neutralization of the molecule.

        Returns:
            Standardizer: the object itself
        """
        self.flag |= StandardizationFlag.NEUTRALIZE
        return self

    def desalt(self) -> Standardizer:
        """Activate desalting during molecule standardization.

        Returns:
            Standardizer: the object itself
        """
        self.flag |= StandardizationFlag.DESALT
        return self

    def keep_salt(self) -> Standardizer:
        """Disables salt removal during molecule standardization.

        Returns:
            Standardizer: the object itself
        """
        self.flag &= ~StandardizationFlag.DESALT
        return self

    def flatten(self) -> Standardizer:
        """Flatten stereo chemistry of the molecule

        Returns:
            Standardizer: the object itself
        """
        self.flag |= StandardizationFlag.FLATTEN_STEREOCHEMISTRY
        return self

    def canonical_tautomer(self) -> Standardizer:
        """This method sets the flag to indicate that canonical tautomer standardization should be applied.

        Returns:
            Standardizer: the instance of the Standardizer class with the flag updated
        """
        self.flag |= StandardizationFlag.CANONICAL_TAUTOMER
        return self

    def get_flag(self) -> StandardizationFlag:
        """Get the configered flag of the pipeline.

        Returns:
            StandardizationFlag: currently configured flag.
        """
        return self.flag

    def set_rdkit_log_level(self, level: int = logging.INFO):
        """Convenience function to set rdkit log level.

        Args:
            level (int, optional): The level to set. Defaults to RDLogger.INFO.
        """
        if level > logging.INFO:
            log.warning("Setting RDKit log level to '%s' might have an effect on pipeline results", level)
        rdkit_logger.setLevel(level)

    def disable_rdkit_logging(self):
        """Convenience function to disable the RdKit log level."""
        self.set_rdkit_log_level(logging.CRITICAL)

    # pylint: disable=too-many-statements
    def standardize(self, mol: str | Chem.rdchem.Mol | None, **kwargs) -> str | Chem.rdchem.Mol | None:
        """Perform standardization of the given molecule.
        Steps according to StandardizeSmiles function with some extras.
        https://github.com/rdkit/rdkit/blob/master/Docs/Notebooks/MolStandardize.ipynb

        Args:
            mol (Union[str, rdkit.Chem.rdchem.Mol]): the input molecule to be standardized

        Raises:
            NotImplementedError: function calls to functions that are not yet implemented

        Returns:
            rdkit.Chem.rdchem.Mol: _description_
        """
        mol_props: dict[str, Any] = {}
        molecule: Chem.rdchem.Mol = Chem.Mol()
        input_molecule: Chem.rdchem.Mol | None = None
        try:
            input_molecule, molecule, mol_props = self._prepare_input_molecule(mol)
            if not input_molecule:
                self.logger.warning("Invalid molecule input to standardizer")
                return input_molecule

            molecule = self._initial_standardize(molecule, mol_props, input_molecule)

            if self.flag == StandardizationFlag.NONE:
                return self.__result(molecule, **kwargs)

            molecule = self._apply_flagged_steps(molecule, mol_props, kwargs)
            molecule = self._final_standardize(molecule, mol_props)

        except IndexError as ex_ception:
            log.error(
                "Standardization of %s failed", utils.mol_get_name(input_molecule) if input_molecule else "unknown"
            )
            log.error(ex_ception)
        except Chem.rdchem.KekulizeException as ex_ception:
            log.error(ex_ception)
            log.warning(
                "Can't kekulize molecule %s", utils.mol_get_name(input_molecule) if input_molecule else "unknown"
            )
            log.warning("Returning potentially 'invalid' input molecule")
            if input_molecule:
                molecule = input_molecule
        except NotImplementedError as ex_ception:
            raise ex_ception from None
        except Exception as ex_ception:
            log.error("Could not standardize molecule")
            log.error(ex_ception)
            return None
        assert isinstance(molecule, Chem.rdchem.Mol)
        self._store_properties(molecule, mol_props)

        return self.__result(molecule, **kwargs)

    def _prepare_input_molecule(
        self, mol: str | Chem.rdchem.Mol | None
    ) -> tuple[Chem.rdchem.Mol | None, Chem.rdchem.Mol, dict[str, Any]]:
        # Do NOT sanitize here, as we want to catch errors later on
        input_molecule = utils.generate_input_molecule(mol, sanitize=False)
        if not input_molecule:
            return None, Chem.Mol(), {}

        molecule = deepcopy(input_molecule)
        mol_props = molecule.GetPropsAsDict(includePrivate=False)
        mol_props["_Name"] = molecule.GetProp("_Name") if molecule.HasProp("_Name") else ""
        # add original molecule SMILES as property
        mol_props["Original_SMILES"] = Chem.MolToSmiles(input_molecule, canonical=False)
        return input_molecule, molecule, mol_props

    def _initial_standardize(
        self,
        molecule: Chem.rdchem.Mol,
        mol_props: dict[str, Any],
        input_molecule: Chem.rdchem.Mol,
    ) -> Chem.rdchem.Mol:
        if not (self.flag & StandardizationFlag.STANDARDIZE):
            return molecule

        # do some inital standardization required for later stages
        try:
            # validate input SMILES
            # sanitize does kinda change the molecule
            rdMolStandardize.ValidateSmiles(mol_props["Original_SMILES"])
        except Chem.rdchem.KekulizeException:
            # catch kekulization error, e.g.: invalid triazol (c1nncn1)
            molecule = self.__repair(molecule, mol_props=mol_props)
            if not molecule:
                return input_molecule
        Chem.SanitizeMol(molecule)
        molecule = Chem.RemoveHs(molecule)
        # disconnect metals required for propper desalting and standardization as well
        molecule = self.disconnect(molecule, mol_props=mol_props)
        # need to santize again, since disconnector may not return a valid molecule
        Chem.SanitizeMol(molecule)
        return molecule

    def _apply_flagged_steps(
        self,
        molecule: Chem.rdchem.Mol,
        mol_props: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Chem.rdchem.Mol:
        self.logger.debug("Performing workflow for molecule: %s ", Chem.MolToSmiles(molecule))

        if self.flag & StandardizationFlag.DESALT:
            self.logger.debug("Desalt molecule/ remove Fragments")
            molecule = self.remove_fragments(molecule, mol_props=mol_props)
            molecule = self.salt_remover.StripMol(molecule, dontRemoveEverything=True, sanitize=False)
            molecule = self.get_largest_fragment(molecule, mol_props=mol_props)

        if self.flag & StandardizationFlag.NEUTRALIZE:
            self.logger.debug("Neutralize molecule")
            # tries to generate a neutral form, not necessarily without charges
            molecule = self.uncharge(molecule, mol_props=mol_props)

        if self.flag & StandardizationFlag.CANONICAL_TAUTOMER:
            molecule = self.get_canonical_tautomer(molecule)
            self.logger.debug("Generate canonical tautomer")
        elif self.flag & StandardizationFlag.FLATTEN_STEREOCHEMISTRY:
            self.logger.debug("Flatten stereochemistry")
            if self._as_smiles:
                kwargs["isomericSmiles"] = False
            else:
                molecule = Chem.MolFromSmiles(Chem.MolToSmiles(molecule, isomericSmiles=False))

        return molecule

    def _final_standardize(self, molecule: Chem.rdchem.Mol, mol_props: dict[str, Any]) -> Chem.rdchem.Mol:
        if not (self.flag & StandardizationFlag.STANDARDIZE):
            self.logger.warning("No standardization steps were applied to the molecule.")
            return molecule

        # standardize at the very end
        self.logger.debug(
            "Perform final rdkit/molvs standardization/normalization steps for: %s",
            utils.mol_get_name(molecule),
        )
        # pylint: disable=unexpected-keyword-arg,too-many-function-args
        molecule = self.normalize(molecule, mol_props=mol_props)
        molecule = self.reionize(molecule)
        Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
        # store canonical SMILES for standardized form...
        # pylint: disable=no-member
        canonical_smiles: str | None = Chem.MolToSmiles(molecule, canonical=True)
        if canonical_smiles:
            molecule.SetProp("Canonical_SMILES", canonical_smiles)
        return molecule

    def _store_properties(self, molecule: Chem.rdchem.Mol, mol_props: dict[str, Any]) -> None:
        for k, v in mol_props.items():
            molecule.SetProp(k, str(v))

    def __result(self, mol: Chem.rdchem.Mol, **kwargs) -> Chem.rdchem.Mol | str | None:
        """Returns the standardized molecule as type molecule or SMILES.
        Simple wrapper function.

        Args:
            mol (Chem.rdchem.Mol): the standardized molecule

        Returns:
            Union[Chem.rdchem.Mol, str]: how the molecule should be returned.
        """
        if not self._as_smiles:
            return mol

        # Filter kwargs to only include valid MolToSmiles parameters
        valid_params = {
            "isomericSmiles",
            "kekuleSmiles",
            "rootedAtAtom",
            "canonical",
            "allBondsExplicit",
            "allHsExplicit",
            "doRandom",
            "ignoreAtomMapNumbers",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        canonical: bool = filtered_kwargs.get("canonical", True)
        filtered_kwargs["canonical"] = canonical

        return Chem.MolToSmiles(mol, **filtered_kwargs)

    def result_is_smiles(self) -> bool:
        return self._as_smiles

    def normalize(self, mol: Chem.rdchem.Mol, mol_props: dict[str, Any] | None = None) -> Chem.rdchem.Mol:
        return self.perform_step_and_extract_log(
            mol=mol,
            func=self.normalizer.normalize,
            pattern=r"Rule applied: (.*?)\n",
            key="applied_transformations",
            mol_props=mol_props,
        )

    def reionize(self, mol: Chem.rdchem.Mol):
        return self.reionizer.reionize(mol)

    def disconnect(self, mol: Chem.rdchem.Mol, mol_props: dict[str, Any] | None = None) -> Chem.rdchem.Mol:
        return self.perform_step_and_extract_log(
            mol=mol,
            func=self.disconnector.Disconnect,
            pattern=rm_pattern,
            key="normalizations_applied",
            mol_props=mol_props,
        )

    def get_largest_fragment(self, mol: Chem.rdchem.Mol, mol_props: dict[str, Any] | None = None) -> Chem.rdchem.Mol:
        return self.perform_step_and_extract_log(
            mol=mol,
            func=self.largest_fragment.choose,
            pattern=rm_pattern,
            key="normalizations_applied",
            mol_props=mol_props,
        )

    def remove_fragments(self, mol: Chem.rdchem.Mol, mol_props: dict[str, Any] | None = None) -> Chem.rdchem.Mol:
        return self.perform_step_and_extract_log(
            mol=mol,
            func=self.fragment_remover.remove,
            pattern=rm_pattern,
            key="normalizations_applied",
            mol_props=mol_props,
        )

    def uncharge(self, mol: Chem.rdchem.Mol, mol_props: dict[str, Any] | None = None) -> Chem.rdchem.Mol:
        return self.perform_step_and_extract_log(
            mol=mol,
            func=self.uncharger.uncharge,
            pattern=rm_pattern,
            key="normalizations_applied",
            mol_props=mol_props,
        )

    def perform_step_and_extract_log(
        self,
        mol: Chem.rdchem.Mol,
        func: Callable,
        pattern: str | None,
        key: str | None,
        mol_props: dict[str, Any] | None = None,
    ) -> Any:
        """
        Performs a step on a given molecule and extracts the log based on the provided pattern and key.

        Args:
            mol (Chem.rdchem.Mol): The molecule to perform the step on.
            func (Callable): The function to apply to the molecule.
            pattern (Optional[str]): The pattern used to extract the log.
            key (Optional[str]): The key used to store the log.
            mol_props (Optional[Dict[str, Any]]):
                Additional properties of the molecule for result storage (default: None).

        Returns:
            Any: The result of applying the function to the molecule.

        Raises:
            Warning: If either the pattern or key is not set, the log extraction cannot be performed.
        """
        # clear logger buffer
        logger_sio.truncate(0)
        logger_sio.seek(0)
        # apply function to molecule
        result: Any = func(mol)
        if mol_props:
            if not (pattern and key):
                log.warning("Pattern or key not set. Can't extract log")
            else:
                utils.extract_log(logger_sio=logger_sio, mol_props=mol_props, pattern=pattern, key=key)
        return result

    def get_canonical_tautomer(self, mol: Chem.rdchem.Mol):
        return self.tautomer_enumerator.Canonicalize(mol)

    def enumerate_tautomers(self, mol: Chem.rdchem.Mol) -> rdMolStandardize.TautomerEnumeratorResult:
        return self.tautomer_enumerator.Enumerate(mol)

    def __repair(self, mol: Chem.rdchem.Mol, mol_props: dict[str, Any] | None = None) -> Chem.rdchem.Mol:
        self.logger.debug("Molecule could not be sanitized. Trying to repair")
        # normalize in place does not work if atoms are added!!!
        return self.perform_step_and_extract_log(
            mol=mol,
            func=self.molecule_fixer.normalize,
            pattern=r"Rule applied: (.*?)\n",
            key="applied_transformations",
            mol_props=mol_props,
        )
