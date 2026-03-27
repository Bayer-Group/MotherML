import logging
import re
from enum import IntFlag, auto, unique
from io import StringIO
from typing import Any

from rdkit import Chem

module_logger: logging.Logger = logging.getLogger(__name__)


@unique
class StandardizationFlag(IntFlag):
    """Enumeration flags that can be used to standardize molecules.

    Args:
        IntFlag (IntFlag): base class of the StandardizationFlag
    """

    NONE = 0
    STANDARDIZE = auto()
    NEUTRALIZE = auto()
    DESALT = auto()
    CANONICAL_TAUTOMER = auto()
    FLATTEN_STEREOCHEMISTRY = auto()
    KEEP_SALT = STANDARDIZE | NEUTRALIZE
    ALL = STANDARDIZE | NEUTRALIZE | DESALT | FLATTEN_STEREOCHEMISTRY | CANONICAL_TAUTOMER

    def __str__(self) -> str:
        return self.name if self.name else ""

    def __repr__(self) -> str:
        return self.__str__()


def get_standardization_flag_from_strings(flags: list[str]) -> StandardizationFlag:
    """
    Converts a list of user-defined flags into a combined StandardizationFlag.

    Args:
        values (List[Flag]): A list of user-defined flags.

    Returns:
        StandardizationFlag: A combined StandardizationFlag based on the input flags.
    """
    _flag_conversion_map: dict[str, int] = {str(value): value for value in StandardizationFlag}
    # fix for py 3.11 and above
    _flag_conversion_map[str(StandardizationFlag.NONE)] = StandardizationFlag.NONE
    flag: StandardizationFlag = StandardizationFlag.NONE
    for value in flags:
        flag |= StandardizationFlag(_flag_conversion_map[value])
    return flag


def mol_get_name(mol: Chem.rdchem.Mol) -> str:
    """Return string representation of molecule.
    Name if available else SMILES

    Args:
        mol (rdkit.Chem.rdchem.Mol): _description_

    Returns:
        str: string representation of molecule
    """
    return mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name") != "" else "unknown molecule"


def extract_log(logger_sio: StringIO, mol_props: dict[str, Any], pattern: str, key: str):
    # extract applied steps
    log_text: str = logger_sio.getvalue()
    # show the normalizations that were applied:
    rules = re.findall(pattern, log_text)
    if len(rules):
        if key in mol_props:
            mol_props[key].extend(rules)
        else:
            mol_props[key] = rules
    task = re.findall("Running (.*?)\n", log_text)
    failure = re.findall("FAILED (.*?)\n", log_text)
    for t, f in zip(task, failure):
        if "Failure" in mol_props:
            mol_props["Failure"].extend(f"{t} failed: {f}")
        else:
            mol_props["Failure"] = [f"{t} failed: {f}"]


def create_rdkit_logger() -> tuple[logging.Logger, StringIO]:
    """Create a logger for RDKit.

    This function creates a logger for the RDKit library.
    It sets up a logger named "rdkit" and adds a handler that uses a StringIO object to capture log messages.
    The log level of the handler is set to INFO, and the log level of the main logger
    is also set to INFO to ensure that INFO messages are sent to the handlers.

    Returns:
        rdkit_logger (logging.Logger): The logger object for RDKit.
        logger_sio (StringIO): The StringIO object used to capture log messages.
    """
    # creating rdkit logger
    rdkit_logger: logging.Logger = logging.getLogger("rdkit")
    logger_sio: StringIO = StringIO()
    # create a handler that uses the StringIO and set its log level:
    handler = logging.StreamHandler(logger_sio)
    handler.setLevel(logging.INFO)
    # add the handler to the Python logger:
    rdkit_logger.addHandler(handler)
    # we also need to change the level of the main logger so that the INFO messages get sent to the handlers:
    rdkit_logger.setLevel(logging.INFO)
    return rdkit_logger, logger_sio


def generate_input_molecule(mol: str | Chem.rdchem.Mol, **kwargs) -> Chem.rdchem.Mol | None:
    """
    Generate an RDKit molecule from a SMILES string or an existing RDKit molecule.

    Parameters:
        mol (Union[str, Chem.rdchem.Mol]): The input molecule, which can be either a SMILES string or an RDKit molecule.

    Returns:
        Optional[Chem.rdchem.Mol]: The generated RDKit molecule.

    Raises:
        None

    Examples:
        >>> generate_input_molecule("CCO")
        <rdkit.Chem.rdchem.Mol object at 0x7f9a2e8d9a80>

        >>> generate_input_molecule(Chem.MolFromSmiles("CCO"))
        <rdkit.Chem.rdchem.Mol object at 0x7f9a2e8d9a80>
    """
    input_molecule: Chem.rdchem.Mol | None = None
    if isinstance(mol, str):
        module_logger.debug("Generate molecule from SMILES input")
        input_molecule = Chem.MolFromSmiles(mol, **kwargs)
    else:
        input_molecule = mol
    return input_molecule
