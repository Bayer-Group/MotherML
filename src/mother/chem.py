import io
import logging
import re
import sys
import typing
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

module_logger = logging.getLogger(__name__)


@lru_cache
def maccs(mol: Chem.Mol) -> np.ndarray:
    """
    Generates MACCS fingerprint for a molecule and finds atom indices
    for non-zero fingerprint bits

    Parameters
    ----------
    mol
        rdkit.Chem.Mol Molecule to generate MACCS fingerprint
    maccs_col
        string denoting the column for MACCS

    Returns
    -------
    np.ndarray of the fingerprint

    """
    fingerprint: npt.NDArray
    with RaiseRDKitErrors():
        fingerprint = np.array(rdMolDescriptors.GetMACCSKeysFingerprint(mol))

    return fingerprint


@lru_cache
def calculate_descriptor(func: typing.Callable, mol: Chem.Mol) -> typing.Union[np.int32, np.float32]:
    """
    Runs a rdkit descriptor function for a molecule

    Parameters
    ----------
    func: function object to run
    mol: rdkit.Chem.Mol Molecule

    Returns
    -------
    The calculated value for the descriptor

    """
    with RaiseRDKitErrors():
        result = func(mol)

        # Mitigating infinite values
        result = np.nan if np.isinf(result) else result

        # Mitigating values higher or lower than 32 bit float
        # this is due the oddities in
        # "imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=np.finfo(np.float64).min)"
        # because this is setting nan values to the minimum of float64, thus to
        # float_info = np.finfo(np.float32)
        # result = float_info.max if result > float_info.max else result
        # result = float_info.min if result < float_info.min else result

    return result


class RDKitException(Exception):
    pass


class RaiseRDKitErrors:
    """
    Context manager to raise RDKitExceptions when rdkit is failing with
    a semi silent "None" and a message on stderr.

    Examples
    --------
    >>> import rdkit.Chem
    >>> with RaiseRDKitErrors():
    >>>     rdkit.Chem.MolFromSmiles(r"CHC")
    """

    def __init__(self) -> None:
        self.saved_stderr = None
        self.stderr = None

    def __enter__(self) -> None:
        self.saved_stderr = sys.stderr
        self.stderr = sys.stderr = io.StringIO()
        rdBase.LogToPythonStderr()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stderr = self.saved_stderr
        stderr = self.stderr.getvalue()
        keywords: typing.List[str] = ["error", "can't"]  # List of keywords to search for
        if stderr != "":
            if any(keyword in stderr.lower() for keyword in keywords):  # Check for any keyword in stderr
                raise RDKitException(self.clean_error_msg(stderr))
            elif "warning" in stderr.lower():
                module_logger.warning(stderr)
            else:
                module_logger.debug(stderr)

    @staticmethod
    def clean_error_msg(stderr) -> str:
        regex = re.compile(r"(\[\d\d:\d\d:\d\d]\s?)")
        stderr = regex.sub("", stderr)
        stderr = stderr.replace("\n", " ")
        stderr = stderr.strip()
        return stderr


def handle_rdkit_exception(col_id: typing.Any, name: str, exception: RDKitException) -> None:
    """
    Handles rdkit exceptions by logging, appending them to the error column
    and setting the success column to false

    Parameters
    ----------
    row: pd.DataFrame row with a column containing success and error columns as labeled by cols
    name: str Prefix for the error message (usually name of the offending function)
    cols: ColumnNames object
    exception: the raised mother.chem.RDKitException

    Returns
    -------
    pd.Series with the updated column

    """
    msg = f"{name} failed for compound {col_id}: {exception}"
    module_logger.error(msg)
