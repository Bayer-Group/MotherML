import logging
import typing

from rdkit.Chem import rdFingerprintGenerator as rdFG

module_logger: logging.Logger = logging.getLogger(__name__)


class FingerprintFactory:
    supported_types: typing.List[str] = list(rdFG.FPType.names.keys())

    def __init__(self, fp_type: str, parameters: typing.Dict[str, typing.Any]) -> None:
        """
        Initialize a FingerprintFactory object with some defaults.

        Parameters
        ----------
        fp_type : str
            The type of fingerprint.
        parameters : Dict[str, Any]
            The parameters for the fingerprint generator.

        """
        if fp_type not in rdFG.FPType.names:
            raise ValueError(f"Unknown fingerprint type: {fp_type}. Allowed types are: {rdFG.FPType.names.keys()}")
        self.fp_type: str = fp_type
        self.parameters: typing.Dict[str, typing.Any] = parameters
        if "n_bits" in self.parameters:
            module_logger.debug("Renaming n_bits to fpSize")
            self.parameters["fpSize"] = self.parameters.pop("n_bits")

        if "countSimulation" not in self.parameters:
            self.parameters["countSimulation"] = False
        if "fpSize" not in self.parameters:
            self.parameters["fpSize"] = 2048
        module_logger.debug("FingerprintFactory initialized with parameters: %s", self.parameters)

    def __str__(self) -> str:
        # Collect all instance attributes and their values in a dictionary
        params = vars(self)
        # Format the dictionary into a string of key-value pairs
        params_str = ", ".join(f"{key}={value}" for key, value in params.items())
        return f"{self.__class__.__name__}({params_str})"

    def __repr__(self) -> str:
        return self.__str__()

    def get_fingerprint_generator(self) -> rdFG.FingerprintGenerator64:
        """
        Returns the fingerprint generator function based on the input string

        Returns
        -------
        fingerprint generator function

        Raises
        ------
        ValueError
            If the fingerprint type is unknown.

        """
        module_logger.info("Creating fingerprint generator '%s'", self.fp_type)
        if self.fp_type == "AtomPairFP":
            return rdFG.GetAtomPairGenerator(**self.parameters)
        elif self.fp_type == "MorganFP":
            return rdFG.GetMorganGenerator(**self.parameters)
        elif self.fp_type == "RDKitFP":
            return rdFG.GetRDKitFPGenerator(**self.parameters)
        elif self.fp_type == "TopologicalTorsionFP":
            return rdFG.GetTopologicalTorsionGenerator(**self.parameters)
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fp_type}")

    @property
    def fpSize(self) -> int:
        """
        Returns the size of the fingerprint.

        Returns
        -------
        int
            The size of the fingerprint.
        """
        return self.parameters["fpSize"]

    @property
    def countSimulation(self) -> bool:
        """
        Returns the countSimulation flag of the fingerprint.

        Returns
        -------
        bool
            The countSimulation flag of the fingerprint.
        """
        return self.parameters["countSimulation"]

    @classmethod
    def is_supported(cls, fp_type: str) -> bool:
        return fp_type in cls.supported_types
