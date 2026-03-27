from typing import Any, Dict, List, Optional, Tuple, Type

import dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import SettingsConfigDict
from rdkit.Chem import rdFingerprintGenerator as rdFG

from .fp_gen import FingerprintFactory


class MaccsFingerprintsParams(BaseModel):
    pass


class FingerprintParams(BaseModel):
    fpSize: int = Field(default=2048, description="Size of the fingerprint")
    countSimulation: bool = Field(default=False, description="Use countSimulation for fingerprint")
    countBounds: Optional[Any] = Field(
        default=None,
        description="""boundaries for count simulation, corresponding bit will be
        set if the count is higher than the number provided for that spot""",
    )
    atomInvariantsGenerator: Optional[Any] = Field(
        default=None, description="atom invariants to be used during fingerprint generation"
    )

    model_config = SettingsConfigDict(
        env_prefix="mother_fingerprint_",
        case_sensitive=True,
        use_enum_values=False,
        extra="allow",  # set to forbid?!?
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
    )


class MorganFingerprintsParams(FingerprintParams):
    radius: int = Field(default=2, description="the number of iterations to grow the fingerprint")
    includeChirality: bool = Field(default=False, description="Whether to include chirality in the fingerprints")
    useBondTypes: bool = Field(
        default=True, description="if set, bond types will be included as a part of the default bond invariants"
    )
    includeRedundantEnvironments: bool = Field(default=False, description="if set, use redundant environments")
    includeRingMembership: bool = Field(default=True, description="if set, include ring membership")
    onlyNonzeroInvariants: bool = Field(default=False, description="???")

    model_config = SettingsConfigDict(
        env_prefix="mother_fingerprint_",
        case_sensitive=True,
        use_enum_values=False,
        extra="allow",  # set to forbid?!?
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
    )


class AtomPairFingerprintsParams(FingerprintParams):
    includeChirality: bool = Field(default=False, description="Include chirality in the fingerprint")
    minDistance: int = Field(
        default=1, description="Minimum distance between atoms to be considered in a pair, default is 1 bond"
    )
    maxDistance: int = Field(
        default=30,
        description="Maximum distance between atoms to be considered in a pair, default is maxPathLen-1 bonds",
    )
    use2D: bool = Field(default=False, description=" if set, the 2D (topological) distance matrix will be used")

    model_config = SettingsConfigDict(
        env_prefix="mother_fingerprint_",
        case_sensitive=True,
        use_enum_values=False,
        extra="allow",  # set to forbid?!?
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
    )


class RDKitFingerprintsParams(FingerprintParams):
    minPath: int = Field(default=1, description="Minimum path length to be included in the fingerprint")
    maxPath: int = Field(default=7, description="Maximum path length to be included in the fingerprint")
    useHs: bool = Field(default=True, description="Include Hs in the fingerprint")
    branchedPaths: bool = Field(default=True, description="Include branched paths in the fingerprint")
    useBondOrder: bool = Field(default=True, description="Use bond order in the fingerprint")
    numBitsPerFeature: int = Field(default=2, description="The number of bits set per path/subgraph found")

    model_config = SettingsConfigDict(
        env_prefix="mother_fingerprint_",
        case_sensitive=True,
        use_enum_values=False,
        extra="allow",  # set to forbid?!?
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
    )


class TopologicalTorsionFingerprintsParams(FingerprintParams):
    includeChirality: bool = Field(default=False, description="Include chirality during descriptor calculation")
    torsionAtomCount: int = Field(default=4, description="The number of atoms to include in the torsions")

    model_config = SettingsConfigDict(
        env_prefix="mother_fingerprint_",
        case_sensitive=True,
        use_enum_values=False,
        extra="allow",  # set to forbid?!?
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
    )


class ChemicalDescriptorsParams(BaseModel):
    omit_prefixes: Tuple[str, ...] = Field(default=(), description="Prefixes to omit in descriptor methods")
    descriptor_prefix: str = Field(default="", description="Prefix for the descriptor name")
    descriptor_list: Optional[List[str]] = Field(default=None, description="List of available rdkit descriptors")

    # @field_validator("descriptor_list")
    # def validate_descriptors(self, v):
    #     # validate using pydantic validator (before?!)
    #     if not set(v).issubset([descriptor for descriptor, _ in Descriptors.descList]):
    #         raise ValidationError("Invalid rdkit descriptor")
    #     return v

    model_config = SettingsConfigDict(
        env_prefix="mother_fingerprint_",
        case_sensitive=True,
        use_enum_values=False,
        extra="allow",  # set to forbid?!?
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
    )


class FeatureGenerationConfig(BaseModel):
    fingerprints: List[Dict[str, Any]] = Field(default=[], description="List of fingerprint generator settings")
    maccs: bool = Field(default=False, description="Flag if maccs fingerprints should be generated")
    chemical_descriptors: Optional[ChemicalDescriptorsParams] = Field(default=None)
    use_counts: bool = Field(default=False, description="Whether to use count fingerprints")

    @field_validator("fingerprints", mode="before")
    @classmethod
    def validate_fp(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # return v if v.
        # check if fp_type is supported
        # TODO implement
        # FingerprintFactory.is_supported(v.key)
        for value in v:
            if not (len(value.keys()) == 1 and FingerprintFactory.is_supported(next(iter(value.keys())))):
                val_keys = rdFG.FPType.names.keys()
                raise ValueError(f"Unknown fingerprint type: {next(iter(value.keys()))}. Allowed types are: {val_keys}")
        return v

    model_config = SettingsConfigDict(
        env_prefix="mother_",
        case_sensitive=False,
        use_enum_values=False,
        extra="ignore",
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
        arbitrary_types_allowed=True,
    )


def get_params_for_fp_type(fp_type: str) -> Type[FingerprintParams]:
    """
    Returns the appropriate fingerprint parameters object based on the given fingerprint type.

    Args:
        fp_type (str): The type of fingerprint. Allowed values are "MorganFP", "AtomPairFP",
                       "RDKitFP", and "TopologicalTorsionFP".

    Returns:
        Type[FingerprintParams]: An instance of the corresponding fingerprint parameters class.

    Raises:
        ValueError: If the provided fingerprint type is not supported.
    """
    if fp_type == "MorganFP":
        return MorganFingerprintsParams
    elif fp_type == "AtomPairFP":
        return AtomPairFingerprintsParams
    elif fp_type == "RDKitFP":
        return RDKitFingerprintsParams
    elif fp_type == "TopologicalTorsionFP":
        return TopologicalTorsionFingerprintsParams
    else:
        raise ValueError(
            f"Unknown fingerprint type: {fp_type}. Allowed types are: {FingerprintFactory.supported_types}"
        )
