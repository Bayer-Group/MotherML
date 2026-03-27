"""Base class for grouping/clustering algorithms."""

import logging
from datetime import datetime
from typing import Any, Iterable, List, Literal, Optional

import numpy as np
import numpy.typing as npt
from rdkit import Chem, DataStructs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from mother import chem
from mother.cv import cv_methods
from mother.feature_generation.core import _TransformOnlyValidMols
from mother.feature_generation.fp_gen import FingerprintFactory

module_logger: logging.Logger = logging.getLogger(__name__)


class _InstantiateFingerprintGenerator(BaseEstimator):
    """A base class for generating molecular fingerprints.

    This class provides functionality for creating and managing molecular fingerprint
    generators, with support for pickling/unpickling. It inherits from scikit-learn's
    BaseEstimator for compatibility with scikit-learn's API.

    Attributes:
        radius (int): The radius for Morgan fingerprint generation. Defaults to 2.
        fp_size (int): The size (length) of generated fingerprints. Defaults to 2048.
        include_chirality (bool): Whether to include chirality information in fingerprints.
            Defaults to True.
        generator: The fingerprint generator instance. Initially None, created during fit.
    """

    def __init__(self, radius: int = 2, fp_size: int = 2048, include_chirality: bool = True):
        """Initialize the fingerprint generator.

        Args:
            radius (int, optional): The radius for Morgan fingerprint generation. Defaults to 2.
            fp_size (int, optional): The size of generated fingerprints. Defaults to 2048.
            include_chirality (bool, optional): Whether to include chirality. Defaults to True.
        """
        self.radius = radius
        self.fp_size = fp_size
        self.include_chirality = include_chirality
        self.generator = None

    def fit(
        self,
        X: Any = None,
        y: Any = None,
    ) -> BaseEstimator:
        """Create and initialize the fingerprint generator.

        This method creates a new fingerprint generator using the configured parameters.
        X and y parameters exist for compatibility with scikit-learn's API but are not used.

        Args:
            X (Any, optional): Ignored. Exists for scikit-learn compatibility.
            y (Any, optional): Ignored. Exists for scikit-learn compatibility.

        Returns:
            BaseEstimator: The fitted estimator (self).
        """

        self.generator = FingerprintFactory(
            "MorganFP", {"radius": self.radius, "fpSize": self.fp_size, "includeChirality": self.include_chirality}
        ).get_fingerprint_generator()

        return self


class DefaultGrouping(BaseEstimator, TransformerMixin):
    """A base class for grouping and clustering algorithms.

    This class provides default functionality for grouping data, implementing
    scikit-learn's BaseEstimator and TransformerMixin interfaces.

    Attributes:
        name (str): The name of the grouping, used in feature naming.
        is_fitted (bool): Whether the estimator has been fitted.
    """

    def __init__(self) -> None:
        """Initialize the DefaultGrouping instance."""
        self.name: str = "groups"
        self.is_fitted: bool = False

    def fit(self, X: Iterable[Any]) -> BaseEstimator:
        """Fit the grouping estimator.

        This method marks the estimator as fitted. Should be called before transform.

        Args:
            X (Iterable[Any]): The training data.

        Returns:
            BaseEstimator: The fitted estimator (self).
        """
        self.is_fitted = True
        return self

    def transform(self, X: Iterable[Any]) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "is_fitted")
        try:
            X_np: npt.NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        except ValueError as e:
            raise ValueError("Input contains non-numeric data, which cannot be converted to a numeric type.") from e
        if np.isnan(X_np).any():
            raise ValueError("'NA' value found in provided list. Please check your Input.")

        return X_np

    def get_output_dimension(self) -> int:
        return 1

    def get_feature_names_out(self, *args: Any, **kwargs: Any) -> List[str]:
        return [self.name]


class TimeSeriesGrouping(DefaultGrouping):
    """A class for grouping time series data.

    This class extends DefaultGrouping to handle datetime-based grouping,
    ensuring data is properly formatted and sorted for time series analysis.

    Args:
        datetime_fmt (str): Format string for parsing datetime strings.
    """

    def __init__(self, datetime_fmt: str) -> None:
        self.datetime_fmt: str = datetime_fmt

    def fit(self, X: Iterable[Any], y=None) -> BaseEstimator:
        module_logger.debug(
            "Checking if all elements in input are of type datetime or can be formatted to the given format string"
        )
        for element in list(X):
            if not isinstance(element, datetime):
                try:
                    datetime.strptime(element, self.datetime_fmt)
                except ValueError:
                    raise ValueError(f"Element '{element}' does not match the datetime format '{self.datetime_fmt}'")
        self.is_fitted: bool = True
        return self

    def transform(self, X: Iterable[Any]) -> npt.NDArray[np.datetime64]:
        module_logger.info("Check if input is sorted and can be used for Time Series Split")
        module_logger.debug("Raise if input is not sorted")
        check_is_fitted(self, "is_fitted")
        X: List[datetime] = [
            datetime.strptime(element, self.datetime_fmt) if not isinstance(element, datetime) else element
            for element in X
        ]
        if X != sorted(X):
            raise ValueError("Provided data are not sorted and can not be used for Time Series Split.")
        module_logger.info("Data is sorted")
        return np.array(X, dtype="datetime64")


class TanimotoGroupingFromMols(
    _InstantiateFingerprintGenerator,
    DefaultGrouping,
    _TransformOnlyValidMols,
):
    """Clustering, based on Tanimoto similarity. Will be calculated on user-specified
    features (Usually morgan fingerprints)"""

    def __init__(
        self, similarity_threshold: float = 0.8, radius: int = 2, fp_size: int = 2048, include_chirality: bool = True
    ) -> None:
        """
        Parameters:
        ----------
        similarity_threshold: float, default=0.8
            The Tanimoto similarity threshold for clustering.
        radius: int, default=2
            Radius for the Morgan fingerprint generator.
        fp_size: int, default=2048
            Length of the generated fingerprint.
        include_chirality: bool, default=True
            Whether to include chirality in the fingerprint generation.
        """
        super().__init__(radius, fp_size, include_chirality)
        self.similarity_threshold: float = similarity_threshold
        self.name = "tanimoto-group"

    def transform(self, valid_compounds: Iterable[chem.Chem.Mol], **kwargs: Any) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "generator")

        fingerprints: tuple[DataStructs.ExplicitBitVect, ...] = self.generator.GetFingerprints(
            valid_compounds, kwargs.get("numThreads", 1)
        )

        clusters: dict[int, List[int]] = cv_methods.tanimoto_sphere_exclusion_clustering(
            fingerprints, similarity_threshold=self.similarity_threshold
        )

        y: npt.NDArray[np.float64] = np.full((len(list(valid_compounds)), 1), np.nan)
        for cluster, member in clusters.items():
            y[member, 0] = cluster

        return y


class HdbscanGroupingFromMols(_InstantiateFingerprintGenerator, DefaultGrouping, _TransformOnlyValidMols):
    """A class to perform HDBSCAN clustering on a dataset of chemical compounds"""

    def __init__(
        self,
        scaffold: Literal["NoScaffold", "Murcko", "GenericMurcko"] = "NoScaffold",
        min_cluster_size: int = 5,
        radius: int = 2,
        fp_size: int = 2048,
        include_chirality: bool = True,
    ) -> None:
        """
        Parameters:
        ----------
        scaffold: (Literal["NoScaffold", "Murcko", "GenericMurcko"])Literal ()
            A string literal that specifies the type of scaffold to be used.
            Must be one of the following values:
            - "NoScaffold": Indicates no scaffold is used.
            - "Murcko": Indicates the use of the Murcko scaffold.
            - "GenericMurcko": Indicates a generic version of the Murcko scaffold.
        min_cluster_size: int, default = 5
            Integer indicating the smallest size grouping which should be considered a cluster.
        radius: int, default=2
            Radius for the Morgan fingerprint generator.
        fp_size: int, default=2048
            Length of the generated fingerprint.
        include_chirality: bool, default=True
            Whether to include chirality in the fingerprint generation.

        """

        super().__init__(radius, fp_size, include_chirality)
        self.scaffold = scaffold
        self.min_cluster_size = min_cluster_size
        self.name = "hdbscan-group"

    def transform(self, valid_compounds: Iterable[chem.Chem.Mol], **kwargs: Any) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "generator")

        self.molecules: List[Chem.Mol] = cv_methods.murcko_scaffold_reduction(
            mol_series=valid_compounds, scaffold=self.scaffold
        )

        fingerprints: tuple[DataStructs.ExplicitBitVect, ...] = self.generator.GetFingerprints(
            self.molecules, kwargs.get("numThreads", 1)
        )

        clusters: dict[int, List[int]] = cv_methods.hdbscan_clustering(
            fingerprints, min_cluster_size=self.min_cluster_size
        )

        y: npt.NDArray[np.float64] = np.full((len(list(valid_compounds)), 1), np.nan)
        for cluster, member in clusters.items():
            y[member, 0] = cluster

        return y

    def get_murcko_scaffolds_out(self) -> List[Chem.Mol]:
        """
        If performed, returns the outcome of the Murcko scaffolding.
        Scaffolds were generated from the input molecules. If no scaffolding was performed,
        the input molecules are returned.

        Returns:
        -------
        self.molecules: List[Chem.Mol]
            List of generated molecular representatives as Murcko scaffolds or same molecules,
            each represented as RDKit molecular objects
        """
        return self.molecules


class KMedoidsGroupingFromMols(_InstantiateFingerprintGenerator, DefaultGrouping, _TransformOnlyValidMols):
    """Apply K-medoids (PAM) on molecular fingerprints"""

    def __init__(
        self,
        scaffold: Literal["NoScaffold", "GenericMurcko", "Murcko"] = "NoScaffold",
        clusters_number: int = 2,
        iteration_method: str = "random",
        max_iter: int = 100,
        random_state: Optional[int] = None,
        radius: int = 2,
        fp_size: int = 2048,
        include_chirality: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        scaffold: Literal["NoScaffold", "Murcko", "GenericMurcko"], default="NoScaffold"
            Type of scaffold to use
        clusters_number: int, default=5
            Number of clusters to generate
        initiation_method: str, default="random"
            Method to initialize medoids ("random", "first" or "build")
        max_iter: int, default=100
            Maximum number of iterations for the algorithm
        random_state: Optional[int], default=None
            Random seed for reproducibility
        radius: int, default=2
            Radius for Morgan fingerprint generation
        fp_size: int, default=2048
            Length of generated fingerprint
        include_chirality: bool, default=True
            Whether to include chirality in fingerprint generation
        """

        super().__init__(radius, fp_size, include_chirality)

        self.name = "kmedoids-group"
        self.scaffold = scaffold
        self.clusters_number = clusters_number
        self.iteration_method = iteration_method
        self.max_iter = max_iter
        self.random_state = random_state

    def transform(self, valid_compounds: Iterable[chem.Chem.Mol], **kwargs: Any) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "generator")

        self.molecules: List[Chem.Mol] = cv_methods.murcko_scaffold_reduction(
            mol_series=valid_compounds, scaffold=self.scaffold
        )

        fingerprints: tuple[DataStructs.ExplicitBitVect, ...] = self.generator.GetFingerprints(
            self.molecules, kwargs.get("numThreads", 1)
        )

        clusters = cv_methods.kmedoids_clustering(
            fingerprints,
            clusters_number=self.clusters_number,
            random_state=self.random_state,
            initiation_method=self.iteration_method,
            max_iter=self.max_iter,
        )

        y: npt.NDArray[np.float64] = np.full((len(list(valid_compounds)), 1), np.nan)
        for cluster, members in clusters.items():
            y[members, 0] = cluster

        return y

    def get_murcko_scaffolds_out(self) -> List[Chem.Mol]:
        """
        Returns the Murcko scaffolds if scaffolding was performed,
        otherwise returns the input molecules.

        Returns
        -------
        List[Chem.Mol]
            List of molecular representatives as Murcko scaffolds or original molecules
        """
        return self.molecules
