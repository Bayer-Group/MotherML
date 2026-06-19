import collections
import logging
import warnings
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.SimDivFilters import rdSimDivPickers
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import pairwise_distances

module_logger = logging.getLogger(__name__)


def np_to_bv(fv: np.ndarray) -> DataStructs.ExplicitBitVect:
    """Convert numpy array to RDKit ExplicitBitVect. Best solution according to
    https://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg06656.html"""
    bv: DataStructs.ExplicitBitVect = DataStructs.ExplicitBitVect(len(fv))
    on_bits = np.nonzero(fv)[0]  # Get indices where array is non-zero
    if len(on_bits) > 0:
        bv.SetBitsFromList(on_bits.tolist())
    return bv


def tanimoto_sphere_exclusion_clustering(
    fingerprints: Union[np.ndarray, List[DataStructs.ExplicitBitVect], tuple[DataStructs.ExplicitBitVect]],
    similarity_threshold: float = 0.8,
):
    """
    Performs sphere exclusion clustering on tanimoto similarities of Morgan fingerprints

    Parameters
    ----------
    fingerprints: Union[np.ndarray, typing.List[DataStructs.ExplicitBitVect], tuple[DataStructs.ExplicitBitVect]]
        A list of fingerprints generated for the molecules or their scaffolds
    similarity_threshold: float
        A float value representing the Tanimoto similarity threshold for clustering.
        The default value is 0.8.


    Returns
    -------
    Indices to use in cross validation groups

    """
    if isinstance(fingerprints, np.ndarray):
        fingerprints = [np_to_bv(fv) for fv in fingerprints]
    elif isinstance(fingerprints, tuple):
        fingerprints = list(fingerprints)

    module_logger.info(f"Applying clustering with Tanimoto similarity: {similarity_threshold}")

    if not fingerprints:
        module_logger.warning("No valid fingerprints provided for clustering")
        return {}

    centroids = _pick_centroids(fingerprints, 1 - similarity_threshold)
    if not centroids:
        module_logger.warning("No centroids found during clustering")
        return {0: [i for i in range(len(fingerprints))]}

    clusters = _assign_points_to_clusters(centroids, fingerprints)
    module_logger.info(f"Found: {len(clusters)} clusters")

    return clusters


def _pick_centroids(fingerprints: List[DataStructs.ExplicitBitVect], distance_threshold: float) -> List[int]:
    """Pick centroids for sphere exclusion clustering using RDKit's LeaderPicker.

    Args:
        fingerprints (List[DataStructs.ExplicitBitVect]): List of molecular fingerprints to cluster
        distance_threshold (float): Distance threshold for selecting centroids. This is
            typically 1 - similarity_threshold.

    Returns:
        List[int]: Indices of the selected centroid molecules
    """
    # noinspection PyArgumentList
    module_logger.debug(f"Picking centroids with distance threshold: {distance_threshold}")
    picker = rdSimDivPickers.LeaderPicker()
    try:
        centroids = picker.LazyBitVectorPick(fingerprints, len(fingerprints), distance_threshold)
        result = list(centroids)  # Convert to list to ensure we return List[int]
        module_logger.debug(f"Identified {len(result)} cluster centers")
        return result
    except Exception as e:
        module_logger.warning(f"Error picking centroids: {str(e)}")
        return []


def _assign_points_to_clusters(
    centroids: List[int], fingerprints: List[DataStructs.ExplicitBitVect]
) -> Dict[int, List[int]]:
    """Assign molecules to clusters based on Tanimoto similarity to centroids.

    This function calculates Tanimoto similarities between each molecule and all centroids,
    then assigns each molecule to the cluster of its most similar centroid.

    Args:
        centroids (List[int]): List of indices representing the centroid molecules
        fingerprints (List[DataStructs.ExplicitBitVect]): List of molecular fingerprints
            for all molecules in the dataset

    Returns:
        Dict[int, List[int]]: Dictionary mapping cluster IDs (0 to n-1) to lists of
            molecule indices belonging to each cluster
    """
    clusters: Dict[int, List[int]] = collections.defaultdict(list)
    similarities: np.ndarray = np.zeros((len(centroids), len(fingerprints)))

    for idx, centroid in enumerate(centroids):
        clusters[idx].append(centroid)
        similarities[idx, :] = DataStructs.BulkTanimotoSimilarity(fingerprints[centroid], fingerprints)
        similarities[idx, idx] = 0

    best: np.ndarray = np.argmax(similarities, axis=0)
    for idx, best_idx in enumerate(best):
        if idx not in centroids:
            clusters[best_idx].append(idx)
    return clusters


def hdbscan_clustering(
    fingerprints: Union[np.ndarray, List[DataStructs.ExplicitBitVect], tuple[DataStructs.ExplicitBitVect]],
    min_cluster_size: int = 5,
) -> Dict[int, List[int]]:
    """
    Apply the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
    clustering algorithm.

    Parameters
    ----------
    fingerprints: Union[np.ndarray, typing.List[DataStructs.ExplicitBitVect], tuple[DataStructs.ExplicitBitVect]]
        A list of fingerprints generated for the molecules or their scaffolds.
    min_cluster_size: int, default = 5
        Smallest size grouping that should be considered a cluster.

    Returns
    -------
    Dict[int, List[int]]
        A dictionary of cluster numbers as keys and values as lists of molecule indices.
    """
    from sklearn.cluster import HDBSCAN

    if isinstance(fingerprints, tuple):
        fingerprints = list(fingerprints)

    if len(fingerprints) == 0:
        module_logger.warning("No valid fingerprints provided for clustering")
        return {}

    module_logger.info("Applying HDBSCAN clustering to the dataset")

    # sklearn.cluster.HDBSCAN expects a 2D numeric feature matrix.
    # If the input is already a numpy array (n_samples × n_bits), use it directly.
    # If it is a list of RDKit ExplicitBitVect objects, convert to a 2D uint8 matrix.
    if isinstance(fingerprints, np.ndarray):
        X = fingerprints
    else:
        nbits = fingerprints[0].GetNumBits()
        X = np.zeros((len(fingerprints), nbits), dtype=np.uint8)
        for i, fp in enumerate(fingerprints):
            DataStructs.ConvertToNumpyArray(fp, X[i])

    clusters: Dict[int, List[int]] = collections.defaultdict(list)
    clusterer: HDBSCAN = HDBSCAN(min_samples=1, metric="jaccard", min_cluster_size=min_cluster_size)
    cluster_labels: np.ndarray = clusterer.fit_predict(X)

    module_logger.info(f"HDBSCAN clustering grouped data into {len(set(cluster_labels))} groups")

    for idx, best_idx in enumerate(cluster_labels):
        clusters[best_idx].append(idx)

    return clusters


def kmedoids_clustering(
    fingerprints: Union[np.ndarray, List[DataStructs.ExplicitBitVect], tuple[DataStructs.ExplicitBitVect]],
    clusters_number: int = 2,
    random_state: Optional[int] = None,
    initiation_method: str = "random",
    max_iter: int = 100,
) -> Dict[int, List[int]]:
    """
    Perform k-medoids clustering using Jaccard distance.

    Parameters
    ----------
    fingerprints: Union[np.ndarray, List[DataStructs.ExplicitBitVect], tuple[DataStructs.ExplicitBitVect]]
        A list of fingerprints generated for the molecules or their scaffolds
    clusters_number: int, default=2
        Number of clusters to generate
    random_state: Optional[int], default=None
        Random seed for reproducibility
    initiation_method: str, default="random"
        Method to initialize medoids ("random", "first" or "build")
    max_iter: int, default=100
        Maximum number of iterations for the algorithm

    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping cluster IDs to lists of molecule indices
    """
    try:
        from kmedoids import pam

    except ImportError as import_error:
        from mother.errors import ExtrasDependencyImportError

        raise ExtrasDependencyImportError("clustering", import_error)
    warnings.filterwarnings(action="ignore", category=DataConversionWarning)

    if isinstance(fingerprints, np.ndarray):
        fp_array: np.ndarray = fingerprints
    else:
        fp_array = np.array([list(fp) for fp in fingerprints])

    module_logger.info(f"Applying PAM k-medoids clustering to create {clusters_number} clusters")

    distance_matrix: np.ndarray = pairwise_distances(fp_array, metric="jaccard")

    # Initialize and fit k-medoids
    kmed: Any = pam(
        diss=distance_matrix,
        medoids=clusters_number,
        max_iter=max_iter,
        init=initiation_method,
        random_state=random_state,
    )

    clusters: Dict[int, List[int]] = collections.defaultdict(list)
    cluster_labels: np.ndarray = kmed.labels

    for idx, label in enumerate(cluster_labels):
        clusters[int(label)].append(idx)

    module_logger.info(f"PAM k-medoids clustering grouped data into {len(clusters)} clusters")
    return dict(clusters)


def murcko_scaffold_reduction(
    mol_series: Iterable[Union[Chem.rdchem.Mol, str]],
    scaffold: Literal["NoScaffold", "GenericMurcko", "Murcko"] = "NoScaffold",
) -> List[Chem.rdchem.Mol]:
    """
    Generates Murcko scaffold or generic Murcko scaffold from a series of molecules or SMILES strings.

    Parameters:
    -----------
    mol_series: Iterable[Union[Chem.rdchem.Mol, str]]
        RDKit molecule objects or SMILES strings representing molecules/scaffolds in the dataset
    scaffold: (Literal["NoScaffold", "Murcko", "GenericMurcko"])
        A string literal that specifies the type of scaffold to be used.
        Must be one of the following values:
        - "NoScaffold": Indicates no scaffold is used.
        - "Murcko": Indicates the use of the Murcko scaffold.
        - "GenericMurcko": Indicates a generic version of the Murcko scaffold.

    Returns:
    -------
    murcko_scaffold: list[Chem.rdchem.Mol]
        List of generated molecular representatives as Murcko scaffolds or same molecules,
        each represented as RDKit molecular objects
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mols: List[Union[Chem.rdchem.Mol, str]] = list(mol_series).copy()

    if scaffold == "GenericMurcko":
        generic_scaffolds: List[Chem.rdchem.Mol] = []

        for ind, mol in enumerate(mols):
            generic_scaffold: Chem.rdchem.Mol = MurckoScaffold.MakeScaffoldGeneric(mol)
            generic_scaffolds.append(generic_scaffold)
        return generic_scaffolds

    elif scaffold == "Murcko":
        return [
            mol
            if Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) == ""
            else MurckoScaffold.GetScaffoldForMol(mol)
            for mol in mols
        ]

    return mols
