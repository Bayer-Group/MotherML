try:
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import plotly.express as px
    import umap
except ImportError as import_error:
    from mother.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("report", import_error)

import logging
import typing

import numpy as np
import pandas as pd
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdFMCS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

module_logger = logging.getLogger(__name__)


def update_plot(
    n_neighbors: int, min_dist: float, df: pd.DataFrame, features: pd.DataFrame, cluster_col: str, expt_col: str
) -> None:
    """Update UMAP plot with new parameters.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for UMAP. Must be positive.
    min_dist : float
        Minimum distance parameter for UMAP. Must be between 0 and 1.
    df : pandas.DataFrame
        DataFrame containing cluster labels and experimental values
    features : pandas.DataFrame
        DataFrame containing feature vectors
    cluster_col : str
        Name of column containing cluster labels
    expt_col : str
        Name of column containing experimental values

    Raises
    ------
    ValueError
        If n_neighbors is not positive or min_dist is not between 0 and 1
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if not 0 <= min_dist <= 1:
        raise ValueError("min_dist must be between 0 and 1")

    scaler: StandardScaler = StandardScaler()
    scaled_features: np.ndarray = scaler.fit_transform(features)

    reducer: umap.UMAP = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
    embedding = reducer.fit_transform(scaled_features)

    umap_df: pd.DataFrame = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "Cluster": df[cluster_col],
            "Experimental": df[expt_col],
        }
    )
    umap_df["Cluster"] = umap_df["Cluster"].astype(str)

    fig = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color="Cluster",
        title="Interactive UMAP Projection",
        labels={"UMAP1": "UMAP 1", "UMAP2": "UMAP 2"},
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        hover_data=["Experimental"],
        width=800,
        height=600,
    )
    fig.show()


def plot_interactive_umap(df: pd.DataFrame, features: pd.DataFrame, cluster_col: str, expt_col: str) -> None:
    """
    Create an interactive UMAP plot with adjustable parameters.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the dataset to be visualized. It should contain the cluster_col collecting cluster labels
        and expt_col collecting experimental parameter values.
    features : pandas.DataFrame
        DataFrame containing the feature vectors, preferably used for clustering.
    cluster_col : str
        Column name for the column containing cluster labels.
    expt_col : str
        Column name containing experimentally determined target values.
    """

    n_neighbors_slider = widgets.IntSlider(value=15, min=1, max=200, step=1, description="n_neighbors:")
    min_dist_slider = widgets.FloatSlider(value=0.1, min=0.0, max=1.0, step=0.01, description="min_dist:")

    ui = widgets.VBox([n_neighbors_slider, min_dist_slider])
    out = widgets.interactive_output(
        update_plot,
        {
            "n_neighbors": n_neighbors_slider,
            "min_dist": min_dist_slider,
            "df": widgets.fixed(df),
            "features": widgets.fixed(features),
            "cluster_col": widgets.fixed(cluster_col),
            "expt_col": widgets.fixed(expt_col),
        },
    )
    display(ui, out)


def get_short_cluster_summary(cluster_col: str, df: pd.DataFrame, thresholds: list[int]) -> None:
    """
    Generate a comprehensive summary of the clustering results.

    Parameters:
    ----------
    cluster_col : str
        The name of the column containing cluster labels in the DataFrame.
        This column should contain categorical values representing different clusters.
    df : pd.DataFrame
        The DataFrame containing the clustering results.
    thresholds: list[int]
        A list of thresholds indicating the sizes of clusters to check.
        This function will count how many clusters match the specified threshold requirement.
        Must be non-negative integers.

    Raises:
    -------
    ValueError
        If cluster_col is not in df, df contains null values, or thresholds contains negative values.
    TypeError
        If thresholds contains non-integer values.
    """
    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in the DataFrame")
    if df[cluster_col].isnull().any():
        raise ValueError(f"Column '{cluster_col}' contains null values")

    try:
        if not all(isinstance(t, (int, np.integer)) for t in thresholds):
            raise TypeError("All thresholds must be integers")
        if any(t < 0 for t in thresholds):
            raise ValueError("All thresholds must be non-negative")
    except TypeError:
        raise TypeError("Thresholds must be a list of integers")

    if df.empty:
        module_logger.info("DataFrame is empty - no clusters to analyze")
        return

    cluster_counts: pd.Series = df[cluster_col].value_counts().sort_index()

    for threshold in sorted(thresholds):
        clusters_of_size: int = sum(1 for count in cluster_counts if count == threshold)
        module_logger.info(f"Number of clusters of size {threshold}: {clusters_of_size}")


def get_silhouette(X: np.ndarray, cluster_labels: np.ndarray, metric: str = "jaccard"):
    """
        Calculate and log the average silhouette score for the given clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    cluster_labels : array-like, shape (n_samples,)
        Cluster labels for each sample.
    metric : str, optional, default="jaccard"
        The metric to use for distance computation.
    """

    silhouette = silhouette_score(X, cluster_labels, metric=metric)
    module_logger.info(f"Average Silhouette Score: {silhouette:.4f}")
    return silhouette


def _filter_unique_mols(molecules: typing.List[Chem.Mol]) -> typing.List[Chem.Mol]:
    """
    Filters the input list of molecules to return only unique and valid RDKit molecule objects.

    Parameters
    ----------
    molecules: List[Chem.Mol]
        A list of RDKit molecule objects.

    Returns
    -------
    List[Chem.Mol]
        A list of unique and valid RDKit molecule objects.
    """
    unique_mols: set[str] = set()
    valid_mols: typing.List[Chem.Mol] = []

    for mol in molecules:
        if mol is None:
            module_logger.warning("Encountered None molecule, skipping.")
            continue
        smiles: str = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smiles not in unique_mols:
            unique_mols.add(smiles)
            valid_mols.append(mol)

    return valid_mols


def _align_to_mcs(input_molecules: typing.List[Chem.Mol]) -> typing.List[Chem.Mol]:
    """
    Aligns the input molecules to their maximum common substructure (MCS) and computes 2D coordinates.
    This function modifies the input molecules in place to ensure they are aligned and have 2D coordinates.

    Parameters:
    ----------
    input_molecules: typing.List[Chem.Mol]
        A list of RDKit molecule objects to be aligned.

    Returns:
    --------
    input_molecules: typing.List[Chem.Mol]
        A list of RDKit molecule objects with computed 2D coordinates and aligned to MCS.
    """
    for mol in input_molecules:
        if mol is not None:
            rdDepictor.Compute2DCoords(mol)

    if len(input_molecules) > 1:
        mcs = rdFMCS.FindMCS(input_molecules, completeRingsOnly=True)
        mcs_mol: typing.Optional[Chem.Mol] = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is not None:
            rdDepictor.Compute2DCoords(mcs_mol)
            for mol in input_molecules:
                rdDepictor.GenerateDepictionMatching2DStructure(mol, mcs_mol, acceptFailure=True)

    return input_molecules


def visualize_scaffolds(molecules: typing.List[Chem.Mol], max_mols: int = 70):
    """
    Visualizes a list of unique molecular scaffolds from a collection of RDKit molecule objects.

    This function generates a grid image of the unique scaffolds derived from the input
    molecules, computes their 2D coordinates, and attempts to align them based on
    their maximum common substructure (MCS).

    Parameters
    ----------
    molecules: List[Chem.Mol]
        A list of RDKit molecule objects to visualize. The function will extract unique
        scaffolds from these molecules for the visualization.

    max_mols: int, optional
        The maximum number of molecules to display in the visualization grid. Default is 70.

    Returns
    -------
    img:
        A grid image containing the 2D representations of the unique scaffolds,
        with optional legends indicating the scaffold numbers.

    Notes
    -----
    - The function will compute 2D coordinates for each scaffold and attempt to align
      them based on their MCS. If MCS alignment fails, it will fall back to basic
      2D coordinates.
    - The function filters out any empty SMILES representations when extracting unique
      scaffolds.

    Example
    -------
    >>> from rdkit import Chem
    >>> molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")]
    >>> img = visualize_scaffolds(molecules)
    >>> img.show()
    """

    if not molecules:
        raise ValueError("No valid molecules to visualize")

    unique_scaffolds_mols = _filter_unique_mols(molecules)

    aligned_scaffolds = _align_to_mcs(unique_scaffolds_mols)

    # Limit number of molecules if needed
    if max_mols:
        aligned_scaffolds = aligned_scaffolds[:max_mols]

    legends = [f"Scaffold {n}" for n in range(1, len(aligned_scaffolds) + 1)]

    img = Draw.MolsToGridImage(
        aligned_scaffolds,
        molsPerRow=4,
        subImgSize=(150, 150),
        legends=legends,
    )

    return img


def plot_cv_indices(cv, X, y, group, n_splits, ax=None, lw=10, **kwargs):
    """
    Create a visualization of cross-validation behavior, showing how data is split into training and test sets.

    This function creates a detailed visualization that shows:
    1. How samples are split into training/test sets across different CV iterations
    2. The distribution of class labels
    3. The distribution of group labels (if using group-based CV)

    The plot contains multiple rows:
    - One row per CV split showing training (0) and test (1) sample assignments
    - One row showing the class distribution
    - One row showing the group distribution

    Parameters
    ----------
    cv : cross-validation generator
        The cross-validation splitting strategy (e.g., KFold, GroupKFold).
    X : array-like of shape (n_samples, n_features)
        The data to fit. Used only to determine the number of samples.
    y : array-like of shape (n_samples,)
        The target variable for coloring the class distribution row.
    group : array-like of shape (n_samples,)
        Group labels for the samples, used for coloring the group distribution row
        and for group-based CV strategies.
    n_splits : int
        Number of splits/folds in the cross-validation.
    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the visualization. Default is the current axes.
    lw : float, optional
        Line width of the samples in the plot. Default is 10.
    **kwargs : dict, optional
        Additional keyword arguments to pass to matplotlib plotting functions.
        Supported arguments:
        - fontsize: int, size of the title font (default: 15)

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot, showing CV splits, class distribution,
        and group distribution.

    Notes
    -----
    - The CV splits are shown using a coolwarm colormap (blue for training, red for test)
    - Class and group distributions are shown using the 'Paired' colormap
    - Each sample's assignment is shown as a horizontal line
    """
    # Create axis if none provided
    if ax is None:
        _, ax = plt.subplots()

    # Get colormaps for data and CV splits
    cmap_data = plt.colormaps.get_cmap("Paired")
    cmap_cv = plt.colormaps.get_cmap("coolwarm")

    # Check if using group-based CV
    use_groups = "Group" in type(cv).__name__
    groups = group if use_groups else None

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data)
    ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, len(X)],
    )
    title = kwargs.get("title", type(cv).__name__)
    ax.set_title(title, fontsize=kwargs.get("fontsize", 15))
