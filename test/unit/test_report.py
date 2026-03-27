import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from sklearn.model_selection import KFold

from mother.cv.report import cluster as cluster_report

rng = np.random.default_rng(42)


@pytest.fixture
def sample_data():
    """Common fixture providing sample data for testing."""

    n_samples = 100
    df = pd.DataFrame(
        {
            "cluster": rng.integers(0, 5, n_samples),
            "condition": rng.choice(["A", "B", "C"], n_samples),
            "experimental": rng.normal(0, 1, n_samples),
        }
    )
    features = pd.DataFrame(rng.random((n_samples, 10)))
    return df, features


@pytest.fixture
def sample_molecules():
    # Create some sample molecules
    smiles_list = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic Acid
        "CC(=O)O",  # Duplicate Acetic Acid
        "CCO",  # Duplicate of Ethanol
    ]
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]


@pytest.fixture
def cv_data():
    """Fixture providing data for cross-validation visualization."""
    n_samples = 50
    X = rng.random((n_samples, 5))
    y = rng.integers(0, 2, n_samples)
    groups = rng.integers(0, 5, n_samples)
    return X, y, groups


class TestInteractiveUMAP:
    """Tests for UMAP visualization functions."""

    def test_plot_interactive_umap(self, sample_data, monkeypatch):
        df, features = sample_data

        # Disable notebook/UI output
        monkeypatch.setattr(cluster_report, "display", lambda *args, **kwargs: None)

        # Disable Plotly window/render call
        class _DummyFig:
            def show(self, *args, **kwargs):
                return None

        monkeypatch.setattr(cluster_report.px, "scatter", lambda *args, **kwargs: _DummyFig())

        cluster_report.plot_interactive_umap(
            df=df,
            features=features,
            cluster_col="cluster",
            expt_col="experimental",
        )

    def test_update_plot_input_validation(self, sample_data):
        df, features = sample_data

        # Test with invalid n_neighbors
        with pytest.raises(ValueError):
            cluster_report.update_plot(
                n_neighbors=-1, min_dist=0.1, df=df, features=features, cluster_col="cluster", expt_col="experimental"
            )

        # Test with invalid min_dist
        with pytest.raises(ValueError):
            cluster_report.update_plot(
                n_neighbors=15, min_dist=-0.1, df=df, features=features, cluster_col="cluster", expt_col="experimental"
            )


@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame for testing."""
    data = {"cluster": ["A", "B", "A", "C", "B", "B"]}
    return pd.DataFrame(data)


@pytest.fixture
def thresholds():
    """Fixture to provide a list of valid thresholds for testing."""
    return [1, 2, 3]


@pytest.fixture
def empty_dataframe():
    """Fixture to create an empty DataFrame."""
    return pd.DataFrame(columns=["cluster"])


@pytest.fixture
def dataframe_without_cluster_col():
    """Fixture for a DataFrame without the specified cluster column."""
    return pd.DataFrame({"other_col": [1, 2, 3, 4, 5]})


class TestGetShortClusterSummary:
    def test_valid_input(self, sample_dataframe, thresholds):
        """Test with valid input."""
        try:
            cluster_report.get_short_cluster_summary("cluster", sample_dataframe, thresholds)
        except Exception:
            pytest.fail("get_short_cluster_summary raised an exception unexpectedly!")

    def test_null_values_in_cluster_column(self):
        """Test when the cluster column contains null values."""
        data = {"cluster": ["A", "B", None, "C"]}
        df = pd.DataFrame(data)
        thresholds = [1]

        with pytest.raises(ValueError, match="Column 'cluster' contains null values"):
            cluster_report.get_short_cluster_summary("cluster", df, thresholds)

    def test_negative_threshold(self, sample_dataframe):
        """Test when thresholds contain negative values."""
        thresholds = [-1]

        with pytest.raises(ValueError, match="All thresholds must be non-negative"):
            cluster_report.get_short_cluster_summary("cluster", sample_dataframe, thresholds)

    def test_non_integer_thresholds(self, sample_dataframe):
        """Test when thresholds contain non-integer values."""
        thresholds = [1.5, 1, "2"]

        with pytest.raises(TypeError, match="Thresholds must be a list of integers"):
            cluster_report.get_short_cluster_summary("cluster", sample_dataframe, thresholds)

    def test_empty_dataframe(self, empty_dataframe, thresholds):
        """Test with an empty DataFrame."""
        try:
            cluster_report.get_short_cluster_summary("cluster", empty_dataframe, thresholds)
        except Exception:
            pytest.fail("get_short_cluster_summary raised an exception unexpectedly!")

    def test_cluster_count_logging(self, caplog, sample_dataframe):
        """Test logging of cluster counts."""
        thresholds = [1, 2]

        cluster_report.get_short_cluster_summary("cluster", sample_dataframe, thresholds)

        assert "Number of clusters of size 1: 1" in caplog.text
        assert "Number of clusters of size 2: 1" in caplog.text

    def test_missing_cluster_col(self, dataframe_without_cluster_col, thresholds):
        """Test that ValueError is raised when cluster_col is not in the DataFrame."""
        with pytest.raises(ValueError, match="Column 'cluster' not found in the DataFrame"):
            cluster_report.get_short_cluster_summary("cluster", dataframe_without_cluster_col, thresholds)


class TestSilhouetteScore:
    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine", "jaccard"])
    def test_get_silhouette_different_metrics(self, sample_data, metric):
        df, features = sample_data
        score = cluster_report.get_silhouette(features, df["cluster"], metric=metric)
        assert isinstance(score, float)
        assert -1 <= score <= 1

    def test_get_silhouette_edge_cases(self, sample_data):
        df, features = sample_data

        # Single cluster case
        df_single = df.copy()
        df_single["cluster"] = 0
        with pytest.raises(ValueError):
            cluster_report.get_silhouette(features, df_single["cluster"])


@pytest.fixture
def invalid_molecules():
    return [None, Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC(=O)O"), Chem.MolFromSmiles("InvalidSmiles")]


@pytest.fixture
def valid_molecules():
    """Fixture to create a list of valid RDKit molecule objects."""
    return [
        Chem.MolFromSmiles("CCO"),  # Ethanol
        Chem.MolFromSmiles("CCN"),  # Ethylamine
        Chem.MolFromSmiles("CCC"),  # Propane
        Chem.MolFromSmiles("C1CCCCC1"),  # Cyclohexane
    ]


@pytest.fixture
def empty_molecules():
    """Fixture to create an empty list of RDKit molecule objects."""
    return []


class TestVisualizeScaffolds:
    def test_filter_unique_mols(self, sample_molecules):
        unique_mols = cluster_report._filter_unique_mols(sample_molecules)
        unique_smiles = [Chem.MolToSmiles(mol) for mol in unique_mols]
        expected_smiles = ["CCO", "CC(=O)O"]

        assert len(unique_smiles) == len(expected_smiles), "Outputs have different lengths."
        assert unique_smiles[0] in expected_smiles, f"Expected {expected_smiles}, but got {unique_smiles}"
        assert unique_smiles[1] in expected_smiles, f"Expected {expected_smiles}, but got {unique_smiles}"
        assert all(isinstance(mol, Chem.Mol) for mol in unique_mols), "Not all items are valid RDKit Mol objects"

    def test_filter_unique_mols_with_invalid(self, caplog, invalid_molecules):
        unique_mols = cluster_report._filter_unique_mols(invalid_molecules)
        assert len(unique_mols) == 2, "Should have 2 valid molecules after filtering None"
        assert "Encountered None molecule" in caplog.text
        assert all(isinstance(mol, Chem.Mol) for mol in unique_mols)

    def test_align_molecules(self, sample_molecules):
        aligned_scaffolds = cluster_report._align_to_mcs(sample_molecules)
        assert aligned_scaffolds is not None
        assert len(aligned_scaffolds) == len(sample_molecules)
        assert all(mol.GetNumConformers() > 0 for mol in aligned_scaffolds)

    def test_visualize_scaffolds_basic(self, valid_molecules):
        """Test basic visualization with valid molecules."""
        img = cluster_report.visualize_scaffolds(valid_molecules)
        assert img is not None

    def test_visualize_scaffolds_empty(self, empty_molecules):
        """Test visualization with empty molecule list."""
        with pytest.raises(ValueError, match="No valid molecules to visualize"):
            cluster_report.visualize_scaffolds(empty_molecules)

    @pytest.mark.parametrize("max_mols", [1, 10, 1000])
    def test_visualize_scaffolds_max_mols(self, valid_molecules, max_mols):
        """Test limiting number of molecules in visualization."""
        img = cluster_report.visualize_scaffolds(valid_molecules, max_mols=max_mols)
        assert img is not None

    def test_none_molecule(self):
        """Test handling of None molecules."""
        input_mols = [None, None, None]
        result = cluster_report._filter_unique_mols(input_mols)
        assert result == []


@pytest.fixture
def sample_cv_data():
    np.random.seed(42)
    X = rng.random((50, 4))
    y = rng.integers(0, 2, 50)
    group = rng.integers(0, 3, 50)
    n_splits = 5
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    return X, y, group, n_splits, cv


class TestPlotCVIndices:
    def test_plot_cv_indices_runs_and_returns_ax(self, sample_cv_data):
        X, y, group, n_splits, cv = sample_cv_data
        _, ax = plt.subplots()
        cluster_report.plot_cv_indices(cv, X, y, group, n_splits, ax=ax)

        # Check basic plot properties
        assert len(ax.collections) > 0  # Should have scatter points
        assert ax.get_xlabel() == "Sample index"
        assert ax.get_ylabel() == "CV iteration"

    def test_plot_cv_indices_axes_limits(self, sample_cv_data):
        X, y, group, n_splits, cv = sample_cv_data
        fig, ax = plt.subplots()
        cluster_report.plot_cv_indices(cv, X, y, group, n_splits, ax=ax)
        assert ax.get_xlim() == (0, len(X))
        assert ax.get_ylim() == (n_splits + 2.2, -0.2)
        plt.close(fig)

    def test_plot_cv_indices_invalid_cv(self, sample_cv_data):
        X, y, group, n_splits, _ = sample_cv_data
        fig, ax = plt.subplots()
        with pytest.raises(AttributeError):
            cluster_report.plot_cv_indices(None, X, y, group, n_splits, ax=ax)
        plt.close(fig)

    def test_plot_cv_indices_mismatched_group_length(self, sample_cv_data):
        X, y, group, n_splits, cv = sample_cv_data
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            cluster_report.plot_cv_indices(cv, X, y, group[:10], n_splits, ax=ax)
        plt.close(fig)

    def test_plot_cv_indices_different_n_splits(self):
        X = rng.random((30, 2))
        y = rng.integers(0, 2, 30)
        group = rng.integers(0, 2, 30)
        n_splits = 3
        cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        fig, ax = plt.subplots()
        cluster_report.plot_cv_indices(cv, X, y, group, n_splits, ax=ax)
        assert ax.get_ylim() == (n_splits + 2.2, -0.2)
        plt.close(fig)
