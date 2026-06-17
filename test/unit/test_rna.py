from typing import Tuple

try:
    import anndata
except ImportError as import_error:
    from mother.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rna", import_error)
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import the classes to test
from mother.ml.rna import RNA, LogisticRegressionL1FeatureSelector, ScanpyPreprocessor


@pytest.fixture
def sample_rna_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Create RNA-seq like count data for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.negative_binomial(n=5, p=0.5, size=(100, 1000)), columns=[f"gene_{i}" for i in range(1000)]
    )
    y = pd.Series(np.random.randint(0, 2, 100))
    return (X, y)


@pytest.fixture
def small_rna_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Create a small RNA-seq like dataset for faster tests."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.negative_binomial(n=5, p=0.5, size=(20, 50)), columns=[f"gene_{i}" for i in range(50)])
    y = pd.Series(np.random.randint(0, 2, 20))
    return (X, y)


@pytest.fixture
def anndata_rna_object() -> anndata.AnnData:
    """Create an AnnData object with RNA-seq like data for testing."""
    np.random.seed(42)
    X = np.random.negative_binomial(n=5, p=0.5, size=(30, 100))
    adata = anndata.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(100)]
    adata.obs_names = [f"cell_{i}" for i in range(30)]
    return adata


# Tests for LogisticRegressionL1FeatureSelector
class TestLogisticRegressionL1FeatureSelector:
    def test_init(self) -> None:
        """Test initialization of LogisticRegressionL1FeatureSelector."""
        selector = LogisticRegressionL1FeatureSelector(n_features=5, cv=3, random_state=42)
        assert selector.n_features == 5
        assert selector.cv == 3
        assert selector.random_state == 42
        assert selector.selected_features_ is None

    def test_fit(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit method with RNA-seq data."""
        X, y = sample_rna_data
        selector = LogisticRegressionL1FeatureSelector(n_features=10)
        result = selector.fit(X, y)

        assert result is selector
        assert hasattr(selector, "selected_features_")
        assert isinstance(selector.selected_features_, list)
        assert len(selector.selected_features_) <= 10
        assert all(feat in X.columns for feat in selector.selected_features_)

    def test_fit_more_features_than_available(self, small_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit method when n_features > number of available features."""
        X, y = small_rna_data
        selector = LogisticRegressionL1FeatureSelector(n_features=100)
        selector.fit(X, y)  # small_rna_data has only 50 features

        assert selector.selected_features_ == X.columns.tolist()

    def test_transform(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform method."""
        X, y = sample_rna_data
        selector = LogisticRegressionL1FeatureSelector(n_features=10)
        selector.fit(X, y)

        X_transformed = selector.transform(X)
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] <= 10
        assert list(X_transformed.columns) == selector.selected_features_

    def test_transform_unfitted(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform method with unfitted selector."""
        X, _ = sample_rna_data
        selector = LogisticRegressionL1FeatureSelector()
        with pytest.raises(ValueError, match="not fitted yet"):
            selector.transform(X)

    def test_get_feature_names_out(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_feature_names_out method."""
        X, y = sample_rna_data
        selector = LogisticRegressionL1FeatureSelector(n_features=10)
        selector.fit(X, y)

        feature_names = selector.get_feature_names_out()
        assert isinstance(feature_names, list)
        assert feature_names == selector.selected_features_

    def test_get_feature_names_out_unfitted(self) -> None:
        """Test get_feature_names_out method with unfitted selector."""
        selector = LogisticRegressionL1FeatureSelector()
        with pytest.raises(ValueError, match="not fitted yet"):
            selector.get_feature_names_out()


# Tests for ScanpyPreprocessor
class TestScanpyPreprocessor:
    def test_init(self) -> None:
        """Test initialization of ScanpyPreprocessor."""
        processor = ScanpyPreprocessor(min_genes=100, min_cells=5, target_sum=1e5, max_fraction=0.1, n_bins=15)
        assert processor.min_genes == 100
        assert processor.min_cells == 5
        assert processor.target_sum == float(1e5)
        assert processor.max_fraction == 0.1
        assert processor.n_bins == 15

    def test_prepare_anndata(self, small_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test _prepare_anndata method."""
        X, _ = small_rna_data
        processor = ScanpyPreprocessor()
        adata = processor._prepare_anndata(X)

        assert isinstance(adata, anndata.AnnData)
        assert adata.shape == X.shape
        np.testing.assert_array_equal(adata.X, X.values)

    def test_transform(self, small_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform method."""
        X, _ = small_rna_data
        processor = ScanpyPreprocessor()
        processor.fit(X)
        result = processor.transform(X)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape
        # RNA data should be transformed (normalized and log-transformed)
        assert not np.array_equal(result.values, X.values)

    def test_transform_different_sizes(
        self, small_rna_data: Tuple[pd.DataFrame, pd.Series], sample_rna_data: Tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test transform method with different sized data."""
        small_X, _ = small_rna_data
        large_X, _ = sample_rna_data

        processor = ScanpyPreprocessor()
        processor.fit(small_X)

        # Test with larger dataset (more rows but same columns)
        large_X_subset = large_X.iloc[:, : small_X.shape[1]]
        large_X_subset.columns = small_X.columns

        result = processor.transform(large_X_subset)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == large_X_subset.shape

        # Test with smaller dataset
        smaller_X = small_X.iloc[:10, :]
        result = processor.transform(smaller_X)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == smaller_X.shape

    def test_handles_sparse_counts(self) -> None:
        """Test that processor can handle sparse count matrices typical in RNA-seq."""
        # Create sparse RNA-seq like data (many zeros)
        rng = np.random.default_rng(42)
        sparse_counts = np.zeros((20, 30))
        # Add counts in random positions (10% of matrix)
        indices = rng.choice(sparse_counts.size, size=int(sparse_counts.size * 0.1), replace=False)
        rows, cols = np.unravel_index(indices, sparse_counts.shape)
        sparse_counts[rows, cols] = rng.negative_binomial(n=5, p=0.5, size=len(indices))

        sparse_df = pd.DataFrame(sparse_counts, columns=[f"gene_{i}" for i in range(30)])

        processor = ScanpyPreprocessor()
        processor.fit(sparse_df)
        result = processor.transform(sparse_df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sparse_df.shape


# Tests for RNA class
class TestRNA:
    def test_init(self) -> None:
        """Test initialization of RNA class."""
        rna = RNA(n_features=15, n_bins=25, normalisation_method="Scanpy")
        assert rna.n_features == 15
        assert rna.n_bins == 25
        assert rna.normalisation_method == "Scanpy"

    def test_init_invalid_normalisation_method(self) -> None:
        """Test initialization with invalid normalization method."""
        with pytest.raises(ValueError, match="Invalid normalization method"):
            RNA(normalisation_method="InvalidMethod")

    def test_build_pipeline(self) -> None:
        """Test _build_pipeline method."""
        rna = RNA()
        pipeline = rna._build_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 3
        assert pipeline.steps[0][0] == "normalisation"
        assert pipeline.steps[1][0] == "lasso_feature_selection"
        assert pipeline.steps[2][0] == "discretisation"

    def test_fit(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit method."""
        X, y = sample_rna_data
        rna = RNA(n_features=10)
        result = rna.fit(X, y)

        assert result is rna
        assert rna.pipeline is not None

    def test_transform(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform method."""
        X, y = sample_rna_data
        rna = RNA(n_features=10)
        rna.fit(X, y)
        result = rna.transform(X)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == X.shape[0]
        assert result.shape[1] <= 10  # Should have at most n_features columns
        # Check that the result has been discretized (should contain integers)
        assert np.all(result.dtypes == "int64") or np.all(result.dtypes == "int32")
        assert all(pd.api.types.is_integer_dtype(result[col]) for col in result.columns)

    def test_transform_unfitted(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform method with unfitted RNA."""
        X, _ = sample_rna_data
        rna = RNA()
        with pytest.raises(ValueError, match="not fitted yet"):
            rna.transform(X)

    def test_get_params(self) -> None:
        """Test get_params method."""
        rna = RNA(n_features=15, n_bins=25, normalisation_method="UQ")
        params = rna.get_params()

        assert isinstance(params, dict)
        assert params["n_features"] == 15
        assert params["n_bins"] == 25
        assert params["normalisation_method"] == "UQ"

    def test_set_params(self) -> None:
        """Test set_params method."""
        rna = RNA()
        result = rna.set_params(n_features=15, n_bins=25, normalisation_method="UQ")

        assert result is rna
        assert rna.n_features == 15
        assert rna.n_bins == 25
        assert rna.normalisation_method == "UQ"
        assert rna.pipeline is None  # Should reset pipeline

    def test_set_params_invalid_normalisation_method(self) -> None:
        """Test set_params with invalid normalization method."""
        rna = RNA()
        with pytest.raises(ValueError, match="Invalid normalization method"):
            rna.set_params(normalisation_method="InvalidMethod")

    @pytest.mark.parametrize(
        "norm_method",
        [
            "Scanpy",
            pytest.param(
                "UQ",
                marks=pytest.mark.xfail(
                    reason="rnanorm<=2.2.0 calls deprecated sklearn._validate_data removed in sklearn>=1.8",
                    strict=True,
                ),
            ),
            pytest.param(
                "CUF",
                marks=pytest.mark.xfail(
                    reason="rnanorm<=2.2.0 calls deprecated sklearn._validate_data removed in sklearn>=1.8",
                    strict=True,
                ),
            ),
            "CPM",
        ],
    )
    def test_different_normalisation_methods(
        self, small_rna_data: Tuple[pd.DataFrame, pd.Series], norm_method: str
    ) -> None:
        """Test RNA with different normalization methods."""
        X, y = small_rna_data
        rna = RNA(normalisation_method=norm_method, n_features=5)
        rna.fit(X, y)
        result = rna.transform(X)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == X.shape[0]

    def test_full_pipeline_workflow(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test the full RNA pipeline workflow."""
        X, y = sample_rna_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        rna = RNA(n_features=10, n_bins=5)
        rna.fit(X_train, y_train)

        # Transform training data
        X_train_transformed = rna.transform(X_train)
        assert isinstance(X_train_transformed, pd.DataFrame)
        assert X_train_transformed.shape[0] == X_train.shape[0]
        assert X_train_transformed.shape[1] <= 10

        # Transform test data
        X_test_transformed = rna.transform(X_test)
        assert isinstance(X_test_transformed, pd.DataFrame)
        assert X_test_transformed.shape[0] == X_test.shape[0]
        assert X_test_transformed.shape[1] <= 10

        # Check that train and test have the same columns
        assert list(X_train_transformed.columns) == list(X_test_transformed.columns)

        # Check that the result has been discretized (should contain integers)
        assert np.all(X_train_transformed.dtypes == "int64") or np.all(X_train_transformed.dtypes == "int32")
        assert np.all(X_test_transformed.dtypes == "int64") or np.all(X_test_transformed.dtypes == "int32")

    def test_handles_highly_expressed_genes(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that the pipeline can handle datasets with highly expressed genes."""
        X, y = sample_rna_data

        # Add some highly expressed genes (10x higher expression)
        rng = np.random.default_rng(42)
        high_expr_cols = rng.choice(X.columns, size=5, replace=False)
        X_modified = X.copy()
        X_modified[high_expr_cols] = X_modified[high_expr_cols] * 10

        rna = RNA(n_features=10)
        rna.fit(X_modified, y)
        result = rna.transform(X_modified)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == X_modified.shape[0]
        assert result.shape[1] <= 10

    def test_handles_zero_count_genes(self, sample_rna_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that the pipeline can handle genes with zero counts."""
        X, y = sample_rna_data

        # Add some genes with zero counts
        rng = np.random.default_rng(42)
        zero_cols = rng.choice(X.columns, size=10, replace=False)
        X_modified = X.copy()
        X_modified[zero_cols] = 0

        rna = RNA(n_features=10)
        rna.fit(X_modified, y)
        result = rna.transform(X_modified)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == X_modified.shape[0]
        assert result.shape[1] <= 10
