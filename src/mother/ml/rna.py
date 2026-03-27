import logging
from typing import Any, Dict, List, Optional, Union

try:
    import anndata
    import scanpy as sc
    from rnanorm import CPM, CUF, UQ
except ImportError as import_error:
    from mother.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rna", import_error)
import numpy as np
import pandas as pd
import sklearn.pipeline as skl_pipe
from feature_engine.discretisation import GeometricWidthDiscretiser
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Set up module-level logger
module_logger = logging.getLogger(__name__)


class LogisticRegressionL1FeatureSelector(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    Feature selection using Lasso with cross-validation.

    Parameters
    ----------
    n_features : int, default=10
        Number of features to select based on importance scores.
    cv : int, default=5
        Number of cross-validation folds for LogisticRegressionCV.
    random_state : int, default=42
        Random state for reproducibility.
    """

    def __init__(self, n_features: int = 10, cv: int = 5, random_state: int = 42) -> None:
        """
        Initialize the LogisticRegressionL1FeatureSelector.

        Parameters
        ----------
        n_features : int
            Number of features to select.
        cv : int
            Number of cross-validation folds.
        random_state : int
            Random state for reproducibility.
        """
        self.n_features: int = n_features
        self.cv: int = cv
        self.random_state: int = random_state
        self.selected_features_: Optional[List[str]] = None
        self.feature_selector: Optional[SelectFromModel] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionL1FeatureSelector":
        """
        Fit the model and select features based on Lasso coefficients.

        Parameters
        ----------
        X : DataFrame
            Input features.
        y : Series
            Target labels.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.n_features is None:  # If n_features is None, just take all features with non-zero importance
            threshold: float = np.finfo(float).eps
            max_features: Optional[int] = None
        elif (
            self.n_features >= X.shape[1] and self.n_features is not None
        ):  # Handle case where we want more features than available
            module_logger.warning(
                f"Number of features requested ({self.n_features}) is >= available features: ({X.shape[1]})."
            )
            module_logger.warning("Returning all features.")
            self.selected_features_ = X.columns.tolist()
            return self
        else:  # If n_features is not None, take only n top features
            threshold = -np.inf
            max_features = self.n_features

        # Apply MinMaxScaler to scale the data between 0 and 1
        scaler: MinMaxScaler = MinMaxScaler()
        X_scaled: np.ndarray = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Create feature selector with LogisticRegressionCV
        self.feature_selector = SelectFromModel(
            LogisticRegressionCV(
                cv=self.cv,
                random_state=self.random_state,
                penalty="l1",
                class_weight="balanced",
                solver="liblinear",
            ),
            threshold=threshold,  # Use n_features instead of threshold
            max_features=max_features,
        )

        # Encode y if it is of type object
        if y is not None and y.dtype == "object":
            le: LabelEncoder = LabelEncoder()
            y = le.fit_transform(y)

        # Fit the selector
        self.feature_selector.fit(X, y)

        # Get selected feature names
        feature_mask: np.ndarray = self.feature_selector.get_support()
        self.selected_features_ = X.columns[feature_mask].tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data to select the features identified by the model.

        Parameters
        ----------
        X : DataFrame
            Input features.

        Returns
        -------
        X_transformed : DataFrame
            Transformed data with only the selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("This LogisticRegressionL1FeatureSelector instance is not fitted yet.")
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get the names of the selected features.

        Parameters
        ----------
        input_features : array-like, default=None
            Input feature names for compatibility.

        Returns
        -------
        selected_features : list
            List of selected feature names.
        """
        if self.selected_features_ is None:
            raise ValueError("This LogisticRegressionL1FeatureSelector instance is not fitted yet.")
        return self.selected_features_


class ScanpyPreprocessor(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    ScanpyPreprocessor performs preprocessing steps for single-cell RNA sequencing data using Scanpy.

    Parameters
    ----------
    min_genes : int, default=200
        Minimum number of genes per cell.
    min_cells : int, default=3
        Minimum number of cells per gene.
    target_sum : float, default=1e4
        Target sum for normalization.
    max_fraction : float, default=0.2
        Maximum fraction of counts to exclude.
    n_bins : int, default=10
        Number of bins for scaling.
    random_state : int, default=42
        Random state for reproducibility.
    """

    def __init__(
        self,
        min_genes: int = 200,
        min_cells: int = 3,
        target_sum: float = 1e4,
        max_fraction: float = 0.2,
        n_bins: int = 10,
    ) -> None:
        """
        Initialize class
        """
        self.min_genes: int = min_genes
        self.min_cells: int = min_cells
        self.target_sum: float = target_sum
        self.max_fraction: float = max_fraction
        self.n_bins: int = n_bins
        self.normalisation_factors: Optional[np.ndarray] = None

    def _prepare_anndata(self, X: pd.DataFrame) -> anndata.AnnData:
        """
        Prepare AnnData object from DataFrame.

        Parameters
        ----------
        X : DataFrame
            Input features.

        Returns
        -------
        adata : AnnData
            AnnData object.
        """
        X = X.fillna(X.median())
        if not isinstance(X.index, pd.RangeIndex):
            X.index = X.index.astype(str)
        return anndata.AnnData(X)

    def fit(
        self,
        X: Union[pd.DataFrame, anndata.AnnData],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "ScanpyPreprocessor":
        """
        Fit the ScanpyPreprocessor to the data.

        Parameters
        ----------
        X : DataFrame or AnnData
            Input features.
        y : Series or array-like, default=None
            Target labels (not used).

        Returns
        -------
        X_transformed : DataFrame
            Transformed data.
        """
        return self

    def transform(self, X: Union[pd.DataFrame, anndata.AnnData]) -> pd.DataFrame:
        """
        Transform the data using the fitted ScanpyPreprocessor.

        Parameters
        ----------
        X : DataFrame or AnnData
            Input features.

        Returns
        -------
        X_transformed : DataFrame
            Transformed data.
        """
        # Convert to AnnData if necessary
        if isinstance(X, pd.DataFrame):
            adata: anndata.AnnData = self._prepare_anndata(X)
        else:
            adata = X.copy()

        # Apply normalization factors
        sc.pp.normalize_total(
            adata,
            target_sum=self.target_sum,
            exclude_highly_expressed=True,
            inplace=True,
        )

        # Find the minimum value in the array and add offset to ensure positivity
        min_value: float = np.min(adata.X)
        if min_value < 0:
            adata.X = adata.X + abs(min_value) + 1

        # Log transform
        sc.pp.log1p(adata)

        # Convert back to DataFrame
        X_transformed: pd.DataFrame = adata.to_df()
        X_transformed = X_transformed.clip(upper=1e6)
        X_transformed.index = (
            X.index if hasattr(X, "index") else pd.RangeIndex(start=0, stop=X_transformed.shape[0], step=1)
        )

        return X_transformed


class RNA(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    RNA class to build a processing pipeline for RNA sequencing data.

    Parameters
    ----------
    n_features : int, default=20
        Number of features to select.
    n_bins : int, default=20
        Number of bins for discretization.
    normalisation_method : str, default="Scanpy"
        The normalization method to use. Must be one of "CUF", "UQ", "CPM", or "Scanpy".
    random_state : int, default=42
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_features: int = 20,
        n_bins: int = 20,
        normalisation_method: str = "Scanpy",
        random_state: int = 42,
    ) -> None:
        """
        Initialize class
        """
        super().__init__()
        self.n_features: int = n_features
        self.n_bins: int = n_bins
        self.normalisation_method: str = normalisation_method
        self.random_state: int = random_state
        self.pipeline: Optional[skl_pipe.Pipeline] = None

        # Define available normalization methods
        self.normalisation_methods_dict: Dict[str, Union[ScanpyPreprocessor, UQ, CUF, CPM]] = {
            "Scanpy": ScanpyPreprocessor(),
            "UQ": UQ().set_output(transform="pandas"),  # Upper quartile normalisation
            "CUF": CUF().set_output(transform="pandas"),  # Counts adjusted with Upper quartile factors normalization.
            "CPM": CPM().set_output(transform="pandas"),  # Counts per million normalization.
        }

        if normalisation_method not in self.normalisation_methods_dict:
            raise ValueError(
                f"Invalid normalization method: {normalisation_method}. "
                f"Valid methods are: {', '.join(self.normalisation_methods_dict.keys())}."
            )

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> "RNA":
        """
        Fit the RNA processing pipeline to the data.

        Parameters
        ----------
        X : DataFrame
            Input features.
        y : Series or array-like, default=None
            Target labels.

        Returns
        -------
        self : object
            Returns self.
        """
        module_logger.info("Building and fitting RNA processing pipeline")
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        module_logger.info("Finished fitting RNA processing pipeline")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted RNA processing pipeline.

        Parameters
        ----------
        X : DataFrame
            Input features.

        Returns
        -------
        X_transformed : DataFrame
            Transformed data.
        """
        if self.pipeline is None:
            raise ValueError("This RNA instance is not fitted yet.")
        return self.pipeline.transform(X)

    def _build_pipeline(self) -> skl_pipe.Pipeline:
        """
        Build RNA processing pipeline consisting of filling NAs, count normalization,
        discretization, and feature selection.

        Returns
        -------
        pipeline : Pipeline
            The constructed sklearn Pipeline.
        """
        normalisation_object = self.normalisation_methods_dict[self.normalisation_method]

        pipeline = skl_pipe.Pipeline(
            [
                ("normalisation", normalisation_object),
                (
                    "lasso_feature_selection",
                    LogisticRegressionL1FeatureSelector(n_features=self.n_features, random_state=self.random_state),
                ),
                ("discretisation", GeometricWidthDiscretiser(bins=self.n_bins)),
            ]
        )

        return pipeline

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "n_features": self.n_features,
            "n_bins": self.n_bins,
            "normalisation_method": self.normalisation_method,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Dict[str, Any]) -> "RNA":
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)

        # Update normalization method if changed
        if "normalisation_method" in params:
            if params["normalisation_method"] not in self.normalisation_methods_dict:
                raise ValueError(
                    f"Invalid normalization method: {params['normalisation_method']}. "
                    f"Valid methods are: {', '.join(self.normalisation_methods_dict.keys())}."
                )

        # Reset pipeline to force rebuilding with new parameters
        self.pipeline = None

        return self
