# Neural Oblivious Decision Ensembles
# Author: Sergey Popov, Julian Qian
# https://github.com/Qwicen/node
# For license information, see https://github.com/Qwicen/node/blob/master/LICENSE.md
"""
Neural Oblivious Decision Ensembles (NODE)

This module implements NODE, a neural network architecture for tabular data that combines
the interpretability and efficiency of decision tree ensembles with the flexibility and
power of deep learning. NODE uses differentiable oblivious decision trees that can be
trained end-to-end using gradient descent.

Key Features:
- Supports both classification and regression tasks on tabular data
- Utilizes differentiable oblivious decision trees for end-to-end training
- Sparse activation functions (entmax15, sparsemax, sparsemoid) for interpretability
- Compatible with scikit-learn API through skorch wrappers
- Automatic input/output dimension detection via InputOutputShapeSetter callback
- Support for mixed data types (continuous and categorical features)
- Multiple head architectures: subset, linear, MLP, and flow (probabilistic)
- Flow head implements NodeFlow architecture (Wielopolski, Furman & Zięba, 2024)
- Hyperparameter optimization integration with Optuna via MotherTuner

Architecture Overview:
    Input Data → Embedding Layer → Dense ODST Blocks → Head Layer → Predictions

    1. Embedding Layer: Preprocesses features (normalization, categorical embeddings)
    2. Dense ODST Blocks: Core NODE computation with oblivious decision trees
    3. Head Layer: Converts tree outputs to final predictions (subset/linear/mlp/flow)

Flow Head (NodeFlow Architecture):
    NODE Embeddings → (Optional Tanh MLP + Dropout) → Conditional Normalizing Flow

    The flow head implements the NodeFlow architecture, which combines NODE with
    conditional normalizing flows for probabilistic regression. This provides:
    - Flexible uncertainty quantification
    - Non-parametric density estimation
    - Multiple flow architectures (GMM, NICE, RealNVP, NAF, UNAF, NSF, BPF) via Zuko library

Usage Examples:

    # Basic Classification
    from mother.ml.models.m_node import NODEClassifier

    clf = NODEClassifier(
        num_trees=2048,
        depth=6,
        num_layers=1,
        max_epochs=100,
        lr=0.01,
        device='cpu'
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Regression with MLP Head
    from mother.ml.models.m_node import NODERegressor

    reg = NODERegressor(
        head_type='mlp',
        mlp_hidden_dims=[256, 128],
        num_trees=2048,
        max_epochs=100,
        lr=0.01
    )
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    # Probabilistic Regression with Flow Head (NodeFlow)
    from mother.ml.models.m_node import NODERegressor

    reg = NODERegressor(
        head_type='flow',
        flow_type='NSF',  # Neural Spline Flow
        num_trees=2048,
        max_epochs=100,
        lr=0.01
    )
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)  # Point predictions
    samples = reg.predict_flow(X_test, num_samples=1000)  # Uncertainty samples

References:
    Popov, S., Morozov, S., & Babenko, A. (2019).
    Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data.
    arXiv:1909.06312.

    Wielopolski, P., Furman, O., & Zięba, M. (2024).
    NodeFlow: Towards End-to-end Flexible Probabilistic Regression on Tabular Data.
    Entropy, 26(7), 593.
"""

import logging
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import skorch
import torch
import torch.nn as nn
from optuna import Trial
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.net import NeuralNet
from torch import Tensor

from mother.ml.core import AbstractMotherPipeline
from mother.ml.models.m_head_utils import compute_flow_mode_and_uncertainty
from mother.ml.models.m_heads import FlowHead, MLPHead
from mother.ml.models.m_node_utils import (
    DenseODSTBlock,
    Embedding1dLayer,
    Lambda,
    entmax15,
    entmoid15,
    sparsemax,
    sparsemoid,
)

# Setup module logger
module_logger = logging.getLogger(__name__)

# Standardized quantile defaults matching Mother convention (TabPFN, RandomForest, etc.).
# Kept as a module-level constant because it is used as a default argument value in
# method signatures (e.g. ``predict_uncertainty(quantiles=DEFAULT_QUANTILES)``),
# where ``self`` is not yet available.  Shared across NODERegressor and NODEClassifier.
DEFAULT_QUANTILES: list[float] = [0.25, 0.5, 0.75]

# Fixed early-stopping patience used when a validation split is active.
# Defined at module level rather than as an ``__init__`` parameter because it is
# a project-wide convention, not a per-instance tunable.
_EARLY_STOPPING_PATIENCE: int = 20


# ==============================================================================
# MODULE-LEVEL HELPERS
# ==============================================================================


def _prepare_for_dataframe(
    arr: Optional[npt.NDArray[Any]],
) -> Optional[Union[npt.NDArray[Any], List[npt.NDArray[Any]]]]:
    """Reshape an array for insertion into a ``pd.DataFrame`` column.

    - ``None`` → ``None``
    - 1-D array → pass through
    - 2-D with single column → flatten to 1-D
    - 2-D multi-column → list of row vectors (one per cell)

    This avoids the ``ValueError: Must have equal len keys and value``
    that pandas raises when assigning a 2-D array to a single column.
    """
    if arr is None:
        return None
    if not hasattr(arr, "shape"):
        return arr
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr.flatten()
    # Multi-target: each row becomes a list entry
    return [row for row in arr]


# ==============================================================================
# SKORCH CALLBACKS FOR AUTO DIMENSION DETECTION
# ==============================================================================


def _is_string_or_object_dtype(series: pd.Series) -> bool:
    """Check if a pandas Series has string or object dtype.

    Handles both legacy ``object`` dtype and the ``StringDtype`` introduced
    as default for string columns in pandas 3.0.
    """
    return series.dtype == "object" or pd.api.types.is_string_dtype(series)


class InputOutputShapeSetter(skorch.callbacks.Callback):
    """
    Callback to auto-detect input/output dimensions and handle categorical features.

    Handles:
    - Splitting features into categorical vs continuous based on the explicitly
      declared ``categorical_columns`` (mirroring CatBoost's ``cat_features``).
      No automatic detection is performed.
    - Setting input_dim and output_dim based on training data
    - Label encoding and embedding setup for categorical features

    Categorical features must be declared explicitly. Any non-numeric column
    that is not declared categorical raises an error.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        max_embedding_dim: int = 16,
        min_embedding_dim: int = 2,
    ) -> None:
        self.categorical_columns = categorical_columns
        self.max_embedding_dim = max_embedding_dim
        self.min_embedding_dim = min_embedding_dim

        # Will be set during training
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.continuous_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.categorical_embedding_dims_: List[Tuple[int, int]] = []
        self.feature_names_: List[str] = []

    def _detect_feature_types(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> Tuple[List[str], List[str]]:
        """Split features into continuous vs categorical based on explicit declaration.

        Categorical columns must be declared explicitly via ``categorical_columns``
        (mirroring CatBoost's ``cat_features``). No automatic detection is
        performed: any non-numeric column that is not declared categorical raises
        an error.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)

            # Only explicitly declared columns are categorical. Everything else is
            # treated as continuous (no auto-detection).
            if self.categorical_columns is not None:
                categorical_cols = [col for col in self.categorical_columns if col in X.columns]
            else:
                categorical_cols = []
            continuous_cols = [col for col in X.columns if col not in categorical_cols]

            # CRITICAL VALIDATION: continuous columns must be numeric. Non-numeric
            # columns (string/object or 'category' dtype) must be declared
            # categorical — NODE never auto-detects them.
            for col in continuous_cols:
                if _is_string_or_object_dtype(X[col]) or isinstance(X[col].dtype, pd.CategoricalDtype):
                    raise ValueError(
                        f"Column '{col}' has a non-numeric dtype ({X[col].dtype}) but is not "
                        f"declared categorical. NODE does not auto-detect categorical features. "
                        f"Please either: 1) List '{col}' in the 'cat_features' parameter "
                        f"(e.g. NODERegressor(cat_features=['{col}', ...])), or "
                        f"2) Convert '{col}' to numeric dtype before passing to NODE."
                    )

            return continuous_cols, categorical_cols
        else:
            # For numpy arrays, treat all as continuous (backward compatibility)
            n_features: int = X.shape[1] if hasattr(X, "shape") else len(X[0])
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]
            return list(range(n_features)), []

    def _calculate_embedding_dim(self, n_categories: int) -> int:
        """Calculate appropriate embedding dimension for categorical feature."""
        embedding_dim: int = int(n_categories**0.6)  # Common heuristic: n^0.6
        return max(self.min_embedding_dim, min(self.max_embedding_dim, embedding_dim))

    def _setup_categorical_encoders(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> None:
        """Set up label encoders for categorical features."""
        if isinstance(X, pd.DataFrame) and self.categorical_columns_:
            for col in self.categorical_columns_:
                # Create and fit label encoder
                le: LabelEncoder = LabelEncoder()
                le.fit(X[col].astype(str))  # Convert to string to handle mixed types
                self.label_encoders_[col] = le

                # Calculate embedding dimension
                n_categories: int = len(le.classes_)
                embedding_dim: int = self._calculate_embedding_dim(n_categories)
                self.categorical_embedding_dims_.append((n_categories, embedding_dim))

    def _prepare_data_for_node(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Convert DataFrame input to numpy array for NODE (NO dictionaries!)."""
        if isinstance(X, pd.DataFrame):
            # Create a copy to avoid modifying original
            X_processed: pd.DataFrame = X.copy()

            # Encode categorical columns (both object/string and category dtypes)
            for col in X_processed.columns:
                # Check if it's a categorical column (object/string dtype or category dtype)
                is_object = _is_string_or_object_dtype(X_processed[col])
                is_category = isinstance(X_processed[col].dtype, pd.CategoricalDtype)

                if is_object or is_category:
                    # If it's a designated categorical column, use proper label encoder
                    if col in self.categorical_columns_ and col in self.label_encoders_:
                        try:
                            X_processed[col] = self.label_encoders_[col].transform(X_processed[col].astype(str))
                        except ValueError:
                            # Handle unseen categories by assigning to first category
                            encoded: npt.NDArray[np.int_] = np.zeros(len(X_processed), dtype=int)
                            known_mask: pd.Series = (
                                X_processed[col].astype(str).isin(self.label_encoders_[col].classes_)
                            )
                            if known_mask.any():
                                encoded[known_mask] = self.label_encoders_[col].transform(
                                    X_processed[col].astype(str)[known_mask]
                                )
                            X_processed[col] = encoded
                    else:
                        # For non-categorical object/category columns, create a temporary encoder
                        temp_le: LabelEncoder = LabelEncoder()
                        X_processed[col] = temp_le.fit_transform(X_processed[col].astype(str))

            # Return as numpy array - NO dictionaries!
            return X_processed.values.astype(np.float32)
        else:
            # Already numpy array
            return np.asarray(X, dtype=np.float32)

    def on_train_begin(
        self,
        net: NeuralNet,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        y: Union[pd.Series, npt.NDArray[Any]],
    ) -> None:
        """Enhanced dimension detection with categorical feature support."""
        # Use original DataFrame if available, otherwise use X
        original_X: Union[pd.DataFrame, npt.NDArray[np.float32]] = getattr(net, "_original_X_train", X)

        # === FEATURE TYPE DETECTION ===
        self.continuous_columns_, self.categorical_columns_ = self._detect_feature_types(original_X)

        # === CATEGORICAL FEATURE SETUP ===
        if self.categorical_columns_:
            self._setup_categorical_encoders(original_X)

        # === INPUT DIMENSION DETECTION ===
        # Total input dimension is all features combined
        if hasattr(original_X, "shape") and len(original_X.shape) >= 2:
            input_dim: int = original_X.shape[1]  # Number of features
        elif hasattr(original_X, "columns"):  # DataFrame
            input_dim: int = len(original_X.columns)
        else:
            input_dim: int = 1  # Fallback for edge cases

        # === OUTPUT DIMENSION DETECTION ===
        # Determine output dimension based on the target data
        if hasattr(y, "ndim") and y.ndim > 1 and y.shape[1] > 1:
            # Multi-dimensional output (regression with multiple targets OR multi-label classification)
            output_dim: int = y.shape[1]
            target_type: str = "multi_target"
        elif "classifier" in str(type(net)).lower():
            # Classification: count unique classes
            if hasattr(y, "numpy"):
                y_array: npt.NDArray[Any] = y.numpy()
            else:
                y_array: npt.NDArray[Any] = np.asarray(y)
            output_dim: int = len(np.unique(y_array))
            target_type: str = "single_target"
        else:
            # Single-dimensional output (single regression target)
            output_dim: int = 1
            target_type: str = "single_target"

        # === STORE PREPROCESSING INFO IN NETWORK ===
        # This allows the network to handle DataFrames during prediction
        net.categorical_columns_ = self.categorical_columns_
        net.continuous_columns_ = self.continuous_columns_
        net.label_encoders_ = self.label_encoders_
        net.categorical_embedding_dims_ = self.categorical_embedding_dims_
        net.feature_names_ = self.feature_names_
        net.target_type_ = target_type
        net._prepare_data_for_node = self._prepare_data_for_node

        # === UPDATE MODULE PARAMETERS ===
        # Only update parameters that have actually changed to avoid unnecessary re-initialization.
        # Data-detected dimensions take precedence over user-specified values.
        update_params: Dict[str, Any] = {}

        # Get current parameters
        current_params: Dict[str, Any] = net.get_params()

        # Only add parameters that have changed
        if current_params.get("module__input_dim") != input_dim:
            update_params["module__input_dim"] = input_dim
        if current_params.get("module__output_dim") != output_dim:
            update_params["module__output_dim"] = output_dim

        # CRITICAL: If we're updating dimensions, preserve head_type to avoid it resetting to default
        # This ensures flow heads and other non-default heads work correctly after re-initialization
        if update_params and hasattr(net, "head_type"):
            existing_head_type = net.head_type
            if current_params.get("module__head_type") != existing_head_type:
                update_params["module__head_type"] = existing_head_type

        # Only update if there are actual changes
        if update_params:
            net.set_params(**update_params)

        # === LOGGING ===
        module_logger.info("InputOutputShapeSetter:")
        module_logger.info(f"  - Total input features: {input_dim}")
        module_logger.info(f"  - Output dimension: {output_dim}")
        module_logger.info(f"  - Target type: {target_type}")
        if self.continuous_columns_:
            module_logger.info(f"  - Continuous features ({len(self.continuous_columns_)}): {self.continuous_columns_}")
        if self.categorical_columns_:
            module_logger.info(
                f"  - Categorical features ({len(self.categorical_columns_)}): {self.categorical_columns_}"
            )
            module_logger.info(f"  - Categorical embeddings: {self.categorical_embedding_dims_}")


class LossFunctionSetter(skorch.callbacks.Callback):
    """
    Callback to auto-set appropriate loss function based on task type.

    Sets CrossEntropyLoss for classification (BCEWithLogitsLoss for multi-label)
    and MSELoss for regression. Only overrides if not explicitly provided by the user.
    """

    def on_train_begin(
        self,
        net: Union["NODEClassifier", "NODERegressor"],
        X: Union[pd.DataFrame, npt.NDArray[np.float32], None] = None,
        y: Union[pd.Series, npt.NDArray[Any], None] = None,
        **kwargs: Any,
    ) -> None:
        """Set appropriate loss function if not explicitly provided.

        Delegates to ``net._set_loss(y)`` which is implemented separately
        by ``NODEClassifier`` and ``NODERegressor``.
        """
        if hasattr(net, "_user_provided_criterion"):
            return

        net._set_loss(y)


# ==============================================================================
# NODE BACKBONE - DEPENDS ON DENSE ODST BLOCK AND EMBEDDING
# ==============================================================================


class NODEBackbone(nn.Module):
    """NODE backbone: assembles Embedding + Dense ODST Block from a config namespace.

    This class is used internally by ``NODEModel`` (PyTorch Tabular style) and
    constructs the core computation graph:  Dense ODST layers with sparse
    activation functions.  Embedding is built lazily via ``_build_embedding_layer``.

    Args:
        config: Namespace / object with attributes:
            ``continuous_dim``, ``embedded_cat_dim``, ``embedding_dims``,
            ``embedding_dropout``, ``batch_norm_continuous_input``,
            ``num_trees``, ``num_layers``, ``output_dim``,
            ``additional_tree_output_dim``, ``max_features``,
            ``input_dropout``, ``depth``, ``choice_function``,
            ``bin_function``, ``initialize_response``,
            ``initialize_selection_logits``, ``threshold_init_beta``,
            ``threshold_init_cutoff``.
    """

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__()
        self.hparams = config

        self.hparams.node_input_dim = (self.hparams.continuous_dim or 0) + (self.hparams.embedded_cat_dim or 0)

        # Map function names to actual functions
        if self.hparams.choice_function == "sparsemax":
            choice_func = sparsemax
        else:
            choice_func = entmax15

        if self.hparams.bin_function == "sparsemoid":
            bin_func = sparsemoid
        else:
            bin_func = entmoid15

        self.dense_block = DenseODSTBlock(
            input_dim=self.hparams.node_input_dim,
            num_trees=self.hparams.num_trees,
            num_layers=self.hparams.num_layers,
            tree_output_dim=self.hparams.output_dim + self.hparams.additional_tree_output_dim,
            max_features=self.hparams.max_features,
            input_dropout=self.hparams.input_dropout,
            depth=self.hparams.depth,
            choice_function=choice_func,
            bin_function=bin_func,
            initialize_response_=getattr(nn.init, self.hparams.initialize_response + "_"),
            initialize_selection_logits_=getattr(nn.init, self.hparams.initialize_selection_logits + "_"),
            threshold_init_beta=self.hparams.threshold_init_beta,
            threshold_init_cutoff=self.hparams.threshold_init_cutoff,
        )
        self.output_dim = self.hparams.output_dim + self.hparams.additional_tree_output_dim

    def _build_embedding_layer(self) -> Embedding1dLayer:
        """Create the embedding layer for continuous + categorical features."""
        return Embedding1dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_embedding_dims=self.hparams.embedding_dims,
            embedding_dropout=self.hparams.embedding_dropout,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass features through the Dense ODST Block.

        Args:
            x: Embedded features ``[batch_size, node_input_dim]``.

        Returns:
            Tree outputs ``[batch_size, num_layers * num_trees, tree_output_dim]``.
        """
        x = self.dense_block(x)
        return x


# Note: MLPHead and FlowHead are imported from standalone modules.


class LinearHead(nn.Module):
    """Single linear projection from flattened tree outputs to ``output_dim``.

    Args:
        input_dim: Expected flattened dimension ``num_layers * num_trees * tree_output_dim``.
        output_dim: Target prediction dimension.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten ``[batch, trees, dim]`` → ``[batch, trees*dim]`` and project."""
        return self.net(x.reshape(x.shape[0], -1))


class NODEModel(nn.Module):
    """Full NODE model following PyTorch Tabular conventions.

    Composes ``embedding_layer → backbone (Dense ODST) → head`` and
    supports data-aware initialization of ODST thresholds.

    Note:
        This class is **not** used by the sklearn-compatible wrappers
        (``NODEClassifier`` / ``NODERegressor``).  Those use
        ``CompletePyTorchTabularNODE`` directly.  This class exists for
        PyTorch Tabular integration.
    """

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__()
        self.hparams = config
        self._build_network()

    def data_aware_initialization(self, datamodule: Any) -> None:
        """Performs data-aware initialization for NODE."""
        module_logger.info(
            "Data Aware Initialization of NODE using a forward pass with "
            f"{self.hparams.data_aware_init_batch_size} batch size...."
        )
        # Need a big batch to initialize properly
        alt_loader = datamodule.train_dataloader(batch_size=self.hparams.data_aware_init_batch_size)
        batch = next(iter(alt_loader))
        for k, v in batch.items():
            if isinstance(v, list) and (len(v) == 0):
                continue
            if not isinstance(v, list) and hasattr(v, "to"):
                batch[k] = v.to(self.device)

        # single forward pass to initialize the ODST
        with torch.no_grad():
            self(batch)

    @property
    def backbone(self) -> NODEBackbone:
        return self._backbone

    @property
    def embedding_layer(self) -> Embedding1dLayer:
        return self._embedding_layer

    @property
    def head(self) -> nn.Module:
        return self._head

    def _build_network(self) -> None:
        self._backbone = NODEBackbone(self.hparams)
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Build the appropriate head based on head_type
        self._head = self._build_head()

    def _build_head(self) -> nn.Module:
        """Build the appropriate head based on head_type configuration."""
        head_type = getattr(self.hparams, "head_type", "subset")

        # Calculate head input dimension
        # Tree outputs are: [batch_size, num_layers, num_trees, total_output_dim]
        # Head input is the flattened tree outputs: num_layers * num_trees * total_output_dim
        head_input_dim = self.hparams.num_layers * self.hparams.num_trees * self.hparams.total_output_dim
        head_output_dim = self.hparams.output_dim

        if head_type == "subset":
            # Original NODE behavior - subset and mean
            return Lambda(self.subset)
        elif head_type == "linear":
            # Linear head (tree dropout applied before head)
            return LinearHead(head_input_dim, head_output_dim)
        elif head_type == "mlp":
            # MLP head with adaptive architecture
            mlp_hidden_dims = getattr(self.hparams, "mlp_hidden_dims", None)
            mlp_dropout = getattr(self.hparams, "mlp_dropout", 0)

            # If None, create adaptive funnel architecture derived from first hidden layer
            # First layer size adapts to NODE output, subsequent layers form a funnel
            # Architecture: [first_hidden, first_hidden//2, first_hidden//4]
            if mlp_hidden_dims is None:
                first_hidden = max(128, head_input_dim // 4)  # 25% of NODE output
                mlp_hidden_dims = [
                    first_hidden,  # First hidden layer (tunable)
                    first_hidden // 2,  # Second layer: 50% of first
                    first_hidden // 4,  # Third layer: 25% of first
                ]

            return MLPHead(head_input_dim, head_output_dim, mlp_hidden_dims, dropout=mlp_dropout)
        elif head_type == "flow":
            # Flow head for probabilistic regression (tree dropout applied before head)
            flow_type = getattr(self.hparams, "flow_type", "NICE")
            flow_transforms = getattr(self.hparams, "flow_transforms", 3)
            flow_bins = getattr(self.hparams, "flow_bins", 8)
            flow_degree = getattr(self.hparams, "flow_degree", 16)
            flow_signal = getattr(self.hparams, "flow_signal", 16)
            flow_components = getattr(self.hparams, "flow_components", 8)
            return FlowHead(
                head_input_dim,
                head_output_dim,
                flow_type=flow_type,
                flow_transforms=flow_transforms,
                flow_bins=flow_bins,
                flow_degree=flow_degree,
                flow_signal=flow_signal,
                flow_components=flow_components,
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def subset(self, x: torch.Tensor) -> torch.Tensor:
        """Subset head: slice first ``output_dim`` dims and average across trees."""
        return x[..., : self.hparams.output_dim].mean(dim=-2)

    def forward(self, x_dict: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """Embedding → backbone → head."""
        x = self.embedding_layer(x_dict)
        x = self.backbone(x)
        x = self.head(x)
        return x


class CompletePyTorchTabularNODE(nn.Module):
    r"""
    Complete NODE module: Embedding → Dense ODST Blocks → Head → Output.

    This is the ``nn.Module`` instantiated by ``NODEClassifier`` and
    ``NODERegressor`` via Skorch.  It combines differentiable oblivious
    decision trees with sparse activations for deep learning on tabular data.

    Supports:
    - Continuous and categorical features (via embedding layer)
    - Multiple head types: ``subset``, ``linear``, ``mlp``, ``flow``
    - Classification and regression tasks
    - Tree-level dropout for regularisation

    Args:
        input_dim: Number of input features (``None`` until auto-detected).
        output_dim: Prediction dimension (classes for clf, targets for reg).
        num_layers: Number of stacked ODST layers with dense connections.
        num_trees: Number of oblivious decision trees per layer.
        additional_tree_output_dim: Extra per-tree output dimensions beyond
            ``output_dim``.  Acts as auxiliary capacity during training;
            only the ``subset`` head discards them at inference.
        depth: Tree depth — each tree has 2\ :sup:`depth` leaves.
        choice_function: Sparse feature selector (``"entmax15"`` or ``"sparsemax"``).
        bin_function: Soft binning function (``"entmoid15"`` or ``"sparsemoid"``).
        max_features: Cap on concatenated feature dim between layers.
        input_dropout: Dropout on features between ODST layers.
        head_type: Prediction head — ``"subset"``, ``"linear"``, ``"mlp"``, or ``"flow"``.
        mlp_hidden_dims: Hidden sizes for MLP head (``None`` = auto funnel).
        mlp_dropout: Dropout inside MLP head.
        mlp_activation: Activation for MLP head (``"ReLU"``, ``"GELU"``, ``"LeakyReLU"``).
        tree_dropout: Probability of dropping entire trees before the head.
        flow_type: Normalizing flow architecture
            (``"GMM"``, ``"NICE"``, ``"RealNVP"``, ``"NAF"``,
            ``"UNAF"``, ``"NSF"``, ``"BPF"``).
        flow_transforms: Number of flow transformation layers
            (NICE, RealNVP, NAF, UNAF).
        flow_bins: Number of spline bins for NSF.
        flow_degree: Polynomial degree for BPF (default 16).
        flow_signal: Hidden signal dimension for NAF/UNAF (default 16).
        flow_components: Number of mixture components for GMM (default 8).
    """

    def __init__(
        self,
        input_dim: Optional[int],
        output_dim: int,
        num_layers: int = 1,
        num_trees: int = 2048,
        additional_tree_output_dim: int = 3,
        depth: int = 6,
        choice_function: str = "entmax15",  # "entmax15" or "sparsemax"
        bin_function: str = "entmoid15",  # "entmoid15" or "sparsemoid"
        max_features: Optional[int] = None,
        input_dropout: float = 0.0,
        initialize_response: str = "normal",  # "normal" or "uniform"
        initialize_selection_logits: str = "uniform",  # "uniform" or "normal"
        threshold_init_beta: float = 1.0,
        threshold_init_cutoff: float = 1.0,
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
        head_type: str = "subset",  # "subset", "linear", "mlp", or "flow"
        mlp_hidden_dims: Optional[List[int]] = None,  # e.g. [512, 256]; None = auto funnel
        mlp_dropout: float = 0.1,  # only used when head_type="mlp"
        mlp_activation: str = "ReLU",  # "ReLU", "GELU", or "LeakyReLU"; only for head_type="mlp"
        tree_dropout: float = 0.0,  # drop entire trees before head (regularization)
        flow_type: str = "NICE",  # Flow architecture; only for head_type="flow"
        flow_transforms: int = 3,  # Transform layers (NICE, RealNVP, NAF, UNAF)
        flow_bins: int = 8,  # Spline bins (NSF)
        flow_degree: int = 16,  # Polynomial degree (BPF)
        flow_signal: int = 16,  # Hidden signal dim (NAF, UNAF)
        flow_components: int = 8,  # Mixture components (GMM)
    ) -> None:
        super().__init__()

        # Store configuration
        self.continuous_dim = input_dim
        self.embedded_cat_dim = 0
        self.embedding_dims = []
        self.embedding_dropout = embedding_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.output_dim = output_dim
        self.head_type = head_type
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout = mlp_dropout
        self.mlp_activation = mlp_activation
        self.tree_dropout = tree_dropout
        self.flow_type = flow_type
        self.flow_transforms = flow_transforms
        self.flow_bins = flow_bins
        self.flow_degree = flow_degree
        self.flow_signal = flow_signal
        self.flow_components = flow_components

        # ODST parameters
        self.additional_tree_output_dim = additional_tree_output_dim
        self.num_trees = num_trees
        self.num_layers = num_layers
        self.depth = depth
        self.max_features = max_features
        self.input_dropout = input_dropout
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.initialize_response = initialize_response
        self.initialize_selection_logits = initialize_selection_logits
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff

        # Total input dim for ODST (continuous + categorical embeddings)
        # input_dim/output_dim can be None initially — InputShapeSetter callback sets them
        self.node_input_dim = (self.continuous_dim or 0) + (self.embedded_cat_dim or 0)
        self._build_modules()

    def _build_modules(self) -> None:
        """Build the dense block, embedding layer, and head when input_dim is known."""
        # Map string names to functions
        choice_func: Callable[[Tensor, int], Tensor]
        if self.choice_function == "sparsemax":
            choice_func = sparsemax
        else:
            choice_func = entmax15

        bin_func: Callable[[Tensor], Tensor]
        if self.bin_function == "sparsemoid":
            bin_func = sparsemoid
        else:
            bin_func = entmoid15

        # Dense ODST Block
        self.dense_block = DenseODSTBlock(
            input_dim=self.node_input_dim,
            num_trees=self.num_trees,
            num_layers=self.num_layers,
            tree_output_dim=self.output_dim + self.additional_tree_output_dim,
            max_features=self.max_features,
            input_dropout=self.input_dropout,
            depth=self.depth,
            choice_function=choice_func,
            bin_function=bin_func,
            initialize_response_=getattr(nn.init, self.initialize_response + "_"),
            initialize_selection_logits_=getattr(nn.init, self.initialize_selection_logits + "_"),
            threshold_init_beta=self.threshold_init_beta,
            threshold_init_cutoff=self.threshold_init_cutoff,
        )

        # Embedding layer
        continuous_dim_for_embedding: int = self.continuous_dim if self.continuous_dim is not None else 1
        self.embedding_layer = Embedding1dLayer(
            continuous_dim=continuous_dim_for_embedding,
            categorical_embedding_dims=self.embedding_dims,
            embedding_dropout=self.embedding_dropout,
            batch_norm_continuous_input=self.batch_norm_continuous_input,
        )

        # Output head
        if self.head_type == "subset":
            self.head = Lambda(self.subset)

        elif self.head_type == "linear":
            total_output_dim: int = self.output_dim + self.additional_tree_output_dim
            linear_input_dim: int = self.num_layers * self.num_trees * total_output_dim
            self.head = LinearHead(input_dim=linear_input_dim, output_dim=self.output_dim)

        elif self.head_type == "mlp":
            total_output_dim: int = self.output_dim + self.additional_tree_output_dim
            mlp_input_dim: int = self.num_layers * self.num_trees * total_output_dim
            self.head = MLPHead(
                input_dim=mlp_input_dim,
                output_dim=self.output_dim,
                hidden_dims=self.mlp_hidden_dims,
                dropout=self.mlp_dropout,
                activation=self.mlp_activation,
            )

        elif self.head_type == "flow":
            total_output_dim: int = self.output_dim + self.additional_tree_output_dim
            flow_input_dim: int = self.num_layers * self.num_trees * total_output_dim
            self.head = FlowHead(
                input_dim=flow_input_dim,
                output_dim=self.output_dim,
                flow_type=self.flow_type,
                flow_transforms=self.flow_transforms,
                flow_bins=self.flow_bins,
                flow_degree=self.flow_degree,
                flow_signal=self.flow_signal,
                flow_components=self.flow_components,
            )

        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}. Choose 'subset', 'linear', 'mlp', or 'flow'.")

    def subset(self, x: torch.Tensor) -> torch.Tensor:
        """Original NODE head: take first ``output_dim`` dims and mean across trees."""
        return x[..., : self.output_dim].mean(dim=-2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: raw features → embedding → ODST blocks → (tree dropout) → head.

        Args:
            x: [batch_size, input_dim]

        Returns:
            Predictions [batch_size, output_dim] (or flow distribution for flow heads).
        """
        # Prepare input dict for embedding layer
        if hasattr(self, "categorical_columns_") and hasattr(self, "continuous_columns_"):
            continuous_indices: List[int] = []
            categorical_indices: List[int] = []
            all_columns: List[str] = getattr(self, "feature_names_", [])
            if all_columns:
                for i, col in enumerate(all_columns):
                    if col in self.continuous_columns_:
                        continuous_indices.append(i)
                    elif col in self.categorical_columns_:
                        categorical_indices.append(i)

            continuous_data: Optional[Tensor] = x[:, continuous_indices] if continuous_indices else None
            categorical_data: Optional[Tensor] = x[:, categorical_indices].long() if categorical_indices else None
            x_dict: Dict[str, Optional[Tensor]] = {"continuous": continuous_data, "categorical": categorical_data}
        else:
            x_dict = {"continuous": x, "categorical": None}

        x = self.embedding_layer(x_dict)
        x = self.dense_block(x)

        # Tree dropout: randomly drop entire trees during training
        if self.training and hasattr(self, "tree_dropout") and self.tree_dropout > 0:
            mask = torch.bernoulli(torch.ones_like(x[..., :1]) * (1 - self.tree_dropout))
            x = x * mask / (1 - self.tree_dropout)

        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Sklearn-compatible wrappers (skorch-based)
# ---------------------------------------------------------------------------


class BaseNODEEstimator(NeuralNet, AbstractMotherPipeline):
    """
    Abstract base class for NODE estimators containing shared functionality.

    This class implements all common methods for both NODERegressor and NODEClassifier,
    reducing code duplication and ensuring consistent behavior across both estimators.

    Inherits from NeuralNet first to ensure proper MRO for sklearn compatibility methods.
    """

    # Type annotations for dynamic attributes added by InputOutputShapeSetter callback
    categorical_columns_: List[str]
    continuous_columns_: List[str]
    label_encoders_: dict
    categorical_embedding_dims_: List[tuple]
    feature_names_: List[str]
    target_type_: str
    _prepare_data_for_node: Callable
    _is_dataframe_input: bool

    def _store_node_parameters(
        self,
        num_layers: int,
        num_trees: int,
        additional_tree_output_dim: int,
        depth: int,
        choice_function: str,
        bin_function: str,
        max_features: Optional[int],
        input_dropout: float,
        initialize_response: str,
        initialize_selection_logits: str,
        threshold_init_beta: float,
        threshold_init_cutoff: float,
        embedding_dropout: float,
        batch_norm_continuous_input: bool,
        head_type: str,
        mlp_hidden_dims: Optional[List[int]],
        mlp_dropout: float,
        mlp_activation: str,
        tree_dropout: float,
        flow_type: str,
        flow_transforms: int,
        flow_bins: int,
        flow_degree: int,
        flow_signal: int,
        flow_components: int,
        callbacks: Optional[List[Any]],
        cat_features: Optional[List[str]] = None,
    ) -> None:
        """Persist all NODE-specific parameters as instance attributes.

        This is required for ``sklearn.clone()`` which re-creates the
        estimator from ``get_params()`` → ``__init__(**params)``.
        """
        self.num_layers = num_layers
        self.num_trees = num_trees
        self.additional_tree_output_dim = additional_tree_output_dim
        self.depth = depth
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.max_features = max_features
        self.input_dropout = input_dropout
        self.initialize_response = initialize_response
        self.initialize_selection_logits = initialize_selection_logits
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff
        self.embedding_dropout = embedding_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.head_type = head_type
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout = mlp_dropout
        self.mlp_activation = mlp_activation
        self.tree_dropout = tree_dropout
        self.flow_type = flow_type
        self.flow_transforms = flow_transforms
        self.flow_bins = flow_bins
        self.flow_degree = flow_degree
        self.flow_signal = flow_signal
        self.flow_components = flow_components
        self.cat_features = cat_features
        self.callbacks = callbacks
        self._original_callbacks = callbacks

    def _prepare_callbacks(self, callbacks: Optional[List[Any]], train_split: Optional[Any] = None) -> List[Any]:
        """Ensure essential callbacks are present.

        Always injects:
        - ``InputOutputShapeSetter`` – auto-detects input/output dimensions and
          applies the declared ``cat_features`` as categorical columns
        - ``LossFunctionSetter`` – configures criterion based on head_type

        When a validation split is active (``train_split`` is not None),
        also injects:
        - ``EarlyStopping`` (patience=20, monitor valid_loss)
        """
        callbacks_list = callbacks[:] if callbacks is not None else []
        has_shape_setter = any(isinstance(cb, InputOutputShapeSetter) for cb in callbacks_list)
        has_loss_setter = any(isinstance(cb, LossFunctionSetter) for cb in callbacks_list)

        if not has_shape_setter:
            callbacks_list = [
                InputOutputShapeSetter(categorical_columns=getattr(self, "cat_features", None))
            ] + callbacks_list
        if not has_loss_setter:
            callbacks_list = [LossFunctionSetter()] + callbacks_list

        # Only add early-stopping when validation data exists
        if train_split is not None:
            if not any(isinstance(cb, EarlyStopping) for cb in callbacks_list):
                callbacks_list.append(EarlyStopping(patience=_EARLY_STOPPING_PATIENCE, monitor="valid_loss"))
        return callbacks_list

    @property
    def _is_flow_head(self) -> bool:
        """Check whether the fitted module uses a flow head."""
        return hasattr(self, "module_") and hasattr(self.module_, "head_type") and self.module_.head_type == "flow"

    def _has_active_dropout(self, *, include_mlp: bool = True) -> bool:
        """Check whether any dropout source is configured with a non-zero rate.

        Args:
            include_mlp: Whether to count MLP-head dropout. Set to ``False``
                when checking dropout for flow heads (MLP dropout is irrelevant).
        """
        if hasattr(self, "input_dropout") and self.input_dropout > 0:
            return True
        if hasattr(self, "module_"):
            if hasattr(self.module_, "input_dropout") and self.module_.input_dropout > 0:
                return True
            if hasattr(self.module_, "tree_dropout") and self.module_.tree_dropout > 0:
                return True
            if include_mlp and hasattr(self.module_, "mlp_dropout") and self.module_.mlp_dropout > 0:
                return True
        return False

    def _prepare_fit_X(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Convert DataFrame to float32 numpy array for the DataLoader.

        Stores the original DataFrame so ``InputOutputShapeSetter`` can detect
        categorical columns during ``on_train_begin``.
        """
        self._is_dataframe_input = hasattr(X, "columns")

        if self._is_dataframe_input:
            self._original_X_train = X
            X_processed = X.copy()
            for col in X_processed.columns:
                if _is_string_or_object_dtype(X_processed[col]) or isinstance(
                    X_processed[col].dtype, pd.CategoricalDtype
                ):
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            return X_processed.values.astype(np.float32)

        return np.asarray(X, dtype=np.float32)

    def _build_skorch_init_params(
        self,
        *,
        output_dim_placeholder: int,
        criterion: type,
        optimizer: type,
        lr: float,
        max_epochs: int,
        batch_size: int,
        iterator_train__shuffle: bool,
        train_split: Optional[Any],
        callbacks_list: List[Any],
        device: str,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Build the ``kwargs`` dict for ``super().__init__()`` (Skorch).

        Centralises the ~30 ``module__*`` parameter assignments that are
        identical between ``NODERegressor`` and ``NODEClassifier``.
        """
        return dict(
            module=CompletePyTorchTabularNODE,
            module__input_dim=1,  # placeholder — auto-detected
            module__output_dim=output_dim_placeholder,
            module__num_layers=self.num_layers,
            module__num_trees=self.num_trees,
            module__additional_tree_output_dim=self.additional_tree_output_dim,
            module__depth=self.depth,
            module__choice_function=self.choice_function,
            module__bin_function=self.bin_function,
            module__max_features=self.max_features,
            module__input_dropout=self.input_dropout,
            module__initialize_response=self.initialize_response,
            module__initialize_selection_logits=self.initialize_selection_logits,
            module__threshold_init_beta=self.threshold_init_beta,
            module__threshold_init_cutoff=self.threshold_init_cutoff,
            module__embedding_dropout=self.embedding_dropout,
            module__batch_norm_continuous_input=self.batch_norm_continuous_input,
            module__head_type=self.head_type,
            module__mlp_hidden_dims=self.mlp_hidden_dims,
            module__mlp_dropout=self.mlp_dropout,
            module__mlp_activation=self.mlp_activation,
            module__tree_dropout=self.tree_dropout,
            module__flow_type=self.flow_type,
            module__flow_transforms=self.flow_transforms,
            module__flow_bins=self.flow_bins,
            module__flow_degree=self.flow_degree,
            module__flow_signal=self.flow_signal,
            module__flow_components=self.flow_components,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            iterator_train__shuffle=iterator_train__shuffle,
            train_split=train_split,
            callbacks=callbacks_list,
            device=device,
            **extra,
        )

    def _create_node_module(self, output_dim_placeholder: int = 1) -> "CompletePyTorchTabularNODE":
        """Instantiate ``CompletePyTorchTabularNODE`` from stored parameters.

        Uses placeholder values for ``input_dim`` / ``output_dim`` which are
        overwritten by ``InputOutputShapeSetter`` at train time.
        """
        return CompletePyTorchTabularNODE(
            input_dim=1,  # Placeholder - auto-detected by InputOutputShapeSetter
            output_dim=output_dim_placeholder,  # Placeholder - auto-detected by InputOutputShapeSetter
            num_layers=self.num_layers,
            num_trees=self.num_trees,
            additional_tree_output_dim=self.additional_tree_output_dim,
            depth=self.depth,
            choice_function=self.choice_function,
            bin_function=self.bin_function,
            max_features=self.max_features,
            input_dropout=self.input_dropout,
            initialize_response=self.initialize_response,
            initialize_selection_logits=self.initialize_selection_logits,
            threshold_init_beta=self.threshold_init_beta,
            threshold_init_cutoff=self.threshold_init_cutoff,
            embedding_dropout=self.embedding_dropout,
            batch_norm_continuous_input=self.batch_norm_continuous_input,
            head_type=self.head_type,
            mlp_hidden_dims=self.mlp_hidden_dims,
            mlp_dropout=self.mlp_dropout,
            mlp_activation=self.mlp_activation,
            tree_dropout=self.tree_dropout,
            flow_type=self.flow_type,
            flow_transforms=self.flow_transforms,
            flow_bins=self.flow_bins,
            flow_degree=self.flow_degree,
            flow_signal=self.flow_signal,
            flow_components=self.flow_components,
        )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for sklearn compatibility.

        Excludes dynamically constructed 'module' and 'module__*' params so
        sklearn clone() works correctly (our __init__ reconstructs the module).
        """
        params = super().get_params(deep=deep)

        # Remove dynamically constructed parameters that shouldn't be passed to __init__
        params.pop("module", None)  # Module is constructed from NODE params

        # Remove all module__* parameters - they're created automatically in __init__ from NODE params
        # This prevents duplicate parameter errors during cloning
        params_to_remove = [key for key in params.keys() if key.startswith("module__")]
        for key in params_to_remove:
            params.pop(key, None)

        # Ensure we return the original callbacks list for sklearn compatibility
        if hasattr(self, "_original_callbacks"):
            params["callbacks"] = self._original_callbacks

        return params

    def set_params(self, **params: Any) -> "BaseNODEEstimator":
        """
        Set parameters for sklearn compatibility.

        Syncs NODE architecture params to their module__ counterparts so
        skorch knows to re-initialize the module with new values.
        """
        # List of NODE parameters that need to be synced to module
        node_params = [
            "num_layers",
            "num_trees",
            "additional_tree_output_dim",
            "depth",
            "choice_function",
            "bin_function",
            "max_features",
            "input_dropout",
            "initialize_response",
            "initialize_selection_logits",
            "threshold_init_beta",
            "threshold_init_cutoff",
            "embedding_dropout",
            "batch_norm_continuous_input",
            "head_type",
            "mlp_hidden_dims",
            "mlp_dropout",
            "mlp_activation",
            "tree_dropout",
            "flow_type",
            "flow_transforms",
            "flow_bins",
            "flow_degree",
            "flow_signal",
            "flow_components",
        ]

        # For each NODE parameter being set, also set the module__ version
        # This ensures skorch re-initializes the module with the new parameter
        params_to_add = {}
        for param_name in node_params:
            if param_name in params:
                # Also set module__param_name so skorch passes it to module __init__
                params_to_add[f"module__{param_name}"] = params[param_name]

        # Merge the additional module__ parameters
        params.update(params_to_add)

        return super().set_params(**params)  # type: ignore

    def __sklearn_clone__(self) -> "BaseNODEEstimator":
        """Custom sklearn cloning: excludes 'module' which is constructed dynamically."""
        # Get clean parameters without 'module'
        params = self.get_params(deep=False)

        # Create new instance with clean parameters
        return self.__class__(**params)

    def _prepare_input_data(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Prepare input data for prediction, handling DataFrame inputs."""
        return self._prepare_data_for_node(X)

    def get_embeddings(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """
        Extract learned representations from NODE tree layers (before the head).

        Useful for dimensionality reduction, transfer learning, clustering, and
        understanding learned representations.

        Args:
            X: Input data (n_samples, n_features).

        Returns:
            Flattened tree outputs (n_samples, num_layers * num_trees * total_output_dim).

        Raises:
            ValueError: If model has not been fitted.
        """
        import torch

        # Ensure model is fitted
        if not hasattr(self, "module_"):
            raise ValueError("Model must be fitted before extracting embeddings. Call .fit(X, y) first.")

        # Prepare input data (handle DataFrames, scaling, etc.)
        X_prepared = self._prepare_input_data(X)

        # Set model to evaluation mode
        self.module_.eval()

        # Extract embeddings
        with torch.no_grad():
            # Convert to tensor
            X_tensor = torch.tensor(X_prepared, dtype=torch.float32)

            # Forward through embedding layer if it exists
            # The embedding layer expects a dict with 'continuous' and 'categorical' keys
            if hasattr(self.module_, "embedding_layer") and self.module_.embedding_layer is not None:
                # Split features into continuous/categorical using the same logic as forward()
                if (
                    hasattr(self.module_, "categorical_columns_")
                    and hasattr(self.module_, "continuous_columns_")
                    and hasattr(self.module_, "feature_names_")
                    and self.module_.feature_names_
                ):
                    continuous_indices = [
                        i
                        for i, col in enumerate(self.module_.feature_names_)
                        if col in self.module_.continuous_columns_
                    ]
                    categorical_indices = [
                        i
                        for i, col in enumerate(self.module_.feature_names_)
                        if col in self.module_.categorical_columns_
                    ]
                    continuous_data = X_tensor[:, continuous_indices] if continuous_indices else None
                    categorical_data = X_tensor[:, categorical_indices].long() if categorical_indices else None
                    x_dict: Dict[str, Optional[Tensor]] = {
                        "continuous": continuous_data,
                        "categorical": categorical_data,
                    }
                else:
                    x_dict = {"continuous": X_tensor, "categorical": None}
                X_embedded = self.module_.embedding_layer(x_dict)
            else:
                X_embedded = X_tensor

            # Forward through the dense block (NODE layers) to get tree outputs
            # This is the representation before the head
            tree_outputs = self.module_.dense_block(X_embedded)

            # Flatten the tree outputs to get embeddings
            # Shape: (batch_size, num_layers, num_trees, total_output_dim) -> (batch_size, -1)
            embeddings = tree_outputs.reshape(tree_outputs.shape[0], -1)

            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()

        return embeddings_np

    def _predict_uncertainty_mc_dropout(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        num_samples: int = 100,
        quantiles: Optional[List[float]] = None,
        return_dataframe: bool = False,
        use_std: bool = True,
    ) -> Union[npt.NDArray[np.float32], pd.DataFrame]:
        """
        Monte Carlo Dropout for uncertainty estimation (shared by regressor and classifier).

        Performs multiple forward passes with dropout active to estimate prediction uncertainty.
        Uses the model's configured dropout settings (input_dropout, tree_dropout, mlp_dropout).
        If all dropouts are 0, falls back to deterministic prediction.

        Args:
            X: Input features
            num_samples: Number of forward passes with dropout (default: 100)
            quantiles: Optional quantiles to compute (e.g., [0.025, 0.5, 0.975])
            return_dataframe: If True, return DataFrame with std/IQR and quantile columns
            use_std: If True, use standard deviation; if False, use IQR (default: True)

        Returns:
            Array of std/IQR values or DataFrame with std/IQR and quantiles
        """
        # Determine if this is a classifier based on the instance type
        is_classifier = isinstance(self, NeuralNetClassifier)

        # Check if ANY dropout is configured in the model
        has_dropout = False
        if hasattr(self, "input_dropout") and self.input_dropout > 0:
            has_dropout = True
        if hasattr(self.module_, "tree_dropout") and self.module_.tree_dropout > 0:
            has_dropout = True
        if hasattr(self.module_, "mlp_dropout") and self.module_.mlp_dropout > 0:
            has_dropout = True

        # If no dropout configured, fall back to regular predict
        if not has_dropout:
            module_logger.warning(
                "No dropout configured in model (input_dropout, tree_dropout, mlp_dropout all 0). "
                "Falling back to deterministic predict() without uncertainty estimation. "
                "Set at least one dropout > 0 to enable Monte Carlo Dropout uncertainty."
            )
            if is_classifier:
                predictions = self.predict_proba(X)  # type: ignore
            else:
                predictions = self.predict(X)  # type: ignore

            # Return zeros for IQR since there's no uncertainty
            if quantiles and return_dataframe:
                output_dim = getattr(self.module_, "output_dim", 1)
                data_dict = {}

                uncertainty_key = "std" if use_std else "iqr"
                if is_classifier:
                    # Classifier: columns like class_0_std/class_0_iqr, class_0_q_0.025, etc.
                    for class_idx in range(output_dim):
                        data_dict[f"class_{class_idx}_{uncertainty_key}"] = np.zeros(len(predictions))
                        for q in quantiles:
                            data_dict[f"class_{class_idx}_q_{q}"] = predictions[:, class_idx]
                else:
                    # Regressor: columns like target_0_std/target_0_iqr or just std/iqr
                    if output_dim == 1:
                        data_dict[uncertainty_key] = np.zeros(len(predictions))
                        for q in quantiles:
                            data_dict[f"q_{q}"] = predictions if predictions.ndim == 1 else predictions.flatten()
                    else:
                        for target_idx in range(output_dim):
                            data_dict[f"target_{target_idx}_{uncertainty_key}"] = np.zeros(len(predictions))
                            for q in quantiles:
                                data_dict[f"target_{target_idx}_q_{q}"] = predictions[:, target_idx]
                return pd.DataFrame(data_dict)
            else:
                return np.zeros_like(
                    predictions
                    if predictions.ndim > 1
                    else predictions.reshape(-1, getattr(self.module_, "output_dim", 1))
                )

        # Use model's configured dropout settings for MC Dropout
        module_logger.info(f"MC Dropout: using model's dropout configuration for {num_samples} samples")

        # Get the model and ensure it's in eval mode
        model = self.module_
        model.eval()

        # Prepare input - use the callback's data preparation (bound during fit)
        X = self._prepare_data_for_node(X)

        # Keep the whole model in eval mode (so BatchNorm uses its running statistics
        # and every other stateful layer stays deterministic) and then switch ON
        # *only* the dropout mechanisms for MC-dropout. We deliberately do NOT put the
        # entire model into training mode, which previously also enabled BatchNorm
        # training and other train-only behaviour.
        model.eval()
        # tree_dropout is gated on the top module's own self.training flag.
        model.training = True
        for _m in model.modules():
            # input_dropout is gated on each DenseODSTBlock's self.training flag;
            # mlp_dropout is implemented with standard nn.Dropout layers.
            if isinstance(_m, (DenseODSTBlock, nn.Dropout)):
                _m.training = True

        all_predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                sample_predictions = []

                # Iterate through batches
                for batch in self.get_iterator(X, training=False):
                    Xi = batch[0] if isinstance(batch, (tuple, list)) else batch
                    Xi = Xi.to(self.device)

                    # Use model's forward method which includes appropriate dropout
                    # - Subset/Linear/Flow: tree dropout applied in forward pass
                    # - MLP: internal dropout layers in MLP head are active
                    predictions = model(Xi)

                    sample_predictions.append(predictions.detach().cpu().numpy())

                # Concatenate all batch predictions for this sample
                sample_predictions_np = np.concatenate(sample_predictions, axis=0)
                all_predictions.append(sample_predictions_np)

        # Restore model to eval mode
        model.eval()

        # Stack predictions: shape (num_samples, n_samples, n_outputs)
        all_predictions_np = np.stack(all_predictions, axis=0)

        # Compute uncertainty measure across MC samples
        if use_std:
            # Standard deviation across MC samples
            uncertainty = np.std(all_predictions_np, axis=0)
        else:
            # IQR (75th - 25th percentile) across MC samples (more robust to outliers)
            q75 = np.percentile(all_predictions_np, 75, axis=0)
            q25 = np.percentile(all_predictions_np, 25, axis=0)
            uncertainty = q75 - q25

        # If no quantiles requested, return std/IQR only
        if quantiles is None:
            # Flatten if single output dimension
            if hasattr(self, "module_") and getattr(self.module_, "output_dim", 1) == 1 and not is_classifier:
                return uncertainty.flatten()
            else:
                return uncertainty

        # Compute requested quantiles
        quantile_values = []
        for q in quantiles:
            q_vals = np.percentile(all_predictions_np, q * 100, axis=0)
            quantile_values.append(q_vals)

        # Build result based on return_dataframe flag
        if return_dataframe:
            output_dim = getattr(self.module_, "output_dim", 1)
            data_dict = {}
            uncertainty_key = "std" if use_std else "iqr"

            if is_classifier:
                # Classifier: columns like class_0_std/class_0_iqr, class_0_q_0.025, etc.
                for class_idx in range(output_dim):
                    data_dict[f"class_{class_idx}_{uncertainty_key}"] = uncertainty[:, class_idx]
                    for i, q in enumerate(quantiles):
                        data_dict[f"class_{class_idx}_q_{q}"] = quantile_values[i][:, class_idx]
            else:
                # Regressor: columns like target_0_std/target_0_iqr or just std/iqr
                if output_dim == 1:
                    data_dict[uncertainty_key] = uncertainty.flatten()
                    for i, q in enumerate(quantiles):
                        data_dict[f"q_{q}"] = quantile_values[i].flatten()
                else:
                    for target_idx in range(output_dim):
                        data_dict[f"target_{target_idx}_{uncertainty_key}"] = uncertainty[:, target_idx]
                        for i, q in enumerate(quantiles):
                            data_dict[f"target_{target_idx}_q_{q}"] = quantile_values[i][:, target_idx]

            return pd.DataFrame(data_dict)
        else:
            # Return as numpy array
            result_list = [uncertainty] + quantile_values
            result_np = np.concatenate([arr.reshape(arr.shape[0], -1) for arr in result_list], axis=1)

            # Flatten if single output dimension (regressor only)
            if hasattr(self, "module_") and getattr(self.module_, "output_dim", 1) == 1 and not is_classifier:
                return result_np.flatten()
            else:
                return result_np

    def get_hyperparameter_space(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        y: Union[pd.Series, pd.DataFrame, npt.NDArray[Any]],
        trial: Trial,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Generic hyperparameter space for NODE models.

        Tunes architecture (layers, trees, depth), learning rate, dropout and
        sparse-activation functions.  Head-specific parameters are delegated to
        :meth:`suggested_params_head`, which is overridden by ``NODERegressor``
        and ``NODEClassifier`` to include the head types they support.
        """
        suggested_params = {
            prefix + "num_layers": trial.suggest_int(prefix + "num_layers", 1, 4, log=False),
            prefix + "num_trees": trial.suggest_int(prefix + "num_trees", 256, 2048, step=256, log=False),
            prefix + "additional_tree_output_dim": trial.suggest_int(
                prefix + "additional_tree_output_dim", 0, 4, log=False
            ),
            prefix + "depth": trial.suggest_int(prefix + "depth", 3, 6, log=False),
            prefix + "lr": trial.suggest_float(prefix + "lr", 1e-4, 5e-2, log=True),
        }

        # Tune dropout parameters (architectural regularization).
        # Fine step (0.01) so Optuna can reach the low-dropout regime that
        # normalizing-flow heads favour (best ~0.008-0.05; Werner & Schmidt-Thieme 2025);
        # all three dropout knobs (input/tree/mlp) stay available.
        # input_dropout: Applied to combined features between ODST layers
        suggested_params[prefix + "input_dropout"] = trial.suggest_float(prefix + "input_dropout", 0.0, 0.3, step=0.01)
        # tree_dropout: Applied after NODE layers, before head (architectural)
        suggested_params[prefix + "tree_dropout"] = trial.suggest_float(prefix + "tree_dropout", 0.0, 0.3, step=0.01)

        # Head-specific tuning is delegated entirely to subclass overrides.
        # The base implementation is a no-op; NODERegressor / NODEClassifier
        # handle both tune_head=True (suggest head type) and tune_head=False
        # (tune params for the fixed head) — same pattern as _set_loss.
        suggested_params = self.suggested_params_head(trial, suggested_params, y, prefix)

        suggested_params[prefix + "choice_function"] = trial.suggest_categorical(
            prefix + "choice_function", ("entmax15", "sparsemax")
        )
        suggested_params[prefix + "bin_function"] = trial.suggest_categorical(
            prefix + "bin_function", ("entmoid15", "sparsemoid")
        )

        return suggested_params

    def _suggest_mlp_params(
        self,
        trial: Trial,
        suggested_params: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        """Suggest MLP head hyperparameters.

        Called by :meth:`suggested_params_head` when the selected (or fixed)
        head type is ``"mlp"``.  Extracted as a helper so both
        ``NODERegressor`` and ``NODEClassifier`` can reuse it.
        """
        # Calculate head input dimension
        num_layers = suggested_params.get(prefix + "num_layers", self.num_layers)
        num_trees = suggested_params.get(prefix + "num_trees", self.num_trees)
        additional_output = suggested_params.get(prefix + "additional_tree_output_dim", self.additional_tree_output_dim)
        total_output_dim = (
            len(self.cat_features) + len(self.cont_features) + additional_output
            if hasattr(self, "cat_features") and hasattr(self, "cont_features")
            else 1 + additional_output
        )
        expected_input_dim = num_layers * num_trees * total_output_dim

        # Determine number of MLP hidden layers (respect user's if set, else tune 1-4)
        if hasattr(self, "mlp_hidden_dims") and self.mlp_hidden_dims is not None:
            num_mlp_layers = len(self.mlp_hidden_dims)
        else:
            num_mlp_layers = trial.suggest_int(prefix + "mlp_num_layers", 1, 4)

        # Tune first hidden layer (10-50% of NODE output), derive rest with 2x compression
        min_hidden = max(64, expected_input_dim // 10)
        max_hidden = expected_input_dim // 2
        step = max(16, expected_input_dim // 64)
        max_hidden = min_hidden + ((max_hidden - min_hidden) // step) * step
        hidden_dim_1 = trial.suggest_int(prefix + "mlp_hidden_dim_1", min_hidden, max_hidden, step=step, log=False)

        # Progressive compression: [first, first//2, first//4, ...]
        mlp_hidden_dims = [hidden_dim_1]
        for i in range(1, num_mlp_layers):
            layer_dim = max(16, hidden_dim_1 // (2**i))
            mlp_hidden_dims.append(layer_dim)

        suggested_params[prefix + "mlp_hidden_dims"] = mlp_hidden_dims
        suggested_params[prefix + "mlp_dropout"] = trial.suggest_float(prefix + "mlp_dropout", 0.0, 0.5, log=False)
        suggested_params[prefix + "mlp_activation"] = trial.suggest_categorical(
            prefix + "mlp_activation", ("ReLU", "GELU", "LeakyReLU", "ELU", "SiLU")
        )
        return suggested_params

    def suggested_params_loss(
        self,
        trial: Trial,
        suggested_params: Dict[str, Any],
        y: Union[pd.DataFrame, pd.Series, npt.NDArray[Any]],
        prefix: str,
    ) -> Dict[str, Any]:
        return suggested_params

    def suggested_params_head(
        self,
        trial: Trial,
        suggested_params: Dict[str, Any],
        y: Union[pd.DataFrame, pd.Series, npt.NDArray[Any]],
        prefix: str,
    ) -> Dict[str, Any]:
        """Suggest head-type and its associated hyperparameters.

        Base implementation is a no-op.  ``NODEClassifier`` and
        ``NODERegressor`` override this to suggest head types and
        their associated parameters (MLP dims, flow architecture, etc.).
        """
        return suggested_params

    def default_parameters(self, prefix: str = "") -> Dict[str, Any]:
        """Return default hyperparameters for the general NODE architecture.

        Only covers parameters tuned by the base-class
        :meth:`get_hyperparameter_space` (backbone, dropout, activation
        functions).  Head-specific defaults are added by
        ``NODERegressor`` and ``NODEClassifier``.
        """
        return {
            prefix + "lr": 0.03,
            prefix + "depth": 6,
            prefix + "num_layers": 1,
            prefix + "num_trees": 2048,
            prefix + "additional_tree_output_dim": 3,
            prefix + "choice_function": "entmax15",
            prefix + "bin_function": "entmoid15",
            prefix + "input_dropout": 0.05,
            prefix + "tree_dropout": 0.0,
        }


class NODERegressor(BaseNODEEstimator):
    """
    Neural Oblivious Decision Ensembles (NODE) for regression tasks.

    Key Features:
        - Automatic dimension detection for single/multi-target regression
        - Flow head: probabilistic predictions with sampling (head_type='flow')
        - MLP head: non-linear transformations (head_type='mlp')
        - Mixed data types: continuous and categorical features. Categorical
          columns must be declared explicitly via ``cat_features`` (like
          CatBoost); they are never auto-detected.
        - DataFrame and numpy array support

    Example:
        >>> reg = NODERegressor(num_trees=2048, depth=6, max_epochs=100)
        >>> reg.fit(X_train, y_train)
        >>> predictions = reg.predict(X_test)

        >>> # Declare categorical columns explicitly:
        >>> reg = NODERegressor(cat_features=["city", "education"], max_epochs=100)
        >>> reg.fit(X_train_df, y_train)

        >>> # For probabilistic predictions with flow head:
        >>> # IMPORTANT: Flow heads require standardized targets for numerical stability
        >>> from sklearn.preprocessing import StandardScaler
        >>> y_scaler = StandardScaler()
        >>> y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        >>> reg_prob = NODERegressor(head_type="flow", max_epochs=100)
        >>> reg_prob.fit(X_train, y_train_scaled)
        >>> predictions_scaled = reg_prob.predict(X_test)  # Mode of distribution
        >>> predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

    Note: BaseNODEEstimator is listed first to ensure our get_params() method
    takes precedence over skorch's, which is critical for proper sklearn cloning.
    """

    def __init__(
        self,
        # ====================================================================
        # Core Architecture (most important parameters)
        # ====================================================================
        num_trees: int = 2048,  # Number of trees in ensemble
        depth: int = 6,  # Tree depth (complexity)
        num_layers: int = 1,  # Number of NODE layers
        # ====================================================================
        # Head Configuration (prediction layer)
        # ====================================================================
        head_type: str = "mlp",  # "subset", "linear", "mlp", or "flow" (probabilistic)
        mlp_hidden_dims: Optional[List[int]] = None,  # MLP hidden layer sizes; default [128, 64, 32]
        mlp_activation: str = "ReLU",  # "ReLU", "GELU", or "LeakyReLU" (if head_type="mlp")
        flow_type: str = "NICE",  # Flow architecture (if head_type="flow")
        flow_transforms: int = 3,  # Transform layers (NICE, RealNVP, NAF, UNAF)
        flow_bins: int = 8,  # Spline bins (NSF)
        flow_degree: int = 16,  # Polynomial degree (BPF)
        flow_signal: int = 16,  # Hidden signal dim (NAF, UNAF)
        flow_components: int = 8,  # Mixture components (GMM)
        # ====================================================================
        # Dropout & Regularization (for uncertainty estimation)
        # ====================================================================
        input_dropout: float = 0.05,  # Dropout on input features (low: flow heads are dropout-sensitive)
        tree_dropout: float = 0.0,  # Dropout on trees (0.0 = off)
        mlp_dropout: float = 0.1,  # Dropout in MLP head (if head_type="mlp")
        embedding_dropout: float = 0.0,  # Dropout on categorical embeddings
        # ====================================================================
        # Training Configuration
        # ====================================================================
        max_epochs: int = 100,  # Number of training epochs
        lr: float = 0.01,  # Learning rate
        batch_size: int = 128,  # Batch size for training
        optimizer: type = torch.optim.Adam,  # Optimizer class
        criterion: type = nn.MSELoss,  # Loss function
        device: str = "cuda" if torch.cuda.is_available() else "cpu",  # Device (cuda/cpu)
        # ====================================================================
        # Advanced Architecture (usually keep defaults)
        # ====================================================================
        choice_function: str = "entmax15",  # Feature selection: "entmax15" or "sparsemax"
        bin_function: str = "entmoid15",  # Binning function: "entmoid15" or "sparsemoid"
        additional_tree_output_dim: int = 3,  # Additional output dimensions per tree
        max_features: Optional[int] = None,  # Max features per split (None = all)
        initialize_response: str = "normal",  # Response init: "normal" or "uniform"
        initialize_selection_logits: str = "uniform",  # Selection init: "uniform" or "normal"
        threshold_init_beta: float = 1.0,  # Beta for threshold initialization
        threshold_init_cutoff: float = 1.0,  # Cutoff for threshold initialization
        batch_norm_continuous_input: bool = False,  # Batch norm on continuous features
        # ====================================================================
        # Framework Integration (Mother/Skorch compatibility)
        # ====================================================================
        target_type: str = "single_target",  # "single_target" or "multi_target"
        model_type: str = "regression",  # Model type for Mother framework
        task_weights: Optional[List[float]] = None,  # Weights for multi-task regression
        cat_features: Optional[List[str]] = None,  # Column names to treat as categorical (like CatBoost)
        iterator_train__shuffle: bool = True,  # Shuffle training data
        train_split: Optional[Any] = None,  # Validation split (None = no validation)
        callbacks: Optional[List[Any]] = None,  # Additional Skorch callbacks
        tune_head: bool = True,  # Tune head params during hyperparameter search
        **kwargs: Any,
    ) -> None:
        # Store Mother framework compatibility parameters
        if model_type != "regression":
            raise ValueError("model_type for NODERegressor must be 'regression'.")
        self.model_type = model_type
        self.target_type = target_type
        self.task_weights = task_weights

        # Resolve mutable default for mlp_hidden_dims
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [128, 64, 32]

        # Store all NODE parameters using base class method
        self._store_node_parameters(
            num_layers,
            num_trees,
            additional_tree_output_dim,
            depth,
            choice_function,
            bin_function,
            max_features,
            input_dropout,
            initialize_response,
            initialize_selection_logits,
            threshold_init_beta,
            threshold_init_cutoff,
            embedding_dropout,
            batch_norm_continuous_input,
            head_type,
            mlp_hidden_dims,
            mlp_dropout,
            mlp_activation,
            tree_dropout,
            flow_type,
            flow_transforms,
            flow_bins,
            flow_degree,
            flow_signal,
            flow_components,
            callbacks,
            cat_features,
        )

        # Prepare callbacks list (inject EarlyStopping when val split active)
        callbacks_list = self._prepare_callbacks(callbacks, train_split=train_split)

        super().__init__(
            **self._build_skorch_init_params(
                output_dim_placeholder=1,
                criterion=criterion,
                optimizer=optimizer,
                lr=lr,
                max_epochs=max_epochs,
                batch_size=batch_size,
                iterator_train__shuffle=iterator_train__shuffle,
                train_split=train_split,
                callbacks_list=callbacks_list,
                device=device,
                **kwargs,
            )
        )

        # store the tuning parameters
        self.tune_head = tune_head

    def _set_loss(self, y: Union[pd.Series, npt.NDArray[Any], None] = None) -> None:
        """Set appropriate loss for regression tasks.

        Defaults to ``MSELoss``.  Flow heads use their own negative
        log-likelihood internally, but ``MSELoss`` is still used by the
        skorch wrapper for validation scoring.
        """
        if not isinstance(self.criterion_, nn.MSELoss):
            module_logger.info("LossFunctionSetter: Using MSELoss for regression")
            self.criterion = nn.MSELoss
            self.criterion_ = nn.MSELoss()

    def get_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        X: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute loss with head-type-specific handling.

        - Flow heads: negative log-probability loss
        - Other heads: standard criterion with shape alignment and NaN masking
          for multi-task regression
        """
        if self._is_flow_head:
            # For flow heads, y_pred is the flow distribution conditioned on X
            # We need to compute the negative log probability directly
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(-1)  # Add feature dimension for flow head
            loss = -y_pred.log_prob(y_true)  # -log p(y_true | X) where y_pred = flow(X)
            loss = loss.mean()
            return loss

        # For other head types, handle tensor shape mismatch
        if hasattr(y_pred, "dim"):
            if y_pred.dim() == 2 and y_pred.size(1) == 1 and y_true.dim() == 1:
                y_pred = y_pred.squeeze(1)  # Convert [N, 1] to [N]
            elif y_pred.dim() == 1 and y_true.dim() == 2 and y_true.size(1) == 1:
                y_true = y_true.squeeze(1)  # Convert [N, 1] to [N]

        # Handle NaN values in multi-task regression targets
        has_nan = torch.isnan(y_true).any()

        if has_nan:
            is_multitask = y_true.dim() > 1 and y_true.shape[-1] > 1

            if is_multitask:
                mask = ~torch.isnan(y_true)
                has_any_valid = mask.any(dim=-1)

                if not has_any_valid.all():
                    # Some samples have all NaN targets - raise an informative exception
                    invalid_indices = torch.where(~has_any_valid)[0].cpu().numpy()
                    num_invalid = len(invalid_indices)
                    num_total = len(y_true)
                    raise ValueError(
                        f"Found {num_invalid} sample(s) out of {num_total} with all NaN targets "
                        f"in multi-task regression. "
                        f"Sample indices with all NaN: {invalid_indices.tolist()[:10]}"
                        f"{'...' if num_invalid > 10 else ''}. "
                        f"For multi-task regression with missing values, each sample must have "
                        f"at least one valid (non-NaN) target. "
                        f"Please remove or impute these samples before training."
                    )

                # Use reduction='none' to get per-element loss, then per-target mean
                criterion_instance = self.criterion_ if hasattr(self, "criterion_") else self.criterion()

                original_reduction = getattr(criterion_instance, "reduction", "mean")
                if hasattr(criterion_instance, "reduction"):
                    criterion_instance.reduction = "none"

                # Replace NaN with zeros to prevent NaN gradients
                y_true_safe = torch.where(mask, y_true, torch.zeros_like(y_true))
                loss_all = criterion_instance(y_pred, y_true_safe)

                # Restore original reduction setting
                if hasattr(criterion_instance, "reduction"):
                    criterion_instance.reduction = original_reduction

                # Mask out losses for NaN targets, compute per-target mean
                loss_masked = torch.where(mask, loss_all, torch.tensor(0.0, device=loss_all.device))
                valid_counts = mask.sum(dim=0).float()
                loss_per_target = loss_masked.sum(dim=0) / valid_counts.clamp(min=1.0)

                # Apply task weights if provided
                if self.task_weights is not None:
                    if not isinstance(self.task_weights, Tensor):
                        task_weights_tensor = torch.tensor(
                            self.task_weights, dtype=loss_per_target.dtype, device=loss_per_target.device
                        )
                    else:
                        task_weights_tensor = self.task_weights.to(loss_per_target.device)

                    if task_weights_tensor.shape[0] != loss_per_target.shape[0]:
                        raise ValueError(
                            f"task_weights length ({task_weights_tensor.shape[0]}) must match "
                            f"number of targets ({loss_per_target.shape[0]})"
                        )

                    # Normalize weights so weighted avg == unweighted when all weights equal
                    normalized_weights = task_weights_tensor * len(task_weights_tensor) / task_weights_tensor.sum()
                    weighted_loss = (loss_per_target * normalized_weights).mean()
                    return weighted_loss
                else:
                    return loss_per_target.mean()
            else:
                # Single-task regression with NaN - not supported
                num_nan = torch.isnan(y_true).sum().item()
                total = y_true.numel()
                raise ValueError(
                    f"Found {num_nan} NaN value(s) out of {total} in single-target regression. "
                    f"NaN values in targets are not supported for single-target regression. "
                    f"Please remove or impute samples with NaN targets before training. "
                    f"For multi-target regression with missing values, ensure y has shape (n_samples, n_targets) "
                    f"with n_targets > 1, where each sample has at least one valid (non-NaN) target."
                )

        # Filter kwargs to only include params accepted by the criterion
        if kwargs:
            criterion_instance = self.criterion_ if hasattr(self, "criterion_") else self.criterion()
            try:
                criterion_callable = (
                    criterion_instance.forward if hasattr(criterion_instance, "forward") else criterion_instance
                )
                criterion_sig = signature(criterion_callable)
                accepted_params = set(criterion_sig.parameters.keys()) - {"self"}
                criterion_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
            except (ValueError, TypeError):
                criterion_kwargs = {k: v for k, v in kwargs.items() if k not in ["X", "training"]}
        else:
            criterion_kwargs = kwargs

        return super().get_loss(y_pred, y_true, *args, **criterion_kwargs)

    def fit(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        y: Union[pd.Series, npt.NDArray[np.float32]],
        **fit_params: Any,
    ) -> "NODERegressor":
        """Enhanced fit method with DataFrame support."""
        # Store whether input was DataFrame for later use
        self._is_dataframe_input = hasattr(X, "columns")

        # For DataFrames, store original for callback processing
        if self._is_dataframe_input:
            self._original_X_train = X
            # Convert DataFrame to numpy for PyTorch DataLoader compatibility
            # The callback will detect categorical features and set up encoders
            # But we need numeric data for DataLoader, so encode object/category columns temporarily
            X_processed = X.copy()

            for col in X_processed.columns:
                # Encode both object/string and category dtypes
                if _is_string_or_object_dtype(X_processed[col]) or isinstance(
                    X_processed[col].dtype, pd.CategoricalDtype
                ):
                    # Temporary encoding for DataLoader compatibility
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))

            X = X_processed.values.astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        y = np.asarray(y, dtype=np.float32)
        return super().fit(X, y, **fit_params)  # type: ignore

    def predict(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        num_samples: int = 1000,
    ) -> npt.NDArray[np.float32]:
        """
        Enhanced predict method with DataFrame support.

        Args:
            X: Input features
            num_samples: Number of samples to draw for flow head predictions (default: 1000)
                        More samples = better mode estimate but slower

        Returns:
            Predictions array

        Note:
            For flow heads, predictions are the mode of the distribution, estimated by
            selecting the sample with highest log probability.
        """
        # Use the callback's data preparation
        X = self._prepare_data_for_node(X)

        # Check if using flow head by inspecting the actual module
        is_flow_head = (
            hasattr(self, "module_") and hasattr(self.module_, "head_type") and self.module_.head_type == "flow"
        )

        if is_flow_head:
            # For flow heads, use predict_flow_head and handle flow sampling there
            # This avoids duplicating the batching and tensor conversion logic
            predictions = self.predict_flow_head(X, num_samples=num_samples)
            # Flatten only if output_dim is 1
            if hasattr(self, "module_") and getattr(self.module_, "output_dim", 1) == 1:
                return predictions.flatten()
            else:
                return predictions
        else:
            return super().predict(X)

    def predict_flow_head(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        num_samples: int = 200,
    ) -> npt.NDArray[np.float32]:
        """
        Predict using the flow head for probabilistic regression.

        Since not all zuko flow distributions support .mode property directly,
        we approximate the mode through sampling:

        1. Sample from the flow distribution
        2. Calculate log_prob for all samples (vectorized)
        3. Select the sample with highest log_prob as the mode estimate

        Args:
            X: Input features
            num_samples: Number of samples to draw from the flow distribution.
                        More samples give better mode estimates but slower. Default: 200
                        NOTE: Mode estimation requires standardized targets during training
                        for numerical stability of log_prob calculations.

        Returns:
            Predictions (mode) from the flow distribution

        Note:
            For best results with flow heads, standardize your targets before training:
            ```python
            from sklearn.preprocessing import StandardScaler

            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            model.fit(X_train, y_train_scaled)
            predictions_scaled = model.predict(X_test)
            predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
            ```
        """
        from .m_head_utils import compute_flow_mode_and_uncertainty

        self.module_.eval()
        modes: List[Tensor] = []

        # Use torch.no_grad() to skip gradient computation during inference
        with torch.no_grad():
            # Use skorch's built-in forward method which handles batching and device placement
            for yp in self.forward_iter(X, training=False):
                # Use shared utility function for mode computation (vectorized)
                mode, _ = compute_flow_mode_and_uncertainty(yp, num_samples)
                modes.append(mode)

        # Concatenate and convert to numpy
        modes_np: npt.NDArray[np.float32] = torch.cat(modes, 0).cpu().numpy()

        # Ensure we return the right shape - predict expects 2D
        if len(modes_np.shape) == 1:
            modes_np = modes_np.reshape(-1, 1)

        return modes_np

    def predict_uncertainty(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        return_quantiles: bool = False,
        quantiles: List[float] = DEFAULT_QUANTILES,
        uncertainty_for_opt: bool = False,
        num_samples: int = 100,
        use_std: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, npt.NDArray[np.float32]]]:
        """
        Predict with uncertainty estimation for regression (Mother framework compatible).

        This method matches the interface of other Mother estimators (CatBoost, RandomForest,
        TabPFN) on the ``ranker_update`` branch, returning predictions along with uncertainty
        estimates in a standardised DataFrame.

        Three uncertainty estimation methods:
        1. **Flow head with dropout**: Provides both data uncertainty (from flow) and
           knowledge uncertainty (from MC Dropout) — the gold standard.
        2. **Flow head without dropout**: Returns data uncertainty only from flow distribution.
        3. **Non-flow heads with dropout**: Returns knowledge uncertainty from MC Dropout.

        Args:
            X: Input features.
            return_quantiles: If True, also return quantile predictions (default False).
                Only supported for flow heads, where quantiles are sampled from the
                learned conditional distribution. Requesting quantiles for a non-flow
                (MC-dropout) head raises ``ValueError``.
            quantiles: List of quantiles to calculate the uncertainty.
                Default: ``[0.25, 0.5, 0.75]`` (``DEFAULT_QUANTILES``).
            uncertainty_for_opt: If True, return only total uncertainty for
                optimisation / active learning (default False).
            num_samples: Number of samples for uncertainty estimation (default 100).
            use_std: If True, use standard deviation; if False, use IQR
                for uncertainty (default True).
            **kwargs: Additional keyword arguments (ignored, for pipeline compatibility).

        Returns:
            Union[pd.DataFrame, tuple[pd.DataFrame, np.ndarray]]:
                - If ``return_quantiles=False`` (default): A DataFrame with columns:
                    - ``'mean_predictions'``: The mean predictions for each sample.
                    - ``'knowledge_uncertainty'``: Epistemic uncertainty (from MC Dropout if
                      available, else ``None``).
                    - ``'data_uncertainty'``: Aleatoric uncertainty (from flow if available,
                      else ``None``).
                    - ``'total_uncertainty'``: Combined / primary uncertainty measure.
                - If ``return_quantiles=True``: A tuple containing:
                    - The DataFrame described above.
                    - ``np.ndarray`` of quantile values with shape
                      ``(n_samples, n_quantiles)``.
                - If ``uncertainty_for_opt=True``: ``pd.Series`` of ``total_uncertainty``.

        Example:
            >>> # Flow head with dropout (both uncertainties)
            >>> reg = NODERegressor(head_type="flow", input_dropout=0.1, max_epochs=100)
            >>> reg.fit(X_train, y_train)
            >>> results = reg.predict_uncertainty(X_test)
            >>> print(results[["mean_predictions", "total_uncertainty"]].head())
            >>> # With quantiles (like TabPFN / RandomForest)
            >>> results, quantiles_array = reg.predict_uncertainty(
            ...     X_test, return_quantiles=True, quantiles=[0.025, 0.5, 0.975]
            ... )
        """
        # Check if using flow head
        is_flow_head = (
            hasattr(self, "module_") and hasattr(self.module_, "head_type") and self.module_.head_type == "flow"
        )

        has_dropout = False
        if hasattr(self, "input_dropout") and self.input_dropout > 0:
            has_dropout = True
        if hasattr(self.module_, "input_dropout") and self.module_.input_dropout > 0:
            has_dropout = True
        if hasattr(self.module_, "tree_dropout") and self.module_.tree_dropout > 0:
            has_dropout = True
        # Note: mlp_dropout is NOT checked here for flow heads since it only applies to MLP heads
        if not is_flow_head:
            if hasattr(self.module_, "mlp_dropout") and self.module_.mlp_dropout > 0:
                has_dropout = True

        if return_quantiles and not is_flow_head:
            raise ValueError(
                "Quantiles are only available for flow heads (head_type='flow'). "
                "Non-flow heads estimate uncertainty via MC-dropout, which yields a "
                "mean and std/IQR but not a calibrated predictive distribution. "
                "Set return_quantiles=False."
            )

        index = X.index if isinstance(X, pd.DataFrame) else None

        # Defensive copy to avoid mutating the default list
        quantiles = list(quantiles)

        # Ensure DEFAULT_QUANTILES are included for IQR calculation
        for q in DEFAULT_QUANTILES:
            if q not in quantiles:
                quantiles.append(q)
        quantiles = sorted(quantiles)

        # Compute quantiles only for flow heads
        quantile_predictions = None
        if return_quantiles and is_flow_head:
            X_prep = self._prepare_data_for_node(X)
            self.module_.eval()

            all_quantiles = []
            with torch.no_grad():
                for yp in self.forward_iter(X_prep, training=False):
                    # Sample from flow distributions
                    samples = yp.sample(torch.Size([num_samples]))  # Shape: (n_samples, batch_size, output_dim)

                    # Compute quantiles
                    batch_quantiles = []
                    for q in quantiles:
                        q_vals = torch.quantile(samples, q, dim=0)  # Shape: (batch_size, output_dim)
                        batch_quantiles.append(q_vals)

                    # Stack quantiles: (n_quantiles, batch_size, output_dim)
                    batch_quantiles_stacked = torch.stack(batch_quantiles, dim=0)
                    all_quantiles.append(batch_quantiles_stacked)

            # Concatenate batches: (n_quantiles, total_samples, output_dim)
            quantile_predictions = torch.cat(all_quantiles, dim=1).cpu().numpy()

            # Transpose to (total_samples, n_quantiles) for single target
            # or (total_samples, n_quantiles, output_dim) for multi-target
            quantile_predictions = np.transpose(quantile_predictions, (1, 0, 2))

            # Flatten last dimension if single target
            if quantile_predictions.shape[2] == 1:
                quantile_predictions = quantile_predictions.squeeze(axis=2)  # Shape: (total_samples, n_quantiles)

        # Flow head with dropout: use combined uncertainty (best option)
        if is_flow_head and has_dropout:
            # Get the full stats to access total_uncertainty
            stats = self.predict_with_combined_uncertainty(
                X,
                num_mc_samples=num_samples,
                num_flow_samples=100,
                return_all=True,
            )

            pred = stats["predictions"]
            knowledge_unc = stats["knowledge_uncertainty"]
            data_unc = stats["data_uncertainty"]  # Always 1D (scalar per sample)
            total_unc = stats["total_uncertainty"]  # Always 1D (scalar per sample)

            results = pd.DataFrame(
                {
                    "pred": _prepare_for_dataframe(pred),
                    "mean_predictions": _prepare_for_dataframe(pred),
                    "knowledge_uncertainty": _prepare_for_dataframe(knowledge_unc),
                    "data_uncertainty": data_unc,  # Always scalar
                    "total_uncertainty": total_unc,  # Always scalar
                },
                index=index,
            )

        # Flow head without dropout: use flow uncertainty only
        elif is_flow_head:
            X_prep = self._prepare_data_for_node(X)
            self.module_.eval()

            all_modes = []
            data_unc_list = []

            with torch.no_grad():
                for yp in self.forward_iter(X_prep, training=False):
                    # Point prediction = mode (max log_prob sample)
                    mode_pred, _ = compute_flow_mode_and_uncertainty(yp, num_samples)
                    # Data uncertainty = differential entropy H[p] (aleatoric)
                    fsamples = yp.sample(torch.Size([num_samples]))
                    entropy = -yp.log_prob(fsamples).mean(dim=0)
                    all_modes.append(mode_pred)
                    data_unc_list.append(entropy)

            # Concatenate and convert to numpy
            predictions = torch.cat(all_modes, 0).cpu().numpy()
            uncertainties = torch.cat(data_unc_list, 0).cpu().numpy().flatten()  # Always 1D

            results = pd.DataFrame(
                {
                    "pred": _prepare_for_dataframe(predictions),
                    "mean_predictions": _prepare_for_dataframe(predictions),
                    "knowledge_uncertainty": None,
                    "data_uncertainty": uncertainties,  # Always scalar (1D array)
                    "total_uncertainty": uncertainties,  # Always scalar (1D array)
                },
                index=index,
            )

        # Non-flow heads: use MC Dropout (std/IQR only — no quantiles)
        else:
            predictions = self.predict(X)

            uncertainties = super()._predict_uncertainty_mc_dropout(
                X,
                num_samples=num_samples,
                quantiles=None,
                return_dataframe=False,
                use_std=use_std,
            )

            results = pd.DataFrame(
                {
                    "pred": _prepare_for_dataframe(predictions),
                    "mean_predictions": _prepare_for_dataframe(predictions),
                    "knowledge_uncertainty": _prepare_for_dataframe(uncertainties),
                    "data_uncertainty": None,
                    "total_uncertainty": _prepare_for_dataframe(uncertainties),
                },
                index=index,
            )

        if uncertainty_for_opt:
            # Return total_uncertainty for active-learning / optimisation,
            # matching TabPFN / RandomForest convention.
            return results.loc[:, "total_uncertainty"]

        if return_quantiles:
            return results, quantile_predictions

        return results

    def predict_quantiles(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        quantiles: Optional[List[float]] = None,
        num_samples: int = 200,
    ) -> npt.NDArray[np.float32]:
        """
        Predict quantiles at inference time (compatible with TabPFN / RandomForest interface).

        For **flow heads**, quantiles are computed by sampling from the learned
        conditional distribution $p(y \\mid x)$.
        Only supported for flow heads; non-flow heads raise ``ValueError`` because
        they do not model a predictive distribution.

        Args:
            X: Input features.
            quantiles: List of quantiles in [0, 1] to compute.
                If None, uses ``[0.025, 0.25, 0.5, 0.75, 0.975]``.
            num_samples: Number of flow samples or MC Dropout forward passes.
                Default: 200.

        Returns:
            Array of shape ``(n_samples, n_quantiles)`` for single-target or
            ``(n_samples, n_quantiles, n_targets)`` for multi-target regression.

        Example:
            >>> reg = NODERegressor(head_type="flow", flow_type="NICE")
            >>> reg.fit(X_train, y_train)
            >>> q = reg.predict_quantiles(X_test, quantiles=[0.025, 0.5, 0.975])
            >>> lower, median, upper = q[:, 0], q[:, 1], q[:, 2]
        """
        is_flow_head = (
            hasattr(self, "module_") and hasattr(self.module_, "head_type") and self.module_.head_type == "flow"
        )
        if not is_flow_head:
            raise ValueError(
                "predict_quantiles() is only available for flow heads (head_type='flow'). "
                "Non-flow heads do not model a predictive distribution."
            )

        if quantiles is None:
            quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

        # Validate
        invalid = [q for q in quantiles if not 0 <= q <= 1]
        if invalid:
            raise ValueError(f"Quantiles must be in [0, 1]. Got invalid values: {invalid}")
        quantiles = sorted(quantiles)

        # predict_uncertainty internally appends DEFAULT_QUANTILES for IQR calculation.
        # We call it with the user's quantiles, then filter back to only what was requested.
        merged = sorted(set(quantiles) | set(DEFAULT_QUANTILES))

        _, all_quantile_predictions = self.predict_uncertainty(
            X,
            num_samples=num_samples,
            return_quantiles=True,
            quantiles=list(merged),
        )

        # Filter to only the user-requested quantile columns
        user_indices = [merged.index(q) for q in quantiles]
        if all_quantile_predictions.ndim == 2:
            return all_quantile_predictions[:, user_indices]
        else:
            # multi-target: (n_samples, n_quantiles, n_targets)
            return all_quantile_predictions[:, user_indices, :]

    def predict_with_combined_uncertainty(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        num_mc_samples: int = 50,
        num_flow_samples: int = 100,
        return_all: bool = False,
    ) -> Union[
        Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
        Dict[str, Optional[npt.NDArray[np.float32]]],
    ]:
        """
        Decompose prediction uncertainty into knowledge (epistemic) and data (aleatoric) components.

        Flow-head only. Requires dropout > 0 for knowledge uncertainty.

        Algorithm (information-theoretic / BALD decomposition):
            Treat the ``num_mc_samples`` MC-dropout passes as an ensemble of flows
            ``{p_t(y|x)}``. Using differential entropies estimated by Monte-Carlo
            (``H[p] = -E_{y~p}[log p(y)] ≈ -(1/S) Σ_s log p(y_s)``, ``y_s ~ p``):

            * ``data``  (aleatoric)  = ``(1/T) Σ_t H[p_t]``     — expected entropy
            * ``total``              = ``H[(1/T) Σ_t p_t]``     — mixture entropy
            * ``knowledge`` (epist.) = ``total - data``        — mutual information (≥ 0)

            ``data`` and ``total`` are differential entropies (nats) and may be
            negative for peaked flows; the mutual-information ``knowledge`` term is
            provably non-negative (clamped at 0 to absorb Monte-Carlo noise) so the
            identity ``total == data + knowledge`` holds exactly.

            The ``knowledge`` term above is exactly BALD with continuous entropy
            (``BALD_H`` in Werner & Schmidt-Thieme, 2025): a *scalar-entropy*
            acquisition score computed by first aggregating each flow ``p_t`` into a
            single number ``H[p_t]`` and then subtracting from the mixture entropy.

        Sources / lineage (name the right ones):
            * MI decomposition: Houlsby et al. 2011 (BALD).
            * MC-dropout ensemble approximation: Gal, Islam & Ghahramani 2017.
            * Continuous / regression via differential entropy: Depeweg et al. 2018.
            * Flow ensemble + *sampled* entropy ``H = -(1/S) Σ_s log p(y_s)`` combined
              by subtraction: this is exactly the ``NFlows Out`` method of Berry &
              Meger 2023 (AAAI 2023, pp. 6806-6814; arXiv:2308.13498). Werner &
              Schmidt-Thieme 2025 (BALSA) label this baseline ``BALD_H``.
            IMPORTANT attribution details:
              - We estimate the entropy by SAMPLING (as ``NFlows Out`` does), NOT on a
                fixed grid; the grid/trapezoidal variant is BALSA's own ``BALD_H``.
              - Dropout lives in the NODE trunk / flow-head conditioner with RANDOM
                masks (matches BALSA's stated setup), whereas ``NFlows Out`` uses FIXED
                masks inside the flow's bijective transforms. The decomposition maths is
                identical; only the location/type of the injected noise differs.

        Relation to BALSA (Bayesian Active Learning by Distribution Disagreement):
            BALSA (Werner & Schmidt-Thieme, 2025; arXiv:2501.01248) is an
            active-learning acquisition function that improves on BALD_H for
            normalizing-flow regression. The insight: collapsing each ``p_t`` to a
            single entropy value throws away most of the distributional information,
            and Shannon-entropy / std / least-confidence scores empirically pick poor
            query points for flows. BALSA instead measures the *disagreement between
            the flows directly* with a full distributional distance ``φ`` rather than
            the ``H[mixture] - mean H`` subtraction:

                BALD_H(x)  = Σ_t ( H[p̄] - H[p_t] )          (current code)
                BALSA(x)   = Σ_t φ( p_t , p̄ )                (distribution distance)

            with ``p̄ = (1/T) Σ_t p_t`` the mixture ("average") flow. Two variants of
            ``φ`` and two ways to form ``p̄`` are proposed:

            * ``BALSA_KL`` — φ = KL divergence between densities. Best performer
              overall (``BALSA_KL Pair`` was SOTA across 4 datasets). Two flavours:
                - *Grid*: normalise ``y`` to [0, 1], evaluate every ``p_t`` on a fixed
                  grid (≈200 points) to get likelihood vectors, average them into
                  ``p̄``, then Σ_t KL(p_t, p̄).
                - *Pair*: skip ``p̄`` entirely and sum KL over the ``T-1`` consecutive
                  i.i.d. dropout pairs, Σ_t KL(p_t, p_{t+1}).
            * ``BALSA_EMD`` — φ = Earth-Mover's / Wasserstein distance over i.i.d.
              samples of consecutive pairs, Σ_t EMD(y'_t, y'_{t+1}), y'_t ~ p_t
              (pair-only, since EMD needs samples not grid densities).

            Recommended MC-dropout rate for BALSA is low (~0.05), a full order of
            magnitude below the classic 0.5 used for classification BALD.

        Integrating BALSA into this estimator (not yet implemented):
            All ingredients already exist in this method — no re-training needed:

            * ``dists_by_batch[b][t]`` holds the ``T`` per-pass zuko flow objects and
              ``samples_by_batch[b][t]`` the ``S`` samples drawn from each. The
              ``lp_stack`` cross-evaluation (``log p_{t'}(y_{t,s})`` for all ``t'``)
              already computes everything ``BALSA_KL Pair`` needs, because
              ``KL(p_t, p_{t+1}) ≈ (1/S) Σ_s [log p_t(y_s) - log p_{t+1}(y_s)]``,
              ``y_s ~ p_t`` — i.e. a cheap slice of the tensor we build for the
              mixture-entropy term (essentially free).
            * ``BALSA_KL Grid`` needs a fixed 1-D grid over the (normalised) target
              range and ``flow.log_prob`` evaluated on it, then a mean over ``t`` and a
              trapezoidal KL — a handful of extra tensor ops.
            * ``BALSA_EMD`` needs ``scipy.stats.wasserstein_distance`` (or a sorted-
              sample 1-D EMD) on the per-pass sample sets already stored.

            Suggested surface: a sibling ``acquisition_score(X, method="balsa_kl_pair"
            | "balsa_kl_grid" | "balsa_emd" | "bald")`` returning one score per row for
            pool-based active-learning point selection. It would reuse this method's
            MC-dropout collection loop and simply swap the final reduction. The
            existing ``knowledge_uncertainty`` (BALD_H) already serves as the
            ``"bald"`` baseline. Multi-target (D > 1) would need a per-dimension or
            joint-grid extension, as the paper only covers scalar targets.

        Args:
            X: Input features [n_samples, n_features]
            num_mc_samples: MC Dropout forward passes (default: 50)
            num_flow_samples: Samples from flow per pass (default: 100)
            return_all: If True, return dict with all stats; else return tuple.

        Returns:
            If return_all=False:
                (predictions, knowledge_uncertainty, data_uncertainty)
            If return_all=True:
                Dict with 'predictions', 'knowledge_uncertainty', 'data_uncertainty',
                'total_uncertainty', 'mc_means', 'mc_uncertainties', 'mc_stds'.
        """
        # Check if using flow head
        is_flow_head = (
            hasattr(self, "module_") and hasattr(self.module_, "head_type") and self.module_.head_type == "flow"
        )

        if not is_flow_head:
            raise ValueError(
                "predict_with_combined_uncertainty() only works with flow heads. "
                "Current head_type is not 'flow'. Use predict_uncertainty() instead."
            )

        # Check if ANY dropout is configured that affects the flow head
        # Note: mlp_dropout is NOT relevant for flow heads (only for MLP heads)
        has_dropout = False
        if hasattr(self, "input_dropout") and self.input_dropout > 0:
            has_dropout = True
        if hasattr(self.module_, "input_dropout") and self.module_.input_dropout > 0:
            has_dropout = True
        if hasattr(self.module_, "tree_dropout") and self.module_.tree_dropout > 0:
            has_dropout = True

        # Prepare data
        X = self._prepare_data_for_node(X)

        # If no dropout configured there is only a single flow p(y|x), so the
        # epistemic (mutual-information) term is exactly 0. Data uncertainty is the
        # flow's differential entropy H[p] = -E_{y~p}[log p(y)] (aleatoric).
        if not has_dropout:
            self.module_.eval()

            all_means = []
            data_unc_list = []

            with torch.no_grad():
                for yp in self.forward_iter(X, training=False):
                    samples = yp.sample(torch.Size([num_flow_samples]))  # [S, batch, output_dim]
                    log_p = yp.log_prob(samples)  # [S, batch]
                    # Differential entropy (Monte-Carlo estimate) = data uncertainty
                    entropy = -log_p.mean(dim=0)  # [batch]
                    all_means.append(samples.mean(dim=0))  # [batch, output_dim]
                    data_unc_list.append(entropy)

            # Concatenate and convert to numpy
            predictions = torch.cat(all_means, 0).cpu().numpy()
            data_uncertainty = torch.cat(data_unc_list, 0).cpu().numpy()

            # No knowledge uncertainty without dropout
            knowledge_uncertainty = None

            # Flatten if single target
            output_dim = getattr(self.module_, "output_dim", 1)
            if output_dim == 1:
                predictions = predictions.flatten()
                data_uncertainty = data_uncertainty.flatten()

            if return_all:
                return {
                    "predictions": predictions,
                    "knowledge_uncertainty": knowledge_uncertainty,
                    "data_uncertainty": data_uncertainty,
                    "total_uncertainty": data_uncertainty,  # Only data uncertainty
                    "mc_means": None,  # No MC samples without dropout
                    "mc_stds": None,
                }
            else:
                return predictions, knowledge_uncertainty, data_uncertainty

        # ------------------------------------------------------------------
        # Flow + MC-dropout: information-theoretic (BALD) decomposition.
        #
        # Treat the T = num_mc_samples dropout passes as an ensemble of flows
        # {p_t(y|x)}.  With differential entropies estimated by Monte-Carlo
        # (H[p] = -E_{y~p}[log p(y)] ~= -(1/S) sum_s log p(y_s), y_s ~ p):
        #
        #     data  (aleatoric) = (1/T) sum_t H[p_t]            (expected entropy)
        #     total             = H[(1/T) sum_t p_t]            (mixture entropy)
        #     knowledge (epist.) = total - data                 (mutual information >= 0)
        #
        # data/total are DIFFERENTIAL entropies (nats) and may be negative for
        # peaked flows; the mutual-information (knowledge) term is provably >= 0.
        #
        # IMPORTANT: dropout is enabled but the model stays in eval() so BatchNorm
        # keeps its running statistics (see MC-dropout notes).
        # ------------------------------------------------------------------
        # Enable MC-dropout WITHOUT enabling BatchNorm training: keep the whole
        # model in eval() (so BatchNorm keeps using its running statistics and is
        # never updated) and switch ON *only* the dropout mechanisms. Mirrors
        # _predict_uncertainty_mc_dropout and keeps all three dropout paths active:
        #   * tree_dropout   - gated on the top module's own `.training` flag
        #   * input_dropout  - gated on each DenseODSTBlock's `.training` flag
        #   * mlp_dropout    - standard nn.Dropout layers (flow-head conditioner)
        model = self.module_
        model.eval()
        model.training = True
        for _m in model.modules():
            if isinstance(_m, (DenseODSTBlock, nn.Dropout)):
                _m.training = True

        # Collect, per batch, the T stochastic flow distributions and S samples
        # drawn from each.  The distribution objects are stored so we can later
        # cross-evaluate log p_{t'}(y_{t,s}) for the mixture-entropy term.
        # Call the module directly (not forward_iter) so skorch does not reset the
        # training flags per batch, and iterate with training=False so the batch
        # order is stable across the T passes (dists_by_batch[b] stays aligned).
        dists_by_batch = []  # dists_by_batch[b][t]
        samples_by_batch = []  # samples_by_batch[b][t] : (S, B, D)

        try:
            with torch.no_grad():
                for t in range(num_mc_samples):
                    for b, batch in enumerate(self.get_iterator(X, training=False)):
                        Xi = batch[0] if isinstance(batch, (tuple, list)) else batch
                        Xi = Xi.to(self.device)
                        yp = model(Xi)  # flow distribution for this dropout pass
                        if t == 0:
                            dists_by_batch.append([])
                            samples_by_batch.append([])
                        samp = yp.sample(torch.Size([num_flow_samples]))  # (S, B, D)
                        dists_by_batch[b].append(yp)
                        samples_by_batch[b].append(samp)
        finally:
            model.eval()

        T = num_mc_samples
        log_T = float(np.log(T))

        data_list = []
        total_list = []
        pred_list = []
        per_pass_entropy_list = []  # each (T, B) -> mc_uncertainties
        per_pass_mean_list = []  # each (T, B, D) -> mc_means

        with torch.no_grad():
            for b in range(len(dists_by_batch)):
                dists_b = dists_by_batch[b]  # list length T
                samples_b = samples_by_batch[b]  # list length T of (S, B, D)

                per_source_mix = []  # each (S, B): log p-bar(y_{t,s})
                per_source_self = []  # each (S, B): log p_t(y_{t,s})
                for t in range(T):
                    samp_t = samples_b[t]  # (S, B, D)
                    # log p_{t'}(y_{t,s}) for every t' -> (T, S, B)
                    lp_stack = torch.stack([dists_b[tp].log_prob(samp_t) for tp in range(T)], dim=0)
                    # Mixture density: log p-bar = logsumexp_t' log p_t'  - log T
                    per_source_mix.append(torch.logsumexp(lp_stack, dim=0) - log_T)  # (S, B)
                    per_source_self.append(lp_stack[t])  # (S, B)

                mix_all = torch.stack(per_source_mix, dim=0)  # (T, S, B)
                self_all = torch.stack(per_source_self, dim=0)  # (T, S, B)

                # Differential entropies (per sample in batch)
                total_list.append(-mix_all.mean(dim=(0, 1)))  # (B,)
                data_list.append(-self_all.mean(dim=(0, 1)))  # (B,)

                # Per-pass diagnostics
                per_pass_entropy_list.append(-self_all.mean(dim=1))  # (T, B)
                samp_stack = torch.stack(samples_b, dim=0)  # (T, S, B, D)
                per_pass_mean_list.append(samp_stack.mean(dim=1))  # (T, B, D)

                # Point prediction: mean over all pooled samples
                pred_list.append(samp_stack.mean(dim=(0, 1)))  # (B, D)

        data_uncertainty = torch.cat(data_list, dim=0).detach().cpu().numpy()  # (N,)
        total_raw = torch.cat(total_list, dim=0).detach().cpu().numpy()  # (N,)
        predictions = torch.cat(pred_list, dim=0).detach().cpu().numpy()  # (N, D)

        # Knowledge = mutual information = total - data. Clamp at 0 (Monte-Carlo
        # noise can push the Jensen gap slightly negative), then re-derive total so
        # the additive identity total == data + knowledge holds exactly.
        knowledge_uncertainty = np.maximum(total_raw - data_uncertainty, 0.0)
        total_uncertainty = data_uncertainty + knowledge_uncertainty

        # Per-pass diagnostics: (T, N) -> (num_mc, N, 1); means (num_mc, N, D)
        mc_unc = torch.cat(per_pass_entropy_list, dim=1).unsqueeze(-1).detach().cpu().numpy()
        mc_means = torch.cat(per_pass_mean_list, dim=1).detach().cpu().numpy()

        # Flatten per-sample scores to 1D
        data_uncertainty = data_uncertainty.flatten()
        knowledge_uncertainty = knowledge_uncertainty.flatten()
        total_uncertainty = total_uncertainty.flatten()

        # Flatten predictions if single target
        output_dim = getattr(self.module_, "output_dim", 1)
        if output_dim == 1:
            predictions = predictions.flatten()

        if return_all:
            return {
                "predictions": predictions,
                "knowledge_uncertainty": knowledge_uncertainty,
                "data_uncertainty": data_uncertainty,
                "total_uncertainty": total_uncertainty,
                "mc_means": mc_means,  # [num_mc, total_samples, output_dim]
                "mc_uncertainties": mc_unc,  # [num_mc, total_samples, 1] - per-pass entropy
                "mc_stds": mc_unc,  # legacy alias for backward compatibility
            }
        else:
            return predictions, knowledge_uncertainty, data_uncertainty

    def suggested_params_head(
        self,
        trial: Trial,
        suggested_params: Dict[str, Any],
        y: Union[pd.DataFrame, pd.Series, npt.NDArray[Any]],
        prefix: str,
    ) -> Dict[str, Any]:
        """Suggest head-type and associated hyperparameters for regression.

        When ``tune_head=True`` the head type is tuned among
        ``(subset, mlp, linear)``.  The ``flow`` head is **not**
        included in automatic tuning — it must be selected explicitly
        by setting ``head_type='flow'`` with ``tune_head=False``, which
        will then tune the flow-specific architecture parameters.
        """
        if self.tune_head:
            suggested_params[prefix + "head_type"] = trial.suggest_categorical(
                prefix + "head_type", ("subset", "mlp", "linear")
            )
            selected_head = suggested_params[prefix + "head_type"]
        else:
            selected_head = self.head_type

        # MLP-specific parameters (reuse base-class helper)
        if selected_head == "mlp":
            suggested_params = self._suggest_mlp_params(trial, suggested_params, prefix)

        # Flow-specific parameters (regression only)
        if selected_head == "flow":
            suggested_params = self._suggest_flow_params(trial, suggested_params, prefix)

        return suggested_params

    def _suggest_flow_params(
        self,
        trial: Trial,
        suggested_params: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        """Suggest flow head hyperparameters.

        Called by :meth:`suggested_params_head` when the selected (or fixed)
        head type is ``"flow"``.  Tunes the normalizing-flow architecture,
        number of transforms, and type-specific parameters.
        """
        suggested_params[prefix + "flow_type"] = trial.suggest_categorical(
            prefix + "flow_type",
            ("GMM", "NICE", "RealNVP", "NAF", "UNAF", "NSF", "BPF"),
        )

        selected_flow_type = suggested_params[prefix + "flow_type"]

        # Coupling / autoregressive flow transforms
        if selected_flow_type in ("NICE", "RealNVP", "NAF", "UNAF"):
            suggested_params[prefix + "flow_transforms"] = trial.suggest_int(prefix + "flow_transforms", 2, 5)

        # Hidden signal dimension (NAF / UNAF)
        if selected_flow_type in ("NAF", "UNAF"):
            suggested_params[prefix + "flow_signal"] = trial.suggest_int(prefix + "flow_signal", 8, 32)

        # Mixture components (GMM)
        if selected_flow_type == "GMM":
            suggested_params[prefix + "flow_components"] = trial.suggest_int(prefix + "flow_components", 4, 16)

        # Spline bins (NSF)
        if selected_flow_type == "NSF":
            suggested_params[prefix + "flow_bins"] = trial.suggest_int(prefix + "flow_bins", 4, 16)

        # Polynomial degree (BPF)
        if selected_flow_type == "BPF":
            suggested_params[prefix + "flow_degree"] = trial.suggest_int(prefix + "flow_degree", 8, 32)

        return suggested_params

    def default_parameters(self, prefix: str = "") -> Dict[str, Any]:
        """Default hyperparameters for NODE regression.

        Extends the base architecture defaults with the default head type
        and, when head tuning is active, sensible starting points for
        MLP and flow parameters that Optuna can refine.
        """
        defaults = super().default_parameters(prefix)
        defaults[prefix + "head_type"] = "subset"

        if self.tune_head:
            # MLP defaults (Optuna starting point when MLP is sampled)
            defaults[prefix + "mlp_dropout"] = 0.1
            defaults[prefix + "mlp_activation"] = "ReLU"
            # Flow defaults (Optuna starting point when flow is sampled)
            defaults[prefix + "flow_type"] = "NICE"
            defaults[prefix + "flow_transforms"] = 3

        return defaults


class NODEClassifier(BaseNODEEstimator, NeuralNetClassifier):
    """
    Sklearn-compatible NODE classifier for tabular data.

    Supported:
        - Binary classification (CrossEntropyLoss)
        - Multiclass classification (CrossEntropyLoss)
        - Multi-label binary (BCEWithLogitsLoss)

    Key Features:
        - Auto dimension detection via InputOutputShapeSetter callback
        - Mixed data types with categorical embeddings. Categorical columns must
          be declared explicitly via ``cat_features`` (like CatBoost); they are
          never auto-detected.
        - Head types: subset (default), linear, mlp
        - Uncertainty via Monte Carlo Dropout (requires input_dropout > 0)

    MC Dropout Uncertainty:
        Multiple forward passes with dropout enabled produce a distribution of
        predictions. Spread (std/IQR) across passes measures model confidence.
        Low spread = confident; high spread = uncertain.

    Examples:
        >>> clf = NODEClassifier(num_trees=2048, depth=6, max_epochs=100)
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> probabilities = clf.predict_proba(X_test)

        >>> # With uncertainty
        >>> clf = NODEClassifier(input_dropout=0.1, max_epochs=100)
        >>> clf.fit(X_train, y_train)
        >>> results = clf.predict_uncertainty(X_test, num_samples=100)
    """

    def __init__(
        self,
        # ====================================================================
        # Core Architecture (most important parameters)
        # ====================================================================
        num_trees: int = 2048,  # Number of trees in ensemble
        depth: int = 6,  # Tree depth (complexity)
        num_layers: int = 1,  # Number of NODE layers
        # ====================================================================
        # Head Configuration (prediction layer)
        # ====================================================================
        head_type: str = "subset",  # "subset", "linear", or "mlp" (flow not supported)
        mlp_hidden_dims: Optional[List[int]] = None,  # MLP hidden layer sizes; default [128, 64, 32]
        mlp_activation: str = "ReLU",  # "ReLU", "GELU", or "LeakyReLU" (if head_type="mlp")
        # ====================================================================
        # Dropout & Regularization (for uncertainty estimation)
        # ====================================================================
        input_dropout: float = 0.0,  # Dropout on input features (use > 0 for uncertainty)
        tree_dropout: float = 0.0,  # Dropout on trees (0.0 = off)
        mlp_dropout: float = 0.1,  # Dropout in MLP head (if head_type="mlp")
        embedding_dropout: float = 0.0,  # Dropout on categorical embeddings
        # ====================================================================
        # Training Configuration
        # ====================================================================
        max_epochs: int = 100,  # Number of training epochs
        lr: float = 0.01,  # Learning rate
        batch_size: int = 128,  # Batch size for training
        optimizer: type = torch.optim.Adam,  # Optimizer class
        criterion: type = nn.CrossEntropyLoss,  # Loss function
        device: str = "cuda" if torch.cuda.is_available() else "cpu",  # Device (cuda/cpu)
        # ====================================================================
        # Advanced Architecture (usually keep defaults)
        # ====================================================================
        choice_function: str = "entmax15",  # Feature selection: "entmax15" or "sparsemax"
        bin_function: str = "entmoid15",  # Binning function: "entmoid15" or "sparsemoid"
        additional_tree_output_dim: int = 3,  # Additional output dimensions per tree
        max_features: Optional[int] = None,  # Max features per split (None = all)
        initialize_response: str = "normal",  # Response init: "normal" or "uniform"
        initialize_selection_logits: str = "uniform",  # Selection init: "uniform" or "normal"
        threshold_init_beta: float = 1.0,  # Beta for threshold initialization
        threshold_init_cutoff: float = 1.0,  # Cutoff for threshold initialization
        batch_norm_continuous_input: bool = False,  # Batch norm on continuous features
        flow_type: str = "NICE",  # Reserved for future use (not supported in classification)
        flow_transforms: int = 3,  # Reserved for future use (not supported in classification)
        flow_bins: int = 8,  # Reserved for future use (not supported in classification)
        flow_degree: int = 16,  # Reserved for future use (not supported in classification)
        flow_signal: int = 16,  # Reserved for future use (not supported in classification)
        flow_components: int = 8,  # Reserved for future use (not supported in classification)
        # ====================================================================
        # Framework Integration (Mother/Skorch compatibility)
        # ====================================================================
        model_type: str = "classification_binary",  # Model type for Mother framework
        cat_features: Optional[List[str]] = None,  # Column names to treat as categorical (like CatBoost)
        iterator_train__shuffle: bool = True,  # Shuffle training data
        train_split: Optional[Any] = None,  # Validation split (None = no validation)
        callbacks: Optional[List[Any]] = None,  # Additional Skorch callbacks
        tune_head: bool = True,  # Tune head params during hyperparameter search
        **kwargs: Any,
    ) -> None:
        # Store Mother framework compatibility parameters
        if model_type not in ["classification_binary", "classification_multiclass", "classification_multilabel"]:
            module_logger.warning(
                f"model_type '{model_type}' is unusual for NODEClassifier. "
                f"Expected 'classification_binary', 'classification_multiclass', or 'classification_multilabel'."
            )
        self.model_type = model_type

        # Validate head_type: flow is not supported for classification
        if head_type == "flow":
            raise ValueError(
                "head_type='flow' is not supported for classification. Flow heads are only available for NODERegressor."
            )

        # Resolve mutable default for mlp_hidden_dims
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [128, 64, 32]

        # Store all NODE parameters using base class method
        self._store_node_parameters(
            num_layers,
            num_trees,
            additional_tree_output_dim,
            depth,
            choice_function,
            bin_function,
            max_features,
            input_dropout,
            initialize_response,
            initialize_selection_logits,
            threshold_init_beta,
            threshold_init_cutoff,
            embedding_dropout,
            batch_norm_continuous_input,
            head_type,
            mlp_hidden_dims,
            mlp_dropout,
            mlp_activation,
            tree_dropout,
            flow_type,
            flow_transforms,
            flow_bins,
            flow_degree,
            flow_signal,
            flow_components,
            callbacks,
            cat_features,
        )

        # Prepare callbacks list (inject EarlyStopping when val split active)
        callbacks_list = self._prepare_callbacks(callbacks, train_split=train_split)

        # Disable NeuralNetClassifier's default valid_acc scorer when a validation
        # split is active — it calls predict() on a Skorch Subset which is
        # incompatible with NODE's custom data pipeline.
        if train_split is not None:
            kwargs.setdefault("callbacks__valid_acc", None)

        super().__init__(
            **self._build_skorch_init_params(
                output_dim_placeholder=2,
                criterion=criterion,
                optimizer=optimizer,
                lr=lr,
                max_epochs=max_epochs,
                batch_size=batch_size,
                iterator_train__shuffle=iterator_train__shuffle,
                train_split=train_split,
                callbacks_list=callbacks_list,
                device=device,
                **kwargs,
            )
        )

        # Store the tuning parameters
        self.tune_head = tune_head

    def _set_loss(self, y: Union[pd.Series, npt.NDArray[Any], None] = None) -> None:
        """Set appropriate loss for classification tasks.

        Uses ``BCEWithLogitsLoss`` for multi-label targets (2-D *y* with
        more than one column) and ``CrossEntropyLoss`` for standard
        single-label classification.
        """
        if y is not None and hasattr(y, "shape") and len(y.shape) > 1 and y.shape[1] > 1:
            module_logger.info("LossFunctionSetter: Detected multi-label classification, using BCEWithLogitsLoss")
            self.criterion = nn.BCEWithLogitsLoss
            self.criterion_ = nn.BCEWithLogitsLoss()
        elif not isinstance(self.criterion_, nn.CrossEntropyLoss):
            module_logger.info("LossFunctionSetter: Using CrossEntropyLoss for classification")
            self.criterion = nn.CrossEntropyLoss
            self.criterion_ = nn.CrossEntropyLoss()

    def fit(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        y: Union[pd.Series, npt.NDArray[Any]],
        **fit_params: Any,
    ) -> "NODEClassifier":
        """Enhanced fit method with DataFrame support."""
        # Store whether input was DataFrame for later use
        self._is_dataframe_input = hasattr(X, "columns")

        # For DataFrames, store original for callback processing
        if self._is_dataframe_input:
            self._original_X_train = X
            # Convert DataFrame to numpy for PyTorch DataLoader compatibility
            # The callback will detect categorical features and set up encoders
            # But we need numeric data for DataLoader, so encode object/category columns temporarily
            X_processed = X.copy()

            for col in X_processed.columns:
                # Encode both object/string and category dtypes
                if _is_string_or_object_dtype(X_processed[col]) or isinstance(
                    X_processed[col].dtype, pd.CategoricalDtype
                ):
                    # Temporary encoding for DataLoader compatibility
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))

            X = X_processed.values.astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        # Convert y to appropriate dtype based on criterion
        # BCELoss/BCEWithLogitsLoss need float32; CrossEntropyLoss needs int64.
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        # Multi-label: shape (n, k) with k > 1 → float32 for BCEWithLogitsLoss
        if hasattr(y, "shape") and len(y.shape) > 1 and y.shape[1] > 1:
            y = np.asarray(y, dtype=np.float32)
        else:
            # For single-target classification, ensure y is 1D
            if hasattr(y, "shape") and len(y.shape) > 1:
                y = y.flatten()

            criterion_class = self.criterion if isinstance(self.criterion, type) else type(self.criterion)
            if criterion_class in (nn.BCELoss, nn.BCEWithLogitsLoss):
                y = np.asarray(y, dtype=np.float32)
            else:
                # Default to int64 for CrossEntropyLoss and most classification losses
                y = np.asarray(y, dtype=np.int64)

        return super().fit(X, y, **fit_params)  # type: ignore

    def predict(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> npt.NDArray[Any]:
        """Enhanced predict method with DataFrame support and multi-label handling."""
        X = self._prepare_input_data(X)

        # For multi-label classification (BCEWithLogitsLoss), use sigmoid threshold
        if isinstance(self.criterion_, nn.BCEWithLogitsLoss):
            # Get probabilities (sigmoid outputs) and threshold at 0.5
            probas = self.predict_proba(X)
            return (probas > 0.5).astype(int)
        else:
            # Standard multiclass classification
            return super().predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Enhanced predict_proba method with DataFrame support and multi-label handling."""
        X = self._prepare_input_data(X)

        # For multi-label classification (BCEWithLogitsLoss), apply sigmoid to logits
        if isinstance(self.criterion_, nn.BCEWithLogitsLoss):
            # Use forward_iter to get raw logits
            y_probas = []
            for yp in self.forward_iter(X, training=False):
                # yp contains raw logits, apply sigmoid
                probas = torch.sigmoid(yp).cpu().numpy()
                y_probas.append(probas)
            return np.concatenate(y_probas, 0)
        else:
            # Standard multiclass classification - use skorch's default (applies softmax)
            return super().predict_proba(X)

    def predict_uncertainty(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        return_quantiles: bool = False,
        quantiles: List[float] = DEFAULT_QUANTILES,
        uncertainty_for_opt: bool = False,
        num_samples: int = 100,
        use_std: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, npt.NDArray[np.float32]]]:
        """
        Predict with uncertainty estimation for classification (Mother framework compatible).

        This method matches the interface of other Mother classifiers (CatBoost),
        returning predictions along with uncertainty estimates in a standardised
        DataFrame.

        Args:
            X: Input features.
            return_quantiles: Not supported for classification. Quantiles are only
                available for flow (regression) heads; passing True raises ``ValueError``.
            quantiles: Accepted for interface compatibility but unused.
            uncertainty_for_opt: If True, return only ``knowledge_uncertainty``
                for optimisation (default False).
            num_samples: Number of MC Dropout forward passes (default 100).
            use_std: If True, use std; if False, use IQR for uncertainty (default True).
            **kwargs: Additional keyword arguments (ignored, for pipeline compatibility).

        Returns:
            pd.DataFrame:
                - If ``return_quantiles=False``: DataFrame with columns:
                    - ``'mean_predictions'``: Mean-over-dropout probability of the
                      reported class (class 1 for binary, predicted-class prob for
                      multiclass).
                    - ``'knowledge_uncertainty'``: Epistemic uncertainty — mutual
                      information ``total - data`` (matches CatBoost).
                    - ``'data_uncertainty'``: Aleatoric uncertainty — mean per-pass
                      entropy (expected entropy) across MC-dropout passes.
                    - ``'total_uncertainty'``: Entropy of the mean predictive
                      distribution.
                - If ``uncertainty_for_opt=True``: ``pd.DataFrame`` with 1 column
                    ``'knowledge_uncertainty'``.

        Raises:
            ValueError: If ``return_quantiles=True`` (quantiles require a flow head).
        """
        if return_quantiles:
            raise ValueError(
                "Quantiles are only available for flow heads. NODE classification "
                "estimates uncertainty via MC-dropout, not a calibrated predictive "
                "distribution. Set return_quantiles=False."
            )

        from scipy.stats import entropy

        # Monte-Carlo dropout probability samples: (num_samples, n_datapoints, n_classes)
        probabilities = self._mc_dropout_proba_samples(X, num_samples)

        # Predictive (mean) distribution across MC passes.
        mean_probs = probabilities.mean(axis=0)  # (n_datapoints, n_classes)
        pred = mean_probs.argmax(axis=1)

        # Uncertainty decomposition matching CatBoost (Malinin et al.):
        #   total     = entropy of the mean predictive distribution  H(mean_p)
        #   data      = mean per-pass entropy (expected entropy)      E_t[H(p_t)]
        #   knowledge = total - data (mutual information; 0 when dropout inactive)
        total_uncertainty = entropy(mean_probs, axis=1)
        per_pass_entropy = entropy(probabilities, axis=2)  # (num_samples, n_datapoints)
        data_uncertainty = per_pass_entropy.mean(axis=0)
        knowledge_uncertainty = total_uncertainty - data_uncertainty

        index = X.index if isinstance(X, pd.DataFrame) else None

        # mean_predictions: mean-over-dropout probability of the reported class.
        if mean_probs.shape[1] == 2:
            mean_predictions = mean_probs[:, 1]
        else:
            mean_predictions = mean_probs.max(axis=1)

        results = pd.DataFrame(
            {
                "pred": pred,
                "mean_predictions": mean_predictions,
                "knowledge_uncertainty": knowledge_uncertainty,
                "data_uncertainty": data_uncertainty,
                "total_uncertainty": total_uncertainty,
            },
            index=index,
        )

        if uncertainty_for_opt:
            return pd.DataFrame(
                {"knowledge_uncertainty": results["knowledge_uncertainty"]},
                index=index,
            )

        return results

    def _mc_dropout_proba_samples(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float32]], num_samples: int
    ) -> npt.NDArray[np.float32]:
        """Run ``num_samples`` MC-dropout forward passes and return class probabilities.

        Dropout layers are kept active during inference so that each pass yields a
        different probability vector. Uses sigmoid for multi-label (BCE) criteria and
        softmax otherwise.

        Returns:
            Array of shape ``(num_samples, n_datapoints, n_classes)``.
        """
        model = self.module_
        model.eval()
        X_prep = self._prepare_data_for_node(X)

        # Enable dropout for MC sampling.
        model.train()

        use_sigmoid = isinstance(self.criterion_, nn.BCEWithLogitsLoss)
        all_probs = []
        with torch.no_grad():
            for _ in range(num_samples):
                batch_probs = []
                for batch in self.get_iterator(X_prep, training=False):
                    Xi = batch[0] if isinstance(batch, (tuple, list)) else batch
                    Xi = Xi.to(self.device)
                    logits = model(Xi)
                    probs = torch.sigmoid(logits) if use_sigmoid else torch.softmax(logits, dim=1)
                    batch_probs.append(probs.detach().cpu().numpy())
                all_probs.append(np.concatenate(batch_probs, axis=0))

        # Restore eval mode.
        model.eval()

        return np.stack(all_probs, axis=0)

    def predict_quantiles(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        quantiles: Optional[List[float]] = None,
        num_samples: int = 200,
    ) -> npt.NDArray[np.float32]:
        """
        Not supported for classification.

        Quantiles are only available for flow (regression) heads, which model a
        predictive distribution. NODE classification estimates uncertainty via
        MC-dropout entropy and therefore does not expose predictive quantiles.

        Raises:
            ValueError: Always, because classification has no flow head.
        """
        raise ValueError(
            "predict_quantiles() is only available for flow heads (regression). "
            "NODE classification does not model a predictive distribution; use "
            "predict_uncertainty() for MC-dropout uncertainty instead."
        )

    def suggested_params_head(
        self,
        trial: Trial,
        suggested_params: Dict[str, Any],
        y: Union[pd.DataFrame, pd.Series, npt.NDArray[Any]],
        prefix: str,
    ) -> Dict[str, Any]:
        """Suggest head-type and associated hyperparameters for classification.

        When ``tune_head=True`` the head type itself is tuned among
        ``(subset, mlp, linear)``.  Flow heads are not supported for
        classification.  When ``tune_head=False`` the head type is fixed
        but head-specific params (e.g. MLP dims) are still tuned.
        """
        if self.tune_head:
            suggested_params[prefix + "head_type"] = trial.suggest_categorical(
                prefix + "head_type", ("subset", "mlp", "linear")
            )
            selected_head = suggested_params[prefix + "head_type"]
        else:
            selected_head = self.head_type

        if selected_head == "mlp":
            suggested_params = self._suggest_mlp_params(trial, suggested_params, prefix)

        return suggested_params

    def default_parameters(self, prefix: str = "") -> Dict[str, Any]:
        """Default hyperparameters for NODE classification.

        Extends the base architecture defaults with the default head type
        and, when head tuning is active, sensible MLP starting points
        for Optuna.  Flow heads are not supported for classification.
        """
        defaults = super().default_parameters(prefix)
        defaults[prefix + "head_type"] = "subset"

        if self.tune_head:
            # MLP defaults (Optuna starting point when MLP is sampled)
            defaults[prefix + "mlp_dropout"] = 0.1
            defaults[prefix + "mlp_activation"] = "ReLU"

        return defaults
