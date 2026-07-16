"""
Neural Network Head Architectures for Regression and Classification

This module provides standalone head architectures that can be used independently
or as part of larger models like NODE. It includes both deterministic (MLP) and
probabilistic (Flow) heads for flexible modeling of different types of predictions.

Available Heads:
- MLPHead: Multi-layer perceptron for deterministic predictions
- FlowHead: Flow-based architecture for probabilistic regression

Key Features:
- Configurable architectures
- Multiple activation functions and flow types
- Dropout regularization
- Hyperparameter optimization support via Optuna
- Skorch wrapper for scikit-learn compatibility
- Can be used standalone or integrated into other models

Usage Examples:

    # MLP Head for regression
    from mother.ml.models.m_heads import MLPHeadRegressor

    reg = MLPHeadRegressor(
        input_dim=512,
        output_dim=1,
        hidden_dims=[256, 128],
        max_epochs=100,
        lr=0.001
    )
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    # MLP Head for classification
    from mother.ml.models.m_heads import MLPHeadClassifier

    clf = MLPHeadClassifier(
        input_dim=512,
        output_dim=3,  # 3 classes
        hidden_dims=[256, 128],
        max_epochs=100,
        lr=0.001
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Flow Head for probabilistic regression
    from mother.ml.models.m_heads import FlowHeadRegressor

    reg = FlowHeadRegressor(
        input_dim=512,
        output_dim=1,
        flow_type="NSF",
        max_epochs=100,
        lr=0.001
    )
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)  # Point predictions
    samples = reg.predict_flow(X_test, num_samples=1000)  # Full distribution

References:
    NodeFlow Architecture (flow head):
        Wielopolski, P., Furman, O., & Zięba, M. (2024).
        NodeFlow: Towards End-to-end Flexible Probabilistic Regression on Tabular Data.
        Entropy, 26(7), 593.
        https://doi.org/10.3390/e26070593

    Uncertainty decomposition (MC-dropout heads):
        The classification MC-dropout split (predictive entropy = expected entropy +
        mutual information) follows the CatBoost virtual-ensemble decomposition of
        Malinin, Prokhorenkova & Ustimenko (2021), "Uncertainty in Gradient Boosting
        via Ensembles" (arXiv:2006.10562) — itself the BALD mutual-information split of
        Houlsby et al. (2011) approximated with MC-dropout (Gal, Islam & Ghahramani,
        2017). The flow-head regression analogue (differential-entropy BALD over an
        MC-dropout flow ensemble, entropy estimated by sampling) corresponds to the
        ``NFlows Out`` method of Berry & Meger (2023); see ``m_node.py``
        ``predict_with_combined_uncertainty`` for the full attribution.

Authors: Julian Qian, Sergey Popov
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from optuna import Trial
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import Callback, EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit

try:
    import zuko
except ModuleNotFoundError:  # pragma: no cover - zuko is only needed for flow heads
    zuko = None  # type: ignore[assignment]

from mother.ml.core import AbstractMotherPipeline
from mother.ml.models.m_head_utils import compute_flow_mode_and_uncertainty

module_logger = logging.getLogger(__name__)

# Default quantiles for the standardised predict_uncertainty interface
# (mirrors the convention used by CatBoost / TabPFN / RandomForest / NODE).
DEFAULT_QUANTILES: List[float] = [0.25, 0.5, 0.75]


def _prepare_for_dataframe(values: npt.NDArray[np.float32]) -> Any:
    """Convert a prediction / uncertainty array into a column suitable for a DataFrame.

    Single-target arrays are flattened to 1D; multi-target arrays are converted to a
    list of per-row vectors so each cell holds the full target vector. This mirrors the
    helper used by the NODE estimators so head outputs share the same layout.
    """
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.flatten()
    return [row for row in arr]


# ============================================================================
# SHARED UTILITIES
# ============================================================================


class DimensionSetter(Callback):
    """Automatically detect and set ``input_dim`` / ``output_dim`` from training data.

    Runs at ``on_train_begin`` and updates the module's dimension parameters
    based on the actual data shapes.  This allows users to create standalone
    MLP / Flow heads without specifying dimensions upfront.

    Detection rules:
    - **input_dim**: ``X.shape[1]``
    - **output_dim for classification**: number of unique values in *y*,
      or ``len(net.classes_)`` if Skorch has already detected them.
    - **output_dim for regression**: ``y.shape[1]`` if 2-D, else ``1``.
    """

    def on_train_begin(self, net: Any, X: Any = None, y: Any = None, **kwargs: Any) -> None:
        """Detect dimensions from data and update module parameters."""
        # Skip if dimensions are already properly set (not default placeholders).
        # Use NeuralNet.get_params (via super()) to access module__* keys that
        # our overridden get_params() strips out.
        raw_params = super(type(net), net).get_params()
        current_input_dim = raw_params.get("module__input_dim", 1)

        # Only auto-detect if using default placeholder value
        if current_input_dim != 1:
            return  # Dimensions already set by user

        # Get actual dimensions from data
        if isinstance(X, pd.DataFrame):
            input_dim = X.shape[1]
        elif hasattr(X, "shape"):
            input_dim = X.shape[1] if len(X.shape) > 1 else 1
        else:
            input_dim = len(X[0]) if len(X) > 0 else 1

        # Detect output dimension
        # For classification, check if we can infer number of classes
        if y is not None:
            if hasattr(net, "classes_"):  # Classification task (skorch detected classes)
                # Number of classes detected by skorch
                output_dim = len(net.classes_)
            elif isinstance(net, NeuralNetClassifier):
                # Classification without pre-detected classes: infer the number of
                # classes from the unique target values. This "few unique values"
                # heuristic must NOT be applied to regressors, where an integer-valued
                # target (e.g. counts) with few unique values would otherwise corrupt
                # the regressor's output shape.
                if isinstance(y, pd.DataFrame):
                    output_dim = int(pd.Series(y.values.ravel()).nunique())
                elif isinstance(y, pd.Series):
                    output_dim = int(y.nunique())
                else:
                    output_dim = int(len(np.unique(np.asarray(y))))
            elif hasattr(y, "shape"):
                # Regression: output dimension is purely shape-based.
                output_dim = y.shape[1] if len(y.shape) > 1 else 1
            else:
                output_dim = 1
        else:
            output_dim = 1

        # Update the network's module parameters and force re-initialization
        net.set_params(module__input_dim=input_dim, module__output_dim=output_dim)
        if net.initialized_:
            net.initialize()


# ============================================================================
# MLP HEAD - Deterministic Predictions
# ============================================================================


class MLPHead(nn.Module):
    """
    Multi-layer perceptron head for neural networks.

    This head takes flattened feature representations and applies a series of linear layers
    with nonlinear activations to produce final predictions. It's a general-purpose
    architecture that can learn sophisticated mappings from input features to targets.

    The MLP processes inputs through:
    1. Input layer: Linear transformation of input features
    2. Hidden layers: Linear + Activation + Dropout (repeated)
    3. Output layer: Final linear transformation to target dimension

    Architecture:
        Input → [Linear → Activation → Dropout]* → Linear → Output

    Args:
        input_dim: Total dimension of input features
        output_dim: Target output dimension (e.g., number of classes or regression targets)
        hidden_dims: List of hidden layer sizes [512, 256, ...]
        dropout: Dropout rate for regularization (default: 0.1)
        activation: Activation function name - "ReLU", "GELU", or "LeakyReLU" (default: "ReLU")
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "ReLU",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation_name = activation
        self.batch_norm = batch_norm

        # === ACTIVATION FUNCTION FACTORY ===
        def _make_activation() -> nn.Module:
            if activation == "ReLU":
                return nn.ReLU()
            elif activation == "GELU":
                return nn.GELU()
            elif activation == "LeakyReLU":
                return nn.LeakyReLU()
            elif activation == "ELU":
                return nn.ELU()
            elif activation == "SiLU":
                return nn.SiLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        # === MLP LAYER CONSTRUCTION ===
        # Architecture: Input → [Linear → BatchNorm → Activation → Dropout]* → Linear → Output
        # Each hidden block uses batch normalization for stable training and
        # a fresh activation instance (required for nn.Sequential).
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:  # Hidden layer (not final output)
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(_make_activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # === WEIGHT INITIALIZATION (Kaiming) ===
        # Proper init prevents vanishing/exploding gradients and speeds up convergence.
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass for MLP head.

        Args:
            x: Input features with shape [batch_size, input_dim]
               Can also accept [batch_size, ...] and will flatten automatically
               Can be passed as positional or keyword argument (for Skorch DataFrame compatibility)
            **kwargs: Additional keyword arguments. If x is None, will extract from kwargs

        Returns:
            Final predictions with shape [batch_size, output_dim]
        """
        # Handle case where Skorch passes DataFrame as kwargs
        if x is None:
            # Skorch with DataFrames passes data as {'feature_0': tensor, 'feature_1': tensor, ...}
            # We need to concatenate them into a single tensor
            if kwargs:
                # Concatenate the per-column tensors in DataFrame column order.
                # Iterate values() (insertion order) rather than sorted(keys()) so the
                # column order matches the array input path and is not scrambled for
                # names like 'feature_10' vs 'feature_2'.
                tensors = [v for v in kwargs.values() if isinstance(v, torch.Tensor)]
                if tensors:
                    # Each tensor is [batch_size] or [batch_size, 1], concatenate along feature dimension
                    # First ensure all are 2D
                    tensors_2d = [t.view(-1, 1) if t.dim() == 1 else t for t in tensors]
                    x = torch.cat(tensors_2d, dim=1)
                else:
                    raise ValueError("No input data provided to forward()")
            else:
                raise ValueError("No input data provided to forward()")

        # Flatten higher-dimensional inputs to [batch_size, input_dim]
        if x.dim() > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

        # Apply MLP transformation → [batch_size, output_dim]
        return self.mlp(x)


def _suggest_adaptive_hidden_dims(
    trial: Trial,
    input_dim: int,
    *,
    layers_key: str,
    width_key: str,
    max_layers: int = 4,
) -> List[int]:
    """Suggest an input-adaptive funnel MLP architecture.

    Shared by the standalone MLP head (:class:`BaseMLPHeadEstimator`) and the flow
    head's MLP trunk (:class:`BaseFlowHeadEstimator`) so both size their hidden
    layers identically:

    - **depth** ``[1, max_layers]`` (``layers_key``);
    - **first-layer width** scaled to the input dimension (``width_key``): from
      ``max(64, input_dim // 2)`` up to ``input_dim * 2``, step-aligned so Optuna
      only proposes reachable values;
    - **funnel**: each subsequent layer halves the previous width, floored at 32.

    Args:
        trial: Optuna trial used to suggest the depth and first-layer width.
        input_dim: Number of input features (drives the width scaling).
        layers_key: Full trial parameter name for the number of hidden layers.
        width_key: Full trial parameter name for the first hidden-layer width.
        max_layers: Maximum number of hidden layers (default 4).

    Returns:
        The list of hidden-layer sizes (the funnel architecture).
    """
    num_layers = trial.suggest_int(layers_key, 1, max_layers, log=False)

    # First hidden layer scaled to the data with a generous floor so small
    # datasets still get a usable trunk.
    min_hidden = max(64, input_dim // 2)
    max_hidden = max(min_hidden, input_dim * 2)
    step = max(32, input_dim // 16)
    # Ensure max_hidden is reachable from min_hidden with the chosen step.
    max_hidden = min_hidden + ((max_hidden - min_hidden) // step) * step

    first_hidden = trial.suggest_int(width_key, min_hidden, max_hidden, step=step, log=False)

    # Funnel: each subsequent layer halves the previous width (floored at 32).
    hidden_dims = [first_hidden]
    for i in range(1, num_layers):
        hidden_dims.append(max(32, first_hidden // (2**i)))
    return hidden_dims


class BaseMLPHeadEstimator:
    """
    Base mixin for MLP Head estimators with hyperparameter optimization support.

    This class provides:
    - Hyperparameter space definition for Optuna
    - Default parameter values
    - Common functionality for regression and classification
    """

    def get_hyperparameter_space(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        y: Union[pd.Series, pd.DataFrame, npt.NDArray[Any]],
        trial: Trial,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Define hyperparameter search space for MLP head.

        This method is called by MotherTuner during hyperparameter optimization.
        It suggests optimal values for:
        - Number of hidden layers (architecture depth)
        - Size of first hidden layer (then derives subsequent layers)
        - Dropout rate (regularization strength)
        - Activation function (nonlinearity type)
        - Learning rate

        Args:
            X: Input features (used to determine input dimension)
            y: Target values (not used but required by interface)
            trial: Optuna trial object for suggesting hyperparameters
            prefix: Prefix for parameter names (default: "")

        Returns:
            Dictionary of suggested hyperparameters
        """
        suggested_params = {}

        # Get input dimension from data
        if isinstance(X, pd.DataFrame):
            input_dim = X.shape[1]
        else:
            input_dim = X.shape[1] if hasattr(X, "shape") else len(X[0])

        # === ARCHITECTURE HYPERPARAMETERS ===

        # Depth + input-adaptive funnel widths, shared with the flow head's MLP
        # trunk so both heads size their layers identically.
        hidden_dims = _suggest_adaptive_hidden_dims(
            trial,
            input_dim,
            layers_key=prefix + "num_hidden_layers",
            width_key=prefix + "hidden_dim_1",
            max_layers=4,
        )
        suggested_params[prefix + "hidden_dims"] = hidden_dims

        # === REGULARIZATION HYPERPARAMETERS ===

        # Dropout rate (probability of dropping units during training)
        # Range: 0.0 to 0.5
        # Higher = more regularization but may hurt learning
        suggested_params[prefix + "dropout"] = trial.suggest_float(prefix + "dropout", 0.0, 0.5, log=False)

        # Batch normalization between hidden layers
        # Stabilizes training and often improves generalization
        suggested_params[prefix + "batch_norm"] = trial.suggest_categorical(prefix + "batch_norm", (True, False))

        # === ACTIVATION FUNCTION ===

        # Type of nonlinearity between layers
        # ReLU: Fast and simple, good default
        # GELU: Smoother, often better for complex patterns
        # LeakyReLU: Prevents dead neurons, good for deep networks
        suggested_params[prefix + "activation"] = trial.suggest_categorical(
            prefix + "activation", ("ReLU", "GELU", "LeakyReLU")
        )

        # === OPTIMIZATION HYPERPARAMETERS ===

        # Learning rate for optimizer
        # Range: 1e-5 to 1e-2 (log scale)
        # Lower = more stable but slower training
        # Higher = faster but may overshoot optimal weights
        suggested_params[prefix + "lr"] = trial.suggest_float(prefix + "lr", 1e-5, 1e-2, log=True)

        return suggested_params

    def default_parameters(self, prefix: str = "") -> Dict[str, Any]:
        """
        Return default hyperparameters for MLP head.

        These defaults provide a good starting point for most tasks:
        - 3-layer funnel architecture [256, 128, 64]
        - 10% dropout for regularization
        - Batch normalization enabled
        - ReLU activation (simple and effective)
        - Learning rate of 0.001

        Args:
            prefix: Prefix for parameter names (default: "")

        Returns:
            Dictionary of default parameters
        """
        return {
            prefix + "hidden_dims": [256, 128, 64],
            prefix + "dropout": 0.1,
            prefix + "batch_norm": True,
            prefix + "activation": "ReLU",
            prefix + "lr": 0.001,
        }


class MLPHeadRegressor(NeuralNetRegressor, BaseMLPHeadEstimator, AbstractMotherPipeline):
    """
    MLP Head Regressor with scikit-learn API via Skorch.

    This wrapper enables the MLP head to be used as a drop-in replacement for
    scikit-learn regressors. It automatically handles:
    - Input/output dimension detection
    - Training loop with early stopping
    - Prediction interface
    - Integration with scikit-learn pipelines
    - Hyperparameter optimization with Optuna

    Inherits from NeuralNetRegressor first to ensure proper MRO for sklearn compatibility.

    Args:
        input_dim: Input feature dimension (required)
        output_dim: Output dimension (default: 1 for single-target regression)
        hidden_dims: List of hidden layer sizes (default: [256, 128, 64])
        dropout: Dropout rate (default: 0.05)
        activation: Activation function name (default: "ReLU")
        max_epochs: Maximum training epochs (default: 500)
        lr: Learning rate (default: 0.005)
        **kwargs: Additional arguments passed to NeuralNetRegressor.
            Notable Skorch kwargs: optimizer, optimizer__weight_decay,
            batch_size, train_split, callbacks, device.

    Example:
        >>> from mother.ml.models.m_heads import MLPHeadRegressor
        >>> reg = MLPHeadRegressor(input_dim=20, output_dim=1, max_epochs=50)
        >>> reg.fit(X_train, y_train)
        >>> predictions = reg.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int = 1,  # Placeholder - automatically detected from data
        output_dim: int = 1,  # Placeholder - automatically detected from data
        hidden_dims: Union[List[int], None] = None,
        dropout: float = 0.05,
        batch_norm: bool = True,
        activation: str = "ReLU",
        max_epochs: int = 500,
        lr: float = 0.005,
        **kwargs: Any,
    ) -> None:
        # Set defaults
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # ── Sensible training defaults ──────────────────────────────────
        # AdamW provides proper weight-decay decoupling for better generalisation
        kwargs.setdefault("optimizer", torch.optim.AdamW)
        kwargs.setdefault("optimizer__weight_decay", 1e-2)

        # Validation split for early stopping / LR scheduling (user can override)
        kwargs.setdefault("train_split", ValidSplit(cv=0.1))

        # Build default callbacks: DimensionSetter + EarlyStopping + LR scheduler
        callbacks = list(kwargs.get("callbacks", []))
        if not any(isinstance(cb, DimensionSetter) for cb in callbacks):
            callbacks.append(DimensionSetter())
        if not any(isinstance(cb, EarlyStopping) for cb in callbacks):
            callbacks.append(EarlyStopping(patience=20, monitor="valid_loss"))
        if not any(isinstance(cb, LRScheduler) for cb in callbacks):
            callbacks.append(
                LRScheduler(
                    policy="ReduceLROnPlateau",
                    monitor="valid_loss",
                    patience=7,
                    factor=0.5,
                )
            )
        kwargs["callbacks"] = callbacks

        # Initialize Skorch regressor
        super().__init__(
            module=MLPHead,
            module__input_dim=input_dim,
            module__output_dim=output_dim,
            module__hidden_dims=hidden_dims,
            module__dropout=dropout,
            module__batch_norm=batch_norm,
            module__activation=activation,
            max_epochs=max_epochs,
            lr=lr,
            **kwargs,
        )

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters, implementing AbstractMotherPipeline requirement with proper MRO."""
        # Use super() to follow MRO: NeuralNetRegressor -> BaseMLPHeadEstimator -> AbstractMotherPipeline
        params: dict = super().get_params(deep=deep)
        # Re-expose module__<head_param> as bare constructor arguments so that
        # sklearn.clone() (which rebuilds via __class__(**get_params())) preserves the
        # head configuration instead of silently reverting to constructor defaults.
        head_params: List[str] = ["input_dim", "output_dim", "hidden_dims", "dropout", "batch_norm", "activation"]
        for name in head_params:
            module_key = f"module__{name}"
            if hasattr(self, module_key):
                params[name] = getattr(self, module_key)
        # Remove module__* parameters to avoid conflicts during sklearn cloning
        params_to_remove: List[str] = [key for key in params.keys() if key.startswith("module__")]
        for key in params_to_remove:
            params.pop(key, None)
        params.pop("module", None)  # Also remove 'module' itself
        return params

    def set_params(self, **params: Any) -> "MLPHeadRegressor":
        """Set parameters, implementing AbstractMotherPipeline requirement with proper MRO."""
        # List of MLP Head parameters that need to be synced to module
        head_params: List[str] = ["input_dim", "output_dim", "hidden_dims", "dropout", "batch_norm", "activation"]

        # For each head parameter being set, also set the module__ version
        # and remove the bare name so that skorch doesn't reject it.
        for param_name in head_params:
            if param_name in params:
                params[f"module__{param_name}"] = params.pop(param_name)

        # Use super() to follow MRO: NeuralNetRegressor -> BaseMLPHeadEstimator -> AbstractMotherPipeline
        return super().set_params(**params)

    def __sklearn_clone__(self) -> "MLPHeadRegressor":
        """Custom sklearn cloning to avoid passing 'module' parameter."""
        params: dict = self.get_params(deep=False)
        # Remove 'module' if present (Skorch adds it automatically)
        params.pop("module", None)
        return self.__class__(**params)

    def fit(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        y: Optional[npt.NDArray[np.float32]] = None,
        **fit_params: Any,
    ) -> "MLPHeadRegressor":
        """Fit the model, ensuring correct dtype and target shape."""
        if isinstance(X, np.ndarray) and X.dtype == np.float64:
            X = X.astype(np.float32)
        if y is not None:
            if isinstance(y, np.ndarray) and y.dtype == np.float64:
                y = y.astype(np.float32)
            # Skorch MSELoss needs 2D targets to match [batch, output_dim] predictions
            if isinstance(y, np.ndarray) and y.ndim == 1:
                y = y.reshape(-1, 1)
        return super().fit(X, y, **fit_params)

    def predict(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Predict, returning 1D array for sklearn compatibility."""
        if isinstance(X, np.ndarray) and X.dtype == np.float64:
            X = X.astype(np.float32)
        preds = super().predict(X)
        # Flatten [batch, 1] → [batch] for sklearn compatibility
        if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.ravel()
        return preds

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        return_quantiles: bool = False,
        quantiles: List[float] = DEFAULT_QUANTILES,
        uncertainty_for_opt: bool = False,
        num_samples: int = 100,
        use_std: bool = True,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Predict with uncertainty estimation using MC Dropout (Mother framework compatible).

        This standalone MLP head has no probabilistic output, so uncertainty comes purely
        from Monte-Carlo dropout: multiple stochastic forward passes give a predictive
        ``mean`` and ``std``. This matches the interface of the other Mother estimators
        (CatBoost, TabPFN, RandomForest, NODE).

        Args:
            X: Input features.
            return_quantiles: Not supported for MC-dropout heads. Quantiles are only
                available for flow heads; passing True raises ``ValueError``.
            quantiles: Accepted for interface compatibility but unused.
            uncertainty_for_opt: If True, return only ``total_uncertainty`` as a Series
                for optimisation / active learning (default False).
            num_samples: Number of MC Dropout forward passes (default 100).
            use_std: If True, use standard deviation; if False, use IQR (default True).
            **kwargs: Additional arguments (ignored, for interface compatibility).

        Returns:
            Union[pd.DataFrame, pd.Series]:
                - Default: DataFrame with columns ``pred``, ``mean_predictions``,
                  ``knowledge_uncertainty``, ``data_uncertainty`` (None), ``total_uncertainty``.
                - If ``uncertainty_for_opt=True``: ``pd.Series`` of ``total_uncertainty``.

        Raises:
            ValueError: If ``return_quantiles=True`` (quantiles require a flow head).
        """
        if return_quantiles:
            raise ValueError(
                "Quantiles are only available for flow heads. The MLP head estimates "
                "uncertainty via MC-dropout, which yields a predictive mean and std/IQR "
                "but not a calibrated predictive distribution. Set return_quantiles=False."
            )

        index = X.index if isinstance(X, pd.DataFrame) else None

        # Deterministic point prediction (dropout off).
        point_pred = self.predict(X)

        # Collect MC Dropout samples: shape (num_samples, n_datapoints, output_dim).
        samples = self._mc_dropout_samples(X, num_samples)

        mean_pred = samples.mean(axis=0)
        if use_std:
            uncertainty = samples.std(axis=0)
        else:
            uncertainty = np.percentile(samples, 75, axis=0) - np.percentile(samples, 25, axis=0)

        results = pd.DataFrame(
            {
                "pred": _prepare_for_dataframe(point_pred),
                "mean_predictions": _prepare_for_dataframe(mean_pred),
                "knowledge_uncertainty": _prepare_for_dataframe(uncertainty),
                "data_uncertainty": None,
                "total_uncertainty": _prepare_for_dataframe(uncertainty),
            },
            index=index,
        )

        if uncertainty_for_opt:
            return results.loc[:, "total_uncertainty"]

        return results

    def _mc_dropout_samples(self, X: pd.DataFrame, num_samples: int) -> npt.NDArray[np.float32]:
        """Run ``num_samples`` stochastic forward passes with dropout active.

        Returns an array of shape ``(num_samples, n_datapoints, output_dim)``.
        """
        # Convert to float32
        if isinstance(X, np.ndarray) and X.dtype == np.float64:
            X_input = X.astype(np.float32)
        else:
            X_input = X

        # Convert to tensor
        if not isinstance(X_input, torch.Tensor):
            X_tensor = torch.tensor(
                X_input.values if isinstance(X_input, pd.DataFrame) else X_input, dtype=torch.float32
            )
        else:
            X_tensor = X_input

        # Move to same device as model
        device = next(self.module_.parameters()).device
        X_tensor = X_tensor.to(device)

        # Enable dropout for inference (MC Dropout) while keeping BatchNorm layers in
        # eval mode so their running statistics are not updated and predictions do not
        # become batch-dependent during uncertainty sampling.
        self.module_.eval()
        for _m in self.module_.modules():
            if isinstance(_m, nn.Dropout):
                _m.train()

        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.module_(X_tensor)
                predictions.append(pred.cpu().numpy())

        # Return to eval mode
        self.module_.eval()

        # Stack predictions: shape (num_samples, n_datapoints, output_dim)
        return np.stack(predictions, axis=0)


class MLPHeadClassifier(NeuralNetClassifier, BaseMLPHeadEstimator, AbstractMotherPipeline):
    """
    MLP Head Classifier with scikit-learn API via Skorch.

    This wrapper enables the MLP head to be used as a drop-in replacement for
    scikit-learn classifiers. It automatically handles:
    - Input/output dimension detection
    - Training loop with early stopping
    - Prediction interface with class labels
    - Probability predictions
    - Integration with scikit-learn pipelines
    - Hyperparameter optimization with Optuna

    Inherits from NeuralNetClassifier first to ensure proper MRO for sklearn compatibility.

    Args:
        input_dim: Input feature dimension (required)
        output_dim: Number of classes (required)
        hidden_dims: List of hidden layer sizes (default: [256, 128, 64])
        dropout: Dropout rate (default: 0.05)
        activation: Activation function name (default: "ReLU")
        max_epochs: Maximum training epochs (default: 500)
        lr: Learning rate (default: 0.005)
        **kwargs: Additional arguments passed to NeuralNetClassifier.
            Notable Skorch kwargs: optimizer, optimizer__weight_decay,
            batch_size, train_split, callbacks, criterion, device.

    Example:
        >>> from mother.ml.models.m_heads import MLPHeadClassifier
        >>> clf = MLPHeadClassifier(input_dim=20, output_dim=3, max_epochs=50)
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> probabilities = clf.predict_proba(X_test)
    """

    def __init__(
        self,
        input_dim: int = 1,  # Placeholder - automatically detected from data
        output_dim: int = 1,  # Placeholder - automatically detected from data
        hidden_dims: Union[List[int], None] = None,
        dropout: float = 0.05,
        batch_norm: bool = True,
        activation: str = "ReLU",
        max_epochs: int = 500,
        lr: float = 0.005,
        **kwargs: Any,
    ) -> None:
        # Set defaults
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # ── Sensible training defaults ──────────────────────────────────────
        # AdamW provides proper weight-decay decoupling for better generalisation
        kwargs.setdefault("optimizer", torch.optim.AdamW)
        kwargs.setdefault("optimizer__weight_decay", 1e-2)

        kwargs.setdefault("train_split", ValidSplit(cv=0.1))

        callbacks = list(kwargs.get("callbacks", []))
        if not any(isinstance(cb, DimensionSetter) for cb in callbacks):
            callbacks.append(DimensionSetter())
        if not any(isinstance(cb, EarlyStopping) for cb in callbacks):
            callbacks.append(EarlyStopping(patience=20, monitor="valid_loss"))
        if not any(isinstance(cb, LRScheduler) for cb in callbacks):
            callbacks.append(
                LRScheduler(
                    policy="ReduceLROnPlateau",
                    monitor="valid_loss",
                    patience=7,
                    factor=0.5,
                )
            )
        kwargs["callbacks"] = callbacks

        # Initialize Skorch classifier
        # NOTE: Skorch NeuralNetClassifier defaults to NLLLoss which expects
        # log-probabilities, but our MLPHead outputs raw logits.
        # We must use CrossEntropyLoss which applies LogSoftmax internally.
        kwargs.setdefault("criterion", nn.CrossEntropyLoss)

        super().__init__(
            module=MLPHead,
            module__input_dim=input_dim,
            module__output_dim=output_dim,
            module__hidden_dims=hidden_dims,
            module__dropout=dropout,
            module__batch_norm=batch_norm,
            module__activation=activation,
            max_epochs=max_epochs,
            lr=lr,
            **kwargs,
        )

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters, implementing AbstractMotherPipeline requirement with proper MRO."""
        # Use super() to follow MRO: NeuralNetClassifier -> BaseMLPHeadEstimator -> AbstractMotherPipeline
        params: dict = super().get_params(deep=deep)
        # Re-expose module__<head_param> as bare constructor arguments so that
        # sklearn.clone() (which rebuilds via __class__(**get_params())) preserves the
        # head configuration instead of silently reverting to constructor defaults.
        head_params: List[str] = ["input_dim", "output_dim", "hidden_dims", "dropout", "batch_norm", "activation"]
        for name in head_params:
            module_key = f"module__{name}"
            if hasattr(self, module_key):
                params[name] = getattr(self, module_key)
        # Remove module__* parameters to avoid conflicts during sklearn cloning
        params_to_remove: List[str] = [key for key in params.keys() if key.startswith("module__")]
        for key in params_to_remove:
            params.pop(key, None)
        params.pop("module", None)  # Also remove 'module' itself
        return params

    def set_params(self, **params: Any) -> "MLPHeadClassifier":
        """Set parameters, implementing AbstractMotherPipeline requirement with proper MRO."""
        # List of MLP Head parameters that need to be synced to module
        head_params: List[str] = ["input_dim", "output_dim", "hidden_dims", "dropout", "batch_norm", "activation"]

        # For each head parameter being set, also set the module__ version
        # and remove the bare name so that skorch doesn't reject it.
        for param_name in head_params:
            if param_name in params:
                params[f"module__{param_name}"] = params.pop(param_name)

        # Use super() to follow MRO: NeuralNetClassifier -> BaseMLPHeadEstimator -> AbstractMotherPipeline
        return super().set_params(**params)

    def __sklearn_clone__(self) -> "MLPHeadClassifier":
        """Custom sklearn cloning to avoid passing 'module' parameter."""
        params: dict = self.get_params(deep=False)
        # Remove 'module' if present (Skorch adds it automatically)
        params.pop("module", None)
        return self.__class__(**params)

    def fit(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        y: Optional[npt.NDArray[Any]] = None,
        **fit_params: Any,
    ) -> "MLPHeadClassifier":
        """Fit the model, ensuring correct dtypes.

        - X is cast to float32 for PyTorch
        - y is cast to int64 for CrossEntropyLoss (requires Long targets)
        """
        if isinstance(X, np.ndarray) and X.dtype == np.float64:
            X = X.astype(np.float32)
        if y is not None and isinstance(y, np.ndarray) and not np.issubdtype(y.dtype, np.int64):
            y = y.astype(np.int64)
        return super().fit(X, y, **fit_params)

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        return_quantiles: bool = False,
        quantiles: List[float] = DEFAULT_QUANTILES,
        uncertainty_for_opt: bool = False,
        num_samples: int = 100,
        use_std: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Predict with uncertainty estimation using MC Dropout (Mother framework compatible).

        Multiple stochastic forward passes with dropout active produce per-class
        probabilities. The uncertainty is decomposed as CatBoost does (Malinin et al.):
        ``total_uncertainty`` = entropy of the mean (predictive) probability;
        ``data_uncertainty`` (aleatoric) = mean per-pass entropy (expected entropy);
        ``knowledge_uncertainty`` (epistemic) = ``total - data`` (mutual information).
        This matches the standardised interface of the other Mother classifiers.

        Args:
            X: Input features.
            return_quantiles: Not supported for MC-dropout heads. Quantiles are only
                available for flow heads; passing True raises ``ValueError``.
            quantiles: Accepted for interface compatibility but unused.
            uncertainty_for_opt: If True, return only ``knowledge_uncertainty`` as a
                single-column DataFrame for optimisation (default False).
            num_samples: Number of MC Dropout forward passes (default 100).
            use_std: Unused for classification; kept for interface compatibility.
            **kwargs: Additional arguments (ignored).

        Returns:
            pd.DataFrame:
                - Default: DataFrame with columns ``pred``, ``mean_predictions``
                  (mean-over-dropout probability of the reported class),
                  ``knowledge_uncertainty`` (mutual information, total - data),
                  ``data_uncertainty`` (mean per-pass entropy),
                  ``total_uncertainty`` (entropy of the mean probability).
                - If ``uncertainty_for_opt=True``: single-column ``knowledge_uncertainty``
                  DataFrame.

        Raises:
            ValueError: If ``return_quantiles=True`` (quantiles require a flow head).
        """
        from scipy.stats import entropy

        if return_quantiles:
            raise ValueError(
                "Quantiles are only available for flow heads. The MLP classifier estimates "
                "uncertainty via MC-dropout entropy, not a calibrated predictive distribution. "
                "Set return_quantiles=False."
            )

        index = X.index if isinstance(X, pd.DataFrame) else None

        # Convert to float32
        if isinstance(X, np.ndarray) and X.dtype == np.float64:
            X_input = X.astype(np.float32)
        else:
            X_input = X

        # Convert to tensor
        if not isinstance(X_input, torch.Tensor):
            X_tensor = torch.tensor(
                X_input.values if isinstance(X_input, pd.DataFrame) else X_input, dtype=torch.float32
            )
        else:
            X_tensor = X_input

        # Move to same device as model
        device = next(self.module_.parameters()).device
        X_tensor = X_tensor.to(device)

        # Enable dropout for inference (MC Dropout) while keeping BatchNorm layers in
        # eval mode so their running statistics are not updated and predictions do not
        # become batch-dependent during uncertainty sampling.
        self.module_.eval()
        for _m in self.module_.modules():
            if isinstance(_m, nn.Dropout):
                _m.train()

        probabilities = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.module_(X_tensor)
                probs = torch.softmax(logits, dim=1)
                probabilities.append(probs.cpu().numpy())

        # Return to eval mode
        self.module_.eval()

        # Stack probabilities: shape (num_samples, n_datapoints, n_classes)
        probabilities = np.stack(probabilities, axis=0)

        # Predictive (mean) distribution across MC dropout passes.
        mean_probs = probabilities.mean(axis=0)  # shape: (n_datapoints, n_classes)

        # Most likely class.
        mean_pred = mean_probs.argmax(axis=1)

        # Uncertainty decomposition matching CatBoost (Malinin et al.):
        #   total     = entropy of the mean predictive distribution  H(mean_p)
        #   data      = mean per-pass entropy (expected entropy)      E_t[H(p_t)]
        #   knowledge = total - data (mutual information; 0 when dropout inactive)
        total_uncertainty = entropy(mean_probs, axis=1)
        per_pass_entropy = entropy(probabilities, axis=2)  # (num_samples, n_datapoints)
        data_uncertainty = per_pass_entropy.mean(axis=0)
        knowledge_uncertainty = total_uncertainty - data_uncertainty

        # mean_predictions: mean-over-dropout probability of the reported class.
        if mean_probs.shape[1] == 2:
            mean_predictions = mean_probs[:, 1]
        else:
            mean_predictions = mean_probs.max(axis=1)

        results = pd.DataFrame(
            {
                "pred": mean_pred,
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


# ============================================================================
# FLOW HEAD - Probabilistic Predictions
# ============================================================================


class FlowHead(nn.Module):
    """
    Flow-based head for probabilistic regression with conditional normalizing flows.

    Architecture follows NodeFlow (Wielopolski, Furman & Zięba, 2024):
    Input Embeddings → Conditional Normalizing Flow → Probabilistic Predictions

    The flow head models the conditional distribution p(y|x) using normalizing flows,
    providing:
    - Flexible non-parametric density estimation
    - Uncertainty quantification (aleatoric + epistemic)
    - Mode/mean/median predictions
    - Full predictive distribution via sampling

    Args:
        input_dim: Dimension of input embeddings (conditioning context)
        output_dim: Target output dimension (regression targets)
        flow_type: Type of normalizing flow architecture. Options:
            - "GMM": Gaussian Mixture Model
                * Simple mixture of Gaussians — fastest, strong baseline
                * Best for: Quick experiments, small data

            - "NICE": Non-linear Independent Components Estimation (2014, default)
                * Simple additive coupling layers
                * Very fast, robust across datasets
                * Best for: General use, fast training

            - "RealNVP": Density estimation using Real NVP (2016)
                * Affine coupling layers — slightly more expressive than NICE
                * Best for: General use with a bit more capacity

            - "NAF": Neural Autoregressive Flow (2018)
                * Neural network–based autoregressive transforms
                * Best for: Chemical / molecular data

            - "UNAF": Unconstrained Monotonic Neural Networks (2019)
                * Unconstrained monotonic transforms — very expressive
                * Best for: Chemical / molecular data, complex patterns

            - "NSF": Neural Spline Flow (2019)
                * Monotonic rational-quadratic splines
                * Best for: Multi-modal distributions

            - "BPF": Bernstein-Polynomial Flow (2020)
                * Smooth monotonic Bernstein polynomial transforms
                * Best for: Small data, molecular property prediction
        flow_transforms: Number of transformation layers (default: 3)
            Used by NICE, RealNVP, NAF, UNAF.
        flow_bins: Number of spline bins for NSF (default: 8)
        flow_degree: Polynomial degree for BPF (default: 16)
        flow_signal: Hidden signal dimension for NAF/UNAF (default: 16)
        flow_components: Number of mixture components for GMM (default: 8)
        mlp_hidden_dims: Optional list of hidden sizes for an MLP encoder placed
            *before* the flow (default: None = condition the flow directly on the
            raw input). When provided, the input is first mapped to an embedding of
            size ``mlp_hidden_dims[-1]`` and the flow is conditioned on that
            embedding. Together with ``mlp_dropout`` this enables MC-dropout
            epistemic uncertainty for the standalone flow head (analogous to how the
            NODE+flow head derives epistemic uncertainty from its trunk dropout).
        mlp_dropout: Dropout rate applied inside the MLP encoder (default: 0.0).
            Only has an effect when ``mlp_hidden_dims`` is set. Must be > 0 to obtain
            knowledge (epistemic) uncertainty via MC-dropout.
        mlp_activation: Activation for the MLP encoder (default: "ReLU"). One of
            "ReLU", "GELU", "LeakyReLU", "ELU", "SiLU", "Tanh".
        mlp_batch_norm: Whether to apply batch normalisation inside the MLP encoder
            (default: True). Only has an effect when ``mlp_hidden_dims`` is set. The
            encoder mirrors the standalone :class:`MLPHead` block layout
            ``Linear -> BatchNorm -> activation -> Dropout`` so a flow head with an
            MLP encoder is defined exactly like the standalone MLP head, just with the
            normalising flow attached afterwards.
    """

    SUPPORTED_FLOW_TYPES = ("GMM", "NICE", "RealNVP", "NAF", "UNAF", "NSF", "BPF")

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        flow_type: str = "NICE",
        flow_transforms: int = 3,
        flow_bins: int = 8,
        flow_degree: int = 16,
        flow_signal: int = 16,
        flow_components: int = 8,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_dropout: float = 0.0,
        mlp_activation: str = "ReLU",
        mlp_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if zuko is None:  # pragma: no cover - exercised only when the optional dep is absent
            raise ModuleNotFoundError(
                "zuko is required for FlowHead / FlowHeadRegressor. Install the optional "
                "dependencies, e.g. `pip install mother-ml[node]`."
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.flow_type = flow_type
        self.flow_transforms = flow_transforms
        self.flow_bins = flow_bins
        self.flow_degree = flow_degree
        self.flow_signal = flow_signal
        self.flow_components = flow_components
        self.mlp_hidden_dims = list(mlp_hidden_dims) if mlp_hidden_dims else None
        self.mlp_dropout = mlp_dropout
        self.mlp_activation = mlp_activation
        self.mlp_batch_norm = mlp_batch_norm

        # === OPTIONAL MLP ENCODER (conditioner) ===
        # When hidden dims are given, the flow is conditioned on an MLP embedding
        # instead of the raw input. Dropout layers inside this encoder are what make
        # MC-dropout epistemic uncertainty possible for the standalone flow head.
        # The per-layer block layout (Linear -> BatchNorm -> activation -> Dropout)
        # deliberately mirrors the standalone ``MLPHead`` so the MLP part of a flow
        # head is defined exactly like the standalone MLP head, with the flow attached
        # afterwards.
        if self.mlp_hidden_dims:

            def _make_activation() -> nn.Module:
                if mlp_activation == "ReLU":
                    return nn.ReLU()
                elif mlp_activation == "GELU":
                    return nn.GELU()
                elif mlp_activation == "LeakyReLU":
                    return nn.LeakyReLU()
                elif mlp_activation == "ELU":
                    return nn.ELU()
                elif mlp_activation == "SiLU":
                    return nn.SiLU()
                elif mlp_activation == "Tanh":
                    return nn.Tanh()
                else:
                    raise ValueError(f"Unsupported mlp_activation: {mlp_activation}")

            encoder_layers: List[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in self.mlp_hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                if mlp_batch_norm:
                    encoder_layers.append(nn.BatchNorm1d(hidden_dim))
                encoder_layers.append(_make_activation())
                if mlp_dropout > 0:
                    encoder_layers.append(nn.Dropout(mlp_dropout))
                prev_dim = hidden_dim
            self.encoder: Optional[nn.Module] = nn.Sequential(*encoder_layers)

            for module in self.encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            context_dim = self.mlp_hidden_dims[-1]
        else:
            self.encoder = None
            context_dim = input_dim

        # Build normalizing flow based on specified type
        # Flow is conditioned on input embeddings (context_dim)
        # Zuko provides many pre-built architectures
        if flow_type == "GMM":
            self.net = zuko.flows.GMM(features=output_dim, context=context_dim, components=flow_components)
        elif flow_type == "NICE":
            self.net = zuko.flows.NICE(features=output_dim, context=context_dim, transforms=flow_transforms)
        elif flow_type == "RealNVP":
            self.net = zuko.flows.RealNVP(features=output_dim, context=context_dim, transforms=flow_transforms)
        elif flow_type == "NAF":
            self.net = zuko.flows.NAF(
                features=output_dim, context=context_dim, transforms=flow_transforms, signal=flow_signal
            )
        elif flow_type == "UNAF":
            self.net = zuko.flows.UNAF(
                features=output_dim, context=context_dim, transforms=flow_transforms, signal=flow_signal
            )
        elif flow_type == "NSF":
            self.net = zuko.flows.NSF(features=output_dim, context=context_dim, bins=flow_bins)
        elif flow_type == "BPF":
            self.net = zuko.flows.BPF(features=output_dim, context=context_dim, degree=flow_degree)
        else:
            raise ValueError(f"Unsupported flow_type: {flow_type}. Choose from {self.SUPPORTED_FLOW_TYPES}.")

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """
        Forward pass: Input embeddings → Conditional flow.

        Args:
            x: Input embeddings (conditioning context)
               Shape: [batch_size, input_dim] or [batch_size, ...]
               Can be passed as positional or keyword argument (for Skorch DataFrame compatibility)
            **kwargs: Additional keyword arguments. If x is None, will extract from kwargs

        Returns:
            Flow distribution conditioned on input embeddings
        """
        # Handle case where Skorch passes DataFrame as kwargs
        if x is None:
            # Skorch with DataFrames passes data as {'feature_0': tensor, 'feature_1': tensor, ...}
            # We need to concatenate them into a single tensor
            if kwargs:
                # Concatenate the per-column tensors in DataFrame column order.
                # Iterate values() (insertion order) rather than sorted(keys()) so the
                # conditioning context matches the array input path and is not scrambled
                # for names like 'feature_10' vs 'feature_2'.
                tensors = [v for v in kwargs.values() if isinstance(v, torch.Tensor)]
                if tensors:
                    # Each tensor is [batch_size] or [batch_size, 1], concatenate along feature dimension
                    # First ensure all are 2D
                    tensors_2d = [t.view(-1, 1) if t.dim() == 1 else t for t in tensors]
                    x = torch.cat(tensors_2d, dim=1)
                else:
                    raise ValueError("No input data provided to forward()")
            else:
                raise ValueError("No input data provided to forward()")

        # Flatten higher-dimensional inputs to [batch_size, input_dim]
        if x.dim() > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

        # Optionally encode the input through the MLP conditioner before the flow.
        # Dropout layers here stay active during MC-dropout uncertainty sampling.
        if self.encoder is not None:
            x = self.encoder(x)

        # Return the conditional flow distribution p(y | x)
        return self.net(x)


class BaseFlowHeadEstimator:
    """
    Base mixin for Flow Head estimators with hyperparameter optimization support.

    This class provides:
    - Hyperparameter space definition for Optuna
    - Default parameter values
    - Common functionality for probabilistic regression
    """

    def get_hyperparameter_space(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        y: Union[pd.Series, pd.DataFrame, npt.NDArray[Any]],
        trial: Trial,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Flow head.

        This method is called by MotherTuner during hyperparameter optimization.
        It suggests optimal values for:
        - Flow type (architecture choice)
        - Learning rate

        Args:
            X: Input features (used to determine input dimension)
            y: Target values (not used but required by interface)
            trial: Optuna trial object for suggesting hyperparameters
            prefix: Prefix for parameter names (default: "")

        Returns:
            Dictionary of suggested hyperparameters
        """
        suggested_params = {}

        # === FLOW ARCHITECTURE ===

        # Type of normalizing flow
        # GMM: Fast Gaussian mixture baseline
        # NICE: Fast default, additive coupling layers
        # RealNVP: Affine coupling layers, slightly more expressive than NICE
        # NAF: Neural autoregressive, expressive on molecular data
        # UNAF: Unconstrained monotonic networks, very expressive
        # NSF: Spline-based, flexible but needs tuning
        # BPF: Bernstein-polynomial, smooth monotonic transforms
        suggested_params[prefix + "flow_type"] = trial.suggest_categorical(
            prefix + "flow_type",
            ("GMM", "NICE", "RealNVP", "NAF", "UNAF", "NSF", "BPF"),
        )

        # Get the selected flow type (from current trial)
        selected_flow_type = suggested_params[prefix + "flow_type"]

        # === FLOW ARCHITECTURE HYPERPARAMETERS ===

        # Number of transformation layers (for coupling / autoregressive flows)
        if selected_flow_type in ("NICE", "RealNVP", "NAF", "UNAF"):
            suggested_params[prefix + "flow_transforms"] = trial.suggest_int(prefix + "flow_transforms", 2, 5)

        # Hidden signal dimension (for NAF / UNAF)
        if selected_flow_type in ("NAF", "UNAF"):
            suggested_params[prefix + "flow_signal"] = trial.suggest_int(prefix + "flow_signal", 8, 32)

        # Number of mixture components (for GMM)
        if selected_flow_type == "GMM":
            suggested_params[prefix + "flow_components"] = trial.suggest_int(prefix + "flow_components", 4, 16)

        # Number of spline bins (for NSF only)
        if selected_flow_type == "NSF":
            suggested_params[prefix + "flow_bins"] = trial.suggest_int(prefix + "flow_bins", 4, 16)

        # Polynomial degree (for BPF only)
        if selected_flow_type == "BPF":
            suggested_params[prefix + "flow_degree"] = trial.suggest_int(prefix + "flow_degree", 8, 32)

        # === MLP ENCODER (conditioner placed BEFORE the flow) ===
        # The MLP trunk gives the flow richer conditioning features and — with
        # dropout — provides MC-dropout epistemic uncertainty. It is tuned with the
        # SAME search space as the standalone MLP head (shared adaptive-sizing helper
        # plus dropout / activation / batch-norm), so the two are fully aligned.
        if isinstance(X, pd.DataFrame):
            input_dim = X.shape[1]
        else:
            input_dim = X.shape[1] if hasattr(X, "shape") else len(X[0])

        mlp_hidden_dims = _suggest_adaptive_hidden_dims(
            trial,
            input_dim,
            layers_key=prefix + "mlp_num_layers",
            width_key=prefix + "mlp_hidden_dim_1",
            max_layers=4,
        )
        suggested_params[prefix + "mlp_hidden_dims"] = mlp_hidden_dims
        suggested_params[prefix + "mlp_dropout"] = trial.suggest_float(prefix + "mlp_dropout", 0.0, 0.5, log=False)
        suggested_params[prefix + "mlp_batch_norm"] = trial.suggest_categorical(
            prefix + "mlp_batch_norm", (True, False)
        )
        suggested_params[prefix + "mlp_activation"] = trial.suggest_categorical(
            prefix + "mlp_activation", ("ReLU", "GELU", "LeakyReLU")
        )

        # === OPTIMIZATION HYPERPARAMETERS ===

        # Learning rate for optimizer
        # Range: 1e-5 to 1e-2 (log scale)
        # Flow models often need lower learning rates than standard networks
        suggested_params[prefix + "lr"] = trial.suggest_float(prefix + "lr", 1e-5, 1e-2, log=True)

        return suggested_params

    def default_parameters(self, prefix: str = "") -> Dict[str, Any]:
        """
        Return default hyperparameters for Flow head.

        These defaults provide a good starting point for most tasks:
        - NICE flow (fast default)
        - 3 transformation layers (good balance)
        - 8 spline bins (good balance for NSF)
        - A 2-layer MLP encoder ``[256, 128]`` with 0.1 dropout before the flow
        - Learning rate of 0.001

        Args:
            prefix: Prefix for parameter names (default: "")

        Returns:
            Dictionary of default parameters
        """
        return {
            prefix + "flow_type": "NICE",
            prefix + "flow_transforms": 3,
            prefix + "mlp_hidden_dims": [256, 128],
            prefix + "mlp_dropout": 0.1,
            prefix + "lr": 0.001,
        }


class FlowHeadRegressor(NeuralNetRegressor, BaseFlowHeadEstimator, AbstractMotherPipeline):
    """
    Flow Head Regressor with scikit-learn API via Skorch.

    This wrapper enables the Flow head to be used as a drop-in replacement for
    scikit-learn regressors with probabilistic predictions. It automatically handles:
    - Input/output dimension detection
    - Training loop with negative log-likelihood loss
    - Point predictions (mode/mean/median)
    - Full distribution sampling
    - Integration with scikit-learn pipelines
    - Hyperparameter optimization with Optuna

    The Flow head models p(y|x) using conditional normalizing flows, providing:
    - Flexible non-parametric density estimation
    - Uncertainty quantification
    - Sampling from predictive distribution

    Args:
        input_dim: Input feature dimension (required)
        output_dim: Output dimension (default: 1 for single-target regression)
        flow_type: Type of flow architecture (default: "NICE")
            Options: GMM, NICE, RealNVP, NAF, UNAF, NSF, BPF
        flow_transforms: Number of transformation layers (default: 3)
            Used by NICE, RealNVP, NAF, UNAF.
        flow_bins: Number of spline bins for NSF (default: 8)
        flow_degree: Polynomial degree for BPF (default: 16)
        flow_signal: Hidden signal dimension for NAF/UNAF (default: 16)
        flow_components: Number of mixture components for GMM (default: 8)
        mlp_hidden_dims: Hidden sizes for the MLP encoder placed *before* the flow.
            Default ``"auto"`` builds a reasonable 2-layer encoder ``[256, 128]`` so the
            standalone flow head has an MLP trunk and (with ``mlp_dropout`` > 0) MC-dropout
            uncertainty out of the box. Pass an explicit list to control the layers, or
            ``None`` / ``[]`` to condition the flow directly on the raw input (flow-alone,
            aleatoric uncertainty only). When an MLP is used it is defined exactly like the
            standalone :class:`MLPHeadRegressor`
            (``Linear -> BatchNorm -> activation -> Dropout`` per layer) with the flow
            attached afterwards, and — together with ``mlp_dropout`` > 0 — unlocks the
            same flow + MC-dropout uncertainty decomposition as the NODE flow head.
        mlp_dropout: Dropout rate for the MLP encoder (default: 0.1). Must be > 0 to
            obtain knowledge (epistemic) uncertainty via MC-dropout. Set to 0.0 for a
            deterministic encoder (aleatoric uncertainty only).
        mlp_activation: Activation for the MLP encoder (default: "ReLU").
        mlp_batch_norm: Whether to use batch norm in the MLP encoder (default: True).
        max_epochs: Maximum training epochs (default: 100)
        lr: Learning rate (default: 0.001)
        **kwargs: Additional arguments passed to NeuralNetRegressor

    Example:
        >>> from mother.ml.models.m_heads import FlowHeadRegressor
        >>> reg = FlowHeadRegressor(input_dim=20, output_dim=1, flow_type="NICE")
        >>> reg.fit(X_train, y_train)
        >>> predictions = reg.predict(X_test)  # Point predictions
        >>> samples = reg.predict_flow(X_test, num_samples=1000)  # Distribution
        >>> # Default encoder ([256, 128], dropout 0.1) -> flow + MC-dropout uncertainties
        >>> results = reg.predict_uncertainty(X_test)  # knowledge + data uncertainty
        >>> # Opt out of the MLP encoder for a pure flow (aleatoric only)
        >>> reg = FlowHeadRegressor(input_dim=20, mlp_hidden_dims=None)

    Note:
        The default loss function for flow models is negative log-likelihood (NLL).
        This is automatically handled by the flow distribution's log_prob method.
    """

    def __init__(
        self,
        input_dim: int = 1,  # Placeholder - automatically detected from data
        output_dim: int = 1,  # Placeholder - automatically detected from data
        flow_type: str = "NICE",
        flow_transforms: int = 3,
        flow_bins: int = 8,
        flow_degree: int = 16,
        flow_signal: int = 16,
        flow_components: int = 8,
        mlp_hidden_dims: Union[str, List[int], None] = "auto",
        mlp_dropout: float = 0.1,
        mlp_activation: str = "ReLU",
        mlp_batch_norm: bool = True,
        max_epochs: int = 100,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> None:
        # Flow models use negative log-likelihood loss by default
        # Don't pass criterion as a method reference - Skorch will call it during forward
        # We'll override get_loss instead

        # Resolve the "auto" architecture into a concrete, reasonable default encoder
        # (2 hidden layers). None / [] keep the flow conditioned on the raw input.
        if isinstance(mlp_hidden_dims, str):
            if mlp_hidden_dims != "auto":
                raise ValueError(f"mlp_hidden_dims string must be 'auto', got {mlp_hidden_dims!r}.")
            mlp_hidden_dims = [256, 128]

        # ── Sensible training defaults ──────────────────────────────────
        # AdamW provides proper weight-decay decoupling for better generalisation
        kwargs.setdefault("optimizer", torch.optim.AdamW)
        kwargs.setdefault("optimizer__weight_decay", 1e-2)

        # Validation split for early stopping / LR scheduling (user can override)
        kwargs.setdefault("train_split", ValidSplit(cv=0.1))

        # Build default callbacks: DimensionSetter + EarlyStopping + LR scheduler
        callbacks = list(kwargs.get("callbacks", []))
        if not any(isinstance(cb, DimensionSetter) for cb in callbacks):
            callbacks.append(DimensionSetter())
        if not any(isinstance(cb, EarlyStopping) for cb in callbacks):
            callbacks.append(EarlyStopping(patience=20, monitor="valid_loss"))
        if not any(isinstance(cb, LRScheduler) for cb in callbacks):
            callbacks.append(
                LRScheduler(
                    policy="ReduceLROnPlateau",
                    monitor="valid_loss",
                    patience=7,
                    factor=0.5,
                )
            )
        kwargs["callbacks"] = callbacks

        # Initialize Skorch regressor
        super().__init__(
            module=FlowHead,
            module__input_dim=input_dim,
            module__output_dim=output_dim,
            module__flow_type=flow_type,
            module__flow_transforms=flow_transforms,
            module__flow_bins=flow_bins,
            module__flow_degree=flow_degree,
            module__flow_signal=flow_signal,
            module__flow_components=flow_components,
            module__mlp_hidden_dims=mlp_hidden_dims,
            module__mlp_dropout=mlp_dropout,
            module__mlp_activation=mlp_activation,
            module__mlp_batch_norm=mlp_batch_norm,
            max_epochs=max_epochs,
            lr=lr,
            **kwargs,
        )

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters, implementing AbstractMotherPipeline requirement with proper MRO."""
        # Use super() to follow MRO: NeuralNetRegressor -> BaseFlowHeadEstimator -> AbstractMotherPipeline
        params: dict = super().get_params(deep=deep)
        # Re-expose module__<head_param> as bare constructor arguments so that
        # sklearn.clone() (which rebuilds via __class__(**get_params())) preserves the
        # flow-head configuration instead of silently reverting to constructor defaults.
        head_params: List[str] = [
            "input_dim",
            "output_dim",
            "flow_type",
            "flow_transforms",
            "flow_bins",
            "flow_degree",
            "flow_signal",
            "flow_components",
            "mlp_hidden_dims",
            "mlp_dropout",
            "mlp_activation",
            "mlp_batch_norm",
        ]
        for name in head_params:
            module_key = f"module__{name}"
            if hasattr(self, module_key):
                params[name] = getattr(self, module_key)
        # Remove module__* parameters to avoid conflicts during sklearn cloning
        params_to_remove = [key for key in params.keys() if key.startswith("module__")]
        for key in params_to_remove:
            params.pop(key, None)
        params.pop("module", None)  # Also remove 'module' itself
        return params

    def set_params(self, **params: Any) -> "FlowHeadRegressor":
        """Set parameters, implementing AbstractMotherPipeline requirement with proper MRO."""
        # List of Flow Head parameters that need to be synced to module
        head_params: List[str] = [
            "input_dim",
            "output_dim",
            "flow_type",
            "flow_transforms",
            "flow_bins",
            "flow_degree",
            "flow_signal",
            "flow_components",
            "mlp_hidden_dims",
            "mlp_dropout",
            "mlp_activation",
            "mlp_batch_norm",
        ]

        # For each head parameter being set, also set the module__ version
        # and remove the bare name so that skorch doesn't reject it.
        for param_name in head_params:
            if param_name in params:
                params[f"module__{param_name}"] = params.pop(param_name)

        # Use super() to follow MRO: NeuralNetRegressor -> BaseFlowHeadEstimator -> AbstractMotherPipeline
        return super().set_params(**params)

    def __sklearn_clone__(self) -> "FlowHeadRegressor":
        """Custom sklearn cloning to avoid passing 'module' parameter."""
        params: dict = self.get_params(deep=False)
        # Remove 'module' if present (Skorch adds it automatically)
        params.pop("module", None)
        return self.__class__(**params)

    def get_loss(self, y_pred: Any, y_true: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Negative log-likelihood loss for flow models.

        Args:
            y_pred: Flow distribution from model forward pass
            y_true: Ground truth targets

        Returns:
            Negative log-likelihood (mean over batch)
        """
        # Ensure y_true has the right shape
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(-1)

        # Compute negative log-likelihood
        # log_prob returns log p(y|x), we want to minimize -log p(y|x)
        log_prob = y_pred.log_prob(y_true)
        return -log_prob.mean()

    def fit(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        y: Optional[npt.NDArray[np.float32]] = None,
        **fit_params: Any,
    ) -> "FlowHeadRegressor":
        """Fit the model, ensuring correct dtype."""
        # Convert to float32 to match PyTorch's default dtype
        if isinstance(X, np.ndarray) and X.dtype == np.float64:
            X = X.astype(np.float32)
        if isinstance(y, np.ndarray) and y.dtype == np.float64:
            y = y.astype(np.float32)
        return super().fit(X, y, **fit_params)

    def predict_flow(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float32]], num_samples: int = 1000
    ) -> npt.NDArray[np.float32]:
        """
        Generate samples from the predictive distribution.

        This method provides the full predictive distribution by sampling from
        the conditional flow p(y|x) for each input.

        Args:
            X: Input features [n_samples, n_features]
            num_samples: Number of samples to draw per input (default: 1000)

        Returns:
            Samples from predictive distribution [n_samples, num_samples, output_dim]

        Example:
            >>> samples = reg.predict_flow(X_test, num_samples=1000)
            >>> # samples.shape = [100, 1000, 1] for 100 test samples
            >>> # Compute statistics
            >>> mean_pred = samples.mean(axis=1)
            >>> std_pred = samples.std(axis=1)
            >>> median_pred = np.median(samples, axis=1)
        """
        self.module_.eval()
        with torch.no_grad():
            # Convert input to tensor
            if not isinstance(X, torch.Tensor):
                # Handle DataFrame
                if isinstance(X, pd.DataFrame):
                    X = X.values
                X = torch.tensor(X, dtype=torch.float32)

            # Move to same device as model
            device = next(self.module_.parameters()).device
            X = X.to(device)

            # Get flow distribution
            dist = self.module_(X)

            # Sample from distribution
            samples = dist.sample((num_samples,))  # [num_samples, batch_size, output_dim]

            # Transpose to [batch_size, num_samples, output_dim]
            samples = samples.permute(1, 0, 2)

            # Convert to numpy
            return samples.cpu().numpy()

    def predict(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float32]], num_samples: int = 100
    ) -> npt.NDArray[np.float32]:
        """
        Generate point predictions (maximum log-likelihood point).

        For flow models, we find the maximum likelihood estimate by:
        1. Sampling from the distribution
        2. Evaluating log probability for each sample
        3. Returning the sample with highest log probability (MAP estimate)

        Args:
            X: Input features [n_samples, n_features]
            num_samples: Number of samples to draw for finding MAP estimate (default: 100)
                        Higher values = more accurate but slower

        Returns:
            Point predictions [n_samples, output_dim] or [n_samples] if output_dim=1
        """
        self.module_.eval()
        with torch.no_grad():
            # Convert input to tensor
            if not isinstance(X, torch.Tensor):
                # Handle DataFrame
                if isinstance(X, pd.DataFrame):
                    X_np = X.values
                else:
                    X_np = X
                X_tensor = torch.tensor(X_np, dtype=torch.float32)
            else:
                X_tensor = X

            # Move to same device as model
            device = next(self.module_.parameters()).device
            X_tensor = X_tensor.to(device)

            # Get flow distribution
            dist = self.module_(X_tensor)

            # Use shared utility to compute mode
            mode_predictions, _ = compute_flow_mode_and_uncertainty(dist, num_samples)

            # Convert to numpy
            predictions = mode_predictions.cpu().numpy()

            # Flatten if single output dimension
            if predictions.shape[1] == 1:
                predictions = predictions.flatten()

            return predictions

    def _flow_has_mc_dropout(self) -> bool:
        """Whether this flow head has an MLP encoder with active dropout.

        Returns ``True`` only when an MLP encoder was configured (``mlp_hidden_dims``)
        *and* ``mlp_dropout > 0`` — the condition under which MC-dropout can provide
        epistemic (knowledge) uncertainty for the standalone flow head, mirroring how
        the NODE flow head derives epistemic uncertainty from its trunk dropout.
        """
        module = getattr(self, "module_", None)
        if module is None:
            return False
        return getattr(module, "encoder", None) is not None and getattr(module, "mlp_dropout", 0.0) > 0

    def _to_input_tensor(self, X: Union[pd.DataFrame, npt.NDArray[np.float32]]) -> torch.Tensor:
        """Convert ``X`` to a float32 tensor on the module's device."""
        if isinstance(X, torch.Tensor):
            X_tensor = X
        else:
            X_np = X.values if isinstance(X, pd.DataFrame) else X
            if isinstance(X_np, np.ndarray) and X_np.dtype == np.float64:
                X_np = X_np.astype(np.float32)
            X_tensor = torch.tensor(X_np, dtype=torch.float32)
        device = next(self.module_.parameters()).device
        return X_tensor.to(device)

    def predict_with_combined_uncertainty(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float32]],
        num_mc_samples: int = 30,
        num_flow_samples: int = 100,
        return_all: bool = False,
    ) -> Union[Dict[str, Any], tuple]:
        """Decompose predictive uncertainty into epistemic + aleatoric parts.

        Mirrors the NODE flow head's information-theoretic (BALD) decomposition for the
        standalone flow head. Requires an MLP encoder with ``mlp_dropout > 0``; the
        ``num_mc_samples`` dropout passes are treated as an ensemble of flows
        ``{p_t(y|x)}`` and, with differential entropies estimated by sampling
        (``H[p] = -E_{y~p}[log p(y)] ~= -(1/S) sum_s log p(y_s)``):

            data  (aleatoric)  = (1/T) sum_t H[p_t]      (expected entropy)
            total              = H[(1/T) sum_t p_t]      (mixture entropy)
            knowledge (epist.) = total - data            (mutual information >= 0)

        ``data`` / ``total`` are differential entropies (nats) and may be negative for
        peaked flows; the knowledge (mutual-information) term is always >= 0 and
        ``total == data + knowledge`` holds exactly.

        When no MLP-encoder dropout is configured there is a single flow, so knowledge
        uncertainty is undefined (``None``) and data uncertainty is the flow's
        differential entropy ``H[p]``.

        Args:
            X: Input features.
            num_mc_samples: Number of MC-dropout forward passes ``T`` (default 30).
                The mixture-entropy term is ``O(T^2 * num_flow_samples)``.
            num_flow_samples: Number of samples ``S`` drawn from each flow (default 100).
            return_all: If True, return a dict with per-pass diagnostics; otherwise a
                ``(predictions, knowledge_uncertainty, data_uncertainty)`` tuple.

        Returns:
            Either a tuple ``(predictions, knowledge_uncertainty, data_uncertainty)`` or,
            when ``return_all=True``, a dict with keys ``predictions``,
            ``knowledge_uncertainty``, ``data_uncertainty``, ``total_uncertainty``,
            ``mc_means``, ``mc_uncertainties`` (per-pass entropy) and ``mc_stds`` (alias).
        """
        X_tensor = self._to_input_tensor(X)
        output_dim = getattr(self.module_, "output_dim", 1)

        # ── Single flow (no MC-dropout): aleatoric differential entropy only ──
        if not self._flow_has_mc_dropout():
            self.module_.eval()
            with torch.no_grad():
                dist = self.module_(X_tensor)
                samples = dist.sample(torch.Size([num_flow_samples]))  # (S, N, D)
                log_p = dist.log_prob(samples)  # (S, N)
                data = -log_p.mean(dim=0)  # (N,)
                predictions_t = samples.mean(dim=0)  # (N, D)
            predictions = predictions_t.cpu().numpy()
            data_uncertainty = data.cpu().numpy()
            if output_dim == 1:
                predictions = predictions.flatten()
                data_uncertainty = data_uncertainty.flatten()
            if return_all:
                return {
                    "predictions": predictions,
                    "knowledge_uncertainty": None,
                    "data_uncertainty": data_uncertainty,
                    "total_uncertainty": data_uncertainty,
                    "mc_means": None,
                    "mc_uncertainties": None,
                    "mc_stds": None,
                }
            return predictions, None, data_uncertainty

        # ── Flow + MC-dropout: BALD entropy decomposition ──
        # Enable MC-dropout WITHOUT enabling BatchNorm training: keep the whole model
        # in eval() (so BatchNorm keeps using its running statistics) and switch ON
        # only the nn.Dropout layers inside the MLP encoder.
        model = self.module_
        model.eval()
        for _m in model.modules():
            if isinstance(_m, nn.Dropout):
                _m.train()

        dists: List[Any] = []
        samples_list: List[torch.Tensor] = []
        try:
            with torch.no_grad():
                for _ in range(num_mc_samples):
                    dist = model(X_tensor)  # flow distribution for this dropout pass
                    dists.append(dist)
                    samples_list.append(dist.sample(torch.Size([num_flow_samples])))  # (S, N, D)
        finally:
            model.eval()

        T = num_mc_samples
        log_T = float(np.log(T))

        with torch.no_grad():
            per_source_mix = []  # each (S, N): log p-bar(y_{t,s})
            per_source_self = []  # each (S, N): log p_t(y_{t,s})
            for t in range(T):
                samp_t = samples_list[t]  # (S, N, D)
                # log p_{t'}(y_{t,s}) for every t' -> (T, S, N)
                lp_stack = torch.stack([dists[tp].log_prob(samp_t) for tp in range(T)], dim=0)
                per_source_mix.append(torch.logsumexp(lp_stack, dim=0) - log_T)  # (S, N)
                per_source_self.append(lp_stack[t])  # (S, N)

            mix_all = torch.stack(per_source_mix, dim=0)  # (T, S, N)
            self_all = torch.stack(per_source_self, dim=0)  # (T, S, N)

            total = -mix_all.mean(dim=(0, 1))  # (N,)  mixture entropy
            data = -self_all.mean(dim=(0, 1))  # (N,)  expected entropy
            per_pass_entropy = -self_all.mean(dim=1)  # (T, N)
            samp_stack = torch.stack(samples_list, dim=0)  # (T, S, N, D)
            per_pass_mean = samp_stack.mean(dim=1)  # (T, N, D)
            predictions_t = samp_stack.mean(dim=(0, 1))  # (N, D)

        data_uncertainty = data.detach().cpu().numpy()
        total_raw = total.detach().cpu().numpy()
        predictions = predictions_t.detach().cpu().numpy()

        # Knowledge = mutual information = total - data. Clamp at 0 (Monte-Carlo noise
        # can push the Jensen gap slightly negative), then re-derive total so the
        # additive identity total == data + knowledge holds exactly.
        knowledge_uncertainty = np.maximum(total_raw - data_uncertainty, 0.0)
        total_uncertainty = data_uncertainty + knowledge_uncertainty

        mc_unc = per_pass_entropy.unsqueeze(-1).detach().cpu().numpy()  # (T, N, 1)
        mc_means = per_pass_mean.detach().cpu().numpy()  # (T, N, D)

        data_uncertainty = data_uncertainty.flatten()
        knowledge_uncertainty = knowledge_uncertainty.flatten()
        total_uncertainty = total_uncertainty.flatten()
        if output_dim == 1:
            predictions = predictions.flatten()

        if return_all:
            return {
                "predictions": predictions,
                "knowledge_uncertainty": knowledge_uncertainty,
                "data_uncertainty": data_uncertainty,
                "total_uncertainty": total_uncertainty,
                "mc_means": mc_means,
                "mc_uncertainties": mc_unc,
                "mc_stds": mc_unc,  # legacy alias
            }
        return predictions, knowledge_uncertainty, data_uncertainty

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        return_quantiles: bool = False,
        quantiles: List[float] = DEFAULT_QUANTILES,
        uncertainty_for_opt: bool = False,
        num_samples: int = 1000,
        num_mc_samples: int = 30,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, pd.Series, tuple[pd.DataFrame, npt.NDArray[np.float32]]]:
        """
        Predict with uncertainty estimation (Mother framework compatible).

        The flow head reports uncertainty in one of two regimes:

        * **Flow alone** (no MLP encoder, or ``mlp_dropout == 0``): uncertainty is purely
          aleatoric and comes from the conditional flow ``p(y|x)`` via negative
          log-likelihood: ``data_uncertainty = -log_prob(mode)``, ``knowledge_uncertainty``
          is ``None``. This is the default, backward-compatible behaviour.
        * **Flow + MLP encoder with dropout** (``mlp_hidden_dims`` set and
          ``mlp_dropout > 0``): the same flow + MC-dropout decomposition as the NODE flow
          head becomes available — ``data_uncertainty`` (expected differential entropy,
          aleatoric), ``knowledge_uncertainty`` (mutual information, epistemic) and their
          sum ``total_uncertainty`` (see :meth:`predict_with_combined_uncertainty`).

        Because the flow is a full probabilistic model, this head can also return genuine
        predictive quantiles sampled from the distribution (unlike dropout-only heads
        which only expose a mean and std).

        Args:
            X: Input features.
            return_quantiles: If True, also return quantile predictions sampled from the
                flow distribution (default False).
            quantiles: List of quantiles to compute. Default ``[0.25, 0.5, 0.75]``
                (``DEFAULT_QUANTILES``).
            uncertainty_for_opt: If True, return only ``total_uncertainty`` as a Series
                for optimisation / active learning (default False).
            num_samples: Number of samples drawn from the flow for the mode and quantiles
                (default 1000).
            num_mc_samples: Number of MC-dropout forward passes used when an MLP encoder
                with dropout is present (default 30). Ignored for the flow-alone regime.
            **kwargs: Additional arguments (ignored).

        Returns:
            Union[pd.DataFrame, pd.Series, tuple[pd.DataFrame, np.ndarray]]:
                - Default: DataFrame with columns ``pred``, ``mean_predictions``,
                  ``knowledge_uncertainty`` (``None`` unless MC-dropout is active),
                  ``data_uncertainty`` and ``total_uncertainty``.
                - If ``return_quantiles=True``: ``(DataFrame, quantile_array)`` where the
                  array has shape ``(n_samples, n_quantiles)`` (single target) or
                  ``(n_samples, n_quantiles, output_dim)`` (multi-target).
                - If ``uncertainty_for_opt=True``: ``pd.Series`` of ``total_uncertainty``.

        Example:
            >>> reg = FlowHeadRegressor(input_dim=10, output_dim=1)
            >>> reg.fit(X_train, y_train)
            >>> results = reg.predict_uncertainty(X_test, num_samples=1000)
            >>> results, q = reg.predict_uncertainty(X_test, return_quantiles=True)
            >>> # Flow + MLP dropout -> epistemic + aleatoric decomposition
            >>> reg = FlowHeadRegressor(input_dim=10, mlp_hidden_dims=[64], mlp_dropout=0.1)
            >>> reg.fit(X_train, y_train)
            >>> results = reg.predict_uncertainty(X_test)  # knowledge_uncertainty populated
        """
        index = X.index if isinstance(X, pd.DataFrame) else None

        # Defensive copy; ensure DEFAULT_QUANTILES are included for consistency.
        quantiles = list(quantiles)
        for q in DEFAULT_QUANTILES:
            if q not in quantiles:
                quantiles.append(q)
        quantiles = sorted(quantiles)

        X_tensor = self._to_input_tensor(X)

        # Quantiles are sampled from the (dropout-off) conditional flow p(y|x), matching
        # the NODE flow head convention of drawing quantiles from a single eval() pass.
        quantile_predictions = None
        if return_quantiles:
            self.module_.eval()
            with torch.no_grad():
                dist = self.module_(X_tensor)
                samples = dist.sample(torch.Size([num_samples]))  # (num_samples, N, output_dim)
                q_stack = torch.stack([torch.quantile(samples, q, dim=0) for q in quantiles], dim=1)
                quantile_predictions = q_stack.cpu().numpy()
                if quantile_predictions.shape[2] == 1:
                    quantile_predictions = quantile_predictions.squeeze(axis=2)

        if self._flow_has_mc_dropout():
            # Flow + MC-dropout: full epistemic/aleatoric decomposition (like NODE flow).
            stats = self.predict_with_combined_uncertainty(
                X,
                num_mc_samples=num_mc_samples,
                num_flow_samples=min(num_samples, 100),
                return_all=True,
            )
            predictions_col = _prepare_for_dataframe(stats["predictions"])
            results = pd.DataFrame(
                {
                    "pred": predictions_col,
                    "mean_predictions": predictions_col,
                    "knowledge_uncertainty": _prepare_for_dataframe(stats["knowledge_uncertainty"]),
                    "data_uncertainty": stats["data_uncertainty"],
                    "total_uncertainty": stats["total_uncertainty"],
                },
                index=index,
            )
        else:
            # Flow alone: aleatoric NLL of the mode (backward-compatible behaviour).
            self.module_.eval()
            with torch.no_grad():
                dist = self.module_(X_tensor)
                mode_pred, data_uncertainty = compute_flow_mode_and_uncertainty(dist, num_samples)
                mode_pred = mode_pred.cpu().numpy()
                data_uncertainty = data_uncertainty.cpu().numpy()

            predictions_col = _prepare_for_dataframe(mode_pred)
            results = pd.DataFrame(
                {
                    "pred": predictions_col,
                    "mean_predictions": predictions_col,
                    "knowledge_uncertainty": None,  # No dropout in standalone head
                    "data_uncertainty": data_uncertainty,  # Negative log-likelihood (aleatoric)
                    "total_uncertainty": data_uncertainty,  # Only source of uncertainty
                },
                index=index,
            )

        if uncertainty_for_opt:
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
        """Predict quantiles by sampling the learned conditional flow ``p(y|x)``.

        Compatible with the TabPFN / RandomForest / NODE flow-head interface. Quantiles
        are always drawn from the (dropout-off) flow distribution.

        Args:
            X: Input features.
            quantiles: List of quantiles in ``[0, 1]``. If None, uses
                ``[0.025, 0.25, 0.5, 0.75, 0.975]``.
            num_samples: Number of flow samples used to estimate the quantiles
                (default 200).

        Returns:
            Array of shape ``(n_samples, n_quantiles)`` for single-target or
            ``(n_samples, n_quantiles, output_dim)`` for multi-target regression.
        """
        if quantiles is None:
            quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

        invalid = [q for q in quantiles if not 0 <= q <= 1]
        if invalid:
            raise ValueError(f"Quantiles must be in [0, 1]. Got invalid values: {invalid}")
        quantiles = sorted(quantiles)

        # predict_uncertainty internally appends DEFAULT_QUANTILES; request the union then
        # filter back to only the user-requested columns.
        merged = sorted(set(quantiles) | set(DEFAULT_QUANTILES))

        _, all_quantile_predictions = self.predict_uncertainty(
            X,
            num_samples=num_samples,
            return_quantiles=True,
            quantiles=list(merged),
        )

        user_indices = [merged.index(q) for q in quantiles]
        if all_quantile_predictions.ndim == 2:
            return all_quantile_predictions[:, user_indices]
        return all_quantile_predictions[:, user_indices, :]
