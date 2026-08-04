"""
MLP Head Architectures for Regression and Classification.

Deterministic multi-layer perceptron heads that can be used standalone or as part
of larger models like NODE. Provides an ``nn.Module`` (``MLPHead``) plus Skorch /
scikit-learn estimator wrappers with Optuna hyperparameter support and MC-dropout
uncertainty.

Usage Examples:

    # MLP Head for regression
    from mother.ml.models.m_mlp import MLPHeadRegressor

    reg = MLPHeadRegressor(input_dim=512, output_dim=1, hidden_dims=[256, 128])
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    # MLP Head for classification
    from mother.ml.models.m_mlp import MLPHeadClassifier

    clf = MLPHeadClassifier(input_dim=512, output_dim=3, hidden_dims=[256, 128])
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

References:
    Uncertainty decomposition (MC-dropout heads):
        The classification MC-dropout split (predictive entropy = expected entropy +
        mutual information) follows the CatBoost virtual-ensemble decomposition of
        Malinin, Prokhorenkova & Ustimenko (2021), "Uncertainty in Gradient Boosting
        via Ensembles" (arXiv:2006.10562) — itself the BALD mutual-information split of
        Houlsby et al. (2011) approximated with MC-dropout (Gal, Islam & Ghahramani,
        2017).

Authors: Julian Qian, Sergey Popov
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from optuna import Trial
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit

from mother.ml.core import AbstractMotherPipeline
from mother.ml.models.m_head_utils import (
    DEFAULT_QUANTILES,
    DimensionSetter,
    _prepare_for_dataframe,
    _suggest_adaptive_hidden_dims,
)

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
        >>> from mother.ml.models.m_mlp import MLPHeadRegressor
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
            return_quantiles: Not supported for this classifier estimator.
                Passing True raises ``ValueError``.
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
            ValueError: If ``return_quantiles=True`` (classification here does not expose quantiles).
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
        >>> from mother.ml.models.m_mlp import MLPHeadClassifier
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
        - y dtype follows criterion requirements:
          * CrossEntropyLoss -> int64
          * BCEWithLogitsLoss -> float32
        """
        if isinstance(X, np.ndarray) and X.dtype == np.float64:
            X = X.astype(np.float32)
        if y is not None and isinstance(y, np.ndarray):
            criterion_obj = getattr(self, "criterion", nn.CrossEntropyLoss)
            criterion_cls = criterion_obj if isinstance(criterion_obj, type) else type(criterion_obj)
            if issubclass(criterion_cls, nn.BCELoss) and not issubclass(criterion_cls, nn.BCEWithLogitsLoss):
                raise ValueError(
                    "MLPHeadClassifier does not support nn.BCELoss because this model outputs logits. "
                    "Use nn.BCEWithLogitsLoss instead."
                )

            use_float_targets = issubclass(criterion_cls, nn.BCEWithLogitsLoss)

            if use_float_targets:
                if not np.issubdtype(y.dtype, np.floating) or y.dtype == np.float64:
                    y = y.astype(np.float32)
            else:
                if not np.issubdtype(y.dtype, np.int64):
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
                "Quantiles are not available for MLPHeadClassifier. This estimator models "
                "classification uncertainty via MC-dropout entropy, not a calibrated predictive distribution. "
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

        use_sigmoid = isinstance(self.criterion_, nn.BCEWithLogitsLoss)

        probabilities = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.module_(X_tensor)
                probs = torch.sigmoid(logits) if use_sigmoid else torch.softmax(logits, dim=1)
                probabilities.append(probs.cpu().numpy())

        # Return to eval mode
        self.module_.eval()

        # Stack probabilities: shape (num_samples, n_datapoints, n_classes)
        probabilities = np.stack(probabilities, axis=0)

        # Predictive (mean) distribution across MC dropout passes.
        mean_probs = probabilities.mean(axis=0)  # shape: (n_datapoints, n_classes)

        if use_sigmoid:
            # Multi-label path: independent Bernoulli probabilities per label.
            eps = 1e-12

            def _binary_entropy(p: np.ndarray) -> np.ndarray:
                p = np.clip(p, eps, 1.0 - eps)
                return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

            mean_pred = (mean_probs >= 0.5).astype(np.int64)

            # Average per-label uncertainties into one scalar per sample.
            total_uncertainty = _binary_entropy(mean_probs).mean(axis=1)
            per_pass_entropy = _binary_entropy(probabilities).mean(axis=2)  # (num_samples, n_datapoints)
            data_uncertainty = per_pass_entropy.mean(axis=0)
            knowledge_uncertainty = total_uncertainty - data_uncertainty

            # Expose full per-label predictive probabilities.
            mean_predictions = _prepare_for_dataframe(mean_probs.astype(np.float32))
            pred_out = _prepare_for_dataframe(mean_pred.astype(np.float32))
        else:
            # Most likely class — map the argmax *index* back to the real class label
            # via skorch's ``classes_`` so that non-0..C-1 / string label sets are
            # preserved (consistent with ``predict()`` and the other Mother classifiers).
            pred_out = self.classes_[mean_probs.argmax(axis=1)]

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
                "pred": pred_out,
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
