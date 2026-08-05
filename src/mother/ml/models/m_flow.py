"""
Flow Head Architecture for Probabilistic Regression.

Flow-based (conditional normalizing flow) head for probabilistic regression, usable
standalone or as part of larger models like NODE. Provides an ``nn.Module``
(``FlowHead``) plus a Skorch / scikit-learn regressor wrapper with Optuna
hyperparameter support, full-distribution sampling, quantiles and an
information-theoretic (BALD) uncertainty decomposition.

Usage Examples:

    # Flow Head for probabilistic regression
    from mother.ml.models.m_flow import FlowHeadRegressor

    reg = FlowHeadRegressor(input_dim=512, output_dim=1, flow_type="NSF")
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)  # Point predictions
    samples = reg.predict_flow(X_test, num_samples=1000)  # Full distribution

References:
    NodeFlow Architecture (flow head):
        Wielopolski, P., Furman, O., & Zięba, M. (2024).
        NodeFlow: Towards End-to-end Flexible Probabilistic Regression on Tabular Data.
        Entropy, 26(7), 593.
        https://doi.org/10.3390/e26070593

    Uncertainty decomposition (flow + MC-dropout):
        The flow-head regression analogue (differential-entropy BALD over an
        MC-dropout flow ensemble, entropy estimated by sampling) corresponds to the
        ``NFlows Out`` method of Berry & Meger (2023); see ``m_node.py``
        ``predict_with_combined_uncertainty`` for the full attribution.

Authors: Julian Qian, Sergey Popov
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from optuna import Trial
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit

try:
    import zuko
except ModuleNotFoundError:  # pragma: no cover - zuko is only needed for flow heads
    zuko = None  # type: ignore[assignment]

from mother.ml.core import AbstractMotherPipeline
from mother.ml.models.m_head_utils import (
    DEFAULT_QUANTILES,
    DimensionSetter,
    _prepare_for_dataframe,
    _suggest_adaptive_hidden_dims,
    compute_flow_mode_and_uncertainty,
)

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
        mlp_activation: Activation for the MLP encoder (default: "GELU"). One of
            "ReLU", "GELU", "LeakyReLU", "ELU", "SiLU", "Tanh". GELU is a smooth
            default that pairs well with the downstream normalising flow.
        mlp_norm: Normalisation applied inside each MLP-encoder block (default:
            "batch"). One of "batch" (BatchNorm1d), "layer" (LayerNorm) or "none".
            Only has an effect when ``mlp_hidden_dims`` is set. The block layout is
            ``Linear -> Norm -> activation -> Dropout``. "batch" mirrors the standalone
            :class:`MLPHead`; prefer "layer" for wide pretrained-embedding inputs
            (e.g. CheMeleon) or small / variable batch sizes, where BatchNorm running
            statistics get noisy; "none" disables normalisation.
    """

    SUPPORTED_FLOW_TYPES = ("GMM", "NICE", "RealNVP", "NAF", "UNAF", "NSF", "BPF")

    @staticmethod
    def _move_nested_tensors_to_device(obj: Any, device: torch.device, visited: Optional[set[int]] = None) -> Any:
        """Recursively move plain tensors inside nested objects to ``device``.

        This is primarily for distribution objects returned by zuko flows, where
        some tensors may be stored as plain attributes rather than registered
        module parameters/buffers.
        """
        if visited is None:
            visited = set()

        oid = id(obj)
        if oid in visited:
            return obj
        visited.add(oid)

        if isinstance(obj, torch.Tensor):
            if obj.device != device:
                return obj.to(device)
            return obj

        if isinstance(obj, nn.Parameter):
            return obj

        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                obj[k] = FlowHead._move_nested_tensors_to_device(v, device, visited)
            return obj

        if isinstance(obj, list):
            for i, v in enumerate(obj):
                obj[i] = FlowHead._move_nested_tensors_to_device(v, device, visited)
            return obj

        if isinstance(obj, tuple):
            return tuple(FlowHead._move_nested_tensors_to_device(v, device, visited) for v in obj)

        if hasattr(obj, "__dict__"):
            for attr_name, attr_val in list(vars(obj).items()):
                try:
                    new_val = FlowHead._move_nested_tensors_to_device(attr_val, device, visited)
                    if new_val is not attr_val:
                        setattr(obj, attr_name, new_val)
                except Exception:
                    continue
        return obj

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
        mlp_activation: str = "GELU",
        mlp_norm: str = "batch",
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
        self.mlp_norm = mlp_norm

        # === OPTIONAL MLP ENCODER (conditioner) ===
        # When hidden dims are given, the flow is conditioned on an MLP embedding
        # instead of the raw input. Dropout layers inside this encoder are what make
        # MC-dropout epistemic uncertainty possible for the standalone flow head.
        # The per-layer block layout (Linear -> Norm -> activation -> Dropout) mirrors
        # the standalone ``MLPHead``; ``mlp_norm`` selects BatchNorm / LayerNorm / none.
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
                if mlp_norm == "batch":
                    encoder_layers.append(nn.BatchNorm1d(hidden_dim))
                elif mlp_norm == "layer":
                    encoder_layers.append(nn.LayerNorm(hidden_dim))
                elif mlp_norm != "none":
                    raise ValueError(f"Unsupported mlp_norm: {mlp_norm!r}. Choose 'batch', 'layer' or 'none'.")
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
        dist = self.net(x)
        # Some zuko distribution internals are plain tensors created lazily.
        # Keep them aligned with the context device used for this forward pass.
        self._move_nested_tensors_to_device(dist, x.device)
        return dist


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
        suggested_params[prefix + "mlp_dropout"] = trial.suggest_float(prefix + "mlp_dropout", 0.0, 0.3, log=False)
        suggested_params[prefix + "mlp_norm"] = trial.suggest_categorical(
            prefix + "mlp_norm", ("batch", "layer", "none")
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
        - A single-layer MLP encoder ``[512]`` with 0.1 dropout before the flow
        - Learning rate of 0.001

        Args:
            prefix: Prefix for parameter names (default: "")

        Returns:
            Dictionary of default parameters
        """
        return {
            prefix + "flow_type": "NICE",
            prefix + "flow_transforms": 3,
            prefix + "mlp_hidden_dims": [512],
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
            Default ``"auto"`` builds a single standard encoder layer ``[512]`` so the
            standalone flow head has an MLP trunk and (with ``mlp_dropout`` > 0) MC-dropout
            uncertainty out of the box. Pass an explicit list to control the layers, or
            ``None`` / ``[]`` to condition the flow directly on the raw input (flow-alone,
            aleatoric uncertainty only). When an MLP is used it is defined exactly like the
            standalone :class:`MLPHeadRegressor`
            (``Linear -> Norm -> activation -> Dropout`` per layer, ``Norm`` selected by
            ``mlp_norm``) with the flow attached afterwards, and — together with
            ``mlp_dropout`` > 0 — unlocks the
            same flow + MC-dropout uncertainty decomposition as the NODE flow head.
        mlp_dropout: Dropout rate for the MLP encoder (default: 0.1). Must be > 0 to
            obtain knowledge (epistemic) uncertainty via MC-dropout. Set to 0.0 for a
            deterministic encoder (aleatoric uncertainty only).
        mlp_activation: Activation for the MLP encoder (default: "GELU").
        mlp_norm: Normalisation inside the MLP encoder: "batch", "layer" or "none"
            (default: "batch"). Use "layer" for wide pretrained-embedding inputs
            (e.g. CheMeleon) or small batches where BatchNorm statistics are noisy.
        max_epochs: Maximum training epochs (default: 100)
        lr: Learning rate (default: 0.001)
        **kwargs: Additional arguments passed to NeuralNetRegressor

    Example:
        >>> from mother.ml.models.m_flow import FlowHeadRegressor
        >>> reg = FlowHeadRegressor(input_dim=20, output_dim=1, flow_type="NICE")
        >>> reg.fit(X_train, y_train)
        >>> predictions = reg.predict(X_test)  # Point predictions
        >>> samples = reg.predict_flow(X_test, num_samples=1000)  # Distribution
        >>> # Default encoder ([512], dropout 0.1) -> flow + MC-dropout uncertainties
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
        mlp_activation: str = "GELU",
        mlp_norm: str = "batch",
        max_epochs: int = 100,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> None:
        # Flow models use negative log-likelihood loss by default
        # Don't pass criterion as a method reference - Skorch will call it during forward
        # We'll override get_loss instead

        # Resolve the "auto" architecture into a single standard encoder layer (width 512,
        # a good match for wide pretrained embeddings such as ~2048-dim CheMeleon). One MLP
        # layer gives the flow a learned conditioner (and, with dropout, MC-dropout
        # uncertainty); HPO can still widen/deepen it. None / [] keep the flow conditioned on
        # the raw input. Inside NODE the flow head is built without this encoder, since the
        # NODE trunk already provides the feature layers.
        if isinstance(mlp_hidden_dims, str):
            if mlp_hidden_dims != "auto":
                raise ValueError(f"mlp_hidden_dims string must be 'auto', got {mlp_hidden_dims!r}.")
            mlp_hidden_dims = [512]

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
            module__mlp_norm=mlp_norm,
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
            "mlp_norm",
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
            "mlp_norm",
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
            Either a tuple ``(predictions, knowledge_uncertainty, data_uncertainty)`` --
            where ``predictions`` is the mode (MAP), matching :meth:`predict` -- or,
            when ``return_all=True``, a dict with keys ``predictions`` (mode),
            ``mean_predictions`` (mean of the sampling distribution),
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
                # Point estimates from the SAME draws: ``mode`` is the MAP sample
                # (consistent with predict()); ``mean`` is the sampling-distribution
                # mean. Both are exposed so callers can pick either.
                mean_pred_t = samples.mean(dim=0)  # (N, D)
                best_idx = log_p.argmax(dim=0)  # (N,)
                batch_arange = torch.arange(samples.shape[1], device=samples.device)
                mode_pred_t = samples[best_idx, batch_arange, :]  # (N, D)
            predictions = mode_pred_t.cpu().numpy()
            mean_predictions = mean_pred_t.cpu().numpy()
            data_uncertainty = data.cpu().numpy()
            if output_dim == 1:
                predictions = predictions.flatten()
                mean_predictions = mean_predictions.flatten()
                data_uncertainty = data_uncertainty.flatten()
            if return_all:
                return {
                    "predictions": predictions,
                    "mean_predictions": mean_predictions,
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
            # Mean of the MC sampling distribution (pooled over passes and draws).
            mean_pred_t = samp_stack.mean(dim=(0, 1))  # (N, D)
            # Mode = pooled MC sample with the highest mixture log-density p̄ (MAP),
            # so the reported point estimate matches predict()'s mode convention.
            n_batch = samp_stack.shape[2]
            m_pool = samp_stack.shape[0] * samp_stack.shape[1]
            mix_pooled = mix_all.reshape(m_pool, n_batch)  # (M, N)
            samp_pooled = samp_stack.reshape(m_pool, n_batch, samp_stack.shape[3])  # (M, N, D)
            best_idx = mix_pooled.argmax(dim=0)  # (N,)
            batch_arange = torch.arange(n_batch, device=samp_pooled.device)
            mode_pred_t = samp_pooled[best_idx, batch_arange, :]  # (N, D)

        data_uncertainty = data.detach().cpu().numpy()
        total_raw = total.detach().cpu().numpy()
        predictions = mode_pred_t.detach().cpu().numpy()
        mean_predictions = mean_pred_t.detach().cpu().numpy()

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
            mean_predictions = mean_predictions.flatten()

        if return_all:
            return {
                "predictions": predictions,
                "mean_predictions": mean_predictions,
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

                * **Flow + MLP encoder with dropout** (default; ``mlp_hidden_dims='auto'`` and
                    ``mlp_dropout=0.1``): the same flow + MC-dropout decomposition as the NODE flow
                    head becomes available — ``data_uncertainty`` (expected differential entropy,
                    aleatoric), ``knowledge_uncertainty`` (mutual information, epistemic) and their
                    sum ``total_uncertainty`` (see :meth:`predict_with_combined_uncertainty`).
                * **Flow alone / deterministic encoder** (set ``mlp_hidden_dims=None`` or ``[]``,
                    and/or set ``mlp_dropout=0.0``): uncertainty is purely aleatoric and is the
                    flow's differential entropy ``H[p(y|x)]`` estimated by sampling
                    (``data_uncertainty = -E_{y~p}[log p(y)]``, the NFlows-Out sampled-entropy
                    definition from BALSA), with ``knowledge_uncertainty`` set to ``None``. This
                    reports the same aleatoric quantity as
                    :meth:`predict_with_combined_uncertainty` for the same configuration.

        Because the flow is a full probabilistic model, this head can also return genuine
        predictive quantiles sampled from the distribution (unlike dropout-only heads
        which only expose a mean and std).

        Args:
            X: Input features.
            return_quantiles: If True, also return quantile predictions sampled from the
                flow distribution (default False).
            quantiles: List of quantiles to compute. Default ``[0.25, 0.5, 0.75]``
                (``DEFAULT_QUANTILES``).
            uncertainty_for_opt: If True, return only a single uncertainty column
                as a Series for optimisation / active learning (default False).
                Returns the epistemic ``knowledge_uncertainty`` (the BALD/BALSA
                acquisition signal) when MC-dropout is active, falling back to
                ``total_uncertainty`` for the flow-alone regime, which exposes no
                epistemic estimate.
            num_samples: Number of samples drawn from the flow for the mode and quantiles
                (default 1000).
            num_mc_samples: Number of MC-dropout forward passes used when an MLP encoder
                with dropout is present (default 30). Ignored for the flow-alone regime.
            **kwargs: Additional arguments (ignored).

        Returns:
            Union[pd.DataFrame, pd.Series, tuple[pd.DataFrame, np.ndarray]]:
                - Default: DataFrame with columns ``pred`` (mode / MAP, matching
                  :meth:`predict`), ``mean_predictions`` (mean of the sampling
                  distribution), ``knowledge_uncertainty`` (``None`` unless MC-dropout
                  is active), ``data_uncertainty`` and ``total_uncertainty``.
                - If ``return_quantiles=True``: ``(DataFrame, quantile_array)`` where the
                  array has shape ``(n_samples, n_quantiles)`` (single target) or
                  ``(n_samples, n_quantiles, output_dim)`` (multi-target).
                - If ``uncertainty_for_opt=True``: ``pd.Series`` of
                  ``knowledge_uncertainty`` (epistemic), or of ``total_uncertainty``
                  when no epistemic estimate is available (flow-alone regime).

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
            # ``pred`` is the mode (MAP, matching predict()); ``mean_predictions`` is the
            # mean of the (MC) sampling distribution.
            results = pd.DataFrame(
                {
                    "pred": _prepare_for_dataframe(stats["predictions"]),
                    "mean_predictions": _prepare_for_dataframe(stats["mean_predictions"]),
                    "knowledge_uncertainty": _prepare_for_dataframe(stats["knowledge_uncertainty"]),
                    "data_uncertainty": stats["data_uncertainty"],
                    "total_uncertainty": stats["total_uncertainty"],
                },
                index=index,
            )
        else:
            # Flow alone: the aleatoric term is the flow's differential entropy
            # H[p(y|x)] estimated by sampling (-E_{y~p}[log p(y)], the NFlows-Out
            # sampled-entropy definition from BALSA) — the SAME quantity that
            # predict_with_combined_uncertainty reports for this configuration, so the
            # two public helpers stay consistent. ``pred`` is the mode (MAP, matching
            # predict()); ``mean_predictions`` is the mean of the sampling distribution.
            stats = self.predict_with_combined_uncertainty(
                X,
                num_flow_samples=num_samples,
                return_all=True,
            )
            results = pd.DataFrame(
                {
                    "pred": _prepare_for_dataframe(stats["predictions"]),
                    "mean_predictions": _prepare_for_dataframe(stats["mean_predictions"]),
                    "knowledge_uncertainty": None,  # No dropout in standalone head
                    "data_uncertainty": stats["data_uncertainty"],  # sampled differential entropy
                    "total_uncertainty": stats["total_uncertainty"],  # == data (only source)
                },
                index=index,
            )

        if uncertainty_for_opt:
            # Epistemic (knowledge) uncertainty is the active-learning acquisition
            # signal (BALD/BALSA); fall back to total_uncertainty for the flow-alone
            # regime, which exposes no epistemic estimate.
            if results["knowledge_uncertainty"].notna().any():
                return results.loc[:, "knowledge_uncertainty"]
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
