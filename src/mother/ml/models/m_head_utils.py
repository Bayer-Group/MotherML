"""
Utility functions for head layers (MLP, Flow, etc.).

This module provides utilities shared by the prediction head layers
(:mod:`mother.ml.models.m_mlp` and :mod:`mother.ml.models.m_flow`): dimension
auto-detection, dataframe formatting, adaptive MLP sizing, and flow mode /
uncertainty computation.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from optuna import Trial
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback

# Default quantiles for the standardised predict_uncertainty interface
# (mirrors the convention used by CatBoost / TabPFN / RandomForest / NODE).
DEFAULT_QUANTILES: List[float] = [0.25, 0.5, 0.75]


def compute_flow_mode_and_uncertainty(dist, num_samples: int = 100):
    """
    Compute mode predictions and uncertainty from a flow distribution.

    This function samples from a flow distribution, finds the mode (sample with
    highest log probability), and computes uncertainty as negative log-likelihood.

    The approach follows the NodeFlow paper (Wielopolski, Furman & Zięba, 2024):
    - Mode: Sample with maximum log probability (MAP estimate)
    - Uncertainty: Negative log-likelihood of the mode
        * High log_prob → low -log_prob (low uncertainty)
        * Low log_prob → high -log_prob (high uncertainty)

    Uses fully vectorized operations for both log_prob computation and mode
    extraction, leveraging zuko's native support for batched log_prob calls.
    This provides 18-100x speedup over sequential per-sample evaluation.

    Args:
        dist: Flow distribution object with sample() and log_prob() methods
            Expected to be a zuko flow distribution or compatible interface
        num_samples: Number of samples to draw for finding mode (default: 100)
            Higher values give more accurate mode estimates but are slower

    Returns:
        Tuple of (mode_predictions, uncertainties):
        - mode_predictions: torch.Tensor of shape [batch_size, output_dim]
            The sample with highest log probability for each input
        - uncertainties: torch.Tensor of shape [batch_size]
            Negative log-likelihood of the mode (data uncertainty)

    Example:
        >>> # Get flow distribution from model
        >>> dist = flow_model(x)  # x: [batch_size, input_dim]
        >>> # Compute mode and uncertainty
        >>> mode, uncertainty = compute_flow_mode_and_uncertainty(dist, num_samples=100)
        >>> # mode: [batch_size, output_dim]
        >>> # uncertainty: [batch_size] with higher values = more uncertain

    References:
        Wielopolski, P., Furman, O., & Zięba, M. (2024).
        NodeFlow: Towards End-to-end Flexible Probabilistic Regression on Tabular Data.
        Entropy, 26(7), 593.
        https://doi.org/10.3390/e26070593
    """
    # Inference-only helper: no autograd graph is needed for mode/uncertainty.
    with torch.no_grad():
        # Sample from distribution
        samples = dist.sample((num_samples,))  # [num_samples, batch_size, output_dim]

        # Vectorized log_prob: zuko supports batched evaluation natively
        # Passing [num_samples, batch_size, output_dim] directly returns [num_samples, batch_size]
        log_probs = dist.log_prob(samples)  # [num_samples, batch_size]

        # Find the sample with highest log_prob for each input (mode)
        best_log_probs, best_indices = log_probs.max(dim=0)  # [batch_size]

        # Vectorized mode extraction using advanced indexing
        # best_indices: [batch_size], need to gather from samples: [num_samples, batch_size, output_dim]
        batch_arange = torch.arange(samples.shape[1], device=samples.device)
        mode_predictions = samples[best_indices, batch_arange, :]  # [batch_size, output_dim]

        # Data uncertainty: negative log-likelihood of the mode
        # High log_prob → low -log_prob (low uncertainty)
        # Low log_prob → high -log_prob (high uncertainty)
        uncertainties = -best_log_probs  # [batch_size]

    return mode_predictions, uncertainties


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
        current_output_dim = raw_params.get("module__output_dim", 1)

        # Auto-detect only the dimension(s) still at the placeholder value (1). If the
        # user set BOTH input_dim and output_dim explicitly there is nothing to do;
        # otherwise we still detect the missing one (e.g. a user who sets input_dim but
        # leaves output_dim at 1 for multi-class classification).
        if current_input_dim != 1 and current_output_dim != 1:
            return  # Both dimensions already set by user

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
                    # Multi-label targets are typically shape [n_samples, n_labels]
                    # with values in {0, 1}. In that case output_dim must be n_labels,
                    # not the number of unique scalar values.
                    output_dim = int(y.shape[1]) if y.shape[1] > 1 else int(pd.Series(y.values.ravel()).nunique())
                elif isinstance(y, pd.Series):
                    output_dim = int(y.nunique())
                else:
                    y_arr = np.asarray(y)
                    output_dim = (
                        int(y_arr.shape[1]) if y_arr.ndim > 1 and y_arr.shape[1] > 1 else int(len(np.unique(y_arr)))
                    )
            elif hasattr(y, "shape"):
                # Regression: output dimension is purely shape-based.
                output_dim = y.shape[1] if len(y.shape) > 1 else 1
            else:
                output_dim = 1
        else:
            output_dim = 1

        # Update only the placeholder dimensions and force re-initialization so a
        # user-provided input_dim / output_dim is never overwritten by detection.
        new_params: Dict[str, int] = {}
        if current_input_dim == 1:
            new_params["module__input_dim"] = input_dim
        if current_output_dim == 1:
            new_params["module__output_dim"] = output_dim
        if new_params:
            net.set_params(**new_params)
            if net.initialized_:
                net.initialize()


def _suggest_adaptive_width(trial: Trial, input_dim: int, *, width_key: str) -> int:
    """Suggest a single input-adaptive hidden-layer width.

    Width is sampled log-uniformly in ``[max(64, input_dim // 8), input_dim]``. The
    low floor lets tuning reach compact, ChemProp-style readouts (e.g. ~300 for a
    2048-dim foundation embedding such as CheMeleon); the ceiling is capped at
    input_dim to avoid over-parameterised heads. Log scale puts finer granularity on
    the small widths that tend to win on frozen embeddings while still spanning to the
    ceiling. Shared by the multi-layer funnel/constant helper and single-layer heads
    (e.g. the flow head's one-layer MLP conditioner).
    """
    min_hidden = max(64, input_dim // 8)
    max_hidden = max(min_hidden, input_dim)
    return trial.suggest_int(width_key, min_hidden, max_hidden, log=True)


def _suggest_adaptive_hidden_dims(
    trial: Trial,
    input_dim: int,
    *,
    layers_key: str,
    width_key: str,
    max_layers: int = 4,
    shape_key: Optional[str] = None,
) -> List[int]:
    """Suggest an input-adaptive MLP architecture.

    Shared by the standalone MLP head (``BaseMLPHeadEstimator``) and the flow head's
    MLP trunk (``BaseFlowHeadEstimator``) so both size their hidden layers identically:

    - **depth** ``[1, max_layers]`` (``layers_key``);
    - **first-layer width** log-uniform in ``[max(64, input_dim // 8), input_dim * 2]``
      (``width_key``), via :func:`_suggest_adaptive_width`;
    - **shape** (``shape_key``, optional): ``"funnel"`` halves each subsequent layer
      (floored at 32); ``"constant"`` keeps every layer at the first-layer width. When
      ``shape_key`` is ``None`` the architecture is always a funnel (backwards compatible).

    Constant width mirrors the ChemProp / foundation-embedding readout convention and
    often fits wide pretrained embeddings (e.g. CheMeleon) better than a funnel; offering
    both lets tuning pick per dataset. It is a single independent categorical, so no
    nested/conditional search space is introduced.

    Args:
        trial: Optuna trial used to suggest the depth, first-layer width and shape.
        input_dim: Number of input features (drives the width scaling).
        layers_key: Full trial parameter name for the number of hidden layers.
        width_key: Full trial parameter name for the first hidden-layer width.
        max_layers: Maximum number of hidden layers (default 4).
        shape_key: Optional trial parameter name for the funnel/constant choice. When
            omitted, a funnel is always used (no extra hyperparameter is sampled).

    Returns:
        The list of hidden-layer sizes.
    """
    num_layers = trial.suggest_int(layers_key, 1, max_layers, log=False)

    first_hidden = _suggest_adaptive_width(trial, input_dim, width_key=width_key)

    # Shape: "funnel" halves each subsequent layer (floored at 32); "constant" keeps the
    # first-layer width. Only sampled when shape_key is given, and as a single independent
    # categorical so Optuna never sees a nested/conditional space.
    shape = "funnel"
    if shape_key is not None:
        shape = trial.suggest_categorical(shape_key, ("funnel", "constant"))

    hidden_dims = [first_hidden]
    for i in range(1, num_layers):
        hidden_dims.append(first_hidden if shape == "constant" else max(32, first_hidden // (2**i)))
    return hidden_dims
