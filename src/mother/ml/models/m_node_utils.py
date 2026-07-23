"""
NODE Architecture Utilities

This module contains the core building blocks of the Neural Oblivious Decision
Ensembles (NODE) architecture:
- Sparse activation functions (sparsemax, entmax15, sparsemoid, entmoid15)
- Base module classes (ModuleWithInit, Lambda, Residual)
- Embedding layer for tabular data
- ODST (Oblivious Differentiable Sparsemax Tree) - the fundamental tree structure
- DenseODSTBlock - stacks multiple ODST layers

These are the "raw" NODE architecture components that are independent of
the Skorch/sklearn wrappers and can be used standalone in PyTorch.
"""

from typing import Any, Callable, Dict, Optional, Tuple
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.jit import script

# ==============================================================================
# UTILITY FUNCTIONS FOR SPARSE ACTIVATIONS
# ==============================================================================


def _make_ix_like(X: Tensor, dim: int) -> Tensor:
    """Create index tensor matching shape of X along specified dimension."""
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X: Tensor, dim: int) -> Tensor:
    """Roll specified dimension to last position."""
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() + dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


# ==============================================================================
# SPARSE ACTIVATION FUNCTIONS
# ==============================================================================
# Implementation of entmax (Peters et al., 2019) and sparsemax (Martins & Astudillo, 2016)
# Author: Ben Peters, Vlad Niculae <vlad@vene.ro>


class Entmoid15(Function):
    """A highly optimized equivalent of lambda x: Entmax15([x, 0])"""

    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @script
    def _forward(x: Tensor) -> Tensor:
        x_abs, is_pos = abs(x), x >= 0
        tau = (x_abs + torch.sqrt(F.relu(8 - x_abs**2))) / 2
        tau.masked_fill_(tau <= x_abs, 2.0)
        y_neg = 0.25 * F.relu(tau - x_abs, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    @script
    def _backward(output: Tensor, grad_output: Tensor) -> Tensor:
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


def sparsemoid(input: Tensor) -> Tensor:
    """Sparse sigmoid-like activation for binary splits."""
    return (0.5 * input + 0.5).clamp_(0, 1)


def _sparsemax_threshold_and_support(X: Tensor, dim: int = -1, k: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Core computation for sparsemax: optimal threshold and support size."""
    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


def _entmax_threshold_and_support(X: Tensor, dim: int = -1, k: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Core computation for 1.5-entmax: optimal threshold and support size."""
    if k is None or k >= X.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt**2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean**2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star, support_size


class SparsemaxFunction(Function):
    """Sparsemax activation function (PyTorch autograd)."""

    @classmethod
    def forward(cls, ctx: Any, X: Tensor, dim: int = -1, k: Optional[int] = None) -> Tensor:
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


class Entmax15Function(Function):
    """1.5-entmax activation function (PyTorch autograd)."""

    @classmethod
    def forward(cls, ctx: Any, X: Tensor, dim: int = 0, k: Optional[int] = None) -> Tensor:
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @classmethod
    def backward(cls, ctx: Any, dY: Tensor) -> Tuple[Tensor, None, None]:
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


def sparsemax(X: Tensor, dim: int = -1, k: Optional[int] = None) -> Tensor:
    """Sparsemax: normalizing sparse transform (a la softmax).

    Solves the projection:  min_p ||x - p||_2   s.t.  p >= 0, sum(p) == 1.

    References:
        Martins, A. & Astudillo, R. (2016). From Softmax to Sparsemax.
    """
    return SparsemaxFunction.apply(X, dim, k)


def entmax15(X: Tensor, dim: int = -1, k: Optional[int] = None) -> Tensor:
    """1.5-entmax: normalizing sparse transform (a la softmax).

    Solves: max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.
    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.
    """
    return Entmax15Function.apply(X, dim, k)


# Convenience aliases
entmoid15 = Entmoid15.apply


# ==============================================================================
# BASE MODULE CLASSES
# ==============================================================================


class ModuleWithInit(nn.Module):
    """Base class for pytorch module with data-aware initializer on first batch."""

    def __init__(self) -> None:
        super().__init__()
        # Persistent buffer (NOT an nn.Parameter): it is a bookkeeping flag, not a
        # trainable weight, so it must stay out of model.parameters() / optimisers
        # while still being saved in state_dict.
        self.register_buffer("_is_initialized_tensor", torch.tensor(0, dtype=torch.uint8))
        self._is_initialized_bool: Optional[bool] = None
        # A cached python bool mirrors the buffer to avoid a tensor .item() sync on
        # every forward call.
        # please DO NOT use these flags in child modules

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initialize module tensors using first batch of data."""
        raise NotImplementedError("Please implement initialize() in subclass")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)


class Lambda(nn.Module):
    """A wrapper for a lambda function as a pytorch module."""

    def __init__(self, func: Callable) -> None:
        """Initialize lambda module
        Args:
            func: any function/callable
        """
        super().__init__()
        self.func = func

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class Residual(nn.Module):
    """Residual connection wrapper: output = layer(x) + x."""

    def __init__(self, layer: Callable[..., Tensor]) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Apply residual connection: output = layer(input) + input."""
        return self.layer(x, **kwargs) + x


# ==============================================================================
# EMBEDDING LAYER
# ==============================================================================


class Embedding1dLayer(nn.Module):
    """
    Embedding layer for tabular data with continuous and categorical features.

    Handles:
    - Continuous features: optional BatchNorm normalization
    - Categorical features: learned dense embeddings with dropout
    - Concatenation of both feature types into a single tensor
    """

    def __init__(
        self,
        continuous_dim: int = 0,
        categorical_embedding_dims: Optional[list] = None,
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
    ) -> None:
        super().__init__()

        if categorical_embedding_dims is None:
            categorical_embedding_dims = []

        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.embedding_dropout = embedding_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input

        # Categorical embedding layers
        if len(categorical_embedding_dims) > 0:
            self.cat_embedding_layers = nn.ModuleList(
                [nn.Embedding(vocab_size, embedding_dim) for vocab_size, embedding_dim in categorical_embedding_dims]
            )
            self.embedding_dropout_layer = nn.Dropout(embedding_dropout)
        else:
            self.cat_embedding_layers = None

        # Optional batch normalization for continuous features
        if batch_norm_continuous_input and continuous_dim > 0:
            self.cont_batch_norm = nn.BatchNorm1d(continuous_dim)
        else:
            self.cont_batch_norm = None

    @property
    def embedded_cat_dim(self) -> int:
        """Total dimension of all categorical embeddings combined."""
        if self.cat_embedding_layers is not None:
            return sum([embedding_dim for vocab_size, embedding_dim in self.categorical_embedding_dims])
        else:
            return 0

    def forward(self, x_dict: Dict[str, Optional[Tensor]]) -> Tensor:
        """
        Process continuous and categorical features into a single tensor.

        Args:
            x_dict: Dict with 'continuous' and/or 'categorical' tensors.

        Returns:
            Concatenated feature tensor [batch_size, total_feature_dim].
        """
        continuous = x_dict.get("continuous", None)
        categorical = x_dict.get("categorical", None)

        # Process continuous features
        if continuous is not None and self.continuous_dim > 0:
            if self.cont_batch_norm is not None:
                continuous = self.cont_batch_norm(continuous)
        else:
            continuous = None

        # Process categorical features through embeddings
        if categorical is not None and self.cat_embedding_layers is not None:
            cat_embed = []
            for i, embedding_layer in enumerate(self.cat_embedding_layers):
                embedded_feature = embedding_layer(categorical[:, i])
                cat_embed.append(embedded_feature)
            categorical = torch.cat(cat_embed, dim=1)
            if self.embedding_dropout > 0:
                categorical = self.embedding_dropout_layer(categorical)
        else:
            categorical = None

        # Concatenate
        if continuous is not None and categorical is not None:
            x = torch.cat([continuous, categorical], dim=1)
        elif continuous is not None:
            x = continuous
        elif categorical is not None:
            x = categorical
        else:
            raise ValueError("Both continuous and categorical inputs are None")

        return x


# ==============================================================================
# ODST - OBLIVIOUS DIFFERENTIABLE SPARSEMAX TREE
# ==============================================================================


class ODST(ModuleWithInit):
    """
    Oblivious Differentiable Sparsemax Tree (ODST) — core building block of NODE.

    An oblivious decision tree where all nodes at the same depth share the same
    splitting feature and threshold. Differentiable via soft split functions
    (sparsemoid/entmoid15) and sparse feature selection (sparsemax/entmax15),
    enabling end-to-end gradient-based optimization.

    Key properties:
    - Oblivious structure: same feature/threshold at each depth level → efficient vectorization
    - Soft decisions: probabilistic splits instead of hard left/right
    - Sparse feature selection: focuses on most relevant features per depth level
    - Data-aware initialization: thresholds set from data quantiles for stable training
    """

    def __init__(
        self,
        in_features: int,
        num_trees: int,
        depth: int = 6,
        tree_output_dim: int = 1,
        flatten_output: bool = True,
        choice_function: Callable = entmax15,
        bin_function: Callable = entmoid15,
        initialize_response_: Callable = nn.init.normal_,
        initialize_selection_logits_: Callable = nn.init.uniform_,
        threshold_init_beta: float = 1.0,
        threshold_init_cutoff: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.num_trees = num_trees
        self.tree_dim = tree_output_dim
        self.flatten_output = flatten_output
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff
        self.random_state = random_state

        # Leaf response values: [num_trees, tree_output_dim, 2^depth]
        self.response = nn.Parameter(torch.zeros([num_trees, tree_output_dim, 2**depth]), requires_grad=True)
        initialize_response_(self.response)

        # Feature selection logits: [in_features, num_trees, depth]
        self.feature_selection_logits = nn.Parameter(torch.zeros([in_features, num_trees, depth]), requires_grad=True)
        initialize_selection_logits_(self.feature_selection_logits)

        # Decision thresholds and temperatures (initialized from data in initialize())
        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )
        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )

        # Pre-computed binary codes for mapping soft decisions to leaf indices
        with torch.no_grad():
            indices = torch.arange(2**self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            # Shape: [depth, 2^depth, 2]
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)

    # Re-evaluate matmul strategy every N forward passes so we adapt as
    # entmax15/entmoid15 sparsity patterns evolve during training.
    _STRATEGY_RECHECK_INTERVAL = 500

    def _detect_matmul_strategy(self, input: Tensor, feature_selectors_2d: Tensor) -> str:
        """
        Choose matmul strategy based on sparsity of the operands.

        Re-evaluated every ``_STRATEGY_RECHECK_INTERVAL`` forward passes so the
        strategy can adapt as entmax15 feature selectors sharpen during training.

        Rules (from benchmarking on typical NODE workloads):
        - If either matrix is >50% zeros, use sparse mm on whichever is sparser.
          Input sparsity is common with fingerprints; selector sparsity comes from
          entmax but is rarely >50% in practice.
        - Otherwise stick with dense matmul — it wins for normal tabular data.

        Strategies:
        - "dense":            input @ selector                          (plain matmul)
        - "input_sparse":     sparse.mm(sparse_input, dense_selector)   (exploits input zeros)
        - "selector_sparse":  sparse.mm(sparse_sel.T, input.T).T       (exploits entmax zeros)
        """
        _SPARSITY_THRESHOLD = 0.5

        with torch.no_grad():
            inp_sparsity = (input == 0).sum().item() / input.numel()
            sel_sparsity = (feature_selectors_2d == 0).sum().item() / feature_selectors_2d.numel()

        if inp_sparsity > _SPARSITY_THRESHOLD or sel_sparsity > _SPARSITY_THRESHOLD:
            # At least one operand is very sparse — use sparse mm on the sparser one
            if inp_sparsity >= sel_sparsity:
                strategy = "input_sparse"
            else:
                strategy = "selector_sparse"
        else:
            strategy = "dense"

        old = getattr(self, "_matmul_strategy", None)
        if old is None:
            warn(
                f"ODST matmul strategy: '{strategy}' "
                f"(input_sparsity={inp_sparsity:.0%}, selector_sparsity={sel_sparsity:.0%}, "
                f"features={input.shape[1]}, trees×depth={feature_selectors_2d.shape[1]})"
            )
        elif old != strategy:
            warn(
                f"ODST matmul strategy changed: '{old}' → '{strategy}' "
                f"(input_sparsity={inp_sparsity:.0%}, selector_sparsity={sel_sparsity:.0%})"
            )
        return strategy

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass: feature selection → threshold comparison → soft leaf routing → response.

        Steps:
        1. Sparse feature selection via choice_function (sparsemax/entmax)
        2. Extract selected feature values (strategy auto-detected on first batch)
        3. Compare to learned thresholds, scaled by temperature
        4. Soft binary decisions via bin_function (sparsemoid/entmoid)
        5. Compute leaf probabilities from decision products
        6. Weighted sum of leaf responses

        Args:
            input: [batch_size, in_features]

        Returns:
            [batch_size, num_trees * tree_output_dim] if flatten_output else
            [batch_size, num_trees, tree_output_dim]
        """
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)

        # 1. Sparse feature selection
        feature_logits = self.feature_selection_logits  # [in_features, num_trees, depth]
        feature_selectors = self.choice_function(feature_logits, dim=0)
        feature_selectors_2d: Tensor = feature_selectors.reshape(feature_selectors.shape[0], -1)

        # 2. Feature value extraction (strategy auto-detected, rechecked periodically)
        self._forward_count = getattr(self, "_forward_count", 0) + 1
        if self._forward_count == 1 or self._forward_count % self._STRATEGY_RECHECK_INTERVAL == 0:
            self._matmul_strategy = self._detect_matmul_strategy(input, feature_selectors_2d)
        strategy = self._matmul_strategy

        if strategy == "input_sparse":
            input_sparse: Tensor = input.to_sparse()
            feature_values = torch.sparse.mm(input_sparse, feature_selectors_2d)
        elif strategy == "selector_sparse":
            nonzero_mask: Tensor = feature_selectors_2d != 0
            indices: Tensor = nonzero_mask.nonzero(as_tuple=False).t()
            values: Tensor = feature_selectors_2d[nonzero_mask]
            sparse_sel: Tensor = torch.sparse_coo_tensor(
                indices,
                values,
                feature_selectors_2d.shape,
                device=feature_selectors_2d.device,
                dtype=feature_selectors_2d.dtype,
            ).coalesce()
            feature_values = torch.sparse.mm(sparse_sel.t(), input.t()).t()
        else:
            feature_values = input @ feature_selectors_2d

        feature_values = feature_values.reshape(input.shape[0], self.num_trees, self.depth)

        # 3. Threshold comparison with temperature scaling
        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)
        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # [batch_size, num_trees, depth, 2]

        # 4. Soft binary decisions
        bins = self.bin_function(threshold_logits)

        # 5. Leaf probability computation via binary code matching
        bin_matches = torch.einsum("btds,dcs->btdc", bins, self.bin_codes_1hot)
        response_weights = torch.prod(bin_matches, dim=-2)
        # [batch_size, num_trees, 2^depth]

        # 6. Weighted response aggregation
        response = torch.einsum("bnd,ncd->bnc", response_weights, self.response)

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, input: Tensor, eps: float = 1e-6) -> None:
        """
        Data-aware initialization of thresholds and temperatures from first batch.

        Sets thresholds to data quantiles and temperatures based on data distribution
        to ensure meaningful initial decisions and proper gradient flow.
        """
        assert len(input.shape) == 2

        if input.shape[0] < 256:
            warn(
                "Data-aware initialization is performed on less than 256 data points. "
                "This may reduce threshold initialization quality on some datasets. "
                "Prefer at least 256 samples for stable initialization; 512+ can be more robust "
                "when memory allows. You can run manual initialization before training, ideally "
                "under torch.no_grad() for memory efficiency."
            )

        with torch.no_grad():
            if not isinstance(input, torch.Tensor):
                input_tensor = torch.as_tensor(input, dtype=torch.float32)
            else:
                input_tensor = input

            # Compute feature values using current selection weights
            feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
            feature_values = torch.einsum("bi,ind->bnd", input_tensor, feature_selectors)

            # Initialize thresholds from sampled data quantiles (Beta distribution)
            rng = np.random.default_rng(self.random_state)
            percentiles_q = 100 * rng.beta(
                self.threshold_init_beta,
                self.threshold_init_beta,
                size=[self.num_trees, self.depth],
            )

            feature_values_np = feature_values.detach().cpu().numpy()
            thresholds = np.zeros([self.num_trees, self.depth])
            for tree_idx in range(self.num_trees):
                for depth_idx in range(self.depth):
                    thresholds[tree_idx, depth_idx] = np.percentile(
                        feature_values_np[:, tree_idx, depth_idx], percentiles_q[tree_idx, depth_idx]
                    )

            self.feature_thresholds.data[...] = torch.as_tensor(
                thresholds,
                dtype=feature_values.dtype,
                device=feature_values.device,
            )

            # Initialize temperatures from data spread around thresholds
            feature_threshold_diffs = abs(feature_values - self.feature_thresholds).detach().cpu().numpy()
            temperatures = np.zeros([self.num_trees, self.depth])
            for tree_idx in range(self.num_trees):
                for depth_idx in range(self.depth):
                    temperatures[tree_idx, depth_idx] = np.percentile(
                        feature_threshold_diffs[:, tree_idx, depth_idx], q=100 * min(1.0, self.threshold_init_cutoff)
                    )

            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.feature_selection_logits.shape[0]},"
            f" num_trees={self.num_trees},"
            f" depth={self.depth},"
            f" tree_dim={self.tree_dim},"
            f" flatten_output={self.flatten_output})"
        )


# ==============================================================================
# DENSE ODST BLOCK
# ==============================================================================


class DenseODSTBlock(nn.Sequential):
    """
    Dense block of ODST layers with skip connections.

    Stacks multiple ODST layers where each layer receives all previous outputs
    (like DenseNet). Supports dimension capping to prevent memory explosion.
    """

    def __init__(
        self,
        input_dim: int,
        num_trees: int,
        num_layers: int,
        tree_output_dim: int = 1,
        max_features: Optional[int] = None,
        input_dropout: float = 0.0,
        flatten_output: bool = False,
        Module: type = ODST,
        **kwargs: Any,
    ) -> None:
        """
        Build a dense stack of ODST layers.

        Args:
            input_dim: Number of input features.
            num_trees: Number of trees per layer.
            num_layers: Number of ODST layers to stack.
            tree_output_dim: Output dimension per tree (output_dim + additional).
            max_features: Cap on concatenated feature dimension between layers.
                Prevents memory explosion with many layers. ``None`` = no cap.
            input_dropout: Dropout rate on concatenated features between layers.
            flatten_output: If True, return ``[batch, layers*trees*dim]``;
                otherwise ``[batch, layers*trees, dim]``.
            Module: ODST class (or compatible) to use for each layer.
            **kwargs: Forwarded to each ``Module(...)`` constructor.
        """
        # Ensure max_features never shrinks below the original input width.
        # The dense forward path always preserves all original features, so
        # allowing a smaller cap would make later ODST layers expect fewer
        # features than they actually receive.
        effective_max_features = max_features
        if effective_max_features is not None and effective_max_features < input_dim:
            warn(
                f"max_features={effective_max_features} is smaller than input_dim={input_dim}; "
                f"using max_features={input_dim} to keep dimensions consistent."
            )
            effective_max_features = input_dim

        layers = []
        for _ in range(num_layers):
            oddt = Module(input_dim, num_trees, tree_output_dim=tree_output_dim, flatten_output=True, **kwargs)

            # Cap dimension growth between layers (prevents memory issues)
            input_dim = min(input_dim + num_trees * tree_output_dim, effective_max_features or float("inf"))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers = num_layers
        self.layer_dim = num_trees
        self.tree_dim = tree_output_dim
        self.max_features = effective_max_features
        self.flatten_output = flatten_output
        self.input_dropout = input_dropout

    def forward(self, x: Tensor) -> Tensor:
        """Forward with dense (DenseNet-style) connections and optional input dropout.

        Each layer receives the concatenation of the original features and all
        previous layer outputs.  If ``max_features`` is set, the concatenated
        tensor is trimmed to keep the original features plus the most recent
        tail to stay within budget.

        Args:
            x: Input features ``[batch_size, input_dim]``.

        Returns:
            Tree outputs.  Shape depends on ``flatten_output``:
            - ``True``:  ``[batch, num_layers * num_trees * tree_output_dim]``
            - ``False``: ``[batch, num_layers * num_trees, tree_output_dim]``
        """
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                # Only trim when there is a positive tail to keep. A non-positive
                # value (max_features <= initial_features) would turn the negative
                # slice into a positive start index and corrupt the features.
                if tail_features > 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
                else:
                    # Keep only original features when no tail fits the budget.
                    layer_inp = layer_inp[..., :initial_features]
            if self.training and self.input_dropout:
                # Apply dropout to combined features (continuous + categorical embeddings)
                # This regularizes all input features, not just categorical
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs
