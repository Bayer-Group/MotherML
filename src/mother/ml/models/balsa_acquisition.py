"""
balsa_acquisition.py
====================
BALSA-style acquisition scores for active learning with normalising-flow estimators.

Implements four acquisition methods:

``"bald"``
    BALD_H / NFlows Out mutual information.
    ``knowledge = H[p̄] - (1/T) Σ_t H[p_t]``
    Identical to the ``knowledge_uncertainty`` column produced by
    ``NODERegressor.predict_uncertainty()`` / ``predict_with_combined_uncertainty()``.

``"balsa_kl_pair"``  *(recommended)*
    BALSA KL-Pair: ``Σ_{t=0}^{T-2} KL(p_t ∥ p_{t+1})``.
    Nearly free — reuses the log-prob cross-evaluation already built for BALD.
    Works unchanged for multi-target (D > 1).

``"balsa_kl_grid"``
    BALSA KL-Grid: ``Σ_t KL(p_t ∥ p̄)`` via an adaptive quantile grid and
    trapezoidal quadrature. Zero MC variance. **Single-target (D = 1) only** — it
    integrates the joint density along a 1-D grid; for D > 1 use ``balsa_kl_pair``
    or ``balsa_emd``.

``"balsa_emd"``
    BALSA EMD: ``Σ_{t=0}^{T-2} W₁(p_t, p_{t+1})`` via sorted samples (D=1)
    or sliced-Wasserstein (D > 1).

Algorithm lineage
-----------------
*   BALD functional:  Houlsby et al. (2011).  arXiv:1112.5745
*   MC-dropout approximation:  Gal, Islam & Ghahramani (2017).  ICML
*   Continuous targets / differential entropy:  Depeweg et al. (2018).  ICML
*   Flow ensemble + sampled entropy (``NFlows Out``):  Berry & Meger (2023).
    AAAI 2023 pp. 6806-6814;  arXiv:2308.13498
*   BALSA distribution-disagreement scores:  Werner & Schmidt-Thieme (2025).
    arXiv:2501.01248

Supported estimators
--------------------
*   ``NODERegressor`` with ``head_type="flow"``
*   ``FlowHeadRegressor``

Usage
-----
::

    from mother.ml.models.m_node import NODERegressor
    from mother.ml.models.balsa_acquisition import acquisition_score

    reg = NODERegressor(head_type="flow", input_dropout=0.05)
    reg.fit(X_train, y_train)

    # BALD_H baseline  (= knowledge_uncertainty from predict_uncertainty)
    scores = acquisition_score(reg, X_pool, method="bald")

    # BALSA KL-Pair  (recommended: nearly free, fully batched)
    scores = acquisition_score(reg, X_pool, method="balsa_kl_pair")

    # BALSA KL-Grid  (deterministic, adaptive grid)
    scores = acquisition_score(reg, X_pool, method="balsa_kl_grid")

    # BALSA EMD  (1-D Wasserstein on samples)
    scores = acquisition_score(reg, X_pool, method="balsa_emd")

    # Select the 10 most informative points from the pool
    top_idx = scores.argsort()[::-1][:10]
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)

__all__ = ["acquisition_score"]

# --------------------------------------------------------------------------- #
#  Type aliases                                                                #
# --------------------------------------------------------------------------- #

# A "distribution batch" is whatever zuko returns from a flow forward pass.
# We don't import zuko directly so as to not make it a hard dependency here.
_DistBatch = Any
_DistsByBatch = List[List[_DistBatch]]  # [batch][mc_pass]
_SamplesByBatch = List[List[torch.Tensor]]  # [batch][mc_pass] -> (S, B, D)


# --------------------------------------------------------------------------- #
#  MC-dropout activation helpers                                               #
# --------------------------------------------------------------------------- #


def _enable_mc_dropout_node(model: nn.Module) -> None:
    """Enable tree/input/mlp dropout while keeping BatchNorm in eval mode.

    Mirrors the logic in ``NODERegressor.predict_with_combined_uncertainty``:
    assign ``.training = True`` directly (never call ``.train()``) so that
    BatchNorm layers are not flipped into training mode.
    """
    try:
        from mother.ml.models.m_node_utils import DenseODSTBlock  # type: ignore[import]
    except ImportError:
        DenseODSTBlock = None  # NODE not installed — fall through to Dropout only

    model.eval()
    model.training = True  # enables tree_dropout in the top module
    for _m in model.modules():
        if DenseODSTBlock is not None and isinstance(_m, DenseODSTBlock):
            _m.training = True  # enables input_dropout
        elif isinstance(_m, nn.Dropout):
            _m.training = True  # enables mlp_dropout in the flow conditioner


def _enable_mc_dropout_flow_head(module: nn.Module) -> None:
    """Enable only ``nn.Dropout`` layers for a ``FlowHeadRegressor``.

    Mirrors ``MLPHeadRegressor._mc_dropout_samples``: keep the full module in
    eval (so BatchNorm uses running statistics) and switch only Dropout to train.
    """
    module.eval()
    for _m in module.modules():
        if isinstance(_m, nn.Dropout):
            _m.train()


def _restore_eval(model: nn.Module) -> None:
    """Restore full eval mode after a MC-dropout collection pass."""
    model.eval()


# --------------------------------------------------------------------------- #
#  Estimator introspection                                                     #
# --------------------------------------------------------------------------- #


def _is_node_flow(estimator: Any) -> bool:
    """Return True if *estimator* is a ``NODERegressor`` with ``head_type="flow"``."""
    try:
        from mother.ml.models.m_node import NODERegressor  # type: ignore[import]
    except ImportError:
        return False
    return (
        isinstance(estimator, NODERegressor)
        and hasattr(estimator, "module_")
        and getattr(estimator.module_, "head_type", None) == "flow"
    )


def _is_flow_head(estimator: Any) -> bool:
    """Return True if *estimator* is a ``FlowHeadRegressor``."""
    try:
        from mother.ml.models.m_heads import FlowHeadRegressor  # type: ignore[import]
    except ImportError:
        return False
    return isinstance(estimator, FlowHeadRegressor)


def _has_any_dropout(estimator: Any) -> bool:
    """Check whether the estimator has at least one dropout path > 0."""
    module = getattr(estimator, "module_", None)
    if module is None:
        return False
    for attr in ("input_dropout", "tree_dropout"):
        if getattr(module, attr, 0) > 0:
            return True
    for m in module.modules():
        if isinstance(m, nn.Dropout) and m.p > 0:
            return True
    return False


def _to_tensor(X: npt.NDArray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(X, dtype=np.float32)
    return torch.tensor(arr, device=device)


def _prepare_numpy(X: Union[pd.DataFrame, npt.NDArray]) -> npt.NDArray:
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32)
    arr = np.asarray(X)
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr


# --------------------------------------------------------------------------- #
#  Core MC-flow collection loop                                                #
# --------------------------------------------------------------------------- #


def _collect_mc_flow_distributions(
    estimator: Any,
    X: Union[pd.DataFrame, npt.NDArray],
    num_mc_samples: int,
    num_flow_samples: int,
    batch_size: Optional[int] = None,
) -> Tuple[_DistsByBatch, _SamplesByBatch]:
    """Run *num_mc_samples* MC-dropout passes and collect distribution objects + samples.

    Returns
    -------
    dists_by_batch : list[list[dist]]
        ``dists_by_batch[b][t]`` — zuko flow distribution for batch *b*, pass *t*.
    samples_by_batch : list[list[Tensor]]
        ``samples_by_batch[b][t]`` — shape ``(S, B, D)`` samples from that distribution.

    The two nested lists are indexed ``[batch_chunk][mc_pass]`` so that all
    downstream reductions can iterate over chunks independently without loading
    the full dataset into GPU memory at once.
    """
    # Validate the estimator *class* first (independent of fit state) so that an
    # unfitted NODERegressor yields a clear NotFittedError below rather than a
    # misleading TypeError.
    try:
        from mother.ml.models.m_node import NODERegressor  # type: ignore[import]
    except ImportError:
        NODERegressor = None  # type: ignore[assignment]
    _is_node = NODERegressor is not None and isinstance(estimator, NODERegressor)

    if not (_is_node or _is_flow_head(estimator)):
        raise TypeError(
            f"acquisition_score requires a NODERegressor (head_type='flow') or "
            f"FlowHeadRegressor, got {type(estimator).__name__!r}."
        )

    # Reject unfitted estimators up front: ``module_`` is created during ``fit``,
    # and accessing it below would otherwise raise a cryptic AttributeError.
    if getattr(estimator, "module_", None) is None:
        raise NotFittedError(
            f"{type(estimator).__name__} is not fitted yet. Call `.fit(X, y)` before `acquisition_score`."
        )

    # A NODERegressor must use the flow head for flow-based acquisition scores.
    if _is_node and getattr(estimator.module_, "head_type", None) != "flow":
        raise ValueError(
            "acquisition_score requires NODERegressor(head_type='flow'); "
            f"got head_type={getattr(estimator.module_, 'head_type', None)!r}."
        )

    if not _has_any_dropout(estimator):
        warnings.warn(
            "The estimator has no active dropout (all dropout rates are 0). "
            "All MC passes will be identical and acquisition scores will be 0. "
            "Set input_dropout / tree_dropout > 0 (e.g. 0.05) for meaningful results.",
            UserWarning,
            stacklevel=4,
        )

    dists_by_batch: _DistsByBatch = []
    samples_by_batch: _SamplesByBatch = []
    model = estimator.module_

    try:
        if _is_node_flow(estimator):
            # ------------------------------------------------------------------
            # NODE path: prepare data with NODE's own preprocessor, then iterate
            # with skorch's get_iterator so the batch size / device handling is
            # consistent with predict_with_combined_uncertainty.
            # ------------------------------------------------------------------
            X_prep = estimator._prepare_data_for_node(X)
            _enable_mc_dropout_node(model)

            with torch.no_grad():
                for t in range(num_mc_samples):
                    for b, batch in enumerate(estimator.get_iterator(X_prep, training=False)):
                        Xi = batch[0] if isinstance(batch, (tuple, list)) else batch
                        Xi = Xi.to(estimator.device)
                        yp = model(Xi)
                        samp = yp.sample(torch.Size([num_flow_samples]))  # (S, B, D)
                        if t == 0:
                            dists_by_batch.append([])
                            samples_by_batch.append([])
                        dists_by_batch[b].append(yp)
                        samples_by_batch[b].append(samp)

        else:
            # ------------------------------------------------------------------
            # FlowHeadRegressor path: collect in batches to avoid pool-size OOM.
            # Prefer the estimator's own skorch iterator so batch_size/device
            # semantics match training/predict; fall back to manual chunking.
            # ------------------------------------------------------------------
            X_np = _prepare_numpy(X)
            device = next(model.parameters()).device
            _enable_mc_dropout_flow_head(model)

            flow_batches: List[torch.Tensor] = []
            if batch_size is None:
                try:
                    dataset = estimator.get_dataset(X_np)
                    for batch in estimator.get_iterator(dataset, training=False):
                        Xi = batch[0] if isinstance(batch, (tuple, list)) else batch
                        if not isinstance(Xi, torch.Tensor):
                            Xi = torch.as_tensor(Xi)
                        flow_batches.append(Xi)
                except Exception:
                    # Keep a robust fallback for custom/skipped skorch iterator paths.
                    estimator_bs = getattr(estimator, "batch_size", None)
                    batch_size = int(estimator_bs) if estimator_bs and int(estimator_bs) > 0 else len(X_np)

            if batch_size is not None:
                for start in range(0, len(X_np), batch_size):
                    stop = min(start + batch_size, len(X_np))
                    flow_batches.append(torch.as_tensor(X_np[start:stop], dtype=torch.float32))

            with torch.no_grad():
                for t in range(num_mc_samples):
                    for b, Xi in enumerate(flow_batches):
                        Xi = Xi.to(device=device, dtype=torch.float32)
                        yp = model(Xi)
                        samp = yp.sample(torch.Size([num_flow_samples]))  # (S, B, D)
                        if t == 0:
                            dists_by_batch.append([])
                            samples_by_batch.append([])
                        dists_by_batch[b].append(yp)
                        samples_by_batch[b].append(samp)

    finally:
        _restore_eval(model)

    return dists_by_batch, samples_by_batch


# --------------------------------------------------------------------------- #
#  BALD_H / NFlows Out                                                         #
# --------------------------------------------------------------------------- #


def _bald_score(
    dists_by_batch: _DistsByBatch,
    samples_by_batch: _SamplesByBatch,
) -> npt.NDArray[np.float32]:
    """BALD_H mutual information via sampled differential entropies.

    knowledge = H[p̄] − (1/T) Σ_t H[p_t]

    This is identical to the ``knowledge_uncertainty`` column produced by
    ``predict_with_combined_uncertainty`` / ``predict_uncertainty``.
    """
    out: List[torch.Tensor] = []

    with torch.no_grad():
        for b in range(len(dists_by_batch)):
            dists_b = dists_by_batch[b]
            samples_b = samples_by_batch[b]
            T = len(dists_b)
            log_T = math.log(T)

            per_mix: List[torch.Tensor] = []
            per_self: List[torch.Tensor] = []

            for t in range(T):
                samp_t = samples_b[t]  # (S, B, D)
                # Cross-evaluate: log p_{t'}(y_{t,s}) for every t' → (T, S, B)
                lp_stack = torch.stack([dists_b[tp].log_prob(samp_t) for tp in range(T)], dim=0)
                per_mix.append(torch.logsumexp(lp_stack, dim=0) - log_T)  # (S, B)
                per_self.append(lp_stack[t])  # (S, B)

            mix_all = torch.stack(per_mix, dim=0)  # (T, S, B)
            self_all = torch.stack(per_self, dim=0)  # (T, S, B)

            total = -mix_all.mean(dim=(0, 1))  # (B,)  H[p̄]
            data = -self_all.mean(dim=(0, 1))  # (B,)  (1/T) Σ H[p_t]

            # Clamp at 0: Jensen guarantees ≥ 0 but float32 noise can produce −ε
            out.append(torch.clamp(total - data, min=0.0))

    return torch.cat(out).cpu().numpy()


# --------------------------------------------------------------------------- #
#  BALSA KL-Pair (sampled, nearly free)                                        #
# --------------------------------------------------------------------------- #


def _balsa_kl_pair(
    dists_by_batch: _DistsByBatch,
    samples_by_batch: _SamplesByBatch,
    reduction: str = "sum",
) -> npt.NDArray[np.float32]:
    """BALSA KL-Pair acquisition score.

    KL(p_t ∥ p_{t+1}) ≈ (1/S) Σ_s [log p_t(y_s) − log p_{t+1}(y_s)],  y_s ~ p_t

    Sums over the T−1 consecutive pairs (or averages if ``reduction="mean"``).
    This reuses the log-prob cross-evaluation already computed for BALD — it is
    therefore nearly free.  Works unchanged for multi-target (D > 1) because
    ``flow.log_prob`` accepts vector inputs.
    """
    out: List[torch.Tensor] = []

    with torch.no_grad():
        for b in range(len(dists_by_batch)):
            dists_b = dists_by_batch[b]
            samples_b = samples_by_batch[b]
            T = len(dists_b)
            B = samples_b[0].shape[1]
            device = samples_b[0].device

            kl_sum = torch.zeros(B, device=device)

            for t in range(T - 1):
                samp_t = samples_b[t]  # (S, B, D)
                log_pt = dists_b[t].log_prob(samp_t)  # (S, B)
                log_pt1 = dists_b[t + 1].log_prob(samp_t)  # (S, B)
                kl_t = (log_pt - log_pt1).mean(dim=0)  # (B,)
                kl_sum = kl_sum + kl_t

            if reduction == "mean" and T > 1:
                kl_sum = kl_sum / (T - 1)

            out.append(kl_sum.clamp(min=0.0))

    return torch.cat(out).cpu().numpy()


# --------------------------------------------------------------------------- #
#  BALSA KL-Grid (adaptive quantile grid, trapezoidal quadrature)              #
# --------------------------------------------------------------------------- #


def _balsa_kl_grid(
    dists_by_batch: _DistsByBatch,
    samples_by_batch: _SamplesByBatch,
    grid_size: int = 200,
    grid_range: Optional[Tuple[float, float]] = None,
    reduction: str = "sum",
) -> npt.NDArray[np.float32]:
    """BALSA KL-Grid acquisition score.

    KL(p_t ∥ p̄) is integrated via the trapezoidal rule on a 1-D grid.
    When ``grid_range=None`` (default), the grid is built adaptively from
    pooled samples — one quantile grid per batch item — so it self-locates
    wherever the flows' mass is without requiring a known target range.  This
    makes it safe for OOD inputs.

    Single-target (D = 1) only: the score integrates the *joint* flow density
    along a 1-D grid, which is not a valid per-marginal decomposition for
    D > 1.  A ``NotImplementedError`` is raised for multi-target flows — use
    ``method="balsa_kl_pair"`` or ``method="balsa_emd"`` instead.
    """
    out: List[torch.Tensor] = []

    with torch.no_grad():
        for b in range(len(dists_by_batch)):
            dists_b = dists_by_batch[b]
            samples_b = samples_by_batch[b]
            T = len(dists_b)
            S, B, D = samples_b[0].shape
            if D > 1:
                raise NotImplementedError(
                    "balsa_kl_grid supports single-target (D=1) flows only: it "
                    "integrates the joint density on a 1-D grid, which is not a "
                    "valid per-marginal KL for D>1. Use method='balsa_kl_pair' or "
                    "method='balsa_emd' for multi-target flows."
                )
            device = samples_b[0].device
            log_T = math.log(T)

            # ------------------------------------------------------------------
            # Build the evaluation grid.
            # Fixed grid:    nodes shape (G, 1, 1) → broadcast over (B, D).
            # Adaptive grid: nodes shape (G, B, D) — one grid per input per dim.
            # ------------------------------------------------------------------
            if grid_range is not None:
                lo, hi = float(grid_range[0]), float(grid_range[1])
                nodes_1d = torch.linspace(lo, hi, grid_size, device=device)  # (G,)
                # Expand to (G, B, D) for uniform treatment
                nodes = nodes_1d.view(grid_size, 1, 1).expand(grid_size, B, D)  # (G, B, D)
            else:
                # Pool all T*S samples, quantile-sort per (batch_item, marginal)
                all_samp = torch.cat(samples_b, dim=0)  # (T*S, B, D)
                pct = torch.linspace(0.0, 1.0, grid_size, device=device)  # (G,)
                # torch.quantile over the last dim: input (B, D, T*S) → (G, B, D)
                nodes = torch.quantile(
                    all_samp.permute(1, 2, 0).contiguous(),  # (B, D, T*S)
                    pct,
                    dim=-1,
                )  # (G, B, D)

            # ------------------------------------------------------------------
            # Evaluate log p_t at every grid node for every expert t.
            # zuko distributions accept batched inputs; we flatten (G, B) → G*B
            # and reshape back.
            # ------------------------------------------------------------------
            log_p_list: List[torch.Tensor] = []
            for t in range(T):
                # nodes: (G, B, D). zuko broadcasts the leading (G, B) dims
                # against the flow's batch_shape (B,), returning (G, B) —
                # exactly as the sample-based methods evaluate (S, B, D) → (S, B).
                lp = dists_b[t].log_prob(nodes)  # (G, B)
                log_p_list.append(lp)
            log_p = torch.stack(log_p_list, dim=0)  # (T, G, B)

            # Mixture log-density: log p̄ = logsumexp_t(log p_t) − log T
            log_pbar = torch.logsumexp(log_p, dim=0) - log_T  # (G, B)

            # ------------------------------------------------------------------
            # Trapezoidal KL(p_t ∥ p̄) integration over the grid axis.
            # Spacing: Δy shape (G−1, B) — per-input when using adaptive grid.
            # D == 1 is guaranteed above, so dim 0 is the only (and correct) axis.
            # ------------------------------------------------------------------
            delta_y = (nodes[1:, :, 0] - nodes[:-1, :, 0]).abs()  # (G−1, B)

            kl_sum = torch.zeros(B, device=device)
            for t in range(T):
                p_t = log_p[t].exp()  # (G, B)
                kl_int = p_t * (log_p[t] - log_pbar)  # (G, B)
                mid = 0.5 * (kl_int[:-1] + kl_int[1:])  # (G−1, B)
                kl_sum = kl_sum + (mid * delta_y).sum(dim=0)  # (B,)

            if reduction == "mean":
                kl_sum = kl_sum / T

            out.append(kl_sum.clamp(min=0.0))

    return torch.cat(out).cpu().numpy()


# --------------------------------------------------------------------------- #
#  BALSA EMD (1-D Wasserstein on samples)                                      #
# --------------------------------------------------------------------------- #


def _balsa_emd(
    samples_by_batch: _SamplesByBatch,
    reduction: str = "sum",
    sliced_directions: int = 0,
) -> npt.NDArray[np.float32]:
    """BALSA EMD acquisition score.

    For single-target (D=1): exact 1-D Wasserstein-1 distance via sorted samples.
    For multi-target (D>1): sliced-Wasserstein approximation — average W1 over
    ``sliced_directions`` random unit projections (default: max(50, 10·D)).

    Sums (or averages if ``reduction="mean"``) over the T−1 consecutive pairs.
    """
    out: List[torch.Tensor] = []

    with torch.no_grad():
        for b in range(len(samples_by_batch)):
            samples_b = samples_by_batch[b]
            T = len(samples_b)
            S, B, D = samples_b[0].shape
            device = samples_b[0].device

            R = sliced_directions if sliced_directions > 0 else max(50, 10 * D)

            # Pre-generate random directions once per batch chunk (D>1 only)
            if D > 1:
                dirs = torch.randn(R, D, device=device)
                dirs = dirs / dirs.norm(dim=1, keepdim=True)  # (R, D)

            emd_sum = torch.zeros(B, device=device)

            for t in range(T - 1):
                a = samples_b[t]  # (S, B, D)
                bv = samples_b[t + 1]  # (S, B, D)

                if D == 1:
                    # Exact 1-D W1: compare sorted order-statistics element-wise
                    a_s, _ = a[..., 0].sort(dim=0)  # (S, B)
                    b_s, _ = bv[..., 0].sort(dim=0)  # (S, B)
                    emd_t = (a_s - b_s).abs().mean(dim=0)  # (B,)
                else:
                    # Sliced-Wasserstein over R random unit directions
                    # a projection: (R, S, B) via einsum over the D dimension
                    a_proj = torch.einsum("sbd,rd->rsb", a, dirs)  # (R, S, B)
                    bv_proj = torch.einsum("sbd,rd->rsb", bv, dirs)  # (R, S, B)
                    a_ps, _ = a_proj.sort(dim=1)  # (R, S, B)
                    bv_ps, _ = bv_proj.sort(dim=1)  # (R, S, B)
                    # W1 per direction per batch item, then average over R
                    emd_t = (a_ps - bv_ps).abs().mean(dim=1).mean(dim=0)  # (B,)

                emd_sum = emd_sum + emd_t

            if reduction == "mean" and T > 1:
                emd_sum = emd_sum / (T - 1)

            out.append(emd_sum.clamp(min=0.0))

    return torch.cat(out).cpu().numpy()


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

_VALID_METHODS = frozenset({"bald", "balsa_kl_pair", "balsa_kl_grid", "balsa_emd"})


def acquisition_score(
    estimator: Any,
    X: Union[pd.DataFrame, npt.NDArray],
    method: str = "balsa_kl_pair",
    num_mc_samples: int = 50,
    num_flow_samples: int = 200,
    batch_size: Optional[int] = None,
    grid_size: int = 200,
    grid_range: Optional[Tuple[float, float]] = None,
    reduction: str = "sum",
    sliced_directions: int = 0,
) -> npt.NDArray[np.float32]:
    """Compute a BALSA-style acquisition score for pool-based active learning.

    Each method returns one non-negative score per input row.  Higher score →
    more informative point to label next.

    Parameters
    ----------
    estimator : NODERegressor (head_type="flow") or FlowHeadRegressor
        A **fitted** flow-head estimator.  The estimator must have at least one
        dropout rate > 0 (``input_dropout``, ``tree_dropout``, or the conditioner
        ``dropout``) for the MC passes to differ from one another.
        The BALSA paper recommends low dropout (~0.05).
    X : DataFrame or array, shape (N, F)
        Pool of candidate inputs to score.
    method : {"bald", "balsa_kl_pair", "balsa_kl_grid", "balsa_emd"}
        Acquisition function to use.

        ``"bald"``
            BALD_H mutual information (NFlows Out baseline).
            Same quantity as ``predict_uncertainty``'s ``knowledge_uncertainty``
            column.  No distribution-distance overhead.

        ``"balsa_kl_pair"``  *(recommended default)*
            Σ_{t} KL(p_t ∥ p_{t+1}) over consecutive pairs.  Nearly free
            on top of the BALD collection loop.  Works for D > 1.

        ``"balsa_kl_grid"``
            Σ_t KL(p_t ∥ p̄) via an adaptive quantile grid and trapezoidal
            integration.  Deterministic (zero MC variance).  **Single-target
            (D = 1) only** — it integrates the joint density along a 1-D grid,
            so multi-target flows raise ``NotImplementedError`` (use
            ``"balsa_kl_pair"`` or ``"balsa_emd"`` instead).  Pass
            ``grid_range=(lo, hi)`` to use a fixed grid instead of the adaptive
            per-query one.

        ``"balsa_emd"``
            Σ_t W₁(p_t, p_{t+1}) via sorted samples (D=1) or
            sliced-Wasserstein with ``sliced_directions`` random projections
            (D > 1).

    num_mc_samples : int
        Number of MC-dropout forward passes T (default 50).
    num_flow_samples : int
        Samples drawn from each flow pass S (default 200).
    batch_size : int or None
        Optional batch size for FlowHeadRegressor pool scoring. If ``None``
        (default), the estimator's own iterator / configured batch size is
        used when available. For NODERegressor, batching always follows
        ``estimator.get_iterator``.
    grid_size : int
        Number of grid nodes G for ``"balsa_kl_grid"`` (default 200).
    grid_range : (float, float) or None
        Fixed ``(lo, hi)`` target range for the grid.  ``None`` (default) uses
        an adaptive per-query quantile grid — robust to OOD inputs and unknown
        target ranges.
    reduction : {"sum", "mean"}
        How to aggregate over the T (or T−1) pair terms (default ``"sum"``,
        matching the BALSA paper).
    sliced_directions : int
        Number of random projection directions for sliced-Wasserstein (D > 1,
        ``"balsa_emd"`` only).  0 (default) → ``max(50, 10 * D)``.

    Returns
    -------
    scores : ndarray, shape (N,), dtype float32
        Non-negative acquisition score per input row.

    Raises
    ------
    TypeError
        If *estimator* is not a ``NODERegressor`` (flow head) or
        ``FlowHeadRegressor``.
    ValueError
        If *method* is not one of the valid options.

    Examples
    --------
    **NODE flow head:**

    >>> from mother.ml.models.m_node import NODERegressor
    >>> from mother.ml.models.balsa_acquisition import acquisition_score
    >>>
    >>> reg = NODERegressor(head_type="flow", input_dropout=0.05)
    >>> reg.fit(X_train, y_train)
    >>>
    >>> scores = acquisition_score(reg, X_pool, method="balsa_kl_pair")
    >>> top10 = scores.argsort()[::-1][:10]

    **Standalone FlowHeadRegressor** (requires ``mlp_dropout > 0`` in the MLP conditioner):

    >>> from mother.ml.models.m_heads import FlowHeadRegressor
    >>> reg = FlowHeadRegressor(flow_type="NSF", mlp_dropout=0.05)
    >>> reg.fit(X_train, y_train)
    >>>
    >>> scores = acquisition_score(reg, X_pool, method="bald")

    **All four methods:**

    >>> methods = ["bald", "balsa_kl_pair", "balsa_kl_grid", "balsa_emd"]
    >>> for m in methods:
    ...     s = acquisition_score(reg, X_pool, method=m)
    ...     print(f"{m}: mean={s.mean():.4f}")

    References
    ----------
    Werner & Schmidt-Thieme (2025). *Bayesian Active Learning by Distribution
    Disagreement (BALSA).* arXiv:2501.01248.

    Berry & Meger (2023). *Normalizing Flow Ensembles for Rich Aleatoric and
    Epistemic Uncertainty Modeling.* AAAI 2023; arXiv:2308.13498.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {sorted(_VALID_METHODS)}, got {method!r}.")
    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}.")

    dists_by_batch, samples_by_batch = _collect_mc_flow_distributions(
        estimator,
        X,
        num_mc_samples,
        num_flow_samples,
        batch_size=batch_size,
    )

    if method == "bald":
        return _bald_score(dists_by_batch, samples_by_batch)

    if method == "balsa_kl_pair":
        return _balsa_kl_pair(dists_by_batch, samples_by_batch, reduction=reduction)

    if method == "balsa_kl_grid":
        return _balsa_kl_grid(
            dists_by_batch,
            samples_by_batch,
            grid_size=grid_size,
            grid_range=grid_range,
            reduction=reduction,
        )

    # method == "balsa_emd"
    return _balsa_emd(
        samples_by_batch,
        reduction=reduction,
        sliced_directions=sliced_directions,
    )
