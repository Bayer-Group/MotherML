"""Training losses for Mother's PyTorch neural-network estimators.

The losses mask missing labels, apply class weights, and average over
ensemble members and tasks, so estimators such as TabM (and later NODE) share
one implementation of MSE, MAE, log-cosh, Huber, Tweedie, cross-entropy, focal,
and user-supplied losses.
"""

from typing import Any, Callable, TypeAlias

import numpy as np
import numpy.typing as npt

from mother.errors import ExtrasDependencyImportError

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:
    raise ExtrasDependencyImportError("torch", error) from error

CustomLoss: TypeAlias = Callable[[Tensor, Tensor], Tensor]


def inverse_frequency_alpha(targets: npt.NDArray[Any]) -> float | npt.NDArray[np.float64]:
    """Return the focal-loss positive-class weight from inverse class frequency.

    Lin et al. (2017), "Focal Loss for Dense Object Detection", Section 3.1:
    "α may be set by inverse class frequency". Normalizing the weights
    ``1 / p`` and ``1 / (1 - p)`` to sum to one gives ``alpha = 1 - p``, the
    fraction of negative labels, where ``p`` is the positive-class fraction.

    Args:
        targets: Encoded binary labels: a ``(rows,)`` vector of class indices
            with ``-1`` for missing labels, or a ``(rows, tasks)`` matrix of
            ``0``/``1`` labels with ``NaN`` for missing labels.

    Returns:
        One alpha for a label vector, or one alpha per task for a matrix.
    """
    if targets.ndim == 1:
        return float(1 - np.mean(targets[targets >= 0] == 1))
    return 1 - np.nanmean(targets, axis=0)


def focal_alpha_from_strength(targets: npt.NDArray[Any], strength: float) -> float | npt.NDArray[np.float64]:
    """Return per-task focal alpha, moved from 0.5 by ``strength`` toward or away from inverse frequency.

    ``alpha = 0.5 + strength * (inverse_frequency_alpha - 0.5)``. ``strength``
    ``1`` is inverse class frequency, ``0`` weights both classes equally, and
    negative values give the rare class less weight. Lin et al. (2017) tuned
    alpha jointly with gamma and found that with gamma 2 the rare class works
    best down-weighted (alpha 0.25, Section 4.1), because the focal term
    already removes most loss from the easy, frequent class. One shared
    strength keeps each task's alpha tied to its own class balance while
    tuning a single parameter.

    Args:
        targets: Encoded binary labels, as for :func:`inverse_frequency_alpha`.
        strength: Value in [-1, 1].

    Returns:
        One alpha for a label vector, or one alpha per task for a matrix.
    """
    return 0.5 + strength * (inverse_frequency_alpha(targets) - 0.5)


class EnsembleLoss(nn.Module):
    """Train every ensemble member against the same targets.

    Ensemble models such as TabM predict ``k`` outputs per row (one per
    member), so predictions have shape ``(batch, members, outputs)``. Models
    without ensemble members, such as NODE, pass a members axis of size 1. Each member is
    scored independently and the per-member losses are averaged, so every
    member receives its own gradient signal.

    Two target layouts are supported:

    * **Single-target classification**: ``targets`` is a ``(batch,)`` vector
      of encoded class indices, with ``-1`` marking missing labels.
      Predictions are ``(batch, members, classes)`` logits.
    * **Everything else** (regression and multitarget classification):
      ``targets`` is a ``(batch, tasks)`` matrix with ``NaN`` marking missing
      labels. Predictions are ``(batch, members, tasks)``. The loss is
      averaged over observed labels per task, then averaged across the tasks
      that have at least one observed label in the batch, so every task
      counts equally regardless of how many labels it has.
    """

    def __init__(
        self,
        classification: bool,
        multitarget: bool = False,
        loss: str | CustomLoss | None = None,
        tweedie_variance_power: float = 1.5,
        huber_delta: float = 1.0,
        focal_alpha: float | npt.NDArray[np.float64] = 0.5,
        focal_gamma: float = 2.0,
        class_weights: Tensor | None = None,
    ) -> None:
        """Create the regression or classification loss.

        Args:
            classification: Use cross-entropy when true, otherwise mean
                squared error.
            multitarget: Use binary cross-entropy for multitarget classification.
            loss: ``"mse"``, ``"mae"``, ``"logcosh"``, ``"huber"``,
                ``"tweedie"``, ``"cross_entropy"``, ``"focal"``, or a callable
                returning unreduced losses. ``None`` selects MSE or
                cross-entropy.
            tweedie_variance_power: Tweedie power in (1, 2); predictions are
                log means.
            huber_delta: Residual size, in standardized target units, where
                Huber loss switches from quadratic to linear.
            focal_alpha: Focal weight of the positive class, either shared or
                one value per task (see :func:`inverse_frequency_alpha`).
            focal_gamma: Focal down-weighting of well-classified samples.
            class_weights: Optional per-class weights, shaped ``(classes,)``
                for single-target or ``(tasks, 2)`` for multitarget
                classification. Losses become weighted means.
        """
        super().__init__()
        self.classification = classification
        self.multitarget = multitarget
        self.loss = loss if loss is not None else ("cross_entropy" if classification else "mse")
        self.tweedie_variance_power = tweedie_variance_power
        self.huber_delta = huber_delta
        self.register_buffer("focal_alpha", torch.as_tensor(focal_alpha, dtype=torch.float32))
        self.focal_gamma = focal_gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the scalar training loss averaged over ensemble members.

        Args:
            predictions: Member outputs shaped ``(batch, members, outputs)``.
            targets: A ``(batch, tasks)`` matrix with NaN for missing labels,
                or an encoded single-target class vector with -1 for missing
                labels.

        Returns:
            Scalar loss.
        """
        if self.classification and not self.multitarget:
            return self._single_target_classification_loss(predictions, targets)
        return self._task_matrix_loss(predictions, targets)

    # ------------------------------------------------------------------
    # Single-target classification: targets are class indices, -1 = missing
    # ------------------------------------------------------------------

    def _single_target_classification_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Softmax loss over class logits, one row per (sample, member) pair."""
        observed = targets >= 0
        if not observed.any():
            raise ValueError("A training batch must contain at least one observed target.")

        # Flatten (observed_batch, members, classes) -> (observed_batch * members, classes)
        # and repeat each label once per member so rows line up.
        n_members = predictions.shape[1]
        logits = predictions[observed].flatten(0, 1)
        labels = targets[observed].repeat_interleave(n_members, dim=0)
        class_weights = None if self.class_weights is None else self.class_weights.to(logits.dtype)

        if callable(self.loss):
            losses = self._custom_losses(logits, labels)
        elif self.loss == "focal":
            losses = self._softmax_focal_losses(logits, labels)
        else:
            # PyTorch already computes the class-weighted mean.
            return nn.functional.cross_entropy(logits, labels, weight=class_weights)

        if class_weights is None:
            return losses.mean()
        sample_weights = class_weights[labels]
        return (sample_weights * losses).sum() / sample_weights.sum()

    def _softmax_focal_losses(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Focal loss ``-alpha * (1 - p_t)^gamma * log(p_t)`` on softmax probabilities."""
        log_pt = logits.log_softmax(dim=-1).gather(1, labels[:, None]).squeeze(1)
        alpha = torch.where(labels == 1, self.focal_alpha, 1 - self.focal_alpha)
        return -alpha * self._focal_modulation(log_pt) * log_pt

    def _focal_modulation(self, log_pt: Tensor) -> Tensor:
        """Focal factor ``(1 - p_t)^gamma``, kept differentiable for ``gamma < 1``.

        The derivative of ``x^gamma`` is infinite at ``x = 0`` when ``gamma < 1``,
        so a confidently correct sample (``p_t == 1``) would yield ``inf * 0 = NaN``
        gradients. Clamping ``1 - p_t`` away from zero avoids that.
        """
        one_minus_pt = -torch.expm1(log_pt)
        return one_minus_pt.clamp_min(torch.finfo(log_pt.dtype).eps).pow(self.focal_gamma)

    # ------------------------------------------------------------------
    # Regression and multitarget classification: (batch, tasks) matrix, NaN = missing
    # ------------------------------------------------------------------

    def _task_matrix_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Per-task masked mean of element-wise losses, averaged over present tasks."""
        observed = torch.isfinite(targets)  # (batch, tasks)
        task_present = observed.any(dim=0)  # (tasks,)
        if not task_present.any():
            raise ValueError("A training batch must contain at least one observed target.")

        # Replace missing labels with 0 so the loss stays finite, then broadcast
        # (batch, tasks) -> (batch, members, tasks) to match the predictions.
        targets = torch.where(observed, targets, 0).unsqueeze(1).expand_as(predictions)
        observed = observed.unsqueeze(1).expand_as(predictions)

        losses = self._elementwise_losses(predictions, targets)

        weights = observed.to(losses.dtype)
        if self.class_weights is not None:
            # class_weights is (tasks, 2): column 0 weights negatives, column 1 positives.
            task_weights = self.class_weights.to(losses.dtype)
            weights = weights * torch.where(targets == 1, task_weights[:, 1], task_weights[:, 0])

        # torch.where (rather than multiplying by a 0 weight) keeps a NaN/inf
        # loss at a masked position from leaking into the sum.
        weighted_loss_per_task = torch.where(observed, losses * weights, 0).sum(dim=(0, 1))
        weight_per_task = weights.sum(dim=(0, 1))
        return (weighted_loss_per_task[task_present] / weight_per_task[task_present]).mean()

    def _elementwise_losses(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Unreduced loss for every (sample, member, task) element."""
        if callable(self.loss):
            return self._custom_losses(predictions, targets)
        if self.classification:
            return self._sigmoid_losses(predictions, targets)
        if self.loss == "tweedie":
            return self._tweedie_losses(predictions, targets)
        if self.loss == "mae":
            return (predictions - targets).abs()
        if self.loss == "logcosh":
            return self._logcosh_losses(predictions - targets)
        if self.loss == "huber":
            return nn.functional.huber_loss(predictions, targets, reduction="none", delta=self.huber_delta)
        return (predictions - targets).square()

    def _sigmoid_losses(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Binary cross-entropy per task, optionally reshaped into focal loss."""
        losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.loss != "focal":
            return losses
        # For binary cross-entropy, losses == -log(p_t).
        alpha = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        return alpha * self._focal_modulation(-losses) * losses

    def _tweedie_losses(self, log_means: Tensor, targets: Tensor) -> Tensor:
        """Tweedie negative log-likelihood (up to a constant) for log-mean predictions."""
        power = self.tweedie_variance_power
        return -targets * torch.exp((1 - power) * log_means) / (1 - power) + torch.exp((2 - power) * log_means) / (
            2 - power
        )

    @staticmethod
    def _logcosh_losses(residuals: Tensor) -> Tensor:
        """Numerically stable ``log(cosh(r)) = |r| + log1p(exp(-2|r|)) - log(2)``."""
        abs_residuals = residuals.abs()
        return abs_residuals + nn.functional.softplus(-2 * abs_residuals) - np.log(2)

    def _custom_losses(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Apply a user loss, which must return one loss per target element."""
        losses = self.loss(predictions, targets)
        if not isinstance(losses, Tensor) or losses.shape != targets.shape:
            raise ValueError(
                f"A custom loss_function must return unreduced losses with shape {tuple(targets.shape)}, "
                f"got {getattr(losses, 'shape', type(losses))}; use reduction='none' for PyTorch losses."
            )
        return losses
