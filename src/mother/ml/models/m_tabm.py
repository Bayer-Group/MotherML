"""Mother estimators for TabM, trained with skorch."""

import logging
from typing import Any, Callable, Literal, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from optuna.trial import Trial
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import Tags
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, column_or_1d, validate_data

import mother.ml.properties as props
from mother.errors import ExtrasDependencyImportError
from mother.ml.core import AbstractMotherPipeline

try:
    import torch
    from skorch import NeuralNet
    from torch import Tensor, nn
except ImportError as error:
    raise ExtrasDependencyImportError("tabm", error) from error

from mother.ml.models.tabm_network import Model, make_parameter_groups
from mother.ml.models.tabm_utils import compute_bins

Features: TypeAlias = pd.DataFrame | npt.NDArray[Any]
CustomLoss: TypeAlias = Callable[["Tensor", "Tensor"], "Tensor"]
module_logger = logging.getLogger(__name__)
_EMBEDDING_TYPES = ("linear", "linear_relu", "periodic", "pwl")
_EMBEDDING_DEFAULTS = {
    "d_embedding": 16,
    "num_emb_n_bins": 48,
    "num_emb_n_frequencies": 48,
    "num_emb_frequency_init_scale": 0.01,
    "num_emb_activation": True,
}


class _EnsembleLoss(nn.Module):
    """Train every TabM ensemble member against the same targets.

    TabM predicts ``k`` outputs per row (one per ensemble member), so
    predictions have shape ``(batch, members, outputs)``. Each member is
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
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        class_weights: Tensor | None = None,
    ) -> None:
        """Create the regression or classification loss.

        Args:
            classification: Use cross-entropy when true, otherwise mean
                squared error.
            multitarget: Use binary cross-entropy for multitarget classification.
            loss: ``"mse"``, ``"mae"``, ``"logcosh"``, ``"tweedie"``,
                ``"cross_entropy"``, ``"focal"``, or a callable returning
                unreduced losses. ``None`` selects MSE or cross-entropy.
            tweedie_variance_power: Tweedie power in (1, 2); predictions are
                log means.
            focal_alpha: Focal weight of the positive class.
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
        self.focal_alpha = focal_alpha
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
        return -alpha * (1 - log_pt.exp()).pow(self.focal_gamma) * log_pt

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
        return (predictions - targets).square()

    def _sigmoid_losses(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Binary cross-entropy per task, optionally reshaped into focal loss."""
        losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.loss != "focal":
            return losses
        probabilities = logits.sigmoid()
        pt = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        return alpha * (1 - pt).pow(self.focal_gamma) * losses

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


class _TabMNet(NeuralNet):
    """Skorch wrapper that configures TabM's grouped optimizer parameters."""

    def initialize_optimizer(self, triggered_directly: Any = None) -> "_TabMNet":
        """Create AdamW parameter groups during skorch initialization.

        TabM excludes biases and selected normalization/embedding parameters
        from weight decay. Skorch calls this hook after constructing the
        PyTorch module.
        """
        self.optimizer_ = self.optimizer(
            make_parameter_groups(self.module_),
            lr=self.lr,
            **self.get_params_for("optimizer"),
        )
        return self


class _BaseTabMMother(BaseEstimator, AbstractMotherPipeline):
    """Shared sklearn-style preprocessing and training for TabM estimators.

    Mother owns input validation, feature/target preprocessing, and
    model-specific dimension discovery. Skorch owns module initialization and
    the mini-batch training loop. The fitted skorch network is stored in
    ``net_``.
    """

    net_: _TabMNet

    def __init__(
        self,
        *,
        target_type: Literal["single_target", "multi_target"] = "single_target",
        model_type: props.ModelType | None = None,
        n_blocks: int | None = None,
        d_block: int = 256,
        dropout: float = 0.1,
        k: int = 32,
        arch_type: Literal["tabm", "tabm-mini", "auto"] = "tabm",
        num_emb_type: Literal["none", "auto", "linear", "linear_relu", "periodic", "pwl"] = "none",
        d_embedding: int = 16,
        num_emb_n_bins: int = 48,
        num_emb_n_frequencies: int = 48,
        num_emb_frequency_init_scale: float = 0.01,
        num_emb_activation: bool = True,
        loss_function: Literal["auto", "mse", "mae", "logcosh", "tweedie", "cross_entropy", "focal"]
        | CustomLoss
        | None = None,
        tweedie_variance_power: float = 1.5,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        lr: float = 0.002,
        weight_decay: float = 0.0003,
        max_epochs: int = 100,
        batch_size: int = 256,
        cat_features: Sequence[str | int] | None = None,
        scale_features: bool = True,
        random_state: int = 42,
        device: str = "cpu",
        verbose: int = 0,
    ) -> None:
        """Configure a TabM estimator.

        Args:
            target_type: ``"single_target"`` (default) accepts one target.
                ``"multi_target"`` trains a shared network on target columns;
                classification requires two observed classes per column.
                A target vector is treated as a one-column matrix in
                multitarget mode, retaining multitarget output shapes.
                Both modes allow missing target labels, but not missing inputs.
            model_type: Task type, matching the other Mother estimators.
                Regressors accept ``"regression"``; classifiers accept
                ``"classification_binary"`` or ``"classification_multiclass"``.
                ``None`` (default) infers the task from the estimator class and
                training labels. ``"classification_binary"`` requires exactly
                two classes per task; ``"classification_multiclass"`` requires
                ``target_type="single_target"``.
            n_blocks: Number of blocks in the MLP backbone. ``None`` (default)
                follows pytabkit's depth heuristic: two blocks when PWL
                embeddings are active on numeric features, otherwise three.
                An explicit positive integer is used as given. MotherTuner
                searches depths from one to four regardless of this setting.
                The fitted depth is available as ``n_blocks_``.
            d_block: Width of each MLP block.
            dropout: Dropout rate used by the MLP backbone.
            k: Number of ensemble members.
            arch_type: Ensemble architecture. ``"tabm"`` (default) gives
                members their own small adjustments throughout the shared
                network. ``"tabm-mini"`` uses member-specific input adjustments
                and output heads with a shared backbone. ``"auto"`` lets
                MotherTuner compare both variants, starting with ``"tabm"``;
                direct fitting requires a specific variant. Explicit variants
                remain fixed during tuning. ``k`` controls member count for
                either variant.
            num_emb_type: Numerical input representation. ``"none"`` gives
                each numeric value directly to the network. The other options
                turn a value into learned signals: ``"linear"`` learns a
                separate rescaling and offset for each feature;
                ``"linear_relu"`` adds a threshold-like step that can treat
                lower and higher values differently; ``"periodic"`` uses
                repeating wave-like signals for cyclic patterns; and ``"pwl"``
                learns a representation across value intervals, which can
                capture different effects in different parts of a feature's
                range. These are alternative representations, not new data.
                ``"auto"`` lets MotherTuner compare no embedding and the four
                embedding types;
                choose a specific type for direct fitting.
            d_embedding: Embedding width per numeric feature when embeddings
                are enabled.
            num_emb_n_bins: Requested quantile-bin count for PWL embeddings.
            num_emb_n_frequencies: Number of learned frequencies per feature
                for periodic embeddings.
            num_emb_frequency_init_scale: Standard deviation used to initialize
                periodic embedding frequencies.
            num_emb_activation: Whether to apply ReLU after periodic
                embeddings.
            loss_function: Training loss, mirroring the losses MotherTuner
                searches for CatBoost. ``None`` (default) uses ``"mse"`` for
                regression and ``"cross_entropy"`` for classification.
                Regressors also accept ``"mae"``, ``"logcosh"``, and
                ``"tweedie"``; Tweedie needs nonnegative targets with a
                positive value per task and predicts through a log link.
                Classifiers also accept ``"focal"`` for binary tasks.
                ``"auto"`` lets MotherTuner choose between the task's losses,
                offering Tweedie and focal only when the data allow them;
                choose a specific loss for direct fitting.
                A callable ``loss(predictions, targets)`` supplies a custom
                loss and is never tuned. It must return unreduced losses with
                the targets' shape, like PyTorch losses with
                ``reduction="none"``; Mother then masks missing labels,
                applies class weights, and averages over members and tasks.
                Regression and multitarget classification pass
                ``(rows, members, tasks)`` outputs and float targets, using
                standardized targets for regression. Custom regression
                outputs are treated as standardized predictions and mapped
                back with the target scaler, so the loss must compare them
                with targets directly (identity link); log-link losses such as
                Poisson are unsupported. Scale parameters such as Huber's
                ``delta`` are in target standard deviations. Single-target
                classification passes ``(rows * members, classes)`` logits and
                integer class targets. Use a module-level function or
                ``functools.partial`` so fitted models can be pickled.
            tweedie_variance_power: Tweedie variance power, in (1, 2).
            focal_alpha: Focal loss weight of the positive class, in (0, 1).
                ``0.5`` weights both classes equally.
            focal_gamma: Positive focal loss exponent; larger values
                down-weight well-classified samples more strongly.
            lr: AdamW learning rate.
            weight_decay: AdamW weight decay for parameters not excluded by
                TabM's parameter-group rules.
            max_epochs: Fixed number of skorch training epochs.
            batch_size: Number of samples per training mini-batch.
            cat_features: Categorical DataFrame column names or array column
                indices. All other columns are treated as numeric.
            scale_features: Standardize numeric features using training data.
            random_state: Seed used for model initialization and training.
            device: PyTorch device, such as ``"cpu"`` or ``"cuda"``.
            verbose: Verbosity level passed to skorch.
        """
        self.target_type = target_type
        self.model_type = model_type
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.k = k
        self.arch_type = arch_type
        self.num_emb_type = num_emb_type
        self.d_embedding = d_embedding
        self.num_emb_n_bins = num_emb_n_bins
        self.num_emb_n_frequencies = num_emb_n_frequencies
        self.num_emb_frequency_init_scale = num_emb_frequency_init_scale
        self.num_emb_activation = num_emb_activation
        self.loss_function = loss_function
        self.tweedie_variance_power = tweedie_variance_power
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.cat_features = cat_features
        self.scale_features = scale_features
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters using sklearn's estimator convention.

        Args:
            deep: Whether nested estimator parameters should be included.

        Returns:
            Constructor parameters, optionally including nested parameters.
        """
        return super().get_params(deep=deep)

    def set_params(self, **params: Any) -> "_BaseTabMMother":
        """Update constructor parameters using sklearn's estimator convention.

        Args:
            **params: Parameter names and replacement values.

        Returns:
            This estimator, for chaining.
        """
        return super().set_params(**params)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = self.target_type == "multi_target"
        tags.target_tags.single_output = self.target_type == "single_target"
        tags.target_tags.two_d_labels = self.target_type == "multi_target"
        return tags

    def _validate_parameters(self) -> None:
        """Raise ``ValueError`` when a constructor parameter is unsupported."""
        if self.target_type not in ("single_target", "multi_target"):
            raise ValueError("target_type must be 'single_target' or 'multi_target'.")
        allowed_model_types = (
            ("classification_binary", "classification_multiclass")
            if isinstance(self, ClassifierMixin)
            else ("regression",)
        )
        if self.model_type is not None and self.model_type not in allowed_model_types:
            raise ValueError(
                f"model_type for {type(self).__name__} must be None or one of {allowed_model_types}, "
                f"got {self.model_type!r}."
            )
        if self.model_type == "classification_multiclass" and self.target_type == "multi_target":
            raise ValueError("TabM multitarget classification supports only binary tasks.")
        for name in (
            "n_blocks",
            "d_block",
            "k",
            "max_epochs",
            "batch_size",
            "d_embedding",
            "num_emb_n_frequencies",
        ):
            value = getattr(self, name)
            if name == "n_blocks" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.arch_type not in ("tabm", "tabm-mini", "auto"):
            raise ValueError("arch_type must be 'tabm', 'tabm-mini', or 'auto'.")
        if self.num_emb_type not in ("none", "auto", *_EMBEDDING_TYPES):
            raise ValueError(f"num_emb_type must be 'none', 'auto', or one of {_EMBEDDING_TYPES}.")
        if isinstance(self.num_emb_n_bins, bool) or not isinstance(self.num_emb_n_bins, int) or self.num_emb_n_bins < 2:
            raise ValueError("num_emb_n_bins must be an integer of at least 2.")
        if not np.isfinite(self.num_emb_frequency_init_scale) or self.num_emb_frequency_init_scale <= 0:
            raise ValueError("num_emb_frequency_init_scale must be positive and finite.")
        if not isinstance(self.num_emb_activation, bool):
            raise ValueError("num_emb_activation must be a boolean.")
        allowed_losses = self._task_losses()
        if (
            self.loss_function is not None
            and not callable(self.loss_function)
            and self.loss_function not in ("auto", *allowed_losses)
        ):
            raise ValueError(
                f"loss_function for {type(self).__name__} must be None, 'auto', a callable, or one of {allowed_losses}."
            )
        if not 1 < self.tweedie_variance_power < 2:
            raise ValueError("tweedie_variance_power must be in (1, 2).")
        if not 0 < self.focal_alpha < 1:
            raise ValueError("focal_alpha must be in (0, 1).")
        if not np.isfinite(self.focal_gamma) or self.focal_gamma <= 0:
            raise ValueError("focal_gamma must be positive and finite.")
        if isinstance(self, ClassifierMixin) and self.auto_class_weights not in (
            "default",
            None,
            "Balanced",
            "SqrtBalanced",
            "None",
        ):
            raise ValueError("auto_class_weights must be 'default', None, 'Balanced', 'SqrtBalanced', or 'None'.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite.")

    def _prepare_features(self, X: Features, *, fitting: bool) -> dict[str, npt.NDArray[Any]]:
        """Validate and encode features into the inputs expected by TabM.

        Numeric and categorical data are returned separately because the
        PyTorch module accepts them as distinct inputs. During fitting this
        method learns the feature schema and preprocessing; prediction reuses
        that fitted state.

        Args:
            X: Dense feature matrix or DataFrame.
            fitting: Reset and learn preprocessing state when true; otherwise
                validate against and reuse the fitted state.

        Returns:
            A mapping containing ``x_num`` and/or ``x_cat`` arrays.
        """
        values = validate_data(self, X, reset=fitting, dtype=None, ensure_all_finite=False)
        if fitting:
            self.categorical_indices_: list[int] = []
            for feature in self.cat_features if self.cat_features is not None else ():
                if isinstance(feature, str):
                    if not isinstance(X, pd.DataFrame) or feature not in X.columns:
                        raise ValueError(f"Categorical feature {feature!r} requires a matching DataFrame column.")
                    index = X.columns.get_loc(feature)
                    if not isinstance(index, int):
                        raise ValueError("Categorical column names must be unique.")
                elif isinstance(feature, int) and not isinstance(feature, bool) and 0 <= feature < values.shape[1]:
                    index = feature
                else:
                    raise ValueError(f"Invalid categorical feature: {feature!r}.")
                if index in self.categorical_indices_:
                    raise ValueError(f"Duplicate categorical feature: {feature!r}.")
                self.categorical_indices_.append(index)
            self.numeric_indices_ = [i for i in range(values.shape[1]) if i not in self.categorical_indices_]

        result: dict[str, npt.NDArray[Any]] = {}
        if self.numeric_indices_:
            numeric = np.asarray(values[:, self.numeric_indices_], dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError("TabM numeric features must be finite; impute missing values before fitting.")
            if fitting:
                self.scaler_ = StandardScaler() if self.scale_features else None
                if self.scaler_ is not None:
                    self.scaler_.fit(numeric)
            if self.scaler_ is not None:
                numeric = self.scaler_.transform(numeric)
            result["x_num"] = np.asarray(numeric, dtype=np.float32)

        if self.categorical_indices_:
            categorical = values[:, self.categorical_indices_]
            if pd.isna(categorical).any():
                raise ValueError("TabM categorical features must be nonmissing; impute them before fitting.")
            if fitting:
                self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
                self.encoder_.fit(categorical)
                self.cat_cardinalities_ = [len(categories) for categories in self.encoder_.categories_]
            encoded = self.encoder_.transform(categorical)
            for index, cardinality in enumerate(self.cat_cardinalities_):
                encoded[encoded[:, index] == -1, index] = cardinality
            result["x_cat"] = np.asarray(encoded, dtype=np.int64)
        elif fitting:
            self.cat_cardinalities_ = []
        return result

    def fit(self, X: Features, y: Any) -> "_BaseTabMMother":
        """Fit preprocessing and train the TabM ensemble.

        Mother derives the feature counts and output dimensions from the
        training data, then supplies them as module parameters. Skorch does not
        infer TabM's input size; it initializes that configured module and
        performs its mini-batch training loop.

        Args:
            X: Dense training features as a DataFrame or NumPy array.
            y: Target values or labels, with one column per task in
                ``multi_target`` mode. Missing labels are masked during training.

        Returns:
            This fitted estimator.
        """
        self._validate_parameters()
        # A failed refit must not expose a stale trained network with new encoders.
        if hasattr(self, "net_"):
            del self.net_
        if self.arch_type == "auto":
            raise ValueError("arch_type='auto' is only supported during MotherTuner optimization.")
        if self.num_emb_type == "auto":
            raise ValueError("num_emb_type='auto' is only supported during MotherTuner optimization.")
        if self.loss_function == "auto":
            raise ValueError("loss_function='auto' is only supported during MotherTuner optimization.")
        loss = self.loss_ = self._resolved_loss()
        targets = self._target_array(y)
        if len(targets) != len(X):
            raise ValueError("X and y must contain the same number of samples.")
        classification = isinstance(self, ClassifierMixin)
        self.n_outputs_ = targets.shape[1]
        if classification:
            prepared_targets, output_dim = self._prepare_classification_targets(targets)
            if loss == "focal" and output_dim > 2:
                raise ValueError("Focal loss supports only binary classification tasks.")
            self.class_weights_ = self._class_weights(prepared_targets, output_dim, loss)
            observed_rows = (~pd.isna(targets)).any(axis=1)
        else:
            numeric_targets = np.asarray(np.where(pd.isna(targets), np.nan, targets), dtype=np.float64)
            if np.isinf(numeric_targets).any():
                raise ValueError("TabM regression targets must be finite or missing (NaN).")
            observed_rows = np.isfinite(numeric_targets).any(axis=1)
            if not np.isfinite(numeric_targets).any(axis=0).all():
                raise ValueError("Each regression task requires at least one observed finite target.")
            if loss == "tweedie":
                if not self._tweedie_compatible(numeric_targets):
                    raise ValueError("Tweedie loss requires nonnegative targets with a positive value per task.")
                # Rescaling without centering keeps targets nonnegative for the log link.
                self.target_scaler_ = StandardScaler(with_mean=False).fit(numeric_targets)
            else:
                self.target_scaler_ = StandardScaler().fit(numeric_targets)
            prepared_targets = self.target_scaler_.transform(numeric_targets).astype(np.float32)
            output_dim = self.n_outputs_
        if not observed_rows.all():
            module_logger.warning(
                "Excluding %d training rows with no observed target labels.", int((~observed_rows).sum())
            )
            X = X.iloc[observed_rows] if isinstance(X, pd.DataFrame) else X[observed_rows]
            prepared_targets = prepared_targets[observed_rows]

        # Mother supplies the discovered dimensions and fitted preprocessing
        # artifacts; skorch receives a fully configured PyTorch module factory.
        inputs = self._prepare_features(X, fitting=True)
        self.bins_: list[Tensor] | None = None
        if self.num_emb_type == "pwl" and self.numeric_indices_:
            if len(X) < 3:
                raise ValueError("Piecewise-linear embeddings require at least 3 training samples.")
            self.bins_ = compute_bins(torch.from_numpy(inputs["x_num"]), n_bins=min(self.num_emb_n_bins, len(X) - 1))
        self.n_blocks_ = self.n_blocks if self.n_blocks is not None else (2 if self.bins_ is not None else 3)

        # Skorch instantiates Model from these module__ parameters, including
        # the dimensions Mother discovered from X and y.
        net = _TabMNet(
            module=Model,
            module__n_num_features=len(self.numeric_indices_),
            module__cat_cardinalities=self.cat_cardinalities_,
            module__n_classes=output_dim,
            module__backbone={
                "type": "MLP",
                "n_blocks": self.n_blocks_,
                "d_block": self.d_block,
                "dropout": self.dropout,
            },
            module__bins=self.bins_,
            module__num_embeddings=self._numerical_embedding_config(),
            module__arch_type=self.arch_type,
            module__k=self.k,
            module__share_training_batches=True,
            criterion=_EnsembleLoss,
            criterion__classification=classification,
            criterion__multitarget=self.target_type == "multi_target",
            criterion__loss=loss,
            criterion__tweedie_variance_power=self.tweedie_variance_power,
            criterion__focal_alpha=self.focal_alpha,
            criterion__focal_gamma=self.focal_gamma,
            criterion__class_weights=(
                torch.as_tensor(self.class_weights_, dtype=torch.float32)
                if classification and self.class_weights_ is not None
                else None
            ),
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=self.weight_decay,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            iterator_valid__generator=torch.Generator().manual_seed(self.random_state),
            train_split=None,
            device=self.device,
            verbose=self.verbose,
        )
        device = torch.device(self.device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.random_state)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(self.random_state)
            # Skorch now owns batching, epochs, loss evaluation, and optimizer
            # steps; its history is available on the fitted ``net_``.
            net.fit(inputs, prepared_targets)
        self.net_ = net
        return self

    def _class_weights(
        self, prepared_targets: npt.NDArray[Any], output_dim: int, loss: str
    ) -> npt.NDArray[np.float64] | None:
        """Compute CatBoost-style class weights from the observed training labels."""
        mode = self.auto_class_weights
        if mode == "default":
            mode = "Balanced" if self.target_type == "single_target" and loss == "cross_entropy" else None
        if mode in (None, "None"):
            return None
        if self.target_type == "single_target":
            counts = np.bincount(prepared_targets[prepared_targets >= 0], minlength=output_dim).astype(np.float64)
        else:
            positives = np.nansum(prepared_targets, axis=0)
            counts = np.column_stack([np.isfinite(prepared_targets).sum(axis=0) - positives, positives])
        weights = counts.max(axis=-1, keepdims=True) / counts
        return np.sqrt(weights) if mode == "SqrtBalanced" else weights

    def _task_losses(self) -> tuple[str, ...]:
        """Return the task's losses: the default first, the data-dependent one last."""
        if isinstance(self, ClassifierMixin):
            return ("cross_entropy", "focal")
        return ("mse", "mae", "logcosh", "tweedie")

    def _resolved_loss(self) -> str:
        """Return the configured loss, resolving ``None`` to the task default."""
        return self.loss_function if self.loss_function is not None else self._task_losses()[0]

    @staticmethod
    def _tweedie_compatible(targets: npt.NDArray[np.float64]) -> bool:
        """Check for nonnegative observed targets with a positive value per task."""
        observed = np.isfinite(targets)
        return bool((targets[observed] >= 0).all() and np.where(observed, targets, 0).max(axis=0).min() > 0)

    def _target_array(self, y: Any) -> npt.NDArray[Any]:
        """Validate target dimensions without replacing missing labels."""
        if self.target_type == "single_target":
            return np.asarray(column_or_1d(y))[:, None]
        values = np.asarray(y)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("multi_target requires a 2D target array with at least one column.")
        return values

    def _prepare_classification_targets(self, targets: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], int]:
        """Encode observed labels independently for each classification task."""
        missing = pd.isna(targets)
        if missing.all(axis=0).any():
            raise ValueError("Each classification task requires observed labels and at least two classes.")
        self.label_encoders_: list[LabelEncoder] = []
        encoded = np.full(targets.shape, np.nan, dtype=np.float32)
        for task in range(self.n_outputs_):
            observed = targets[~missing[:, task], task]
            # Object arrays preserve missing string labels but sklearn expects
            # numeric labels to have a numeric dtype when validating their type.
            labels = pd.Series(observed).infer_objects().to_numpy()
            if labels.dtype.kind in "biuf" and not np.isfinite(labels).all():
                raise ValueError("Classification labels must be finite or missing.")
            check_classification_targets(labels)
            encoder = LabelEncoder().fit(labels)
            if len(encoder.classes_) < 2:
                raise ValueError(f"Classification task {task} requires at least two classes.")
            if self.target_type == "multi_target" and len(encoder.classes_) != 2:
                raise ValueError("Multitarget classification supports only binary tasks (two classes per column).")
            if self.model_type == "classification_binary" and len(encoder.classes_) != 2:
                raise ValueError(
                    f"model_type='classification_binary' requires two classes, but task {task} has "
                    f"{len(encoder.classes_)}. Use model_type='classification_multiclass' or None."
                )
            encoded[~missing[:, task], task] = encoder.transform(labels)
            self.label_encoders_.append(encoder)
        if self.target_type == "single_target":
            self.label_encoder_ = self.label_encoders_[0]
            self.classes_ = self.label_encoder_.classes_
            return np.where(missing[:, 0], -1, encoded[:, 0]).astype(np.int64), len(self.classes_)
        self.classes_ = [encoder.classes_ for encoder in self.label_encoders_]
        return encoded, self.n_outputs_

    def _numerical_embedding_config(self) -> dict[str, Any] | None:
        """Build the network configuration for the selected numeric embedding."""
        if not self.numeric_indices_ or self.num_emb_type == "none":
            return None
        if self.num_emb_type == "linear":
            return {"type": "LinearEmbeddings", "d_embedding": self.d_embedding}
        if self.num_emb_type == "linear_relu":
            return {"type": "LinearReLUEmbeddings", "d_embedding": self.d_embedding}
        if self.num_emb_type == "periodic":
            return {
                "type": "PeriodicEmbeddings",
                "d_embedding": self.d_embedding,
                "n_frequencies": self.num_emb_n_frequencies,
                "frequency_init_scale": self.num_emb_frequency_init_scale,
                "activation": self.num_emb_activation,
                "lite": False,
            }
        if self.num_emb_type == "pwl":
            return {
                "type": "PiecewiseLinearEmbeddings",
                "d_embedding": self.d_embedding,
                "activation": False,
                "version": "B",
            }
        raise ValueError(f"Unsupported numerical embedding type: {self.num_emb_type!r}.")

    def _member_predictions(self, X: Features) -> Tensor:
        """Run the fitted network and concatenate ensemble predictions.

        Skorch performs inference in batches; concatenating those outputs
        restores one ``(rows, members, outputs)`` tensor for Mother to
        aggregate.
        """
        check_is_fitted(self, "net_")
        inputs = self._prepare_features(X, fitting=False)
        return torch.cat(list(self.net_.forward_iter(inputs, training=False, device="cpu")), dim=0)

    def get_hyperparameter_space(self, X: Features, y: Any, trial: Trial, prefix: str = "") -> dict[str, Any]:
        """Suggest tunable TabM parameters for a MotherTuner trial.

        Args:
            X: Training features required by Mother's tuning interface; not
                accessed by this implementation.
            y: Training targets, used to decide whether Tweedie or focal loss
                is applicable when ``loss_function="auto"``.
            trial: Optuna trial used to suggest parameter values.
            prefix: Prefix for parameter names, used by rooted pipelines.

        Returns:
            Suggested parameter values, with names prefixed when requested.
        """
        parameters = {
            prefix + "lr": trial.suggest_float(prefix + "lr", 1e-4, 1e-2, log=True),
            prefix + "weight_decay": trial.suggest_float(prefix + "weight_decay", 1e-6, 1e-2, log=True),
            prefix + "n_blocks": trial.suggest_int(prefix + "n_blocks", 1, 4),
            prefix + "d_block": trial.suggest_categorical(prefix + "d_block", [64, 128, 256, 512]),
            prefix + "dropout": trial.suggest_float(prefix + "dropout", 0.0, 0.5),
            prefix + "k": trial.suggest_categorical(prefix + "k", [8, 16, 32]),
        }
        if self.arch_type == "auto":
            parameters[prefix + "arch_type"] = trial.suggest_categorical(prefix + "arch_type", ["tabm", "tabm-mini"])
        embedding_type = self.num_emb_type
        if embedding_type == "auto":
            embedding_type = trial.suggest_categorical(prefix + "num_emb_type", ["none", *_EMBEDDING_TYPES])
            parameters[prefix + "num_emb_type"] = embedding_type
        if embedding_type in _EMBEDDING_TYPES:
            parameters[prefix + "d_embedding"] = trial.suggest_int(prefix + "d_embedding", 4, 64, log=True)
        if embedding_type == "periodic":
            parameters[prefix + "num_emb_n_frequencies"] = trial.suggest_int(
                prefix + "num_emb_n_frequencies", 8, 128, log=True
            )
            parameters[prefix + "num_emb_frequency_init_scale"] = trial.suggest_float(
                prefix + "num_emb_frequency_init_scale", 1e-3, 1e-1, log=True
            )
            parameters[prefix + "num_emb_activation"] = trial.suggest_categorical(
                prefix + "num_emb_activation", [False, True]
            )
        elif embedding_type == "pwl":
            parameters[prefix + "num_emb_n_bins"] = trial.suggest_int(prefix + "num_emb_n_bins", 2, 128, log=True)
        loss = self._resolved_loss()
        if loss == "auto":
            losses = list(self._task_losses())
            if not self._alternative_loss_applicable(y):
                losses.pop()
            loss = trial.suggest_categorical(prefix + "loss_function", losses)
            parameters[prefix + "loss_function"] = loss
        # Ranges match CatBoost's Tweedie and focal tuning in m_catboost.py.
        if loss == "tweedie":
            parameters[prefix + "tweedie_variance_power"] = trial.suggest_float(
                prefix + "tweedie_variance_power", 1.000001, 1.999999
            )
        elif loss == "focal":
            if self.loss_function == "auto":
                # CatBoost disables class weights when tuning selects focal loss.
                parameters[prefix + "auto_class_weights"] = "None"
            parameters[prefix + "focal_alpha"] = trial.suggest_float(prefix + "focal_alpha", 0.000001, 0.999999)
            parameters[prefix + "focal_gamma"] = trial.suggest_float(prefix + "focal_gamma", 0.000001, 7)
        return parameters

    def _alternative_loss_applicable(self, y: Any) -> bool:
        """Whether ``y`` allows Tweedie (regression) or focal (classification) loss."""
        targets = self._target_array(y)
        if isinstance(self, ClassifierMixin):
            return all(pd.Series(targets[:, task]).dropna().nunique() == 2 for task in range(targets.shape[1]))
        numeric = np.asarray(np.where(pd.isna(targets), np.nan, targets), dtype=np.float64)
        return self._tweedie_compatible(numeric)

    def default_parameters(self, prefix: str = "") -> dict[str, Any]:
        """Return Mother's default tuning values with an optional prefix.

        Args:
            prefix: Prefix for parameter names, used by rooted pipelines.

        Returns:
            Default values for parameters exposed to MotherTuner.
        """
        parameters = {
            prefix + name: value
            for name, value in {
                "lr": 0.002,
                "weight_decay": 0.0003,
                "n_blocks": 3,
                "d_block": 256,
                "dropout": 0.1,
                "k": 32,
            }.items()
        }
        if self.arch_type == "auto":
            parameters[prefix + "arch_type"] = "tabm"
        embedding_type = self.num_emb_type
        if embedding_type == "auto":
            embedding_type = "pwl"
            parameters[prefix + "num_emb_type"] = embedding_type
        if embedding_type in _EMBEDDING_TYPES:
            parameters[prefix + "d_embedding"] = _EMBEDDING_DEFAULTS["d_embedding"]
        if embedding_type == "periodic":
            parameters.update(
                {
                    prefix + "num_emb_n_frequencies": _EMBEDDING_DEFAULTS["num_emb_n_frequencies"],
                    prefix + "num_emb_frequency_init_scale": _EMBEDDING_DEFAULTS["num_emb_frequency_init_scale"],
                    prefix + "num_emb_activation": _EMBEDDING_DEFAULTS["num_emb_activation"],
                }
            )
        elif embedding_type == "pwl":
            parameters[prefix + "num_emb_n_bins"] = _EMBEDDING_DEFAULTS["num_emb_n_bins"]
        loss = self._resolved_loss()
        if loss == "auto":
            loss = self._task_losses()[0]
            parameters[prefix + "loss_function"] = loss
        if loss == "tweedie":
            parameters[prefix + "tweedie_variance_power"] = 1.5
        elif loss == "focal":
            parameters[prefix + "focal_alpha"] = 0.5
            parameters[prefix + "focal_gamma"] = 2.0
        return parameters


class TabMRegressorMother(RegressorMixin, _BaseTabMMother):
    """Single- or multitarget TabM regressor with missing-label support.

    Numeric targets are standardized during fitting and converted back to
    their original units for predictions. Tweedie loss only rescales targets
    and maps member outputs through ``exp`` before averaging. The final
    prediction is the mean across the ensemble members trained by the skorch
    network.
    Use ``target_type="multi_target"`` for a two-dimensional target matrix.
    Missing labels are masked; each task needs an observed finite training
    value. Input features must still be nonmissing.
    """

    def _regression_members(self, X: Features) -> Tensor:
        """Return float64 member predictions in scaled target units."""
        members = self._member_predictions(X).double()
        return members.exp() if self.loss_ == "tweedie" else members

    def predict(self, X: Features) -> npt.NDArray[np.float64]:
        """Predict target values in their original scale.

        Args:
            X: Dense features with the same schema used during fitting.

        Returns:
            Float64 predictions in original target units: a vector for
            single-target mode, or a matrix with one column per task.
        """
        predictions = self._regression_members(X).mean(dim=1).numpy()
        restored = self.target_scaler_.inverse_transform(predictions)
        return restored.ravel() if self.target_type == "single_target" else restored

    def score(self, X: Features, y: Any, sample_weight: Any = None) -> float:
        """Return mean task R-squared, evaluating only observed target labels.

        Each task needs at least two observed evaluation labels. External
        sklearn scorers are not automatically missing-label aware.
        """
        predictions = self.predict(X).reshape(len(X), self.n_outputs_)
        raw_targets = self._target_array(y)
        targets = np.asarray(np.where(pd.isna(raw_targets), np.nan, raw_targets), dtype=np.float64)
        if targets.shape != predictions.shape or np.isinf(targets).any():
            raise ValueError("Evaluation targets must match predictions and contain only finite values or NaN.")
        scores = []
        for task in range(self.n_outputs_):
            observed = np.isfinite(targets[:, task])
            if observed.sum() < 2:
                raise ValueError("Each regression task requires at least two observed labels for scoring.")
            weights = None if sample_weight is None else np.asarray(sample_weight)[observed]
            scores.append(r2_score(targets[observed, task], predictions[observed, task], sample_weight=weights))
        return float(np.mean(scores))

    def predict_uncertainty(self, X: Features, uncertainty_for_opt: bool = False) -> pd.DataFrame:
        """Return member-mean predictions and population standard deviation.

        Knowledge uncertainty is member spread (ddof=0) in original target units,
        not a standard error or calibrated interval. Shared parameters can limit
        diversity. Data and total uncertainty are unavailable and remain unset.

        Args:
            X: Dense features with the same schema used during fitting.
            uncertainty_for_opt: Return only knowledge uncertainty when true.

        Returns:
            A DataFrame containing predictions and uncertainty fields, or just
            knowledge uncertainty when ``uncertainty_for_opt`` is true.
            Multitarget columns use ``target_i_`` prefixes in task order.
        """
        members = self._regression_members(X)
        predictions = self.target_scaler_.inverse_transform(members.mean(dim=1).numpy())
        knowledge = members.std(dim=1, correction=0).numpy() * self.target_scaler_.scale_
        index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(predictions))
        result = pd.DataFrame(index=index)
        for task in range(self.n_outputs_):
            prefix = "" if self.target_type == "single_target" else f"target_{task}_"
            if not uncertainty_for_opt:
                result[prefix + "pred"] = predictions[:, task]
                result[prefix + "mean_predictions"] = predictions[:, task]
            result[prefix + "knowledge_uncertainty"] = knowledge[:, task]
            if not uncertainty_for_opt:
                result[prefix + "data_uncertainty"] = None
                result[prefix + "total_uncertainty"] = None
        return result


class TabMClassifierMother(ClassifierMixin, _BaseTabMMother):
    """Single-target classification or multitarget binary classification.

    Mother encodes labels as integer class indices for training and maps
    predictions back to the original labels. Each ensemble member's logits are
    converted to probabilities before the member probabilities are averaged.
    Multitarget mode uses independent sigmoid outputs and requires two observed
    classes per task; single-target mode uses softmax and also supports
    multiclass labels. Missing labels are masked in either mode.
    """

    def __init__(
        self,
        *,
        target_type: Literal["single_target", "multi_target"] = "single_target",
        model_type: props.ModelType | None = None,
        n_blocks: int | None = None,
        d_block: int = 256,
        dropout: float = 0.1,
        k: int = 32,
        arch_type: Literal["tabm", "tabm-mini", "auto"] = "tabm",
        num_emb_type: Literal["none", "auto", "linear", "linear_relu", "periodic", "pwl"] = "none",
        d_embedding: int = 16,
        num_emb_n_bins: int = 48,
        num_emb_n_frequencies: int = 48,
        num_emb_frequency_init_scale: float = 0.01,
        num_emb_activation: bool = True,
        loss_function: Literal["auto", "cross_entropy", "focal"] | CustomLoss | None = None,
        tweedie_variance_power: float = 1.5,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        auto_class_weights: Literal["default", "Balanced", "SqrtBalanced", "None"] | None = "default",
        lr: float = 0.002,
        weight_decay: float = 0.0003,
        max_epochs: int = 100,
        batch_size: int = 256,
        cat_features: Sequence[str | int] | None = None,
        scale_features: bool = True,
        random_state: int = 42,
        device: str = "cpu",
        verbose: int = 0,
    ) -> None:
        """Configure a TabM classifier.

        Parameters other than ``auto_class_weights`` are described in
        ``_BaseTabMMother.__init__``.

        Args:
            auto_class_weights: Class weighting, matching CatBoost's option.
                ``"Balanced"`` weights each class by the largest class count
                divided by its own count; ``"SqrtBalanced"`` uses the square
                root of that ratio; and ``None`` or ``"None"`` disables
                weighting. Multitarget weights are computed per task.
                ``"default"`` mirrors Mother's CatBoost classifier when the
                option is omitted: ``"Balanced"`` for single-target
                cross-entropy, otherwise no weighting.
        """
        super().__init__(
            target_type=target_type,
            model_type=model_type,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            k=k,
            arch_type=arch_type,
            num_emb_type=num_emb_type,
            d_embedding=d_embedding,
            num_emb_n_bins=num_emb_n_bins,
            num_emb_n_frequencies=num_emb_n_frequencies,
            num_emb_frequency_init_scale=num_emb_frequency_init_scale,
            num_emb_activation=num_emb_activation,
            loss_function=loss_function,
            tweedie_variance_power=tweedie_variance_power,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            batch_size=batch_size,
            cat_features=cat_features,
            scale_features=scale_features,
            random_state=random_state,
            device=device,
            verbose=verbose,
        )
        self.auto_class_weights = auto_class_weights

    def predict_proba(self, X: Features) -> npt.NDArray[np.float32] | list[npt.NDArray[np.float32]]:
        """Predict class probabilities by averaging member probabilities.

        Args:
            X: Dense features with the same schema used during fitting.

        Returns:
            A class-probability matrix for single-target mode. Multitarget mode
            returns a list of two-column matrices, one per task, in each
            task's ``classes_`` order.
        """
        probabilities = self._classification_probabilities(X).mean(dim=1).numpy()
        if self.target_type == "single_target":
            return probabilities[:, 0, :]
        return [probabilities[:, task, :] for task in range(self.n_outputs_)]

    def _classification_probabilities(self, X: Features) -> Tensor:
        """Return member probabilities with explicit task and class axes."""
        members = self._member_predictions(X)
        if self.target_type == "single_target":
            return members.softmax(dim=-1).unsqueeze(2)
        positive = members.sigmoid()
        return torch.stack([1 - positive, positive], dim=-1)

    def predict(self, X: Features) -> npt.NDArray[Any]:
        """Predict the original class label with highest averaged probability.

        Args:
            X: Dense features with the same schema used during fitting.

        Returns:
            Decoded class labels: a vector in single-target mode or a matrix
            with one column per task in multitarget mode.
        """
        probabilities = self._classification_probabilities(X).mean(dim=1).numpy()
        labels = [
            encoder.inverse_transform(probabilities[:, task, :].argmax(axis=1))
            for task, encoder in enumerate(self.label_encoders_)
        ]
        if self.target_type == "single_target":
            return labels[0]
        result = np.empty((len(X), self.n_outputs_), dtype=object)
        for task, values in enumerate(labels):
            result[:, task] = values
        # Preserve ordinary numeric output dtypes when tasks share that dtype.
        if all(values.dtype == labels[0].dtype for values in labels):
            return np.column_stack(labels)
        return result

    def score(self, X: Features, y: Any, sample_weight: Any = None) -> float:
        """Return mean task accuracy using only observed evaluation labels."""
        predictions = self.predict(X).reshape(len(X), self.n_outputs_)
        targets = self._target_array(y)
        if targets.shape != predictions.shape:
            raise ValueError("Evaluation target shape must match predictions.")
        scores = []
        for task in range(self.n_outputs_):
            observed = ~pd.isna(targets[:, task])
            if not observed.any():
                raise ValueError("Each classification task requires observed labels for scoring.")
            actual = np.asarray(targets[observed, task].tolist())
            predicted = np.asarray(predictions[observed, task].tolist())
            weights = None if sample_weight is None else np.asarray(sample_weight)[observed]
            scores.append(accuracy_score(actual, predicted, sample_weight=weights))
        return float(np.mean(scores))

    def predict_uncertainty(self, X: Features, uncertainty_for_opt: bool = False) -> pd.DataFrame:
        """Decompose predictive entropy into expected member entropy and disagreement.

        All entropies use natural logarithms (nats). Data uncertainty is the mean
        member entropy; knowledge uncertainty is the generalized Jensen-Shannon
        divergence between members. Independence is not required, but shared
        parameters can limit disagreement: these are not calibrated noise or
        Bayesian posterior estimates.

        Probability columns follow ``classes_``. ``mean_predictions`` is unset
        for classification, matching Mother's classification fallback.
        Multitarget outputs use ``target_i_`` prefixes with independent
        probability and entropy fields for each binary task.

        Args:
            X: Dense features with the same schema used during fitting.
            uncertainty_for_opt: Return only knowledge uncertainty when true.

        Returns:
            A DataFrame containing decoded predictions, per-class
            probabilities, and uncertainty fields, or just
            ``knowledge_uncertainty`` when ``uncertainty_for_opt`` is true.
        """
        probabilities = self._classification_probabilities(X)
        mean_probabilities = probabilities.mean(dim=1).numpy()
        # Float64 and xlogy preserve small differences and define 0 * log(0) as 0.
        member_probabilities = probabilities.double()
        ensemble_mean = member_probabilities.mean(dim=1)
        total = -torch.special.xlogy(ensemble_mean, ensemble_mean).sum(dim=-1)
        data = -torch.special.xlogy(member_probabilities, member_probabilities).sum(dim=-1).mean(dim=1)
        knowledge = (total - data).clamp_min(0)
        index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(mean_probabilities))
        result = pd.DataFrame(index=index)
        for task, encoder in enumerate(self.label_encoders_):
            prefix = "" if self.target_type == "single_target" else f"target_{task}_"
            if not uncertainty_for_opt:
                result[prefix + "pred"] = encoder.inverse_transform(mean_probabilities[:, task, :].argmax(axis=1))
                for i in range(len(encoder.classes_)):
                    result[prefix + f"proba_{i}"] = mean_probabilities[:, task, i]
                result[prefix + "mean_predictions"] = None
            result[prefix + "knowledge_uncertainty"] = knowledge[:, task].numpy()
            if not uncertainty_for_opt:
                result[prefix + "data_uncertainty"] = data[:, task].numpy()
                result[prefix + "total_uncertainty"] = total[:, task].numpy()
        return result
