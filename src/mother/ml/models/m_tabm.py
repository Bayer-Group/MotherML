"""Mother estimators for TabM, trained with skorch."""

from typing import Any, Literal, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from optuna.trial import Trial
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, column_or_1d, validate_data

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
_EMBEDDING_TYPES = ("linear", "linear_relu", "periodic", "pwl")
_EMBEDDING_DEFAULTS = {
    "d_embedding": 16,
    "num_emb_n_bins": 48,
    "num_emb_n_frequencies": 48,
    "num_emb_frequency_init_scale": 0.01,
    "num_emb_activation": True,
}


class _EnsembleLoss(nn.Module):
    """Apply the task loss to every ensemble member independently.

    The network returns predictions with shape ``(batch, members, outputs)``.
    Each row's target is repeated for every member, then the task loss is
    averaged over all rows and members.
    """

    def __init__(self, classification: bool) -> None:
        """Create the regression or classification loss.

        Args:
            classification: Use cross-entropy when true, otherwise mean
                squared error.
        """
        super().__init__()
        self.classification = classification

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the mean loss across samples and ensemble members.

        Args:
            predictions: Member outputs with batch, member, and output
                dimensions.
            targets: One target per input row.

        Returns:
            Scalar cross-entropy or mean-squared-error loss.
        """
        members = predictions.shape[1]
        repeated = targets.repeat_interleave(members, dim=0)
        if self.classification:
            return nn.functional.cross_entropy(predictions.flatten(0, 1), repeated)
        return nn.functional.mse_loss(predictions.flatten(0, 1), repeated)


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
                ``"auto"`` lets MotherTuner compare the four embedding types;
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

    def _validate_parameters(self) -> None:
        """Raise ``ValueError`` when a constructor parameter is unsupported."""
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
            y: Single-target regression values or classification labels.

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
        targets = column_or_1d(y)
        if len(targets) != len(X):
            raise ValueError("X and y must contain the same number of samples.")
        classification = isinstance(self, ClassifierMixin)
        if classification:
            check_classification_targets(targets)
            self.label_encoder_ = LabelEncoder().fit(targets)
            self.classes_ = self.label_encoder_.classes_
            if len(self.classes_) < 2:
                raise ValueError("TabM classification requires at least two classes.")
            prepared_targets = self.label_encoder_.transform(targets).astype(np.int64)
            output_dim = len(self.classes_)
        else:
            targets = np.asarray(targets, dtype=np.float64)
            if not np.isfinite(targets).all():
                raise ValueError("TabM supports only finite, single-target regression labels.")
            self.target_scaler_ = StandardScaler().fit(targets[:, None])
            prepared_targets = self.target_scaler_.transform(targets[:, None]).astype(np.float32)
            output_dim = 1

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
            y: Training targets required by the interface; not accessed here.
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
            embedding_type = trial.suggest_categorical(prefix + "num_emb_type", list(_EMBEDDING_TYPES))
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
        return parameters

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
        return parameters


class TabMRegressorMother(RegressorMixin, _BaseTabMMother):
    """Single-target TabM regressor with standardized targets.

    Numeric targets are standardized during fitting and converted back to
    their original units for predictions. The final prediction is the mean
    across the ensemble members trained by the skorch network.
    """

    def predict(self, X: Features) -> npt.NDArray[np.float64]:
        """Predict target values in their original scale.

        Args:
            X: Dense features with the same schema used during fitting.

        Returns:
            One float64 prediction per input row, in the original target units.
        """
        predictions = self._member_predictions(X).double().mean(dim=1).numpy()
        return self.target_scaler_.inverse_transform(predictions).ravel()

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
            ``knowledge_uncertainty`` when ``uncertainty_for_opt`` is true.
        """
        members = self._member_predictions(X).double()
        predictions = self.target_scaler_.inverse_transform(members.mean(dim=1).numpy()).ravel()
        knowledge = members.std(dim=1, correction=0).numpy().ravel() * self.target_scaler_.scale_[0]
        index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(predictions))
        if uncertainty_for_opt:
            return pd.DataFrame({"knowledge_uncertainty": knowledge}, index=index)
        return pd.DataFrame(
            {
                "pred": predictions,
                "mean_predictions": predictions,
                "knowledge_uncertainty": knowledge,
                "data_uncertainty": None,
                "total_uncertainty": None,
            },
            index=index,
        )


class TabMClassifierMother(ClassifierMixin, _BaseTabMMother):
    """Binary or multiclass TabM classifier with probability-space averaging.

    Mother encodes labels as integer class indices for training and maps
    predictions back to the original labels. Each ensemble member's logits are
    converted to probabilities before the member probabilities are averaged.
    """

    def predict_proba(self, X: Features) -> npt.NDArray[np.float32]:
        """Predict class probabilities by averaging member probabilities.

        Args:
            X: Dense features with the same schema used during fitting.

        Returns:
            A probability matrix with one column per class in ``classes_``.
        """
        return self._member_predictions(X).softmax(dim=-1).mean(dim=1).numpy()

    def predict(self, X: Features) -> npt.NDArray[Any]:
        """Predict the original class label with highest averaged probability.

        Args:
            X: Dense features with the same schema used during fitting.

        Returns:
            One decoded class label per input row.
        """
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def predict_uncertainty(self, X: Features, uncertainty_for_opt: bool = False) -> pd.DataFrame:
        """Decompose predictive entropy into expected member entropy and disagreement.

        All entropies use natural logarithms (nats). Data uncertainty is the mean
        member entropy; knowledge uncertainty is the generalized Jensen-Shannon
        divergence between members. Independence is not required, but shared
        parameters can limit disagreement: these are not calibrated noise or
        Bayesian posterior estimates.

        Probability columns follow ``classes_``. ``mean_predictions`` is unset
        for classification, matching Mother's classification fallback.

        Args:
            X: Dense features with the same schema used during fitting.
            uncertainty_for_opt: Return only knowledge uncertainty when true.

        Returns:
            A DataFrame containing decoded predictions, per-class
            probabilities, and uncertainty fields, or just
            ``knowledge_uncertainty`` when ``uncertainty_for_opt`` is true.
        """
        probabilities = self._member_predictions(X).softmax(dim=-1)
        mean_probabilities = probabilities.mean(dim=1).numpy()
        # Float64 and xlogy preserve small differences and define 0 * log(0) as 0.
        member_probabilities = probabilities.double()
        ensemble_mean = member_probabilities.mean(dim=1)
        total = -torch.special.xlogy(ensemble_mean, ensemble_mean).sum(dim=-1)
        data = -torch.special.xlogy(member_probabilities, member_probabilities).sum(dim=-1).mean(dim=1)
        knowledge = (total - data).clamp_min(0)
        index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(mean_probabilities))
        if uncertainty_for_opt:
            return pd.DataFrame({"knowledge_uncertainty": knowledge.numpy()}, index=index)
        result = pd.DataFrame({"pred": self.classes_[mean_probabilities.argmax(axis=1)]}, index=index)
        for i in range(len(self.classes_)):
            result[f"proba_{i}"] = mean_probabilities[:, i]
        result["mean_predictions"] = None
        result["knowledge_uncertainty"] = knowledge.numpy()
        result["data_uncertainty"] = data.numpy()
        result["total_uncertainty"] = total.numpy()
        return result
