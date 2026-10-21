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

Features: TypeAlias = pd.DataFrame | npt.NDArray[Any]


class _EnsembleLoss(nn.Module):
    def __init__(self, classification: bool) -> None:
        super().__init__()
        self.classification = classification

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        members = predictions.shape[1]
        repeated = targets.repeat_interleave(members, dim=0)
        if self.classification:
            return nn.functional.cross_entropy(predictions.flatten(0, 1), repeated)
        return nn.functional.mse_loss(predictions.flatten(0, 1), repeated)


class _TabMNet(NeuralNet):
    def initialize_optimizer(self, triggered_directly: Any = None) -> "_TabMNet":
        self.optimizer_ = self.optimizer(
            make_parameter_groups(self.module_),
            lr=self.lr,
            **self.get_params_for("optimizer"),
        )
        return self


class _BaseTabMMother(BaseEstimator, AbstractMotherPipeline):
    net_: _TabMNet

    def __init__(
        self,
        *,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        k: int = 32,
        arch_type: Literal["tabm", "tabm-mini"] = "tabm",
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
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.k = k
        self.arch_type = arch_type
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
        return super().get_params(deep=deep)

    def set_params(self, **params: Any) -> "_BaseTabMMother":
        return super().set_params(**params)

    def _validate_parameters(self) -> None:
        for name in ("n_blocks", "d_block", "k", "max_epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.arch_type not in ("tabm", "tabm-mini"):
            raise ValueError("arch_type must be 'tabm' or 'tabm-mini'.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite.")

    def _prepare_features(self, X: Features, *, fitting: bool) -> dict[str, npt.NDArray[Any]]:
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
            numeric = np.asarray(values[:, self.numeric_indices_], dtype=np.float32)
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
        self._validate_parameters()
        # A failed refit must not expose a stale trained network with new encoders.
        if hasattr(self, "net_"):
            del self.net_
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
            targets = np.asarray(targets, dtype=np.float32)
            if not np.isfinite(targets).all():
                raise ValueError("TabM supports only finite, single-target regression labels.")
            self.target_scaler_ = StandardScaler().fit(targets[:, None])
            prepared_targets = self.target_scaler_.transform(targets[:, None]).astype(np.float32)
            output_dim = 1

        inputs = self._prepare_features(X, fitting=True)
        net = _TabMNet(
            module=Model,
            module__n_num_features=len(self.numeric_indices_),
            module__cat_cardinalities=self.cat_cardinalities_,
            module__n_classes=output_dim,
            module__backbone={
                "type": "MLP",
                "n_blocks": self.n_blocks,
                "d_block": self.d_block,
                "dropout": self.dropout,
            },
            module__bins=None,
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
            net.fit(inputs, prepared_targets)
        self.net_ = net
        return self

    def _member_predictions(self, X: Features) -> Tensor:
        check_is_fitted(self, "net_")
        inputs = self._prepare_features(X, fitting=False)
        return torch.cat(list(self.net_.forward_iter(inputs, training=False, device="cpu")), dim=0)

    def get_hyperparameter_space(self, X: Features, y: Any, trial: Trial, prefix: str = "") -> dict[str, Any]:
        return {
            prefix + "lr": trial.suggest_float(prefix + "lr", 1e-4, 1e-2, log=True),
            prefix + "weight_decay": trial.suggest_float(prefix + "weight_decay", 1e-6, 1e-2, log=True),
            prefix + "n_blocks": trial.suggest_int(prefix + "n_blocks", 1, 4),
            prefix + "d_block": trial.suggest_categorical(prefix + "d_block", [64, 128, 256, 512]),
            prefix + "dropout": trial.suggest_float(prefix + "dropout", 0.0, 0.5),
            prefix + "k": trial.suggest_categorical(prefix + "k", [8, 16, 32]),
        }

    def default_parameters(self, prefix: str = "") -> dict[str, Any]:
        return {
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


class TabMRegressorMother(RegressorMixin, _BaseTabMMother):
    """Single-target TabM regression with skorch and standardized targets."""

    def predict(self, X: Features) -> npt.NDArray[np.float32]:
        predictions = self._member_predictions(X).mean(dim=1).numpy()
        return self.target_scaler_.inverse_transform(predictions).ravel().astype(np.float32)


class TabMClassifierMother(ClassifierMixin, _BaseTabMMother):
    """Binary or multiclass TabM classification with probability-space averaging."""

    def predict_proba(self, X: Features) -> npt.NDArray[np.float32]:
        return self._member_predictions(X).softmax(dim=-1).mean(dim=1).numpy()

    def predict(self, X: Features) -> npt.NDArray[Any]:
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
