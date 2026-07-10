import logging
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from mother.errors import ExtrasDependencyImportError

module_logger = logging.getLogger(__name__)


def _check_chemprop() -> None:
    """Validate chemprop availability and raise a Mother-style extras error if missing."""
    try:
        import chemprop  # noqa: F401
    except ImportError as import_error:
        raise ExtrasDependencyImportError("gnn", import_error) from import_error


def _default_chemeleon_embedder(
    checkpoint_path: Optional[str] = None,
    output_dim: int = 2048,
    device: str = "cpu",
) -> Callable[[Sequence[str]], np.ndarray]:
    """Build a default CheMeleon embedder callable from chemprop.

    The public API of chemprop has changed across versions. We keep this logic
    intentionally defensive and raise a clear error when the expected loader is
    not available.
    """
    _check_chemprop()
    import chemprop  # type: ignore

    model = None
    load_fns = [
        "load_model",
        "load_checkpoint",
        "load_from_checkpoint",
    ]
    for fn_name in load_fns:
        loader = getattr(chemprop, fn_name, None)
        if callable(loader):
            kwargs = {}
            loader_varnames = loader.__code__.co_varnames if hasattr(loader, "__code__") else ()
            if checkpoint_path is not None:
                if "checkpoint_path" in loader_varnames:
                    kwargs["checkpoint_path"] = checkpoint_path
                elif "path" in loader_varnames:
                    kwargs["path"] = checkpoint_path
            if "device" in loader_varnames:
                kwargs["device"] = device
            try:
                model = loader(**kwargs) if kwargs else loader()
                break
            except TypeError:
                # Loader exists but has different signature.
                continue

    if model is None:
        raise RuntimeError(
            "chemprop is installed but a supported model loader could not be found. "
            "Please install a compatible chemprop>=2 release or provide a custom embedder."
        )

    if not hasattr(model, "fingerprint"):
        raise RuntimeError(
            "Loaded chemprop model does not expose a 'fingerprint' method. Please provide a custom embedder callable."
        )

    def _embed(smiles_batch: Sequence[str]) -> np.ndarray:
        arr = np.asarray(model.fingerprint(smiles_batch), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("CheMeleon embedder must return a 2D array.")
        if arr.shape[1] != output_dim:
            raise ValueError(f"Expected embedding size {output_dim} but embedder returned {arr.shape[1]} features.")
        return arr

    return _embed


class CheMeleonFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer creating CheMeleon embeddings from SMILES."""

    def __init__(
        self,
        output_dim: int = 2048,
        batch_size: int = 256,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        embedder: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
    ) -> None:
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.embedder = embedder

    def fit(self, X: Iterable[str], y=None) -> "CheMeleonFingerprintTransformer":
        if self.embedder is None:
            self.embedder_ = _default_chemeleon_embedder(
                checkpoint_path=self.checkpoint_path,
                output_dim=self.output_dim,
                device=self.device,
            )
        else:
            self.embedder_ = self.embedder
        self.is_fitted_ = True
        return self

    def _to_smiles(self, value) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, Chem.Mol):
            return Chem.MolToSmiles(value)
        return None

    def transform(self, X: Iterable[str]) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")

        values = np.array(list(X), dtype=object).reshape(-1)
        out = np.full((len(values), self.output_dim), np.nan, dtype=np.float32)
        if len(values) == 0:
            return out

        smiles = np.array([self._to_smiles(value) for value in values], dtype=object)
        valid_mask = np.array(
            [
                isinstance(smiles_str, str) and smiles_str != "" and Chem.MolFromSmiles(smiles_str) is not None
                for smiles_str in smiles
            ],
            dtype=bool,
        )

        n_invalid = int((~valid_mask).sum())
        if n_invalid:
            module_logger.info("Skipping %s invalid SMILES entries during CheMeleon featurization", n_invalid)

        valid_smiles = smiles[valid_mask].tolist()
        if not valid_smiles:
            return out

        rows = []
        for start in range(0, len(valid_smiles), self.batch_size):
            batch = valid_smiles[start : start + self.batch_size]
            batch_embeddings = np.asarray(self.embedder_(batch), dtype=np.float32)
            if batch_embeddings.ndim != 2:
                raise ValueError("CheMeleon embedder must return a 2D array.")
            if batch_embeddings.shape[1] != self.output_dim:
                raise ValueError(f"Expected embedding size {self.output_dim} but received {batch_embeddings.shape[1]}.")
            rows.append(batch_embeddings)

        out[valid_mask, :] = np.vstack(rows)
        return out

    def get_output_dimension(self) -> int:
        return self.output_dim

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return [f"CheMeleonGNNFP_{i}" for i in range(self.output_dim)]


class CheMeleonFingerprintFactory:
    """Factory creating sklearn transformers for CheMeleon GNN fingerprints."""

    def __init__(
        self,
        output_dim: int = 2048,
        batch_size: int = 256,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        embedder: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
    ) -> None:
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.embedder = embedder

    def get_fingerprint_generator(self) -> CheMeleonFingerprintTransformer:
        if self.embedder is None:
            # Keep dependency optional until the factory is actively used.
            _check_chemprop()
        return CheMeleonFingerprintTransformer(
            output_dim=self.output_dim,
            batch_size=self.batch_size,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            embedder=self.embedder,
        )
