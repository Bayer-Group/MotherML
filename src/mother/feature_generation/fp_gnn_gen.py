import logging
from pathlib import Path
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
    """Build a default CheMeleon embedder callable from current chemprop."""
    _check_chemprop()
    import torch
    from chemprop.data import (  # type: ignore
        MoleculeDatapoint,
        MoleculeDataset,
        collate_batch,
    )
    from chemprop.models import load_model  # type: ignore

    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for CheMeleon embeddings with the current chemprop API.")

    model = load_model(path=Path(checkpoint_path))

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    if not hasattr(model, "fingerprint"):
        raise RuntimeError(
            "Loaded chemprop model does not expose a 'fingerprint' method. Please provide a custom embedder callable."
        )

    def _embed(smiles_batch: Sequence[str]) -> np.ndarray:
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi) for smi in smiles_batch])
        batch = collate_batch([dataset[i] for i in range(len(dataset))])
        bmg, V_d, X_d, *_ = batch

        bmg.to(device)
        if V_d is not None:
            V_d = V_d.to(device)
        if X_d is not None:
            X_d = X_d.to(device)

        with torch.no_grad():
            fps = model.fingerprint(bmg, V_d, X_d)
        arr = np.asarray(fps.detach().cpu().numpy(), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("CheMeleon embedder must return a 2D array.")
        if arr.shape[1] != output_dim:
            raise ValueError(f"Expected embedding size {output_dim} but embedder returned {arr.shape[1]} features.")
        return arr

    return _embed


class CheMeleonFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer creating CheMeleon embeddings from RDKit Mol objects."""

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

    def fit(self, X: Iterable, y=None) -> "CheMeleonFingerprintTransformer":
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

    def transform(self, X: Iterable) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")

        values = np.array(list(X), dtype=object).reshape(-1)
        out = np.full((len(values), self.output_dim), np.nan, dtype=np.float32)
        if len(values) == 0:
            return out

        valid_mask = np.array([isinstance(compound, Chem.Mol) for compound in values], dtype=bool)

        n_invalid = int((~valid_mask).sum())
        if n_invalid:
            module_logger.info("Skipping %s invalid molecule entries during CheMeleon featurization", n_invalid)

        valid_mols = values[valid_mask].tolist()
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
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
