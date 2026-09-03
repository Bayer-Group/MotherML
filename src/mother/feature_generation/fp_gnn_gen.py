import hashlib
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from mother.errors import ExtrasDependencyImportError

module_logger = logging.getLogger(__name__)

_CHEMELEON_ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
_CHEMELEON_CACHE_PATH = Path.home() / ".cache" / "mother" / "chemeleon_mp.pt"
_CHEMELEON_SHA256 = "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"


def _sha256(path: Path) -> str:
    """Compute the SHA256 hex digest of a file, reading it in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_default_chemeleon_checkpoint() -> Path:
    """Return path to chemeleon_mp.pt, downloading from Zenodo on first use."""
    if _CHEMELEON_CACHE_PATH.exists() and _sha256(_CHEMELEON_CACHE_PATH) == _CHEMELEON_SHA256:
        return _CHEMELEON_CACHE_PATH

    if _CHEMELEON_CACHE_PATH.exists():
        module_logger.warning("Cached CheMeleon checkpoint failed SHA256 verification; replacing it.")

    if not _CHEMELEON_CACHE_PATH.exists() or _sha256(_CHEMELEON_CACHE_PATH) != _CHEMELEON_SHA256:
        _CHEMELEON_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Optional[Path] = None
        try:
            module_logger.info("Downloading CheMeleon checkpoint from Zenodo to %s", _CHEMELEON_CACHE_PATH)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".tmp",
                prefix="chemeleon_mp.",
                dir=_CHEMELEON_CACHE_PATH.parent,
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                with urllib.request.urlopen(_CHEMELEON_ZENODO_URL, timeout=60) as response:
                    shutil.copyfileobj(response, tmp_file)

            if _sha256(tmp_path) != _CHEMELEON_SHA256:
                raise RuntimeError("Downloaded CheMeleon checkpoint failed SHA256 verification.")
            tmp_path.replace(_CHEMELEON_CACHE_PATH)  # atomic on POSIX; avoids partial files
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
    return _CHEMELEON_CACHE_PATH


def _check_chemprop() -> None:
    """Validate chemprop availability and raise a Mother-style extras error if missing."""
    try:
        import chemprop  # noqa: F401
    except ImportError as import_error:
        raise ExtrasDependencyImportError("chemprop", import_error) from import_error


def _default_chemeleon_embedder(
    checkpoint_path: Optional[str] = None,
    output_dim: int = 2048,
    device: str = "cpu",
) -> Callable[[Sequence[str]], np.ndarray]:
    """Build a default CheMeleon embedder callable from current chemprop."""
    _check_chemprop()
    import torch
    from chemprop import models as cp_models  # type: ignore
    from chemprop import nn as cnn  # type: ignore
    from chemprop.data import (  # type: ignore
        MoleculeDatapoint,
        MoleculeDataset,
        collate_batch,
    )

    if checkpoint_path is None:
        checkpoint_path = str(get_default_chemeleon_checkpoint())

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "hyper_parameters" not in ckpt or "state_dict" not in ckpt:
        raise RuntimeError(
            "Unsupported CheMeleon checkpoint format for current chemprop loader. "
            "Expected a dict with 'hyper_parameters' and 'state_dict'."
        )

    mp = cnn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    agg = cnn.MeanAggregation()
    ffn = cnn.RegressionFFN(input_dim=mp.output_dim)
    model = cp_models.MPNN(mp, agg, ffn, batch_norm=False).to(device).eval()

    def _embed(smiles_batch: Sequence[str]) -> np.ndarray:
        """Run a batch of SMILES through the loaded CheMeleon MPNN and return their fingerprints."""
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi) for smi in smiles_batch])
        batch = collate_batch([dataset[i] for i in range(len(dataset))])
        bmg, V_d, X_d, *_ = batch

        # BatchMolGraph.to mutates in-place and returns None.
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
        """Validate and store CheMeleon embedding configuration for later use in `fit`."""
        if output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.embedder = embedder

    def fit(self, X: Iterable, y: object | None = None) -> "CheMeleonFingerprintTransformer":
        """Load the CheMeleon embedder (or use the injected one) and mark the transformer as fitted."""
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
        """Convert RDKit Mol objects to CheMeleon fingerprints, NaN-filling any invalid molecules."""
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
        """Return the number of embedding dimensions produced by this transformer."""
        return self.output_dim

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> List[str]:
        """Return sklearn-style output feature names, one per embedding dimension."""
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
        """Validate and store the configuration used to build fingerprint transformers."""
        if output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.embedder = embedder

    def get_fingerprint_generator(self) -> CheMeleonFingerprintTransformer:
        """Build a `CheMeleonFingerprintTransformer` using this factory's stored configuration."""
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
