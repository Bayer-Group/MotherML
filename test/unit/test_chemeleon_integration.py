"""End-to-end check of the default CheMeleon embedder against a real Chemprop install.

The unit tests in ``test_fpFactory.py`` replace Chemprop with stubs. This test saves a small, randomly
initialised ``BondMessagePassing`` encoder in the same checkpoint format as the published CheMeleon
weights and loads it through the production path, so checkpoint loading, batching and fingerprint
values are all exercised against the installed Chemprop version.
"""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from chemprop import nn as cnn  # noqa: E402
from chemprop.data import BatchMolGraph  # noqa: E402
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer  # noqa: E402

from mother.feature_generation.fp_gnn_gen import (  # noqa: E402
    CheMeleonFingerprintFactory,
)

pytestmark = pytest.mark.slow

_HIDDEN_DIM = 16
_SMILES = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "C1CCNCC1", "OC(=O)CCl"]


def _save_encoder_checkpoint(path):
    """Save a small encoder as {"hyper_parameters", "state_dict"}, the layout of chemeleon_mp.pt."""
    torch.manual_seed(0)
    hyper_parameters = {
        "d_v": 72,
        "d_e": 14,
        "d_h": _HIDDEN_DIM,
        "bias": False,
        "depth": 2,
        "dropout": 0.0,
        "activation": "relu",
        "undirected": False,
        "d_vd": None,
        "V_d_transform": None,
        "graph_transform": None,
    }
    encoder = cnn.BondMessagePassing(**hyper_parameters)
    torch.save({"hyper_parameters": hyper_parameters, "state_dict": encoder.state_dict()}, path)
    return encoder.eval()


def _reference_fingerprints(encoder, smiles):
    """Mean-pool the encoder's atom embeddings per molecule, bypassing MPNN and collate_batch."""
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    bmg = BatchMolGraph([featurizer(Chem.MolFromSmiles(smi)) for smi in smiles])
    with torch.no_grad():
        atom_embeddings = encoder(bmg)
    return np.stack([atom_embeddings[bmg.batch == i].mean(dim=0).numpy() for i in range(len(smiles))])


@pytest.mark.parametrize("return_separate_columns", [True, False])
def test_chemeleon_embedder_loads_local_checkpoint_with_real_chemprop(tmp_path, return_separate_columns):
    checkpoint_path = tmp_path / "encoder.pt"
    encoder = _save_encoder_checkpoint(checkpoint_path)
    expected = _reference_fingerprints(encoder, _SMILES)

    # An invalid row in the middle and batch_size=2 give full batches, a partial last batch and a
    # NaN row that must not shift the rows after it.
    molecules = pd.Series(
        [Chem.MolFromSmiles(smi) for smi in _SMILES[:2]] + [None] + [Chem.MolFromSmiles(smi) for smi in _SMILES[2:]],
        index=pd.Index([10, 11, 12, 13, 14, 15], name="molecule_id"),
    )
    factory = CheMeleonFingerprintFactory(
        output_dim=_HIDDEN_DIM,
        batch_size=2,
        checkpoint_path=str(checkpoint_path),
        return_separate_columns=return_separate_columns,
    )

    result = factory.get_fingerprint_generator().fit_transform(molecules)

    if return_separate_columns:
        fingerprints = result
    else:
        pd.testing.assert_index_equal(result.index, molecules.index)
        fingerprints = np.stack(result["CheMeleonGNNFP"])
    assert fingerprints.shape == (len(molecules), _HIDDEN_DIM)
    assert fingerprints.dtype == np.float32
    assert np.isnan(fingerprints[2]).all()
    valid = np.delete(fingerprints, 2, axis=0)
    assert np.isfinite(valid).all()
    # Distinct molecules must get distinct fingerprints, so a constant output cannot pass.
    assert len(np.unique(valid.round(5), axis=0)) == len(_SMILES)
    np.testing.assert_allclose(valid, expected, rtol=1e-5, atol=1e-6)


def test_chemeleon_embedder_rejects_checkpoint_with_wrong_output_dim(tmp_path):
    checkpoint_path = tmp_path / "encoder.pt"
    _save_encoder_checkpoint(checkpoint_path)
    transformer = CheMeleonFingerprintFactory(
        output_dim=2048, checkpoint_path=str(checkpoint_path)
    ).get_fingerprint_generator()

    with pytest.raises(ValueError, match=f"does not match the CheMeleon checkpoint embedding size {_HIDDEN_DIM}"):
        transformer.fit([Chem.MolFromSmiles("CCO")])
