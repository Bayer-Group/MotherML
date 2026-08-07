#!/usr/bin/env python3
"""Benchmark MLP, NODE (subset/mlp/flow heads) and Ridge on FreeSolv with CheMeleon embeddings.

Usage:
    python scripts/chemeleon_freesolv_benchmark.py --ckpt /tmp/chemeleon_mp.pt

The CheMeleon checkpoint path defaults to /tmp/chemeleon_mp.pt (same location used
by the earlier MLP-only benchmark).  Pass --ckpt to override.

Import order note: mother/sklearn must be imported before chemprop/lightning to avoid
a native library conflict with torchmetrics/lightning torch bindings.
"""

from __future__ import annotations

import argparse
import os
import sys
import types

os.environ.setdefault("MPLBACKEND", "Agg")

# Stub the interactive plotting module before chemprop triggers its aimsim import.
sys.modules.setdefault("aimsim.utils.plotting_scripts", types.ModuleType("aimsim.utils.plotting_scripts"))

import warnings

import numpy as np
import torch

torch.set_num_threads(1)
warnings.filterwarnings("ignore")

# ── mother / sklearn imports first (avoids torchmetrics/lightning conflict) ──
import pandas as pd

# ── chemprop (after mother) ───────────────────────────────────────────────────
from chemprop import models as cp_models
from chemprop import nn as cnn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mother.cv.cv_methods import tanimoto_sphere_exclusion_clustering

from mother.feature_generation.fp_gnn_gen import get_default_chemeleon_checkpoint
from mother.ml.models.m_catboost import CatboostRegressorMother
from mother.ml.models.m_flow import FlowHeadRegressor
from mother.ml.models.m_mlp import MLPHeadRegressor
from mother.ml.models.m_node import NODERegressor
from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

LOG_PATH = "/tmp/chemeleon_freesolv_benchmark.txt"
_log_fh = open(LOG_PATH, "w")


def out(msg: str) -> None:
    print(msg, flush=True)
    _log_fh.write(msg + "\n")
    _log_fh.flush()


def _load_data() -> tuple[list[str], np.ndarray]:
    """Return (smiles, y) from FreeSolv.  Tries DeepChem S3 first, local CSV fallback."""
    REMOTE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
    LOCAL = "examples/notebooks/freesolv_train.csv"
    try:
        df = pd.read_csv(REMOTE)
        src = "DeepChem SAMPL (full FreeSolv)"
    except Exception:
        df = pd.read_csv(LOCAL)
        src = "local freesolv_train.csv"
    df = df.dropna(subset=["smiles", "expt"]).reset_index(drop=True)
    out(f"Data: {src}  |  n={len(df)}  |  target=expt (hydration free energy kcal/mol)")
    return df["smiles"].tolist(), df["expt"].to_numpy(dtype=np.float32)


def _build_embedder(ckpt_path: str):
    """Load CheMeleon MPNN from checkpoint and return an embed(smiles) callable."""
    if not os.path.exists(ckpt_path):
        out(f"Checkpoint not found at {ckpt_path} — using cached/auto-downloaded copy ...")
        ckpt_path = str(get_default_chemeleon_checkpoint())
    out(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=True)
    mp = cnn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    agg = cnn.MeanAggregation()
    ffn = cnn.RegressionFFN(input_dim=mp.output_dim)
    mpnn = cp_models.MPNN(mp, agg, ffn, batch_norm=False).eval()

    def embed(smis: list[str], bs: int = 64) -> np.ndarray:
        parts: list[np.ndarray] = []
        for i in range(0, len(smis), bs):
            chunk = smis[i : i + bs]
            ds = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in chunk])
            bmg, V_d, X_d, *_ = collate_batch([ds[j] for j in range(len(ds))])
            with torch.no_grad():
                fp = mpnn.fingerprint(bmg, V_d, X_d)
            parts.append(fp.cpu().numpy().astype(np.float32))
        return np.concatenate(parts, 0)

    return embed


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

HEADER = f"{'Model':<45}  {'RMSE':>6}  {'MAE':>6}  {'R²':>7}"
SEP = "-" * len(HEADER)


def _report(name: str, model, Xtr, Xte, ytr_s, yte, ym, ysd) -> dict:
    """Fit model on standardized targets, evaluate on original scale."""
    try:
        model.fit(Xtr, ytr_s)
        raw = model.predict(Xte)
        if hasattr(raw, "flatten"):
            raw = raw.flatten()
        pred = raw * ysd + ym
        rmse = root_mean_squared_error(yte, pred)
        mae = mean_absolute_error(yte, pred)
        r2 = r2_score(yte, pred)
        out(f"{name:<45}  {rmse:>6.3f}  {mae:>6.3f}  {r2:>7.3f}")
        return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "status": "ok"}
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)[:80]
        out(f"{name:<45}  FAILED: {msg}")
        return {"model": name, "rmse": None, "mae": None, "r2": None, "status": f"failed: {msg}"}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="/tmp/chemeleon_mp.pt", help="Path to CheMeleon MPNN checkpoint (.pt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100, help="Default max epochs (overridden per family)")
    parser.add_argument("--mlp-epochs", type=int, default=None, help="Max epochs for MLP models (default: --epochs)")
    parser.add_argument("--node-epochs", type=int, default=None, help="Max epochs for NODE models (default: --epochs // 2)")
    parser.add_argument("--flow-epochs", type=int, default=None, help="Max epochs for standalone flow models (default: --epochs)")
    parser.add_argument("--node-flow-epochs", type=int, default=None, help="Max epochs for NODE+flow head (default: --node-epochs)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # ── Data ─────────────────────────────────────────────────────────────────
    smiles, y = _load_data()

    # ── CheMeleon embeddings ──────────────────────────────────────────────────
    out(f"\nEmbedding with CheMeleon MPNN from {args.ckpt} ...")
    embed = _build_embedder(args.ckpt)
    X = embed(smiles)
    out(f"Fingerprint matrix: {X.shape}")

    # ── Cluster-based train/test split (MACCS + Tanimoto sphere exclusion) ──
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    maccs_fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    clusters = tanimoto_sphere_exclusion_clustering(maccs_fps, similarity_threshold=0.5)

    # Sort clusters by size descending; hold out the largest ~20% of molecules
    cluster_ids_by_size = sorted(clusters, key=lambda c: len(clusters[c]), reverse=True)
    test_idx, train_idx = set(), set()
    target_test = int(0.2 * len(smiles))
    for cid in cluster_ids_by_size:
        if len(test_idx) < target_test:
            test_idx.update(clusters[cid])
        else:
            train_idx.update(clusters[cid])
    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)

    out(f"Clusters: {len(clusters)}  |  Split: train={len(train_idx)}  test={len(test_idx)}  "
        f"(MACCS Tanimoto threshold=0.5)\n")

    Xtr = X[train_idx].astype(np.float32)
    Xte = X[test_idx].astype(np.float32)
    ytr = y[train_idx]
    yte = y[test_idx]

    xs = StandardScaler().fit(Xtr)
    Xtr = xs.transform(Xtr).astype(np.float32)
    Xte = xs.transform(Xte).astype(np.float32)

    # Standardize target (required for numerical stability of flow heads)
    ym, ysd = ytr.mean(), ytr.std()
    ytr_s = ((ytr - ym) / ysd).astype(np.float32)

    mlp_epochs = args.mlp_epochs or args.epochs
    node_epochs = args.node_epochs or args.epochs
    flow_epochs = args.flow_epochs or args.epochs
    node_flow_epochs = args.node_flow_epochs or node_epochs

    # ── Benchmark ─────────────────────────────────────────────────────────────
    out(f"=== FreeSolv benchmark  |  CheMeleon {X.shape[1]}-dim embeddings ===\n")
    out(f"Epochs  MLP={mlp_epochs}  NODE={node_epochs}  Flow={flow_epochs}  NODE+flow={node_flow_epochs}\n")
    out(HEADER)
    out(SEP)

    results = []

    # Ridge (linear probe) — strong baseline for foundation embeddings on small data
    results.append(_report(
        "Lasso (alpha=0.01)",
        Lasso(alpha=0.01, max_iter=10000),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))

    results.append(_report(
        "CatBoost (default)",
        CatboostRegressorMother(verbose=0),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))

    # pytabkit auto-detects GPU; early stopping disabled — 2048-dim embeddings need more epochs to converge
    results.append(_report(
        "RealMLP-TD",
        RealMLP_TD_Regressor(n_epochs=mlp_epochs * 5, n_refit=1, val_fraction=0.1, use_early_stopping=False),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))
    results.append(_report(
        "TabM-D",
        TabM_D_Regressor(n_epochs=mlp_epochs * 5, val_fraction=0.1),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))

    # MLP replications from previous run
    results.append(_report(
        "MLP default [512,256,128] GELU/BN",
        MLPHeadRegressor(max_epochs=mlp_epochs, lr=2.5e-3, device=args.device, verbose=0),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))
    results.append(_report(
        "MLP CheMeleon-style [512] LayerNorm",
        MLPHeadRegressor(
            hidden_dims=[512], norm="layer", activation="GELU", dropout=0.1,
            max_epochs=mlp_epochs, lr=2.5e-3, device=args.device, verbose=0,
        ),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))

    _node_base = dict(num_trees=256, depth=4, num_layers=2, input_dropout=0.05,
                      tree_dropout=0.0, lr=5e-3, batch_size=64, device=args.device)

    # NODE — subset head
    results.append(_report(
        "NODE subset head",
        NODERegressor(head_type="subset", max_epochs=node_epochs, **_node_base),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))

    # NODE — linear head
    results.append(_report(
        "NODE linear head",
        NODERegressor(head_type="linear", max_epochs=node_epochs, **_node_base),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))

    # NODE — MLP heads: several width/depth combos
    for dims, act in [([256], "ReLU"), ([256, 128], "GELU"), ([512, 256], "GELU"), ([512, 256, 128], "GELU")]:
        label = f"NODE MLP head {dims} {act}"
        results.append(_report(
            label,
            NODERegressor(head_type="mlp", mlp_hidden_dims=dims, mlp_dropout=0.1,
                          mlp_activation=act, max_epochs=node_epochs, **_node_base),
            Xtr, Xte, ytr_s, yte, ym, ysd,
        ))

    # Standalone flow head (no NODE backbone) — MLP encoder + normalizing flow
    results.append(_report(
        "Flow [512] LayerNorm + NICE",
        FlowHeadRegressor(
            flow_type="NICE", flow_transforms=3,
            mlp_hidden_dims=[512], mlp_norm="layer", mlp_dropout=0.1, mlp_activation="GELU",
            max_epochs=flow_epochs, lr=2.5e-3, device=args.device,
        ),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))
    results.append(_report(
        "Flow [512] LayerNorm + NSF",
        FlowHeadRegressor(
            flow_type="NSF", flow_bins=8,
            mlp_hidden_dims=[512], mlp_norm="layer", mlp_dropout=0.1, mlp_activation="GELU",
            max_epochs=flow_epochs, lr=2.5e-3, device=args.device,
        ),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))

    # NODE — flow heads: lower lr to stabilise the joint NODE+flow optimisation
    _node_flow_base = {**_node_base, "lr": 1e-3}
    results.append(_report(
        "NODE flow head NSF (point pred)",
        NODERegressor(head_type="flow", flow_type="NSF", flow_bins=8,
                      max_epochs=node_flow_epochs, **_node_flow_base),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))
    results.append(_report(
        "NODE flow head NICE (point pred)",
        NODERegressor(head_type="flow", flow_type="NICE", flow_transforms=3,
                      max_epochs=node_flow_epochs, **_node_flow_base),
        Xtr, Xte, ytr_s, yte, ym, ysd,
    ))

    out(SEP)
    out(f"\nLog written to {LOG_PATH}")

    # Summary table sorted by RMSE
    valid = [r for r in results if r["rmse"] is not None]
    if valid:
        out("\n=== Ranked by RMSE (lower is better) ===")
        for r in sorted(valid, key=lambda x: x["rmse"]):
            out(f"  {r['model']:<45}  RMSE={r['rmse']:.3f}  R²={r['r2']:.3f}")

    _log_fh.close()


if __name__ == "__main__":
    main()
