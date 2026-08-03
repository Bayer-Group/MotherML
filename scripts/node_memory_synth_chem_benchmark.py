#!/usr/bin/env python3
"""Benchmark NODE GPU memory on synthetic chemistry-like datasets.

This script creates high-dimensional sparse fingerprint-like datasets and runs
NODEClassifier/NODERegressor with a set of practical and stress configs,
recording wall time and peak CUDA memory statistics.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mother.ml.models.m_node import NODEClassifier, NODERegressor


@dataclass
class RunResult:
    dataset: str
    task: str
    config_name: str
    status: str
    fit_seconds: float
    peak_allocated_gb: float
    peak_reserved_gb: float
    error: str | None


def _gpu_stats() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_res = torch.cuda.max_memory_reserved() / (1024**3)
    return peak_alloc, peak_res


def _make_sparse_fingerprint_matrix(
    n_samples: int,
    n_features: int,
    on_bits: int,
    rng: np.random.Generator,
    as_counts: bool,
) -> np.ndarray:
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_samples):
        idx = rng.choice(n_features, size=on_bits, replace=False)
        if as_counts:
            X[i, idx] = rng.integers(1, 4, size=on_bits, endpoint=False).astype(np.float32)
        else:
            X[i, idx] = 1.0
    return X


def _run_classifier(
    X: np.ndarray,
    y: np.ndarray,
    config_name: str,
    cfg: dict[str, Any],
    device: str,
    dataset_name: str,
) -> RunResult:
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = NODEClassifier(
        max_epochs=cfg.get("max_epochs", 12),
        batch_size=cfg["batch_size"],
        num_layers=cfg["num_layers"],
        num_trees=cfg["num_trees"],
        depth=cfg["depth"],
        additional_tree_output_dim=cfg.get("additional_tree_output_dim", 3),
        max_layers_retained=cfg.get("max_layers_retained", 1),
        lr=cfg.get("lr", 0.005),
        input_dropout=cfg.get("input_dropout", 0.05),
        tree_dropout=cfg.get("tree_dropout", 0.05),
        head_type="subset",
        device=device,
        verbose=0,
    )

    start = time.perf_counter()
    try:
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        peak_alloc, peak_res = _gpu_stats()
        return RunResult(dataset_name, "classification", config_name, "ok", fit_seconds, peak_alloc, peak_res, None)
    except Exception as exc:  # noqa: BLE001
        fit_seconds = time.perf_counter() - start
        peak_alloc, peak_res = _gpu_stats()
        return RunResult(
            dataset_name,
            "classification",
            config_name,
            "error",
            fit_seconds,
            peak_alloc,
            peak_res,
            repr(exc),
        )


def _run_regressor(
    X: np.ndarray,
    y: np.ndarray,
    config_name: str,
    cfg: dict[str, Any],
    device: str,
    dataset_name: str,
) -> RunResult:
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train).astype(np.float32)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = NODERegressor(
        max_epochs=cfg.get("max_epochs", 12),
        batch_size=cfg["batch_size"],
        num_layers=cfg["num_layers"],
        num_trees=cfg["num_trees"],
        depth=cfg["depth"],
        additional_tree_output_dim=cfg.get("additional_tree_output_dim", 3),
        max_layers_retained=cfg.get("max_layers_retained", 1),
        lr=cfg.get("lr", 0.005),
        input_dropout=cfg.get("input_dropout", 0.05),
        tree_dropout=cfg.get("tree_dropout", 0.05),
        head_type="mlp",
        mlp_hidden_dims=[128, 64, 32],
        device=device,
        verbose=0,
    )

    start = time.perf_counter()
    try:
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        peak_alloc, peak_res = _gpu_stats()
        return RunResult(dataset_name, "regression", config_name, "ok", fit_seconds, peak_alloc, peak_res, None)
    except Exception as exc:  # noqa: BLE001
        fit_seconds = time.perf_counter() - start
        peak_alloc, peak_res = _gpu_stats()
        return RunResult(
            dataset_name,
            "regression",
            config_name,
            "error",
            fit_seconds,
            peak_alloc,
            peak_res,
            repr(exc),
        )


def main() -> None:
    rng = np.random.default_rng(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets: list[dict[str, Any]] = [
        {
            "name": "chem_synth_cls_binary_3048d",
            "task": "classification",
            "X": _make_sparse_fingerprint_matrix(12000, 3048, 64, rng, as_counts=False),
            "y": rng.integers(0, 2, size=12000, endpoint=False),
        },
        {
            "name": "chem_synth_cls_counts_3048d",
            "task": "classification",
            "X": _make_sparse_fingerprint_matrix(12000, 3048, 80, rng, as_counts=True),
            "y": rng.integers(0, 3, size=12000, endpoint=False),
        },
        {
            "name": "chem_synth_reg_3048d",
            "task": "regression",
            "X": _make_sparse_fingerprint_matrix(100000, 3048, 72, rng, as_counts=True),
            "y": rng.normal(0.0, 1.0, size=100000).astype(np.float32),
        },
    ]

    configs: dict[str, dict[str, Any]] = {
        "default_practical": {
            "num_layers": 2,
            "num_trees": 512,
            "depth": 4,
            "batch_size": 1001,
            "max_layers_retained": 1,
            "max_epochs": 12,
            "lr": 0.005,
        },
        "low_memory": {
            "num_layers": 2,
            "num_trees": 256,
            "depth": 4,
            "batch_size": 1001,
            "max_layers_retained": 1,
            "max_epochs": 12,
            "lr": 0.005,
        },
        "higher_capacity": {
            "num_layers": 3,
            "num_trees": 2048,
            "depth": 6,
            "batch_size": 256,
            "max_layers_retained": 1,
            "max_epochs": 12,
            "lr": 0.003,
        },
    }

    results: list[RunResult] = []

    for ds in datasets:
        for cfg_name, cfg in configs.items():
            if ds["task"] == "classification":
                out = _run_classifier(ds["X"], ds["y"], cfg_name, cfg, device, ds["name"])
            else:
                out = _run_regressor(ds["X"], ds["y"], cfg_name, cfg, device, ds["name"])
            results.append(out)
            print(
                f"[{out.status}] {out.dataset} | {out.task} | {out.config_name} "
                f"| fit={out.fit_seconds:.1f}s | peak_alloc={out.peak_allocated_gb:.2f}GB "
                f"| peak_reserved={out.peak_reserved_gb:.2f}GB"
            )
            if out.error:
                print(f"  error: {out.error}")

    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    output_dir = Path(".")
    df.to_csv(output_dir / "node_memory_synth_chem_results.csv", index=False)
    with (output_dir / "node_memory_synth_chem_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\nSaved:")
    print("- node_memory_synth_chem_results.csv")
    print("- node_memory_synth_chem_results.json")


if __name__ == "__main__":
    main()
