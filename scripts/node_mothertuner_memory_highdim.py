#!/usr/bin/env python3
"""Benchmark NODERegressor memory during MotherTuner optimization.

This script builds a high-dimensional synthetic regression dataset (default:
3000 features with most informative), then runs MotherTuner in the standard
MotherML way:
- estimator.get_hyperparameter_space(...) is used by the tuner
- estimator.default_parameters() are enqueued as the first trial

It reports wall time, peak CPU RSS, and peak CUDA memory (if available), and
writes a JSON summary for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mother.ml import PipelineWithHyperparameterRooting
from mother.ml.models.m_node import NODERegressor
from mother.optimization import MotherTuner

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None


@dataclass
class BenchmarkResult:
    status: str
    error: Optional[str]
    n_samples: int
    n_features: int
    n_informative: int
    n_trials_optuna: int
    n_splits_cv: int
    duration_seconds: float
    baseline_rss_gb: Optional[float]
    peak_rss_gb: Optional[float]
    rss_delta_gb: Optional[float]
    peak_cuda_allocated_gb: float
    peak_cuda_reserved_gb: float
    best_trial_number: Optional[int]
    best_trial_value: Optional[float]
    best_params: Optional[dict[str, Any]]


class PeakRSSMonitor:
    """Track process peak RSS via lightweight polling in a background thread."""

    def __init__(self, poll_interval_seconds: float = 0.05) -> None:
        self.poll_interval_seconds = poll_interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline_rss_bytes: Optional[int] = None
        self.peak_rss_bytes: Optional[int] = None

    def start(self) -> None:
        if psutil is None:
            return

        process = psutil.Process()
        current = process.memory_info().rss
        self.baseline_rss_bytes = current
        self.peak_rss_bytes = current

        def _run() -> None:
            assert self.peak_rss_bytes is not None
            while not self._stop.is_set():
                rss_now = process.memory_info().rss
                if rss_now > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss_now
                self._stop.wait(self.poll_interval_seconds)

        self._thread = threading.Thread(target=_run, name="peak-rss-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if psutil is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _bytes_to_gb(num_bytes: Optional[int]) -> Optional[float]:
    if num_bytes is None:
        return None
    return num_bytes / (1024**3)


def _cuda_peaks_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated() / (1024**3)
    reserved = torch.cuda.max_memory_reserved() / (1024**3)
    return allocated, reserved


def make_high_dimensional_dataset(
    *,
    n_samples: int,
    n_features: int,
    informative_ratio: float,
    noise: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a 3000-feature style regression dataset with most features informative."""
    rng = np.random.default_rng(random_state)
    n_informative = max(2, min(n_features, int(round(n_features * informative_ratio))))

    # Build a design matrix and make most columns predictive by assigning
    # non-zero weights to a high fraction of features.
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)).astype(np.float32)

    coef = np.zeros(n_features, dtype=np.float32)
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)
    coef[informative_idx] = rng.normal(loc=0.0, scale=1.0, size=n_informative).astype(np.float32)

    y = X @ coef + rng.normal(loc=0.0, scale=noise, size=n_samples).astype(np.float32)

    # Standardize target for stable training and cleaner optimization dynamics.
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    feature_names = [f"fp_{i:04d}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    return X_df, y_series


def build_pipeline(device: str, max_epochs: int, batch_size: int) -> PipelineWithHyperparameterRooting:
    """Create NODERegressor wrapped in the Mother pipeline rooter."""
    model = NODERegressor(
        # Keep defaults for NODE hyperparameter interfaces used by MotherTuner.
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        head_type="mlp",
        tune_head=True,
        train_split=None,
        # Keep this silent in benchmark output.
        verbose=0,
    )
    return PipelineWithHyperparameterRooting([("regressor", model)])


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    X, y = make_high_dimensional_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        informative_ratio=args.informative_ratio,
        noise=args.noise,
        random_state=args.seed,
    )

    n_informative = max(2, min(args.n_features, int(round(args.n_features * args.informative_ratio))))

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    pipeline = build_pipeline(device=device, max_epochs=args.max_epochs, batch_size=args.batch_size)
    tuner = MotherTuner(
        scorer="neg_root_mean_squared_error",
        n_trials_optuna=args.n_trials,
        n_threads_optuna=args.n_jobs,
        n_startup_trials=max(1, min(args.n_trials, args.n_startup_trials)),
        seed=args.seed,
        tuning_direction="maximize",
    )

    default_params = pipeline.default_parameters()
    cv = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)

    monitor = PeakRSSMonitor(poll_interval_seconds=args.rss_poll_seconds)

    start = time.perf_counter()
    monitor.start()

    status = "ok"
    error = None
    best_trial_number: Optional[int] = None
    best_trial_value: Optional[float] = None
    best_params: Optional[dict[str, Any]] = None

    try:
        _ = tuner.optimize(
            estimator=pipeline,
            X=X,
            y=y,
            cross_validation=cv,
            default_parameters=default_params,
        )
        if tuner.study is not None and tuner.study.best_trial is not None:
            best_trial_number = tuner.study.best_trial.number
            best_trial_value = tuner.study.best_trial.value
            best_params = dict(tuner.study.best_trial.params)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = repr(exc)
    finally:
        monitor.stop()

    duration_seconds = time.perf_counter() - start
    peak_allocated_gb, peak_reserved_gb = _cuda_peaks_gb()

    baseline_gb = _bytes_to_gb(monitor.baseline_rss_bytes)
    peak_gb = _bytes_to_gb(monitor.peak_rss_bytes)
    delta_gb = (peak_gb - baseline_gb) if (peak_gb is not None and baseline_gb is not None) else None

    return BenchmarkResult(
        status=status,
        error=error,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=n_informative,
        n_trials_optuna=args.n_trials,
        n_splits_cv=args.cv_splits,
        duration_seconds=duration_seconds,
        baseline_rss_gb=baseline_gb,
        peak_rss_gb=peak_gb,
        rss_delta_gb=delta_gb,
        peak_cuda_allocated_gb=peak_allocated_gb,
        peak_cuda_reserved_gb=peak_reserved_gb,
        best_trial_number=best_trial_number,
        best_trial_value=best_trial_value,
        best_params=best_params,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NODERegressor memory with MotherTuner.")
    parser.add_argument("--n-samples", type=int, default=1800, help="Number of rows in the synthetic dataset.")
    parser.add_argument("--n-features", type=int, default=3000, help="Number of features (high-dimensional setup).")
    parser.add_argument(
        "--informative-ratio",
        type=float,
        default=0.9,
        help="Fraction of informative features (e.g., 0.9 means 90%% informative).",
    )
    parser.add_argument("--noise", type=float, default=0.2, help="Gaussian noise std used for target generation.")
    parser.add_argument("--n-trials", type=int, default=8, help="Number of Optuna/MotherTuner trials.")
    parser.add_argument("--n-startup-trials", type=int, default=2, help="Optuna startup trials.")
    parser.add_argument("--cv-splits", type=int, default=3, help="KFold split count for tuning.")
    parser.add_argument("--max-epochs", type=int, default=15, help="NODE max epochs per fit.")
    parser.add_argument("--batch-size", type=int, default=512, help="NODE training batch size.")
    parser.add_argument("--n-jobs", type=int, default=1, help="MotherTuner n_threads_optuna.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. 'auto' selects CUDA when available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--rss-poll-seconds",
        type=float,
        default=0.05,
        help="CPU RSS polling interval in seconds (smaller captures peaks better).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("node_mothertuner_memory_highdim_result.json"),
        help="Path to write the benchmark JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)

    config_dict = vars(args).copy()
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)

    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "script": "node_mothertuner_memory_highdim.py",
        "config": config_dict,
        "result": asdict(result),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("NODERegressor + MotherTuner high-dimensional memory benchmark")
    print(f"status={result.status}")
    print(
        f"dataset: n_samples={result.n_samples}, n_features={result.n_features}, "
        f"n_informative={result.n_informative}"
    )
    print(
        f"tuning: trials={result.n_trials_optuna}, cv_splits={result.n_splits_cv}, "
        f"duration={result.duration_seconds:.2f}s"
    )
    print(
        f"cpu_rss_gb: baseline={result.baseline_rss_gb}, peak={result.peak_rss_gb}, "
        f"delta={result.rss_delta_gb}"
    )
    print(
        f"cuda_peak_gb: allocated={result.peak_cuda_allocated_gb:.4f}, "
        f"reserved={result.peak_cuda_reserved_gb:.4f}"
    )
    if result.best_trial_number is not None:
        print(f"best_trial: number={result.best_trial_number}, value={result.best_trial_value}")
    if result.error:
        print(f"error={result.error}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
