#!/usr/bin/env python3
"""Run NODERegressor with official MotherCV + MotherTuner defaults on high-dimensional data.

This benchmark uses the actual mother.pipeline_utils.mother_cv() entrypoint with a
MotherTuner instance, records memory/runtime, and saves tuned hyperparameters.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mother.ml import PipelineWithHyperparameterRooting
from mother.ml.models.m_node import NODERegressor
from mother.optimization import MotherTuner
from mother.pipeline_utils import mother_cv

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class RunResult:
    status: str
    error: Optional[str]
    traceback: Optional[str]
    duration_seconds: float
    baseline_rss_gb: Optional[float]
    peak_rss_gb: Optional[float]
    rss_delta_gb: Optional[float]
    peak_cuda_allocated_gb: float
    peak_cuda_reserved_gb: float
    rmse_mean: Optional[float]
    rmse_std: Optional[float]
    best_trial_number: Optional[int]
    best_trial_value: Optional[float]
    best_params: Optional[dict[str, Any]]


@dataclass
class StandaloneComplexityResult:
    batch_size: int
    status: str
    error: Optional[str]
    traceback: Optional[str]
    duration_seconds: float
    baseline_rss_gb: Optional[float]
    peak_rss_gb: Optional[float]
    rss_delta_gb: Optional[float]
    peak_cuda_allocated_gb: float
    peak_cuda_reserved_gb: float
    train_rmse: Optional[float]


class PeakRSSMonitor:
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

        self._thread = threading.Thread(target=_run, daemon=True)
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


def make_dataset(
    *,
    n_samples: int,
    n_features: int,
    informative_ratio: float,
    noise: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)
    n_informative = max(2, min(n_features, int(round(n_features * informative_ratio))))

    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)).astype(np.float32)
    coef = np.zeros(n_features, dtype=np.float32)
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)
    coef[informative_idx] = rng.normal(loc=0.0, scale=1.0, size=n_informative).astype(np.float32)

    y = X @ coef + rng.normal(loc=0.0, scale=noise, size=n_samples).astype(np.float32)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    feature_names = [f"fp_{i:04d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")


def build_pipeline(device: str, max_epochs: int, batch_size: int) -> PipelineWithHyperparameterRooting:
    # Keep NODE architecture defaults to follow official default behavior.
    model = NODERegressor(
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_split=None,
        verbose=0,
    )
    return PipelineWithHyperparameterRooting([("regressor", model)])


def _parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a comma-separated list of integers.")
    if any(v <= 0 for v in values):
        raise ValueError("All values in comma-separated integer lists must be positive.")
    return values


def run_standalone_complexity_sweep(args: argparse.Namespace, *, device: str) -> list[StandaloneComplexityResult]:
    X, y = make_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        informative_ratio=args.informative_ratio,
        noise=args.noise,
        random_state=args.seed,
    )

    results: list[StandaloneComplexityResult] = []
    for batch_size in args.standalone_batch_sizes:
        prefix = f"[standalone][batch_size={batch_size}]"
        print(
            f"{prefix} start | epochs={args.standalone_epochs} "
            f"layers={args.standalone_num_layers} trees={args.standalone_num_trees}"
        )

        model = NODERegressor(
            device=device,
            max_epochs=args.standalone_epochs,
            batch_size=batch_size,
            num_layers=args.standalone_num_layers,
            num_trees=args.standalone_num_trees,
            depth=args.standalone_depth,
            additional_tree_output_dim=args.standalone_additional_tree_output_dim,
            train_split=None,
            verbose=0,
        )

        monitor = PeakRSSMonitor(poll_interval_seconds=args.rss_poll_seconds)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        gc.collect()
        monitor.start()
        start = time.perf_counter()

        status = "ok"
        error = None
        traceback_text = None
        train_rmse: Optional[float] = None

        try:
            model.fit(X, y)
            pred = model.predict(X)
            train_rmse = float(root_mean_squared_error(y.astype(float), np.asarray(pred, dtype=float)))
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            traceback_text = traceback.format_exc()
        finally:
            duration = time.perf_counter() - start
            monitor.stop()

        peak_cuda_allocated_gb, peak_cuda_reserved_gb = _cuda_peaks_gb()
        baseline_rss_gb = _bytes_to_gb(monitor.baseline_rss_bytes)
        peak_rss_gb = _bytes_to_gb(monitor.peak_rss_bytes)
        rss_delta_gb = None
        if baseline_rss_gb is not None and peak_rss_gb is not None:
            rss_delta_gb = peak_rss_gb - baseline_rss_gb

        result = StandaloneComplexityResult(
            batch_size=batch_size,
            status=status,
            error=error,
            traceback=traceback_text,
            duration_seconds=duration,
            baseline_rss_gb=baseline_rss_gb,
            peak_rss_gb=peak_rss_gb,
            rss_delta_gb=rss_delta_gb,
            peak_cuda_allocated_gb=peak_cuda_allocated_gb,
            peak_cuda_reserved_gb=peak_cuda_reserved_gb,
            train_rmse=train_rmse,
        )
        results.append(result)

        print(
            f"{prefix} done | status={result.status} duration={result.duration_seconds:.2f}s "
            f"peak_rss_gb={result.peak_rss_gb} peak_cuda_allocated_gb={result.peak_cuda_allocated_gb:.4f}"
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_experiment(args: argparse.Namespace) -> tuple[RunResult, dict[str, Any]]:
    X, y = make_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        informative_ratio=args.informative_ratio,
        noise=args.noise,
        random_state=args.seed,
    )

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    estimator = build_pipeline(device=device, max_epochs=args.max_epochs, batch_size=args.batch_size)

    # MotherTuner defaults come from class defaults; only set scorer + trial budget knobs requested.
    tuner = MotherTuner(
        scorer="neg_root_mean_squared_error",
        n_trials_optuna=args.n_trials,
        n_startup_trials=min(args.n_startup_trials, args.n_trials),
        n_threads_optuna=args.n_jobs,
        seed=args.seed,
    )

    outer_cv = KFold(n_splits=args.outer_cv_splits, shuffle=True, random_state=args.seed)
    inner_cv = KFold(n_splits=args.inner_cv_splits, shuffle=True, random_state=args.seed)

    # Enable visible tuning logs from Mother/Optuna internals.
    logging.basicConfig(
        level=getattr(logging, args.tuning_log_level),
        format="%(asctime)s | %(name)30s | %(levelname)8s | %(message)s",
    )
    logging.getLogger("mother.optimization.core").setLevel(getattr(logging, args.tuning_log_level))
    logging.getLogger("mother.pipeline_utils").setLevel(getattr(logging, args.tuning_log_level))

    monitor = PeakRSSMonitor(poll_interval_seconds=args.rss_poll_seconds)
    monitor.start()
    start = time.perf_counter()

    status = "ok"
    error = None
    traceback_text = None
    rmse_mean: Optional[float] = None
    rmse_std: Optional[float] = None

    best_trial_number: Optional[int] = None
    best_trial_value: Optional[float] = None
    best_params: Optional[dict[str, Any]] = None

    perf_data_dict: dict[str, Any] = {}

    perf_data_container: dict[str, Any] = {}
    est_dict_container: dict[str, Any] = {}
    worker_error: dict[str, Exception] = {}
    worker_traceback: dict[str, str] = {}
    done = threading.Event()

    def _run_mother_cv() -> None:
        try:
            perf_data, est_dict = mother_cv(
                estimator=estimator,
                cv=outer_cv,
                inner_cv=inner_cv,
                X=X,
                y=y,
                tuner=tuner,
                return_estimators=True,
                prediction_prefix="pred_",
            )
            perf_data_container["perf_data"] = perf_data
            est_dict_container["est_dict"] = est_dict
        except Exception as exc:  # noqa: BLE001
            worker_error["error"] = exc
            worker_traceback["traceback"] = traceback.format_exc()
        finally:
            done.set()

    worker = threading.Thread(target=_run_mother_cv, name="mothercv-worker", daemon=True)
    worker.start()

    last_seen_trials = -1
    last_seen_complete = -1

    try:
        while not done.wait(timeout=args.progress_seconds):
            if not args.print_tuning_progress:
                continue

            if tuner.study is None:
                print("[tuning] study not initialized yet")
                continue

            trials = list(tuner.study.trials)
            n_total = len(trials)
            n_complete = sum(t.state.name == "COMPLETE" for t in trials)
            n_pruned = sum(t.state.name == "PRUNED" for t in trials)
            n_fail = sum(t.state.name == "FAIL" for t in trials)

            if n_total == last_seen_trials and n_complete == last_seen_complete:
                continue

            last_seen_trials = n_total
            last_seen_complete = n_complete

            best_text = "n/a"
            if n_complete > 0:
                try:
                    best_text = f"{float(tuner.study.best_trial.value):.6f}"
                except Exception:  # noqa: BLE001
                    best_text = "n/a"

            print(
                "[tuning] trials={total}/{target} complete={complete} pruned={pruned} fail={fail} best={best}".format(
                    total=n_total,
                    target=args.n_trials,
                    complete=n_complete,
                    pruned=n_pruned,
                    fail=n_fail,
                    best=best_text,
                )
            )

        worker.join()

        if worker_error:
            raise worker_error["error"]

        perf_data = perf_data_container["perf_data"]
        est_dict = est_dict_container["est_dict"]

        pred_cols = [c for c in perf_data.columns if c.startswith("pred_")]
        if pred_cols:
            pred_series = perf_data[pred_cols[0]].astype(float)
            true_series = perf_data[y.name].astype(float)
            rmse_per_row = (true_series - pred_series) ** 2
            rmse_mean = float(root_mean_squared_error(true_series, pred_series))
            rmse_std = float(np.sqrt(np.std(rmse_per_row.to_numpy(dtype=float))))

        if tuner.study is not None:
            best_trial_number = tuner.study.best_trial.number
            best_trial_value = tuner.study.best_trial.value
            best_params = dict(tuner.study.best_trial.params)

        perf_data_dict = {
            "n_rows": int(perf_data.shape[0]),
            "columns": list(perf_data.columns),
            "estimator_count": len(est_dict.get("estimators", [])),
        }
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        traceback_text = worker_traceback.get("traceback", traceback.format_exc())
    finally:
        duration = time.perf_counter() - start
        monitor.stop()

    peak_cuda_allocated_gb, peak_cuda_reserved_gb = _cuda_peaks_gb()
    baseline_rss_gb = _bytes_to_gb(monitor.baseline_rss_bytes)
    peak_rss_gb = _bytes_to_gb(monitor.peak_rss_bytes)
    rss_delta_gb = None
    if baseline_rss_gb is not None and peak_rss_gb is not None:
        rss_delta_gb = peak_rss_gb - baseline_rss_gb

    result = RunResult(
        status=status,
        error=error,
        traceback=traceback_text,
        duration_seconds=duration,
        baseline_rss_gb=baseline_rss_gb,
        peak_rss_gb=peak_rss_gb,
        rss_delta_gb=rss_delta_gb,
        peak_cuda_allocated_gb=peak_cuda_allocated_gb,
        peak_cuda_reserved_gb=peak_cuda_reserved_gb,
        rmse_mean=rmse_mean,
        rmse_std=rmse_std,
        best_trial_number=best_trial_number,
        best_trial_value=best_trial_value,
        best_params=best_params,
    )

    details = {
        "device": device,
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "inner_cv_splits": args.inner_cv_splits,
        "outer_cv_splits": args.outer_cv_splits,
        "n_trials": args.n_trials,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "performance_table": perf_data_dict,
    }

    return result, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--n-features", type=int, default=3000)
    parser.add_argument("--informative-ratio", type=float, default=0.9)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--outer-cv-splits", type=int, default=3)
    parser.add_argument("--inner-cv-splits", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--n-startup-trials", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Legacy single tuning batch size (used only if --tuning-batch-sizes is not provided).",
    )
    parser.add_argument(
        "--tuning-batch-sizes",
        type=str,
        default="256,512,1000",
        help="Comma-separated batch sizes for MotherTuner + MotherCV runs.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tuning-log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--progress-seconds", type=float, default=10.0)
    parser.add_argument("--print-tuning-progress", action="store_true", default=True)
    parser.add_argument("--no-print-tuning-progress", dest="print_tuning_progress", action="store_false")
    parser.add_argument("--rss-poll-seconds", type=float, default=0.05)
    parser.add_argument(
        "--run-standalone-high-complexity",
        action="store_true",
        help="Also run standalone (non-tuning) high-complexity NODE fits across batch sizes.",
    )
    parser.add_argument(
        "--standalone-epochs",
        type=int,
        default=50,
        help="Epoch count for standalone high-complexity runs.",
    )
    parser.add_argument(
        "--standalone-batch-sizes",
        type=str,
        default="256,512,1000",
        help="Comma-separated batch sizes for standalone high-complexity runs.",
    )
    parser.add_argument("--standalone-num-layers", type=int, default=8)
    parser.add_argument("--standalone-num-trees", type=int, default=256)
    parser.add_argument("--standalone-depth", type=int, default=6)
    parser.add_argument("--standalone-additional-tree-output-dim", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_artifacts/node_mothercv_tuner_default_run.json"),
    )
    args = parser.parse_args()
    args.tuning_batch_sizes = _parse_int_list(args.tuning_batch_sizes)
    args.standalone_batch_sizes = _parse_int_list(args.standalone_batch_sizes)
    return args


def main() -> None:
    args = parse_args()

    tuning_runs: list[dict[str, Any]] = []
    for tuning_batch_size in args.tuning_batch_sizes:
        run_args = argparse.Namespace(**vars(args))
        run_args.batch_size = tuning_batch_size
        print(f"[tuning] running MotherCV + MotherTuner for batch_size={tuning_batch_size}")
        result, details = run_experiment(run_args)
        tuning_runs.append(
            {
                "batch_size": tuning_batch_size,
                "details": details,
                "result": asdict(result),
            }
        )

    # Preserve a primary result for backwards-compatible console summary fields.
    primary = tuning_runs[0]
    result = RunResult(**primary["result"])
    details = primary["details"]

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    standalone_results: list[StandaloneComplexityResult] = []
    if args.run_standalone_high_complexity:
        print("[standalone] running high-complexity batch-size sweep")
        standalone_results = run_standalone_complexity_sweep(args, device=device)

    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "script": "node_mothercv_tuner_default_run.py",
        "config": {
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "informative_ratio": args.informative_ratio,
            "noise": args.noise,
            "outer_cv_splits": args.outer_cv_splits,
            "inner_cv_splits": args.inner_cv_splits,
            "n_trials": args.n_trials,
            "n_startup_trials": args.n_startup_trials,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "tuning_batch_sizes": args.tuning_batch_sizes,
            "n_jobs": args.n_jobs,
            "seed": args.seed,
            "device": args.device,
            "run_standalone_high_complexity": args.run_standalone_high_complexity,
            "standalone_epochs": args.standalone_epochs,
            "standalone_batch_sizes": args.standalone_batch_sizes,
            "standalone_num_layers": args.standalone_num_layers,
            "standalone_num_trees": args.standalone_num_trees,
            "standalone_depth": args.standalone_depth,
            "standalone_additional_tree_output_dim": args.standalone_additional_tree_output_dim,
        },
        "details": details,
        "result": asdict(result),
        "tuning_runs": tuning_runs,
        "standalone_results": [asdict(item) for item in standalone_results],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("NODE MotherCV + MotherTuner default benchmark")
    for item in tuning_runs:
        run = item["result"]
        print(f"[tuning][batch_size={item['batch_size']}] status={run['status']}")
        print(f"[tuning][batch_size={item['batch_size']}] duration_seconds={run['duration_seconds']:.2f}")
        print(
            "[tuning][batch_size={bs}] cpu_rss_gb: baseline={baseline}, peak={peak}, delta={delta}".format(
                bs=item["batch_size"],
                baseline=run["baseline_rss_gb"],
                peak=run["peak_rss_gb"],
                delta=run["rss_delta_gb"],
            )
        )
        print(
            "[tuning][batch_size={bs}] cuda_peak_gb: allocated={allocated:.4f}, reserved={reserved:.4f}".format(
                bs=item["batch_size"],
                allocated=float(run["peak_cuda_allocated_gb"]),
                reserved=float(run["peak_cuda_reserved_gb"]),
            )
        )
        if run.get("best_params") is not None:
            print(f"[tuning][batch_size={item['batch_size']}] best_trial_number={run['best_trial_number']}")
            print(f"[tuning][batch_size={item['batch_size']}] best_trial_value={run['best_trial_value']}")
            print(f"[tuning][batch_size={item['batch_size']}] best_params=")
            print(json.dumps(run["best_params"], indent=2, sort_keys=True))
        if run.get("rmse_mean") is not None:
            print(f"[tuning][batch_size={item['batch_size']}] rmse_mean={run['rmse_mean']}")
            print(f"[tuning][batch_size={item['batch_size']}] rmse_std_proxy={run['rmse_std']}")
    if standalone_results:
        print("standalone_high_complexity=")
        print(json.dumps([asdict(item) for item in standalone_results], indent=2))
    if result.error:
        print(f"error={result.error}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
