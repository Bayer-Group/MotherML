#!/usr/bin/env python3
"""Hardcoded NODE memory stability test for MotherCV and MotherTuner.

This script intentionally avoids CLI parameters and uses fixed settings:
- Dense ChEMELEON-like synthetic data (continuous descriptors)
- n_samples=10000, n_features=3000
- batch_size=512
- max_epochs=40
- n_trials=40

Execution order:
1) Standalone high-complexity fit (no CV, no tuner)
2) mother_cv without tuning, return_estimators=False
3) mother_cv without tuning, return_estimators=True
4) mother_cv with MotherTuner optimization

Outputs are written to fixed artifact paths.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

# Ensure imports work when executing this file directly via `python scripts/...`.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from node_mothercv_tuner_default_run import (
    RunResult,
    StandaloneComplexityResult,
    run_experiment,
    run_standalone_complexity_sweep,
)

OUTPUT_JSON = Path("benchmark_artifacts/node_memory_stability_chemeleon_10k_3k_bs512_e40_t40.json")
OUTPUT_SUMMARY = Path("benchmark_artifacts/node_memory_stability_chemeleon_10k_3k_bs512_e40_t40_summary.txt")


def hardcoded_config() -> SimpleNamespace:
    # Dense ChEMELEON-like setup: continuous descriptor matrix (non-sparse).
    return SimpleNamespace(
        n_samples=10000,
        n_features=3000,
        informative_ratio=0.9,
        noise=0.2,
        outer_cv_splits=3,
        inner_cv_splits=3,
        n_trials=40,
        n_startup_trials=12,
        max_epochs=40,
        batch_size=512,
        tuning_batch_sizes=[512],
        n_jobs=1,
        seed=42,
        device="auto",
        tuning_log_level="WARNING",
        progress_seconds=10.0,
        print_tuning_progress=True,
        rss_poll_seconds=0.05,
        run_standalone_high_complexity=True,
        standalone_epochs=40,
        standalone_batch_sizes=[512],
        standalone_num_layers=1,
        standalone_num_trees=2048,
        standalone_depth=6,
        standalone_additional_tree_output_dim=3,
        output=OUTPUT_JSON,
    )


def sanitize_details(details: dict[str, Any]) -> dict[str, Any]:
    """Hide any table column names from output while keeping useful shape stats."""
    out = dict(details)
    perf = dict(out.get("performance_table", {}))
    if "columns" in perf:
        perf["n_columns"] = len(perf["columns"])
        perf.pop("columns", None)
    out["performance_table"] = perf
    return out


def summarize_run(name: str, result: RunResult) -> str:
    return (
        f"{name}: status={result.status}, duration_s={result.duration_seconds:.2f}, "
        f"peak_rss_gb={result.peak_rss_gb}, peak_cuda_allocated_gb={result.peak_cuda_allocated_gb:.4f}, "
        f"peak_cuda_reserved_gb={result.peak_cuda_reserved_gb:.4f}"
    )


def summarize_standalone(item: StandaloneComplexityResult) -> str:
    return (
        f"standalone(batch={item.batch_size}): status={item.status}, duration_s={item.duration_seconds:.2f}, "
        f"peak_rss_gb={item.peak_rss_gb}, peak_cuda_allocated_gb={item.peak_cuda_allocated_gb:.4f}, "
        f"peak_cuda_reserved_gb={item.peak_cuda_reserved_gb:.4f}"
    )


def main() -> None:
    args = hardcoded_config()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    print("[config] hardcoded memory stability run")
    print(
        json.dumps(
            {
                "dataset_profile": "chemeleon_dense",
                "n_samples": args.n_samples,
                "n_features": args.n_features,
                "batch_size": args.batch_size,
                "epochs": args.max_epochs,
                "tuning_trials": args.n_trials,
                "device": device,
                "output_json": str(args.output),
                "output_summary": str(OUTPUT_SUMMARY),
            },
            indent=2,
        )
    )

    # Stage 1: highest-complexity standalone first.
    print("[stage 1/4] standalone high complexity")
    standalone_results = run_standalone_complexity_sweep(args, device=device)

    # Stage 2: mother_cv without tuning and without returning estimators.
    print("[stage 2/4] mother_cv no tuner, return_estimators=False")
    no_tuner_no_models_result, no_tuner_no_models_details = run_experiment(
        args,
        use_tuner=False,
        return_estimators=False,
    )

    # Stage 3: mother_cv without tuning and with returned estimators.
    print("[stage 3/4] mother_cv no tuner, return_estimators=True")
    no_tuner_with_models_result, no_tuner_with_models_details = run_experiment(
        args,
        use_tuner=False,
        return_estimators=True,
    )

    # Stage 4: mother_cv with tuner (single hardcoded batch size 512).
    print("[stage 4/4] mother_cv with MotherTuner (40 trials)")
    tuning_result, tuning_details = run_experiment(
        args,
        use_tuner=True,
        return_estimators=True,
    )

    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "script": "node_memory_stability_hardcoded.py",
        "config": {
            "dataset_profile": "chemeleon_dense",
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "batch_size": args.batch_size,
            "epochs": args.max_epochs,
            "tuning_trials": args.n_trials,
            "device": device,
            "standalone_num_layers": args.standalone_num_layers,
            "standalone_num_trees": args.standalone_num_trees,
            "standalone_depth": args.standalone_depth,
            "standalone_additional_tree_output_dim": args.standalone_additional_tree_output_dim,
        },
        "stage_standalone_high_complexity": [asdict(item) for item in standalone_results],
        "stage_mothercv_no_tuner_no_estimators": {
            "details": sanitize_details(no_tuner_no_models_details),
            "result": asdict(no_tuner_no_models_result),
        },
        "stage_mothercv_no_tuner_with_estimators": {
            "details": sanitize_details(no_tuner_with_models_details),
            "result": asdict(no_tuner_with_models_result),
        },
        "stage_mothercv_tuner": {
            "details": sanitize_details(tuning_details),
            "result": asdict(tuning_result),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "NODE memory stability hardcoded run",
        summarize_standalone(standalone_results[0]) if standalone_results else "standalone: not run",
        summarize_run("mother_cv no tuner (return_estimators=False)", no_tuner_no_models_result),
        summarize_run("mother_cv no tuner (return_estimators=True)", no_tuner_with_models_result),
        summarize_run("mother_cv with tuner", tuning_result),
        f"saved_json={args.output}",
    ]

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines:
        print(line)

    if tuning_result.best_params is not None:
        print("best_params_effective=")
        print(json.dumps(tuning_result.best_params, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
