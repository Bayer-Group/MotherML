# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`mother-ml` (imported as `mother`) is a scikit-learn-based ML framework for predicting properties from chemical molecules: SMILES preprocessing, molecular feature generation, chemical-similarity-aware cross-validation, model training/tuning, and RNA/transcriptomics preprocessing. Every capability is exposed as an sklearn `transformer` or `estimator`, so all methods compose via sklearn `Pipeline`, `ColumnTransformer`, and `FeatureUnion`.

The build backend is `uv_build` with `module-name = "mother"`; source lives under `src/mother/`. Python 3.11–3.14.

## Commands

Uses `uv` + `poe` (poethepoet). Run tasks with `uv run poe <task>`.

- **Setup**: `uv sync` (base) or `uv sync --all-extras --all-groups` (full dev). Optional extras: `report`, `rna`, `torch` (~3GB), `tabpfn`, `clustering`.
- **Unit tests**: `uv run poe test-unit` — runs `pytest test -m 'not slow and not serial' -n auto` then serial tests with `-n 0`.
- **Slow tests**: `uv run poe test-slow` (marker `slow`).
- **Single test**: `uv run pytest test/unit/test_ml.py::TestClass::test_name` (add `-p no:cacheprovider` if needed). Markers: `slow`, `serial` (serial tests modify global sklearn config and must not run in parallel).
- **Coverage**: `uv run poe coverage` (writes `coverage-report.xml` / xunit); `uv run poe serve-coverage` for HTML.
- **Style (check)**: `uv run poe check-style` = `check-sort-imports` (isort, black profile) + `check-format` (ruff format). Apply with `uv run poe style`.
- **Static analysis**: `uv run poe check-static-analysis` = `check-lint` (ruff, E+F) + `check-types` (mypy on `src`, strict: `disallow_untyped_defs`).
- **Lint autofix**: `uv run poe lint`.
- **Docs**: `uv run poe docs` (build) / `uv run poe serve-docs` (mkdocs serve from `mkdocs/`).
- **Pre-commit**: `uv run poe install-hook` then `uv run poe check-hook`.

Line length: ruff 120, pylint 100. Note: `CONTRIBUTING.md` references some task names (`poe test`, `test-acceptance`, `check-docs`) that do not exist in `pyproject.toml` — use the verified names above.

## Architecture

The whole framework layers custom classes on top of sklearn so that a molecule-to-prediction workflow is one composable sklearn object. Data generally flows: **SMILES DataFrame → standardized SMILES → rdkit mol objects → feature DataFrame → feature selection → model**.

### Hyperparameter "rooting" (central abstraction)
`src/mother/ml/core.py` defines `AbstractMotherPipeline` (ABC) — an sklearn-compatible estimator that additionally requires `get_hyperparameter_space(X, y, trial, prefix)` and `default_parameters(prefix)`. The subclasses `PipelineWithHyperparameterRooting`, `ColumnTransformerWithHyperparameterRooting`, and `FeatureUnionWithHyperparameterRooting` subclass the corresponding sklearn compose classes AND `_SklComposeWithHyperparameterRooting`. Their key trick: they walk their own steps, call each step's `get_hyperparameter_space`, and concatenate the results with sklearn-style `step_name__param` prefixes. This lets Optuna tune parameters across an entire nested pipeline. When adding a tunable step, implement these two methods so the space propagates.

`AbstractMotherPipeline` also defines the uniform `predict_uncertainty(X)` contract returning a DataFrame with columns `pred, mean_predictions, knowledge_uncertainty, data_uncertainty, total_uncertainty` (per target, prefixed for multi-target; `proba_*` columns for classifiers). Model backends override it; the base provides a fallback. Keep this output schema stable — `mother_cv` and downstream code depend on it.

### Model registry (dynamic discovery)
`src/mother/ml/__init__.py` holds `MotherModelRegistry`, a singleton that scans `src/mother/ml/models/m_*.py`, imports each, and registers every class subclassing `AbstractMotherPipeline`. Backends live in `models/`: `m_catboost.py` (default; regressor/classifier/ranker/GP-regressor), `m_randomForest.py`, `m_lasso.py`, `m_tabpfn.py`. Look up via `ml.get_model_class(name)`, `ml.get_model_class_by_algorithm_and_type(algorithm, model_type)`, `ml.get_available_algorithms()`. **To add a new model**: drop a `m_<algo>.py` in `models/` with a class extending `AbstractMotherPipeline` (auto-discovered), or use the `@ml.register_model("algo")` decorator. Import failures for a backend (e.g. missing torch extra) are caught and logged, not fatal.

### Configuration system (pydantic-settings)
`src/mother/settings.py` `MotherSettings` is the top-level config aggregating per-module pydantic config models (`InputConfig`, `PipelineConfig`, `PreprocessingConfig`, `FeatureGenerationConfig`, `CVSettings`, `ModelConfig`, `MotherTunerConfig`). Load/save via `MotherSettings.create()`, `.load_from_yaml()`, `.dump_to_yaml()`; defaults come from `src/mother/data/mother_config.yaml`. Env override prefix is `mother_` (and `mother_ml_` for model config). Every transformer can also be built directly from a plain dict — settings are a convenience layer, not required.

### `pipeline_utils.py` (the "mother takes care" builders)
Factory functions that assemble sklearn objects from settings/dicts: `get_preprocessing_pipeline`, `get_feature_generation_pipeline`, `get_feature_selection_pipeline` (builds a `ColumnTransformerWithHyperparameterRooting` from `FeatureSelectionFlags`), `get_model`, `get_ranking_pipeline`, and `mother_cv` — the framework's own cross-validation loop that returns a tidy DataFrame of per-fold predictions + uncertainty + group/iteration metadata, with optional nested CV when a `MotherTuner` is passed.

### Module map (`src/mother/`)
- `preprocessing/` — `StandardizerTransformer` (SMILES cleanup via `flags`: STANDARDIZE/DESALT/NEUTRALIZE), `SmilesToMolTransformer` (→ rdkit mols).
- `feature_generation/` — `MaccsFingerprints`, `MorganFingerprints`, `ChemicalDescriptors`, `FingerprintsGeneric`; combine with `FeatureUnion`.
- `cv/` — `TanimotoGroupingFromMols` (chemical-similarity groups), `TimeSeriesGrouping`, `DefaultGrouping`; feed groups into sklearn `GroupKFold`/`StratifiedGroupKFold`.
- `optimization/` — `MotherTuner` (Optuna-based, consumes the rooted hyperparameter space).
- `ml/` — models, registry, `estimators.py` (`MotherSelectFromModel`, `MotherBorutaPy`, permutation/catboost importance selectors), `properties.py` (`ModelType`, `TargetType`, `FeatureSelectionFlags` IntFlag), `rna.py` (RNA pipeline: normalization UQ/CUF/CPM/Scanpy + feature selection + discretization).
- `errors.py` — `ExtrasDependencyImportError` (friendly "pip install mother[extra]" message), `ConfigurationError`.

## Conventions

- **Commits use Conventional Commits** — versioning/changelog are automated by `python-semantic-release` (`feat`→minor, `fix`/`perf`/etc.→patch; `chore`/`ci`/`refactor`/`style`/`test` excluded from changelog). Never bump the version manually; CI does it. Tag format `v{version}`.
- **Never push to `main` directly** — all changes via PR (fork or maintainer branch). Reference issues with `Fixes #N`.
- New behavior needs tests under `test/unit/` (`test_*.py`); shared fixtures in `test/unit/conftest.py`. Use the `serial` marker for anything that mutates global sklearn config (e.g. `set_config(enable_metadata_routing=...)`).
- Example workflows (the best usage reference) live in `examples/notebooks/` organized `01_basics` … `05_advanced`.
