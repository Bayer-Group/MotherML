# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-15

### Added

- New notebook covering the `predict` and `predict_uncertainty` interface.
- Updated notebook examples for `predict_uncertainty` usage with `CatboostGaussianProcessRegressorMother`.

### Changed

- Harmonized prediction outputs across CatBoost, RandomForest, and TabPFN backends.
- Improved `mother_cv` typing and estimator return behavior.
- Added internal workflow skill to update docs and changelog.

### Fixed

- CatBoost regressor uncertainty outputs for data and total uncertainty.
- RandomForest regressor uncertainty outputs for data and knowledge uncertainty.
- Multi-target prediction fallback behavior and uncertainty handling in pipelines.
- Single-target enforcement and `**kwargs` support in `CatboostGaussianProcessRegressorMother` prediction methods.
- Quantile validation error messaging in `CatboostRegressorMother`.
- Notebook text/formatting issues in prediction interface guide.
- Probability DataFrame index alignment consistency in prediction outputs.

### Tests

- Expanded tests for unified `predict` behavior across regression and classification.
- Added/updated tests for uncertainty predictions and pipeline integration consistency.

## [1.0.0] - 2026-04-17

### Added

- First stable release baseline for the 1.x series.
