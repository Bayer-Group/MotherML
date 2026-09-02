# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## v1.1.3 (2026-08-31)

### Bug Fixes

- Fix use count based fingerprints as default, including TabPFN precision handling and test updates
  ([#75](https://github.com/Bayer-Group/MotherML/pull/75),
  [`74cac2ec`](https://github.com/Bayer-Group/MotherML/commit/74cac2ec0f93fa7f436c5fc6f4f5c8f711ae1a8a))

- (tabpfn) Restore upstream TabPFN precision handling with float32 for all operations


## v1.1.2 (2026-08-25)

### Bug Fixes

- Solving the `max_features` parameter not being correctly passed to the superclass
  ([#40](https://github.com/Bayer-Group/MotherML/pull/40),
  [`4fd6869c`](https://github.com/Bayer-Group/MotherML/commit/4fd6869c3b3218cedf96060bcf4afe84ba70ece5))


## v1.1.1 (2026-08-21)

### Chores

- Update changelog for version 1.1.0 with bug fixes and new features
  ([`8a69fa24`](https://github.com/Bayer-Group/MotherML/commit/8a69fa246b449f257dacd93d0cbbd28b28090532))


## v1.1.0 (2026-08-20)

### Bug Fixes

- (tabpfn) Corrected a bug concerning unsupported BFloat16 for tabpfn. Now force np.float32 for all
  operations. ([#21](https://github.com/Bayer-Group/MotherML/pull/21),
  [`ca45d17`](https://github.com/Bayer-Group/MotherML/commit/ca45d17e86659d1ee00dae5398737982bd7da63a))

### Features

- (tabicl) add MotherML wrappers, embeddings and uncertainty support
  ([#21](https://github.com/Bayer-Group/MotherML/pull/21),
  [`ca45d17`](https://github.com/Bayer-Group/MotherML/commit/ca45d17e86659d1ee00dae5398737982bd7da63a))


## v1.0.6 (2026-08-18)

### Bug Fixes

- Declare scipy explicitly in base dependencies
  ([#66](https://github.com/Bayer-Group/MotherML/pull/66),
  [`5436111`](https://github.com/Bayer-Group/MotherML/commit/5436111089649394145affef55992369f9401203))

- Local TabPFN import drift and declare SciPy explicitly
  ([#66](https://github.com/Bayer-Group/MotherML/pull/66),
  [`5436111`](https://github.com/Bayer-Group/MotherML/commit/5436111089649394145affef55992369f9401203))


## v1.0.5 (2026-08-17)

### Bug Fixes

- Make MVS default boostrap for catboost tuning
  ([#61](https://github.com/Bayer-Group/MotherML/pull/61),
  [`7141902`](https://github.com/Bayer-Group/MotherML/commit/7141902546738c6a7535bba233f53e87d9c7f4ec))

- Make MVS the default bootstrap for catboost tuning
  ([#61](https://github.com/Bayer-Group/MotherML/pull/61),
  [`7141902`](https://github.com/Bayer-Group/MotherML/commit/7141902546738c6a7535bba233f53e87d9c7f4ec))


## v1.0.4 (2026-07-23)

### Bug Fixes

- Correct typo in workflow configuration for error handling
  ([`1a54319`](https://github.com/Bayer-Group/MotherML/commit/1a5431905e32432f1ce261358b6d17b66d7b6c85))


## v1.0.3 (2026-07-21)

### Bug Fixes

- Add missing readme field in project metadata
  ([`3caa87b`](https://github.com/Bayer-Group/MotherML/commit/3caa87b1e10e2b8ef717199fa36e4b42640a118e))

- Update Python version and actions in workflow configuration
  ([`f14ef68`](https://github.com/Bayer-Group/MotherML/commit/f14ef688432e879ad40c4f06a5bc804fffeefc11))

### Chores

- Uv audit
  ([`e6ac96c`](https://github.com/Bayer-Group/MotherML/commit/e6ac96c67294d314fe4442e7f5c1b5702c485909))


## v1.0.2 (2026-07-21)

### Bug Fixes

- Update entry for docs-python-fences hook to use 'uv run poe'
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

### Documentation

- Add CLAUDE.md for project guidance and command usage
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Enhance documentation and add python code validation for markdown files
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Enhance SKILL.md for changelog management and update __init__.py for version handling
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Remove single-source dependency from pyproject.toml and uv.lock
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Update README and examples for improved clarity and navigation
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))

- Update SKILL.md and CHANGELOG.md for improved documentation workflow and versioning
  ([#55](https://github.com/Bayer-Group/MotherML/pull/55),
  [`f09baf4`](https://github.com/Bayer-Group/MotherML/commit/f09baf4e26b9464d9f3be5a38f8e748798bcf38d))


## v1.0.1 (2026-06-15)

### Bug Fixes

- **ml**: Unify predict_uncertainty output schema across all model backends and pipelines
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Add data and total uncertainty outputs to CatboostRegressorMother predictions
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Add knowledge and data uncertainty outputs to RandomForestRegressorMother predictions
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Enforce single target type in CatboostGaussianProcessRegressorMother, update predict()
  and predict_uncertainty() to accept **kwargs
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Harmonize prediction outputs across model backends
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Improve error message for quantile validation in CatboostRegressorMother
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **models**: Initialize quantile_array in CatboostRegressorMother predict method
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **notebook**: Correct minor text and formatting issues in prediction interface guide
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **notebook**: Update predict_uncertainty interface examples and add
  CatboostGaussianProcessRegressorMother ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **pipeline**: Add overloads for mother_cv function to enhance type hints
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **pipeline**: Improve mother_cv function to conditionally return estimators
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **pipeline**: Update multi-target prediction fallback to clarify uncertainty handling
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- **tests**: Enhance uncertainty predictions in CatboostRegressorMother and
  GaussianProcessRegressorMother tests ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

### Chores

- Added skill to use ai for changelog and docs update
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

- Uv audit ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))

### Testing

- **models**: Add regression and classification tests for unified predict behavior
  ([#16](https://github.com/Bayer-Group/MotherML/pull/16),
  [`2931c6e`](https://github.com/Bayer-Group/MotherML/commit/2931c6e6360753d1ba0b65c1a56695d3bcf86303))


## v1.0.0 (2026-04-17)

- Initial Release
