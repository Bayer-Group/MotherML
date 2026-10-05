# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- version list -->

## v2.0.1 (2026-10-01)

### Bug Fixes

- Reintegrate release into workflow.yml
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

- Update job dependencies in workflow.yml for correct execution order
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

### Chores

- Fix CI/CD workflows for release and manual PyPI publishing
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

- Update action versions and set timeouts for workflows
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

### Continuous Integration

- Enhance workflow validation and release process
  ([#93](https://github.com/Bayer-Group/MotherML/pull/93),
  [`ef72ce0`](https://github.com/Bayer-Group/MotherML/commit/ef72ce0ab4b5acbb79c2fc1262ec9a9fbcc4dffb))

---

**Detailed Changes**: [v2.0.0...v2.0.1](https://github.com/Bayer-Group/MotherML/compare/v2.0.0...v2.0.1)


## v2.0.0 (2026-09-29)

### Bug Fixes

- **catboost**: Fix loss_function=None handling, set_params rollback, and doc example
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost**: Harden ranker loss-function handling and GP regressor tuning
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost**: Reject top=/mode=Classic conflicts, fix stale top/max_pairs, fix avg_ndcg_score direction
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Apply mode=NDCG fix-up to top= embedded via set_params
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Compare embedded loss-string params numerically
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Make set_params atomic and validate ranking array shapes
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Require explicit mode for embedded top=, revert avg_ndcg_score shift
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Validate malformed top/max_pairs tokens, fix avg_ndcg_score DataFrame iteration
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking**: Validate top/max_pairs family gaps, dedupe group-indexing
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost,ranking,tabpfn**: Harden loss-function validation, atomic set_params, uncertainty wrapper consistency
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ml**: Stop GP unpickling from re-enabling disabled tuning flags
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Enable metadata routing in tutorial, document pred normalization
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Reject 2-D group_id/group_ids instead of silently flattening
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Reject missing values in avg_ndcg_score groups
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Reject zero-column ensemble arrays, document larger-is-better convention
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Validate k strictly and clean up top-k analysis docs/notebook
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Validate missing group IDs in top-k analysis
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **tabpfn**: Honor configured device for pre-fitted models
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **tabpfn**: Validate pre-fitted model before embedding extraction
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

### Chores

- Fix example notebook formatting and nbqa check
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Remove deprecated unit tests for example notebooks
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Remove unmaintained polaris-lib dependency and fix example notebooks (still issues)
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Speedup env creation for style check
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Update actions to latest versions in workflow
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Update workflow for Python version and remove test-pypi-publish job
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- Uv audit
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Remove PR status doc, not meant for the repo
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

### Documentation

- Add CI/CD & Release Overview to documentation
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Document larger-is-better convention in avg_ndcg_score
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Update PR status doc for top=/mode fix, mother_cv note, test fixes
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **tabpfn**: Clarify device/precision are not enforced for pre-fitted models
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

### Features

- **ranking**: Add CatBoost ranker tuning and uncertainty workflows
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **ranking**: Add functions to convert scores to ranks and score matrices to ranks
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

### Refactoring

- Remove wrong _TransformOnlyValidMols usage
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

- **catboost**: Move ranking utility functions to utils module
  ([#45](https://github.com/Bayer-Group/MotherML/pull/45),
  [`5b477ae`](https://github.com/Bayer-Group/MotherML/commit/5b477ae1303d6b7c05b706ac2742e84bfe3d3599))

---

**Detailed Changes**: [v1.2.0...v2.0.0](https://github.com/Bayer-Group/MotherML/compare/v1.2.0...v2.0.0)

**Resolved Issues**: [#25](https://github.com/Bayer-Group/MotherML/issues/25),
[#33](https://github.com/Bayer-Group/MotherML/issues/33),
[#57](https://github.com/Bayer-Group/MotherML/issues/57).


## v1.2.0 (2026-09-01)

### Bug Fixes

- Potential fix for pull request finding
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Remove changelog file reference from semantic release configuration
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Update actions to latest versions in workflow and sync package version in uv.lock
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Update release preflight script to validate commits against the main branch and remove outdated report file
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

### Chores

- Uv audit
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

### Continuous Integration

- Enhance release workflow with preflight checks and reporting
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

### Documentation

- Add changelog file configuration to pyproject.toml
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Add initial configuration for Zensical documentation site
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Add markdown extensions and update mkdocstrings configuration
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Add missing changelog entries for v1.1.1, v1.1.2, and v1.1.3
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

- Update README to include 'Why Mother?" section
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

### Features

- Add release preflight workflow and reporting script
  ([#72](https://github.com/Bayer-Group/MotherML/pull/72),
  [`ab22ecf`](https://github.com/Bayer-Group/MotherML/commit/ab22ecfa997b15d6ca163cdcf56391be08bc5082))

---

**Detailed Changes**: [v1.1.3...v1.2.0](https://github.com/Bayer-Group/MotherML/compare/v1.1.3...v1.2.0)


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
