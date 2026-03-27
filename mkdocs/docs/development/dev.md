# Development Setup

This python based project includes a lot of best pratices that have been established within CLS. To get familiar with the most basic ones please see have a look at the `pyproject.toml` file. If you are not familiar with [uv](#using-uv) please go ahead and read the following sections. It is highly recommended to use uv to properly handle python dependencies within this project.

To provide a stable model an essential part is clean code. To support this, several things have been preconfigured here.

For example,

- python code linting
- code formatting
- unit tests
- credential handling
- additional checks via pre-commit hooks

Most of the above mentioned examples is discussed in more detail below.


## Development installation

In case you want to develop mother further, do the following steps:

- install uv following the steps outlined here (<https://docs.astral.sh/uv/getting-started/installation/>)
- clone the repository
- create the uv venv with base dependencies: `uv sync`
- optionally install extras: `uv sync --extra report --extra torch --extra tabpfn`
- run unittests `uv run poe test-unit`
- run acceptance tests `uv run poe test-acceptance`

### Understanding Extras vs Dependency Groups

Mother uses two different mechanisms for optional dependencies:

#### Extras (User-Facing)

Extras are pip-installable optional dependencies that end users can install:

| Extra | Description | Installation |
|-------|-------------|--------------|
| `all` | All optional features | `uv sync --extra all` |
| `report` | Visualization and reporting tools | `uv sync --extra report` |
| `rna` | RNA sequence analysis | `uv sync --extra rna` |
| `torch` | PyTorch neural network support | `uv sync --extra torch` |
| `tabpfn` | TabPFN model support | `uv sync --extra tabpfn` |
| `clustering` | Chemical compound clustering | `uv sync --extra clustering` |

**Install multiple extras:**
```bash
uv sync --extra report --extra torch --extra tabpfn
```

#### Dependency Groups (Development Only)

Dependency groups are development dependencies that are NOT published with the package:

| Group | Description | Installation |
|-------|-------------|--------------|
| `examples` | Dependencies for example notebooks (polaris-lib, shap) | `uv sync --group examples` |
| `docs` | Documentation building tools (mkdocs, etc.) | `uv sync --group docs` |
| `test_duration` | Test performance analysis (pytest-html, pytest-xdist) | `uv sync --group test_duration` |

**Install multiple groups:**
```bash
uv sync --group examples --group docs --group test_duration
```

**Complete developer setup:**
```bash
# Install base dependencies + all extras + all dev groups
uv sync --all-extras --all-groups
```

## Developing a new feature

- create a new feature branch `git checkout -b feature`
- use pre-commit hooks
- make changes to mother and add tests accordingly
- run unittests `uv run poe test-unit`
- run acceptance tests `uv run poe test-acceptance`
- `git push` changes to a separate branch describing the feature
- Do not forget to use semantic versioning commands in your commit message (version is bumped by semantic-versioning)
- create a pull request on GitHub

## Publishing a new package version to Artifactory

A new package will be pushed via the mother CI/CD pipeline. No manual steps required.


## Codespaces

Skip local install and run mother from codespaces; all compute runs on github. Automate install using docker. **This is a work in progress. User docs come after this is tested.**

!!! warning
     Use the directory `test/data_secret` for local files. The `data_secret` directory is for files which should be NOT kept in github. This includes large and/or secret data. This directory is gitignored so will not be pushed to our bayer github.

### Run Tests

Confirm mother is working by running some tests.

- Go to "Testing" tab, wait until loaded and select tests to run
- There are three types of test to run. Each github commit triggers running all these tests. See github actions for more info.
  - Acceptance are long running tests. Run these remotely
  - Unit test we can run locally, as they are quick to run.

## Using a virtual environment

Key for propper project setup is a virtual environment.
The recommended way is to use `uv`, which manages virtual environments automatically:

```bash
uv sync
```

Alternatively, you can manually create a virtual environment:

- Create the virtual environment: `python -m virtualenv .venv`
- Initiate the virtual environment: `source .venv/bin/activate`
- Then install the base requirements: `pip install -r requirements.dev.txt`

## Using uv

If you want to use [uv](https://docs.astral.sh/uv/) as your dependency manager use the provided `pyproject.toml` file.
`uv` automatically manages virtual environments and lockfiles for you.
Consistency of the requirements files is ensured if the pre-commit hooks are enabled.

## Development

To properly develop please start by enabling the [pre-commit hooks](#pre-commit-hooks) if not already done using `pre-commit install`.

### Pull Requests

Merging your changes to the main branch is just allowed via PRs. For that please provide clean commit messages for your changes. When merging the PR please use `Squash and merge` and ensure linear history. The changelog is automatically created via github actions and takes the changelog from the PR message. See [here](https://github.com/mikepenz/release-changelog-builder-action).

## Testing

`pytest` is the prefered testing package on the one included in the `requirements.dev.txt`. A github actions workflow runs tests before merges, but you should run your own tests before pushing to avoid the github actions failing.

### Estimating Test Durations

Some of our unit tests are pretty slow. But which one, we do not exactly now. There is the possibility to use `--durations` to get durations for each test.
However, this does not lead to consistent timings.

I installed the modules `uv add --group test_duration pytest-xdist[psutil] pytest-html` to see how this would improve timings (according to ChatGPT).

```bash
pytest test --durations=0 --html=pytest_report.html --self-contained-html -n auto
```

```python title="Parsing Files"
import json

with open("durations.json", "r") as json_file:
    data = json.load(json_file)

test_durations = {item["nodeid"]: item["duration"] for item in data}
for test_name, duration in test_durations.items():
    print(f"Test {test_name} took {duration:.2f} seconds")

```

## Credential Handling

You should **NEVER** add your credentials into any script. Please use the `.env`file.

## Linting

`pylint` is the prefered testing package on the one included in the `requirements.dev.txt`. The github actions workflow is set to fail if the score is below 8, so to make sure your action will not fail run `pylint` before you push.

## Formatting

Formating is done for you by `ruff` with github actions automatically and as a pre-commit hook.

## SonarQube

SonarQube is a great way to analyze your code and make sure its set up properly.
To add your project you can simply link it to your github repository [here](https://sonar.cloud.bayer.com/). Coverage of your project is automatically generated. To learn how python test coverage is handled in sonar qube have a look at the docs [sonarqube coverage](
<https://docs.sonarqube.org/9.7/analyzing-source-code/test-coverage/python-test-coverage/>).

## Pre-Commit Hooks

pre-commit hooks can be enabled via `pre-commit install`. Please ensure to properly activate the python environment. If you use uv use `uv run poe install-hook`. These hooks are simple scripts that run before a commit is performed to ensure consistency of the added code and checks for credentials as well. See `.pre-commit-config.yaml` for provided hooks or add your own if desired.

## Python Semantic Release

This project uses [Python Semantic Release (PSR)](https://python-semantic-release.readthedocs.io/) to automate version management and changelog generation. PSR analyzes commit messages to automatically determine the next version number and generate release notes.

### Conventional Commits

To work effectively with semantic release, all commit messages must follow the [Conventional Commits](https://conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Commit Types

The project recognizes the following commit types:

- **feat**: A new feature (triggers a **minor** version bump)
- **fix**: A bug fix (triggers a **patch** version bump)
- **perf**: A code change that improves performance (triggers a **patch** version bump)
- **docs**: Documentation only changes (triggers a **patch** version bump)
- **style**: Changes that do not affect the meaning of the code (triggers a **patch** version bump)
- **refactor**: A code change that neither fixes a bug nor adds a feature (triggers a **patch** version bump)
- **test**: Adding missing tests or correcting existing tests (triggers a **patch** version bump)
- **build**: Changes that affect the build system or external dependencies (triggers a **patch** version bump)
- **ci**: Changes to CI configuration files and scripts (triggers a **patch** version bump)
- **chore**: Other changes that don't modify src or test files (triggers a **patch** version bump)

#### Breaking Changes

To trigger a **major** version bump, add `BREAKING CHANGE:` in the commit body or append `!` to the type:

```bash
feat!: remove deprecated API endpoint

BREAKING CHANGE: The /api/v1/old-endpoint has been removed. Use /api/v2/new-endpoint instead.
```

#### Examples

```bash
# Minor version bump
feat: add new clustering algorithm support

# Patch version bump
fix: resolve memory leak in preprocessing pipeline
docs: update installation instructions
refactor: optimize feature selection performance

# Major version bump
feat!: redesign configuration system

BREAKING CHANGE: Configuration file format has changed from YAML to JSON.
```

### Configuration

The semantic release configuration is defined in `pyproject.toml`:

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
build_command = "uv build"
upload_to_release = true

[tool.semantic_release.changelog]
mode = "update"
insertion_flag = "..\n    All versions below are listed in reverse chronological order"
changelog_file = "mkdocs/docs/Changelog.md"
output_format = "md"
```

### Automated Release Process

Releases are automatically triggered when commits are pushed to the `main` branch:

1. **Analysis**: PSR analyzes commit messages since the last release
2. **Version Calculation**: Determines the next version based on commit types
3. **Changelog Generation**: Updates `mkdocs/docs/Changelog.md` with new changes
4. **Version Bump**: Updates version in `pyproject.toml`
5. **Build**: Creates distribution packages using uv
6. **Release**: Creates a GitHub release with built artifacts
7. **Publish**: Uploads packages to the configured repository

### Manual Release (if needed)

While releases are automated, you can manually trigger a release:

```bash
# Dry run to see what would happen
uv run semantic-release version --no-push

# Generate changelog only
uv run semantic-release changelog

# Print current version
uv run semantic-release version --print
```

### Best Practices

1. **Consistent Commit Messages**: Always follow conventional commit format
2. **Atomic Commits**: Make each commit focused on a single change
3. **Descriptive Messages**: Write clear, concise commit descriptions
4. **Scope Usage**: Use scopes to indicate which part of the codebase is affected:
   ```bash
   feat(preprocessing): add new normalization method
   fix(ml): correct cross-validation scoring
   docs(api): update docstring examples
   ```
5. **Merge Strategy**: Use "Squash and merge" for PRs to maintain clean commit history
6. **Release Notes**: The changelog is automatically generated from commit messages, so write them for your users

### Troubleshooting

#### No Release Generated
- Ensure commits follow conventional commit format
- Check that commit types are recognized (see configuration above)
- Verify no commits are excluded by `exclude_commit_patterns`

#### Version Not Updated
- Check that `version_toml` path is correct in configuration
- Ensure uv is properly configured
- Verify GitHub Actions has necessary permissions

#### Failed Release
- Check GitHub Actions logs for detailed error messages
- Verify all required secrets and tokens are configured
- Ensure build command succeeds locally

For more details, consult the [Python Semantic Release documentation](https://python-semantic-release.readthedocs.io/).
