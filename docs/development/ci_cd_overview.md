# CI/CD, Checks & Release Overview

A quick agenda of everything that runs on your code — locally before you push, in
CI on every pull request, and during an automated release. Use this as the
onboarding map; each section links to the deeper reference.

## Agenda

1. [Local checks](#local-checks) — what to run before you push
2. [Pre-commit hooks](#pre-commit-hooks) — what runs automatically on commit
3. [CI/CD pipeline](#cicd-pipeline) — the GitHub Actions jobs, in order
4. [Code review & PR approval](#code-review-pr-approval) — how PRs get merged
5. [Release preflight](#release-preflight) — the conventional-commit gate
6. [Automated release](#automated-release) — semantic-release & publishing
7. [Skills](#skills) — the `update-docs` release helper

## Local checks

Run these with `uv run poe <task>` before pushing so CI does not fail on you.
Full setup lives in [General Development](dev.md).

| Task | What it does | CI-enforced |
|------|--------------|-------------|
| `check-style` | `check-sort-imports` (isort, black profile) + `check-format` (ruff format) | Yes |
| `style` | Apply the style fixes above | — |
| `check-lint` | ruff static analysis (`E`+`F`) on `src` and `test` | Yes |
| `lint` | ruff with autofix | — |
| `check-types` | mypy on `src` (strict) | No — currently commented out in CI |
| `check-static-analysis` | `check-lint` + `check-types` | Lint only |
| `test-unit` | Parallel unit tests, then `serial`-marked tests with `-n 0` | Partly (`coverage` excludes serial tests) |
| `test-slow` | `slow`-marked tests | Yes |
| `coverage` | Unit tests + XML coverage / xunit report | Yes |
| `docs` | Build the docs site (zensical) | Yes |
| `docs-python-fences` | Execute python fenced blocks in `docs/*.md` | Yes (pre-commit) |
| `dist-test` | Build the wheel and test it in a clean, no-extras env | Yes |

!!! note
    `check-types` (mypy) is not yet a hard CI gate — it is commented out in
    `.github/workflows/workflow.yml` pending type-annotation cleanup. Run it
    locally anyway to avoid drift.

## Pre-commit hooks

Enable once with `uv run poe install-hook`; run against everything with
`uv run poe check-hook`. Configured in `.pre-commit-config.yaml`:

- **uv**: `uv-lock` keeps the lockfile in sync.
- **Hygiene** (pre-commit-hooks): valid-AST, docstring-first, case-conflict,
  merge-conflict markers, TOML parse, **private-key detection**, end-of-file
  fixer, byte-order-marker, mixed line endings, trailing whitespace.
- **Local**: ruff linter, isort, ruff formatter, and the docs python-fence check.
- **Notebooks** (nbQA): ruff check/format, pyupgrade, isort on `.ipynb` files.

Hooks skip `examples/` and `test/`.

## CI/CD pipeline

All CI and release jobs are defined in `.github/workflows/workflow.yml`. The
separate `release.yml` workflow has been removed. CI runs on pull requests to
`main`, pushes to `main`, and manual `workflow_dispatch`. A manual dispatch runs
checks but does **not** trigger a release or PyPI publication.

The release path runs in this dependency order; a failure blocks its dependent
jobs, not independent branches of the workflow:

1. **style**: `check-style` + `check-lint`.
2. **release-preflight**: after `style`, runs the [preflight audit](#release-preflight)
   with a read-only token, uploads the report artifact, and fails on violations.
3. **test**: after preflight, runs the Python 3.11 / 3.12 / 3.13 / 3.14 matrix
   with `rna`, `report`, `tabpfn`, `tabicl`, and `clustering` extras. Each job runs
   `coverage` (non-slow, non-serial tests), then `dist-test` after a no-extras sync.
4. **test-slow**: after the test matrix passes, runs the slow suite on Python 3.14.
5. **release**: requires `test-slow` and preflight, and runs only on pushes to
   `main` that are not semantic-release's own release commits. It runs PSR on the
   host, updates the version and changelog, regenerates and stages `uv.lock`,
   pushes the release commit and tag, and creates the GitHub release. If a new
   tag was created, it builds distributions with `uv build`, checks them with
   `twine check --strict`, and uploads the `Packages` Actions artifact.
6. **publish-pypi**: after `release`, downloads `Packages` and publishes using
   PyPI Trusted Publishing (OIDC) in the `pypi` environment. It runs only when
   `release` reports a new tag and the repository is `Bayer-Group/MotherML`.

Two other branches run separately from the release path:

- **comment-preflight**: after preflight, posts or updates a sticky report comment
  on same-repository PRs, even when preflight fails. It has an isolated write
  token and never executes PR code. Fork PRs do not receive this comment.
- **build-docs / deploy-docs**: the docs build has no test or release dependency.
  Deployment depends only on a successful docs build and a non-PR run on `main`,
  including a manual dispatch on `main`.

Superseded PR runs are cancelled; pushes to `main` are not. Releases are serialized
with the `release` concurrency group. The generated release commit contains
`[skip ci]`, and the release job also checks for PSR's generated-commit message
to prevent a release loop.

## Code review & PR approval

Never push to `main` — every change lands via PR, squash-merged with linear
history. See [PR Approval Process](pr_approval_guidelines.md) and
`CONTRIBUTING.md`.

- At least one reviewer approval; maintainers are listed in `.github/CODEOWNERS`.
- All CI checks above must be green.
- Tests accompany new behavior; maintaining/increasing coverage is encouraged.
- Docs updated for non-trivial changes.
- Reference issues in the PR/commit body with `Fixes #N`.

## Release preflight

`scripts/release_preflight.py` complements semantic-release — it does **not**
write the changelog. It compares `main..HEAD` and produces
`release-preflight-report.md` with:

- a conventional-commit summary by type,
- **non-conventional or unsupported** commit subjects,
- commits **missing an issue/PR reference** (`#N`).

Allowed types: `feat`, `fix`, `perf`, `docs`, `style`, `refactor`, `test`,
`build`, `ci`, `chore`, `revert`. With `--strict` (how CI runs it), the job
**fails** when any non-conventional commit is present.

Fix a failure by rewriting/squashing commit subjects into Conventional Commit
format and re-pushing; the sticky PR comment updates on the next run.

## Automated release

Versioning and the changelog are automated by `python-semantic-release`; never
bump the version by hand. Commit type drives the bump:

- `feat` → **minor**, `fix` / `perf` / others → **patch**,
- `BREAKING CHANGE:` (or `type!:`) → **major**.

Full details, configuration, and troubleshooting are in the
[Python Semantic Release section of General Development](dev.md#python-semantic-release).

## Skills

The `update-docs` skill (`.github/skills/update-docs/SKILL.md`) runs the manual
side of a release that preflight cannot: it builds/verifies the changelog against
merged PRs, and reconciles GitHub issues and milestones. Preflight is the
automated reminder; `update-docs` is the human-in-the-loop reconciliation — run
it before cutting a release.
