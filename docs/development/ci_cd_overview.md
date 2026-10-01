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
| `test-unit` | Parallel unit tests, then `serial`-marked tests with `-n 0` | Yes (as `coverage`) |
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

Defined in `.github/workflows/workflow.yml`. Triggers: pull requests to `main`,
any `push`, and manual `workflow_dispatch`. Jobs run in dependency order and stop
the pipeline on the first failure:

1. **style** — `check-style` + `check-lint`.
2. **test** — matrix on Python 3.11 / 3.12 / 3.13 / 3.14: full `coverage` run
   (with `rna`, `report`, `tabpfn`, `tabicl`, `clustering` extras), then
   `dist-test` against a no-extras install. Runs only on `main` or PRs.
3. **test-slow** — `slow` suite on Python 3.14.
4. **release-preflight** — see [below](#release-preflight). Read-only token;
   uploads the report artifact and fails on violations.
5. **comment-preflight** — posts/updates a sticky PR comment with the preflight
   report. Isolated write-token job that never runs PR code; same-repo PRs only.
6. **release** — on `main` only: `python-semantic-release` computes the version,
   updates the changelog, builds and checks the wheel. Skipped with
   `[skip release]` in the commit message.
7. **test-pypi-publish** — publishes the dev build to Test PyPI on push to `main`.
8. **publish-pypi** — publishes to PyPI when a release was actually cut.
9. **build-docs / deploy-docs** — build the site; deploy to GitHub Pages on `main`.

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
