#!/usr/bin/env python3
"""Release preflight checks that complement semantic-release.

This script does not write changelog entries. It validates the commit range since
last tag and produces a markdown report to support manual release review steps.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CONVENTIONAL_RE = re.compile(r"^(?P<type>[a-z]+)(\([^)]+\))?(!)?:\s+.+")
ISSUE_RE = re.compile(r"#(?P<num>\d+)")

ALLOWED_TYPES = {
    "feat",
    "fix",
    "perf",
    "docs",
    "style",
    "refactor",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
}


@dataclass
class Commit:
    sha: str
    subject: str


def run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def get_last_tag() -> str:
    return run_git(["describe", "--tags", "--abbrev=0"])


def get_commits(base_tag: str) -> list[Commit]:
    raw = run_git(["log", f"{base_tag}..HEAD", "--pretty=format:%H%x09%s"])
    if not raw:
        return []
    commits: list[Commit] = []
    for line in raw.splitlines():
        sha, subject = line.split("\t", 1)
        commits.append(Commit(sha=sha, subject=subject))
    return commits


def classify(commit: Commit) -> str | None:
    match = CONVENTIONAL_RE.match(commit.subject)
    if not match:
        return None
    commit_type = match.group("type")
    if commit_type not in ALLOWED_TYPES:
        return None
    return commit_type


def build_report(base_tag: str, commits: list[Commit]) -> tuple[str, int]:
    categorized: dict[str, list[Commit]] = defaultdict(list)
    unknown: list[Commit] = []
    missing_issue_refs: list[Commit] = []

    for commit in commits:
        commit_type = classify(commit)
        if commit_type is None:
            unknown.append(commit)
        else:
            categorized[commit_type].append(commit)

        if not ISSUE_RE.search(commit.subject):
            missing_issue_refs.append(commit)

    lines: list[str] = []
    lines.append("# Release Preflight Report")
    lines.append("")
    lines.append(f"Base tag: {base_tag}")
    lines.append(f"Commit count: {len(commits)}")
    lines.append("")

    if not commits:
        lines.append("No commits found since the last tag. Release is likely unnecessary.")
        return "\n".join(lines), 0

    lines.append("## Conventional commit summary")
    lines.append("")
    for key in sorted(categorized.keys()):
        lines.append(f"- {key}: {len(categorized[key])}")
    lines.append("")

    if unknown:
        lines.append("## Non-conventional or unsupported commits")
        lines.append("")
        for commit in unknown:
            lines.append(f"- {commit.sha[:8]} {commit.subject}")
        lines.append("")

    if missing_issue_refs:
        lines.append("## Commits missing issue or PR reference")
        lines.append("")
        for commit in missing_issue_refs:
            lines.append(f"- {commit.sha[:8]} {commit.subject}")
        lines.append("")

    lines.append("## Suggested manual follow-up")
    lines.append("")
    lines.append("- Run the update-docs skill before release for issue and milestone reconciliation.")
    lines.append("- Confirm docs updates for release-specific changes.")

    return "\n".join(lines), len(unknown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Release preflight helper")
    parser.add_argument("--base-tag", default="auto", help="Tag to compare against, or 'auto'")
    parser.add_argument("--output", default="release-preflight-report.md", help="Output markdown file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if non-conventional commits are found",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        base_tag = get_last_tag() if args.base_tag == "auto" else args.base_tag
        commits = get_commits(base_tag)
        report, unknown_count = build_report(base_tag, commits)
        Path(args.output).write_text(report + "\n", encoding="utf-8")

        print(report)

        if args.strict and unknown_count > 0:
            print("\nPreflight failed: non-conventional commits found.", file=sys.stderr)
            return 2
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
