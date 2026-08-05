#!/usr/bin/env python3
"""Validate executable Python code fences in docs markdown files.

This script scans mkdocs/docs/**/*.md for fenced blocks with the `python` language,
executes them sequentially per file, and fails if any block raises an exception.

Execution happens in isolated namespaces per file so examples can build on previous
blocks in the same document but not across documents.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import textwrap
import traceback


def _iter_docs_markdown(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(root.rglob("*.md"))


def _extract_python_blocks(markdown_text: str) -> list[str]:
    # Supports fences like ```python and ```python title="..."
    pattern = r"```python(?:[^\n]*)\n(.*?)\n```"
    return re.findall(pattern, markdown_text, flags=re.S)


def check_docs_python_fences(
    docs_root: pathlib.Path,
    verbose: bool = False,
    skip_missing_extras: bool = True,
) -> int:
    failures: list[tuple[pathlib.Path, int, str]] = []
    executed_blocks = 0
    skipped_blocks = 0

    for file_path in _iter_docs_markdown(docs_root):
        text = file_path.read_text(encoding="utf-8")
        blocks = _extract_python_blocks(text)
        if not blocks:
            continue

        namespace: dict[str, object] = {}
        if verbose:
            rel = file_path.relative_to(docs_root.parent)
            print(f"\n=== {rel}: {len(blocks)} python block(s) ===")

        for idx, block in enumerate(blocks, start=1):
            code = textwrap.dedent(block)
            try:
                exec(code, namespace, namespace)
                executed_blocks += 1
                if verbose:
                    print(f"  block {idx}: OK")
            except Exception as exc:
                if skip_missing_extras and exc.__class__.__name__ == "ExtrasDependencyImportError":
                    skipped_blocks += 1
                    if verbose:
                        print(f"  block {idx}: SKIP (missing optional extra)")
                    continue
                tb = traceback.format_exc()
                failures.append((file_path, idx, tb))
                if verbose:
                    print(f"  block {idx}: FAIL")

    if failures:
        print(
            "Checked docs python fences: "
            f"{executed_blocks} block(s) executed, "
            f"{skipped_blocks} block(s) skipped, "
            f"{len(failures)} failure(s)."
        )
        for file_path, idx, tb in failures:
            rel = file_path.relative_to(docs_root.parent)
            print(f"\n--- {rel} block {idx} ---")
            print(tb)
        return 1

    print(
        "Checked docs python fences: "
        f"{executed_blocks} block(s) executed, "
        f"{skipped_blocks} block(s) skipped, 0 failures."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Python fenced code blocks in docs markdown.")
    parser.add_argument(
        "--docs-root",
        default="mkdocs/docs",
        help="Path to docs root containing markdown files (default: mkdocs/docs)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file and per-block status",
    )
    parser.add_argument(
        "--fail-on-missing-extras",
        action="store_true",
        help="Fail instead of skipping blocks that require optional extras",
    )
    args = parser.parse_args()

    docs_root = pathlib.Path(args.docs_root).resolve()
    if not docs_root.exists():
        print(f"Docs root does not exist: {docs_root}", file=sys.stderr)
        return 2

    return check_docs_python_fences(
        docs_root=docs_root,
        verbose=args.verbose,
        skip_missing_extras=not args.fail_on_missing_extras,
    )


if __name__ == "__main__":
    raise SystemExit(main())
