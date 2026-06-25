#!/usr/bin/env python3
"""
Convert all jupytext .py sources to paired .ipynb files.

Usage:
    python scripts/convert_notebooks.py          # convert and write .ipynb
    python scripts/convert_notebooks.py --check  # verify .ipynb is in sync, exit 1 if not

Run via hatch:
    hatch run convert
    hatch run sync-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import jupytext
except ImportError:
    print("jupytext is not installed.  Run:  pip install jupytext>=1.19", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Discover notebook sources
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "data_wrangling/solutions",
    "datascience_starter_course",
    "xed",
    "scripts",
}
EXCLUDE_FILES = {
    "data_wrangling/notebook_corrected.py",
    "data_wrangling/use_case.py",
    "visualizations/plot_2d_separator.py",
    "visualizations/plot_interactive_forest.py",
}


def is_jupytext_source(path: Path) -> bool:
    """Return True if *path* is a jupytext percent-format notebook source."""
    if path.suffix != ".py":
        return False
    rel = path.relative_to(ROOT).as_posix()
    if any(rel.startswith(d) for d in EXCLUDE_DIRS):
        return False
    if rel in EXCLUDE_FILES:
        return False
    # A jupytext source has a percent-format header in the first 20 lines
    try:
        head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]
    except OSError:
        return False
    return any("format_name: percent" in line for line in head)


def discover() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.py") if is_jupytext_source(p))


# ---------------------------------------------------------------------------
# Cell-source comparison (ignores outputs and execution counts)
# ---------------------------------------------------------------------------


def cell_sources(nb) -> list[tuple[str, str]]:
    return [
        ("".join(c["source"]), c["cell_type"])
        for c in nb.cells
        if "".join(c.get("source", "")).strip()
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: exit 1 if any .ipynb is out of sync with its .py source.",
    )
    args = parser.parse_args()

    sources = discover()
    if not sources:
        print("No jupytext .py sources found.")
        return

    failures: list[str] = []
    converted = 0

    for py_path in sources:
        ipynb_path = py_path.with_suffix(".ipynb")
        nb_from_py = jupytext.read(str(py_path))

        if args.check:
            if not ipynb_path.exists():
                failures.append(f"MISSING  {ipynb_path.relative_to(ROOT)}")
                continue
            nb_from_ipynb = jupytext.read(str(ipynb_path))
            if cell_sources(nb_from_py) != cell_sources(nb_from_ipynb):
                failures.append(f"OUT OF SYNC  {py_path.relative_to(ROOT)}")
        else:
            jupytext.write(nb_from_py, str(ipynb_path))
            print(f"  converted  {py_path.relative_to(ROOT)}")
            converted += 1

    if args.check:
        if failures:
            print("Notebook sync failures:", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
            print(
                "\nRun  hatch run convert  (or  python scripts/convert_notebooks.py)"
                "  to regenerate the .ipynb files.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(f"All {len(sources)} .py/.ipynb pairs are in sync.")
    else:
        print(f"\nConverted {converted} notebooks.")


if __name__ == "__main__":
    main()
