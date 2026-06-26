"""
Command-line entry points for the Xed development workflow.

Installed automatically by ``uv sync`` / ``pip install -e .[dev]``.

uv users:
    uv run xed-lint          # ruff check
    uv run xed-lint-fix      # ruff check --fix
    uv run xed-fmt-check     # ruff format --check
    uv run xed-fmt           # ruff format
    uv run xed-check         # lint + fmt-check + sync-check
    uv run xed-fix           # lint-fix + fmt
    uv run xed-convert       # .py → .ipynb
    uv run xed-sync-check    # verify .ipynb matches .py

hatch users (equivalent):
    hatch run lint / fmt / check / fix / convert / sync-check
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Repository root — two levels up from xed/_cli.py
_ROOT = Path(__file__).parent.parent


def _run(*cmd: str) -> int:
    """Run *cmd* in the repo root and return the exit code."""
    return subprocess.run(list(cmd), cwd=_ROOT).returncode


def _scripts_py() -> str:
    return str(_ROOT / "scripts" / "convert_notebooks.py")


# ── Linting ───────────────────────────────────────────────────────────────────


def lint() -> None:
    """ruff check — PEP 8 and pyflakes violations."""
    sys.exit(_run("ruff", "check", "."))


def lint_fix() -> None:
    """ruff check --fix — auto-fix all fixable violations."""
    sys.exit(_run("ruff", "check", "--fix", "."))


# ── Formatting ────────────────────────────────────────────────────────────────


def fmt_check() -> None:
    """ruff format --check — verify formatting without modifying files."""
    sys.exit(_run("ruff", "format", "--check", "."))


def fmt() -> None:
    """ruff format — apply automatic formatting."""
    sys.exit(_run("ruff", "format", "."))


# ── Combined gates ────────────────────────────────────────────────────────────


def check() -> None:
    """Run lint + fmt-check + sync-check.  Exits 1 on the first failure."""
    for step in (
        ("ruff", "check", "."),
        ("ruff", "format", "--check", "."),
        (sys.executable, _scripts_py(), "--check"),
    ):
        rc = _run(*step)
        if rc != 0:
            sys.exit(rc)
    sys.exit(0)


def fix() -> None:
    """Apply lint-fix + fmt in one step."""
    rc = _run("ruff", "check", "--fix", ".")
    rc |= _run("ruff", "format", ".")
    sys.exit(rc)


# ── Notebook conversion ───────────────────────────────────────────────────────


def convert() -> None:
    """Convert all jupytext .py sources to paired .ipynb files."""
    sys.exit(_run(sys.executable, _scripts_py()))


def sync_check() -> None:
    """Verify every .ipynb is in sync with its .py source.  Exits 1 if not."""
    sys.exit(_run(sys.executable, _scripts_py(), "--check"))
