"""Workdir rooting shared by the file tools.

One rule, one implementation: a path that resolves outside the workdir is
refused, never clamped. Directories that are noise for a coding agent
(``.git``, caches, vendored trees) are skipped by the walking tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

# Directories the walking tools never descend into.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".venv", "venv", "node_modules", ".idea", ".eggs",
})


def make_resolver(workdir: Path) -> tuple[Path, Callable[[str], Path]]:
    """Return ``(root, resolve)``; ``resolve`` maps a relative path into the root."""
    root = Path(workdir).resolve()

    def resolve(rel: str) -> Path:
        p = (root / rel).resolve()
        if p != root and root not in p.parents:
            raise PermissionError(f"path escapes workdir: {rel}")
        return p

    return root, resolve


def rel_to(root: Path, p: Path) -> str:
    """Path as the model should see it: relative to the workdir, forward slashes."""
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def walk_files(start: Path, root: Path):
    """Yield files under ``start`` in deterministic order, skipping noise dirs."""
    if start.is_file():
        yield start
        return
    stack = [start]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir())
        except (PermissionError, OSError):
            continue
        dirs = []
        for child in children:
            if child.is_symlink():
                continue
            if child.is_dir():
                if child.name in SKIP_DIRS:
                    continue
                dirs.append(child)
            elif child.is_file():
                yield child
        stack.extend(reversed(dirs))
