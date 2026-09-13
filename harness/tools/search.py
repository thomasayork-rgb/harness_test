"""Search tools: regex across file contents, glob over paths. Rooted to the workdir."""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from ..registry import ToolRegistry
from .paths import make_resolver, rel_to, walk_files

MAX_FILE_BYTES = 2_000_000        # bigger files are skipped, not read
BINARY_SNIFF_BYTES = 8192         # a NUL byte in here means binary
MAX_LINE_CHARS = 300              # matched lines are trimmed to this in the result


def _matches_glob(rel: str, pattern: str) -> bool:
    """``*.py`` matches the file name, ``src/**/*.py`` matches the whole path."""
    if "/" in pattern:
        return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern.replace("**/", ""))
    return fnmatch.fnmatch(Path(rel).name, pattern)


def _read_text(path: Path) -> str | None:
    """Text of a file, or None if it is binary or too big to be worth reading."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        with path.open("rb") as f:
            head = f.read(BINARY_SNIFF_BYTES)
            if b"\x00" in head:
                return None
            rest = f.read()
    except (OSError, ValueError):
        return None
    return (head + rest).decode("utf-8", errors="replace")


def register_search_tools(registry: ToolRegistry, workdir: Path) -> None:
    root, resolve = make_resolver(workdir)

    @registry.tool(
        "fs_search",
        "Search file contents for a regex under the workdir; returns path, line number and line.\n"
        "Skips binary files, files over 2 MB, .git and cache/vendor directories.",
        {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regular expression"},
            "path": {"type": "string", "description": "file or directory to search, relative to the workdir (default '.')"},
            "glob": {"type": "string", "description": "name filter like '*.py', or a path filter like 'src/**/*.py' (any pattern containing '/')"},
            "ignore_case": {"type": "boolean"},
            "max_results": {"type": "integer", "description": "default 100"},
        }, "required": ["pattern"], "additionalProperties": False},
    )
    def fs_search(pattern: str, path: str = ".", glob: str | None = None,
                  ignore_case: bool = False, max_results: int = 100) -> dict:
        start = resolve(path)
        if not start.exists():
            raise FileNotFoundError(f"no such path: {path}")
        if max_results < 1:
            raise ValueError("max_results must be >= 1")
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ValueError(f"invalid regex {pattern!r}: {e}") from None

        matches: list[dict] = []
        scanned = skipped = 0
        truncated = False
        for file in walk_files(start, root):
            rel = rel_to(root, file)
            if glob and not _matches_glob(rel, glob):
                continue
            text = _read_text(file)
            if text is None:
                skipped += 1
                continue
            scanned += 1
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    matches.append({"path": rel, "line": i, "text": line.strip()[:MAX_LINE_CHARS]})
                    if len(matches) >= max_results:
                        truncated = True
                        break
            if truncated:
                break
        return {"pattern": pattern, "path": path, "glob": glob, "files_scanned": scanned,
                "files_skipped": skipped, "match_count": len(matches), "truncated": truncated,
                "matches": matches}

    @registry.tool(
        "fs_glob",
        "Find files by glob pattern under the workdir, e.g. '**/*.py' or 'src/*.json'.\n"
        "Returns paths relative to the workdir, sorted; skips .git and cache/vendor directories.",
        {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "'*.py' matches the file name at any depth; a pattern with '/' matches the path relative to 'path'"},
            "path": {"type": "string", "description": "directory to search from, relative to the workdir (default '.')"},
            "max_results": {"type": "integer", "description": "default 200"},
        }, "required": ["pattern"], "additionalProperties": False},
    )
    def fs_glob(pattern: str, path: str = ".", max_results: int = 200) -> dict:
        start = resolve(path)
        if not start.exists():
            raise FileNotFoundError(f"no such path: {path}")
        if not start.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        if max_results < 1:
            raise ValueError("max_results must be >= 1")
        found: list[dict] = []
        for file in walk_files(start, root):
            rel_start = rel_to(start, file)
            if not _matches_glob(rel_start, pattern):
                continue
            found.append({"path": rel_to(root, file), "bytes": file.stat().st_size})
        found.sort(key=lambda m: m["path"])
        return {"pattern": pattern, "path": path, "count": len(found),
                "truncated": len(found) > max_results, "files": found[:max_results]}
