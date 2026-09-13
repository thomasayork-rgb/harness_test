"""Exact-string editing, rooted to the workdir.

``fs_edit`` refuses anything ambiguous: the old string must occur exactly once
unless ``replace_all`` is set. The result carries a unified diff of what
changed so the model can verify the edit without re-reading the file.
"""
from __future__ import annotations

import difflib
from pathlib import Path

from ..registry import ToolRegistry
from .paths import make_resolver

MAX_DIFF_LINES = 60


def _diff(path: str, before: str, after: str, context: int = 2) -> str:
    lines = list(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=context))
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"... [diff truncated at {MAX_DIFF_LINES} lines]"]
    return "\n".join(lines)


def register_edit_tools(registry: ToolRegistry, workdir: Path) -> None:
    _root, resolve = make_resolver(workdir)

    @registry.tool(
        "fs_edit",
        "Replace an exact string in a file; the old string must occur exactly once unless replace_all is set.\n"
        "Returns the number of replacements and a unified diff of the change.",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "file to edit, relative to the workdir"},
            "old": {"type": "string", "description": "exact text to replace, including indentation"},
            "new": {"type": "string", "description": "replacement text; empty string deletes"},
            "replace_all": {"type": "boolean", "description": "replace every occurrence instead of requiring exactly one"},
        }, "required": ["path", "old", "new"], "additionalProperties": False},
    )
    def fs_edit(path: str, old: str, new: str, replace_all: bool = False) -> dict:
        p = resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"no such file: {path}")
        if not p.is_file():
            raise IsADirectoryError(f"not a file: {path}")
        if old == "":
            raise ValueError("'old' must not be empty; use fs_write to create or overwrite a file")
        if old == new:
            raise ValueError("'old' and 'new' are identical; nothing to do")
        raw = p.read_bytes()
        if b"\x00" in raw:
            raise ValueError(f"{path} looks binary (contains NUL bytes); refusing to edit")
        try:
            before = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"{path} is not valid UTF-8 text ({e.reason}); refusing to edit") from None

        count = before.count(old)
        if count == 0:
            raise ValueError(
                f"no occurrence of that exact string in {path}; "
                "check whitespace and indentation, or read the file again")
        if count > 1 and not replace_all:
            raise ValueError(
                f"{count} occurrences of that string in {path}; "
                "include more surrounding context to make it unique, or pass replace_all=true")

        after = before.replace(old, new) if replace_all else before.replace(old, new, 1)
        p.write_text(after, encoding="utf-8")
        return {"path": path, "replacements": count if replace_all else 1,
                "bytes": len(after.encode("utf-8")), "diff": _diff(path, before, after)}
