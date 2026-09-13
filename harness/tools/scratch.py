"""A scratch pad rooted in the run directory.

Tool results are evictable: past the context budget the oldest are replaced
with a pointer to their artifact, and a long one is truncated even before that.
Anything the agent wants to still have in twenty steps has to live somewhere
the budget does not reach. ``scratch_write`` is that somewhere - a small
key/value note store under ``runs/<run_id>/scratch/``, written by the agent,
readable again with ``scratch_read`` whatever happened to the conversation.

Notes are per run, not per workdir: they are working memory, not output. A note
the agent wants to keep belongs in the workdir, via ``fs_write``.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..registry import ToolRegistry

VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_NOTE_BYTES = 200_000


def register_scratch_tools(registry: ToolRegistry, run_dir: Path) -> None:
    """Register the scratch pad for one run. The directory is created on write."""
    root = Path(run_dir) / "scratch"

    def resolve(name: str) -> Path:
        # A flat namespace: no directories, no traversal, nothing to get wrong.
        if not isinstance(name, str) or not VALID_NAME.match(name):
            raise ValueError(
                f"invalid note name {name!r}: letters, digits, dot, dash and underscore only, "
                "up to 64 characters, no directories")
        return root / name

    @registry.tool(
        "scratch_write",
        "Save a note for later under a short name; notes survive context eviction.\n"
        "Use it for facts you will need again: paths found, values read, decisions made.",
        {"type": "object", "properties": {
            "name": {"type": "string", "description": "short note name, e.g. 'findings'"},
            "content": {"type": "string"},
            "append": {"type": "boolean", "description": "add to the note instead of replacing it"},
        }, "required": ["name", "content"], "additionalProperties": False},
    )
    def scratch_write(name: str, content: str, append: bool = False) -> dict:
        path = resolve(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if append and path.exists() else ""
        text = existing + content
        if len(text.encode("utf-8")) > MAX_NOTE_BYTES:
            raise ValueError(f"note '{name}' would exceed {MAX_NOTE_BYTES} bytes; keep notes short")
        path.write_text(text, encoding="utf-8")
        return {"name": name, "bytes": len(text.encode("utf-8")), "appended": bool(append)}

    @registry.tool(
        "scratch_read",
        "Read back a note saved with scratch_write.",
        {"type": "object", "properties": {"name": {"type": "string"}},
         "required": ["name"], "additionalProperties": False},
    )
    def scratch_read(name: str) -> dict:
        path = resolve(name)
        if not path.is_file():
            raise FileNotFoundError(f"no note named '{name}'; scratch_list shows what there is")
        text = path.read_text(encoding="utf-8")
        return {"name": name, "chars": len(text), "content": text}

    @registry.tool(
        "scratch_list",
        "List the notes saved so far in this run, with their sizes.",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    )
    def scratch_list() -> dict:
        if not root.is_dir():
            return {"notes": []}
        return {"notes": [{"name": p.name, "bytes": p.stat().st_size}
                          for p in sorted(root.iterdir()) if p.is_file()]}
