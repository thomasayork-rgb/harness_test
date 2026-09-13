"""Basic tools, all rooted to a working directory. Path escapes are refused."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..registry import ToolRegistry
from .paths import make_resolver


def register_basic_tools(registry: ToolRegistry, workdir: Path) -> None:
    root, resolve = make_resolver(workdir)

    @registry.tool(
        "fs_list",
        "List files and directories under a path relative to the workdir.",
        {"type": "object", "properties": {"path": {"type": "string"}, "max_entries": {"type": "integer"}},
         "required": []},
    )
    def fs_list(path: str = ".", max_entries: int = 200) -> dict:
        p = resolve(path)
        if not p.exists():
            raise FileNotFoundError(path)
        if p.is_file():
            return {"path": path, "type": "file", "bytes": p.stat().st_size}
        entries = []
        for child in sorted(p.iterdir()):
            if child.name.startswith(".git"):
                continue
            entries.append({"name": child.name, "type": "dir" if child.is_dir() else "file",
                            "bytes": child.stat().st_size if child.is_file() else None})
            if len(entries) >= max_entries:
                entries.append({"name": "...", "type": "truncated"})
                break
        return {"path": path, "entries": entries}

    @registry.tool(
        "fs_read",
        "Read a text file relative to the workdir. Optional offset/limit in characters.",
        {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"},
                                          "limit": {"type": "integer"}},
         "required": ["path"]},
    )
    def fs_read(path: str, offset: int = 0, limit: int = 20000) -> dict:
        p = resolve(path)
        # Report the path the model asked for; the absolute one is not its business
        # and would follow the run into the trajectory.
        if not p.exists():
            raise FileNotFoundError(f"no such file: {path}")
        if not p.is_file():
            raise IsADirectoryError(f"not a file: {path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "total_chars": len(text), "offset": offset, "content": text[offset: offset + limit]}

    @registry.tool(
        "fs_write",
        "Write a text file relative to the workdir. Creates parent directories. Overwrites.",
        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["path", "content"]},
    )
    def fs_write(path: str, content: str) -> dict:
        p = resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": path, "bytes": len(content.encode("utf-8"))}

    @registry.tool(
        "run_shell",
        "Run a shell command in the workdir. Returns stdout, stderr, exit code. Timeout in seconds.",
        {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
         "required": ["command"]},
    )
    def run_shell(command: str, timeout: int = 60) -> dict:
        # A non-zero exit is a result: the command ran and said no. A timeout is
        # a tool failure, so it raises and lands in the trajectory as kind
        # "error" rather than hiding inside an "ok" result.
        try:
            proc = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"command timed out after {timeout}s: {command}") from None
        return {"command": command, "exit_code": proc.returncode,
                "stdout": proc.stdout[-20000:], "stderr": proc.stderr[-5000:]}
