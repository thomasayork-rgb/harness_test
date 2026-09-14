"""Basic tools, all rooted to a working directory. Path escapes are refused."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from ..registry import ToolRegistry
from .paths import SKIP_DIRS, make_resolver

# Everything the harness configures itself with lives under this prefix, and
# HARNESS_API_KEY is one of them. A shell command inherits the environment, so
# `env` - or anything that dumps it - would put the operator's key in a result,
# an artifact and the trajectory. The child gets the environment without them.
ENV_PREFIX = "HARNESS_"


def child_env() -> dict:
    """The environment a shell command runs with: this process's, minus HARNESS_*."""
    return {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIX)}


# How long to wait for a killed process group to go away before giving up on
# reaping it. Nothing survives SIGKILL, so this is only a floor on how long a
# timeout can take.
REAP_TIMEOUT = 5.0


def kill_group(proc: subprocess.Popen) -> None:
    """Kill the whole group the command was started in.

    ``start_new_session=True`` makes the shell a session and process-group
    leader, so everything it started - a backgrounded ``sleep``, a server, a
    build - is in that group and dies with it. Killing the shell alone leaves
    those children running with the pipes still open, which is what a timeout
    used to do.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        # AttributeError: no process groups here at all (Windows), so the best
        # available is what this used to do - kill the command itself
        proc.kill()


def register_basic_tools(registry: ToolRegistry, workdir: Path) -> None:
    root, resolve = make_resolver(workdir)

    @registry.tool(
        "fs_list",
        "List files and directories under a path relative to the workdir.\n"
        "Skips the directories the search tools skip: .git, caches, vendored trees.",
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
            # the same noise the walking tools skip: .git, caches, vendored trees
            if child.name in SKIP_DIRS:
                continue
            entries.append({"name": child.name, "type": "dir" if child.is_dir() else "file",
                            "bytes": child.stat().st_size if child.is_file() else None})
            if len(entries) >= max_entries:
                entries.append({"name": "...", "type": "truncated"})
                break
        return {"path": path, "entries": entries}

    @registry.tool(
        "fs_read",
        "Read a text file relative to the workdir. Optional offset/limit in characters.\n"
        "A file with NUL bytes in it is refused as binary rather than decoded into noise.",
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
        raw = p.read_bytes()
        # The same guard fs_edit has: decoding a binary with errors="replace"
        # produces thousands of characters of noise that cost context, say
        # nothing, and read as if the file were text.
        if b"\x00" in raw:
            raise ValueError(f"{path} looks binary (contains NUL bytes); refusing to read. "
                             "Use run_shell if you need to inspect it.")
        text = raw.decode("utf-8", errors="replace")
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
        "Run a shell command in the workdir. Returns stdout, stderr, exit code. Timeout in seconds.\n"
        "The command inherits this process's environment without its HARNESS_* variables.",
        {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
         "required": ["command"]},
    )
    def run_shell(command: str, timeout: int = 60) -> dict:
        # A non-zero exit is a result: the command ran and said no. A timeout is
        # a tool failure, so it raises and lands in the trajectory as kind
        # "error" rather than hiding inside an "ok" result.
        proc = subprocess.Popen(command, shell=True, cwd=root, text=True, env=child_env(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # everything the command started goes with it, not just the shell
            kill_group(proc)
            try:
                proc.communicate(timeout=REAP_TIMEOUT)
            except subprocess.TimeoutExpired:   # pragma: no cover - nothing survives SIGKILL
                pass
            raise TimeoutError(f"command timed out after {timeout}s: {command}") from None
        except KeyboardInterrupt:
            # a new session means Ctrl-C at the terminal never reached the
            # command; the run is ending, so take its group with it
            kill_group(proc)
            raise
        return {"command": command, "exit_code": proc.returncode,
                "stdout": out[-20000:], "stderr": err[-5000:]}
