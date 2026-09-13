"""Prompts, and how the system prompt is assembled.

Kept short on purpose; local models pay for every token here.

Three layers, in this order:

  base    the built-in prompt, or the file named by ``--system-prompt``
  global  house rules that apply to every run on this machine: the first
          existing of ``$HARNESS_SYSTEM_PROMPT``,
          ``$XDG_CONFIG_HOME/harness/system.md``, ``~/.config/harness/system.md``
          (skipped entirely with ``--no-global-prompt``)
  append  each ``--append-system-prompt`` file, in the order given

Layers are joined with a blank line, and each one that was loaded is recorded
in the trajectory header as

    {"source": "builtin" | "<path>", "role": "base" | "global" | "append",
     "chars": n}

so a trace says what the model was told, not just what it did. The prompt is
resolved once, at the start of a run: a resumed run replays the system message
already in ``state.json`` and never re-reads a file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

SYSTEM_PROMPT = """You are an agent that completes tasks independently using tools. You cannot ask for help or clarification.

Before each tool call, write 1-3 sentences: what you learned from the last result, and what you are doing next and why. Be concise.

Tools available now are only meta-tools. Discover the rest:
- toolbelt_list: list available tools (name + one line). Optional keyword filter, matched against both.
- toolbelt_inspect: full schema for one tool. Does not activate it.
- toolbelt_add: activate tools so you can call them.
- toolbelt_remove: deactivate tools you no longer need.

Plan with todo_write before doing substantive work, and keep it current (in_progress when you start, completed when done). final_answer is rejected while any todo is pending or in_progress; cancel what you will not do.

Do not assume file names, paths, or contents. List first. Never fabricate a result.

Finish with final_answer(status, content). status is completed, blocked, or failed."""

TEXT_ONLY_NUDGE = "Use a tool. If the work is done, close your todos and call final_answer."

# Where a global prompt may live, in the order they are tried.
GLOBAL_ENV = "HARNESS_SYSTEM_PROMPT"
GLOBAL_XDG_ENV = "XDG_CONFIG_HOME"
GLOBAL_RELATIVE = Path("harness") / "system.md"

BUILTIN = "builtin"
PROVIDED = "(provided)"


class PromptError(Exception):
    """A prompt file that was named but cannot be read. The CLI turns this
    into a usage error."""


def global_prompt_candidates(env: Mapping[str, str] | None = None) -> list[Path]:
    """Every location a global prompt could live, most specific first."""
    env = os.environ if env is None else env
    out: list[Path] = []
    named = env.get(GLOBAL_ENV)
    if named:
        out.append(Path(named).expanduser())
    xdg = env.get(GLOBAL_XDG_ENV)
    if xdg:
        out.append(Path(xdg).expanduser() / GLOBAL_RELATIVE)
    home = env.get("HOME")
    base = Path(home).expanduser() if home else Path.home()
    out.append(base / ".config" / GLOBAL_RELATIVE)
    return out


def global_prompt_path(env: Mapping[str, str] | None = None) -> Path | None:
    """The global prompt file, or None. Only the first existing location is
    used: a machine has one set of house rules, not three."""
    for path in global_prompt_candidates(env):
        if path.is_file():
            return path
    return None


def _read(path: Any) -> str:
    try:
        return Path(path).expanduser().read_text(encoding="utf-8")
    except OSError as e:
        raise PromptError(f"{path}: cannot read prompt file: {e.strerror or e}") from None


def _layer(source: str, role: str, text: str) -> dict:
    return {"source": source, "role": role, "chars": len(text)}


def resolve_system_prompt(
    base: Any = None,
    appends: Iterable[Any] = (),
    *,
    use_global: bool = True,
    env: Mapping[str, str] | None = None,
) -> tuple[str, list[dict]]:
    """``(prompt, sources)`` for the given flags.

    ``base`` is a path replacing the built-in prompt entirely; ``appends`` are
    paths added after the global prompt, in order. A file that cannot be read
    raises ``PromptError`` - a prompt the user asked for and did not get is
    never worth continuing past.
    """
    segments: list[str] = []
    sources: list[dict] = []

    if base is None:
        text = SYSTEM_PROMPT.strip()
        sources.append(_layer(BUILTIN, "base", text))
    else:
        text = _read(base).strip()
        sources.append(_layer(str(Path(base).expanduser()), "base", text))
    segments.append(text)

    if use_global:
        path = global_prompt_path(env)
        if path is not None:
            text = _read(path).strip()
            sources.append(_layer(str(path), "global", text))
            segments.append(text)

    for extra in appends:
        text = _read(extra).strip()
        sources.append(_layer(str(Path(extra).expanduser()), "append", text))
        segments.append(text)

    return "\n\n".join(s for s in segments if s), sources
