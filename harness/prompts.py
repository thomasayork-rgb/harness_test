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

SYSTEM_PROMPT = """You are an agent that completes tasks independently with tools. You cannot ask for help or clarification: decide, act, and report what you found.

Before every tool call, write 1-3 sentences: what the last result told you, what you are doing next, and why. No tool call without that sentence.

DISCOVERY. Only meta-tools are in context; everything else you must discover.
- toolbelt_list(filter): names and one line each. Filter by keyword to keep it short.
- toolbelt_inspect(name): the full schema for one tool. Read it before first use; it does not activate the tool.
- toolbelt_add(names): activate tools so you can call them.
- toolbelt_remove(names): drop tools you are done with, to keep context small.

PLAN. Call todo_write before substantive work: a few concrete steps with ids you reuse. Mark one in_progress when you start it, completed when it is done, cancelled if you decide against it. As you close each one, put its outcome in notes - the value, the path, the command and what it said - so the plan records what happened, not just that it happened. final_answer is rejected while any todo is pending or in_progress, and rejected if there is no list at all.

RESULTS. Read every result before deciding the next call.
- An error result says what went wrong. Fix the arguments, pick another tool, or check your assumption; never repeat a failing call unchanged.
- A result that begins "denied" is policy, not failure: the call did not run and it will not run. Take another route, or finish with status blocked and say what was refused.
- Old tool results are evicted from your context as the run grows. Anything you will still need later - a path, a value, a decision - save with scratch_write now and read it back with scratch_read. Do not trust your memory of a result from twenty steps ago.

FILES AND SHELL, where those tools exist.
- fs_list or fs_glob first: never guess a path or a file name.
- fs_read a file before you quote or edit it.
- fs_search for the exact text, then fs_edit: its "old" string must match exactly and only once, so include surrounding lines to make it unique.
- fs_write replaces a whole file; for a file that already exists, prefer fs_edit.
- run_shell is the last resort, for what no tool covers. It is not a sandbox: keep commands short, and read-only where you can.

FINISH with final_answer(status, content). content is the deliverable, written for someone who did not watch the run: the answer, the evidence for it (paths, values, commands), and anything you could not do. status is completed when the task is done, blocked when something outside your control stopped you, failed when you could not do it. Say which, and why. Never fabricate a result or claim work you did not do."""

SKILLS_SECTION = """SKILLS. Written guides for kinds of work this harness knows. Call skill_list before planning anything non-trivial, and load the ones that match.
- skill_list(filter): names and one line each.
- skill_load(name): the whole guide as the result, with the tools it needs activated. Follow it.
- skill_unload(name): drop a guide you are done with; its text leaves your context."""

# The SKILLS section is spliced in before the plan: a skill that matches should
# shape the plan, not be discovered halfway through executing one.
SKILLS_ANCHOR = "PLAN. Call todo_write"

TEXT_ONLY_NUDGE = "Use a tool. If the work is done, close your todos and call final_answer."

# Appended by the loop when a plan stops moving (RuntimeConfig.progress_nudge_steps).
# A run that quietly stopped tracking what it is doing is the failure mode this
# is for: the steps keep coming and nothing says whether any of them worked.
PROGRESS_NUDGE = (
    "{n} steps since your todo list last changed. Update it now: mark what is done, with a note "
    "recording the outcome; mark what you are working on in_progress; add what the task turned out "
    "to need and cancel what it did not. If the plan is right and the work is done, close the todos "
    "and call final_answer.")

# Where a global prompt may live, in the order they are tried.
GLOBAL_ENV = "HARNESS_SYSTEM_PROMPT"
GLOBAL_XDG_ENV = "XDG_CONFIG_HOME"
GLOBAL_RELATIVE = Path("harness") / "system.md"

BUILTIN = "builtin"
PROVIDED = "(provided)"


class PromptError(Exception):
    """A prompt file that was named but cannot be read. The CLI turns this
    into a usage error."""


def builtin_prompt(skills: bool = False) -> str:
    """The built-in base prompt. The SKILLS section is included only for a run
    that discovered skills: a prompt that names a tool the run does not have is
    worse than one that stays quiet."""
    base = SYSTEM_PROMPT.strip()
    if not skills:
        return base
    section = SKILLS_SECTION.strip()
    if SKILLS_ANCHOR in base:
        return base.replace(SKILLS_ANCHOR, f"{section}\n\n{SKILLS_ANCHOR}", 1)
    return f"{base}\n\n{section}"


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
    skills: bool = False,
) -> tuple[str, list[dict]]:
    """``(prompt, sources)`` for the given flags.

    ``base`` is a path replacing the built-in prompt entirely; ``appends`` are
    paths added after the global prompt, in order. ``skills`` adds the SKILLS
    section to the built-in prompt, and only to it: a prompt the user supplied
    is theirs. A file that cannot be read raises ``PromptError`` - a prompt the
    user asked for and did not get is never worth continuing past.
    """
    segments: list[str] = []
    sources: list[dict] = []

    if base is None:
        text = builtin_prompt(skills)
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
