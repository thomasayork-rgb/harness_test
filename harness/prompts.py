"""Prompts, and how the system prompt is assembled.

Kept short on purpose; local models pay for every token here.

Five layers, in this order:

  base    the built-in prompt, or the file named by ``--system-prompt``
  global  house rules that apply to every run on this machine: the first
          existing of ``$HARNESS_SYSTEM_PROMPT``,
          ``$XDG_CONFIG_HOME/harness/system.md``, ``~/.config/harness/system.md``
          (skipped entirely with ``--no-global-prompt``)
  append  each ``--append-system-prompt`` file, in the order given
  project for a ``--project`` run: how this repository is worked on, and an
          excerpt of its code map index - the packages a reader would start
          from (see harness.codemap)
  task    for a ``--task-id`` run: what that task calls finished, as checks the
          model has to show evidence for (see harness.tasks)

Layers are joined with a blank line, and each one that was loaded is recorded
in the trajectory header as

    {"source": "builtin" | "<path>", "role": "base" | "global" | "append"
               | "project" | "task", "chars": n}

so a trace says what the model was told, not just what it did. The prompt is
resolved once, at the start of a run: a resumed run replays the system message
already in ``state.json`` and never re-reads a file.

The project layer is the one with a budget, because the index of a large
repository is not something to paste into every request: the top packages by
inbound references plus whatever the task is about, and a line telling the
model to read the rest itself. ``--project-prompt-chars`` is the ceiling, and
excerpt lines are dropped from the bottom until the layer fits under it.
"""
from __future__ import annotations

import os
import re
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

# ---- the project and task layers ------------------------------------------

# Where the code map index lives inside a project (see harness.codemap).
INDEX_RELATIVE = Path("docs") / "map" / "INDEX.md"

# How much of the system prompt a project may take, and how much of its index
# is worth quoting. The excerpt is a starting point, not the map.
PROJECT_PROMPT_CHARS = 6000
EXCERPT_PACKAGES = 20

PROJECT_LAYER = """PROJECT. You are working in a git worktree of this repository, on a branch of your own, and the repository keeps a code map: one file per package under docs/map/, generated facts above prose someone wrote.
- Read docs/map/INDEX.md first, then the map file of any package you are about to work in, before you read that package's code.
- A package whose line says [stale], or that has no map file, is not described: load the `map` skill and map it before you change it.
- Before final_answer, update the map file of every package you changed, so that `harness map stale` would be clean for this task's area.
- The harness commits your work on the task branch when the run ends, so you do not have to drive git. Commit mid-run only if you want a finer history.

The most depended-on packages of this project:"""

PROJECT_UNMAPPED = """PROJECT. You are working in a git worktree of this repository, on a branch of your own. The repository has no code map: docs/map/INDEX.md has not been scaffolded.
- Load the `map` skill and map any package you are about to change, before you change it.
- Before final_answer, make sure the map file of every package you changed describes it.
- The harness commits your work on the task branch when the run ends, so you do not have to drive git. Commit mid-run only if you want a finer history."""

# The excerpt is the beginning of the index, never the whole of it.
INDEX_REST = "fs_read docs/map/INDEX.md for the rest."

TASK_LAYER = "Before final_answer show evidence for each of these checks:"

# A line of docs/map/INDEX.md that names a package: "- pkg (inbound 3): what it is".
_INDEX_LINE = re.compile(r"^-\s+(\S+)\s+\(inbound\s+(\d+)\)")

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


def in_area(package: str, area: Iterable[str]) -> bool:
    """Whether a package is one of these areas, or inside one."""
    return any(package == a or package.startswith(str(a) + ".") for a in area)


def index_excerpt(text: str, area: Iterable[str] = (),
                  limit: int = EXCERPT_PACKAGES) -> list[str]:
    """The lines of an INDEX.md worth putting in front of the model.

    The ``limit`` most depended-on packages, plus every package this task is
    about however little depends on it, in the order the index lists them - so
    the excerpt reads like the top of the file it is quoting.
    """
    wanted = list(area or [])
    found: list[tuple[str, int, str]] = []
    for line in text.splitlines():
        match = _INDEX_LINE.match(line.strip())
        if match:
            found.append((match.group(1), int(match.group(2)), line.rstrip()))
    top = {name for name, _, _ in sorted(found, key=lambda r: (-r[1], r[0]))[:limit]}
    return [line for name, _, line in found if name in top or in_area(name, wanted)]


def project_layer(project: Any, area: Iterable[str] = (),
                  limit: int = PROJECT_PROMPT_CHARS) -> tuple[str, str]:
    """``(text, source)`` for the project layer: how this repository is worked
    on, and the head of its map index.

    A project with no index gets the other half of the same instruction - map
    it before you change it - because a model told to consult a map that is not
    there will invent one.
    """
    index = Path(project) / INDEX_RELATIVE
    try:
        text = index.read_text(encoding="utf-8")
    except OSError:
        return PROJECT_UNMAPPED, str(index)
    lines = index_excerpt(text, area)
    while True:
        layer = "\n".join([PROJECT_LAYER, *lines, "", INDEX_REST])
        if len(layer) <= limit or not lines:
            return layer, str(index)
        lines.pop()          # the least depended-on first: the excerpt is a head


def task_layer(task: Mapping[str, Any]) -> tuple[str, str] | None:
    """``(text, source)`` for the task layer, or None when the task says
    nothing about what finishing means."""
    checks = [str(c).strip() for c in (task.get("done_when") or []) if str(c).strip()]
    if not checks:
        return None
    return "\n".join([TASK_LAYER, *(f"- {check}" for check in checks)]), str(task.get("source") or "")


def resolve_system_prompt(
    base: Any = None,
    appends: Iterable[Any] = (),
    *,
    use_global: bool = True,
    env: Mapping[str, str] | None = None,
    skills: bool = False,
    project: Any = None,
    task: Mapping[str, Any] | None = None,
    project_chars: int = PROJECT_PROMPT_CHARS,
) -> tuple[str, list[dict]]:
    """``(prompt, sources)`` for the given flags.

    ``base`` is a path replacing the built-in prompt entirely; ``appends`` are
    paths added after the global prompt, in order. ``skills`` adds the SKILLS
    section to the built-in prompt, and only to it: a prompt the user supplied
    is theirs. A file that cannot be read raises ``PromptError`` - a prompt the
    user asked for and did not get is never worth continuing past.

    ``project`` is the repository a ``--project`` run works on, and ``task``
    what ``harness.tasks.Task.prompt_meta`` describes. Both add a layer after
    the appends, and both are about this run rather than this machine, so they
    come last: the closer to the task, the later it is said.
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

    if project is not None:
        text, source = project_layer(project, (task or {}).get("area") or (), project_chars)
        sources.append(_layer(source, "project", text))
        segments.append(text)

    if task is not None:
        made = task_layer(task)
        if made is not None:
            text, source = made
            sources.append(_layer(source, "task", text))
            segments.append(text)

    return "\n\n".join(s for s in segments if s), sources
