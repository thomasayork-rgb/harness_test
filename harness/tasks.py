"""Tasks: the work a project keeps in its own repository.

A task is a markdown file the project owns::

    tasks/fix-the-port.md

    ---
    status: todo                                   # todo | doing | done | blocked
    area: [harness.tools]                          # the packages it is about
    done_when: [tests pass, map updated]           # what finishing means
    branch: null                                   # filled in by the run
    run_id: null
    ---
    Make run_shell kill the process group on a timeout.

The body is the prompt: ``harness run --project P --task-id fix-the-port`` is
the same run as ``--task "<that body>"``, with three differences. The task's
``done_when`` becomes a layer of the system prompt, so the model is told what
it has to show before it can finish. The task's ``area`` is what the finish
hooks check the code map against. And the file itself is the record: the run
writes ``doing`` with its branch and run id into the frontmatter before the
first request, and the status the run ended with when it is over.

That last part is why the frontmatter is written back rather than kept in the
run directory. Two people, or two runs, looking at one repository can see which
tasks are taken; ``tasks/`` is the one path a project run may write to outside
its worktree, and the one path the dirty check ignores (see harness.project).

Statuses move one way per run::

    todo ---(a run starts)---> doing ---(completed)---> done
                                     ---(blocked/failed)---> blocked
                                     ---(interrupted, step_cap, ...)---> doing

A run that can still be resumed leaves the task ``doing``: the work is not
finished and not abandoned, and the resume closes it. A second run against a
task that is already ``doing`` is refused unless it is forced, because two runs
on one task overwrite each other's branch in the file and neither is wrong.

Nothing here imports or executes project code: a task file is text, parsed by
``harness.frontmatter``, and the body is handed to the model as the task.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .frontmatter import FrontmatterError, NoFrontmatter, dump, parse_frontmatter, split_frontmatter

TASKS_DIR = Path("tasks")

TODO, DOING, DONE, BLOCKED = "todo", "doing", "done", "blocked"
STATUSES = (TODO, DOING, DONE, BLOCKED)

# What a finished run does to the task it ran. A status that is not here is an
# interruption (transport_error, step_cap, stalled, interrupted): the task stays
# `doing` and the resume finishes it.
AFTER_RUN = {"completed": DONE, "blocked": BLOCKED, "failed": BLOCKED}

# The frontmatter keys a task owns, in the order a new block writes them. A file
# that has others keeps them, in the order it already had.
FIELDS = ("status", "area", "done_when", "branch", "run_id")

# What "not set" looks like in the block: dump writes None as `null`, and the
# parser hands the string back (values are strings; the consumer decodes).
EMPTY = ("", "null", "none", "~")

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class TaskError(Exception):
    """A task file that cannot be used, with the reason to report."""


def _optional(value: Any) -> str | None:
    """A scalar that may be absent: ``null`` and ``""`` are None."""
    if not isinstance(value, str):
        return None
    return None if value.strip().lower() in EMPTY else value.strip()


def _list(value: Any, what: str, path: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return [v for v in value if v.strip()]
    raise TaskError(f"{path}: '{what}' must be a list")


@dataclass
class Task:
    """One task file: its frontmatter, and the body that is the prompt."""

    id: str
    path: Path
    status: str
    area: list[str] = field(default_factory=list)
    done_when: list[str] = field(default_factory=list)
    branch: str | None = None
    run_id: str | None = None
    meta: dict = field(default_factory=dict)      # the block as parsed, for writing back
    body: str = ""                                # the body as written, byte for byte

    @property
    def prompt(self) -> str:
        """What the model is given as the task."""
        return self.body.strip()

    def brief(self) -> dict:
        return {"id": self.id, "status": self.status, "area": list(self.area),
                "branch": self.branch, "run_id": self.run_id}

    def prompt_meta(self) -> dict:
        """What the `task` prompt layer needs (see harness.prompts)."""
        return {"id": self.id, "area": list(self.area), "done_when": list(self.done_when),
                "source": str(self.path)}

    def write(self, **updates: Any) -> "Task":
        """Rewrite the frontmatter with these fields, keeping the body.

        The body is what the author wrote and the run was given; only the block
        above it is the harness's to touch.
        """
        current = {"status": self.status, "area": list(self.area),
                   "done_when": list(self.done_when), "branch": self.branch, "run_id": self.run_id}
        meta = dict(self.meta)
        for key in FIELDS:                        # a file that omitted one gets it now
            meta.setdefault(key, current[key])
        meta.update(updates)
        order = list(self.meta) + [k for k in meta if k not in self.meta]
        ordered = {key: meta[key] for key in order}
        try:
            self.path.write_text(dump(ordered, self.body), encoding="utf-8")
        except OSError as e:
            raise TaskError(f"{self.path}: cannot write task: {e.strerror or e}") from None
        return load_file(self.path)

    def mark(self, status: str, **updates: Any) -> "Task":
        """Move the task to ``status`` (with whatever else changed)."""
        if status not in STATUSES:
            raise TaskError(f"unknown task status '{status}'; one of {', '.join(STATUSES)}")
        return self.write(status=status, **updates)


def tasks_dir(project: Any) -> Path:
    return Path(project) / TASKS_DIR


def load_file(path: Any) -> Task:
    """Read one task file. Raises TaskError naming the file."""
    path = Path(path)
    if not _ID.match(path.stem):
        raise TaskError(f"{path}: a task id must look like a name, not {path.stem!r}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise TaskError(f"{path}: cannot read task: {e.strerror or e}") from None
    try:
        meta, _ = parse_frontmatter(text)
        _, body = split_frontmatter(text)
    except NoFrontmatter:
        raise TaskError(f"{path}: a task file needs a '---' frontmatter block") from None
    except FrontmatterError as e:
        raise TaskError(f"{path}: {e}") from None

    status = _optional(meta.get("status")) or TODO
    if status not in STATUSES:
        raise TaskError(f"{path}: status '{status}' is not one of {', '.join(STATUSES)}")
    return Task(id=path.stem, path=path, status=status,
                area=_list(meta.get("area"), "area", path),
                done_when=_list(meta.get("done_when"), "done_when", path),
                branch=_optional(meta.get("branch")), run_id=_optional(meta.get("run_id")),
                meta=meta, body=body)


def load(project: Any, task_id: str) -> Task:
    """One task of a project by id. Raises TaskError when there is no such file."""
    path = tasks_dir(project) / f"{task_id}.md"
    if not path.is_file():
        known = ", ".join(t.id for t in load_all(project)[0]) or "(none)"
        raise TaskError(f"no task '{task_id}' in {tasks_dir(project)}. Tasks: {known}")
    return load_file(path)


def load_all(project: Any) -> tuple[list[Task], list[str]]:
    """``(tasks by id, errors)``. A file that cannot be read is reported, once,
    and left out rather than stopping the command that listed it."""
    found: list[Task] = []
    errors: list[str] = []
    directory = tasks_dir(project)
    try:
        entries = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".md")
    except OSError:
        return [], []
    for path in entries:
        try:
            found.append(load_file(path))
        except TaskError as e:
            errors.append(str(e))
    return found, errors


def next_task(project: Any) -> Task | None:
    """The first task still to do, by id order. None when there is none."""
    for task in load_all(project)[0]:
        if task.status == TODO:
            return task
    return None


def after_run(run_status: str) -> str | None:
    """The task status a run ending in ``run_status`` implies, or None to leave
    the task alone: an interrupted run is not a finished task."""
    return AFTER_RUN.get(run_status)


def close(task: Any, run_status: str) -> str | None:
    """Move a task to what the run's status implies, and say what happened.

    Returns a line for the operator, or None when the run was an interruption
    and the task stays where it is.
    """
    wanted = after_run(run_status)
    if wanted is None:
        return None
    # re-read: the run wrote `doing` into this file after it was first loaded
    task = load_file(task.path if isinstance(task, Task) else task)
    was = task.status
    task.mark(wanted)
    return f"task {task.id}: {was} -> {wanted}"


def format_tasks(found: Iterable[Task]) -> str:
    """The table ``harness tasks list`` prints."""
    rows = [["id", "status", "area", "branch"]]
    for task in found:
        rows.append([task.id, task.status, ", ".join(task.area) or "-", task.branch or "-"])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip()
                     for row in rows)


def format_task(task: Task) -> str:
    """One task in full: what it is, what finishing means, and the prompt."""
    lines = [f"task {task.id}  [{task.status}]  {task.path}"]
    if task.area:
        lines.append("area: " + ", ".join(task.area))
    if task.done_when:
        lines.append("done when:")
        lines.extend(f"  - {item}" for item in task.done_when)
    if task.branch or task.run_id:
        lines.append(f"branch: {task.branch or '-'}  run: {task.run_id or '-'}")
    lines.append("")
    lines.append(task.prompt)
    return "\n".join(lines)
