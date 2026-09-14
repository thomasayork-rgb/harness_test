"""Projects: a run works in a git worktree, never in the repository itself.

``harness run --project PATH`` points a run at a git repository instead of a
directory. The repository is not the workdir: the run gets its own worktree,
cut from HEAD on a branch of its own::

    <runs-dir>/<run_id>/wt        the workdir the agent is given
    task/<run_id>                 the branch it is on

so a run cannot touch the operator's checkout, two runs cannot collide, and
what a run did is a branch someone can read, keep or delete afterwards. The
worktree lives under the run directory, next to the trajectory and the
artifacts, because it is part of the record of that run.

The tree has to be clean first. The worktree is cut from HEAD, so anything
uncommitted is invisible to the agent - it would work from a tree the operator
cannot see, and report against files that do not say what the operator's copy
says. ``tasks/`` is the exception: those files are the harness's own bookkeeping
and are meant to move while a run is in flight. ``--allow-dirty`` says "yes,
really".

Some git is denied by default for a project run (see ``DENY_SHELL_DEFAULTS``):
not because the agent is malicious, but because a worktree is a shared checkout
of one repository and ``git checkout``, ``git reset --hard`` or ``git push``
from inside it reach further than the run. The denial is an ordinary policy
denial - a result the model reads and can route around - and ``--allow-git``
lifts it.

Nothing here imports or executes project code. Every git call is
``subprocess`` with an argument list, rooted at a path this module resolved.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .runtime import RESUMABLE

GIT = "git"
GIT_TIMEOUT = 120.0
WORKTREE_DIR = "wt"
BRANCH_PREFIX = "task/"
STATE_FILE = "state.json"

# Changes here are the harness's own bookkeeping (task frontmatter moves to
# `doing` as a run starts), so they never make a tree dirty.
IGNORED_DIRTY = ("tasks/",)

# Uncommitted map files are the dirty case worth explaining: the worktree is cut
# from HEAD, so the agent would run against a map the operator can see and it
# cannot.
MAP_DIR = "docs/map/"

# Shell commands a project run refuses unless --allow-git. A worktree shares one
# repository with every other worktree of it: these reach outside this run.
DENY_SHELL_DEFAULTS = (
    r"\bgit\s+push\b",
    r"\bgit\s+checkout\b",
    r"\bgit\s+switch\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+worktree\b",
    r"\bgit\s+branch\s+-D\b",
)

# Who the harness is when it has to write a commit and the repository has not
# said who its committer is.
FALLBACK_NAME = "harness"
FALLBACK_EMAIL = "harness@local"

# What a resumed run is told when its worktree had been pruned. Said in the
# resume note and recorded at the seam: the conversation above it may refer to
# edits that are no longer on disk.
RECREATED = ("worktree was recreated from branch {branch} at {sha}; uncommitted changes "
             "from the pruned worktree are gone.")


class ProjectError(Exception):
    """Anything about a project that stops a run, with the reason to print."""


def git(*args: Any, cwd: Any, check: bool = True) -> str:
    """One git command, as an argument list. Returns stdout; raises
    ProjectError naming what failed."""
    argv = [GIT, *[str(a) for a in args]]
    try:
        proc = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True,
                              timeout=GIT_TIMEOUT)
    except FileNotFoundError:
        raise ProjectError("git is not on PATH; --project needs it") from None
    except subprocess.TimeoutExpired:
        raise ProjectError(f"git {' '.join(str(a) for a in args)}: timed out after "
                           f"{GIT_TIMEOUT:g}s") from None
    except OSError as e:
        raise ProjectError(f"git {' '.join(str(a) for a in args)}: {e}") from None
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise ProjectError(f"git {' '.join(str(a) for a in args)} failed in {cwd}: "
                           + (detail[0] if detail else f"exit {proc.returncode}"))
    return proc.stdout


def repo_root(path: Any) -> Path:
    """The top level of the repository ``path`` is in. Raises ProjectError when
    it is not a git repository - a project is a repository or it is nothing."""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise ProjectError(f"--project {path}: no such directory")
    try:
        out = git("rev-parse", "--show-toplevel", cwd=directory)
    except ProjectError:
        raise ProjectError(f"--project {directory}: not a git repository") from None
    return Path(out.strip()).resolve()


def head_sha(repo: Any, rev: str = "HEAD") -> str:
    """The commit ``rev`` names. Raises ProjectError on a repository with no
    commits, or a branch that is gone."""
    try:
        return git("rev-parse", "--verify", f"{rev}^{{commit}}", cwd=repo).strip()
    except ProjectError:
        raise ProjectError(f"{repo}: cannot resolve '{rev}'; a project needs at least "
                           "one commit") from None


def branch_exists(repo: Any, branch: str) -> bool:
    proc = subprocess.run([GIT, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                          cwd=str(repo), capture_output=True, text=True)
    return proc.returncode == 0


def identity_args(repo: Any) -> list[str]:
    """``-c user.name=... -c user.email=...`` when the repository has no
    committer configured, so a harness commit cannot fail on a machine that
    never set one. A repository that has an identity keeps it."""
    name = git("config", "--get", "user.name", cwd=repo, check=False).strip()
    email = git("config", "--get", "user.email", cwd=repo, check=False).strip()
    if name and email:
        return []
    return ["-c", f"user.name={FALLBACK_NAME}", "-c", f"user.email={FALLBACK_EMAIL}"]


def _entry_paths(line: str) -> list[str]:
    """The path(s) one ``git status --porcelain`` line is about."""
    body = line[3:] if len(line) > 3 else ""
    parts = [p.strip() for p in body.split(" -> ")] if " -> " in body else [body]
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] == '"' else p for p in parts]


def status_entries(path: Any) -> list[str]:
    """Every ``git status --porcelain`` line for a repository or worktree.

    Untracked files are listed one by one rather than as a directory: "?? docs/"
    hides whether the map is what is uncommitted, and that is the case worth
    naming.
    """
    out = git("status", "--porcelain", "--untracked-files=all", cwd=path)
    return [line for line in out.splitlines() if line.strip()]


def dirty_entries(path: Any) -> list[str]:
    """The status lines that stop a project run: everything but ``tasks/``."""
    out = []
    for line in status_entries(path):
        paths = _entry_paths(line)
        if paths and all(any(p.startswith(prefix) for prefix in IGNORED_DIRTY) for p in paths):
            continue
        out.append(line)
    return out


def is_dirty(path: Any) -> bool:
    """Whether a worktree has anything uncommitted at all - no exceptions: this
    is about losing an agent's work, not about starting one."""
    return bool(status_entries(path))


def dirty_message(repo: Any, entries: list[str]) -> str:
    """Why a dirty project refuses to start, in the terms of what is dirty."""
    shown = "\n".join(f"  {line}" for line in entries[:10])
    more = f"\n  ... {len(entries) - 10} more" if len(entries) > 10 else ""
    mapped = any(p.startswith(MAP_DIR) for line in entries for p in _entry_paths(line))
    why = (f"\nuncommitted {MAP_DIR} changes matter most: the worktree is cut from HEAD, so the "
           "agent would run without the map you are looking at." if mapped else "")
    return (f"--project {repo} has uncommitted changes:\n{shown}{more}{why}\n"
            "Commit them (or pass --allow-dirty). Changes under "
            + ", ".join(IGNORED_DIRTY) + " are ignored.")


def branch_for(run_id: str) -> str:
    return f"{BRANCH_PREFIX}{run_id}"


def add_worktree(repo: Any, path: Any, branch: str, start: str | None = None) -> None:
    """Check out ``branch`` at ``path``. With ``start``, the branch is created
    there; without it, an existing branch is checked out."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "prune", cwd=repo)          # clear metadata of worktrees that are gone
    if start is None:
        git("worktree", "add", str(path), branch, cwd=repo)
    else:
        git("worktree", "add", str(path), "-b", branch, start, cwd=repo)


def remove_worktree(repo: Any, path: Any, force: bool = False) -> None:
    """Remove a worktree, leaving its branch. The branch is the record."""
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    git(*args, str(path), cwd=repo)


@dataclass
class Worktree:
    """One worktree of a project that belongs to a run."""

    run_id: str
    path: Path
    branch: str | None
    status: str
    dirty: bool

    @property
    def removable(self) -> bool:
        """Finished runs only: a run that can still be resumed keeps its tree."""
        return self.status not in ("running", "unknown") and self.status not in RESUMABLE


def _porcelain_worktrees(repo: Any) -> list[tuple[Path, str | None]]:
    """``(path, branch)`` per worktree git knows about, main one included."""
    out: list[tuple[Path, str | None]] = []
    path: Path | None = None
    branch: str | None = None
    for line in git("worktree", "list", "--porcelain", cwd=repo).splitlines() + [""]:
        if line.startswith("worktree "):
            path = Path(line[len("worktree "):]).resolve()
            branch = None
        elif line.startswith("branch "):
            ref = line[len("branch "):].strip()
            branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        elif not line.strip() and path is not None:
            out.append((path, branch))
            path, branch = None, None
    return out


def run_status(run_dir: Any) -> str:
    """What ``state.json`` says a run ended as, or ``unknown``."""
    try:
        stored = json.loads((Path(run_dir) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    status = stored.get("status")
    return status if isinstance(status, str) and status else "unknown"


def worktrees(repo: Any, runs_dir: Any) -> list[Worktree]:
    """Every worktree of this repository that lives under ``runs_dir``, which
    is every worktree a harness run made, oldest run id first."""
    root = Path(runs_dir).resolve()
    found: list[Worktree] = []
    for path, branch in _porcelain_worktrees(repo):
        if root != path and root not in path.parents:
            continue
        run_dir = path.parent
        found.append(Worktree(run_id=run_dir.name, path=path, branch=branch,
                              status=run_status(run_dir),
                              dirty=is_dirty(path) if path.is_dir() else False))
    return sorted(found, key=lambda w: w.run_id)


@dataclass
class ProjectRun:
    """One run's worktree, and what the trajectory records about it."""

    repo: Path
    run_id: str
    branch: str
    base_sha: str
    worktree: Path
    task_id: str | None = None
    deny_shell_patterns: list[str] = field(default_factory=list)

    def header(self) -> dict:
        """The ``project`` block of the trajectory header."""
        return {"path": str(self.repo), "base_sha": self.base_sha, "branch": self.branch,
                "worktree": str(self.worktree), "task_id": self.task_id}

    def abandon(self) -> None:
        """Best-effort cleanup for a run that never started: drop the worktree,
        keep the branch. Never raises - the caller is already reporting."""
        try:
            remove_worktree(self.repo, self.worktree, force=True)
        except ProjectError:
            pass

    def release(self) -> str:
        """Remove the worktree of a finished run (``--no-keep-worktree``), or
        say why it was kept. The branch always stays."""
        if not self.worktree.is_dir():
            return f"worktree {self.worktree} is already gone"
        if is_dirty(self.worktree):
            return (f"worktree {self.worktree} has uncommitted changes; left in place "
                    f"(branch {self.branch})")
        remove_worktree(self.repo, self.worktree)
        return f"removed worktree {self.worktree}; branch {self.branch} kept"


def start(project: Any, runs_dir: Any, run_id: str, *, allow_dirty: bool = False,
          allow_git: bool = False, task_id: str | None = None) -> ProjectRun:
    """Cut a worktree for this run and hand back where it is.

    Raises ProjectError for anything that should stop the run before it starts:
    not a repository, no commits, or a tree with uncommitted work the agent
    would never see.
    """
    repo = repo_root(project)
    base = head_sha(repo)
    if not allow_dirty:
        entries = dirty_entries(repo)
        if entries:
            raise ProjectError(dirty_message(repo, entries))
    branch = branch_for(run_id)
    if branch_exists(repo, branch):
        raise ProjectError(f"branch {branch} already exists in {repo}; use --run-id to pick "
                           "another run id")
    worktree = (Path(runs_dir).resolve() / run_id / WORKTREE_DIR)
    add_worktree(repo, worktree, branch, start=base)
    return ProjectRun(repo=repo, run_id=run_id, branch=branch, base_sha=base,
                      worktree=worktree, task_id=task_id,
                      deny_shell_patterns=[] if allow_git else list(DENY_SHELL_DEFAULTS))


def reattach(block: Any) -> tuple[Path, str | None]:
    """The worktree a resumed run continues in, and what to tell the model.

    A worktree that is still there is reused as it stands. One that was pruned
    is cut again from its branch - the committed work is all there is, so the
    note says so rather than letting the model assume its edits survived.
    """
    if not isinstance(block, dict):
        raise ProjectError("this run's trajectory has no project block to resume from")
    worktree = Path(block.get("worktree") or "")
    branch = block.get("branch") or ""
    repo = Path(block.get("path") or "")
    if (worktree / ".git").exists():
        return worktree, None
    if worktree.exists():
        try:
            worktree.rmdir()            # an empty leftover: git will not add into it
        except OSError:
            raise ProjectError(f"{worktree} exists but is not a worktree of {repo}; "
                               "move it aside to resume this run") from None
    if not repo.is_dir():
        raise ProjectError(f"project {repo} is gone; nothing to recreate the worktree from")
    if not branch or not branch_exists(repo, branch):
        raise ProjectError(f"worktree {worktree} was removed and branch {branch or '(none)'} "
                           f"is gone from {repo}; this run cannot be resumed")
    sha = head_sha(repo, branch)
    add_worktree(repo, worktree, branch)
    return worktree, RECREATED.format(branch=branch, sha=sha)


def prune(repo: Any, runs_dir: Any, force: bool = False) -> list[str]:
    """Remove the worktrees of finished runs. Returns one line per decision.

    A run that can still be resumed keeps its worktree, and so does one with
    uncommitted work in it unless ``force``: this deletes an agent's output,
    and a resumable run is not finished with it.
    """
    lines: list[str] = []
    found = worktrees(repo, runs_dir)
    width = max((len(wt.run_id) for wt in found), default=0)
    for wt in found:
        name = wt.run_id.ljust(width)
        if not wt.removable:
            lines.append(f"kept    {name}  {wt.status}: not finished")
            continue
        if wt.dirty and not force:
            lines.append(f"kept    {name}  {wt.status}: uncommitted changes "
                         "(--force to remove anyway)")
            continue
        try:
            remove_worktree(repo, wt.path, force=wt.dirty)
        except ProjectError as e:
            lines.append(f"failed  {name}  {e}")
            continue
        lines.append(f"removed {name}  {wt.status}: branch {wt.branch or '(detached)'} kept")
    return lines


def format_worktrees(found: Iterable[Worktree]) -> str:
    """The table ``harness worktree list`` prints."""
    rows = [["run id", "status", "branch", "tree", "path"]]
    for wt in found:
        rows.append([wt.run_id, wt.status, wt.branch or "(detached)",
                     "dirty" if wt.dirty else "clean", str(wt.path)])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip()
                     for row in rows)
