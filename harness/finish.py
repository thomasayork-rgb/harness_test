"""What a project run does between the last step and the footer.

Two hooks, in this order (see ``harness.runtime.RunHooks``):

  verify   only when a final answer was accepted: run the project's tests, and
           ask whether the code map still describes what the run changed. The
           model has just told you it is done; this is the part of that claim a
           machine can check.
  finish   whatever ended the run: commit the worktree on the task branch, so
           the work survives the run that made it.

Neither can change the run's status. A failing test does not turn `completed`
into `failed`: the model said what it thinks, the footer says what the tests
said, and a reader decides. What that buys is a footer that is worth reading::

    "verify": {"test": {"passed": false, "exit": 1, "artifact": "artifacts/verify_test.txt", ...},
               "map_stale": {"area": ["alpha"], "stale": [...], "clean": false}}
    "commit": {"sha": "...", "message": "harness: first — completed", "files": 3}

The test command is the operator's: ``--test-command`` is trusted because the
operator typed it. A command from the project's own ``.harness/project.json``
is the project's code speaking, and is run only with ``--trust-project-plugins``
- the same rule the skills plugins already follow. Its output goes to an
artifact and never into the footer or the trajectory, which is where an
operator's secrets end up if a build log is quoted into a record.

The commit is one per segment: ``git add -A`` and a message that names the task
(or the run) and the status it ended with. Interrupted runs commit too - an
agent's half-finished work is exactly what should not be lost - and a resume
commits again on top.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .codemap import scan, stale_against_tree
from .project import (ProjectRun, entry_paths, git, head_sha, identity_args, is_dirty,
                      status_entries)
from .runtime import RunHooks
from .tools.basic import child_env, kill_group

# Where a project may name its own test command, and how it is read.
PROJECT_CONFIG = Path(".harness") / "project.json"
CONFIG_SOURCE = ".harness/project.json"
FLAG_SOURCE = "--test-command"

# The verify command's output lives here, in full, and nowhere else.
TEST_ARTIFACT = "verify_test.txt"
TEST_TIMEOUT = 600.0

# The commit a finished run leaves behind. The dash is an em dash; the message
# is read by people, in `git log --oneline`, next to messages people wrote.
COMMIT_MESSAGE = "harness: {name} — {status}"

UNTRUSTED = (
    "verify: {config} names a test command, and the run was not started with "
    "--trust-project-plugins, so it was not run. That command is the project's own code.")


def _note(text: str) -> None:
    """Something the operator should know about their own command line. Never
    the model's business: it is about how the run was launched."""
    print(text, file=sys.stderr)


@dataclass
class ProjectFinish:
    """The verify and finish hooks for one project run."""

    project: ProjectRun
    test_command: str | None = None
    test_timeout: float = TEST_TIMEOUT
    trust_project: bool = False
    auto_commit: bool = True
    area: list[str] = field(default_factory=list)

    def hooks(self) -> RunHooks:
        return RunHooks(verify=self.verify, finish=self.finish)

    # ---- the test command --------------------------------------------------

    def _project_config(self) -> dict:
        """``.harness/project.json`` of the worktree, or ``{}``. A file that is
        not readable JSON is not a configuration, and saying so is the CLI's
        job, not this hook's."""
        try:
            stored = json.loads((self.project.worktree / PROJECT_CONFIG).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return stored if isinstance(stored, dict) else {}

    def test_source(self) -> tuple[str | None, str | None, float]:
        """``(command, source, timeout)``: what to run, who said so, how long."""
        if self.test_command:
            return self.test_command, FLAG_SOURCE, self.test_timeout
        stored = self._project_config()
        command = stored.get("test")
        if not isinstance(command, str) or not command.strip():
            return None, None, self.test_timeout
        if not self.trust_project:
            _note(UNTRUSTED.format(config=CONFIG_SOURCE))
            return None, None, self.test_timeout
        timeout = stored.get("timeout")
        return (command, CONFIG_SOURCE,
                float(timeout) if isinstance(timeout, (int, float)) and timeout > 0
                else self.test_timeout)

    def run_test(self, runtime: Any) -> dict | None:
        """The project's tests, in the worktree. None when there are none.

        Same rules as ``run_shell``: the harness's own environment variables are
        stripped, the command gets its own session so a timeout kills whatever
        it started, and a non-zero exit is an answer rather than a crash.
        """
        command, source, timeout = self.test_source()
        if command is None:
            return None
        started = time.monotonic()
        proc = subprocess.Popen(command, shell=True, cwd=str(self.project.worktree), text=True,
                                env=child_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_group(proc)
            timed_out = True
            try:
                output, _ = proc.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:      # pragma: no cover - nothing survives SIGKILL
                output = ""
            output = (output or "") + f"\n[timed out after {timeout:g}s]\n"
        elapsed = int((time.monotonic() - started) * 1000)
        artifact = runtime.writer.write_named_artifact(
            TEST_ARTIFACT, f"$ {command}\n[{source}, exit {proc.returncode}, {elapsed} ms]\n\n"
            + (output or ""))
        return {"source": source, "command": command, "passed": proc.returncode == 0 and not timed_out,
                "exit": proc.returncode, "timed_out": timed_out, "elapsed_ms": elapsed,
                "artifact": artifact}

    # ---- the map -----------------------------------------------------------

    def changed_files(self) -> list[str]:
        """Every path this run touched in its worktree: what it committed on
        the branch, and what it has not committed yet."""
        found: list[str] = []
        if self.project.base_sha:
            diff = git("diff", "--name-only", self.project.base_sha, cwd=self.project.worktree,
                       check=False)
            found.extend(line.strip() for line in diff.splitlines() if line.strip())
        for line in status_entries(self.project.worktree):
            found.extend(entry_paths(line))
        return sorted({path for path in found if path})

    def changed_packages(self) -> list[str]:
        """The packages those files belong to: what a map would have to say
        something new about."""
        root = self.project.worktree
        directories = {package.directory.relative_to(root).as_posix(): name
                       for name, package in scan(root).packages.items()}
        found = {directories[Path(rel).parent.as_posix()]
                 for rel in self.changed_files()
                 if rel.endswith(".py") and Path(rel).parent.as_posix() in directories}
        return sorted(found)

    def map_report(self) -> dict:
        """Whether the map still describes the code, for the area this run was
        about - or, with no task to say, for whatever it changed.

        Asked of the worktree, not of HEAD: the map the agent edited against the
        code the agent edited, which is the pair a reviewer is about to read.
        """
        area = list(self.area) or self.changed_packages()
        if not area:
            return {"area": [], "stale": [], "clean": True}
        report = stale_against_tree(self.project.worktree, area, tree_root=self.project.worktree)
        return {"area": area, "stale": report, "clean": not report}

    # ---- the hooks ---------------------------------------------------------

    def verify(self, runtime: Any) -> dict:
        """What the run's answer looks like from outside the conversation."""
        return {"test": self.run_test(runtime), "map_stale": self.map_report()}

    def finish(self, runtime: Any, status: str) -> dict | None:
        """Commit the worktree on the task branch. None when there is nothing
        to commit, or when the operator asked for none."""
        worktree = self.project.worktree
        if not self.auto_commit or not worktree.is_dir() or not is_dirty(worktree):
            return None
        git("add", "-A", cwd=worktree)
        staged = [line for line in
                  git("diff", "--cached", "--name-only", cwd=worktree).splitlines() if line.strip()]
        if not staged:                             # everything dirty was ignored or already gone
            return None
        message = COMMIT_MESSAGE.format(name=self.project.task_id or self.project.run_id,
                                        status=status)
        git(*identity_args(worktree), "commit", "-m", message, cwd=worktree)
        return {"sha": head_sha(worktree), "message": message, "files": len(staged)}


def project_hooks(project: ProjectRun, *, test_command: str | None = None,
                  test_timeout: float = TEST_TIMEOUT, trust_project: bool = False,
                  auto_commit: bool = True, area: Iterable[str] = ()) -> RunHooks:
    """The hooks a ``--project`` run is given, from the flags it was given."""
    return ProjectFinish(project=project, test_command=test_command, test_timeout=test_timeout,
                         trust_project=trust_project, auto_commit=auto_commit,
                         area=list(area or [])).hooks()
