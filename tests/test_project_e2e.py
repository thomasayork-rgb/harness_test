"""One whole project run, as a process, over HTTP: the point of the layer.

A repository with a committed code map and a task in it. One real `python -m
harness run --project P --task-id ID` against the mock server, in which the
model lists the bundled skills, loads `orient`, reads the index and the map
before the code, plans, changes a package, refreshes that package's map with
`harness map scaffold`, loads `handoff` and reports. Then the harness does the
part that is not the model's: the tests run, the map is checked, the work is
committed on the task branch, and the task file says done.

Everything else is asserted somewhere narrower. What this test is for is that
the pieces fit: the prompt layers, the bundled skills, the task file, the
worktree, verify and the commit, in one run a person could have started.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from harness.tasks import load as load_task
from harness.trajectory import last, read_trajectory
from harness.transport import call

from .gitfixture import commit_all, git, git_repo, write

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "sekret"

FILES = {
    ".gitignore": "__pycache__/\n",
    "alpha/__init__.py": '"""The alpha package."""\nfrom .core import Engine\n\n__all__ = ["Engine"]\n',
    "alpha/core.py": ("from beta.util import helper\n\n\nclass Engine:\n"
                      "    def start(self):\n        return helper('start')\n"),
    "beta/__init__.py": "",
    "beta/util.py": "def helper(what):\n    return f'{what}!'\n",
    "tasks/add-stop.md": ("---\nstatus: todo\narea: [alpha]\n"
                          "done_when: [Engine.stop returns a value, the map for alpha is current]\n"
                          "branch: null\nrun_id: null\n---\n\n"
                          "Give Engine a stop() method beside start(), using the same helper.\n"),
}

PLAN = [{"id": "read", "content": "read the map for alpha", "status": "in_progress"},
        {"id": "change", "content": "add Engine.stop", "status": "pending"},
        {"id": "map", "content": "refresh the map for alpha", "status": "pending"}]


def closed(*ids, **notes):
    return [dict(item, status="completed" if item["id"] in ids else item["status"],
                 **({"notes": notes[item["id"]]} if item["id"] in notes else {}))
            for item in PLAN]


SCRIPT = [
    {"content": "A project run: what guides are there?", "tool_calls": [call("skill_list", {})]},
    {"content": "orient is the one for a mapped repository.",
     "tool_calls": [call("skill_load", {"name": "orient"})]},
    {"content": "It says index, then map file, then code. I need the file tools.",
     "tool_calls": [call("toolbelt_add", {"names": ["fs_read", "fs_edit", "run_shell"]})]},
    {"content": "Planning the three steps.", "tool_calls": [call("todo_write", {"todos": PLAN})]},
    {"content": "The index first.", "tool_calls": [call("fs_read", {"path": "docs/map/INDEX.md"})]},
    {"content": "Then alpha's map, before alpha's code.",
     "tool_calls": [call("fs_read", {"path": "docs/map/alpha.md"})]},
    {"content": "Engine.start is the entry point it named.",
     "tool_calls": [call("fs_read", {"path": "alpha/core.py"})]},
    {"content": "I have what I need.",
     "tool_calls": [call("todo_write", {"todos": closed(
         "read", read="docs/map/alpha.md: Engine.start is the entry point")})]},
    {"content": "Adding stop() beside start().",
     "tool_calls": [call("fs_edit", {"path": "alpha/core.py",
                                     "old": "    def start(self):\n        return helper('start')\n",
                                     "new": "    def start(self):\n        return helper('start')\n\n"
                                            "    def stop(self):\n        return helper('stop')\n"})]},
    {"content": "The map's Purpose no longer describes alpha; the prose first.",
     "tool_calls": [call("fs_edit", {"path": "docs/map/alpha.md",
                                     "old": "The engine: Engine.start is the whole of it.",
                                     "new": "The engine: Engine.start and Engine.stop, both via beta.util.helper."})]},
    {"content": "Then scaffold, which refreshes the generated facts and the index with them.",
     "tool_calls": [call("run_shell", {
         "command": "python3 -m harness map scaffold --project . --package alpha"})]},
    {"content": "Before finishing, what a handover has to say.",
     "tool_calls": [call("skill_load", {"name": "handoff"})]},
    {"content": "Closing the plan with what each step did.",
     "tool_calls": [call("todo_write", {"todos": closed(
         "read", "change", "map", change="alpha/core.py: Engine.stop returns helper('stop')",
         map="docs/map/alpha.md: Purpose names stop, scaffold refreshed the facts")})]},
    {"content": "Reporting as the handoff skill asks.",
     "tool_calls": [call("final_answer", {"status": "completed", "content": (
         "Branch task/e2e: Engine.stop added.\n"
         "Files: alpha/core.py (+3), docs/map/alpha.md (refreshed).\n"
         "Checks: Engine.stop returns a value; the map for alpha is current.\n"
         "Not done: beta was not touched.")})]},
]


def mapped_repo(path: Path) -> Path:
    """Two packages, a committed map with prose, and one task to do."""
    repo = git_repo(path, dict(FILES), message="a small project")
    harness(repo.parent, "map", "scaffold", "--project", str(repo))
    for package, purpose in (("alpha", "The engine: Engine.start is the whole of it."),
                             ("beta", "String helpers alpha leans on.")):
        map_file = repo / "docs" / "map" / f"{package}.md"
        write(map_file, map_file.read_text(encoding="utf-8").replace(
            "## Purpose", f"## Purpose\n\n{purpose}", 1))
    harness(repo.parent, "map", "scaffold", "--project", str(repo))   # the index reads them back
    commit_all(repo, "map the project")
    return repo


def harness(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    """The packaged CLI as a process, with the harness importable from the
    worktree too - the agent runs `map scaffold` in there."""
    proc = subprocess.run([sys.executable, "-m", "harness", *args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=180,
                          env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    assert proc.returncode in (0, 1), proc.stderr
    return proc


def test_a_whole_project_run_from_the_command_line(tmp_path):
    from harness.mockserver import MockOpenAIServer

    repo = mapped_repo(tmp_path / "proj")
    runs = tmp_path / "runs"
    base = git(repo, "rev-parse", "HEAD").strip()

    with MockOpenAIServer(SCRIPT, model="mock-model", api_key=SECRET) as server:
        proc = harness(tmp_path, "--runs-dir", str(runs), "run", "--project", str(repo),
                       "--task-id", "add-stop", "--run-id", "e2e", "--model", "mock-model",
                       "--endpoint", server.base_url, "--api-key", SECRET, "--test-command",
                       'python3 -c "import alpha; print(alpha.Engine().stop())"')
        system = server.requests[0]["body"]["messages"][0]["content"]
    assert proc.returncode == 0, proc.stderr
    assert "Branch task/e2e: Engine.stop added." in proc.stdout

    # what the model was told: the map index, and what this task calls finished
    assert "- alpha (inbound 0): The engine: Engine.start is the whole of it." in system
    assert "fs_read docs/map/INDEX.md for the rest." in system
    assert "- Engine.stop returns a value" in system and "- the map for alpha is current" in system

    records = read_trajectory(runs / "e2e")
    header, footer = records[0], last(records, "footer")
    assert header["project"]["task_id"] == "add-stop"
    assert header["skills"] == {"dirs": ["bundled"], "names": ["handoff", "map", "orient"]}
    assert [s["role"] for s in header["prompt_sources"]] == ["base", "project", "task"]
    steps = [(r["tool"], r["kind"]) for r in records if r["type"] == "step"]
    assert steps == [("skill_list", "ok"), ("skill_load", "ok"), ("toolbelt_add", "ok"),
                     ("todo_write", "ok"), ("fs_read", "ok"), ("fs_read", "ok"), ("fs_read", "ok"),
                     ("todo_write", "ok"), ("fs_edit", "ok"), ("fs_edit", "ok"),
                     ("run_shell", "ok"), ("skill_load", "ok"), ("todo_write", "ok"),
                     ("final_answer", "final_accepted")]

    # the work: in the worktree, on the branch, committed by the harness
    worktree = runs / "e2e" / "wt"
    assert "def stop(self):" in (worktree / "alpha" / "core.py").read_text(encoding="utf-8")
    assert "def stop(self):" not in (repo / "alpha" / "core.py").read_text(encoding="utf-8")
    assert git(worktree, "status", "--porcelain").strip() == ""
    assert git(worktree, "rev-parse", "HEAD~").strip() == base
    assert footer["commit"]["message"] == "harness: add-stop — completed"
    assert sorted(git(worktree, "show", "--name-only", "--format=").split()) == [
        "alpha/core.py", "docs/map/INDEX.md", "docs/map/alpha.md"]   # INDEX before alpha: ASCII
    assert footer["commit"]["files"] == 3

    # ... and what the harness checked about it
    assert footer["verify"]["test"]["passed"] is True
    assert "stop!" in (runs / "e2e" / footer["verify"]["test"]["artifact"]).read_text(encoding="utf-8")
    assert footer["verify"]["map_stale"] == {"area": ["alpha"], "stale": [], "clean": True}
    assert harness(tmp_path, "map", "stale", "--project", str(worktree)).returncode == 0

    task = load_task(repo, "add-stop")
    assert (task.status, task.branch, task.run_id) == ("done", "task/e2e", "e2e")
    assert task.prompt == "Give Engine a stop() method beside start(), using the same helper."

    # what a reader runs afterwards
    summary = harness(tmp_path, "--runs-dir", str(runs), "trace", "e2e", "--summary").stdout
    assert f"project: {repo}  branch task/e2e  from {base[:12]}  task add-stop" in summary
    assert "status: completed  steps: 14" in summary
    assert "verify: tests passed (exit 0," in summary and "map clean for alpha" in summary
    assert f"commit: {footer['commit']['sha'][:12]}  harness: add-stop — completed  (3 file(s))" in summary
    assert "skills: loaded orient, handoff  (3 discovered in 1 directory)" in summary
    assert "[completed] map: refresh the map for alpha" in summary
    assert "note: docs/map/alpha.md: Purpose names stop, scaffold refreshed the facts" in summary

    # the run is a record of itself: every skill it loaded is in its artifacts
    loaded = [r for r in records if r.get("tool") == "skill_load"]
    assert "Orient yourself in a mapped repository" in (
        runs / "e2e" / loaded[0]["artifact"]).read_text(encoding="utf-8")
    assert json.loads((runs / "e2e" / records[1]["artifact"]).read_text(encoding="utf-8")) == [
        {"name": "handoff", "description": "What final_answer must say on a project task: "
                                           "branch, files, map, tests, what is not done."},
        {"name": "map", "description": "Write the prose half of one package's map file: "
                                       "purpose, entry points, invariants, gotchas."},
        {"name": "orient", "description": "Find your way around a mapped repository before "
                                          "changing it: index, map files, then code."}]
