"""Tasks: the file a project keeps, and what a run does to it.

A task is the one thing a project run writes outside its worktree, so what
matters is that the writing is exact (the prompt survives byte for byte, the
frontmatter says who took the task) and that the status a run leaves behind is
the truth: done when it finished, blocked when it could not, and still `doing`
while a run that can be resumed is unfinished.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.cli import main
from harness.mockserver import MockOpenAIServer, chat_completion_payload
from harness.tasks import (TaskError, close, format_task, load, load_all, load_file, next_task,
                           tasks_dir)
from harness.trajectory import read_trajectory
from harness.transport import call

from .gitfixture import git_repo, sample_project, write

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "sekret"


def todo(status, id="work"):
    return call("todo_write", {"todos": [{"id": id, "content": "do the work", "status": status}]},
                id=f"todo_{status}")


def finish(status="completed"):
    return call("final_answer", {"status": status, "content": "the engine starts"}, id="fin")


def plan_and_finish(status="completed"):
    """The shortest run that ends in ``status``."""
    return [{"content": "Planning.", "tool_calls": [todo("in_progress")]},
            {"content": "Done.", "tool_calls": [todo("completed")]},
            {"content": "Reporting.", "tool_calls": [finish(status)]}]


def run_task(runs, repo, server, run_id, task_id="first", *extra):
    return main(["--runs-dir", str(runs), "run", "--model", "mock-model",
                 "--endpoint", server.base_url, "--api-key", SECRET,
                 "--project", str(repo), "--task-id", task_id, "--run-id", run_id, *extra])


def status_of(repo, task_id="first"):
    return load(repo, task_id).status


# ---- the file --------------------------------------------------------------


def test_a_task_is_frontmatter_and_a_prompt(tmp_path):
    repo = sample_project(tmp_path / "p")
    task = load(repo, "first")
    assert task.id == "first" and task.status == "todo"
    assert task.area == ["alpha"]
    assert task.done_when == ["alpha starts the engine", "the map for alpha is current"]
    assert task.branch is None and task.run_id is None      # `null` is not set, not a string
    assert task.prompt == "Make the engine start."
    assert task.brief() == {"id": "first", "status": "todo", "area": ["alpha"],
                            "branch": None, "run_id": None}

    ids = [t.id for t in load_all(repo)[0]]
    assert ids == ["first", "second"]                       # by id, whatever the directory says
    assert next_task(repo).id == "first"


def test_marking_a_task_keeps_the_body_and_the_key_order(tmp_path):
    repo = sample_project(tmp_path / "p")
    task = load(repo, "first")
    body = task.body

    marked = task.mark("doing", branch="task/r1", run_id="r1")
    text = task.path.read_text(encoding="utf-8")
    assert text == ("---\n"
                    "status: doing\n"
                    "area: [alpha]\n"
                    "done_when: [alpha starts the engine, the map for alpha is current]\n"
                    "branch: task/r1\n"
                    "run_id: r1\n"
                    "---\n\nMake the engine start.\n")
    assert marked.body == body and marked.prompt == "Make the engine start."
    assert marked.status == "doing" and marked.branch == "task/r1" and marked.run_id == "r1"

    # a key the harness knows nothing about is kept, where it was
    write(task.path, "---\nowner: someone\nstatus: todo\n---\n\nA prompt.\n")
    load_file(task.path).mark("done")
    assert task.path.read_text(encoding="utf-8") == (
        "---\nowner: someone\nstatus: done\narea: []\ndone_when: []\nbranch: null\n"
        "run_id: null\n---\n\nA prompt.\n")


def test_a_task_file_that_cannot_be_used_says_why(tmp_path):
    repo = sample_project(tmp_path / "p")
    write(tasks_dir(repo) / "broken.md", "just prose, no fence\n")
    write(tasks_dir(repo) / "odd.md", "---\nstatus: wandering\n---\nprompt\n")

    found, errors = load_all(repo)
    assert [t.id for t in found] == ["first", "second"]
    assert any("broken.md: a task file needs a '---' frontmatter block" in e for e in errors)
    assert any("odd.md: status 'wandering' is not one of todo, doing, done, blocked" in e
               for e in errors)
    with pytest.raises(TaskError, match="no task 'nope'"):
        load(repo, "nope")


def test_close_maps_a_run_status_onto_the_task(tmp_path):
    repo = sample_project(tmp_path / "p")
    task = load(repo, "first").mark("doing", branch="task/r", run_id="r")
    assert close(task, "step_cap") is None and status_of(repo) == "doing"
    assert close(task, "interrupted") is None and status_of(repo) == "doing"
    assert close(task, "failed") == "task first: doing -> blocked"
    assert close(load(repo, "first"), "completed") == "task first: blocked -> done"
    assert status_of(repo) == "done"


# ---- the command line ------------------------------------------------------


def test_tasks_list_next_and_show(tmp_path, capsys):
    repo = sample_project(tmp_path / "p")
    load(repo, "first").mark("doing", branch="task/r1", run_id="r1")

    assert main(["tasks", "list", "--project", str(repo)]) == 0
    out = capsys.readouterr()
    assert out.out.splitlines() == ["id      status  area   branch",
                                    "first   doing   alpha  task/r1",
                                    "second  todo    beta   -"]
    assert "2 task(s)" in out.err

    assert main(["tasks", "next", "--project", str(repo)]) == 0
    out = capsys.readouterr().out
    assert out == "second\nMake the helper helpful.\n"      # first is taken

    assert main(["tasks", "show", "second", "--project", str(repo)]) == 0
    shown = capsys.readouterr().out
    assert shown.startswith("task second  [todo]  ")
    assert "area: beta" in shown and "  - the helper is helpful" in shown
    assert shown.endswith("Make the helper helpful.\n")
    assert format_task(load(repo, "second")).endswith("Make the helper helpful.")

    assert main(["tasks", "show", "third", "--project", str(repo)]) == 64
    assert "no task 'third'" in capsys.readouterr().err


def test_tasks_next_exits_1_when_there_is_nothing_to_do(tmp_path, capsys):
    repo = sample_project(tmp_path / "p")
    for task_id in ("first", "second"):
        load(repo, task_id).mark("done")
    assert main(["tasks", "next", "--project", str(repo)]) == 1
    assert "no task with status 'todo'" in capsys.readouterr().err

    plain = git_repo(tmp_path / "plain")                   # a repository with no tasks/ at all
    assert main(["tasks", "list", "--project", str(plain)]) == 0
    assert "no tasks in" in capsys.readouterr().err


# ---- a run on a task -------------------------------------------------------


def test_the_task_is_claimed_before_the_first_request(tmp_path, capsys):
    """The project's copy says `doing`, with the branch and the run, by the
    time the model is asked anything - so a second run can see it is taken."""
    repo = sample_project(tmp_path / "p")
    runs = tmp_path / "runs"
    seen = {}

    def watch(_body):
        seen["file"] = (tasks_dir(repo) / "first.md").read_text(encoding="utf-8")
        return chat_completion_payload({"content": "Planning.", "tool_calls": [todo("in_progress")]},
                                       "mock-model")

    with MockOpenAIServer([watch], model="mock-model", api_key=SECRET) as server:
        assert run_task(runs, repo, server, "claim", "first", "--step-cap", "1") == 3

    assert "status: doing" in seen["file"]
    assert "branch: task/claim" in seen["file"] and "run_id: claim" in seen["file"]
    assert "Make the engine start." in seen["file"]        # the prompt is untouched
    err = capsys.readouterr().err
    assert f"task first: doing  ({tasks_dir(repo) / 'first.md'})" in err
    assert "task left 'doing': the run ended as step_cap" in err

    header = read_trajectory(runs / "claim")[0]
    assert header["task"] == "Make the engine start."      # the body is the task
    assert header["project"]["task_id"] == "first"
    assert header["invocation"]["task_id"] == "first"
    assert status_of(repo) == "doing"                      # still taken: it can be resumed


def test_a_resumed_run_closes_the_task_it_finishes(tmp_path, capsys):
    repo = sample_project(tmp_path / "p")
    runs = tmp_path / "runs"
    with MockOpenAIServer(plan_and_finish(), model="mock-model", api_key=SECRET) as server:
        assert run_task(runs, repo, server, "paused", "first", "--step-cap", "1") == 3
    assert status_of(repo) == "doing"

    with MockOpenAIServer(plan_and_finish(), model="mock-model", api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "resume", "paused", "--endpoint", server.base_url,
                   "--api-key", SECRET, "--step-cap", "20"])
    assert rc == 0
    assert status_of(repo) == "done"
    assert "task first: doing -> done" in capsys.readouterr().err


@pytest.mark.parametrize("final,expected,code", [("completed", "done", 0),
                                                 ("failed", "blocked", 1),
                                                 ("blocked", "blocked", 1)])
def test_the_status_a_run_ends_with_is_the_status_the_task_gets(tmp_path, final, expected, code):
    repo = sample_project(tmp_path / "p")
    runs = tmp_path / "runs"
    with MockOpenAIServer(plan_and_finish(final), model="mock-model", api_key=SECRET) as server:
        assert run_task(runs, repo, server, f"end-{final}", "first") == code
    assert status_of(repo) == expected
    assert load(repo, "first").branch == f"task/end-{final}"   # what to go and look at


def test_a_task_that_is_already_doing_is_refused_and_force_takes_it(tmp_path, capsys):
    repo = sample_project(tmp_path / "p")
    runs = tmp_path / "runs"
    with MockOpenAIServer(plan_and_finish(), model="mock-model", api_key=SECRET) as server:
        assert run_task(runs, repo, server, "first-run", "first", "--step-cap", "1") == 3

        rc = run_task(runs, repo, server, "second-run", "first")
        err = capsys.readouterr().err
        assert rc == 64
        assert "task first is already 'doing' (branch task/first-run, run first-run)" in err
        assert "--force to take it over" in err
        assert not (runs / "second-run").exists()          # nothing was cut, nothing was written

    with MockOpenAIServer(plan_and_finish(), model="mock-model", api_key=SECRET) as server:
        assert run_task(runs, repo, server, "forced", "first", "--force") == 0
    assert status_of(repo) == "done" and load(repo, "first").run_id == "forced"


def test_the_command_line_needs_exactly_one_task(tmp_path, capsys):
    repo = sample_project(tmp_path / "p")
    runs = tmp_path / "runs"
    base = ["--runs-dir", str(runs), "run", "--model", "m", "--endpoint", "http://localhost:1"]

    assert main(base + ["--project", str(repo), "--task-id", "first", "--task", "t"]) == 64
    assert "--task and --task-id given; a run has one task" in capsys.readouterr().err

    assert main(base + ["--task-id", "first"]) == 64
    assert "--task-id needs --project" in capsys.readouterr().err

    assert main(base + ["--project", str(repo), "--task-id", "nope"]) == 64
    err = capsys.readouterr().err
    assert "no task 'nope'" in err and "Tasks: first, second" in err

    assert main(base + ["--project", str(repo)]) == 64
    assert "need --task, --task-file or --task-id" in capsys.readouterr().err
    assert not runs.exists()


def test_the_packaged_cli_runs_a_task_as_a_process(tmp_path):
    repo = sample_project(tmp_path / "p")
    runs = tmp_path / "runs"
    with MockOpenAIServer(plan_and_finish(), model="mock-model", api_key=SECRET) as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run",
             "--model", "mock-model", "--endpoint", server.base_url, "--api-key", SECRET,
             "--project", str(repo), "--task-id", "second", "--run-id", "task-e2e"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    assert proc.returncode == 0, proc.stderr
    assert "task second: doing -> done" in proc.stderr
    task = load(repo, "second")
    assert (task.status, task.branch, task.run_id) == ("done", "task/task-e2e", "task-e2e")
    assert read_trajectory(runs / "task-e2e")[0]["task"] == "Make the helper helpful."
