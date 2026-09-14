"""Projects: the worktree a run works in, and everything that guards it.

A project run is an ordinary run whose workdir is a git worktree the harness
cut for it. What is worth testing is the seam: the worktree is on the right
branch from the right commit, the trajectory says so, the git that would reach
outside the worktree is denied as a policy denial the model can read, and a
resume finds the same worktree - or says out loud that it had to cut a new one.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.cli import main
from harness.mockserver import MockOpenAIServer
from harness.project import (DENY_SHELL_DEFAULTS, ProjectError, dirty_entries, identity_args,
                             reattach, remove_worktree, start)
from harness.trajectory import format_summary, format_trace, read_trajectory
from harness.transport import call

from .gitfixture import branches, git, git_repo, head, sample_project, write

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "sekret"


def todo(status, id="work"):
    return call("todo_write", {"todos": [{"id": id, "content": "do the work", "status": status}]},
                id=f"todo_{status}")


FINISH = call("final_answer", {"status": "completed", "content": "branch task/<run>: one file written"},
              id="fin")


def project_script():
    """list -> add -> plan -> a denied push -> a real edit -> close -> final."""
    return [
        {"content": "Seeing what is here.", "tool_calls": [call("toolbelt_list", {"filter": "fs_"}, id="l")]},
        {"content": "I need to write and to shell out.",
         "tool_calls": [call("toolbelt_add", {"names": ["fs_write", "run_shell"]}, id="a")]},
        {"content": "Planning the change.", "tool_calls": [todo("in_progress")]},
        {"content": "Pushing the branch so it is safe.",
         "tool_calls": [call("run_shell", {"command": "git push origin HEAD"}, id="push")]},
        {"content": "Denied, and fine: I write the file instead.",
         "tool_calls": [call("fs_write", {"path": "NOTES.md", "content": "worktree only\n"}, id="w")]},
        {"content": "Done; closing the plan.", "tool_calls": [todo("completed")]},
        {"content": "Reporting.", "tool_calls": [FINISH]},
    ]


def paused_script():
    """Enough of a run to trip a step cap of 2 and leave it resumable."""
    return [
        {"content": "Activating the file tools.",
         "tool_calls": [call("toolbelt_add", {"names": ["fs_write", "fs_read"]}, id="a")]},
        {"content": "Planning.", "tool_calls": [todo("in_progress")]},
        {"content": "Writing the file.",
         "tool_calls": [call("fs_write", {"path": "NOTES.md", "content": "from the run\n"}, id="w")]},
        {"content": "Closing.", "tool_calls": [todo("completed")]},
        {"content": "Reporting.", "tool_calls": [FINISH]},
    ]


def run_project(runs, repo, server, run_id, *extra):
    return main(["--runs-dir", str(runs), "run", "--task", "work on the project",
                 "--model", "mock-model", "--endpoint", server.base_url, "--api-key", SECRET,
                 "--project", str(repo), "--run-id", run_id, *extra])


# ---- starting a run --------------------------------------------------------


def test_a_project_run_works_in_a_worktree_cut_from_head(tmp_path, capsys):
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    base = head(repo)

    with MockOpenAIServer(project_script(), model="mock-model", api_key=SECRET) as server:
        rc = run_project(runs, repo, server, "wt-run")
    assert rc == 0

    worktree = runs / "wt-run" / "wt"
    assert (worktree / "README.md").is_file() and (worktree / ".git").exists()
    assert (worktree / "NOTES.md").read_text() == "worktree only\n"     # the agent wrote in the worktree
    assert not (repo / "NOTES.md").exists()                             # never in the project
    assert git(worktree, "rev-parse", "--abbrev-ref", "HEAD").strip() == "task/wt-run"
    assert git(worktree, "rev-parse", "HEAD").strip() == base           # cut from HEAD
    assert "task/wt-run" in branches(repo) and head(repo) == base       # the project did not move

    header = read_trajectory(runs / "wt-run")[0]
    assert header["project"] == {"path": str(repo), "base_sha": base, "branch": "task/wt-run",
                                 "worktree": str(worktree), "task_id": None}
    assert header["invocation"]["workdir"] == str(worktree)
    assert header["invocation"]["project"] == str(repo)
    assert header["invocation"]["task_id"] is None
    assert f"project {repo}  branch task/wt-run" in capsys.readouterr().err

    # a reader of the run afterwards is told where the work landed
    records = read_trajectory(runs / "wt-run")
    assert f"project: {repo}  branch task/wt-run  from {base[:12]}" in format_summary(records)
    assert f"worktree {worktree}" in format_summary(records)
    assert f"project: {repo}  branch task/wt-run" in format_trace(records, runs / "wt-run")


def test_cli_project_run_over_http_end_to_end(tmp_path):
    """The packaged CLI, as a process, against the mock server: a real worktree,
    a real branch, a real denial."""
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    base = head(repo)
    with MockOpenAIServer(project_script(), model="mock-model", api_key=SECRET) as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run",
             "--task", "work on the project", "--model", "mock-model",
             "--endpoint", server.base_url, "--api-key", SECRET,
             "--project", str(repo), "--run-id", "proj-e2e"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))

    assert proc.returncode == 0, proc.stderr
    assert "one file written" in proc.stdout
    assert f"branch task/proj-e2e  at {base[:12]}" in proc.stderr

    worktree = runs / "proj-e2e" / "wt"
    assert (worktree / "NOTES.md").read_text() == "worktree only\n"
    assert git(worktree, "rev-parse", "--abbrev-ref", "HEAD").strip() == "task/proj-e2e"
    assert "task/proj-e2e" in branches(repo) and not (repo / "NOTES.md").exists()

    records = read_trajectory(runs / "proj-e2e")
    assert records[0]["project"] == {"path": str(repo), "base_sha": base,
                                     "branch": "task/proj-e2e", "worktree": str(worktree),
                                     "task_id": None}
    steps = [(r["tool"], r["kind"]) for r in records if r["type"] == "step"]
    assert steps == [("toolbelt_list", "ok"), ("toolbelt_add", "ok"), ("todo_write", "ok"),
                     ("run_shell", "denied"), ("fs_write", "ok"), ("todo_write", "ok"),
                     ("final_answer", "final_accepted")]


def test_the_default_git_denials_are_policy_the_model_reads(tmp_path):
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    with MockOpenAIServer(project_script(), model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "denied-run") == 0

    records = read_trajectory(runs / "denied-run")
    steps = [r for r in records if r["type"] == "step"]
    push = next(s for s in steps if s["tool"] == "run_shell")
    assert push["kind"] == "denied"
    assert "denied" in push["result_preview"] and "git" in push["result_preview"]
    assert records[0]["policy"]["deny_shell_patterns"] == list(DENY_SHELL_DEFAULTS)
    assert records[0]["policy"]["deny_tools"] == []
    # the run carried on after the denial and finished
    assert [s["kind"] for s in steps[-2:]] == ["ok", "final_accepted"]


def test_allow_git_lifts_the_defaults_and_explicit_denials_still_apply(tmp_path):
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    script = project_script()
    script[3] = {"content": "Asking git what it knows.",
                 "tool_calls": [call("run_shell", {"command": "git status --porcelain"}, id="st")]}
    with MockOpenAIServer(script, model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "allow-git", "--allow-git",
                           "--deny-shell-pattern", r"\brm\b") == 0

    records = read_trajectory(runs / "allow-git")
    shell = next(s for s in records if s.get("tool") == "run_shell")
    assert shell["kind"] == "ok"                                  # git ran for real
    assert records[0]["policy"]["deny_shell_patterns"] == [r"\brm\b"]


def test_project_and_workdir_are_mutually_exclusive(tmp_path, capsys):
    repo = git_repo(tmp_path / "project")
    rc = main(["--runs-dir", str(tmp_path / "runs"), "run", "--task", "t", "--model", "m",
               "--endpoint", "http://localhost:1", "--project", str(repo),
               "--workdir", str(tmp_path / "elsewhere")])
    assert rc == 64
    assert "--project and --workdir are mutually exclusive" in capsys.readouterr().err


def test_a_project_that_is_not_a_repository_is_a_usage_error(tmp_path, capsys):
    plain = tmp_path / "plain"
    plain.mkdir()
    rc = main(["--runs-dir", str(tmp_path / "runs"), "run", "--task", "t", "--model", "m",
               "--endpoint", "http://localhost:1", "--project", str(plain)])
    assert rc == 64 and "not a git repository" in capsys.readouterr().err
    rc = main(["--runs-dir", str(tmp_path / "runs"), "run", "--task", "t", "--model", "m",
               "--endpoint", "http://localhost:1", "--project", str(tmp_path / "nope")])
    assert rc == 64 and "no such directory" in capsys.readouterr().err


# ---- the dirty rules -------------------------------------------------------


def test_a_dirty_project_refuses_and_an_uncommitted_map_says_why(tmp_path, capsys):
    repo = sample_project(tmp_path / "project")
    write(repo / "alpha" / "core.py", "import os\n")                 # tracked, modified
    write(repo / "docs" / "map" / "alpha.md", "---\nmap_version: 1\n---\n\n## Purpose\n")

    rc = main(["--runs-dir", str(tmp_path / "runs"), "run", "--task", "t", "--model", "m",
               "--endpoint", "http://localhost:1", "--project", str(repo)])
    err = capsys.readouterr().err
    assert rc == 64
    assert "has uncommitted changes" in err
    assert "alpha/core.py" in err and "docs/map/" in err
    assert "the worktree is cut from HEAD, so the agent would run without the map" in err
    assert not (tmp_path / "runs").exists()          # nothing was cut
    assert branches(repo) == ["main"]


def test_changes_under_tasks_never_make_a_project_dirty(tmp_path):
    repo = sample_project(tmp_path / "project")
    write(repo / "tasks" / "first.md", "---\nstatus: doing\n---\nMake the engine start.\n")
    write(repo / "tasks" / "third.md", "---\nstatus: todo\n---\nA new one.\n")
    assert dirty_entries(repo) == []

    runs = tmp_path / "runs"
    with MockOpenAIServer(project_script(), model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "tasks-move") == 0
    assert (runs / "tasks-move" / "wt" / "NOTES.md").is_file()


def test_allow_dirty_starts_anyway_from_head(tmp_path):
    repo = sample_project(tmp_path / "project")
    write(repo / "alpha" / "core.py", "import os\nNEW = 1\n")
    runs = tmp_path / "runs"
    with MockOpenAIServer(project_script(), model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "dirty-ok", "--allow-dirty") == 0
    # the worktree is HEAD's copy, not the operator's edit
    assert "NEW = 1" not in (runs / "dirty-ok" / "wt" / "alpha" / "core.py").read_text()


# ---- keeping, listing and pruning worktrees --------------------------------


def test_no_keep_worktree_removes_a_clean_tree_and_keeps_the_branch(tmp_path, capsys):
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    script = project_script()
    script[4] = {"content": "Nothing to write, actually.",
                 "tool_calls": [call("run_shell", {"command": "true"}, id="noop")]}
    with MockOpenAIServer(script, model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "gone", "--no-keep-worktree") == 0

    assert not (runs / "gone" / "wt").exists()
    assert "task/gone" in branches(repo)
    assert f"removed worktree {runs / 'gone' / 'wt'}" in capsys.readouterr().err
    assert (runs / "gone" / "trajectory.jsonl").is_file()          # the record stays


def test_no_keep_worktree_leaves_a_dirty_tree_alone(tmp_path, capsys):
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    with MockOpenAIServer(project_script(), model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "kept", "--no-keep-worktree") == 0
    err = capsys.readouterr().err
    assert (runs / "kept" / "wt" / "NOTES.md").is_file()           # the agent's work is still there
    assert "has uncommitted changes; left in place" in err


def test_worktree_list_and_prune(tmp_path, capsys):
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    with MockOpenAIServer(project_script(), model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "done-dirty") == 0
    with MockOpenAIServer(paused_script(), model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "paused", "--step-cap", "2") == 3
    # a finished run whose worktree is clean: nothing to lose by removing it
    with MockOpenAIServer([{"content": "Planning.", "tool_calls": [todo("completed")]},
                           {"content": "Reporting.", "tool_calls": [FINISH]}],
                          model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "done-clean") == 0

    assert main(["--runs-dir", str(runs), "worktree", "list", "--project", str(repo)]) == 0
    out = capsys.readouterr().out
    rows = {cells[0]: cells for cells in (line.split() for line in out.splitlines()[1:]) if cells}
    assert rows["done-clean"][:4] == ["done-clean", "completed", "task/done-clean", "clean"]
    assert rows["done-dirty"][:4] == ["done-dirty", "completed", "task/done-dirty", "dirty"]
    assert rows["paused"][:4] == ["paused", "step_cap", "task/paused", "clean"]
    assert rows["paused"][4] == str(runs / "paused" / "wt")

    assert main(["--runs-dir", str(runs), "worktree", "prune", "--project", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "removed done-clean  completed: branch task/done-clean kept" in out
    assert "kept    done-dirty  completed: uncommitted changes" in out
    assert "kept    paused      step_cap: not finished" in out
    assert not (runs / "done-clean" / "wt").exists()
    assert (runs / "done-dirty" / "wt" / "NOTES.md").is_file() and (runs / "paused" / "wt").is_dir()

    assert main(["--runs-dir", str(runs), "worktree", "prune", "--project", str(repo), "--force"]) == 0
    assert "removed done-dirty" in capsys.readouterr().out
    assert not (runs / "done-dirty" / "wt").exists()
    assert (runs / "paused" / "wt").is_dir()                        # still resumable, still there
    assert sorted(branches(repo)) == ["main", "task/done-clean", "task/done-dirty", "task/paused"]


def test_worktree_list_on_something_that_is_not_a_repository(tmp_path, capsys):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert main(["--runs-dir", str(tmp_path / "runs"), "worktree", "list", "--project", str(plain)]) == 64
    assert "not a git repository" in capsys.readouterr().err


# ---- resuming --------------------------------------------------------------


def test_resume_continues_in_the_same_worktree(tmp_path, capsys):
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    with MockOpenAIServer(paused_script(), model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "resumed", "--step-cap", "2") == 3
    worktree = runs / "resumed" / "wt"
    write(worktree / "left-behind.txt", "still here\n")            # uncommitted work in the tree

    rest = [
        {"content": "Writing the file.",
         "tool_calls": [call("fs_write", {"path": "NOTES.md", "content": "from the resume\n"}, id="w")]},
        {"content": "Reading what was left behind.",
         "tool_calls": [call("fs_read", {"path": "left-behind.txt"}, id="r")]},
        {"content": "Closing.", "tool_calls": [todo("completed")]},
        {"content": "Reporting.", "tool_calls": [FINISH]},
    ]
    with MockOpenAIServer(rest, model="mock-model", api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "resume", "resumed", "--endpoint", server.base_url,
                   "--api-key", SECRET, "--step-cap", "20"])
        messages = server.requests[0]["body"]["messages"]
    assert rc == 0

    steps = [r for r in read_trajectory(runs / "resumed") if r["type"] == "step"]
    assert "still here" in steps[-3]["result_preview"]              # the same tree, untouched
    assert (worktree / "NOTES.md").read_text() == "from the resume\n"
    seam = next(r for r in read_trajectory(runs / "resumed") if r["type"] == "resume")
    assert "worktree was recreated" not in seam["note"]
    assert "worktree was recreated" not in messages[-1]["content"]
    assert seam["invocation"]["project"] == str(repo)              # carried forward for the next one
    assert seam["invocation"]["workdir"] == str(worktree)


def test_resume_after_the_worktree_is_gone_cuts_a_new_one_and_says_so(tmp_path, capsys):
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    with MockOpenAIServer(paused_script(), model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "pruned", "--step-cap", "2") == 3
    worktree = runs / "pruned" / "wt"

    # prune leaves it: the run can still be resumed, so its tree is not rubbish
    assert main(["--runs-dir", str(runs), "worktree", "prune", "--project", str(repo)]) == 0
    assert "kept    pruned" in capsys.readouterr().out and worktree.is_dir()

    # an operator who cleaned it up by hand, though, loses whatever was in it
    write(worktree / "left-behind.txt", "will not survive\n")
    remove_worktree(repo, worktree, force=True)
    assert not worktree.exists()

    rest = [
        {"content": "Writing the file again.",
         "tool_calls": [call("fs_write", {"path": "NOTES.md", "content": "after the recreate\n"}, id="w")]},
        {"content": "Closing.", "tool_calls": [todo("completed")]},
        {"content": "Reporting.", "tool_calls": [FINISH]},
    ]
    with MockOpenAIServer(rest, model="mock-model", api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "resume", "pruned", "--endpoint", server.base_url,
                   "--api-key", SECRET, "--step-cap", "20"])
        messages = server.requests[0]["body"]["messages"]
    assert rc == 0

    sha = git(repo, "rev-parse", "task/pruned").strip()
    expected = (f"worktree was recreated from branch task/pruned at {sha}; uncommitted changes "
                "from the pruned worktree are gone.")
    seam = next(r for r in read_trajectory(runs / "pruned") if r["type"] == "resume")
    assert expected in seam["from_detail"] and expected in seam["note"]
    assert expected in messages[-1]["content"]                     # the model was told
    assert worktree.is_dir() and not (worktree / "left-behind.txt").exists()
    assert (worktree / "NOTES.md").read_text() == "after the recreate\n"
    assert git(worktree, "rev-parse", "--abbrev-ref", "HEAD").strip() == "task/pruned"


def test_resume_refuses_when_the_branch_is_gone_too(tmp_path, capsys):
    repo = sample_project(tmp_path / "project")
    runs = tmp_path / "runs"
    with MockOpenAIServer(paused_script(), model="mock-model", api_key=SECRET) as server:
        assert run_project(runs, repo, server, "orphan", "--step-cap", "2") == 3
    remove_worktree(repo, runs / "orphan" / "wt", force=True)
    git(repo, "branch", "-D", "task/orphan")

    rc = main(["--runs-dir", str(runs), "resume", "orphan", "--endpoint", "http://localhost:1",
               "--step-cap", "20"])
    err = capsys.readouterr().err
    assert rc == 64
    assert "branch task/orphan is gone" in err and "cannot be resumed" in err
    assert [r["type"] for r in read_trajectory(runs / "orphan")].count("resume") == 0


# ---- the library underneath ------------------------------------------------


def test_start_refuses_a_branch_that_already_exists(tmp_path):
    repo = git_repo(tmp_path / "project")
    runs = tmp_path / "runs"
    start(repo, runs, "twice")
    with pytest.raises(ProjectError, match="branch task/twice already exists"):
        start(repo, runs, "twice")


def test_identity_args_only_when_the_repository_has_none(tmp_path):
    repo = git_repo(tmp_path / "project")
    assert identity_args(repo) == ["-c", "user.name=harness", "-c", "user.email=harness@local"]
    git(repo, "config", "user.name", "Someone")
    git(repo, "config", "user.email", "someone@example.com")
    assert identity_args(repo) == []


def test_reattach_reuses_a_worktree_and_refuses_a_project_that_is_gone(tmp_path):
    repo = git_repo(tmp_path / "project")
    run = start(repo, tmp_path / "runs", "reattached")
    assert reattach(run.header()) == (run.worktree, None)

    block = dict(run.header(), path=str(tmp_path / "nowhere"))
    remove_worktree(repo, run.worktree)
    with pytest.raises(ProjectError, match="is gone; nothing to recreate"):
        reattach(block)
