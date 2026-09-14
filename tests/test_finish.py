"""The finish hooks: what a project run does after the loop and before the footer.

Two things happen there, and both are about a claim nobody has checked yet. The
model said it was done: `verify` runs the project's tests and asks whether the
code map still describes what was changed, and puts both in the footer without
touching the run's status. Then `finish` commits the worktree on the task
branch, whatever ended the run - a run whose work is only on disk in a worktree
someone will prune is a run that did not happen.

Driven through the runtime, since that is where the hooks are called from, with
the CLI over the mock server for the flags that reach them.
"""
import os
import subprocess
import sys
from pathlib import Path

from harness.cli import main
from harness.codemap import scaffold
from harness.finish import COMMIT_MESSAGE, TEST_ARTIFACT, ProjectFinish, project_hooks
from harness.mockserver import MockOpenAIServer
from harness.project import start
from harness.registry import ToolRegistry
from harness.runtime import AgentRuntime, RunHooks
from harness.tasks import load as load_task
from harness.tools import register_default_tools, register_scratch_tools
from harness.trajectory import format_summary, last, read_trajectory
from harness.transport import FakeTransport, call

from .gitfixture import commit_all, git, sample_project, write

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "sekret"


def todo(status, id="work"):
    return call("todo_write", {"todos": [{"id": id, "content": "do the work", "status": status}]},
                id=f"todo_{status}")


FINISH = call("final_answer", {"status": "completed", "content": "one file written"}, id="fin")


def edit_script(path="NOTES.md", content="from the run\n"):
    """Activate the file tools, plan, write one file, close, finish."""
    return [
        {"content": "I need to write.",
         "tool_calls": [call("toolbelt_add", {"names": ["fs_write"]}, id="a")]},
        {"content": "Planning.", "tool_calls": [todo("in_progress")]},
        {"content": "Writing.", "tool_calls": [call("fs_write", {"path": path, "content": content},
                                                   id="w")]},
        {"content": "Closing.", "tool_calls": [todo("completed")]},
        {"content": "Reporting.", "tool_calls": [FINISH]},
    ]


def drive(repo, runs, script, run_id="finish", task_id=None, hooks=None, **flags):
    """One scripted run in a worktree of ``repo``, with the project's hooks."""
    project = start(repo, runs, run_id, task_id=task_id)
    registry = ToolRegistry()
    register_default_tools(registry, project.worktree)
    rt = AgentRuntime(registry, FakeTransport(script), runs, "fake", run_id=run_id,
                      project=project.header(),
                      hooks=hooks if hooks is not None else project_hooks(project, **flags))
    register_scratch_tools(registry, rt.run_dir)
    return project, rt.run("do the work")


def footer_of(run_dir):
    return last(read_trajectory(run_dir), "footer")


def mapped_project(tmp_path) -> Path:
    """The fixture project with a written, committed map: a clean starting point."""
    repo = sample_project(tmp_path / "p")
    scaffold(repo)
    for package in ("alpha", "alpha.nested", "beta"):
        path = repo / "docs" / "map" / f"{package}.md"
        write(path, path.read_text(encoding="utf-8").replace(
            "## Purpose", f"## Purpose\n\nWhat {package} is for.", 1))
    commit_all(repo, "map the project")
    return repo


# ---- verify ----------------------------------------------------------------


def test_the_footer_carries_the_test_result_and_the_map_check(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    # the empty quotes vanish in the shell, so what the command prints is a
    # string the command itself does not contain: the footer must not have it
    project, res = drive(repo, runs, edit_script(), test_command='echo the tests ra""n')

    verify = footer_of(res.run_dir)["verify"]
    assert verify["test"]["source"] == "--test-command"
    assert verify["test"]["command"] == 'echo the tests ra""n'
    assert verify["test"]["passed"] is True and verify["test"]["exit"] == 0
    assert verify["test"]["timed_out"] is False and verify["test"]["elapsed_ms"] >= 0
    assert verify["test"]["artifact"] == f"artifacts/{TEST_ARTIFACT}"
    assert "the tests ran" not in str(verify)                  # output is in the artifact only
    artifact = (res.run_dir / verify["test"]["artifact"]).read_text(encoding="utf-8")
    assert artifact.startswith('$ echo the tests ra""n\n[--test-command, exit 0,')
    assert "the tests ran" in artifact

    # NOTES.md is not a package, so nothing the run changed needs a map
    assert verify["map_stale"] == {"area": [], "stale": [], "clean": True}


def test_a_failing_command_is_recorded_and_does_not_change_the_status(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    _, res = drive(repo, runs, edit_script(),
                   test_command='echo the failure det""ail >&2; exit 3')

    assert res.status == "completed"                           # the model's word, not the tests'
    test = footer_of(res.run_dir)["verify"]["test"]
    assert test["passed"] is False and test["exit"] == 3
    assert "the failure detail" not in str(footer_of(res.run_dir))
    assert "the failure detail" in (res.run_dir / test["artifact"]).read_text(encoding="utf-8")


def test_a_command_that_hangs_is_killed_and_said_to_have_timed_out(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    _, res = drive(repo, runs, edit_script(), test_command="sleep 30", test_timeout=0.5)

    test = footer_of(res.run_dir)["verify"]["test"]
    assert test["timed_out"] is True and test["passed"] is False
    assert test["elapsed_ms"] < 20000                          # it did not wait for the sleep
    assert "[timed out after 0.5s]" in (res.run_dir / test["artifact"]).read_text(encoding="utf-8")


def test_the_projects_own_test_command_needs_trusting(tmp_path, capsys):
    """.harness/project.json is the project's code, like a skill's plugin."""
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"

    _, res = drive(repo, runs, edit_script(), run_id="untrusted")
    assert footer_of(res.run_dir)["verify"]["test"] is None
    err = capsys.readouterr().err
    assert ".harness/project.json names a test command" in err
    assert "--trust-project-plugins" in err
    assert not (res.run_dir / "artifacts" / TEST_ARTIFACT).exists()

    _, res = drive(repo, runs, edit_script(), run_id="trusted", trust_project=True)
    test = footer_of(res.run_dir)["verify"]["test"]
    assert test["source"] == ".harness/project.json"
    assert test["command"] == 'python3 -c "import sys; sys.exit(0)"' and test["passed"] is True

    # the operator's own command wins, and is never announced as untrusted
    _, res = drive(repo, runs, edit_script(), run_id="both", test_command="true")
    assert footer_of(res.run_dir)["verify"]["test"]["command"] == "true"
    assert ".harness/project.json" not in capsys.readouterr().err


def test_the_configured_timeout_is_the_projects_when_the_project_is_trusted(tmp_path):
    repo = sample_project(tmp_path / "p")
    runs = tmp_path / "runs"
    project = start(repo, runs, "timeouts")
    finish = ProjectFinish(project=project, trust_project=True)
    assert finish.test_source() == ('python3 -c "import sys; sys.exit(0)"',
                                    ".harness/project.json", 30.0)      # the fixture's timeout

    write(project.worktree / ".harness" / "project.json", '{"test": "true"}')
    assert finish.test_source() == ("true", ".harness/project.json", 600.0)
    write(project.worktree / ".harness" / "project.json", "not json at all")
    assert finish.test_source() == (None, None, 600.0)


def test_the_map_check_covers_the_task_area_and_says_when_it_is_stale(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    task = load_task(repo, "first")                            # area: [alpha]
    script = edit_script("alpha/core.py", "import os\n\n\ndef build(name):\n    return name\n")
    project, res = drive(repo, runs, script, run_id="stale", task_id=task.id, area=task.area)

    mapped = footer_of(res.run_dir)["verify"]["map_stale"]
    assert mapped["area"] == ["alpha"] and mapped["clean"] is False
    assert mapped["stale"] == [{"package": "alpha", "reasons": [
        "changed since the map was written: alpha/core.py"]}]

    # ... and the same run with the map refreshed in the worktree is clean
    scaffold(project.worktree, "alpha")
    assert ProjectFinish(project=project, area=task.area).map_report() == {
        "area": ["alpha"], "stale": [], "clean": True}


def test_without_a_task_the_check_covers_whatever_the_run_changed(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    script = edit_script("beta/util.py", "import json\n\n\ndef helper(sep):\n    return sep\n")
    project, res = drive(repo, runs, script, run_id="changed")

    mapped = footer_of(res.run_dir)["verify"]["map_stale"]
    assert mapped["area"] == ["beta"]                          # the package the file belongs to
    assert [e["package"] for e in mapped["stale"]] == ["beta"]
    assert ProjectFinish(project=project).changed_packages() == ["beta"]
    assert "beta/util.py" in ProjectFinish(project=project).changed_files()


def test_verify_runs_only_when_there_is_an_answer_to_verify(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    project = start(repo, runs, "capped")
    registry = ToolRegistry()
    register_default_tools(registry, project.worktree)
    rt = AgentRuntime(registry, FakeTransport(edit_script()), runs, "fake", run_id="capped",
                      project=project.header(),
                      hooks=project_hooks(project, test_command="echo never"))
    rt.config.step_cap = 3
    res = rt.run("do the work")

    assert res.status == "step_cap"
    footer = footer_of(res.run_dir)
    assert footer["verify"] is None                            # nothing claimed it was done
    assert not (res.run_dir / "artifacts" / TEST_ARTIFACT).exists()
    assert footer["commit"]["message"] == "harness: capped — step_cap"   # the work is still saved


# ---- auto-commit -----------------------------------------------------------


def test_a_finished_run_commits_its_worktree_on_the_task_branch(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    base = git(repo, "rev-parse", "HEAD").strip()
    project, res = drive(repo, runs, edit_script(), run_id="commits")

    commit = footer_of(res.run_dir)["commit"]
    assert commit["message"] == "harness: commits — completed"
    assert commit["message"] == COMMIT_MESSAGE.format(name="commits", status="completed")
    assert commit["files"] == 1
    assert commit["sha"] == git(project.worktree, "rev-parse", "HEAD").strip()
    assert git(project.worktree, "rev-parse", "HEAD~").strip() == base
    assert git(project.worktree, "status", "--porcelain").strip() == ""
    assert git(project.worktree, "show", "--name-only", "--format=%s").split() == [
        "harness:", "commits", "—", "completed", "NOTES.md"]
    assert git(repo, "rev-parse", "HEAD").strip() == base      # the project itself did not move


def test_the_task_id_names_the_commit_when_there_is_one(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    _, res = drive(repo, runs, edit_script(), run_id="named", task_id="first")
    assert footer_of(res.run_dir)["commit"]["message"] == "harness: first — completed"


def test_an_interrupted_run_commits_what_it_had(tmp_path):
    """Ctrl-C is exactly when losing the work would hurt."""
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    script = edit_script()[:3] + [KeyboardInterrupt()]
    project, res = drive(repo, runs, script, run_id="stopped")

    assert res.status == "interrupted"
    footer = footer_of(res.run_dir)
    assert footer["verify"] is None
    assert footer["commit"]["message"] == "harness: stopped — interrupted"
    assert footer["commit"]["files"] == 1
    assert git(project.worktree, "show", "--name-only", "--format=").split() == ["NOTES.md"]
    assert (project.worktree / "NOTES.md").is_file()


def test_nothing_to_commit_and_nothing_asked_for(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"

    quiet = [{"content": "Planning.", "tool_calls": [todo("completed")]},
             {"content": "Reporting.", "tool_calls": [FINISH]}]
    _, res = drive(repo, runs, quiet, run_id="clean")
    assert footer_of(res.run_dir)["commit"] is None            # a clean tree: nothing to say

    project, res = drive(repo, runs, edit_script(), run_id="asked-not-to", auto_commit=False)
    assert footer_of(res.run_dir)["commit"] is None
    assert git(project.worktree, "status", "--porcelain").strip() == "?? NOTES.md"


def test_a_hook_that_raises_costs_the_run_its_block_and_not_its_footer(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"

    def boom(runtime, *args):
        raise RuntimeError("git fell over")

    def stopped(runtime, *args):
        raise KeyboardInterrupt()

    _, res = drive(repo, runs, edit_script(), run_id="raises",
                   hooks=RunHooks(verify=boom, finish=stopped))
    footer = footer_of(res.run_dir)
    assert res.status == "completed" and footer["status"] == "completed"
    assert footer["verify"] == {"error": "RuntimeError: git fell over"}
    assert footer["commit"] == {"error": "interrupted"}
    assert "verify: did not run (RuntimeError: git fell over)" in format_summary(
        read_trajectory(res.run_dir))


# ---- through the command line ----------------------------------------------


def run_cli(runs, repo, server, run_id, *extra):
    return main(["--runs-dir", str(runs), "run", "--task", "work on the project",
                 "--model", "mock-model", "--endpoint", server.base_url, "--api-key", SECRET,
                 "--project", str(repo), "--run-id", run_id, *extra])


def test_the_flags_reach_the_hooks_and_the_summary_shows_both(tmp_path, capsys):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    with MockOpenAIServer(edit_script(), model="mock-model", api_key=SECRET) as server:
        assert run_cli(runs, repo, server, "flags", "--test-command", "echo checked") == 0

    records = read_trajectory(runs / "flags")
    footer = last(records, "footer")
    assert footer["verify"]["test"]["command"] == "echo checked"
    assert footer["commit"]["message"] == "harness: flags — completed"
    summary = format_summary(records)
    assert "verify: tests passed (exit 0," in summary and "map clean for (nothing changed)" in summary
    assert f"commit: {footer['commit']['sha'][:12]}  harness: flags — completed  (1 file(s))" in summary

    # what the run was told to do is recorded, so a resumed segment does the same
    invocation = records[0]["invocation"]
    assert invocation["test_command"] == "echo checked" and invocation["auto_commit"] is True
    assert invocation["test_timeout"] == 600.0


def test_no_keep_worktree_now_removes_a_finished_tree_because_it_was_committed(tmp_path, capsys):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    with MockOpenAIServer(edit_script(), model="mock-model", api_key=SECRET) as server:
        assert run_cli(runs, repo, server, "gone", "--no-keep-worktree") == 0

    assert not (runs / "gone" / "wt").exists()
    assert f"removed worktree {runs / 'gone' / 'wt'}" in capsys.readouterr().err
    assert git(repo, "show", "--name-only", "--format=", "task/gone").split() == ["NOTES.md"]


def test_a_resumed_segment_verifies_and_commits_too(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    with MockOpenAIServer(edit_script(), model="mock-model", api_key=SECRET) as server:
        assert run_cli(runs, repo, server, "twice", "--step-cap", "3",
                       "--test-command", "echo checked") == 3
    first = last(read_trajectory(runs / "twice"), "footer")
    assert first["verify"] is None and first["commit"]["message"] == "harness: twice — step_cap"

    rest = [{"content": "Closing.", "tool_calls": [todo("completed")]},
            {"content": "Writing again.",
             "tool_calls": [call("fs_write", {"path": "MORE.md", "content": "more\n"}, id="w2")]},
            {"content": "Reporting.", "tool_calls": [FINISH]}]
    with MockOpenAIServer(rest, model="mock-model", api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "resume", "twice", "--endpoint", server.base_url,
                   "--api-key", SECRET, "--step-cap", "20"])
    assert rc == 0

    second = last(read_trajectory(runs / "twice"), "footer")
    assert second["verify"]["test"]["command"] == "echo checked"    # the recorded command
    assert second["commit"]["message"] == "harness: twice — completed"
    assert second["commit"]["sha"] != first["commit"]["sha"]
    log = git(runs / "twice" / "wt", "log", "--format=%s").splitlines()
    assert log[:2] == ["harness: twice — completed", "harness: twice — step_cap"]


def test_the_packaged_cli_commits_what_a_real_process_did(tmp_path):
    repo = mapped_project(tmp_path)
    runs = tmp_path / "runs"
    with MockOpenAIServer(edit_script(), model="mock-model", api_key=SECRET) as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run",
             "--task", "work on the project", "--model", "mock-model",
             "--endpoint", server.base_url, "--api-key", SECRET, "--project", str(repo),
             "--run-id", "finish-e2e", "--test-command", "python3 -c \"import sys; sys.exit(1)\""],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    assert proc.returncode == 0, proc.stderr

    footer = last(read_trajectory(runs / "finish-e2e"), "footer")
    assert footer["verify"]["test"]["passed"] is False and footer["verify"]["test"]["exit"] == 1
    assert footer["commit"]["message"] == "harness: finish-e2e — completed"
    assert git(runs / "finish-e2e" / "wt", "log", "-1", "--format=%an").strip() == "harness"
