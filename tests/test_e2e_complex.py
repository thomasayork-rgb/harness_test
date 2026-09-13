"""The whole point, end to end: a complex task tracked to completion with
todos and skills.

One `python -m harness run` as a real process against the mock server. The
model lists the skills it has, loads two of them, plans four todos, does the
work with the file tools, records the outcome of each item in its notes,
ignores its plan long enough to be nudged, brings the plan back up to date and
finishes. The trajectory is asserted step for step, the footer todos with their
notes, and the trace --summary a reader would run afterwards.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from harness.cli import main
from harness.trajectory import read_trajectory, summarize
from harness.transport import call

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples" / "skills"

TASK = ("The README says this service listens on port 9090 but the code says otherwise. "
        "Find the port the service really uses, fix the README, and check the project still "
        "passes its own check.")

CONFIG = "[server]\nPORT = 8080\nhost = 0.0.0.0\n"
SERVER = 'import configparser\n\nPORT = 8080\n\n\ndef serve():\n    return PORT\n'
README = "# demo\n\nThe demo service listens on port 9090.\nRun ./check.py before committing.\n"
CHECK = (
    "import configparser, pathlib, re, sys\n"
    "cfg = configparser.ConfigParser()\n"
    "cfg.read('app/config.ini')\n"
    "code = re.search(r'PORT = (\\d+)', pathlib.Path('app/server.py').read_text()).group(1)\n"
    "doc = re.search(r'port (\\d+)', pathlib.Path('README.md').read_text()).group(1)\n"
    "ok = cfg['server']['PORT'] == code == doc\n"
    "print('check:', 'ok' if ok else f'mismatch cfg={cfg[\"server\"][\"PORT\"]} code={code} doc={doc}')\n"
    "sys.exit(0 if ok else 1)\n"
)


def project(tmp_path):
    work = tmp_path / "demo"
    (work / "app").mkdir(parents=True)
    (work / "app" / "config.ini").write_text(CONFIG, encoding="utf-8")
    (work / "app" / "server.py").write_text(SERVER, encoding="utf-8")
    (work / "README.md").write_text(README, encoding="utf-8")
    (work / "check.py").write_text(CHECK, encoding="utf-8")
    return work


def skills(tmp_path):
    """Fixture copies of two shipped examples, so the test is about them too."""
    root = tmp_path / "skills"
    root.mkdir()
    for name in ("investigate", "code-change"):
        shutil.copytree(EXAMPLES / name, root / name)
    return root


def todos(*items):
    """(id, content, status[, notes]) per item."""
    out = []
    for item in items:
        todo = {"id": item[0], "content": item[1], "status": item[2]}
        if len(item) > 3:
            todo["notes"] = item[3]
        out.append(todo)
    return call("todo_write", {"todos": out})


PLAN = (("find", "find the port the service really uses", "in_progress"),
        ("fix", "fix the port in README.md", "pending"),
        ("check", "run ./check.py", "pending"),
        ("report", "report the port, the edit and the check", "pending"))


def script():
    return [
        {"content": "A multi-part task; checking for a skill that covers this kind of work first.",
         "tool_calls": [call("skill_list", {})],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"content": "investigate is exactly the first half. Loading it.",
         "tool_calls": [call("skill_load", {"name": "investigate"})],
         "usage": {"prompt_tokens": 200, "completion_tokens": 10}},
        {"content": "The skill says shape first, then narrow, then read. Planning the four steps.",
         "tool_calls": [todos(*PLAN)],
         "usage": {"prompt_tokens": 300, "completion_tokens": 30}},
        {"content": "Shape of the tree first, as the skill says.",
         "tool_calls": [call("fs_glob", {"pattern": "**/*.py"})],
         "usage": {"prompt_tokens": 400, "completion_tokens": 12}},
        {"content": "Now the exact text, in the config and in the code.",
         "tool_calls": [call("fs_search", {"pattern": "(?i)port"})],
         "usage": {"prompt_tokens": 500, "completion_tokens": 12}},
        {"content": "Reading the config before I quote it.",
         "tool_calls": [call("fs_read", {"path": "app/config.ini"})],
         "usage": {"prompt_tokens": 600, "completion_tokens": 12}},
        {"content": "8080 everywhere but the README. Recording that and starting the edit.",
         "tool_calls": [todos(("find", "find the port the service really uses", "completed",
                               "8080: app/config.ini:2 and app/server.py:3; README says 9090"),
                              ("fix", "fix the port in README.md", "in_progress"))],
         "usage": {"prompt_tokens": 700, "completion_tokens": 40}},
        {"content": "The edit is a different kind of work; loading the skill for it.",
         "tool_calls": [call("skill_load", {"name": "code-change"})],
         "usage": {"prompt_tokens": 800, "completion_tokens": 12}},
        # five steps that never touch the plan: the nudge lands after this run
        {"content": "Reading the README before editing it.",
         "tool_calls": [call("fs_read", {"path": "README.md"})],
         "usage": {"prompt_tokens": 900, "completion_tokens": 12}},
        {"content": "One exact replacement, with enough context to be unique.",
         "tool_calls": [call("fs_edit", {"path": "README.md", "old": "listens on port 9090",
                                         "new": "listens on port 8080"})],
         "usage": {"prompt_tokens": 1000, "completion_tokens": 20}},
        {"content": "Saving the evidence where eviction cannot reach it.",
         "tool_calls": [call("scratch_write", {"name": "port.md",
                                               "content": "port 8080; README fixed"})],
         "usage": {"prompt_tokens": 1100, "completion_tokens": 14}},
        {"content": "Running the project's own check, from its root.",
         "tool_calls": [call("run_shell", {"command": "python3 check.py"})],
         "usage": {"prompt_tokens": 1200, "completion_tokens": 14}},
        # ... and here the harness asks for the plan back
        {"content": "Fair: two of those are done. Bringing the plan up to date.",
         "tool_calls": [todos(("fix", "fix the port in README.md", "completed",
                               "README.md:3 9090 -> 8080, one line in the diff"),
                              ("check", "run ./check.py", "completed",
                               "python3 check.py -> exit 0, 'check: ok'"),
                              ("report", "report the port, the edit and the check", "in_progress"))],
         "usage": {"prompt_tokens": 1300, "completion_tokens": 50}},
        {"content": "Done with the investigate guide; freeing the context it takes.",
         "tool_calls": [call("skill_unload", {"name": "investigate"})],
         "usage": {"prompt_tokens": 1400, "completion_tokens": 12}},
        {"content": "Writing the report closes the last item.",
         "tool_calls": [todos(("report", "report the port, the edit and the check", "completed",
                               "answered in final_answer"))],
         "usage": {"prompt_tokens": 1500, "completion_tokens": 20}},
        {"content": "Every todo is closed; answering.",
         "tool_calls": [call("final_answer", {"status": "completed", "content": (
             "The service listens on 8080 (app/config.ini:2, app/server.py:3). README.md line 3 "
             "said 9090 and now says 8080. python3 check.py exits 0 and prints 'check: ok'.")})],
         "usage": {"prompt_tokens": 1600, "completion_tokens": 60}},
    ]


EXPECTED = [
    (1, "skill_list", "ok"),
    (2, "skill_load", "ok"),
    (3, "todo_write", "ok"),
    (4, "fs_glob", "ok"),
    (5, "fs_search", "ok"),
    (6, "fs_read", "ok"),
    (7, "todo_write", "ok"),
    (8, "skill_load", "ok"),
    (9, "fs_read", "ok"),
    (10, "fs_edit", "ok"),
    (11, "scratch_write", "ok"),
    (12, "run_shell", "ok"),
    (13, "todo_write", "ok"),
    (14, "skill_unload", "ok"),
    (15, "todo_write", "ok"),
    (16, "final_answer", "final_accepted"),
]


def test_a_complex_task_tracked_to_completion_with_todos_and_skills(tmp_path, capsys):
    from harness.mockserver import MockOpenAIServer

    work, skill_dir, runs = project(tmp_path), skills(tmp_path), tmp_path / "runs"
    with MockOpenAIServer(script(), model="mock-model") as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run", "--task", TASK,
             "--model", "mock-model", "--endpoint", server.base_url, "--workdir", str(work),
             "--skills", str(skill_dir), "--run-id", "owner", "--progress-nudge", "5"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
        bodies = [r["body"] for r in server.requests]

    assert proc.returncode == 0, proc.stderr
    assert "The service listens on 8080" in proc.stdout
    assert "status: completed  steps: 16" in proc.stderr

    records = read_trajectory(runs / "owner")
    steps = [r for r in records if r["type"] == "step"]
    assert [(s["step"], s["tool"], s["kind"]) for s in steps] == EXPECTED

    # the work actually happened in the workdir
    assert "listens on port 8080" in (work / "README.md").read_text(encoding="utf-8")
    assert json.loads((runs / "owner" / steps[11]["artifact"]).read_text())["exit_code"] == 0
    assert "check: ok" in (runs / "owner" / steps[11]["artifact"]).read_text()
    assert (runs / "owner" / "scratch" / "port.md").exists()

    # skills: discovered, listed, two loaded, one unloaded, and never
    # activated by hand - the skills declared every tool the run used
    assert records[0]["skills"]["names"] == ["code-change", "investigate"]
    assert json.loads(steps[0]["result_preview"])[0]["name"] == "code-change"
    assert "toolbelt_add" not in [s["tool"] for s in steps]
    loaded = (runs / "owner" / steps[1]["artifact"]).read_text(encoding="utf-8")
    assert "activated fs_glob, fs_search, fs_read, scratch_write" in loaded
    assert "Quote paths and line numbers" not in loaded          # the description, not the body
    assert "`fs_read` every file you intend to quote" in loaded
    second = (runs / "owner" / steps[7]["artifact"]).read_text(encoding="utf-8")
    assert "activated fs_edit, run_shell" in second and "already active: fs_search, fs_read" in second

    # the plan stopped moving for five steps, and exactly one nudge landed
    notes = [r for r in records if r["type"] == "note"]
    assert [(n["kind"], n["step"]) for n in notes] == [("progress_nudge", 12)]
    nudged = bodies[12]["messages"][-1]
    assert nudged["role"] == "user" and nudged["content"] == notes[0]["text"]
    assert "5 steps since your todo list last changed" in nudged["content"]

    # the finish is auditable: every item, and what came of it
    footer = records[-1]
    assert footer["status"] == "completed" and footer["steps"] == 16
    assert footer["todos"] == [
        {"id": "find", "content": "find the port the service really uses", "status": "completed",
         "notes": "8080: app/config.ini:2 and app/server.py:3; README says 9090"},
        {"id": "fix", "content": "fix the port in README.md", "status": "completed",
         "notes": "README.md:3 9090 -> 8080, one line in the diff"},
        {"id": "check", "content": "run ./check.py", "status": "completed",
         "notes": "python3 check.py -> exit 0, 'check: ok'"},
        {"id": "report", "content": "report the port, the edit and the check", "status": "completed",
         "notes": "answered in final_answer"}]

    s = summarize(records)
    assert s["skills_loaded"] == ["investigate", "code-change"] and s["progress_nudges"] == 1
    assert s["tokens_in"] == sum(e["usage"]["prompt_tokens"] for e in script())

    assert main(["--runs-dir", str(runs), "trace", "owner", "--summary"]) == 0
    out = capsys.readouterr().out
    assert "status: completed  steps: 16" in out
    assert "skills: loaded investigate, code-change  (2 discovered in 1 directory)" in out
    assert "progress nudges: 1" in out
    assert "final [completed]: The service listens on 8080" in out
    assert "  [completed] fix: fix the port in README.md" in out
    assert "      note: python3 check.py -> exit 0, 'check: ok'" in out
    assert out.count("note: ") == 4
