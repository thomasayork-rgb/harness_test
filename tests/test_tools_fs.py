"""The new file tools, exercised through the runtime rather than called directly."""
import json
import os
import time
from pathlib import Path

from harness.registry import ToolRegistry
from harness.runtime import AgentRuntime, RuntimeConfig
from harness.tools import register_default_tools
from harness.trajectory import read_trajectory
from harness.transport import FakeTransport, call

TODO_DONE = call("todo_write", {"todos": [{"id": "1", "content": "work", "status": "completed"}]})
FINISH = call("final_answer", {"status": "completed", "content": "done"})


def project(tmp_path):
    """A small tree: two python files, a note, a binary blob, a skipped cache dir."""
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text(
        "import os\n\n\ndef main():\n    print('hello')\n    return 0\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text(
        "VALUE = 1\n\n\ndef helper():\n    return VALUE\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("hello from the notes\nsecond line\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"hello\x00\x00binary")
    (tmp_path / "latin.txt").write_bytes(b"caf\xe9 au lait\n")   # no NUL, still not UTF-8
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.py").write_text("hello cache\n", encoding="utf-8")
    return tmp_path


def drive(tmp_path, workdir, script, **cfg):
    registry = ToolRegistry()
    register_default_tools(registry, workdir)
    runs = tmp_path / "runs"
    rt = AgentRuntime(registry, FakeTransport(script), runs, "fake",
                      RuntimeConfig(**cfg), run_id="fs")
    res = rt.run("exercise the file tools")
    steps = [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]
    return res, steps


def artifact(res, step):
    return (res.run_dir / step["artifact"]).read_text(encoding="utf-8")


def test_search_and_glob_through_the_runtime(tmp_path):
    work = project(tmp_path / "work")
    script = [
        {"content": "Looking for search tools.", "tool_calls": [call("toolbelt_list", {"filter": "search"})]},
        {"content": "Activating them.", "tool_calls": [call("toolbelt_add", {"names": ["fs_search", "fs_glob"]})]},
        {"content": "A bad regex first.", "tool_calls": [call("fs_search", {"pattern": "def ("})]},
        {"content": "Searching a path that does not exist.",
         "tool_calls": [call("fs_search", {"pattern": "hello", "path": "nope"})]},
        {"content": "Searching outside the workdir.",
         "tool_calls": [call("fs_search", {"pattern": "hello", "path": "../elsewhere"})]},
        {"content": "Now the real search.",
         "tool_calls": [call("fs_search", {"pattern": "^def ", "glob": "*.py"})]},
        {"content": "Case-insensitive across everything.",
         "tool_calls": [call("fs_search", {"pattern": "HELLO", "ignore_case": True})]},
        {"content": "Capping results.",
         "tool_calls": [call("fs_search", {"pattern": "e", "max_results": 2})]},
        {"content": "Listing python files.", "tool_calls": [call("fs_glob", {"pattern": "**/*.py"})]},
        {"content": "Globbing a path that is not there.",
         "tool_calls": [call("fs_glob", {"pattern": "*", "path": "nope"})]},
        {"content": "Globbing a file rather than a directory.",
         "tool_calls": [call("fs_glob", {"pattern": "*", "path": "notes.md"})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, work, script)
    assert res.status == "completed"
    assert [s["tool"] for s in steps] == [
        "toolbelt_list", "toolbelt_add", "fs_search", "fs_search", "fs_search",
        "fs_search", "fs_search", "fs_search", "fs_glob", "fs_glob", "fs_glob",
        "todo_write", "final_answer"]
    assert [s["kind"] for s in steps] == [
        "ok", "ok", "error", "error", "error", "ok", "ok", "ok", "ok", "error", "error",
        "ok", "final_accepted"]

    listed = json.loads(artifact(res, steps[0]))
    assert {"name": "fs_search", "description": listed[0]["description"]} in listed
    assert all("regex" in e["description"] or "glob" in e["description"] for e in listed)

    assert "invalid regex" in steps[2]["result_preview"] and "ValueError" in steps[2]["result_preview"]
    assert "no such path: nope" in steps[3]["result_preview"]
    assert "escapes workdir" in steps[4]["result_preview"]

    found = json.loads(artifact(res, steps[5]))
    assert found["match_count"] == 2 and found["truncated"] is False
    assert [(m["path"], m["line"], m["text"]) for m in found["matches"]] == [
        ("src/app.py", 4, "def main():"), ("src/util.py", 4, "def helper():")]

    insensitive = json.loads(artifact(res, steps[6]))
    hit_paths = {m["path"] for m in insensitive["matches"]}
    assert hit_paths == {"notes.md", "src/app.py"}          # binary skipped, __pycache__ skipped
    assert insensitive["files_skipped"] == 1                 # blob.bin

    capped = json.loads(artifact(res, steps[7]))
    assert capped["match_count"] == 2 and capped["truncated"] is True

    globbed = json.loads(artifact(res, steps[8]))
    assert [f["path"] for f in globbed["files"]] == ["src/app.py", "src/util.py"]
    assert globbed["count"] == 2 and globbed["truncated"] is False
    assert "no such path: nope" in steps[9]["result_preview"]
    assert "not a directory: notes.md" in steps[10]["result_preview"]


def test_edit_through_the_runtime(tmp_path):
    work = project(tmp_path / "work")
    script = [
        {"content": "Activating the editor.", "tool_calls": [call("toolbelt_add", {"names": ["fs_edit"]})]},
        {"content": "Editing a file that is not there.",
         "tool_calls": [call("fs_edit", {"path": "src/missing.py", "old": "a", "new": "b"})]},
        {"content": "Old string is not in the file.",
         "tool_calls": [call("fs_edit", {"path": "src/app.py", "old": "def nope()", "new": "x"})]},
        {"content": "Ambiguous: VALUE appears twice.",
         "tool_calls": [call("fs_edit", {"path": "src/util.py", "old": "VALUE", "new": "COUNT"})]},
        {"content": "So replace them all.",
         "tool_calls": [call("fs_edit", {"path": "src/util.py", "old": "VALUE", "new": "COUNT", "replace_all": True})]},
        {"content": "And one unique edit.",
         "tool_calls": [call("fs_edit", {"path": "src/app.py", "old": "    print('hello')\n", "new": "    print('goodbye')\n"})]},
        {"content": "Missing an argument.",
         "tool_calls": [call("fs_edit", {"path": "src/app.py", "old": "import os"})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, work, script)
    assert res.status == "completed"
    assert [s["kind"] for s in steps] == ["ok", "error", "error", "error", "ok", "ok", "error", "ok", "final_accepted"]

    assert "no such file: src/missing.py" in steps[1]["result_preview"]
    assert "no occurrence of that exact string" in steps[2]["result_preview"]
    assert "2 occurrences" in steps[3]["result_preview"] and "replace_all=true" in steps[3]["result_preview"]
    assert "missing required argument(s): new" in steps[6]["result_preview"]

    bulk = json.loads(artifact(res, steps[4]))
    assert bulk["replacements"] == 2
    assert "-VALUE = 1" in bulk["diff"] and "+COUNT = 1" in bulk["diff"]
    assert (work / "src" / "util.py").read_text() == "COUNT = 1\n\n\ndef helper():\n    return COUNT\n"

    one = json.loads(artifact(res, steps[5]))
    assert one["replacements"] == 1 and one["diff"].startswith("--- a/src/app.py")
    assert "-    print('hello')" in one["diff"] and "+    print('goodbye')" in one["diff"]
    assert (work / "src" / "app.py").read_text() == \
        "import os\n\n\ndef main():\n    print('goodbye')\n    return 0\n"

    # the edited file is not corrupted for the tools that read it back
    assert steps[5]["result_bytes"] == len(artifact(res, steps[5]).encode("utf-8"))


def test_edit_refuses_binary_and_empty_old(tmp_path):
    work = project(tmp_path / "work")
    script = [
        {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["fs_edit"]})]},
        {"content": "Binary file.", "tool_calls": [call("fs_edit", {"path": "blob.bin", "old": "hello", "new": "bye"})]},
        {"content": "Undecodable file.", "tool_calls": [call("fs_edit", {"path": "latin.txt", "old": "au", "new": "AU"})]},
        {"content": "Empty old.", "tool_calls": [call("fs_edit", {"path": "notes.md", "old": "", "new": "x"})]},
        {"content": "No-op edit.", "tool_calls": [call("fs_edit", {"path": "notes.md", "old": "x", "new": "x"})]},
        {"content": "A directory.", "tool_calls": [call("fs_edit", {"path": "src", "old": "a", "new": "b"})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, work, script)
    assert res.status == "completed"
    assert [s["kind"] for s in steps[1:6]] == ["error"] * 5
    assert "looks binary" in steps[1]["result_preview"]
    assert "not valid UTF-8" in steps[2]["result_preview"]
    assert "must not be empty" in steps[3]["result_preview"]
    assert "identical" in steps[4]["result_preview"]
    assert "not a file: src" in steps[5]["result_preview"]
    assert (work / "blob.bin").read_bytes() == b"hello\x00\x00binary"
    assert (work / "latin.txt").read_bytes() == b"caf\xe9 au lait\n"


def test_fs_read_errors_name_the_relative_path_only(tmp_path):
    work = project(tmp_path / "work")
    script = [
        {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["fs_read"]})]},
        {"content": "Reading a file that is not there.",
         "tool_calls": [call("fs_read", {"path": "src/missing.py"})]},
        {"content": "Reading a directory.", "tool_calls": [call("fs_read", {"path": "src"})]},
        {"content": "Reading a real file.", "tool_calls": [call("fs_read", {"path": "notes.md"})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, work, script)
    assert [s["kind"] for s in steps[1:4]] == ["error", "error", "ok"]
    assert steps[1]["result_preview"] == "error: FileNotFoundError: no such file: src/missing.py"
    assert steps[2]["result_preview"] == "error: IsADirectoryError: not a file: src"
    assert json.loads(artifact(res, steps[3]))["content"].startswith("hello from the notes")
    # nothing in the trajectory tells the model where the workdir lives on disk
    assert str(work) not in (res.run_dir / "trajectory.jsonl").read_text(encoding="utf-8")


def test_write_and_list_through_the_runtime(tmp_path):
    work = project(tmp_path / "work")
    script = [
        {"content": "Activating the two tools.",
         "tool_calls": [call("toolbelt_add", {"names": ["fs_write", "fs_list"]})]},
        {"content": "Listing the root.", "tool_calls": [call("fs_list", {})]},
        {"content": "Listing one file.", "tool_calls": [call("fs_list", {"path": "notes.md"})]},
        {"content": "Listing a directory that is not there.", "tool_calls": [call("fs_list", {"path": "nope"})]},
        {"content": "Listing outside the workdir.", "tool_calls": [call("fs_list", {"path": "../"})]},
        {"content": "Capping the listing.", "tool_calls": [call("fs_list", {"max_entries": 2})]},
        {"content": "Writing a file, creating its parents.",
         "tool_calls": [call("fs_write", {"path": "out/deep/report.txt", "content": "first\n"})]},
        {"content": "Overwriting it.",
         "tool_calls": [call("fs_write", {"path": "out/deep/report.txt", "content": "second\n"})]},
        {"content": "Writing outside the workdir.",
         "tool_calls": [call("fs_write", {"path": "../escape.txt", "content": "nope"})]},
        {"content": "Writing over a directory.",
         "tool_calls": [call("fs_write", {"path": "src", "content": "nope"})]},
        {"content": "Forgetting an argument.", "tool_calls": [call("fs_write", {"path": "x.txt"})]},
        {"content": "Seeing the new file.", "tool_calls": [call("fs_list", {"path": "out/deep"})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, work, script)
    assert res.status == "completed"
    assert [s["kind"] for s in steps] == [
        "ok", "ok", "ok", "error", "error", "ok", "ok", "ok", "error", "error", "error",
        "ok", "ok", "final_accepted"]

    root = json.loads(artifact(res, steps[1]))
    assert [e["name"] for e in root["entries"]] == [
        "__pycache__", "blob.bin", "latin.txt", "notes.md", "src"]
    assert [e["type"] for e in root["entries"]][0] == "dir"
    assert json.loads(artifact(res, steps[2])) == {"path": "notes.md", "type": "file", "bytes": 33}
    assert "FileNotFoundError: nope" in steps[3]["result_preview"]
    assert "escapes workdir: ../" in steps[4]["result_preview"]
    capped = json.loads(artifact(res, steps[5]))
    assert [e["name"] for e in capped["entries"]] == ["__pycache__", "blob.bin", "..."]

    assert json.loads(steps[6]["result_preview"]) == {"path": "out/deep/report.txt", "bytes": 6}
    assert (work / "out" / "deep" / "report.txt").read_text() == "second\n"
    assert "escapes workdir: ../escape.txt" in steps[8]["result_preview"]
    assert not (tmp_path / "escape.txt").exists() and not (work.parent / "escape.txt").exists()
    assert "IsADirectoryError" in steps[9]["result_preview"]
    assert "missing required argument(s): content" in steps[10]["result_preview"]
    assert [e["name"] for e in json.loads(artifact(res, steps[11]))["entries"]] == ["report.txt"]


def process_state(pid: int) -> str:
    """``R``/``S``/``Z`` for a pid, or ``gone`` when there is no such process.

    Read from /proc, which is Linux; anywhere without it every pid reads as
    ``gone``, which makes the check that uses this weaker, never wrong.
    """
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                return line.split()[1]
    except OSError:
        return "gone"
    return "?"


def wait_until_dead(pid: int, seconds: float = 3.0) -> str:
    """Poll for a process to die. A zombie is dead: it is waiting to be reaped."""
    deadline = time.monotonic() + seconds
    state = process_state(pid)
    while state not in ("gone", "Z") and time.monotonic() < deadline:
        time.sleep(0.05)
        state = process_state(pid)
    return state


def test_a_timed_out_command_takes_what_it_started_with_it(tmp_path):
    """The shell was killed and its children were not, so a backgrounded
    process outlived the tool, the run, and the harness."""
    work = project(tmp_path / "work")
    script = [
        {"content": "Activating the shell.", "tool_calls": [call("toolbelt_add", {"names": ["run_shell"]})]},
        {"content": "Backgrounding something that outlives the command.",
         "tool_calls": [call("run_shell", {"command": "sleep 30 & echo $! > sleep.pid; wait", "timeout": 1})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, work, script)
    assert res.status == "completed"
    assert steps[1]["kind"] == "error"
    assert steps[1]["result_preview"].startswith("error: TimeoutError: command timed out after 1s")

    pid = int((work / "sleep.pid").read_text(encoding="utf-8").strip())
    state = wait_until_dead(pid)
    if state not in ("gone", "Z"):
        os.kill(pid, 9)                                  # do not leave it behind either way
        raise AssertionError(f"the grandchild {pid} was still {state} after the timeout")


def test_run_shell_hands_no_harness_variable_to_the_command(tmp_path, monkeypatch):
    """`env` in a shell command used to put HARNESS_API_KEY in a result, an
    artifact and the trajectory, where anyone reading the run can see it."""
    monkeypatch.setenv("HARNESS_API_KEY", "sk-not-for-the-model")
    monkeypatch.setenv("HARNESS_SKILLS", "/somewhere/skills")
    monkeypatch.setenv("KEPT_BY_THE_CHILD", "yes")
    work = project(tmp_path / "work")
    script = [
        {"content": "Activating the shell.", "tool_calls": [call("toolbelt_add", {"names": ["run_shell"]})]},
        {"content": "What is in my environment?", "tool_calls": [call("run_shell", {"command": "env"})]},
        {"content": "And by name.",
         "tool_calls": [call("run_shell", {"command": "echo \"[$HARNESS_API_KEY]\""})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, work, script)
    assert res.status == "completed"

    env = json.loads(artifact(res, steps[1]))
    assert env["exit_code"] == 0
    assert [line for line in env["stdout"].splitlines() if line.startswith("HARNESS_")] == []
    assert "KEPT_BY_THE_CHILD=yes" in env["stdout"].splitlines()      # only HARNESS_* is taken
    assert json.loads(artifact(res, steps[2]))["stdout"] == "[]\n"    # unset, not empty-by-accident

    blob = (res.run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
    blob += "".join(p.read_text(encoding="utf-8") for p in (res.run_dir / "artifacts").iterdir())
    assert "sk-not-for-the-model" not in blob            # not in the run, anywhere
    assert "HARNESS_API_KEY=" not in blob                # the only HARNESS_ text is what the model typed
    assert "HARNESS_SKILLS" not in blob


def test_run_shell_through_the_runtime(tmp_path):
    work = project(tmp_path / "work")
    (work.parent / "outside.txt").write_text("secret\n", encoding="utf-8")
    script = [
        {"content": "Activating the shell.", "tool_calls": [call("toolbelt_add", {"names": ["run_shell"]})]},
        {"content": "Where am I?", "tool_calls": [call("run_shell", {"command": "pwd && ls"})]},
        {"content": "A command that fails.", "tool_calls": [call("run_shell", {"command": "ls no_such_file"})]},
        {"content": "Something that hangs.",
         "tool_calls": [call("run_shell", {"command": "sleep 5", "timeout": 1})]},
        {"content": "Wrong argument type.", "tool_calls": [call("run_shell", {"command": "true", "timeout": "1"})]},
        {"content": "Missing the command.", "tool_calls": [call("run_shell", {})]},
        {"content": "Reaching outside the workdir.",
         "tool_calls": [call("run_shell", {"command": "cat ../outside.txt"})]},
        {"content": "Writing through the shell.",
         "tool_calls": [call("run_shell", {"command": "echo shelled > from_shell.txt"})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, work, script)
    assert res.status == "completed"
    assert [s["kind"] for s in steps] == [
        "ok", "ok", "ok", "error", "error", "error", "ok", "ok", "ok", "final_accepted"]

    where = json.loads(artifact(res, steps[1]))
    assert where["exit_code"] == 0
    assert where["stdout"].splitlines()[0] == str(work)      # the command starts in the workdir
    assert "notes.md" in where["stdout"]

    failed = json.loads(artifact(res, steps[2]))
    assert failed["exit_code"] != 0 and failed["stderr"]     # a non-zero exit is a result, not an error

    # a timeout is a tool failure: kind "error", not an "ok" result with an error field in it
    assert steps[3]["result_preview"] == "error: TimeoutError: command timed out after 1s: sleep 5"
    assert "argument 'timeout' must be integer" in steps[4]["result_preview"]
    assert "missing required argument(s): command" in steps[5]["result_preview"]

    # run_shell starts in the workdir but is not a sandbox: the shell can still
    # walk out of it. Rooting is an fs_* guarantee; for the shell the lever is
    # the tool-call policy (--deny-tool / --deny-shell-pattern).
    escaped = json.loads(artifact(res, steps[6]))
    assert escaped["exit_code"] == 0 and "secret" in escaped["stdout"]
    assert (work / "from_shell.txt").read_text() == "shelled\n"
