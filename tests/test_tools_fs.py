"""The new file tools, exercised through the runtime rather than called directly."""
import json

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
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, work, script)
    assert res.status == "completed"
    assert [s["tool"] for s in steps] == [
        "toolbelt_list", "toolbelt_add", "fs_search", "fs_search", "fs_search",
        "fs_search", "fs_search", "fs_search", "fs_glob", "todo_write", "final_answer"]
    assert [s["kind"] for s in steps] == [
        "ok", "ok", "error", "error", "error", "ok", "ok", "ok", "ok", "ok", "final_accepted"]

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
