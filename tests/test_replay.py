"""A recorded run, re-driven through the runtime as a regression test."""
import json

from harness.cli import main
from harness.registry import ToolRegistry
from harness.replay import ReplayTransport, compare, replay, turns_from_trajectory
from harness.runtime import AgentRuntime, RuntimeConfig
from harness.tools import register_default_tools
from harness.trajectory import read_trajectory
from harness.transport import FakeTransport, call

PROJECTION = ("step", "call_index", "reasoning", "tool", "args", "kind",
              "result_preview", "result_bytes", "artifact", "tokens_in", "tokens_out", "todo_snapshot")


def project(records):
    return [tuple(r[f] for f in PROJECTION) for r in records if r["type"] == "step"]


def workdir(tmp_path):
    work = tmp_path / "project"
    work.mkdir()
    (work / "config.ini").write_text("[server]\nPORT = 8080\n", encoding="utf-8")
    return work


def registry_for(work):
    r = ToolRegistry()
    register_default_tools(r, work)
    return r


SCRIPT = [
    {"content": "Looking around.", "tool_calls": [call("toolbelt_list", {"filter": "fs_"})],
     "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
    {"content": "Adding and inspecting in one turn.",
     "tool_calls": [call("toolbelt_add", {"names": ["fs_search", "fs_read"]}),
                    call("toolbelt_inspect", {"name": "fs_search"})],
     "usage": {"prompt_tokens": 110, "completion_tokens": 11}},
    {"content": "Let me think about this for a moment.", "tool_calls": [],
     "usage": {"prompt_tokens": 120, "completion_tokens": 12}},
    {"content": "Planning.", "tool_calls": [call("todo_write", {"todos": [
        {"id": "1", "content": "find the port", "status": "in_progress"}]})],
     "usage": {"prompt_tokens": 130, "completion_tokens": 13}},
    {"content": "Searching, and reading a file that is not there.",
     "tool_calls": [call("fs_search", {"pattern": "PORT"}), call("fs_read", {"path": "missing.txt"})],
     "usage": {"prompt_tokens": 140, "completion_tokens": 14}},
    {"content": "Finishing too early.", "tool_calls": [call("final_answer", {"status": "completed", "content": "8080"})],
     "usage": {"prompt_tokens": 150, "completion_tokens": 15}},
    {"content": "Closing the todo.", "tool_calls": [call("todo_write", {"todos": [
        {"id": "1", "content": "find the port", "status": "completed"}]})],
     "usage": {"prompt_tokens": 160, "completion_tokens": 16}},
    {"content": "Finishing.", "tool_calls": [call("final_answer", {"status": "completed", "content": "port 8080"})],
     "usage": {"prompt_tokens": 170, "completion_tokens": 17}},
]


def record(tmp_path, work):
    rt = AgentRuntime(registry_for(work), FakeTransport(SCRIPT), tmp_path / "runs", "fake-model",
                      RuntimeConfig(preview_chars=300, text_only_limit=3), run_id="rec")
    res = rt.run("find the port in the config")
    assert res.status == "completed" and res.steps == 10
    return res


def test_turns_are_reconstructed_from_call_index(tmp_path):
    work = workdir(tmp_path)
    res = record(tmp_path, work)
    turns = turns_from_trajectory(read_trajectory(res.run_dir))
    assert len(turns) == len(SCRIPT)
    assert [[c["name"] for c in t["tool_calls"]] for t in turns] == \
        [[c["name"] for c in (s["tool_calls"] or [])] for s in SCRIPT]
    assert [t["content"] for t in turns] == [s["content"] for s in SCRIPT]
    assert turns[1]["usage"] == {"prompt_tokens": 110, "completion_tokens": 11}
    assert turns[2]["tool_calls"] == []  # the text-only turn survives as a turn


def test_replay_reproduces_the_recording_step_for_step(tmp_path):
    work = workdir(tmp_path)
    original = record(tmp_path, work)
    res = replay(original.run_dir, registry_for(work))

    assert res.run_id == "rec-replay" and res.status == "completed" and res.steps == 10
    assert compare(original.run_dir, res.run_dir) == []
    assert project(read_trajectory(res.run_dir)) == project(read_trajectory(original.run_dir))
    # and the artifacts were rewritten with the same content
    steps = [s for s in read_trajectory(res.run_dir) if s["type"] == "step"]
    for s in steps:
        assert (res.run_dir / s["artifact"]).read_text(encoding="utf-8") == \
               (original.run_dir / s["artifact"]).read_text(encoding="utf-8")
    assert json.loads(steps[5]["result_preview"])["matches"][0]["line"] == 2   # fs_search
    assert "missing.txt" in steps[6]["result_preview"]                          # fs_read, error


def test_replay_detects_tool_drift(tmp_path):
    work = workdir(tmp_path)
    original = record(tmp_path, work)
    (work / "config.ini").write_text("[server]\nPORT = 9999\nPORT_ALT = 1\n", encoding="utf-8")

    res = replay(original.run_dir, registry_for(work), run_id="drifted")
    diffs = compare(original.run_dir, res.run_dir)
    assert diffs and all(d.startswith("step 6:") for d in diffs)   # only the fs_search step moved
    preview = next(d for d in diffs if "result_preview" in d)
    assert "differs at char" in preview                             # not just the first 120 chars
    assert "recorded 1," in preview and "replayed 2," in preview    # match_count 1 -> 2
    assert any("result_bytes: recorded 245" in d for d in diffs)


def test_replay_cli_reports_identical_then_drift(tmp_path, capsys):
    work = workdir(tmp_path)
    record(tmp_path, work)
    runs = str(tmp_path / "runs")

    rc = main(["--runs-dir", runs, "replay", "rec", "--workdir", str(work)])
    out = capsys.readouterr()
    assert rc == 0 and "identical to the recording, step for step" in out.out
    assert "replayed rec -> rec-replay" in out.err and "status: completed  steps: 10" in out.err

    (work / "config.ini").write_text("[server]\nPORT = 1234\n", encoding="utf-8")
    rc = main(["--runs-dir", runs, "replay", "rec", "--workdir", str(work), "--new-run-id", "again"])
    out = capsys.readouterr()
    assert rc == 1
    assert "difference(s) from the recording" in out.out and "step 6: result_preview" in out.out

    rc = main(["--runs-dir", runs, "replay", "nope", "--workdir", str(work)])
    assert rc == 66 and "no trajectory at" in capsys.readouterr().err


def test_replay_running_out_of_recorded_turns_is_a_transport_error(tmp_path):
    work = workdir(tmp_path)
    original = record(tmp_path, work)
    records = read_trajectory(original.run_dir)
    truncated = [r for r in records if r["type"] != "step" or r["step"] <= 4]

    transport = ReplayTransport(truncated)
    res = AgentRuntime(registry_for(work), transport, tmp_path / "runs", transport.model,
                       transport.config, run_id="short").run(transport.task)
    assert res.status == "transport_error"
    assert read_trajectory(res.run_dir)[-1]["detail"] == "replay exhausted after 3 recorded turn(s)"
    assert transport.config.preview_chars == 300  # config came back out of the header
