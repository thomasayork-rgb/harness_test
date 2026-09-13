"""The scratch pad: notes in the run directory that outlive context eviction."""
import json

from harness.cli import main
from harness.mockserver import MockOpenAIServer
from harness.registry import ToolRegistry
from harness.runtime import AgentRuntime, RuntimeConfig
from harness.tools import register_default_tools, register_scratch_tools
from harness.trajectory import read_trajectory
from harness.transport import FakeTransport, call

TODO_DONE = call("todo_write", {"todos": [{"id": "1", "content": "work", "status": "completed"}]})
FINISH = call("final_answer", {"status": "completed", "content": "done"})


def drive(tmp_path, script, run_id="scratch", **cfg):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    registry = ToolRegistry()
    register_default_tools(registry, work)
    rt = AgentRuntime(registry, FakeTransport(script), tmp_path / "runs", "fake",
                      RuntimeConfig(**cfg), run_id=run_id)
    register_scratch_tools(registry, rt.run_dir)
    res = rt.run("keep notes")
    return res, [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]


def test_scratch_write_read_list_through_the_runtime(tmp_path):
    script = [
        {"content": "Activating the pad.",
         "tool_calls": [call("toolbelt_add", {"names": ["scratch_write", "scratch_read", "scratch_list"]})]},
        {"content": "Nothing saved yet.", "tool_calls": [call("scratch_list", {})]},
        {"content": "Saving what I found.",
         "tool_calls": [call("scratch_write", {"name": "findings", "content": "port=8080\n"})]},
        {"content": "Adding to it.",
         "tool_calls": [call("scratch_write", {"name": "findings", "content": "host=0.0.0.0\n", "append": True})]},
        {"content": "Replacing another note.",
         "tool_calls": [call("scratch_write", {"name": "plan", "content": "second version"})]},
        {"content": "Reading it back.", "tool_calls": [call("scratch_read", {"name": "findings"})]},
        {"content": "A note that is not there.", "tool_calls": [call("scratch_read", {"name": "missing"})]},
        {"content": "A name with a path in it.",
         "tool_calls": [call("scratch_write", {"name": "../escape", "content": "nope"})]},
        {"content": "An empty name.", "tool_calls": [call("scratch_read", {"name": ""})]},
        {"content": "Listing again.", "tool_calls": [call("scratch_list", {})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, script)
    assert res.status == "completed"
    assert [s["kind"] for s in steps] == [
        "ok", "ok", "ok", "ok", "ok", "ok", "error", "error", "error", "ok", "ok", "final_accepted"]

    assert json.loads(steps[1]["result_preview"]) == {"notes": []}
    assert json.loads(steps[2]["result_preview"]) == {"name": "findings", "bytes": 10, "appended": False}
    assert json.loads(steps[3]["result_preview"])["appended"] is True
    read_back = json.loads((res.run_dir / steps[5]["artifact"]).read_text())
    assert read_back["content"] == "port=8080\nhost=0.0.0.0\n"
    assert "no note named 'missing'" in steps[6]["result_preview"]
    assert "invalid note name '../escape'" in steps[7]["result_preview"]
    assert "invalid note name ''" in steps[8]["result_preview"]
    assert json.loads(steps[9]["result_preview"])["notes"] == [
        {"name": "findings", "bytes": 23}, {"name": "plan", "bytes": 14}]

    # the notes are files in the run directory, beside the trajectory
    pad = res.run_dir / "scratch"
    assert sorted(p.name for p in pad.iterdir()) == ["findings", "plan"]
    assert (pad / "plan").read_text() == "second version"
    assert not (res.run_dir.parent / "escape").exists()


def test_a_note_outlives_the_eviction_of_the_result_that_wrote_it(tmp_path):
    """The point of the pad: the conversation forgets, the note does not."""
    big = "B" * 900
    script = [
        {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["scratch_write", "scratch_read", "fs_write"]})]},
        {"content": "Saving the fact.",
         "tool_calls": [call("scratch_write", {"name": "fact", "content": "the port is 8080"})]},
    ] + [
        {"content": f"Filling the context {i}.", "tool_calls": [call("fs_write", {"path": f"f{i}.txt", "content": big})]}
        for i in range(6)
    ] + [
        {"content": "What was that fact?", "tool_calls": [call("scratch_read", {"name": "fact"})]},
        {"content": "Wrapping up.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    res, steps = drive(tmp_path, script, run_id="evict",
                       result_context_chars=1000, context_budget_chars=2500)
    assert res.status == "completed"

    state = json.loads((res.run_dir / "state.json").read_text())
    tools = [m for m in state["messages"] if m["role"] == "tool"]
    assert any(m["content"].startswith("[result evicted") for m in tools)   # the budget bit
    assert json.loads(steps[8]["result_preview"])["content"] == "the port is 8080"
    assert (res.run_dir / "scratch" / "fact").read_text() == "the port is 8080"


def test_scratch_pad_over_http_and_across_a_resume(tmp_path):
    runs = tmp_path / "runs"
    work = tmp_path / "work"
    work.mkdir()
    first = [
        {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["scratch_write", "scratch_read"]})]},
        {"content": "Saving a fact.",
         "tool_calls": [call("scratch_write", {"name": "fact", "content": "the port is 8080"})]},
    ]
    with MockOpenAIServer(first) as server:
        assert main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "m",
                     "--endpoint", server.base_url, "--workdir", str(work),
                     "--run-id", "pad", "--step-cap", "2"]) == 3

    rest = [
        {"content": "What did I save?", "tool_calls": [call("scratch_read", {"name": "fact"})]},
        {"content": "Done.", "tool_calls": [TODO_DONE, FINISH]},
    ]
    with MockOpenAIServer(rest) as server:
        assert main(["--runs-dir", str(runs), "resume", "pad", "--endpoint", server.base_url,
                     "--workdir", str(work), "--step-cap", "8"]) == 0

    steps = [r for r in read_trajectory(runs / "pad") if r["type"] == "step"]
    assert [(s["tool"], s["kind"]) for s in steps] == [
        ("toolbelt_add", "ok"), ("scratch_write", "ok"), ("scratch_read", "ok"),
        ("todo_write", "ok"), ("final_answer", "final_accepted")]
    # the resumed segment read back what the first segment wrote
    assert json.loads(steps[2]["result_preview"])["content"] == "the port is 8080"
    assert (runs / "pad" / "scratch" / "fact").read_text() == "the port is 8080"
