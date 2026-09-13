"""harness trace --summary: run stats read back out of the JSONL."""
from harness.cli import main
from harness.registry import ToolRegistry, ToolSpec
from harness.runtime import AgentRuntime, RuntimeConfig
from harness.trajectory import format_summary, format_trace, read_trajectory, summarize
from harness.transport import FakeTransport, call


def registry():
    r = ToolRegistry()
    r.register(ToolSpec("echo", "Echo a string.", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}, lambda s: s))
    r.register(ToolSpec("boom", "Raise.", {"type": "object", "properties": {}, "required": []},
                        lambda: (_ for _ in ()).throw(RuntimeError("kaboom"))))
    return r


def a_run(tmp_path, run_id="sum"):
    script = [
        {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["echo", "boom"]})],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"content": "Thinking out loud.", "tool_calls": [], "usage": {"prompt_tokens": 150, "completion_tokens": 5}},
        {"content": "Echoing twice, then breaking things.",
         "tool_calls": [call("echo", {"s": "one"}), call("echo", {"s": "two"}), call("boom", {})],
         "usage": {"prompt_tokens": 200, "completion_tokens": 20}},
        {"content": "Finishing without a plan.", "tool_calls": [call("final_answer", {"status": "completed", "content": "x"})],
         "usage": {"prompt_tokens": 250, "completion_tokens": 7}},
        {"content": "Right, the plan.", "tool_calls": [
            call("todo_write", {"todos": [{"id": "1", "content": "echo", "status": "completed"}]}),
            call("final_answer", {"status": "completed", "content": "all done"})],
         "usage": {"prompt_tokens": 300, "completion_tokens": 8}},
    ]
    return AgentRuntime(registry(), FakeTransport(script), tmp_path, "fake-model",
                        RuntimeConfig(text_only_limit=5), run_id=run_id).run("summarise me")


def test_summarize_counts_kinds_tools_tokens(tmp_path):
    res = a_run(tmp_path)
    assert res.status == "completed"
    s = summarize(read_trajectory(res.run_dir))

    assert s["run_id"] == "sum" and s["model"] == "fake-model" and s["task"] == "summarise me"
    assert s["status"] == "completed" and s["steps"] == 8 and s["step_cap"] == 250
    assert s["kinds"] == {"ok": 4, "error": 1, "final_accepted": 1, "final_rejected": 1, "text_only": 1}
    assert s["tools"] == {"echo": 2, "boom": 1, "(text only)": 1, "final_answer": 2,
                          "todo_write": 1, "toolbelt_add": 1}
    assert s["errors"] == 1 and s["rejected_finals"] == 1 and s["text_only"] == 1
    # usage counted once per turn, never per tool call
    assert s["tokens_in"] == 1000 and s["tokens_out"] == 50
    assert s["elapsed_ms"] >= 0 and s["wall_s"] is not None and s["wall_s"] >= 0
    assert s["final"] == {"status": "completed", "content": "all done"}


def test_summary_cli_and_step_view_stay_separate(tmp_path, capsys):
    res = a_run(tmp_path, run_id="cli")
    rc = main(["--runs-dir", str(tmp_path), "trace", "cli", "--summary"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "run cli  model=fake-model" in out
    assert "status: completed  steps: 8" in out
    assert "tokens: 1000 in / 50 out" in out
    assert "errors: 1  rejected finals: 1  text-only turns: 1" in out
    assert "kinds: ok 4" in out and "echo" in out and "final [completed]: all done" in out
    assert "s in steps," in out and "s wall" in out

    # the step list is unchanged by the new view
    plain = format_trace(read_trajectory(res.run_dir), res.run_dir)
    assert "toolbelt_add" in plain and "Echoing twice" in plain and "status: completed  steps: 8" in plain
    assert "kinds:" not in plain

    rc = main(["--runs-dir", str(tmp_path), "trace", "cli"])
    assert rc == 0 and capsys.readouterr().out.strip() == plain.strip()


def test_summary_of_an_unfinished_trajectory():
    records = [
        {"type": "header", "run_id": "x", "ts": "2024-01-01T00:00:00Z", "model": "m",
         "harness_version": "0.1.0", "step_cap": 10, "task": "t", "config": {}},
        {"type": "step", "run_id": "x", "step": 1, "ts": "2024-01-01T00:00:03Z", "elapsed_ms": 1500,
         "reasoning": "", "tool": "echo", "args": {}, "kind": "ok", "result_preview": "",
         "result_bytes": 0, "artifact": None, "tokens_in": None, "tokens_out": None, "todo_snapshot": []},
    ]
    s = summarize(records)
    assert s["status"] == "incomplete" and s["steps"] == 1 and s["final"] is None
    assert s["tokens_in"] == 0 and s["wall_s"] is None
    text = format_summary(records)
    assert "status: incomplete  steps: 1" in text and "elapsed: 1.5 s in steps, ? s wall" in text
