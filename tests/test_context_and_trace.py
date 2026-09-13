import json

from harness.cli import main
from harness.context import EVICTED, ContextBudget
from harness.registry import ToolRegistry, ToolSpec
from harness.runtime import AgentRuntime, RuntimeConfig
from harness.trajectory import format_trace, read_trajectory
from harness.transport import FakeTransport, call, normalize_openai


def test_truncate_and_evict_never_touch_reasoning_or_todos():
    b = ContextBudget(result_chars=10, total_chars=80)
    assert b.truncate_result("short", "a") == "short"
    t = b.truncate_result("x" * 50, "artifacts/s.txt")
    assert t.startswith("x" * 10) and "artifacts/s.txt" in t

    msgs = [
        {"role": "system", "content": "S" * 20},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "reasoning one", "tool_calls": [{"function": {"arguments": "{}"}}]},
        {"role": "tool", "content": "R" * 30, "_artifact": "a1"},
        {"role": "assistant", "content": "reasoning two", "tool_calls": [{"function": {"arguments": "{}"}}]},
        {"role": "tool", "content": "T" * 30, "_artifact": "a2", "_protected": True},
        {"role": "assistant", "content": "reasoning three", "tool_calls": [{"function": {"arguments": "{}"}}]},
        {"role": "tool", "content": "R" * 30, "_artifact": "a3"},
    ]
    evicted = b.enforce(msgs)
    assert evicted >= 1
    assert msgs[3]["content"] == EVICTED.format(artifact="a1")  # oldest unprotected went first
    assert msgs[5]["content"] == "T" * 30                        # protected todo result untouched
    assert all("reasoning" in m["content"] for m in msgs if m["role"] == "assistant")
    assert b.enforce(msgs) == 0 or b.size(msgs) <= 80 or msgs[7]["_evicted"]


def test_budget_enforced_inside_runtime(tmp_path):
    r = ToolRegistry()
    r.register(ToolSpec("big", "big", {"type": "object", "properties": {}, "required": []}, lambda: "B" * 900))
    fake = FakeTransport(
        [{"content": "add", "tool_calls": [call("toolbelt_add", {"names": ["big"]})]}]
        + [{"content": f"call {i}", "tool_calls": [call("big", {})]} for i in range(6)]
        + [{"content": "done", "tool_calls": [call("todo_write", {"todos": [{"id": "1", "content": "x", "status": "completed"}]}),
                                             call("final_answer", {"status": "completed", "content": "ok"})]}]
    )
    cfg = RuntimeConfig(result_context_chars=1000, context_budget_chars=3500)
    res = AgentRuntime(r, fake, tmp_path, "fake", cfg).run("t")
    assert res.status == "completed"
    last_msgs = fake.requests[-1]["messages"]
    tool_msgs = [m for m in last_msgs if m["role"] == "tool"]
    assert any(m["content"].startswith("[result evicted") for m in tool_msgs)
    assert tool_msgs[-1]["content"] == "B" * 900  # newest kept
    assert sum(len(m["content"]) for m in last_msgs) <= 3500 + 1000  # bounded


def test_normalize_openai_parses_and_keeps_bad_json_raw():
    payload = {"choices": [{"message": {"content": "hi", "tool_calls": [
        {"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}},
        {"id": "c2", "function": {"name": "g", "arguments": "{oops"}},
    ]}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
    n = normalize_openai(payload)
    assert n["content"] == "hi"
    assert n["tool_calls"][0]["arguments"] == {"a": 1}
    assert n["tool_calls"][1]["arguments"] == "{oops"
    assert n["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}


def test_trace_cli(tmp_path, capsys):
    r = ToolRegistry()
    r.register(ToolSpec("echo", "echo", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}, lambda s: s))
    fake = FakeTransport([
        {"content": "Adding echo.", "tool_calls": [call("toolbelt_add", {"names": ["echo"]})]},
        {"content": "Echoing.", "tool_calls": [call("echo", {"s": "hello " * 200})]},
        {"content": "Finishing.", "tool_calls": [call("todo_write", {"todos": [{"id": "1", "content": "x", "status": "completed"}]}),
                                                call("final_answer", {"status": "completed", "content": "ok"})]},
    ])
    res = AgentRuntime(r, fake, tmp_path, "fake", RuntimeConfig(preview_chars=50), run_id="tr").run("say hello")
    assert res.status == "completed"

    text = format_trace(read_trajectory(res.run_dir), res.run_dir)
    assert "run tr" in text and "toolbelt_add" in text and "echo" in text and "status: completed" in text

    rc = main(["--runs-dir", str(tmp_path), "trace", "tr", "--step", "2"])
    out = capsys.readouterr().out
    assert rc == 0 and "step 2" in out and "Echoing." in out
    assert out.count("hello") == 400  # args (200) + full artifact (200), not the 50-char preview

    rc = main(["--runs-dir", str(tmp_path), "trace", "missing"])
    assert rc == 66

    steps = [x for x in read_trajectory(res.run_dir) if x["type"] == "step"]
    assert len(steps[1]["result_preview"]) == 50 and steps[1]["result_bytes"] == 1200
