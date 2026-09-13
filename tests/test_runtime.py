import json

from harness.registry import ToolRegistry, ToolSpec
from harness.runtime import META_NAMES, AgentRuntime, RuntimeConfig
from harness.trajectory import read_trajectory
from harness.transport import FakeTransport, TransportError, call


def registry_with(n=25):
    r = ToolRegistry()
    for i in range(n):
        r.register(ToolSpec(f"tool_{i:02d}", f"fixture tool {i}",
                            {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                            lambda x, i=i: {"tool": i, "x": x}))
    r.register(ToolSpec("boom", "raises", {"type": "object", "properties": {}, "required": []},
                        lambda: (_ for _ in ()).throw(RuntimeError("kaboom"))))
    r.register(ToolSpec("big", "returns a lot", {"type": "object", "properties": {}, "required": []},
                        lambda: "X" * 10000))
    return r


def todo(*items, merge=True):
    return call("todo_write", {"todos": [{"id": i, "content": c, "status": s} for i, c, s in items], "merge": merge})


def tool_names(request):
    return [t["function"]["name"] for t in request["tools"]]


def test_initial_payload_is_meta_tools_only_then_exactly_added(tmp_path):
    fake = FakeTransport([
        {"content": "Listing tools.", "tool_calls": [call("toolbelt_list", {})]},
        {"content": "Adding two.", "tool_calls": [call("toolbelt_add", {"names": ["tool_03", "tool_07"]})]},
        {"content": "Adding again (idempotent).", "tool_calls": [call("toolbelt_add", {"names": ["tool_03"]})]},
        {"content": "Removing one.", "tool_calls": [call("toolbelt_remove", {"names": ["tool_07"]})]},
        {"content": "Done.", "tool_calls": [todo(("1", "x", "completed")), call("final_answer", {"status": "completed", "content": "ok"})]},
    ])
    rt = AgentRuntime(registry_with(25), fake, tmp_path, "fake")
    res = rt.run("task")
    assert res.status == "completed"
    assert set(tool_names(fake.requests[0])) == META_NAMES
    assert set(tool_names(fake.requests[2])) == META_NAMES | {"tool_03", "tool_07"}
    assert set(tool_names(fake.requests[3])) == META_NAMES | {"tool_03", "tool_07"}
    assert set(tool_names(fake.requests[4])) == META_NAMES | {"tool_03"}
    steps = [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]
    added = json.loads(steps[2]["result_preview"])
    assert added["already_active"] == ["tool_03"] and added["added"] == []


def test_final_answer_gate(tmp_path):
    fake = FakeTransport([
        {"content": "Trying to finish without a plan.", "tool_calls": [call("final_answer", {"status": "completed", "content": "x"})]},
        {"content": "Planning.", "tool_calls": [todo(("a", "first", "in_progress"), ("b", "second", "pending"))]},
        {"content": "Trying again.", "tool_calls": [call("final_answer", {"status": "completed", "content": "x"})]},
        {"content": "Closing a, cancelling b.", "tool_calls": [todo(("a", "first", "completed"), ("b", "second", "cancelled"))]},
        {"content": "Finishing.", "tool_calls": [call("final_answer", {"status": "completed", "content": "done"})]},
    ])
    rt = AgentRuntime(registry_with(3), fake, tmp_path, "fake")
    res = rt.run("task")
    assert res.status == "completed" and res.final == {"status": "completed", "content": "done"}
    steps = [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]
    kinds = [s["kind"] for s in steps]
    assert kinds == ["final_rejected", "ok", "final_rejected", "ok", "final_accepted"]
    first = json.loads(steps[0]["result_preview"])
    assert first["accepted"] is False and "todo" in first["reason"]
    third = json.loads(steps[2]["result_preview"])
    assert third["open_ids"] == ["a", "b"]


def test_no_todo_gate_when_disabled(tmp_path):
    fake = FakeTransport([{"content": "", "tool_calls": [call("final_answer", {"status": "completed", "content": "x"})]}])
    rt = AgentRuntime(registry_with(1), fake, tmp_path, "fake", RuntimeConfig(require_todos=False))
    assert rt.run("t").status == "completed"


def test_scripted_end_to_end_matches_jsonl_step_for_step(tmp_path):
    script = [
        {"content": "Seeing what tools exist.", "tool_calls": [call("toolbelt_list", {"filter": "fixture"})],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"content": "tool_01 looks right; adding and inspecting.",
         "tool_calls": [call("toolbelt_add", {"names": ["tool_01"]}), call("toolbelt_inspect", {"name": "tool_01"})],
         "usage": {"prompt_tokens": 120, "completion_tokens": 20}},
        {"content": "Planning the work.", "tool_calls": [todo(("1", "call tool_01", "in_progress"))],
         "usage": {"prompt_tokens": 130, "completion_tokens": 15}},
        {"content": "Calling the tool.", "tool_calls": [call("tool_01", {"x": 42})],
         "usage": {"prompt_tokens": 140, "completion_tokens": 8}},
        {"content": "Trying to finish early.", "tool_calls": [call("final_answer", {"status": "completed", "content": "42"})],
         "usage": {"prompt_tokens": 150, "completion_tokens": 8}},
        {"content": "Closing the todo.", "tool_calls": [todo(("1", "call tool_01", "completed"))],
         "usage": {"prompt_tokens": 160, "completion_tokens": 8}},
        {"content": "Finishing.", "tool_calls": [call("final_answer", {"status": "completed", "content": "answer is 42"})],
         "usage": {"prompt_tokens": 170, "completion_tokens": 8}},
    ]
    rt = AgentRuntime(registry_with(5), FakeTransport(script), tmp_path, "fake", run_id="e2e")
    res = rt.run("find the answer")
    assert res.status == "completed" and res.steps == 8

    recs = read_trajectory(res.run_dir)
    assert recs[0]["type"] == "header" and recs[0]["run_id"] == "e2e" and recs[0]["task"] == "find the answer"
    assert recs[-1]["type"] == "footer" and recs[-1]["status"] == "completed" and recs[-1]["steps"] == 8
    steps = [r for r in recs if r["type"] == "step"]

    expected = [
        (1, "toolbelt_list", "ok", "Seeing what tools exist.", 100),
        (2, "toolbelt_add", "ok", "tool_01 looks right; adding and inspecting.", 120),
        (3, "toolbelt_inspect", "ok", "", None),  # second call in the same turn: no reasoning, no double-counted usage
        (4, "todo_write", "ok", "Planning the work.", 130),
        (5, "tool_01", "ok", "Calling the tool.", 140),
        (6, "final_answer", "final_rejected", "Trying to finish early.", 150),
        (7, "todo_write", "ok", "Closing the todo.", 160),
        (8, "final_answer", "final_accepted", "Finishing.", 170),
    ]
    assert [(s["step"], s["tool"], s["kind"], s["reasoning"], s["tokens_in"]) for s in steps] == expected
    assert json.loads(steps[4]["result_preview"]) == {"tool": 1, "x": 42}
    # snapshot is the todo state after the step
    assert steps[2]["todo_snapshot"] == [] and steps[3]["todo_snapshot"][0]["status"] == "in_progress"
    assert steps[7]["todo_snapshot"][0]["status"] == "completed"
    for s in steps:
        assert (res.run_dir / s["artifact"]).exists()
    state = json.loads((res.run_dir / "state.json").read_text())
    assert state["status"] == "completed" and state["active_tools"] == ["tool_01"] and state["final"]["content"] == "answer is 42"


def test_failure_semantics_tool_errors_keep_loop_alive(tmp_path):
    fake = FakeTransport([
        {"content": "", "tool_calls": [call("tool_00", {"x": 1})]},                       # not active
        {"content": "", "tool_calls": [call("nope", {})]},                                # unknown
        {"content": "", "tool_calls": [call("toolbelt_add", {"names": ["tool_00", "boom", "big"]})]},
        {"content": "", "tool_calls": [call("tool_00", {"x": "one"})]},                   # bad type
        {"content": "", "tool_calls": [call("tool_00", "not json at all")]},              # unparseable args
        {"content": "", "tool_calls": [call("boom", {})]},                                # raises
        {"content": "", "tool_calls": [call("big", {})]},                                 # big result → truncated in context
        {"content": "", "tool_calls": [todo(("1", "x", "completed")), call("final_answer", {"status": "failed", "content": "gave up"})]},
    ])
    rt = AgentRuntime(registry_with(2), fake, tmp_path, "fake", RuntimeConfig(result_context_chars=500))
    res = rt.run("t")
    assert res.status == "failed"
    steps = [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]
    assert [s["kind"] for s in steps[:7]] == ["error", "error", "ok", "error", "error", "error", "ok"]
    assert "not active" in steps[0]["result_preview"]
    assert "unknown tool" in steps[1]["result_preview"]
    assert "must be integer" in steps[3]["result_preview"]
    assert "JSON object" in steps[4]["result_preview"]
    assert "kaboom" in steps[5]["result_preview"]
    assert steps[6]["result_bytes"] == 10000
    # in-context copy was truncated; artifact holds the full thing
    last_big = [m for m in fake.requests[7]["messages"] if m["role"] == "tool"][-1]
    assert "truncated at 500" in last_big["content"] and len(last_big["content"]) < 700
    assert (res.run_dir / steps[6]["artifact"]).read_text() == "X" * 10000


def test_transport_error_retries_once_then_fails(tmp_path):
    fake = FakeTransport([TransportError("down"), TransportError("still down")])
    rt = AgentRuntime(registry_with(1), fake, tmp_path, "fake")
    res = rt.run("t")
    assert res.status == "transport_error" and res.steps == 0 and len(fake.requests) == 2
    assert read_trajectory(res.run_dir)[-1]["detail"] == "still down"

    fake2 = FakeTransport([TransportError("blip"), {"content": "", "tool_calls": [
        todo(("1", "x", "completed")), call("final_answer", {"status": "completed", "content": "ok"})]}])
    assert AgentRuntime(registry_with(1), fake2, tmp_path, "fake").run("t").status == "completed"


def test_step_cap_and_default(tmp_path):
    assert RuntimeConfig().step_cap == 250
    fake = FakeTransport([{"content": "looping", "tool_calls": [call("toolbelt_list", {})]} for _ in range(10)])
    rt = AgentRuntime(registry_with(1), fake, tmp_path, "fake", RuntimeConfig(step_cap=4))
    res = rt.run("t")
    assert res.status == "step_cap" and res.steps == 4 and res.final is None
    foot = read_trajectory(res.run_dir)[-1]
    assert foot["status"] == "step_cap" and "4" in foot["detail"]


def test_text_only_turns_nudge_then_stall(tmp_path):
    fake = FakeTransport([{"content": "I think the answer is 4.", "tool_calls": []}] * 5)
    rt = AgentRuntime(registry_with(1), fake, tmp_path, "fake", RuntimeConfig(text_only_limit=3))
    res = rt.run("t")
    assert res.status == "stalled" and res.steps == 3
    steps = [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]
    assert all(s["kind"] == "text_only" and s["tool"] is None for s in steps)
    assert steps[0]["reasoning"] == "I think the answer is 4."
    # nudge was appended after the first two text-only turns
    assert fake.requests[1]["messages"][-1]["role"] == "user"


def test_plain_text_cannot_finish(tmp_path):
    fake = FakeTransport([
        {"content": "Final answer: 4", "tool_calls": []},
        {"content": "", "tool_calls": [todo(("1", "x", "completed")), call("final_answer", {"status": "completed", "content": "4"})]},
    ])
    res = AgentRuntime(registry_with(1), fake, tmp_path, "fake").run("t")
    assert res.status == "completed" and res.steps == 3
