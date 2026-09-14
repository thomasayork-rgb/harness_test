import json

from harness.cli import main
from harness.context import EVICTED, SUPERSEDED, ContextBudget
from harness.registry import ToolRegistry, ToolSpec
from harness.runtime import AgentRuntime, RuntimeConfig
from harness.trajectory import format_summary, format_trace, read_trajectory, summarize
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


def test_the_current_turns_results_are_never_evicted(tmp_path):
    """A budget too small to meet - a long system prompt, a tight --context-chars -
    must not leave the model staring at a placeholder where the result it just
    asked for should be. Its only move then is to make the same call again."""
    r = ToolRegistry()
    r.register(ToolSpec("big", "big", {"type": "object", "properties": {}, "required": []}, lambda: "B" * 500))
    fake = FakeTransport([
        {"content": "add", "tool_calls": [call("toolbelt_add", {"names": ["big"]})]},
        {"content": "one", "tool_calls": [call("big", {})]},
        {"content": "two at once", "tool_calls": [call("big", {}), call("big", {})]},
        {"content": "done", "tool_calls": [call("todo_write", {"todos": [{"id": "1", "content": "x", "status": "completed"}]}),
                                           call("final_answer", {"status": "completed", "content": "ok"})]},
    ])
    cfg = RuntimeConfig(result_context_chars=1000, context_budget_chars=10)   # unmeetable
    res = AgentRuntime(r, fake, tmp_path, "fake", cfg).run("t")
    assert res.status == "completed"

    tool_msgs = [m for m in fake.requests[-1]["messages"] if m["role"] == "tool"]
    assert [m["content"] for m in tool_msgs[-2:]] == ["B" * 500, "B" * 500]   # both calls of the turn
    assert all(m["content"].startswith("[result evicted") for m in tool_msgs[:-2])


PLAN = [{"id": f"t{i:02d}", "content": f"step {i}: check the thing and record what it said",
         "status": "pending"} for i in range(12)]


def keep_the_plan_moving(updates: int) -> list[dict]:
    """One todo_write per update, each answering with the whole 12-item list."""
    script = [{"content": "Planning.", "tool_calls": [call("todo_write", {"todos": PLAN})]}]
    for n in range(updates):
        i = n % len(PLAN)
        script.append({"content": f"Update {n}.", "tool_calls": [call("todo_write", {"todos": [
            {"id": PLAN[i]["id"], "content": PLAN[i]["content"], "status": "in_progress",
             "notes": f"round {n}: " + "y" * 60}]}, id=f"t{n}")]})
    script.append({"content": "Closing everything.", "tool_calls": [
        call("todo_write", {"todos": [dict(t, status="completed") for t in PLAN]}, id="close"),
        call("final_answer", {"status": "completed", "content": "ok"}, id="fin")]})
    return script


def test_only_the_newest_todo_list_holds_its_place_in_context(tmp_path):
    """Every todo_write result is protected and carries the whole list, so 40
    updates to a 12-item plan used to leave 80k+ chars of unevictable copies
    against a 60k budget. Only the newest one is the plan."""
    fake = FakeTransport(keep_the_plan_moving(40))
    rt = AgentRuntime(ToolRegistry(), fake, tmp_path / "runs", "fake", run_id="plan")
    res = rt.run("keep the plan moving")
    assert res.status == "completed" and res.steps == 43

    budget = rt.config.context_budget_chars
    assert rt.budget.size(rt.state.messages) <= budget          # under budget at the end
    protected = [m for m in rt.state.messages if m.get("_protected") and m["role"] == "tool"]
    assert len(protected) == 1                                  # the list it just wrote
    assert rt.budget.protected_size(rt.state.messages) == len(protected[0]["content"])
    assert protected[0]["content"].startswith('{\n  "todos"')     # the list it just wrote
    assert protected[0]["_artifact"].endswith("_todo_write.txt")

    # what the model was actually sent on its last turn: one list, 40 pointers
    tool_msgs = [m for m in fake.requests[-1]["messages"] if m["role"] == "tool"]
    lists = [m for m in tool_msgs if m["content"].startswith("{")]
    markers = [m for m in tool_msgs if m["content"].startswith("[superseded")]
    assert len(lists) == 1 and len(markers) == 40
    assert sum(len(m.get("content") or "") for m in fake.requests[-1]["messages"]) <= budget
    assert not any(k.startswith("_") for m in tool_msgs for k in m)

    # each pointer names the artifact that still holds that version in full
    steps = [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]
    assert markers[0]["content"] == SUPERSEDED.format(label="todo list", artifact=steps[0]["artifact"])
    superseded = json.loads((res.run_dir / steps[0]["artifact"]).read_text(encoding="utf-8"))
    assert len(superseded["todos"]) == 12 and len(superseded["open"]) == 12
    # and the trajectory still records every version, whatever context kept
    assert sum(1 for s in steps if s["tool"] == "todo_write") == 42


def test_a_rejected_todo_write_does_not_supersede_the_plan(tmp_path):
    """A todo_write that never produced a list must not push the real one out
    of context: the model would be left planning against a pointer."""
    fake = FakeTransport([
        {"content": "Planning.", "tool_calls": [call("todo_write", {"todos": [
            {"id": "a", "content": "do it", "status": "in_progress"}]}, id="good")]},
        {"content": "A malformed update.", "tool_calls": [
            call("todo_write", {"todos": [{"id": "a", "status": "completed"}]}, id="bad")]},
        {"content": "Properly, then.", "tool_calls": [
            call("todo_write", {"todos": [{"id": "a", "content": "do it", "status": "completed"}]}, id="ok"),
            call("final_answer", {"status": "completed", "content": "done"}, id="fin")]},
    ])
    res = AgentRuntime(ToolRegistry(), fake, tmp_path / "runs", "fake", run_id="reject").run("plan")
    assert res.status == "completed"

    steps = [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]
    assert [s["kind"] for s in steps] == ["ok", "error", "ok", "final_accepted"]
    # on the turn after the bad call the good list was still there in full
    tool_msgs = [m for m in fake.requests[2]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["open"] == ["a"]
    assert "must be a non-empty string" in tool_msgs[1]["content"]


def test_protected_content_over_the_budget_is_said_once(tmp_path):
    """Eviction cannot touch protected results, so a budget they alone exceed
    is a fact the model has to act on - said once, not every step."""
    r = ToolRegistry()
    r.register(ToolSpec("echo", "echo", {"type": "object", "properties": {}, "required": []},
                        lambda: "done"))
    script = [{"content": "Planning.", "tool_calls": [call("todo_write", {"todos": PLAN})]},
              {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["echo"]})]}]
    script += [{"content": f"Working {i}.", "tool_calls": [call("echo", {})]} for i in range(3)]
    script.append({"content": "Closing.", "tool_calls": [
        call("todo_write", {"todos": [dict(t, status="completed") for t in PLAN]}),
        call("final_answer", {"status": "completed", "content": "ok"})]})
    cfg = RuntimeConfig(context_budget_chars=500, progress_nudge_steps=0)
    fake = FakeTransport(script)
    res = AgentRuntime(r, fake, tmp_path / "runs", "fake", cfg, run_id="over").run("plan big")
    assert res.status == "completed"

    records = read_trajectory(res.run_dir)
    notes = [r for r in records if r["type"] == "note"]
    assert [(n["kind"], n["step"]) for n in notes] == [("budget", 1)]      # once, after the plan
    assert sorted(notes[0]) == ["kind", "run_id", "step", "text", "ts", "type"]  # the nudge's shape
    assert "chars of protected results" in notes[0]["text"] and "skill_unload" in notes[0]["text"]
    assert "over the 500 char context budget" in notes[0]["text"]

    # the model was told, on the very next request, and only then
    after = [m for m in fake.requests[1]["messages"] if m["role"] == "user"]
    assert after[-1]["content"] == notes[0]["text"]
    assert sum(m["content"] == notes[0]["text"]
               for m in fake.requests[-1]["messages"] if m["role"] == "user") == 1
    assert summarize(records)["budget_warnings"] == 1
    assert "budget warnings: 1" in format_summary(records)
    assert "-- budget after step 1:" in format_trace(records, res.run_dir)


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


def test_trace_step_points_at_the_turn_it_shares(tmp_path, capsys):
    """Only the first call of a turn carries reasoning and usage. The later
    ones used to print `reasoning: (none)` and `tokens: None / None`, which
    reads like the model said nothing rather than like a shared turn."""
    r = ToolRegistry()
    r.register(ToolSpec("echo", "echo", {"type": "object", "properties": {"s": {"type": "string"}},
                                         "required": ["s"]}, lambda s: s))
    fake = FakeTransport([
        {"content": "Adding echo.", "tool_calls": [call("toolbelt_add", {"names": ["echo"]})],
         "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
        {"content": "Three echoes in one turn.",
         "tool_calls": [call("echo", {"s": "one"}), call("echo", {"s": "two"}), call("echo", {"s": "three"})],
         "usage": {"prompt_tokens": 20, "completion_tokens": 2}},
        {"content": "Done.", "tool_calls": [
            call("todo_write", {"todos": [{"id": "1", "content": "x", "status": "completed"}]}),
            call("final_answer", {"status": "completed", "content": "ok"})]},
    ])
    res = AgentRuntime(r, fake, tmp_path, "fake", run_id="turns").run("echo things")
    assert res.status == "completed" and res.steps == 6

    assert main(["--runs-dir", str(tmp_path), "trace", "turns", "--step", "4"]) == 0
    out = capsys.readouterr().out
    assert "step 4  [ok]  echo" in out
    assert "turn: call 3 of the turn that started at step 2" in out
    assert "tokens in/out: recorded on step 2" in out
    assert "(recorded on step 2)" in out and "(none)" not in out
    assert "None" not in out

    # the first call of a turn still shows its own reasoning and usage
    assert main(["--runs-dir", str(tmp_path), "trace", "turns", "--step", "2"]) == 0
    out = capsys.readouterr().out
    assert "tokens in/out: 20 / 2" in out and "Three echoes in one turn." in out
    assert "turn:" not in out

    # and the step list says so too, instead of leaving the column blank
    listing = format_trace(read_trajectory(res.run_dir), res.run_dir).splitlines()
    rows = [line for line in listing if line[:6].strip().isdigit()]
    assert rows[1].endswith("Three echoes in one turn.")
    assert rows[2].endswith("(continues the turn of step 2)")
    assert rows[3].endswith("(continues the turn of step 2)")
