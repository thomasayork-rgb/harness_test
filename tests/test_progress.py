"""Tracking a plan to completion: notes per todo, the progress nudge, and the
todos the footer records.

Everything here runs through the runtime: the notes ride in the todo snapshots
and the footer, and the nudge is a JSONL record plus a message the next request
carries.
"""
import json

from harness.registry import ToolRegistry, ToolSpec
from harness.runtime import AgentRuntime, RuntimeConfig
from harness.todo import NOTES_MAX, apply_update, signature, validate_todos
from harness.trajectory import read_trajectory
from harness.transport import FakeTransport, call


def registry():
    r = ToolRegistry()
    r.register(ToolSpec("echo", "Echo a string.",
                        {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]},
                        lambda s: s))
    return r


def todo_call(*items, merge=True):
    """``todo_call(("find", "find the port", "completed", "8080 in config.ini:2"))``."""
    todos = []
    for item in items:
        todo = {"id": item[0], "content": item[1], "status": item[2]}
        if len(item) > 3:
            todo["notes"] = item[3]
        todos.append(todo)
    return call("todo_write", {"todos": todos, "merge": merge})


def run(tmp_path, script, run_id="notes", config=None):
    fake = FakeTransport(script)
    rt = AgentRuntime(registry(), fake, tmp_path / "runs", "fake", config or RuntimeConfig(),
                      run_id=run_id)
    return rt.run("track this"), fake


def steps_of(res):
    return [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]


# ---- notes ----------------------------------------------------------------


def test_notes_are_validated_like_everything_else():
    assert validate_todos([{"id": "a", "content": "c", "status": "pending", "notes": "found it"}]) is None
    assert validate_todos([{"id": "a", "content": "c", "status": "pending"}]) is None
    assert validate_todos([{"id": "a", "content": "c", "status": "pending", "notes": 7}]) == \
        "todos[0].notes must be a string"
    assert validate_todos([{"id": "a", "content": "c", "status": "pending", "notes": "x" * (NOTES_MAX + 1)}]) == \
        f"todos[0].notes is {NOTES_MAX + 1} chars; the limit is {NOTES_MAX}"
    assert validate_todos([{"id": "a", "content": "c", "status": "pending", "notes": "x" * NOTES_MAX}]) is None


def test_a_note_survives_an_update_that_does_not_mention_it():
    current = [{"id": "a", "content": "find the port", "status": "in_progress", "notes": "8080"},
               {"id": "b", "content": "report it", "status": "pending"}]

    closed = apply_update(current, [{"id": "a", "content": "find the port", "status": "completed"}], True)
    assert closed[0] == {"id": "a", "content": "find the port", "status": "completed", "notes": "8080"}
    assert closed[1] == current[1]                      # untouched items keep their place

    retold = apply_update(current, [{"id": "a", "content": "find the port", "status": "completed",
                                     "notes": "  8080, conf/config.ini:2  "}], True)
    assert retold[0]["notes"] == "8080, conf/config.ini:2"        # trimmed, and it replaces
    cleared = apply_update(current, [{"id": "a", "content": "find the port", "status": "completed",
                                      "notes": ""}], True)
    assert "notes" not in cleared[0]                     # an empty string is how you clear one

    replaced = apply_update(current, [{"id": "a", "content": "find the port", "status": "completed"}], False)
    assert replaced == [{"id": "a", "content": "find the port", "status": "completed"}]

    assert signature(current) == (("a", "in_progress", "8080"), ("b", "pending", ""))
    assert signature(closed) != signature(current)       # closing an item is movement
    assert signature(retold) != signature(closed)        # so is only changing a note


def test_notes_reach_the_snapshots_the_state_and_the_model(tmp_path):
    script = [
        {"content": "Planning.", "tool_calls": [
            todo_call(("find", "find the port", "in_progress"), ("report", "report it", "pending"))]},
        {"content": "Found it; recording what I found.", "tool_calls": [
            todo_call(("find", "find the port", "completed", "8080, from conf/config.ini:2"))]},
        {"content": "Closing the second one.", "tool_calls": [
            todo_call(("report", "report it", "completed", "answered in final_answer"))]},
        {"content": "Done.", "tool_calls": [call("final_answer", {"status": "completed", "content": "8080"})]},
    ]
    res, fake = run(tmp_path, script)
    assert res.status == "completed"

    steps = steps_of(res)
    assert steps[1]["todo_snapshot"] == [
        {"id": "find", "content": "find the port", "status": "completed",
         "notes": "8080, from conf/config.ini:2"},
        {"id": "report", "content": "report it", "status": "pending"}]
    # the note the model wrote at step 2 is still on the item at step 3
    assert steps[2]["todo_snapshot"][0]["notes"] == "8080, from conf/config.ini:2"

    state = json.loads((res.run_dir / "state.json").read_text())
    assert [t.get("notes") for t in state["todos"]] == ["8080, from conf/config.ini:2",
                                                        "answered in final_answer"]
    # and the model can read its own notes back: the todo result is never evicted
    result = json.loads([m for m in fake.requests[-1]["messages"] if m["role"] == "tool"][-1]["content"])
    assert result["todos"][1]["notes"] == "answered in final_answer" and result["open"] == []


def test_a_bad_note_is_an_error_the_model_can_fix(tmp_path):
    script = [
        {"content": "Planning with an essay for a note.", "tool_calls": [
            todo_call(("a", "do it", "in_progress", "x" * 600))]},
        {"content": "Shorter, then.", "tool_calls": [todo_call(("a", "do it", "completed", "did it"))]},
        {"content": "Done.", "tool_calls": [call("final_answer", {"status": "completed", "content": "ok"})]},
    ]
    res, _ = run(tmp_path, script, run_id="badnote")
    steps = steps_of(res)
    assert steps[0]["kind"] == "error" and "the limit is 500" in steps[0]["result_preview"]
    assert steps[0]["todo_snapshot"] == []               # nothing was persisted
    assert steps[1]["todo_snapshot"][0]["notes"] == "did it"
    assert res.status == "completed"
