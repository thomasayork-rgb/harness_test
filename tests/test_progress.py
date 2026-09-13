"""Tracking a plan to completion: notes per todo, the progress nudge, and the
todos the footer records.

Everything here runs through the runtime: the notes ride in the todo snapshots
and the footer, and the nudge is a JSONL record plus a message the next request
carries.
"""
import json

from harness.cli import main
from harness.mockserver import MockOpenAIServer
from harness.registry import ToolRegistry, ToolSpec
from harness.replay import compare, replay
from harness.resume import resume
from harness.runtime import AgentRuntime, RuntimeConfig
from harness.todo import NOTES_MAX, apply_update, signature, validate_todos
from harness.trajectory import format_summary, format_trace, read_trajectory, summarize
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


# ---- the progress nudge ---------------------------------------------------


def plan_then_drift(n, notes=None):
    """Plan, then n steps of busywork that never touches the plan."""
    return ([{"content": "Planning.", "tool_calls": [todo_call(("a", "do it", "in_progress"))]},
             {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["echo"]})]}]
            + [{"content": f"echo {i}", "tool_calls": [call("echo", {"s": str(i)})]} for i in range(n)]
            + [{"content": "Closing.", "tool_calls": [todo_call(("a", "do it", "completed", notes or "done"))]},
               {"content": "Done.", "tool_calls": [call("final_answer", {"status": "completed", "content": "ok"})]}])


def test_a_plan_that_stops_moving_gets_one_nudge_and_the_counter_resets(tmp_path):
    res, fake = run(tmp_path, plan_then_drift(8), run_id="nudged",
                    config=RuntimeConfig(progress_nudge_steps=4))
    assert res.status == "completed"

    records = read_trajectory(res.run_dir)
    notes = [r for r in records if r["type"] == "note"]
    # 4 steps after the plan last moved (step 1), and 4 after the nudge reset it
    assert [(n["kind"], n["step"]) for n in notes] == [("progress_nudge", 5), ("progress_nudge", 9)]
    assert "4 steps since your todo list last changed" in notes[0]["text"]
    assert "close the todos and call final_answer" in notes[0]["text"]
    # a note is not a step, and does not count toward the cap
    assert [r["step"] for r in records if r["type"] == "step"] == list(range(1, 13))
    assert records[-1]["steps"] == 12

    # the nudge reached the model on the very next request, as a user message
    requests = fake.requests
    after = requests[5]["messages"][-1]
    assert after["role"] == "user" and after["content"] == notes[0]["text"]
    assert requests[4]["messages"][-1]["role"] == "tool"          # and not before
    # the second nudge counted from the first, not from the start of the run
    assert requests[9]["messages"][-1]["content"] == notes[1]["text"]


def test_a_plan_that_keeps_moving_is_never_nudged(tmp_path):
    """The counter watches status and notes, so an agent that records what it
    did as it goes is left alone - even for the same number of steps."""
    script = [{"content": "Planning.", "tool_calls": [todo_call(("a", "one", "in_progress"),
                                                                ("b", "two", "pending"))]},
              {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["echo"]})]}]
    for i in range(4):
        script += [{"content": f"work {i}", "tool_calls": [call("echo", {"s": str(i)})]},
                   {"content": f"noting {i}", "tool_calls": [
                       todo_call(("a", "one", "in_progress", f"step {i} done"))]}]
    script += [{"content": "Closing.", "tool_calls": [todo_call(("a", "one", "completed"),
                                                                ("b", "two", "cancelled"))]},
               {"content": "Done.", "tool_calls": [call("final_answer", {"status": "completed", "content": "ok"})]}]

    res, _ = run(tmp_path, script, run_id="moving", config=RuntimeConfig(progress_nudge_steps=3))
    records = read_trajectory(res.run_dir)
    assert [r for r in records if r["type"] == "note"] == []
    # the note written while the item was open survived being closed
    assert records[-1]["todos"][0]["notes"] == "step 3 done"


def test_the_nudge_waits_for_a_plan_and_can_be_switched_off(tmp_path):
    """No todo list yet is the todo gate's business, not the nudge's."""
    script = ([{"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["echo"]})]}]
              + [{"content": f"echo {i}", "tool_calls": [call("echo", {"s": str(i)})]} for i in range(6)]
              + [{"content": "Planning late.", "tool_calls": [todo_call(("a", "do it", "completed", "ok"))]},
                 {"content": "Done.", "tool_calls": [call("final_answer", {"status": "completed", "content": "ok"})]}])
    res, _ = run(tmp_path, script, run_id="noplan", config=RuntimeConfig(progress_nudge_steps=3))
    assert [r for r in read_trajectory(res.run_dir) if r["type"] == "note"] == []

    res, _ = run(tmp_path, plan_then_drift(10), run_id="off", config=RuntimeConfig(progress_nudge_steps=0))
    assert [r for r in read_trajectory(res.run_dir) if r["type"] == "note"] == []


def test_text_only_turns_count_toward_the_nudge(tmp_path):
    """A model talking to itself is the clearest case of a plan not moving."""
    script = ([{"content": "Planning.", "tool_calls": [todo_call(("a", "do it", "in_progress"))]}]
              + [{"content": "Thinking out loud.", "tool_calls": []} for _ in range(2)]
              + [{"content": "Closing.", "tool_calls": [todo_call(("a", "do it", "completed", "ok"))]},
                 {"content": "Done.", "tool_calls": [call("final_answer", {"status": "completed", "content": "ok"})]}])
    res, fake = run(tmp_path, script, run_id="quiet",
                    config=RuntimeConfig(progress_nudge_steps=2, text_only_limit=5))
    notes = [r for r in read_trajectory(res.run_dir) if r["type"] == "note"]
    assert [n["step"] for n in notes] == [3]
    assert [m["role"] for m in fake.requests[3]["messages"][-2:]] == ["user", "user"]   # after the text-only nudge


# ---- the auditable finish -------------------------------------------------


def test_the_footer_records_the_todos_with_their_notes(tmp_path):
    script = [
        {"content": "Planning.", "tool_calls": [todo_call(("find", "find the port", "in_progress"),
                                                          ("fix", "fix the config", "pending"))]},
        {"content": "Found it.", "tool_calls": [todo_call(("find", "find the port", "completed",
                                                           "8080, conf/config.ini:2"))]},
        {"content": "Not doing the second one.", "tool_calls": [
            todo_call(("fix", "fix the config", "cancelled", "not needed: the port was already right"))]},
        {"content": "Done.", "tool_calls": [
            call("final_answer", {"status": "completed", "content": "The port is 8080."})]},
    ]
    res, _ = run(tmp_path, script, run_id="audit")
    footer = read_trajectory(res.run_dir)[-1]
    assert footer["todos"] == [
        {"id": "find", "content": "find the port", "status": "completed", "notes": "8080, conf/config.ini:2"},
        {"id": "fix", "content": "fix the config", "status": "cancelled",
         "notes": "not needed: the port was already right"}]


def test_a_run_that_never_finished_still_records_the_todos_it_had(tmp_path):
    script = [{"content": "Planning.", "tool_calls": [todo_call(("a", "do it", "in_progress", "started"))]},
              {"content": "Looping.", "tool_calls": [call("toolbelt_list", {})]},
              {"content": "Looping.", "tool_calls": [call("toolbelt_list", {})]}]
    res, _ = run(tmp_path, script, run_id="capped", config=RuntimeConfig(step_cap=2))
    assert res.status == "step_cap"
    footer = read_trajectory(res.run_dir)[-1]
    assert footer["final"] is None
    assert footer["todos"] == [{"id": "a", "content": "do it", "status": "in_progress", "notes": "started"}]


def test_summary_and_trace_show_the_nudges_and_the_closed_plan(tmp_path, capsys):
    res, _ = run(tmp_path, plan_then_drift(8, notes="echoed 0-7, all returned their input"),
                 run_id="readable", config=RuntimeConfig(progress_nudge_steps=4))
    records = read_trajectory(res.run_dir)

    s = summarize(records)
    assert s["progress_nudges"] == 2 and s["steps"] == 12
    assert s["todos"] == [{"id": "a", "content": "do it", "status": "completed",
                           "notes": "echoed 0-7, all returned their input"}]

    text = format_summary(records)
    assert "progress nudges: 2" in text
    assert "todos:\n  [completed] a: do it\n      note: echoed 0-7, all returned their input" in text

    trace = format_trace(records, res.run_dir)
    assert "-- progress_nudge after step 5:" in trace and "since your todo list last changed" in trace
    assert trace.index("-- progress_nudge after step 5") > trace.index("   5  ok")   # at the seam

    rc = main(["--runs-dir", str(tmp_path / "runs"), "trace", "readable", "--summary"])
    out = capsys.readouterr().out
    assert rc == 0 and "progress nudges: 2" in out and "note: echoed 0-7" in out


def test_a_trajectory_with_a_note_still_resumes_and_replays(tmp_path):
    """The note record sits between steps in the same file: everything that
    reads a trajectory back has to walk past it."""
    script = plan_then_drift(8)[:6]          # plan, activate, four echoes: capped mid-drift
    res, _ = run(tmp_path, script, run_id="withnote",
                 config=RuntimeConfig(progress_nudge_steps=3, step_cap=6))
    assert res.status == "step_cap"
    records = read_trajectory(res.run_dir)
    assert [r["type"] for r in records] == [
        "header", "step", "step", "step", "step", "note", "step", "step", "footer"]

    rest = FakeTransport([
        {"content": "Closing.", "tool_calls": [todo_call(("a", "do it", "completed", "echoed"))]},
        {"content": "Done.", "tool_calls": [call("final_answer", {"status": "completed", "content": "ok"})]}])
    out = resume(res.run_dir, registry(), rest, step_cap=10)
    assert out.status == "completed" and out.steps == 8
    resumed = read_trajectory(res.run_dir)
    assert summarize(resumed)["progress_nudges"] == 1
    assert summarize(resumed)["todos"] == [{"id": "a", "content": "do it", "status": "completed",
                                            "notes": "echoed"}]
    assert "-- progress_nudge after step 4" in format_trace(resumed, res.run_dir)

    # and the whole thing replays, notes and all, step for step
    replayed = replay(res.run_dir, registry(), runs_dir=tmp_path / "runs")
    assert compare(res.run_dir, replayed.run_dir) == []
    assert [r["type"] for r in read_trajectory(replayed.run_dir)].count("note") == 1


def test_progress_nudge_is_a_flag_on_run_and_on_resume(tmp_path):
    runs, work = tmp_path / "runs", tmp_path / "project"
    work.mkdir()
    script = [{"content": "Planning.", "tool_calls": [todo_call(("a", "do it", "in_progress"))]},
              {"content": "Listing.", "tool_calls": [call("toolbelt_list", {})]},
              {"content": "Listing.", "tool_calls": [call("toolbelt_list", {})]}]
    with MockOpenAIServer(script, model="mock-model") as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "mock-model",
                   "--endpoint", server.base_url, "--workdir", str(work), "--run-id", "flag",
                   "--progress-nudge", "2", "--step-cap", "3"])
    assert rc == 3
    records = read_trajectory(runs / "flag")
    assert records[0]["config"]["progress_nudge_steps"] == 2
    assert [(r["kind"], r["step"]) for r in records if r["type"] == "note"] == [("progress_nudge", 3)]

    rest = [{"content": "Closing.", "tool_calls": [todo_call(("a", "do it", "completed", "listed"))]},
            {"content": "Done.", "tool_calls": [call("final_answer", {"status": "completed", "content": "ok"})]}]
    with MockOpenAIServer(rest, model="mock-model") as server:
        rc = main(["--runs-dir", str(runs), "resume", "flag", "--endpoint", server.base_url,
                   "--step-cap", "8", "--progress-nudge", "0"])
    assert rc == 0
    records = read_trajectory(runs / "flag")
    seam = next(r for r in records if r["type"] == "resume")
    assert seam["config"]["progress_nudge_steps"] == 0          # the flag reached the new segment
    assert len([r for r in records if r["type"] == "note"]) == 1
