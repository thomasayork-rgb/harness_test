"""Resuming an interrupted run: same run id, same directory, one trajectory."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.cli import main
from harness.mockserver import HttpError, MockOpenAIServer
from harness.registry import ToolRegistry, ToolSpec
from harness.replay import compare, replay
from harness.resume import load, prepare, resume
from harness.runtime import AgentRuntime, ResumeError, RuntimeConfig
from harness.trajectory import format_summary, format_trace, read_trajectory, summarize
from harness.transport import FakeTransport, TransportError, call

REPO_ROOT = Path(__file__).resolve().parents[1]


def registry():
    r = ToolRegistry()
    r.register(ToolSpec("echo", "Echo a string.",
                        {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]},
                        lambda s: s))
    return r


def todo(status):
    return call("todo_write", {"todos": [{"id": "1", "content": "echo things", "status": status}]},
                id=f"todo_{status}")


FINISH = call("final_answer", {"status": "completed", "content": "all echoed"}, id="fin")


def capped_run(tmp_path, run_id="capped"):
    """A run that trips the step cap in the middle of a three-call turn."""
    script = [
        {"content": "Activating echo.", "tool_calls": [call("toolbelt_add", {"names": ["echo"]}, id="a")],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"content": "Planning.", "tool_calls": [todo("in_progress")],
         "usage": {"prompt_tokens": 110, "completion_tokens": 11}},
        {"content": "Three echoes at once.",
         "tool_calls": [call("echo", {"s": "one"}, id="e1"), call("echo", {"s": "two"}, id="e2"),
                        call("echo", {"s": "three"}, id="e3")],
         "usage": {"prompt_tokens": 120, "completion_tokens": 12}},
    ]
    rt = AgentRuntime(registry(), FakeTransport(script), tmp_path / "runs", "fake-model",
                      RuntimeConfig(step_cap=4), run_id=run_id)
    res = rt.run("echo some things")
    assert res.status == "step_cap" and res.steps == 4
    return res


def test_resume_after_the_step_cap_continues_the_same_trajectory(tmp_path):
    first = capped_run(tmp_path)
    rest = FakeTransport([
        {"content": "Closing the todo.", "tool_calls": [todo("completed")],
         "usage": {"prompt_tokens": 200, "completion_tokens": 20}},
        {"content": "Finishing.", "tool_calls": [FINISH],
         "usage": {"prompt_tokens": 210, "completion_tokens": 21}},
    ])
    res = resume(first.run_dir, registry(), rest, step_cap=10)

    assert res.run_id == "capped" and res.run_dir == first.run_dir
    assert res.status == "completed" and res.steps == 6
    assert res.final == {"status": "completed", "content": "all echoed"}

    # one file, one header, a resume seam, a footer per segment
    recs = read_trajectory(res.run_dir)
    assert [r["type"] for r in recs] == [
        "header", "step", "step", "step", "step", "footer", "resume", "step", "step", "footer"]
    seam = recs[6]
    assert seam["from_status"] == "step_cap" and seam["from_step"] == 4 and seam["step_cap"] == 10
    assert "step cap 4 reached" in seam["from_detail"] and seam["model"] == "fake-model"
    assert seam["config"]["step_cap"] == 10 and seam["harness_version"] == recs[0]["harness_version"]
    steps = [r for r in recs if r["type"] == "step"]
    assert [s["step"] for s in steps] == [1, 2, 3, 4, 5, 6]      # the counter carries on
    assert [s["tool"] for s in steps] == ["toolbelt_add", "todo_write", "echo", "echo",
                                          "todo_write", "final_answer"]
    assert recs[5]["status"] == "step_cap" and recs[-1]["status"] == "completed"

    # the second segment picked up the tools, todos and conversation of the first
    opening = rest.requests[0]["messages"]
    assert opening[0]["role"] == "system" and opening[1]["content"] == "echo some things"
    assert opening[-1]["role"] == "user" and "has been resumed" in opening[-1]["content"]
    assert "step_cap" in opening[-1]["content"]
    requested = [tc["id"] for m in opening if m.get("tool_calls") for tc in m["tool_calls"]]
    answered = [m["tool_call_id"] for m in opening if m["role"] == "tool"]
    assert requested == ["a", "todo_in_progress", "e1", "e2", "e3"] == answered   # e3 never ran
    assert "echo" in {t["function"]["name"] for t in rest.requests[0]["tools"]}
    assert not any(k.startswith("_") for m in opening for k in m)

    state = json.loads((res.run_dir / "state.json").read_text())
    assert state["status"] == "completed" and state["step"] == 6 and state["active_tools"] == ["echo"]
    assert [s["step"] for s in steps] == sorted({int((res.run_dir / s["artifact"]).stem.split("_")[1])
                                                 for s in steps})


def test_summary_and_trace_of_a_resumed_run_read_sensibly(tmp_path):
    first = capped_run(tmp_path, run_id="readable")
    resume(first.run_dir, registry(), FakeTransport([
        {"content": "Closing.", "tool_calls": [todo("completed")], "usage": {"prompt_tokens": 200, "completion_tokens": 20}},
        {"content": "Finishing.", "tool_calls": [FINISH], "usage": {"prompt_tokens": 210, "completion_tokens": 21}},
    ]), step_cap=10)

    recs = read_trajectory(first.run_dir)
    s = summarize(recs)
    assert s["status"] == "completed" and s["steps"] == 6 and s["segments"] == 2
    assert s["resumed_from"] == ["step_cap"] and s["step_cap"] == 10
    assert s["tools"] == {"echo": 2, "final_answer": 1, "todo_write": 2, "toolbelt_add": 1}
    assert s["tokens_in"] == 100 + 110 + 120 + 200 + 210      # both segments, once per turn
    assert s["final"] == {"status": "completed", "content": "all echoed"}
    assert s["wall_s"] is not None and s["wall_s"] >= 0

    text = format_summary(recs)
    assert "segments: 2  resumed from: step_cap" in text and "status: completed  steps: 6" in text

    trace = format_trace(recs, first.run_dir)
    assert "-- resumed at step 4 after step_cap" in trace
    assert trace.index("status: step_cap") < trace.index("-- resumed") < trace.index("status: completed")
    assert "   5  ok" in trace and "   6  final_accepted" in trace


def test_resume_after_a_transport_error(tmp_path):
    rt = AgentRuntime(registry(), FakeTransport([TransportError("down"), TransportError("still down")]),
                      tmp_path / "runs", "fake-model", RuntimeConfig(step_cap=20), run_id="died")
    first = rt.run("echo some things")
    assert first.status == "transport_error" and first.steps == 0

    rest = FakeTransport([
        {"content": "Back. Planning.", "tool_calls": [todo("in_progress")]},
        {"content": "Closing.", "tool_calls": [todo("completed")]},
        {"content": "Finishing.", "tool_calls": [FINISH]},
    ])
    # the two-step API: rebuild the runtime, then drive it
    rt, detail = prepare(first.run_dir, registry(), rest)
    assert rt.run_id == "died" and detail == "still down" and rt.state.status == "transport_error"
    assert rt.config.step_cap == 20 and rt.state.messages[-1]["content"] == "echo some things"
    res = rt.resume(detail)
    assert res.status == "completed" and res.steps == 3
    recs = read_trajectory(res.run_dir)
    assert [r["type"] for r in recs] == ["header", "footer", "resume", "step", "step", "step", "footer"]
    assert recs[2]["from_detail"] == "still down" and recs[2]["from_step"] == 0
    assert "transport_error: still down" in rest.requests[0]["messages"][-1]["content"]
    # the cap came out of the recorded config, not the default
    assert recs[2]["config"]["step_cap"] == 20


def test_resume_after_a_stall(tmp_path):
    rt = AgentRuntime(registry(), FakeTransport([{"content": "Thinking.", "tool_calls": []}] * 3),
                      tmp_path / "runs", "fake-model", RuntimeConfig(text_only_limit=3), run_id="quiet")
    first = rt.run("echo some things")
    assert first.status == "stalled"

    rest = FakeTransport([{"content": "Right, tools.", "tool_calls": [todo("completed"), FINISH]}])
    res = resume(first.run_dir, registry(), rest)
    assert res.status == "completed" and res.steps == 5
    # a stalled run ends on an assistant turn; the resume note keeps the
    # transcript ending on a user turn, which is what every provider wants
    msgs = rest.requests[0]["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-1]["role"] == "user"
    assert "stalled" in msgs[-1]["content"]


def test_finished_runs_are_not_resumable(tmp_path):
    script = [{"content": "Done.", "tool_calls": [todo("completed"), FINISH]}]
    done = AgentRuntime(registry(), FakeTransport(script), tmp_path / "runs", "fake-model",
                        run_id="done").run("echo some things")
    assert done.status == "completed"
    with pytest.raises(ResumeError, match="ended with status 'completed'"):
        resume(done.run_dir, registry(), FakeTransport([]))

    state, records = load(done.run_dir)
    assert state.status == "completed" and records[-1]["type"] == "footer"


def test_resume_at_the_cap_needs_a_higher_cap(tmp_path):
    first = capped_run(tmp_path, run_id="stuck")
    with pytest.raises(ResumeError, match="raise it with --step-cap"):
        resume(first.run_dir, registry(), FakeTransport([]))
    # and nothing was written: the trajectory still ends at the first footer
    recs = read_trajectory(first.run_dir)
    assert [r["type"] for r in recs].count("footer") == 1
    assert not any(r["type"] == "resume" for r in recs)


def test_missing_run_is_a_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        resume(tmp_path / "runs" / "nope", registry(), FakeTransport([]))
    (tmp_path / "runs" / "half").mkdir(parents=True)
    (tmp_path / "runs" / "half" / "trajectory.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no state at"):
        resume(tmp_path / "runs" / "half", registry(), FakeTransport([]))


def test_a_resumed_recording_replays_as_one_run(tmp_path):
    first = capped_run(tmp_path, run_id="tape")
    resume(first.run_dir, registry(), FakeTransport([
        {"content": "Closing.", "tool_calls": [todo("completed")]},
        {"content": "Finishing.", "tool_calls": [FINISH]},
    ]), step_cap=10)

    res = replay(first.run_dir, registry())
    assert res.status == "completed" and res.steps == 6      # not stopped by the original cap of 4
    assert compare(first.run_dir, res.run_dir) == []
    assert [r["type"] for r in read_trajectory(res.run_dir)].count("footer") == 1


def test_resume_cli_over_http(tmp_path, capsys):
    runs = tmp_path / "runs"
    work = tmp_path / "project"
    work.mkdir()
    (work / "config.ini").write_text("[server]\nPORT = 8080\n", encoding="utf-8")

    first = [
        {"content": "Activating search.", "tool_calls": [call("toolbelt_add", {"names": ["fs_search"]})],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"content": "Planning, then searching in the same turn.",
         "tool_calls": [call("todo_write", {"todos": [
             {"id": "find", "content": "find the port", "status": "in_progress"}]}),
             call("fs_search", {"pattern": "PORT"}),
             call("fs_search", {"pattern": "host"})],
         "usage": {"prompt_tokens": 200, "completion_tokens": 20}},
    ]
    with MockOpenAIServer(first, model="mock-model") as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "find the port", "--model", "mock-model",
                   "--endpoint", server.base_url, "--workdir", str(work), "--run-id", "half",
                   "--step-cap", "3"])
    assert rc == 3                                        # step_cap
    assert server.served == 2

    rest = [
        {"content": "Closing the todo.", "tool_calls": [call("todo_write", {"todos": [
            {"id": "find", "content": "find the port", "status": "completed"}]})],
         "usage": {"prompt_tokens": 300, "completion_tokens": 30}},
        {"content": "Finishing.", "tool_calls": [call("final_answer", {
            "status": "completed", "content": "The port is 8080."})],
         "usage": {"prompt_tokens": 400, "completion_tokens": 40}},
    ]
    with MockOpenAIServer(rest, model="mock-model") as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "resume", "half",
             "--endpoint", server.base_url, "--workdir", str(work), "--step-cap", "12",
             "--extra-body", '{"temperature": 0}'],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
        bodies = [r["body"] for r in server.requests]

    assert proc.returncode == 0, proc.stderr
    assert "The port is 8080." in proc.stdout
    assert "resume half at step 3 after step_cap" in proc.stderr
    assert "status: completed  steps: 5" in proc.stderr

    recs = read_trajectory(runs / "half")
    assert [r["type"] for r in recs] == [
        "header", "step", "step", "step", "footer", "resume", "step", "step", "footer"]
    steps = [r for r in recs if r["type"] == "step"]
    assert [(s["step"], s["tool"], s["kind"]) for s in steps] == [
        (1, "toolbelt_add", "ok"), (2, "todo_write", "ok"), (3, "fs_search", "ok"),
        (4, "todo_write", "ok"), (5, "final_answer", "final_accepted")]
    assert recs[-1]["final"]["content"] == "The port is 8080."
    assert summarize(recs)["segments"] == 2

    # the model that resumed the run was handed a well-formed transcript: the
    # third call of the interrupted turn is answered even though it never ran
    msgs = bodies[0]["messages"]
    answered = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
    requested = [tc["id"] for m in msgs if m.get("tool_calls") for tc in m["tool_calls"]]
    assert requested == answered and len(requested) == 4
    assert "not executed" in msgs[-2]["content"] and msgs[-1]["role"] == "user"
    assert bodies[0]["model"] == "mock-model"              # the model came back out of state.json
    assert all(b["temperature"] == 0 for b in bodies)
    assert {t["function"]["name"] for t in bodies[0]["tools"]} >= {"fs_search"}

    # a finished run refuses to resume, and says so
    rc = main(["--runs-dir", str(runs), "resume", "half", "--endpoint", "http://127.0.0.1:1/v1"])
    assert rc == 64 and "ended with status 'completed'" in capsys.readouterr().err
    rc = main(["--runs-dir", str(runs), "resume", "nope", "--endpoint", "http://127.0.0.1:1/v1"])
    assert rc == 66 and "no trajectory at" in capsys.readouterr().err


def test_resume_keeps_going_when_the_endpoint_is_still_down(tmp_path):
    runs = tmp_path / "runs"
    with MockOpenAIServer([HttpError(503, "still cold")] * 4) as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "m",
                   "--endpoint", server.base_url, "--workdir", str(tmp_path), "--run-id", "cold"])
        assert rc == 2
        rc = main(["--runs-dir", str(runs), "resume", "cold", "--endpoint", server.base_url,
                   "--workdir", str(tmp_path)])
    assert rc == 2                                          # transport_error again, resumable again
    recs = read_trajectory(runs / "cold")
    assert [r["type"] for r in recs] == ["header", "footer", "resume", "footer"]
    assert server.served == 4                               # two attempts per segment
    assert summarize(recs)["segments"] == 2 and summarize(recs)["status"] == "transport_error"


SHOUT_PLUGIN = '''
def register(registry, workdir):
    @registry.tool("shout", "Upper-case a string.",
                   {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]})
    def shout(s: str) -> str:
        return s.upper()
'''

SECRET = "sekret"


def test_resume_defaults_every_provider_flag_from_the_recording(tmp_path):
    """The run id and the api key are enough: endpoint, provider, workdir,
    --tools, --extra-body and the policy all come back out of the recording."""
    runs, work = tmp_path / "runs", tmp_path / "project"
    work.mkdir()
    (work / "config.ini").write_text("[server]\nPORT = 8080\n", encoding="utf-8")
    plugin = tmp_path / "shout_plugin.py"
    plugin.write_text(SHOUT_PLUGIN, encoding="utf-8")

    script = [
        {"content": "Activating what I need.",
         "tool_calls": [call("toolbelt_add", {"names": ["shout", "fs_read", "run_shell"]}, id="add")]},
        {"content": "Planning.", "tool_calls": [todo("in_progress")]},
        HttpError(500, "endpoint fell over"), HttpError(500, "still over"),   # one retry, then death
        {"content": "Back. Using the plugin tool.", "tool_calls": [call("shout", {"s": "hi"}, id="s1")]},
        {"content": "Reading the config in the workdir.",
         "tool_calls": [call("fs_read", {"path": "config.ini"}, id="r1")]},
        {"content": "Trying something the policy forbids.",
         "tool_calls": [call("run_shell", {"command": "rm -rf /"}, id="sh1")]},
        {"content": "Closing.", "tool_calls": [todo("completed")]},
        {"content": "Finishing.", "tool_calls": [FINISH]},
    ]

    with MockOpenAIServer(script, model="mock-model", api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "find the port", "--model", "mock-model",
                   "--endpoint", server.base_url, "--workdir", str(work), "--run-id", "defaults",
                   "--api-key", SECRET, "--tools", str(plugin), "--extra-body", '{"temperature": 0}',
                   "--deny-shell-pattern", r"rm\s+-rf"])
        assert rc == 2
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "resume", "defaults",
             "--api-key", SECRET],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
        bodies = [r["body"] for r in server.requests]

    assert proc.returncode == 0, proc.stderr
    assert "all echoed" in proc.stdout
    assert f"openai at {server.base_url}  workdir {work}  tools {plugin}" in proc.stderr

    recs = read_trajectory(runs / "defaults")
    steps = [r for r in recs if r["type"] == "step"]
    assert [(s["tool"], s["kind"]) for s in steps] == [
        ("toolbelt_add", "ok"), ("todo_write", "ok"),
        ("shout", "ok"),            # --tools was loaded again
        ("fs_read", "ok"),          # rooted at the recorded workdir, not the cwd
        ("run_shell", "denied"),    # the recorded policy is still in force
        ("todo_write", "ok"), ("final_answer", "final_accepted")]
    assert steps[2]["result_preview"] == "HI"
    assert "PORT = 8080" in steps[3]["result_preview"]
    assert all(b.get("temperature") == 0 for b in bodies)          # recorded --extra-body
    assert bodies[-1]["model"] == "mock-model"

    invocation = recs[0]["invocation"]
    assert invocation == {"endpoint": server.base_url, "provider": "openai", "max_tokens": 4096,
                          "timeout": 120.0, "extra_body": {"temperature": 0},
                          "tools": [str(plugin)], "skills": [], "workdir": str(work)}
    seam = next(r for r in recs if r["type"] == "resume")
    assert seam["invocation"] == invocation                        # carried forward for the next one
    assert seam["policy"] == {"deny_tools": [], "deny_shell_patterns": [r"rm\s+-rf"],
                              "shell_tools": ["run_shell"]}

    # the api key is never recorded state
    blob = (runs / "defaults" / "trajectory.jsonl").read_text() + \
           (runs / "defaults" / "state.json").read_text()
    assert SECRET not in blob and "api_key" not in blob


def test_explicit_flags_override_the_recorded_invocation(tmp_path, capsys):
    runs, work, other = tmp_path / "runs", tmp_path / "project", tmp_path / "elsewhere"
    work.mkdir(), other.mkdir()
    (other / "other.ini").write_text("[other]\n", encoding="utf-8")

    with MockOpenAIServer([HttpError(500, "down")] * 2, model="recorded-model") as dead:
        rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "recorded-model",
                   "--endpoint", dead.base_url, "--workdir", str(work), "--run-id", "override",
                   "--deny-tool", "fs_read"])
    assert rc == 2                                     # transport_error, resumable

    rest = [
        {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["fs_read", "fs_glob"]})]},
        {"content": "Planning.", "tool_calls": [todo("in_progress")]},
        {"content": "fs_read was denied before; it is allowed now.",
         "tool_calls": [call("fs_read", {"path": "other.ini"})]},
        {"content": "And the new denial bites.", "tool_calls": [call("fs_glob", {"pattern": "*"})]},
        {"content": "Closing.", "tool_calls": [todo("completed")]},
        {"content": "Finishing.", "tool_calls": [FINISH]},
    ]
    with MockOpenAIServer(rest, model="live-model") as live:
        rc = main(["--runs-dir", str(runs), "resume", "override", "--endpoint", live.base_url,
                   "--workdir", str(other), "--deny-tool", "fs_glob", "--model", "live-model"])
        bodies = [r["body"] for r in live.requests]
    assert rc == 0

    steps = [r for r in read_trajectory(runs / "override") if r["type"] == "step"]
    assert [(s["tool"], s["kind"]) for s in steps[2:4]] == [("fs_read", "ok"), ("fs_glob", "denied")]
    assert "[other]" in steps[2]["result_preview"]     # the workdir override took effect
    assert bodies[0]["model"] == "live-model"

    seam = next(r for r in read_trajectory(runs / "override") if r["type"] == "resume")
    assert seam["invocation"]["endpoint"] == live.base_url
    assert seam["invocation"]["workdir"] == str(other)
    assert seam["policy"]["deny_tools"] == ["fs_glob"]


def test_resume_without_a_recorded_endpoint_says_so(tmp_path, capsys):
    """A run driven from Python records no invocation; the CLI still needs one."""
    first = capped_run(tmp_path, run_id="bare")
    rc = main(["--runs-dir", str(tmp_path / "runs"), "resume", "bare", "--step-cap", "9"])
    assert rc == 64 and "no --endpoint given and the run recorded none" in capsys.readouterr().err
    assert [r["type"] for r in read_trajectory(first.run_dir)].count("resume") == 0
