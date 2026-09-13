"""End to end over real HTTP: the packaged CLI against the mock server.

This is the only test that exercises ChatCompletionsTransport, the wire
format, and `python -m harness` as a process. Everything else stops at
FakeTransport.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from harness.cli import main
from harness.mockserver import HttpError, MockOpenAIServer
from harness.runtime import META_NAMES
from harness.trajectory import read_trajectory
from harness.transport import call

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = "Find the port in the config and report it"
SECRET = "sekret"


def script():
    """list -> add -> plan -> fs_search -> early final (rejected) -> close -> final."""
    return [
        {"content": "Discovering what tools exist.",
         "tool_calls": [call("toolbelt_list", {"filter": "fs_"})],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10},
         "reasoning_content": "PRIVATE-CHANNEL-MUST-NOT-BE-READ"},
        {"content": "fs_search will do; activating it.",
         "tool_calls": [call("toolbelt_add", {"names": ["fs_search"]})],
         "usage": {"prompt_tokens": 200, "completion_tokens": 12}},
        {"content": "Planning before I search.",
         "tool_calls": [call("todo_write", {"todos": [
             {"id": "find", "content": "find the port", "status": "in_progress"},
             {"id": "report", "content": "report it", "status": "pending"}]})],
         "usage": {"prompt_tokens": 300, "completion_tokens": 14}},
        {"content": "Searching the config for a port.",
         "tool_calls": [call("fs_search", {"pattern": "(?i)port", "glob": "*.ini"})],
         "usage": {"prompt_tokens": 400, "completion_tokens": 16}},
        {"content": "Answering right away.",
         "tool_calls": [call("final_answer", {"status": "completed", "content": "8080"})],
         "usage": {"prompt_tokens": 500, "completion_tokens": 8}},
        {"content": "Right, the gate. Closing the todos.",
         "tool_calls": [call("todo_write", {"todos": [
             {"id": "find", "content": "find the port", "status": "completed"},
             {"id": "report", "content": "report it", "status": "completed"}]})],
         "usage": {"prompt_tokens": 600, "completion_tokens": 18}},
        {"content": "Now finishing.",
         "tool_calls": [call("final_answer", {"status": "completed", "content": "The port is 8080 (config.ini:2)."})],
         "usage": {"prompt_tokens": 700, "completion_tokens": 20}},
    ]


def workdir(tmp_path):
    work = tmp_path / "project"
    (work / "conf").mkdir(parents=True)
    (work / "conf" / "config.ini").write_text("[server]\nPORT = 8080\nhost = 0.0.0.0\n", encoding="utf-8")
    (work / "README.md").write_text("no port here\n", encoding="utf-8")
    return work


def test_cli_run_over_http_end_to_end(tmp_path):
    work = workdir(tmp_path)
    runs = tmp_path / "runs"
    with MockOpenAIServer(script(), model="mock-model", api_key=SECRET) as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run",
             "--task", TASK, "--model", "mock-model", "--endpoint", server.base_url,
             "--workdir", str(work), "--run-id", "http-e2e", "--api-key", SECRET,
             "--preview-chars", "200", "--extra-body", '{"temperature": 0, "top_p": 0.9}'],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        )

    assert proc.returncode == 0, proc.stderr
    assert "The port is 8080 (config.ini:2)." in proc.stdout
    assert "run http-e2e" in proc.stderr and "status: completed  steps: 7" in proc.stderr

    run_dir = runs / "http-e2e"
    assert (run_dir / "trajectory.jsonl").exists() and (run_dir / "state.json").exists()
    recs = read_trajectory(run_dir)
    header, footer = recs[0], recs[-1]
    steps = [r for r in recs if r["type"] == "step"]
    assert header["model"] == "mock-model" and header["task"] == TASK
    assert footer["status"] == "completed" and footer["steps"] == 7
    assert footer["final"] == {"status": "completed", "content": "The port is 8080 (config.ini:2)."}

    assert [(s["step"], s["tool"], s["kind"]) for s in steps] == [
        (1, "toolbelt_list", "ok"),
        (2, "toolbelt_add", "ok"),
        (3, "todo_write", "ok"),
        (4, "fs_search", "ok"),
        (5, "final_answer", "final_rejected"),
        (6, "todo_write", "ok"),
        (7, "final_answer", "final_accepted"),
    ]
    assert [s["tokens_in"] for s in steps] == [100, 200, 300, 400, 500, 600, 700]
    assert [s["tokens_out"] for s in steps] == [10, 12, 14, 16, 8, 18, 20]
    assert steps[0]["reasoning"] == "Discovering what tools exist."
    assert json.loads(steps[4]["result_preview"])["open_ids"] == ["find", "report"]

    # artifacts hold the full result, not the 200-char preview
    hits = json.loads((run_dir / steps[3]["artifact"]).read_text(encoding="utf-8"))
    assert hits["matches"] == [{"path": "conf/config.ini", "line": 2, "text": "PORT = 8080"}]
    assert hits["files_scanned"] == 1
    for s in steps:
        assert (run_dir / s["artifact"]).exists()

    # guarantee 3: the private reasoning channel was never read
    blob = (run_dir / "trajectory.jsonl").read_text() + (run_dir / "state.json").read_text()
    assert "PRIVATE-CHANNEL" not in blob

    # guarantee 1, over the wire: only meta tools in the first request
    bodies = [r["body"] for r in server.requests]
    assert len(bodies) == 7
    assert {t["function"]["name"] for t in bodies[0]["tools"]} == META_NAMES
    assert {t["function"]["name"] for t in bodies[2]["tools"]} == META_NAMES | {"fs_search"}
    assert bodies[0]["tool_choice"] == "auto" and bodies[0]["model"] == "mock-model"
    # --extra-body reached the provider on every request, without touching what the harness owns
    assert all(b["temperature"] == 0 and b["top_p"] == 0.9 for b in bodies)
    assert all(r["headers"]["Authorization"] == f"Bearer {SECRET}" for r in server.requests)
    # the assistant turn and its tool result came back over the wire intact
    roles = [m["role"] for m in bodies[-1]["messages"]]
    assert roles[:3] == ["system", "user", "assistant"] and roles.count("tool") == 6
    assert not any(k.startswith("_") for m in bodies[-1]["messages"] for k in m)


def test_http_error_becomes_transport_error_exit_2(tmp_path, capsys):
    runs = tmp_path / "runs"
    with MockOpenAIServer([HttpError(500, "upstream exploded")] * 2) as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "m",
                   "--endpoint", server.base_url, "--workdir", str(tmp_path), "--run-id", "boom"])
    assert rc == 2
    assert server.served == 2  # one retry, then give up
    footer = read_trajectory(runs / "boom")[-1]
    assert footer["status"] == "transport_error" and "HTTP 500" in footer["detail"]
    assert "upstream exploded" in footer["detail"]


def test_bad_api_key_is_a_transport_error(tmp_path):
    runs = tmp_path / "runs"
    with MockOpenAIServer(script(), api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "m",
                   "--endpoint", server.base_url, "--workdir", str(tmp_path),
                   "--run-id", "auth", "--api-key", "wrong"])
    assert rc == 2
    assert "HTTP 401" in read_trajectory(runs / "auth")[-1]["detail"]


def test_extra_body_must_be_a_json_object_without_reserved_keys(tmp_path, capsys):
    def run(extra):
        return main(["--runs-dir", str(tmp_path / "runs"), "run", "--task", "t", "--model", "m",
                     "--endpoint", "http://127.0.0.1:1/v1", "--workdir", str(tmp_path),
                     "--extra-body", extra])

    assert run("{not json") == 64 and "not valid JSON" in capsys.readouterr().err
    assert run("[1, 2]") == 64 and "must be a JSON object" in capsys.readouterr().err
    assert run('{"messages": []}') == 64 and "may not set messages" in capsys.readouterr().err
