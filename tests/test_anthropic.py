"""The native Messages API transport: translation both ways, then over HTTP."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.anthropic import (DEFAULT_MAX_TOKENS, AnthropicMessagesTransport, normalize_messages,
                               to_messages_request)
from harness.cli import main
from harness.mockserver import HttpError, MockAnthropicServer
from harness.registry import ToolRegistry
from harness.runtime import META_NAMES, AgentRuntime, RuntimeConfig
from harness.trajectory import read_trajectory
from harness.transport import TransportError, call

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-mock-key"
TASK = "Find the port in the config and report it"

TOOLS = [{"type": "function", "function": {
    "name": "fs_read", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def test_request_translation():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "the task"},
        {"role": "assistant", "content": "Reading two files.", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "fs_read", "arguments": '{"path": "a"}'}},
            {"id": "t2", "type": "function", "function": {"name": "fs_read", "arguments": '{"path": "b"}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "contents of a", "_artifact": "artifacts/1.txt"},
        {"role": "tool", "tool_call_id": "t2", "content": "", "_protected": True},
        {"role": "assistant", "content": ""},                  # an empty turn says nothing
        {"role": "user", "content": "use a tool"},
    ]
    body = to_messages_request(messages, TOOLS, "a-model", max_tokens=77, extra={"temperature": 0})

    assert body["model"] == "a-model" and body["max_tokens"] == 77 and body["temperature"] == 0
    assert body["system"] == "system prompt"                   # hoisted out of messages
    assert body["tools"] == [{"name": "fs_read", "description": "Read a file.",
                              "input_schema": TOOLS[0]["function"]["parameters"]}]
    assert body["tool_choice"] == {"type": "auto"}
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user", "user"]
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "Reading two files."},
        {"type": "tool_use", "id": "t1", "name": "fs_read", "input": {"path": "a"}},
        {"type": "tool_use", "id": "t2", "name": "fs_read", "input": {"path": "b"}}]
    # both results of one turn in one user message, so parallel calls stay parallel
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "contents of a"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "(empty result)"}]
    assert not any(k.startswith("_") for m in body["messages"] for k in m)
    assert json.dumps(body)                                    # serialisable as sent


def test_request_translation_keeps_unparseable_arguments():
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{oops"}}]}]
    block = to_messages_request(messages, [], "m")["messages"][0]["content"][0]
    assert block == {"type": "tool_use", "id": "t1", "name": "f",
                     "input": {"__unparsed_arguments": "{oops"}}


def test_response_translation_drops_private_reasoning():
    payload = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
        "content": [
            {"type": "thinking", "thinking": "PRIVATE-CHANNEL", "signature": "sig"},
            {"type": "redacted_thinking", "data": "PRIVATE-CHANNEL"},
            {"type": "text", "text": "Reading the file."},
            {"type": "tool_use", "id": "toolu_1", "name": "fs_read", "input": {"path": "a"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "cache_read_input_tokens": 20,
                  "cache_creation_input_tokens": 5, "output_tokens": 7},
    }
    out = normalize_messages(payload)
    assert out["content"] == "Reading the file."
    assert out["tool_calls"] == [{"id": "toolu_1", "name": "fs_read", "arguments": {"path": "a"}}]
    assert out["usage"] == {"prompt_tokens": 125, "completion_tokens": 7}   # cached input counts
    assert "PRIVATE" not in json.dumps(out)

    assert normalize_messages({"content": [], "usage": None}) == {
        "content": "", "tool_calls": [], "usage": None}
    with pytest.raises(TransportError, match="malformed response"):
        normalize_messages({"error": {"message": "nope"}})


def test_extended_thinking_is_refused_rather_than_half_supported():
    with pytest.raises(ValueError, match="extended thinking"):
        AnthropicMessagesTransport(extra={"thinking": {"type": "adaptive"}})
    assert AnthropicMessagesTransport("https://example.invalid/v1").endpoint == \
        "https://example.invalid/v1/messages"
    assert AnthropicMessagesTransport("https://example.invalid/v1/messages").endpoint == \
        "https://example.invalid/v1/messages"


def script():
    return [
        {"content": "Discovering what tools exist.",
         "tool_calls": [call("toolbelt_list", {"filter": "fs_"})],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10},
         "thinking": "PRIVATE-CHANNEL-MUST-NOT-BE-READ"},
        {"content": "Activating the reader.",
         "tool_calls": [call("toolbelt_add", {"names": ["fs_read"]})],
         "usage": {"prompt_tokens": 200, "completion_tokens": 12}},
        {"content": "Planning, then reading, in one turn.",
         "tool_calls": [call("todo_write", {"todos": [
             {"id": "find", "content": "find the port", "status": "in_progress"}]}),
             call("fs_read", {"path": "config.ini"})],
         "usage": {"prompt_tokens": 300, "completion_tokens": 14}},
        {"content": "Closing the todo.",
         "tool_calls": [call("todo_write", {"todos": [
             {"id": "find", "content": "find the port", "status": "completed"}]})],
         "usage": {"prompt_tokens": 400, "completion_tokens": 16}},
        {"content": "Finishing.",
         "tool_calls": [call("final_answer", {"status": "completed", "content": "The port is 8080."})],
         "usage": {"prompt_tokens": 500, "completion_tokens": 18}},
    ]


def test_cli_run_over_http_with_the_anthropic_provider(tmp_path):
    work = tmp_path / "project"
    work.mkdir()
    (work / "config.ini").write_text("[server]\nPORT = 8080\n", encoding="utf-8")
    runs = tmp_path / "runs"

    with MockAnthropicServer(script(), model="a-model", api_key=SECRET) as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run",
             "--task", TASK, "--model", "a-model", "--provider", "anthropic",
             "--endpoint", server.base_url, "--workdir", str(work), "--run-id", "anthropic-e2e",
             "--api-key", SECRET, "--max-tokens", "1024",
             "--extra-body", '{"temperature": 0}'],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
        requests = list(server.requests)

    assert proc.returncode == 0, proc.stderr
    assert "The port is 8080." in proc.stdout
    assert "status: completed  steps: 6" in proc.stderr

    run_dir = runs / "anthropic-e2e"
    recs = read_trajectory(run_dir)
    steps = [r for r in recs if r["type"] == "step"]
    assert [(s["step"], s["tool"], s["kind"]) for s in steps] == [
        (1, "toolbelt_list", "ok"), (2, "toolbelt_add", "ok"), (3, "todo_write", "ok"),
        (4, "fs_read", "ok"), (5, "todo_write", "ok"), (6, "final_answer", "final_accepted")]
    assert [s["tokens_in"] for s in steps] == [100, 200, 300, None, 400, 500]
    assert json.loads((run_dir / steps[3]["artifact"]).read_text())["content"].startswith("[server]")
    assert recs[-1]["final"] == {"status": "completed", "content": "The port is 8080."}

    # guarantee 3: the thinking block never became reasoning and never landed on disk
    blob = (run_dir / "trajectory.jsonl").read_text() + (run_dir / "state.json").read_text()
    assert "PRIVATE-CHANNEL" not in blob
    assert steps[0]["reasoning"] == "Discovering what tools exist."

    # the wire format, request by request
    assert [r["path"] for r in requests] == ["/v1/messages"] * 5
    heads = [r["headers"] for r in requests]
    assert all(h["x-api-key"] == SECRET for h in heads)
    assert all(h["anthropic-version"] == "2023-06-01" for h in heads)
    assert all("Authorization" not in h for h in heads)
    bodies = [r["body"] for r in requests]
    assert all(b["max_tokens"] == 1024 and b["temperature"] == 0 for b in bodies)
    assert all(b["model"] == "a-model" for b in bodies)
    assert all(b["system"].startswith("You are an agent") for b in bodies)
    assert all("system" not in [m["role"] for m in b["messages"]] for b in bodies)

    # guarantee 1 on the Messages wire: only meta tools first, schemas as input_schema
    assert {t["name"] for t in bodies[0]["tools"]} == META_NAMES
    assert {t["name"] for t in bodies[3]["tools"]} == META_NAMES | {"fs_read"}
    assert all("input_schema" in t and "parameters" not in t for t in bodies[0]["tools"])
    assert bodies[0]["tools"][0]["input_schema"]["type"] == "object"
    assert bodies[0]["tool_choice"] == {"type": "auto"}

    # the two calls of turn 3 came back as two tool_result blocks in one user message
    last = bodies[-1]["messages"]
    turn = next(m for m in last if m["role"] == "assistant"
                and sum(1 for b in m["content"] if b["type"] == "tool_use") == 2)
    ids = [b["id"] for b in turn["content"] if b["type"] == "tool_use"]
    results = next(m for m in last if m["role"] == "user"
                   and len(m["content"]) == 2 and m["content"][0]["type"] == "tool_result")
    assert [b["tool_use_id"] for b in results["content"]] == ids
    assert all(b["content"] for b in results["content"])
    assert last[0]["role"] == "user" and last[0]["content"][0]["text"] == TASK


def test_anthropic_http_error_and_bad_key_are_transport_errors(tmp_path, capsys):
    runs = tmp_path / "runs"
    with MockAnthropicServer([HttpError(529, "overloaded")] * 2) as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "m",
                   "--provider", "anthropic", "--endpoint", server.base_url,
                   "--workdir", str(tmp_path), "--run-id", "over"])
        assert rc == 2 and server.served == 2          # one retry, then give up
    footer = read_trajectory(runs / "over")[-1]
    assert footer["status"] == "transport_error" and "HTTP 529" in footer["detail"]
    assert "overloaded" in footer["detail"]

    with MockAnthropicServer(script(), api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "m",
                   "--provider", "anthropic", "--endpoint", server.base_url,
                   "--workdir", str(tmp_path), "--run-id", "auth", "--api-key", "wrong"])
    assert rc == 2 and "HTTP 401" in read_trajectory(runs / "auth")[-1]["detail"]

    rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "m",
               "--provider", "anthropic", "--endpoint", "http://127.0.0.1:1/v1",
               "--workdir", str(tmp_path), "--extra-body", '{"thinking": {"type": "adaptive"}}'])
    assert rc == 64 and "extended thinking" in capsys.readouterr().err


def test_an_overloaded_messages_api_is_waited_out_as_it_asked(tmp_path):
    """529 is the API saying it is overloaded; Retry-After says for how long."""
    waited: list[float] = []
    done = [{"content": "Back. Closing.", "tool_calls": [call("todo_write", {"todos": [
                {"id": "1", "content": "ask", "status": "completed"}]})]},
            {"content": "Finishing.", "tool_calls": [call("final_answer", {
                "status": "completed", "content": "ok"})]}]
    script = [HttpError(529, "overloaded", retry_after=3),
              HttpError(429, "slow down", retry_after="in a bit"),   # not seconds: unusable
              *done]
    with MockAnthropicServer(script, model="a-model", api_key=SECRET) as server:
        transport = AnthropicMessagesTransport(server.base_url, api_key=SECRET, sleep=waited.append)
        res = AgentRuntime(ToolRegistry(), transport, tmp_path / "runs", "a-model",
                           RuntimeConfig(transport_retries=2), run_id="busy").run("ask the busy API")
        served = server.served

    assert res.status == "completed" and served == 4
    assert waited == [3.0, 4.0]          # the header, then the schedule it fell back to
    assert read_trajectory(res.run_dir)[-1]["status"] == "completed"


def test_resume_over_the_anthropic_provider(tmp_path):
    runs = tmp_path / "runs"
    with MockAnthropicServer(script()[:2], model="a-model") as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", TASK, "--model", "a-model",
                   "--provider", "anthropic", "--endpoint", server.base_url,
                   "--workdir", str(tmp_path), "--run-id", "half-a", "--step-cap", "2"])
    assert rc == 3

    rest = [{"content": "Planning and finishing.", "tool_calls": [
        call("todo_write", {"todos": [{"id": "1", "content": "x", "status": "completed"}]}),
        call("final_answer", {"status": "completed", "content": "done"})]}]
    with MockAnthropicServer(rest, model="a-model") as server:
        rc = main(["--runs-dir", str(runs), "resume", "half-a", "--provider", "anthropic",
                   "--endpoint", server.base_url, "--workdir", str(tmp_path), "--step-cap", "8"])
        body = server.requests[0]["body"]
    assert rc == 0
    recs = read_trajectory(runs / "half-a")
    assert [r["type"] for r in recs].count("resume") == 1 and recs[-1]["status"] == "completed"
    # the resumed transcript translated cleanly: every tool_use answered by a tool_result
    used = [b["id"] for m in body["messages"] if m["role"] == "assistant"
            for b in m["content"] if b["type"] == "tool_use"]
    got = [b["tool_use_id"] for m in body["messages"] if m["role"] == "user"
           for b in m["content"] if isinstance(b, dict) and b["type"] == "tool_result"]
    assert used == got and len(used) == 2
    assert body["messages"][-1]["role"] == "user"
    assert "has been resumed" in body["messages"][-1]["content"][0]["text"]
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS
