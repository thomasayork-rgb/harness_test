"""Scripted model servers, stdlib only: OpenAI-compatible and Messages API.

``FakeTransport`` stops at the runtime boundary; this one exercises everything
below it — HTTP, headers, wire format, JSON-encoded tool arguments, error
statuses — so a run can be driven end to end against
``http://127.0.0.1:<port>/v1`` with the real ``ChatCompletionsTransport``::

    with MockOpenAIServer([{"content": "Listing.", "tool_calls": [call("toolbelt_list", {})]}]) as server:
        subprocess.run([sys.executable, "-m", "harness", "run",
                        "--endpoint", server.base_url, ...])
    server.requests   # every request body it received, in order

Script entries, tried in this order:

  HttpError(status, message,   respond with that HTTP status and an error body,
            retry_after)       with a Retry-After header when one is given
  callable                     called with the parsed request body, returns a
                               full wire payload
  dict with "choices"          served as-is (raw wire payload)
  dict                         {"content", "tool_calls", "usage",
                               "reasoning_content"} in FakeTransport's shape,
                               converted to the wire format

An exhausted script answers 503, which the transport reports as a
``TransportError`` rather than hanging.

``MockAnthropicServer`` is the same machinery pointed at ``/v1/messages``:
``x-api-key`` instead of a bearer token, and scripted responses rendered as
Messages API content blocks. Give a script entry a ``thinking`` key and the
response carries a thinking block, which the harness must drop on the floor.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

CHAT_PATH = "/chat/completions"
MESSAGES_PATH = "/messages"


class Headers(dict):
    """Request headers exactly as received, looked up case-insensitively.

    urllib spells ``x-api-key`` as ``X-api-key`` on the wire; HTTP says that is
    the same header, so a test asserting on one should not depend on which.
    """

    def __init__(self, items) -> None:
        super().__init__(items)
        self._lower = {k.lower(): v for k, v in self.items()}

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return self._lower[str(key).lower()]

    def __contains__(self, key) -> bool:
        return dict.__contains__(self, key) or str(key).lower() in self._lower

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class HttpError:
    """Script entry: make the server answer with an HTTP error.

    ``retry_after`` becomes a ``Retry-After`` header, which is how a real 429
    or 529 says how long to wait; it is sent verbatim, so a test can also send
    something unparseable and watch the transport fall back to its schedule.
    """

    def __init__(self, status: int, message: str = "mock error",
                 retry_after: Any = None) -> None:
        self.status = status
        self.message = message
        self.retry_after = retry_after


def chat_completion_payload(resp: dict, model: str, index: int = 0) -> dict:
    """Turn a FakeTransport-shaped response into an OpenAI wire payload."""
    calls = []
    for i, c in enumerate(resp.get("tool_calls") or []):
        args = c.get("arguments", {})
        calls.append({
            "id": c.get("id") or f"call_{index}_{i}",
            "type": "function",
            "function": {"name": c.get("name", ""),
                         "arguments": args if isinstance(args, str) else json.dumps(args)},
        })
    message: dict[str, Any] = {"role": "assistant", "content": resp.get("content") or None}
    if calls:
        message["tool_calls"] = calls
    # A private reasoning channel, as several providers emit. The runtime must
    # ignore it; it is here so tests can prove that.
    if resp.get("reasoning_content"):
        message["reasoning_content"] = resp["reasoning_content"]
    payload: dict[str, Any] = {
        "id": f"chatcmpl-mock-{index}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if calls else "stop"}],
    }
    usage = resp.get("usage")
    if usage:
        payload["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0),
        }
    return payload


def messages_payload(resp: dict, model: str, index: int = 0) -> dict:
    """Turn a FakeTransport-shaped response into a Messages API wire payload."""
    blocks: list[dict] = []
    # A private reasoning channel, as the real API emits when thinking is on.
    # The harness must never read it; it is here so tests can prove that.
    if resp.get("thinking"):
        blocks.append({"type": "thinking", "thinking": resp["thinking"], "signature": "sig-mock"})
    if resp.get("content"):
        blocks.append({"type": "text", "text": resp["content"]})
    for i, c in enumerate(resp.get("tool_calls") or []):
        args = c.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        blocks.append({"type": "tool_use", "id": c.get("id") or f"toolu_{index}_{i}",
                       "name": c.get("name", ""), "input": args})
    usage = resp.get("usage") or {}
    return {
        "id": f"msg_mock_{index}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens") or 0,
                  "output_tokens": usage.get("completion_tokens") or 0,
                  "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0},
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, mock: "_MockServer", **kwargs) -> None:
        self.mock = mock
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # keep test output clean
        pass

    def _send(self, status: int, payload: dict, headers: dict | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"_unparseable": raw}

        if not self.path.endswith(self.mock.path_suffix):
            self._send(404, {"error": {"message": f"no such path: {self.path}"}})
            return

        headers = Headers(self.headers.items())
        self.mock.record(self.path, body, headers)
        if not self.mock.authorized(headers):
            self._send(401, {"error": {"message": "invalid api key"}})
            return

        entry, index = self.mock.next_entry()
        if entry is None:
            self._send(503, {"error": {"message": "mock script exhausted"}})
            return
        if isinstance(entry, HttpError):
            extra = {"Retry-After": str(entry.retry_after)} if entry.retry_after is not None else None
            self._send(entry.status, {"error": {"message": entry.message}}, extra)
            return
        if callable(entry):
            self._send(200, entry(body))
            return
        if isinstance(entry, dict) and self.mock.is_raw(entry):
            self._send(200, entry)
            return
        self._send(200, self.mock.wire_payload(entry, body.get("model") or self.mock.model, index))


class _MockServer:
    """Lifecycle and scripting; the subclass supplies the provider's dialect."""

    path_suffix = ""

    def __init__(self, script: list, model: str = "mock-model", api_key: str | None = None,
                 host: str = "127.0.0.1") -> None:
        self.script = list(script)
        self.model = model
        self.api_key = api_key
        self.host = host
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self._index = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> str:
        def factory(*args, **kwargs):
            return _Handler(*args, mock=self, **kwargs)

        self._httpd = ThreadingHTTPServer((self.host, 0), factory)
        self._port = self._httpd.server_address[1]
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "_MockServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def port(self) -> int:
        """The port it is on, or was on: a test that asserts on the endpoint a
        run recorded reads this after the server has been shut down."""
        if self._port is None:
            raise RuntimeError("server not started yet")
        return self._port

    @property
    def base_url(self) -> str:
        """What to pass as ``--endpoint``."""
        return f"http://{self.host}:{self.port}/v1"

    # ---- state shared with the handler threads -----------------------------

    def record(self, path: str, body: dict, headers: dict) -> None:
        with self._lock:
            self.requests.append({"path": path, "body": body, "headers": headers})

    def next_entry(self) -> tuple[Any, int]:
        with self._lock:
            if self._index >= len(self.script):
                return None, self._index
            entry = self.script[self._index]
            self._index += 1
            return entry, self._index - 1

    @property
    def served(self) -> int:
        with self._lock:
            return self._index

    # ---- the provider's dialect --------------------------------------------

    def authorized(self, headers) -> bool:
        raise NotImplementedError

    def is_raw(self, entry: dict) -> bool:
        raise NotImplementedError

    def wire_payload(self, entry: dict, model: str, index: int) -> dict:
        raise NotImplementedError


class MockOpenAIServer(_MockServer):
    """Scripted ``/v1/chat/completions``."""

    path_suffix = CHAT_PATH

    def authorized(self, headers) -> bool:
        return not self.api_key or headers.get("Authorization") == f"Bearer {self.api_key}"

    def is_raw(self, entry: dict) -> bool:
        return "choices" in entry

    def wire_payload(self, entry: dict, model: str, index: int) -> dict:
        return chat_completion_payload(entry, model, index)


class MockAnthropicServer(_MockServer):
    """Scripted ``/v1/messages``: x-api-key, anthropic-version, content blocks."""

    path_suffix = MESSAGES_PATH

    def authorized(self, headers) -> bool:
        return not self.api_key or headers.get("x-api-key") == self.api_key

    def is_raw(self, entry: dict) -> bool:
        return entry.get("type") == "message"

    def wire_payload(self, entry: dict, model: str, index: int) -> dict:
        return messages_payload(entry, model, index)
