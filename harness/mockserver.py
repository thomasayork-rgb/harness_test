"""A scripted OpenAI-compatible server, stdlib only.

``FakeTransport`` stops at the runtime boundary; this one exercises everything
below it — HTTP, headers, wire format, JSON-encoded tool arguments, error
statuses — so a run can be driven end to end against
``http://127.0.0.1:<port>/v1`` with the real ``ChatCompletionsTransport``::

    with MockOpenAIServer([{"content": "Listing.", "tool_calls": [call("toolbelt_list", {})]}]) as server:
        subprocess.run([sys.executable, "-m", "harness", "run",
                        "--endpoint", server.base_url, ...])
    server.requests   # every request body it received, in order

Script entries, tried in this order:

  HttpError(status, message)   respond with that HTTP status and an error body
  callable                     called with the parsed request body, returns a
                               full wire payload
  dict with "choices"          served as-is (raw wire payload)
  dict                         {"content", "tool_calls", "usage",
                               "reasoning_content"} in FakeTransport's shape,
                               converted to the wire format

An exhausted script answers 503, which the transport reports as a
``TransportError`` rather than hanging.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

CHAT_PATH = "/chat/completions"


class HttpError:
    """Script entry: make the server answer with an HTTP error."""

    def __init__(self, status: int, message: str = "mock error") -> None:
        self.status = status
        self.message = message


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


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, mock: "MockOpenAIServer", **kwargs) -> None:
        self.mock = mock
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # keep test output clean
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"_unparseable": raw}

        if not self.path.endswith(CHAT_PATH):
            self._send(404, {"error": {"message": f"no such path: {self.path}"}})
            return

        auth = self.headers.get("Authorization")
        self.mock.record(self.path, body, dict(self.headers))
        if self.mock.api_key and auth != f"Bearer {self.mock.api_key}":
            self._send(401, {"error": {"message": "invalid api key"}})
            return

        entry, index = self.mock.next_entry()
        if entry is None:
            self._send(503, {"error": {"message": "mock script exhausted"}})
            return
        if isinstance(entry, HttpError):
            self._send(entry.status, {"error": {"message": entry.message}})
            return
        if callable(entry):
            self._send(200, entry(body))
            return
        if isinstance(entry, dict) and "choices" in entry:
            self._send(200, entry)
            return
        self._send(200, chat_completion_payload(entry, body.get("model") or self.mock.model, index))


class MockOpenAIServer:
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

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> str:
        def factory(*args, **kwargs):
            return _Handler(*args, mock=self, **kwargs)

        self._httpd = ThreadingHTTPServer((self.host, 0), factory)
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

    def __enter__(self) -> "MockOpenAIServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("server not started")
        return self._httpd.server_address[1]

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
