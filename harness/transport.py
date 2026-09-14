"""Model transport.

One adapter, one function signature. ``complete()`` takes messages and tool
schemas and returns a normalized response::

    {
      "content": str,                       # assistant text, "" if none
      "tool_calls": [                       # [] if none
        {"id": str, "name": str, "arguments": dict | str}
      ],
      "usage": {"prompt_tokens": int, "completion_tokens": int} | None,
    }

``arguments`` is a dict when the model produced valid JSON, otherwise the raw
string; the runtime turns that into a validation error the model can see.

Internal message keys starting with ``_`` are stripped before sending.
"""
from __future__ import annotations

import itertools
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol

# A transport error is retried by the loop (RuntimeConfig.transport_retries),
# which waits between attempts: 1s before the first retry, 4s before any later
# one. The waiting is done through the transport's ``sleep``, so a test can
# hand in one that records the schedule instead of living through it.
RETRY_STATUSES = (429, 529)     # rate limited and overloaded: the two worth waiting on
RETRY_AFTER_MAX = 60.0          # a provider that asks for longer gets this instead


def retry_after_seconds(headers: Any, status: int) -> float | None:
    """The wait a 429 or 529 asked for, in seconds and capped, or None.

    Only the seconds form of ``Retry-After`` is read; the HTTP-date form, and
    anything that is not a number, fall back to the schedule.
    """
    if status not in RETRY_STATUSES or headers is None:
        return None
    try:
        value = float(str(headers.get("Retry-After")).strip())
    except (AttributeError, TypeError, ValueError):
        return None
    if value != value or value < 0:      # NaN, or a wait that is not one
        return None
    return min(value, RETRY_AFTER_MAX)


class TransportError(Exception):
    """A request that produced no usable response.

    ``retry_after`` is what the provider asked to be waited, in seconds, when
    it said so in a header; the loop waits that instead of its own schedule.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class Transport(Protocol):
    def complete(self, messages: list[dict], tools: list[dict], model: str) -> dict: ...


def public_messages(messages: list[dict]) -> list[dict]:
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


def normalize_openai(payload: dict) -> dict:
    try:
        msg = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise TransportError(f"malformed response: {e}: {str(payload)[:300]}")
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments", "{}")
        try:
            args: Any = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = raw
        calls.append({"id": tc.get("id") or f"call_{len(calls)}", "name": fn.get("name", ""), "arguments": args})
    usage = payload.get("usage") or None
    if usage is not None:
        usage = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    return {"content": msg.get("content") or "", "tool_calls": calls, "usage": usage}


class ChatCompletionsTransport:
    """OpenAI-compatible ``/v1/chat/completions`` over urllib. Works with
    llama.cpp server, vLLM, LM Studio, Ollama's OpenAI endpoint, or the real thing."""

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        extra: dict | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.extra = extra or {}
        self.sleep = sleep or time.sleep     # how the loop waits between retries

    def complete(self, messages: list[dict], tools: list[dict], model: str) -> dict:
        body: dict = {"model": model, "messages": public_messages(messages), **self.extra}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise TransportError(f"HTTP {e.code}: {detail}",
                                 retry_after=retry_after_seconds(e.headers, e.code)) from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            raise TransportError(str(e)) from e
        return normalize_openai(payload)


class FakeTransport:
    """Scripted responses for tests. Each entry is either a normalized response
    dict or an exception instance to raise. Records every request it receives.

    Any BaseException is raised, not only an Exception: ``KeyboardInterrupt()``
    in a script is how a test puts Ctrl-C in the middle of a run.
    """

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict] = []
        self.delays: list[float] = []

    def sleep(self, seconds: float) -> None:
        """A scripted transport waits for nobody: the retry delay is recorded."""
        self.delays.append(seconds)

    def complete(self, messages: list[dict], tools: list[dict], model: str) -> dict:
        self.requests.append({"messages": public_messages(messages), "tools": tools, "model": model})
        if not self.script:
            raise TransportError("fake transport script exhausted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return {"content": item.get("content", ""), "tool_calls": item.get("tool_calls", []), "usage": item.get("usage")}


_call_ids = itertools.count(1)


def call(name: str, arguments: Any, id: str | None = None) -> dict:
    """Helper for writing FakeTransport scripts.

    A generated id carries a process-wide counter, so two calls of the same
    tool in one turn are still two distinct ids - a transcript with one id
    answered twice is malformed, and a provider handed it back would say so.
    Pass ``id`` when a test wants to name the call itself.
    """
    return {"id": id or f"call_{name}_{next(_call_ids)}", "name": name, "arguments": arguments}
