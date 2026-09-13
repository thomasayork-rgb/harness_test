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

import json
import urllib.error
import urllib.request
from typing import Any, Protocol


class TransportError(Exception):
    pass


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
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.extra = extra or {}

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
            raise TransportError(f"HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            raise TransportError(str(e)) from e
        return normalize_openai(payload)


class FakeTransport:
    """Scripted responses for tests. Each entry is either a normalized response
    dict or an Exception instance to raise. Records every request it receives."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    def complete(self, messages: list[dict], tools: list[dict], model: str) -> dict:
        self.requests.append({"messages": public_messages(messages), "tools": tools, "model": model})
        if not self.script:
            raise TransportError("fake transport script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"content": item.get("content", ""), "tool_calls": item.get("tool_calls", []), "usage": item.get("usage")}


def call(name: str, arguments: Any, id: str | None = None) -> dict:
    """Helper for writing FakeTransport scripts."""
    return {"id": id or f"call_{name}", "name": name, "arguments": arguments}
