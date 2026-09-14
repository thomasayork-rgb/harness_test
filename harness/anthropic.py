"""Anthropic Messages API transport - stdlib only, no SDK.

``POST /v1/messages`` speaks a different shape from the OpenAI-compatible
endpoint the rest of the harness is written against, so this module is a pure
translation layer around the same ``complete()`` contract:

  out  system messages          -> the top-level ``system`` parameter
       user/assistant text      -> content blocks
       assistant tool_calls     -> ``tool_use`` blocks (arguments -> ``input``)
       tool messages            -> ``tool_result`` blocks carrying
                                   ``tool_use_id``, all the results of one
                                   assistant turn gathered into one user
                                   message so parallel calls stay parallel
       tool schemas             -> ``{name, description, input_schema}``

  in   ``text`` blocks          -> ``content``
       ``tool_use`` blocks      -> ``tool_calls`` [{id, name, arguments}]
       ``usage``                -> {prompt_tokens, completion_tokens}

Auth is ``x-api-key`` plus the required ``anthropic-version`` header, and
``max_tokens`` is required on every request, so it has a default here.

Extended thinking is **not supported**. ``thinking`` and ``redacted_thinking``
blocks are private reasoning: guarantee 3 says the harness never reads them,
so they are dropped rather than turned into reasoning text or written to the
trajectory. Because the API requires those blocks to be echoed back on later
turns of the same tool-use conversation, dropping them and continuing would be
worse than not enabling thinking at all - so enabling it through ``extra`` is
refused outright rather than half-honoured.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .transport import TransportError, public_messages, retry_after_seconds

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
# The API rejects empty text and empty tool_result blocks; a tool may legally
# return nothing, so say so rather than send a block that fails validation.
EMPTY_RESULT = "(empty result)"
# Arguments the model produced that were not valid JSON. The runtime hands them
# on as a string; ``input`` must be an object, so keep the text rather than
# invent an empty call.
UNPARSED_KEY = "__unparsed_arguments"
PRIVATE_BLOCKS = ("thinking", "redacted_thinking")


def _text_blocks(content: Any) -> list[dict]:
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    text = (content or "").strip()
    return [{"type": "text", "text": text}] if text else []


def _tool_use(call: dict) -> dict:
    fn = call.get("function") or {}
    raw = fn.get("arguments")
    if isinstance(raw, str):
        try:
            args: Any = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            args = {UNPARSED_KEY: raw}
    else:
        args = raw if isinstance(raw, dict) else {UNPARSED_KEY: str(raw)}
    return {"type": "tool_use", "id": call.get("id") or "", "name": fn.get("name", ""), "input": args}


def to_messages_request(messages: list[dict], tools: list[dict], model: str,
                        max_tokens: int = DEFAULT_MAX_TOKENS, extra: dict | None = None) -> dict:
    """Translate the runtime's OpenAI-shaped request into a Messages API body."""
    system: list[str] = []
    out: list[dict] = []
    for m in public_messages(messages):
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system.append(str(m["content"]))
            continue
        if role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id") or "",
                     "content": m.get("content") or EMPTY_RESULT}
            # every result of one assistant turn belongs in a single user message
            if out and out[-1]["role"] == "user" and out[-1].get("_tool_results"):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block], "_tool_results": True})
            continue
        if role == "assistant":
            blocks = _text_blocks(m.get("content"))
            blocks += [_tool_use(c) for c in m.get("tool_calls") or []]
            if blocks:  # an assistant turn with nothing in it says nothing
                out.append({"role": "assistant", "content": blocks})
            continue
        blocks = _text_blocks(m.get("content"))
        if blocks:
            out.append({"role": "user", "content": blocks})

    body: dict = {"model": model, "max_tokens": max_tokens,
                  "messages": [{k: v for k, v in m.items() if not k.startswith("_")} for m in out]}
    if system:
        body["system"] = "\n\n".join(system)
    if tools:
        body["tools"] = [{"name": t["function"]["name"],
                          "description": t["function"].get("description", ""),
                          "input_schema": t["function"]["parameters"]} for t in tools]
        body["tool_choice"] = {"type": "auto"}
    body.update(extra or {})
    return body


def normalize_messages(payload: dict) -> dict:
    """Translate a Messages API response into the runtime's response shape."""
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise TransportError(f"malformed response: no content blocks: {str(payload)[:300]}")
    text: list[str] = []
    calls: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text":
            text.append(b.get("text") or "")
        elif kind == "tool_use":
            args = b.get("input")
            calls.append({"id": b.get("id") or f"call_{len(calls)}", "name": b.get("name", ""),
                          "arguments": args if isinstance(args, (dict, str)) else {}})
        # thinking / redacted_thinking and anything else: never read, never logged
    usage = payload.get("usage") or None
    if usage is not None:
        usage = {
            # cached input is still input the model read; counting only the
            # uncached part would make a cached run look free
            "prompt_tokens": sum(usage.get(k) or 0 for k in
                                 ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")),
            "completion_tokens": usage.get("output_tokens"),
        }
    return {"content": "".join(text), "tool_calls": calls, "usage": usage}


class AnthropicMessagesTransport:
    """The native Messages API over urllib. ``--provider anthropic``."""

    def __init__(
        self,
        endpoint: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        extra: dict | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        version: str = ANTHROPIC_VERSION,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/messages"):
            self.endpoint += "/messages"
        self.api_key = api_key
        self.timeout = timeout
        self.extra = extra or {}
        self.max_tokens = max_tokens
        self.version = version
        self.sleep = sleep or time.sleep     # how the loop waits between retries
        if any(k in self.extra for k in PRIVATE_BLOCKS):
            raise ValueError(
                "this transport does not support extended thinking: the harness never reads "
                "private reasoning, and the API needs those blocks echoed back")

    def complete(self, messages: list[dict], tools: list[dict], model: str) -> dict:
        body = to_messages_request(messages, tools, model, self.max_tokens, self.extra)
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("anthropic-version", self.version)
        if self.api_key:
            req.add_header("x-api-key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise TransportError(f"HTTP {e.code}: {detail}",
                                 retry_after=retry_after_seconds(e.headers, e.code)) from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            raise TransportError(str(e)) from e
        return normalize_messages(payload)
