"""Replay a recorded run.

A trajectory already contains everything the model contributed: the reasoning
text of each turn, the tool calls it made, and - via ``call_index`` - which
calls belonged to the same turn. ``ReplayTransport`` hands those back to the
runtime as if the model had produced them again, so a recorded run becomes a
regression test: the tools run for real, and any difference in what they
return shows up as a difference in the new trajectory.

    from harness.replay import ReplayTransport, compare, replay

    result = replay("runs/20240101-120000-abc123", registry)
    assert compare("runs/20240101-120000-abc123", result.run_dir) == []

Timing, ids and the artifact bodies are not compared; everything the model or
the tools decided is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import ToolRegistry
from .runtime import AgentRuntime, RunResult, RuntimeConfig
from .trajectory import read_trajectory
from .transport import TransportError, public_messages

COMPARED_FIELDS = ("step", "call_index", "reasoning", "tool", "args", "kind",
                   "result_preview", "result_bytes", "artifact", "tokens_in",
                   "tokens_out", "todo_snapshot")


def _records(source: Any) -> list[dict]:
    """Accept a records list, a run directory, or the trajectory file itself."""
    if isinstance(source, list):
        return source
    path = Path(source)
    return read_trajectory(path.parent if path.name == "trajectory.jsonl" else path)


def turns_from_trajectory(records: list[dict]) -> list[dict]:
    """Group recorded steps back into the model turns that produced them."""
    turns: list[dict] = []
    for rec in records:
        if rec.get("type") != "step":
            continue
        if not turns or rec.get("call_index", 0) == 0:
            usage = None
            if rec.get("tokens_in") is not None or rec.get("tokens_out") is not None:
                usage = {"prompt_tokens": rec.get("tokens_in"), "completion_tokens": rec.get("tokens_out")}
            turns.append({"content": rec.get("reasoning") or "", "tool_calls": [], "usage": usage})
        if rec.get("tool"):
            turns[-1]["tool_calls"].append(
                {"id": f"call_{rec['step']}", "name": rec["tool"], "arguments": rec.get("args")})
    return turns


class ReplayTransport:
    """Serve a recorded trajectory's turns in order. No model, no network."""

    def __init__(self, source: Any) -> None:
        self.records = _records(source)
        self.header = next((r for r in self.records if r.get("type") == "header"), {})
        self.footer = next((r for r in self.records if r.get("type") == "footer"), {})
        self.turns = turns_from_trajectory(self.records)
        self.requests: list[dict] = []
        self.index = 0

    @property
    def run_id(self) -> str:
        return self.header.get("run_id") or "unknown"

    @property
    def task(self) -> str:
        return self.header.get("task") or ""

    @property
    def model(self) -> str:
        return self.header.get("model") or "replay"

    @property
    def config(self) -> RuntimeConfig:
        """The recorded run's config, as far as this version of the harness understands it."""
        known = set(RuntimeConfig.__dataclass_fields__)
        return RuntimeConfig(**{k: v for k, v in (self.header.get("config") or {}).items() if k in known})

    def complete(self, messages: list[dict], tools: list[dict], model: str) -> dict:
        self.requests.append({"messages": public_messages(messages), "tools": tools, "model": model})
        if self.index >= len(self.turns):
            raise TransportError(f"replay exhausted after {len(self.turns)} recorded turn(s)")
        turn = self.turns[self.index]
        self.index += 1
        return {"content": turn["content"],
                "tool_calls": [dict(c) for c in turn["tool_calls"]],
                "usage": dict(turn["usage"]) if turn["usage"] else None}


def replay(run_dir: Any, registry: ToolRegistry, runs_dir: Path | None = None,
           run_id: str | None = None, config: RuntimeConfig | None = None) -> RunResult:
    """Re-drive a recorded run against ``registry``. Returns the new RunResult."""
    transport = ReplayTransport(run_dir)
    target = Path(runs_dir) if runs_dir is not None else Path(run_dir).parent
    return AgentRuntime(registry, transport, target, transport.model,
                        config or transport.config,
                        run_id=run_id or f"{transport.run_id}-replay").run(transport.task)


def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def _first_difference(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _describe(step: Any, field: str, a: Any, b: Any) -> str:
    """A difference line. For long strings, point at where they diverge."""
    if isinstance(a, str) and isinstance(b, str) and (len(a) > 120 or len(b) > 120):
        i = _first_difference(a, b)
        return (f"step {step}: {field}: differs at char {i}: "
                f"recorded {_short(a[i:], 80)} != replayed {_short(b[i:], 80)}")
    return f"step {step}: {field}: recorded {_short(a)} != replayed {_short(b)}"


def compare(original: Any, replayed: Any, fields: tuple = COMPARED_FIELDS) -> list[str]:
    """Differences between two trajectories, ignoring timing and run ids.

    An empty list means the replay reproduced the recording step for step.
    """
    a = [r for r in _records(original) if r.get("type") == "step"]
    b = [r for r in _records(replayed) if r.get("type") == "step"]
    diffs: list[str] = []
    if len(a) != len(b):
        diffs.append(f"step count: {len(a)} recorded, {len(b)} replayed")
    for ra, rb in zip(a, b):
        for field in fields:
            if ra.get(field) != rb.get(field):
                diffs.append(_describe(ra.get("step"), field, ra.get(field), rb.get(field)))
    fa = next((r for r in _records(original) if r.get("type") == "footer"), {})
    fb = next((r for r in _records(replayed) if r.get("type") == "footer"), {})
    for field in ("status", "steps", "final"):
        if fa.get(field) != fb.get(field):
            diffs.append(f"footer {field}: recorded {_short(fa.get(field))} != replayed {_short(fb.get(field))}")
    return diffs
