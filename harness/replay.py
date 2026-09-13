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
the tools decided is. A recording that was resumed replays as one straight
run - the turns of every segment in order, under the configuration the last
segment ended with - so its footer is compared against the recording's last.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .policy import from_description
from .registry import ToolRegistry
from .runtime import AgentRuntime, RunResult, RuntimeConfig, effective_config
from .skills import discover
from .tools.scratch import register_scratch_tools
from .trajectory import last, read_trajectory, setting
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
        self.footer = last(self.records, "footer")
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
        return (last(self.records, "resume") or self.header).get("model") or "replay"

    @property
    def policy(self) -> Any:
        """The tool-call policy the recording ran under, rebuilt. Without it a
        call the original run denied would run for real, and every step after
        it would drift."""
        return from_description(setting(self.records, "policy"))

    @property
    def skills(self) -> Any:
        """The skills the recording could load, rediscovered from the
        directories the header names. Without them a recorded skill_load would
        come back "unknown tool" and drag every later step into drift."""
        recorded = setting(self.records, "skills") or {}
        return discover(recorded.get("dirs") or [],
                        (setting(self.records, "invocation") or {}).get("workdir"))

    @property
    def config(self) -> RuntimeConfig:
        """The config the recording ended under, as far as this version of the
        harness understands it. A resumed recording may have raised the step
        cap; replaying under the original cap would stop short of the tape."""
        return effective_config(self.records)

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
           run_id: str | None = None, config: RuntimeConfig | None = None,
           policy: Any = None, skills: Any = None) -> RunResult:
    """Re-drive a recorded run against ``registry``. Returns the new RunResult.

    The policy defaults to the one the recording ran under, so denied calls
    stay denied instead of running for real; pass one to replay under different
    rules, or ``ToolPolicy()`` to replay under none.

    The scratch pad is registered for the replay's own run directory unless the
    registry already has one, so a recording that wrote notes replays as a run
    that writes notes rather than one calling an unknown tool.
    """
    transport = ReplayTransport(run_dir)
    target = Path(runs_dir) if runs_dir is not None else Path(run_dir).parent
    runtime = AgentRuntime(registry, transport, target, transport.model,
                           config or transport.config,
                           policy=policy if policy is not None else transport.policy,
                           run_id=run_id or f"{transport.run_id}-replay",
                           skills=skills if skills is not None else transport.skills)
    if "scratch_write" not in registry:
        register_scratch_tools(registry, runtime.run_dir)
    return runtime.run(transport.task)


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
    fa, fb = last(_records(original), "footer"), last(_records(replayed), "footer")
    for field in ("status", "steps", "final"):
        if fa.get(field) != fb.get(field):
            diffs.append(f"footer {field}: recorded {_short(fa.get(field))} != replayed {_short(fb.get(field))}")
    return diffs
