"""Agent runtime.

The loop: assemble request → complete → for each tool call: dispatch, record,
append result → enforce context budget → persist state → repeat until
final_answer is accepted, the step cap is hit, the transport dies, or the
model stops calling tools.

Failure semantics (all deliberate, all visible in the trajectory):

  tool arg validation fails   → error string as the tool result; loop continues
  unknown / inactive tool     → error string as the tool result; loop continues
  tool raises                 → error string as the tool result; loop continues
  final_answer with open todo → rejection naming the ids; loop continues
  text-only turn              → recorded as a step; nudge appended;
                                N consecutive → status "stalled"
  transport error             → retry once, then status "transport_error"
  step cap                    → status "step_cap", final null
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .context import ContextBudget
from .prompts import SYSTEM_PROMPT, TEXT_ONLY_NUDGE
from .registry import ToolRegistry, validate_args
from .todo import apply_update, open_ids, validate_todos
from .trajectory import TrajectoryWriter
from .transport import Transport, TransportError

META_TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "toolbelt_list",
        "description": "List available tools: name and one-line description. Optional keyword filter, matched against the name and that one line.",
        "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "toolbelt_inspect",
        "description": "Return the full schema for one tool. Does not activate it.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "toolbelt_add",
        "description": "Activate tools by name so they can be called. Idempotent.",
        "parameters": {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}}}, "required": ["names"]},
    }},
    {"type": "function", "function": {
        "name": "toolbelt_remove",
        "description": "Deactivate tools by name.",
        "parameters": {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}}}, "required": ["names"]},
    }},
    {"type": "function", "function": {
        "name": "todo_write",
        "description": "Create or update the todo list. merge=true updates by id; merge=false replaces the list.",
        "parameters": {"type": "object", "properties": {
            "todos": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
            }, "required": ["id", "content", "status"]}},
            "merge": {"type": "boolean"},
        }, "required": ["todos"]},
    }},
    {"type": "function", "function": {
        "name": "final_answer",
        "description": "Submit the final answer. Rejected while any todo is pending or in_progress.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["completed", "blocked", "failed"]},
            "content": {"type": "string"},
        }, "required": ["status", "content"]},
    }},
]
META_NAMES = {t["function"]["name"] for t in META_TOOLS}

# A turn can be cut short mid-way: the step cap trips between two calls of the
# same turn, or final_answer is accepted and the run ends. The assistant
# message is already in the transcript with every call it asked for, so the
# calls that never ran still need a result - an OpenAI-shaped transcript with a
# tool_call and no matching tool message is malformed, and a resumed run would
# send exactly that back to the provider.
NOT_EXECUTED = "error: not executed: {reason}"
_META_PARAMS = {t["function"]["name"]: t["function"]["parameters"] for t in META_TOOLS}


@dataclass
class RuntimeConfig:
    step_cap: int = 250
    require_todos: bool = True
    result_context_chars: int = 2000
    context_budget_chars: int = 60000
    preview_chars: int = 400
    text_only_limit: int = 3
    transport_retries: int = 1


@dataclass
class RunState:
    run_id: str
    model: str
    status: str = "running"
    step: int = 0
    step_cap: int = 250
    active_tools: list[str] = field(default_factory=list)
    todo_initialized: bool = False
    todos: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    final: dict | None = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class RunResult:
    run_id: str
    status: str
    steps: int
    final: dict | None
    run_dir: Path


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(result)


class AgentRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        transport: Transport,
        runs_dir: Path,
        model: str,
        config: RuntimeConfig | None = None,
        system_prompt: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.registry = registry
        self.transport = transport
        self.model = model
        self.config = config or RuntimeConfig()
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.run_dir = Path(runs_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.budget = ContextBudget(self.config.result_context_chars, self.config.context_budget_chars)
        self.writer = TrajectoryWriter(self.run_dir, preview_chars=self.config.preview_chars)
        self.state = RunState(run_id=self.run_id, model=model, step_cap=self.config.step_cap)

    # ---- request assembly -------------------------------------------------

    def _tools_payload(self) -> list[dict]:
        active = []
        for name in self.state.active_tools:
            spec = self.registry.get(name)
            if spec:
                active.append(spec.schema())
        return META_TOOLS + active

    def _complete(self, messages: list[dict]) -> dict:
        last: Exception | None = None
        for _ in range(self.config.transport_retries + 1):
            try:
                return self.transport.complete(messages, self._tools_payload(), self.model)
            except TransportError as e:
                last = e
        raise last  # type: ignore[misc]

    # ---- dispatch ---------------------------------------------------------

    def _dispatch(self, name: str, args: Any) -> tuple[str, str]:
        """Returns (result_text, kind). kind ∈ ok | error | final_accepted | final_rejected."""
        if name in META_NAMES:
            err = validate_args(_META_PARAMS[name], args)
            if err:
                return f"error: {err}", "error"
            return getattr(self, f"_meta_{name}")(args)

        spec = self.registry.get(name)
        if spec is None:
            return f"error: unknown tool '{name}'. Use toolbelt_list to see available tools.", "error"
        if name not in self.state.active_tools:
            return f"error: tool '{name}' is not active. Call toolbelt_add first.", "error"
        err = validate_args(spec.parameters, args)
        if err:
            return f"error: {err}", "error"
        try:
            result = spec.fn(**args)
        except Exception as e:  # noqa: BLE001 - the model needs to see it, not a crash
            return f"error: {type(e).__name__}: {e}", "error"
        return _to_text(result), "ok"

    def _meta_toolbelt_list(self, args: dict) -> tuple[str, str]:
        return _to_text(self.registry.list(args.get("filter"))), "ok"

    def _meta_toolbelt_inspect(self, args: dict) -> tuple[str, str]:
        spec = self.registry.get(args["name"])
        if spec is None:
            return f"error: unknown tool '{args['name']}'", "error"
        return _to_text(spec.schema()["function"]), "ok"

    def _meta_toolbelt_add(self, args: dict) -> tuple[str, str]:
        added, already, unknown = [], [], []
        for n in args["names"]:
            if n not in self.registry:
                unknown.append(n)
            elif n in self.state.active_tools:
                already.append(n)
            else:
                self.state.active_tools.append(n)
                added.append(n)
        out = {"added": added, "already_active": already, "unknown": unknown, "active": list(self.state.active_tools)}
        return _to_text(out), ("error" if unknown and not added else "ok")

    def _meta_toolbelt_remove(self, args: dict) -> tuple[str, str]:
        removed = [n for n in args["names"] if n in self.state.active_tools]
        self.state.active_tools = [n for n in self.state.active_tools if n not in removed]
        return _to_text({"removed": removed, "active": list(self.state.active_tools)}), "ok"

    def _meta_todo_write(self, args: dict) -> tuple[str, str]:
        err = validate_todos(args["todos"])
        if err:
            return f"error: {err}", "error"
        merge = bool(args.get("merge", True))
        self.state.todos = apply_update(self.state.todos, args["todos"], merge)
        self.state.todo_initialized = True
        return _to_text({"todos": self.state.todos, "open": open_ids(self.state.todos)}), "ok"

    def _meta_final_answer(self, args: dict) -> tuple[str, str]:
        if self.config.require_todos and not self.state.todo_initialized:
            return _to_text({"accepted": False, "reason": "no todo list. Call todo_write first."}), "final_rejected"
        open_ = open_ids(self.state.todos)
        if open_:
            return _to_text({"accepted": False, "reason": "open todos remain", "open_ids": open_}), "final_rejected"
        self.state.final = {"status": args["status"], "content": args["content"]}
        return _to_text({"accepted": True}), "final_accepted"

    def _close_unexecuted(self, calls: list[dict], status: str) -> None:
        """Answer the calls of a turn that was cut short, so the transcript stays well formed."""
        reason = {
            "step_cap": "the step cap was reached before this call ran",
        }.get(status, "the run ended before this call ran")
        for c in calls:
            self.state.messages.append({
                "role": "tool", "tool_call_id": c["id"],
                "content": NOT_EXECUTED.format(reason=reason),
                # no artifact to point at, so keep it out of the eviction pass
                "_artifact": None, "_protected": True, "_unexecuted": True,
            })

    # ---- the loop ---------------------------------------------------------

    def run(self, task: str) -> RunResult:
        cfg = self.config
        st = self.state
        st.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        self.writer.header(run_id=self.run_id, model=self.model, step_cap=cfg.step_cap, task=task, config=asdict(cfg))
        st.save(self.state_path)

        text_only = 0
        status = "running"
        detail: str | None = None

        while status == "running":
            if st.step >= cfg.step_cap:
                status, detail = "step_cap", f"step cap {cfg.step_cap} reached"
                break

            t0 = time.monotonic()
            try:
                resp = self._complete(st.messages)
            except TransportError as e:
                status, detail = "transport_error", str(e)
                break
            elapsed = int((time.monotonic() - t0) * 1000)

            reasoning = (resp.get("content") or "").strip()
            usage = resp.get("usage") or {}
            tok_in, tok_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
            calls = resp.get("tool_calls") or []

            if not calls:
                st.step += 1
                text_only += 1
                art = self.writer.write_artifact(st.step, None, reasoning)
                self.writer.step(run_id=self.run_id, step=st.step, elapsed_ms=elapsed, reasoning=reasoning,
                                 tool=None, args=None, result="", artifact=art, tokens_in=tok_in, tokens_out=tok_out,
                                 todo_snapshot=list(st.todos), kind="text_only", call_index=0)
                st.messages.append({"role": "assistant", "content": reasoning})
                if text_only >= cfg.text_only_limit:
                    status, detail = "stalled", f"{text_only} consecutive turns without a tool call"
                    break
                st.messages.append({"role": "user", "content": TEXT_ONLY_NUDGE})
                st.save(self.state_path)
                continue

            text_only = 0
            st.messages.append({
                "role": "assistant",
                "content": reasoning or None,
                "tool_calls": [{
                    "id": c["id"], "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"]) if not isinstance(c["arguments"], str) else c["arguments"]},
                } for c in calls],
            })

            executed = 0
            for index, c in enumerate(calls):
                first = index == 0
                if st.step >= cfg.step_cap:
                    status, detail = "step_cap", f"step cap {cfg.step_cap} reached"
                    break
                st.step += 1
                t1 = time.monotonic()
                result, kind = self._dispatch(c["name"], c["arguments"])
                dispatch_ms = int((time.monotonic() - t1) * 1000)
                art = self.writer.write_artifact(st.step, c["name"], result)
                self.writer.step(run_id=self.run_id, step=st.step, elapsed_ms=(elapsed if first else 0) + dispatch_ms,
                                 reasoning=reasoning if first else "", tool=c["name"], args=c["arguments"], result=result,
                                 artifact=art, tokens_in=tok_in if first else None, tokens_out=tok_out if first else None,
                                 todo_snapshot=list(st.todos), kind=kind, call_index=index)
                st.messages.append({
                    "role": "tool", "tool_call_id": c["id"],
                    "content": self.budget.truncate_result(result, art),
                    "_artifact": art, "_protected": c["name"] == "todo_write",
                })
                executed += 1
                if kind == "final_accepted":
                    status = st.final["status"]  # type: ignore[index]
                    break

            self._close_unexecuted(calls[executed:], status)
            self.budget.enforce(st.messages)
            st.save(self.state_path)

        st.status = status
        st.save(self.state_path)
        self.writer.footer(run_id=self.run_id, status=status, steps=st.step, final=st.final, detail=detail)
        return RunResult(run_id=self.run_id, status=status, steps=st.step, final=st.final, run_dir=self.run_dir)
