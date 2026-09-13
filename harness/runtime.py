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
  policy denies a call        → the denial as the tool result, kind "denied";
                                loop continues (see harness.policy)

A run that ended in one of RESUMABLE can be picked up again: rebuild the
runtime from the persisted state and call ``resume()`` instead of ``run()``.
The loop is the same loop; only the way the conversation starts differs.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .context import ContextBudget
from .plugins import PluginError, load_tools
from .prompts import BUILTIN, PROVIDED, SYSTEM_PROMPT, TEXT_ONLY_NUDGE, builtin_prompt
from .registry import ToolRegistry, validate_args
from .skills import SkillSet
from .todo import NOTES_MAX, apply_update, open_ids, signature, validate_todos
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
        "description": "Create or update the todo list. merge=true updates by id, keeping a note the update omits; merge=false replaces the list. Record the outcome of each item in its notes.",
        "parameters": {"type": "object", "properties": {
            "todos": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                "notes": {"type": "string", "description": f"what happened with this item: the value, the path, the command and its result (max {NOTES_MAX} chars)"},
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

# The skill meta-tools exist only for a run that discovered skills: a run
# without them starts with exactly the six above, and nothing in context
# mentions a lever that is not there.
SKILL_META_TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "skill_list",
        "description": "List available skills: name and one-line description. Optional keyword filter, matched against the name and that one line.",
        "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "skill_load",
        "description": "Load a skill: its full text becomes this result, and the tools it declares are activated.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "skill_unload",
        "description": "Drop a loaded skill's text from context when you are done with it. The tools it activated stay active.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
]
SKILL_META_NAMES = {t["function"]["name"] for t in SKILL_META_TOOLS}
ALL_META_NAMES = META_NAMES | SKILL_META_NAMES

# Statuses a run can be picked up from. completed/blocked/failed are answers,
# not interruptions, and "running" means some other process still owns the run.
RESUMABLE = ("transport_error", "step_cap", "stalled")

RESUME_NOTE = ("This run was interrupted ({status}: {detail}) and has been resumed. "
               "The conversation above is yours; continue from it. Step {step} of {cap}.")

# A turn can be cut short mid-way: the step cap trips between two calls of the
# same turn, or final_answer is accepted and the run ends. The assistant
# message is already in the transcript with every call it asked for, so the
# calls that never ran still need a result - an OpenAI-shaped transcript with a
# tool_call and no matching tool message is malformed, and a resumed run would
# send exactly that back to the provider.
NOT_EXECUTED = "error: not executed: {reason}"

# A denied call is not an error: the tool did not fail, it never ran. Its own
# kind keeps the two apart in a trace and in trace --summary.
DENIAL = "denied: {reason}"
_META_PARAMS = {t["function"]["name"]: t["function"]["parameters"]
                for t in META_TOOLS + SKILL_META_TOOLS}

# A loaded skill's text is the one tool result that is neither truncated nor
# evicted: the model was told to follow it, so it has to still be there.
SKILL_LOADED = "skill '{name}' loaded ({chars} chars). {tools}"
SKILL_UNLOADED = "[skill '{name}' unloaded; its text is out of context. skill_load reads it again.]"


class ResumeError(Exception):
    """A run that cannot be resumed, with the reason the CLI should print."""


@dataclass
class RuntimeConfig:
    step_cap: int = 250
    require_todos: bool = True
    result_context_chars: int = 2000
    context_budget_chars: int = 60000
    preview_chars: int = 400
    text_only_limit: int = 3
    transport_retries: int = 1
    skill_chars: int = 12000


def config_from(stored: dict | None) -> RuntimeConfig:
    """A RuntimeConfig from a recorded ``config`` block, ignoring fields this
    version of the harness no longer knows about."""
    known = set(RuntimeConfig.__dataclass_fields__)
    return RuntimeConfig(**{k: v for k, v in (stored or {}).items() if k in known})


def effective_config(records: list[dict]) -> RuntimeConfig:
    """The config a trajectory ended under: the header's, or the last resume
    record's if the run was resumed (a resume may raise the step cap)."""
    stored: dict = {}
    for rec in records:
        if rec.get("type") in ("header", "resume") and rec.get("config"):
            stored = rec["config"]
    return config_from(stored)


@dataclass
class RunState:
    run_id: str
    model: str
    status: str = "running"
    step: int = 0
    step_cap: int = 250
    active_tools: list[str] = field(default_factory=list)
    loaded_skills: list[str] = field(default_factory=list)
    todo_initialized: bool = False
    todos: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    final: dict | None = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunState":
        stored = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(stored) - known)
        if unknown:
            raise ResumeError(f"{path}: unknown state field(s): {', '.join(unknown)}")
        return cls(**stored)


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
        state: RunState | None = None,
        policy: Any = None,
        prompt_sources: list[dict] | None = None,
        invocation: dict | None = None,
        skills: SkillSet | None = None,
    ) -> None:
        self.registry = registry
        self.transport = transport
        self.model = model
        self.config = config or RuntimeConfig()
        # what discovery found, if anything: names and descriptions only until
        # the model calls skill_load (see harness.skills)
        self.skills = skills if skills is not None else SkillSet()
        self.system_prompt = system_prompt or builtin_prompt(bool(self.skills))
        # how this run was launched: endpoint, provider, workdir, tool modules.
        # Recorded so `resume` can default to it; never holds the API key.
        self.invocation = dict(invocation or {})
        # where that prompt came from, for the header (see harness.prompts)
        self.prompt_sources = prompt_sources or [
            {"source": BUILTIN if system_prompt is None else PROVIDED,
             "role": "base", "chars": len(self.system_prompt)}]
        # policy(name, args) -> denial message or None; see harness.policy
        self.policy = policy or None
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.run_dir = Path(runs_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.budget = ContextBudget(self.config.result_context_chars, self.config.context_budget_chars)
        self.writer = TrajectoryWriter(self.run_dir, preview_chars=self.config.preview_chars)
        self.state = state or RunState(run_id=self.run_id, model=model, step_cap=self.config.step_cap)
        self.state.step_cap = self.config.step_cap
        # extra keys for the tool message of the call being dispatched, set by a
        # meta-tool that needs one (skill_load protects and un-truncates its own)
        self._annotate: dict = {}

    # ---- request assembly -------------------------------------------------

    def _tools_payload(self) -> list[dict]:
        active = []
        for name in self.state.active_tools:
            spec = self.registry.get(name)
            if spec:
                active.append(spec.schema())
        meta = META_TOOLS + SKILL_META_TOOLS if self.skills else META_TOOLS
        return meta + active

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
        """Returns (result_text, kind). kind ∈ ok | error | denied | final_accepted | final_rejected."""
        if self.policy is not None:
            try:
                verdict = self.policy(name, args)
            except Exception as e:  # noqa: BLE001 - a broken policy must not kill the run
                return f"error: policy raised {type(e).__name__}: {e}", "error"
            if verdict:
                return DENIAL.format(reason=verdict), "denied"
        if name in META_NAMES or (self.skills and name in SKILL_META_NAMES):
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

    def _meta_skill_list(self, args: dict) -> tuple[str, str]:
        return _to_text(self.skills.list(args.get("filter"))), "ok"

    def _meta_skill_load(self, args: dict) -> tuple[str, str]:
        """The skill's text as the tool result, plus the tools it declares.

        Nothing is half-done: an oversized skill or a plugin that will not load
        leaves no tools activated and nothing marked loaded, so the model can
        pick another route with an accurate picture of its context.
        """
        name = args["name"]
        skill = self.skills.get(name)
        if skill is None:
            known = ", ".join(self.skills.names()) or "(none)"
            return f"error: unknown skill '{name}'. Available: {known}", "error"
        if name in self.state.loaded_skills:
            return f"skill '{name}' is already loaded; its text is above in this conversation.", "ok"
        if skill.chars > self.config.skill_chars:
            return (f"error: skill '{name}' is {skill.chars} chars, over the {self.config.skill_chars} "
                    "char limit for one skill (--skill-chars). It was not loaded."), "error"

        if skill.plugin_path is not None:
            try:
                added = load_tools(self.registry, str(skill.plugin_path), self.skills.workdir or Path("."))
            except PluginError as e:
                return f"error: skill '{name}' declares a plugin that will not load: {e}", "error"
            plugin_note = f"plugin: registered {', '.join(added) or 'nothing'}. "
        else:
            plugin_note = ""

        activated, already, unknown = [], [], []
        for tool in skill.tools:
            if tool not in self.registry:
                unknown.append(tool)
            elif tool in self.state.active_tools:
                already.append(tool)
            else:
                self.state.active_tools.append(tool)
                activated.append(tool)
        parts = []
        if activated:
            parts.append("activated " + ", ".join(activated))
        if already:
            parts.append("already active: " + ", ".join(already))
        if unknown:
            parts.append("declared but unknown: " + ", ".join(unknown))
        tools_note = ("tools: " + "; ".join(parts) + "." if parts else "No tools declared.")

        self.state.loaded_skills.append(name)
        self._annotate = {"_protected": True, "_skill": name, "_full": True}
        header = SKILL_LOADED.format(name=name, chars=skill.chars, tools=plugin_note + tools_note)
        return f"{header}\n\n{skill.body}", "ok"

    def _meta_skill_unload(self, args: dict) -> tuple[str, str]:
        name = args["name"]
        if name not in self.state.loaded_skills:
            loaded = ", ".join(self.state.loaded_skills) or "(none)"
            return f"error: skill '{name}' is not loaded. Loaded: {loaded}", "error"
        marker = SKILL_UNLOADED.format(name=name)
        freed = 0
        for m in self.state.messages:
            if m.get("_skill") == name:
                freed += max(0, len(m.get("content") or "") - len(marker))
                m["content"] = marker
                m["_protected"] = False
                m["_skill"] = None
        self.state.loaded_skills.remove(name)
        return _to_text({"unloaded": name, "context_freed_chars": freed,
                         "loaded": list(self.state.loaded_skills)}), "ok"

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

    def _policy_description(self) -> Any:
        """What the header records about the policy in force, if it can say."""
        describe = getattr(self.policy, "describe", None)
        if callable(describe):
            return describe()
        return None if self.policy is None else repr(self.policy)

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
        """Start a fresh run: header, opening conversation, then the loop."""
        cfg = self.config
        st = self.state
        st.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        self.writer.header(run_id=self.run_id, model=self.model, step_cap=cfg.step_cap, task=task,
                           config=asdict(cfg), policy=self._policy_description(),
                           prompt_sources=list(self.prompt_sources),
                           invocation=dict(self.invocation),
                           skills=self.skills.describe())
        st.save(self.state_path)
        return self._loop()

    def resume(self, detail: str | None = None) -> RunResult:
        """Continue an interrupted run from its persisted state.

        ``detail`` is what killed the previous segment (the old footer's
        detail), quoted back to the model in the resume note.

        The trajectory is appended to, not replaced: a ``resume`` record marks
        the seam and the step counter carries on, so one file still tells the
        whole story. The conversation gets one user message saying what
        happened, which also keeps the transcript ending on a user turn however
        the previous segment died.
        """
        cfg = self.config
        st = self.state
        if st.status not in RESUMABLE:
            raise ResumeError(
                f"run {st.run_id} ended with status '{st.status}'; only "
                f"{', '.join(RESUMABLE)} can be resumed")
        if st.step >= cfg.step_cap:
            raise ResumeError(
                f"run {st.run_id} is at step {st.step} with cap {cfg.step_cap}; "
                "raise it with --step-cap to resume")
        note = RESUME_NOTE.format(status=st.status, detail=detail or "no detail",
                                  step=st.step, cap=cfg.step_cap)
        self.writer.resume(run_id=self.run_id, model=self.model, from_status=st.status,
                           from_step=st.step, from_detail=detail, step_cap=cfg.step_cap,
                           config=asdict(cfg), note=note, policy=self._policy_description(),
                           invocation=dict(self.invocation))
        st.messages.append({"role": "user", "content": note})
        st.status = "running"
        st.save(self.state_path)
        return self._loop()

    def _loop(self) -> RunResult:
        cfg = self.config
        st = self.state
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
                self._annotate = {}
                result, kind = self._dispatch(c["name"], c["arguments"])
                dispatch_ms = int((time.monotonic() - t1) * 1000)
                art = self.writer.write_artifact(st.step, c["name"], result)
                self.writer.step(run_id=self.run_id, step=st.step, elapsed_ms=(elapsed if first else 0) + dispatch_ms,
                                 reasoning=reasoning if first else "", tool=c["name"], args=c["arguments"], result=result,
                                 artifact=art, tokens_in=tok_in if first else None, tokens_out=tok_out if first else None,
                                 todo_snapshot=list(st.todos), kind=kind, call_index=index)
                extra = dict(self._annotate)
                whole = extra.pop("_full", False)
                message = {
                    "role": "tool", "tool_call_id": c["id"],
                    "content": result if whole else self.budget.truncate_result(result, art),
                    "_artifact": art,
                    "_protected": c["name"] == "todo_write" or bool(extra.get("_protected")),
                }
                message.update(extra)
                st.messages.append(message)
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
