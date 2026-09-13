"""CLI.

  harness run    --task "..." | --task-file f  --model m  --endpoint http://host:port/v1
                 [--provider openai|anthropic] [--deny-tool NAME] [--deny-shell-pattern RE]
                 [--system-prompt FILE] [--append-system-prompt FILE] [--no-global-prompt]
  harness resume <run_id> --endpoint http://host:port/v1 [--step-cap N]
  harness tools  [--tools mod] [--filter kw]
  harness prompt [--system-prompt FILE] [--append-system-prompt FILE] [--sources]
  harness trace  <run_id> [--step N | --summary]
  harness replay <run_id> [--workdir d] [--tools mod]

Exit codes for run and resume: 0 completed, 1 blocked/failed, 2
transport_error, 3 step_cap, 4 stalled. For replay: 0 identical to the
recording, 1 drifted. A bad command line (no task, unloadable --tools, a run
that cannot be resumed) is 64; an unreadable run directory is 66.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .anthropic import DEFAULT_MAX_TOKENS, AnthropicMessagesTransport
from .plugins import PluginError, load_all
from .policy import ToolPolicy
from .prompts import GLOBAL_ENV, PromptError, resolve_system_prompt
from .registry import ToolRegistry
from .replay import compare
from .resume import prepare as prepare_resume, recorded_invocation, recorded_policy
from .runtime import AgentRuntime, ResumeError, RuntimeConfig
from .tools import register_default_tools, register_scratch_tools
from .trajectory import format_summary, format_trace, read_trajectory
from .replay import replay as replay_run
from .transport import ChatCompletionsTransport, Transport

EXIT = {"completed": 0, "blocked": 1, "failed": 1, "transport_error": 2, "step_cap": 3, "stalled": 4}
USAGE_ERROR = 64
DEFAULT_TIMEOUT = 120.0
# Fields the harness owns; --extra-body may not set them.
RESERVED_BODY_KEYS = ("model", "messages", "tools", "tool_choice")


def _extra_body(raw: str | None) -> dict:
    """Parse --extra-body into provider request parameters, e.g. {"temperature": 0}."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--extra-body is not valid JSON: {e}") from None
    if not isinstance(value, dict):
        raise ValueError(f"--extra-body must be a JSON object, got {type(value).__name__}")
    reserved = [k for k in value if k in RESERVED_BODY_KEYS]
    if reserved:
        raise ValueError(f"--extra-body may not set {', '.join(reserved)}; the harness owns those fields")
    return value


def _transport(a: argparse.Namespace, extra: dict) -> Transport:
    """The provider adapter a run or resume will talk to. Raises ValueError on
    an option the provider refuses."""
    key = a.api_key or os.environ.get("HARNESS_API_KEY")
    if a.provider == "anthropic":
        return AnthropicMessagesTransport(endpoint=a.endpoint, api_key=key, timeout=a.timeout,
                                          extra=extra, max_tokens=a.max_tokens)
    return ChatCompletionsTransport(endpoint=a.endpoint, api_key=key, timeout=a.timeout, extra=extra)


def _policy(a: argparse.Namespace, recorded: Any = None) -> ToolPolicy | None:
    """--deny-tool / --deny-shell-pattern as a policy, or None. Raises ValueError
    on a pattern that is not a regex.

    ``recorded`` is the policy a run already ran under: with no flags given, a
    resume or a replay keeps it rather than quietly dropping the denials.
    """
    if a.deny_tool or a.deny_shell_pattern:
        policy = ToolPolicy(deny_tools=a.deny_tool or [], deny_shell_patterns=a.deny_shell_pattern or [])
        return policy or None
    return recorded


def _invocation(a: argparse.Namespace, workdir: Path, extra: dict) -> dict:
    """What a resume needs to reach the same provider with the same tools.
    Deliberately never the API key: it would end up in the run directory."""
    return {
        "endpoint": a.endpoint,
        "provider": a.provider,
        "max_tokens": a.max_tokens,
        "timeout": a.timeout,
        "extra_body": dict(extra),
        "tools": list(a.tools or []),
        "workdir": str(workdir),
    }


def _system_prompt(a: argparse.Namespace) -> tuple[str, list[dict]]:
    """The effective system prompt and its provenance, for the prompt flags.
    Raises PromptError for a file that was named and cannot be read."""
    return resolve_system_prompt(a.system_prompt, a.append_system_prompt or [],
                                 use_global=not a.no_global_prompt)


def _prompt_flags_given(a: argparse.Namespace) -> list[str]:
    return [flag for flag, value in (("--system-prompt", a.system_prompt),
                                     ("--append-system-prompt", a.append_system_prompt),
                                     ("--no-global-prompt", a.no_global_prompt)) if value]


def _registry(a: argparse.Namespace, workdir: Path) -> ToolRegistry:
    """Built-in tools plus whatever each --tools module contributes."""
    registry = ToolRegistry()
    register_default_tools(registry, workdir)
    added = load_all(registry, getattr(a, "tools", None), workdir)
    if added:
        print(f"loaded {len(added)} tool(s) from --tools: {', '.join(added)}", file=sys.stderr)
    return registry


def _run(a: argparse.Namespace) -> int:
    if a.task_file:
        task = Path(a.task_file).read_text(encoding="utf-8")
    elif a.task:
        task = a.task
    else:
        print("run: need --task or --task-file", file=sys.stderr)
        return USAGE_ERROR
    workdir = Path(a.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        registry = _registry(a, workdir)
    except PluginError as e:
        print(f"run: --tools {e}", file=sys.stderr)
        return USAGE_ERROR
    try:
        extra = _extra_body(a.extra_body)
        transport = _transport(a, extra)
        policy = _policy(a)
    except ValueError as e:
        print(f"run: {e}", file=sys.stderr)
        return USAGE_ERROR
    try:
        system_prompt, prompt_sources = _system_prompt(a)
    except PromptError as e:
        print(f"run: {e}", file=sys.stderr)
        return USAGE_ERROR

    cfg = RuntimeConfig(
        step_cap=a.step_cap,
        require_todos=not a.no_todo_gate,
        result_context_chars=a.result_chars,
        context_budget_chars=a.context_chars,
        preview_chars=a.preview_chars,
    )
    rt = AgentRuntime(registry, transport, Path(a.runs_dir), a.model, cfg, system_prompt=system_prompt,
                      run_id=a.run_id, policy=policy, prompt_sources=prompt_sources,
                      invocation=_invocation(a, workdir, extra))
    # the scratch pad lives in the run directory, so it can only be rooted now
    register_scratch_tools(registry, rt.run_dir)
    print(f"run {rt.run_id}  ->  {rt.run_dir}", file=sys.stderr)
    res = rt.run(task)
    print(f"status: {res.status}  steps: {res.steps}", file=sys.stderr)
    if res.final:
        print(res.final.get("content", ""))
    return EXIT.get(res.status, 1)


def _resume(a: argparse.Namespace) -> int:
    given = _prompt_flags_given(a)
    if given:
        print(f"resume: {', '.join(given)} cannot be used on resume: the system prompt is part of "
              "the conversation in state.json and is never re-resolved", file=sys.stderr)
        return USAGE_ERROR
    run_dir = Path(a.runs_dir) / a.run_id
    try:
        records = read_trajectory(run_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 66
    # everything the run recorded about how it was launched, minus the api key;
    # an explicit flag still wins.
    rec = recorded_invocation(records)
    a.endpoint = a.endpoint or rec.get("endpoint")
    if not a.endpoint:
        print("resume: no --endpoint given and the run recorded none", file=sys.stderr)
        return USAGE_ERROR
    a.provider = a.provider or rec.get("provider") or "openai"
    a.max_tokens = a.max_tokens if a.max_tokens is not None else (rec.get("max_tokens") or DEFAULT_MAX_TOKENS)
    a.timeout = a.timeout if a.timeout is not None else (rec.get("timeout") or DEFAULT_TIMEOUT)
    a.tools = a.tools or list(rec.get("tools") or [])
    workdir = Path(a.workdir or rec.get("workdir") or ".").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        registry = _registry(a, workdir)
    except PluginError as e:
        print(f"resume: --tools {e}", file=sys.stderr)
        return USAGE_ERROR
    try:
        extra = _extra_body(a.extra_body) if a.extra_body else dict(rec.get("extra_body") or {})
        transport = _transport(a, extra)
        policy = _policy(a, recorded_policy(records))
    except ValueError as e:
        print(f"resume: {e}", file=sys.stderr)
        return USAGE_ERROR

    print(f"{a.provider} at {a.endpoint}  workdir {workdir}"
          + (f"  tools {', '.join(a.tools)}" if a.tools else ""), file=sys.stderr)
    try:
        rt, detail = prepare_resume(run_dir, registry, transport, model=a.model,
                                    step_cap=a.step_cap, policy=policy,
                                    invocation=_invocation(a, workdir, extra))
        register_scratch_tools(registry, rt.run_dir)   # same pad, same run directory
        print(f"resume {rt.run_id} at step {rt.state.step} after {rt.state.status}  ->  {rt.run_dir}",
              file=sys.stderr)
        res = rt.resume(detail)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 66
    except ResumeError as e:
        print(f"resume: {e}", file=sys.stderr)
        return USAGE_ERROR
    print(f"status: {res.status}  steps: {res.steps}", file=sys.stderr)
    if res.final:
        print(res.final.get("content", ""))
    return EXIT.get(res.status, 1)


def _tools(a: argparse.Namespace) -> int:
    try:
        registry = _registry(a, Path(a.workdir).resolve())
    except PluginError as e:
        print(f"tools: --tools {e}", file=sys.stderr)
        return USAGE_ERROR
    # listing only: at run time these are rooted in the real run directory
    register_scratch_tools(registry, Path(a.runs_dir) / "<run_id>")
    entries = registry.list(a.filter)
    width = max((len(e["name"]) for e in entries), default=0)
    for e in entries:
        print(f"{e['name']:<{width}}  {e['description']}")
    print(f"\n{len(entries)} tool(s); none are in context until the agent calls toolbelt_add", file=sys.stderr)
    return 0


def _prompt(a: argparse.Namespace) -> int:
    """Print the system prompt a run would start with, or where it came from."""
    try:
        text, sources = _system_prompt(a)
    except PromptError as e:
        print(f"prompt: {e}", file=sys.stderr)
        return USAGE_ERROR
    if not a.sources:
        print(text)
        return 0
    width = max(len(s["role"]) for s in sources)
    for s in sources:
        print(f"{s['role']:<{width}}  {s['source']}  {s['chars']} chars")
    print(f"total: {len(text)} chars")
    return 0


def _replay(a: argparse.Namespace) -> int:
    source = Path(a.runs_dir) / a.run_id
    try:
        registry = _registry(a, Path(a.workdir).resolve())
    except PluginError as e:
        print(f"replay: --tools {e}", file=sys.stderr)
        return USAGE_ERROR
    try:
        res = replay_run(source, registry, runs_dir=Path(a.runs_dir), run_id=a.new_run_id)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 66
    print(f"replayed {a.run_id} -> {res.run_id}  status: {res.status}  steps: {res.steps}", file=sys.stderr)
    diffs = compare(source, res.run_dir)
    if not diffs:
        print("identical to the recording, step for step")
        return 0
    print(f"{len(diffs)} difference(s) from the recording:")
    for line in diffs[:a.max_diffs]:
        print(f"  {line}")
    if len(diffs) > a.max_diffs:
        print(f"  ... {len(diffs) - a.max_diffs} more")
    return 1


def _trace(a: argparse.Namespace) -> int:
    run_dir = Path(a.runs_dir) / a.run_id
    try:
        records = read_trajectory(run_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 66
    print(format_summary(records) if a.summary else format_trace(records, run_dir, step=a.step))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness")
    p.add_argument("--runs-dir", default="runs", help="where run directories are written (default: ./runs)")
    sub = p.add_subparsers(dest="cmd", required=True)

    tools_help = ("module exporting `registry` or `register(registry)`; dotted name or path "
                  "to a .py file. Repeatable.")

    def prompt_args(sp) -> None:
        """How the system prompt is assembled. Shared by run, resume and prompt."""
        sp.add_argument("--system-prompt", metavar="FILE", default=None,
                        help="use this file as the system prompt instead of the built-in one")
        sp.add_argument("--append-system-prompt", action="append", metavar="FILE",
                        help="append this file after the global prompt. Repeatable.")
        sp.add_argument("--no-global-prompt", action="store_true",
                        help=f"ignore the global prompt (${GLOBAL_ENV}, "
                             "$XDG_CONFIG_HOME/harness/system.md, ~/.config/harness/system.md)")

    def model_args(sp, *, model_required: bool, recorded: bool = False) -> None:
        """Everything needed to reach a provider. Shared by run and resume.

        With ``recorded``, every default is None: the run recorded what it was
        launched with, so an omitted flag means "whatever it used", and only a
        flag actually given overrides it.
        """
        was = " (default: what the run recorded)"
        sp.add_argument("--model", required=model_required, default=None,
                        help="model id" + ("" if model_required else was))
        sp.add_argument("--endpoint", required=not recorded, default=None,
                        help="provider base URL, e.g. http://localhost:8080/v1 or "
                             "https://api.anthropic.com/v1" + (was if recorded else ""))
        sp.add_argument("--provider", choices=("openai", "anthropic"),
                        default=None if recorded else "openai",
                        help="wire format: OpenAI-compatible /chat/completions (default) "
                             "or the Anthropic Messages API" + (was if recorded else ""))
        sp.add_argument("--api-key", default=None, help="or set HARNESS_API_KEY (never recorded)")
        sp.add_argument("--max-tokens", type=int, default=None if recorded else DEFAULT_MAX_TOKENS,
                        help="response cap, anthropic provider only "
                             + (was if recorded else f"(default: {DEFAULT_MAX_TOKENS})"))
        sp.add_argument("--timeout", type=float, default=None if recorded else DEFAULT_TIMEOUT,
                        help="seconds per request"
                             + (was if recorded else f" (default: {DEFAULT_TIMEOUT:g})"))
        sp.add_argument("--extra-body", default=None, metavar="JSON",
                        help='extra provider request parameters, e.g. \'{"temperature": 0}\''
                             + (was if recorded else ""))
        sp.add_argument("--workdir", default=None if recorded else ".",
                        help="root for fs_* and run_shell tools" + (was if recorded else ""))
        sp.add_argument("--tools", action="append", metavar="MODULE|PATH", help=tools_help)
        sp.add_argument("--deny-tool", action="append", metavar="NAME",
                        help="refuse this tool; the model sees the refusal as the result. Repeatable.")
        sp.add_argument("--deny-shell-pattern", action="append", metavar="REGEX",
                        help="refuse run_shell commands matching this regex. Repeatable.")

    r = sub.add_parser("run", help="run a task")
    r.add_argument("--task")
    r.add_argument("--task-file")
    model_args(r, model_required=True)
    r.add_argument("--run-id", default=None)
    r.add_argument("--step-cap", type=int, default=250)
    r.add_argument("--no-todo-gate", action="store_true")
    r.add_argument("--result-chars", type=int, default=2000)
    r.add_argument("--context-chars", type=int, default=60000)
    r.add_argument("--preview-chars", type=int, default=400)
    prompt_args(r)
    r.set_defaults(fn=_run)

    rs = sub.add_parser("resume", help="continue an interrupted run (transport_error, step_cap, stalled)")
    rs.add_argument("run_id")
    model_args(rs, model_required=False, recorded=True)
    rs.add_argument("--step-cap", type=int, default=None,
                    help="raise the cap for the rest of the run (default: the cap it ran under)")
    prompt_args(rs)      # accepted so the refusal can explain itself, never applied
    rs.set_defaults(fn=_resume)

    l = sub.add_parser("tools", help="list registered tools (what the agent can discover)")
    l.add_argument("--workdir", default=".", help="root the fs_* tools would be given")
    l.add_argument("--tools", action="append", metavar="MODULE|PATH", help=tools_help)
    l.add_argument("--filter", default=None, help="keyword filter, like toolbelt_list")
    l.set_defaults(fn=_tools)

    pr = sub.add_parser("prompt", help="print the system prompt a run would start with")
    prompt_args(pr)
    pr.add_argument("--sources", action="store_true",
                    help="print where each layer of the prompt came from instead of the prompt")
    pr.set_defaults(fn=_prompt)

    p_replay = sub.add_parser("replay", help="re-drive a recorded run against today's tools")
    p_replay.add_argument("run_id")
    p_replay.add_argument("--workdir", default=".", help="root for fs_* and run_shell tools")
    p_replay.add_argument("--tools", action="append", metavar="MODULE|PATH", help=tools_help)
    p_replay.add_argument("--new-run-id", dest="new_run_id", default=None,
                          help="run id for the replay (default: <run_id>-replay)")
    p_replay.add_argument("--max-diffs", type=int, default=20)
    p_replay.set_defaults(fn=_replay)

    t = sub.add_parser("trace", help="print a readable trace of a run")
    t.add_argument("run_id")
    view = t.add_mutually_exclusive_group()
    view.add_argument("--step", type=int, default=None, help="print one step in full")
    view.add_argument("--summary", action="store_true", help="print run stats instead of the step list")
    t.set_defaults(fn=_trace)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)
