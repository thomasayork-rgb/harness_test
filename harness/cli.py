"""CLI.

  harness run   --task "..." | --task-file f  --model m  --endpoint http://host:port/v1
  harness tools [--tools mod] [--filter kw]
  harness trace <run_id> [--step N | --summary]
  harness replay <run_id> [--workdir d] [--tools mod]

Exit codes for run: 0 completed, 1 blocked/failed, 2 transport_error,
3 step_cap, 4 stalled. For replay: 0 identical to the recording, 1 drifted.
A bad command line (no task, unloadable --tools) is 64; an unreadable run
directory is 66.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .plugins import PluginError, load_all
from .registry import ToolRegistry
from .replay import compare
from .runtime import AgentRuntime, RuntimeConfig
from .tools import register_default_tools
from .trajectory import format_summary, format_trace, read_trajectory
from .replay import replay as replay_run
from .transport import ChatCompletionsTransport

EXIT = {"completed": 0, "blocked": 1, "failed": 1, "transport_error": 2, "step_cap": 3, "stalled": 4}
USAGE_ERROR = 64
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
    except ValueError as e:
        print(f"run: {e}", file=sys.stderr)
        return USAGE_ERROR

    transport = ChatCompletionsTransport(
        endpoint=a.endpoint,
        api_key=a.api_key or os.environ.get("HARNESS_API_KEY"),
        timeout=a.timeout,
        extra=extra,
    )
    cfg = RuntimeConfig(
        step_cap=a.step_cap,
        require_todos=not a.no_todo_gate,
        result_context_chars=a.result_chars,
        context_budget_chars=a.context_chars,
        preview_chars=a.preview_chars,
    )
    rt = AgentRuntime(registry, transport, Path(a.runs_dir), a.model, cfg, run_id=a.run_id)
    print(f"run {rt.run_id}  ->  {rt.run_dir}", file=sys.stderr)
    res = rt.run(task)
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
    entries = registry.list(a.filter)
    width = max((len(e["name"]) for e in entries), default=0)
    for e in entries:
        print(f"{e['name']:<{width}}  {e['description']}")
    print(f"\n{len(entries)} tool(s); none are in context until the agent calls toolbelt_add", file=sys.stderr)
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

    r = sub.add_parser("run", help="run a task")
    r.add_argument("--task")
    r.add_argument("--task-file")
    r.add_argument("--model", required=True)
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8080/v1")
    r.add_argument("--api-key", default=None, help="or set HARNESS_API_KEY")
    r.add_argument("--workdir", default=".", help="root for fs_* and run_shell tools")
    r.add_argument("--tools", action="append", metavar="MODULE|PATH", help=tools_help)
    r.add_argument("--run-id", default=None)
    r.add_argument("--step-cap", type=int, default=250)
    r.add_argument("--no-todo-gate", action="store_true")
    r.add_argument("--result-chars", type=int, default=2000)
    r.add_argument("--context-chars", type=int, default=60000)
    r.add_argument("--preview-chars", type=int, default=400)
    r.add_argument("--timeout", type=float, default=120.0)
    r.add_argument("--extra-body", default=None, metavar="JSON",
                   help='extra provider request parameters, e.g. \'{"temperature": 0}\'')
    r.set_defaults(fn=_run)

    l = sub.add_parser("tools", help="list registered tools (what the agent can discover)")
    l.add_argument("--workdir", default=".", help="root the fs_* tools would be given")
    l.add_argument("--tools", action="append", metavar="MODULE|PATH", help=tools_help)
    l.add_argument("--filter", default=None, help="keyword filter, like toolbelt_list")
    l.set_defaults(fn=_tools)

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
