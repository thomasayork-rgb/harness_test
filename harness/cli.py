"""CLI.

  harness run   --task "..." | --task-file f  --model m  --endpoint http://host:port/v1
  harness trace <run_id> [--step N]

Exit codes for run: 0 completed, 1 blocked/failed, 2 transport_error,
3 step_cap, 4 stalled.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .registry import ToolRegistry
from .runtime import AgentRuntime, RuntimeConfig
from .tools import register_default_tools
from .trajectory import format_trace, read_trajectory
from .transport import ChatCompletionsTransport

EXIT = {"completed": 0, "blocked": 1, "failed": 1, "transport_error": 2, "step_cap": 3, "stalled": 4}


def _run(a: argparse.Namespace) -> int:
    if a.task_file:
        task = Path(a.task_file).read_text(encoding="utf-8")
    elif a.task:
        task = a.task
    else:
        print("run: need --task or --task-file", file=sys.stderr)
        return 64
    workdir = Path(a.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    registry = ToolRegistry()
    register_default_tools(registry, workdir)

    transport = ChatCompletionsTransport(
        endpoint=a.endpoint,
        api_key=a.api_key or os.environ.get("HARNESS_API_KEY"),
        timeout=a.timeout,
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


def _trace(a: argparse.Namespace) -> int:
    run_dir = Path(a.runs_dir) / a.run_id
    try:
        records = read_trajectory(run_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 66
    print(format_trace(records, run_dir, step=a.step))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness")
    p.add_argument("--runs-dir", default="runs", help="where run directories are written (default: ./runs)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a task")
    r.add_argument("--task")
    r.add_argument("--task-file")
    r.add_argument("--model", required=True)
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8080/v1")
    r.add_argument("--api-key", default=None, help="or set HARNESS_API_KEY")
    r.add_argument("--workdir", default=".", help="root for fs_* and run_shell tools")
    r.add_argument("--run-id", default=None)
    r.add_argument("--step-cap", type=int, default=250)
    r.add_argument("--no-todo-gate", action="store_true")
    r.add_argument("--result-chars", type=int, default=2000)
    r.add_argument("--context-chars", type=int, default=60000)
    r.add_argument("--preview-chars", type=int, default=400)
    r.add_argument("--timeout", type=float, default=120.0)
    r.set_defaults(fn=_run)

    t = sub.add_parser("trace", help="print a readable trace of a run")
    t.add_argument("run_id")
    t.add_argument("--step", type=int, default=None, help="print one step in full")
    t.set_defaults(fn=_trace)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)
