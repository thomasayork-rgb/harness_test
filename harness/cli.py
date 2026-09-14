"""CLI.

  harness run    --task "..." | --task-file f  --model m  --endpoint http://host:port/v1
                 [--provider openai|anthropic] [--deny-tool NAME] [--deny-shell-pattern RE]
                 [--system-prompt FILE] [--append-system-prompt FILE] [--no-global-prompt]
  harness resume <run_id> [--step-cap N] [--progress-nudge N] [--force]
                 (provider, tools, skills and policy flags default to the recording)
  harness tools  [--tools mod] [--filter kw]
  harness skills [--skills DIR] [--workdir d] [--filter kw]
  harness prompt [--system-prompt FILE] [--append-system-prompt FILE] [--sources]
  harness bench  TASKS.jsonl --model m --endpoint http://host:port/v1
  harness trace  <run_id> [--step N | --summary]
  harness replay <run_id> [--workdir d] [--tools mod] [--deny-tool NAME]

Exit codes for run and resume: 0 completed, 1 blocked/failed, 2
transport_error, 3 step_cap, 4 stalled, 130 interrupted. For replay: 0 identical to the
recording, 1 drifted. For bench: 0 if every task completed, 1 otherwise. A
bad command line (no task, unloadable --tools, an unreadable prompt file, a
run that cannot be resumed) is 64; an unreadable run directory is 66.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from .skills import SkillSet, discover, search_dirs
from .tools import register_default_tools, register_scratch_tools
from .trajectory import format_summary, format_trace, read_trajectory, summarize
from .replay import replay as replay_run
from .transport import ChatCompletionsTransport, Transport

EXIT = {"completed": 0, "blocked": 1, "failed": 1, "transport_error": 2, "step_cap": 3,
        "stalled": 4, "interrupted": 130}
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


def _invocation(a: argparse.Namespace, workdir: Path, extra: dict,
                skills: SkillSet | None = None) -> dict:
    """What a resume needs to reach the same provider with the same tools and
    the same skills. Deliberately never the API key: it would end up in the run
    directory."""
    return {
        "endpoint": a.endpoint,
        "provider": a.provider,
        "max_tokens": a.max_tokens,
        "timeout": a.timeout,
        "extra_body": dict(extra),
        "tools": list(a.tools or []),
        "skills": [str(d) for d in (skills.dirs if skills else [])],
        "workdir": str(workdir),
    }


def _system_prompt(a: argparse.Namespace, skills: bool = False) -> tuple[str, list[dict]]:
    """The effective system prompt and its provenance, for the prompt flags.
    Raises PromptError for a file that was named and cannot be read."""
    return resolve_system_prompt(a.system_prompt, a.append_system_prompt or [],
                                 use_global=not a.no_global_prompt, skills=skills)


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


def _discover_skills(a: argparse.Namespace, workdir: Path, recorded: Any = None) -> SkillSet:
    """What this command line can see. ``recorded`` is the directory list a run
    already searched: with no --skills given, a resume keeps it rather than
    re-resolving an environment that may have moved.

    Whatever discovery wants to say - a broken skill file, a name found twice -
    is said once, here, and never to the model.
    """
    flags = list(getattr(a, "skills", None) or [])
    if recorded is not None and not flags:
        dirs = [Path(d) for d in recorded]
    else:
        for named in flags:
            if not Path(named).expanduser().is_dir():
                print(f"skills: --skills {named}: no such directory", file=sys.stderr)
        dirs = search_dirs(flags, workdir)
    found = discover(dirs, workdir)
    for line in found.warnings():
        print(f"skills: {line}", file=sys.stderr)
    return found


def _build(a: argparse.Namespace, workdir: Path, run_id: str | None) -> AgentRuntime:
    """Everything one run needs, assembled from the command line. Raises
    PluginError, ValueError or PromptError; nothing is created until they pass."""
    registry = _registry(a, workdir)
    skills = _discover_skills(a, workdir)
    extra = _extra_body(a.extra_body)
    transport = _transport(a, extra)
    policy = _policy(a)
    system_prompt, prompt_sources = _system_prompt(a, skills=bool(skills))
    cfg = RuntimeConfig(
        step_cap=a.step_cap,
        require_todos=not a.no_todo_gate,
        result_context_chars=a.result_chars,
        context_budget_chars=a.context_chars,
        preview_chars=a.preview_chars,
        skill_chars=a.skill_chars,
        progress_nudge_steps=a.progress_nudge,
        trust_project_plugins=a.trust_project_plugins,
    )
    rt = AgentRuntime(registry, transport, Path(a.runs_dir), a.model, cfg, system_prompt=system_prompt,
                      run_id=run_id, policy=policy, prompt_sources=prompt_sources,
                      invocation=_invocation(a, workdir, extra, skills), skills=skills)
    # the scratch pad lives in the run directory, so it can only be rooted now
    register_scratch_tools(registry, rt.run_dir)
    return rt


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
        rt = _build(a, workdir, a.run_id)
    except PluginError as e:
        print(f"run: --tools {e}", file=sys.stderr)
        return USAGE_ERROR
    except (ValueError, PromptError) as e:
        print(f"run: {e}", file=sys.stderr)
        return USAGE_ERROR
    print(f"run {rt.run_id}  ->  {rt.run_dir}", file=sys.stderr)
    res = rt.run(task)
    print(f"status: {res.status}  steps: {res.steps}", file=sys.stderr)
    if res.final:
        print(res.final.get("content", ""))
    return EXIT.get(res.status, 1)


BENCH_FILE = "bench.jsonl"
# (heading, row key, right-align)
BENCH_COLUMNS = (("run id", "run_id", False), ("status", "status", False), ("steps", "steps", True),
                 ("tokens in", "tokens_in", True), ("tokens out", "tokens_out", True),
                 ("elapsed", "elapsed", True))


def _bench_tasks(path: Path) -> list[dict]:
    """One task per JSONL line: {"task": str, "workdir": str?, "run_id": str?}.
    Raises ValueError naming the line that is wrong."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"{path}: {e.strerror or e}") from None
    tasks = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: not valid JSON: {e}") from None
        if not isinstance(entry, dict) or not isinstance(entry.get("task"), str) or not entry["task"].strip():
            raise ValueError(f'{path}:{n}: each line needs a non-empty "task" string')
        tasks.append(entry)
    if not tasks:
        raise ValueError(f"{path}: no tasks")
    return tasks


def _bench(a: argparse.Namespace) -> int:
    """Run a file of tasks back to back under one set of flags, and tabulate."""
    try:
        tasks = _bench_tasks(a.tasks)
    except ValueError as e:
        print(f"bench: {e}", file=sys.stderr)
        return USAGE_ERROR

    rows: list[dict] = []
    for n, entry in enumerate(tasks, 1):
        workdir = Path(entry.get("workdir") or a.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            rt = _build(a, workdir, entry.get("run_id"))
        except PluginError as e:
            print(f"bench: --tools {e}", file=sys.stderr)
            return USAGE_ERROR
        except (ValueError, PromptError) as e:
            print(f"bench: {e}", file=sys.stderr)
            return USAGE_ERROR
        print(f"[{n}/{len(tasks)}] {rt.run_id}  {entry['task'][:60]}", file=sys.stderr)
        res = rt.run(entry["task"])
        stats = summarize(read_trajectory(res.run_dir))
        rows.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "run_id": res.run_id, "task": entry["task"], "workdir": str(workdir),
            "status": res.status, "steps": res.steps,
            "tokens_in": stats["tokens_in"], "tokens_out": stats["tokens_out"],
            "elapsed_ms": stats["elapsed_ms"], "wall_s": stats["wall_s"],
            "errors": stats["errors"], "denied": stats["denied"], "final": res.final,
        })

    path = Path(a.runs_dir) / BENCH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:      # appended: a bench log, not a snapshot
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    table = [[label for label, _, _ in BENCH_COLUMNS]]
    for row in rows:
        cells = dict(row, elapsed=f"{row['elapsed_ms'] / 1000:.1f} s")
        table.append([str(cells[key]) for _, key, _ in BENCH_COLUMNS])
    widths = [max(len(r[i]) for r in table) for i in range(len(BENCH_COLUMNS))]
    for line in table:
        print("  ".join(cell.rjust(w) if right else cell.ljust(w)
                        for cell, w, (_, _, right) in zip(line, widths, BENCH_COLUMNS)).rstrip())
    done = sum(1 for r in rows if r["status"] == "completed")
    other = ", ".join(sorted({r["status"] for r in rows if r["status"] != "completed"}))
    print(f"\n{len(rows)} task(s): {done} completed" + (f", {len(rows) - done} not ({other})" if other else ""))
    print(f"bench log: {path}", file=sys.stderr)
    return 0 if done == len(rows) else 1


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
    skills = _discover_skills(a, workdir, recorded=rec.get("skills"))
    try:
        extra = _extra_body(a.extra_body) if a.extra_body else dict(rec.get("extra_body") or {})
        transport = _transport(a, extra)
        policy = _policy(a, recorded_policy(records))
    except ValueError as e:
        print(f"resume: {e}", file=sys.stderr)
        return USAGE_ERROR

    print(f"{a.provider} at {a.endpoint}  workdir {workdir}"
          + (f"  tools {', '.join(a.tools)}" if a.tools else "")
          + (f"  skills {len(skills)}" if skills else ""), file=sys.stderr)
    try:
        rt, detail = prepare_resume(run_dir, registry, transport, model=a.model,
                                    step_cap=a.step_cap, policy=policy,
                                    invocation=_invocation(a, workdir, extra, skills),
                                    skills=skills, progress_nudge_steps=a.progress_nudge,
                                    trust_project_plugins=a.trust_project_plugins or None,
                                    force=a.force)
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


def _skills(a: argparse.Namespace) -> int:
    """What the agent could load, and where each came from."""
    found = _discover_skills(a, Path(a.workdir).resolve())
    entries = found.list(a.filter)
    width = max((len(e["name"]) for e in entries), default=0)
    for entry in entries:
        skill = found.get(entry["name"])
        gated = "  [plugin: needs --trust-project-plugins]" if skill.plugin and found.is_project_skill(skill) else ""
        print(f"{entry['name']:<{width}}  {entry['description']}  ({skill.source}){gated}")
    where = f"{len(found.dirs)} director" + ("y" if len(found.dirs) == 1 else "ies")
    print(f"\n{len(entries)} skill(s) in {where}; "
          "none are in context until the agent calls skill_load", file=sys.stderr)
    for directory in found.dirs:
        print(f"  {directory}", file=sys.stderr)
    return 0


def _prompt(a: argparse.Namespace) -> int:
    """Print the system prompt a run would start with, or where it came from."""
    try:
        text, sources = _system_prompt(a, skills=bool(_discover_skills(a, None)))
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
        policy = _policy(a)          # no flags: the recording's own policy is used
    except ValueError as e:
        print(f"replay: {e}", file=sys.stderr)
        return USAGE_ERROR
    try:
        res = replay_run(source, registry, runs_dir=Path(a.runs_dir), run_id=a.new_run_id,
                         policy=policy)
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
    skills_help = ("directory of skills (<name>/SKILL.md or <name>.md), merged with "
                   "$HARNESS_SKILLS, the config directory and <workdir>/.harness/skills. "
                   "Repeatable; a later one wins a name clash.")

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
        sp.add_argument("--skills", action="append", metavar="DIR", help=skills_help)
        sp.add_argument("--deny-tool", action="append", metavar="NAME",
                        help="refuse this tool; the model sees the refusal as the result. Repeatable.")
        sp.add_argument("--deny-shell-pattern", action="append", metavar="REGEX",
                        help="refuse run_shell commands matching this regex. Repeatable.")

    def loop_args(sp) -> None:
        """The knobs of the loop itself. Shared by run and bench."""
        sp.add_argument("--step-cap", type=int, default=250)
        sp.add_argument("--no-todo-gate", action="store_true")
        sp.add_argument("--result-chars", type=int, default=2000)
        sp.add_argument("--context-chars", type=int, default=60000)
        sp.add_argument("--preview-chars", type=int, default=400)
        sp.add_argument("--skill-chars", type=int, default=12000,
                        help="refuse to load a skill larger than this (default: 12000)")
        sp.add_argument("--progress-nudge", type=int, default=12, metavar="N",
                        help="ask the model to update its plan after N steps with no change "
                             "to any todo (default: 12; 0 disables)")
        sp.add_argument("--trust-project-plugins", action="store_true",
                        help="let a skill from <workdir>/.harness/skills register its plugin, "
                             "which runs that project's own code (default: refuse)")

    r = sub.add_parser("run", help="run a task")
    r.add_argument("--task")
    r.add_argument("--task-file")
    model_args(r, model_required=True)
    r.add_argument("--run-id", default=None)
    loop_args(r)
    prompt_args(r)
    r.set_defaults(fn=_run)

    b = sub.add_parser("bench", help="run a file of tasks back to back and tabulate them")
    b.add_argument("tasks", metavar="TASKS.jsonl",
                   help='one task per line: {"task": "...", "workdir": "...", "run_id": "..."}')
    model_args(b, model_required=True)
    loop_args(b)
    prompt_args(b)
    b.set_defaults(fn=_bench)

    rs = sub.add_parser("resume",
                        help="continue an interrupted run (transport_error, step_cap, stalled, interrupted)")
    rs.add_argument("run_id")
    model_args(rs, model_required=False, recorded=True)
    rs.add_argument("--step-cap", type=int, default=None,
                    help="raise the cap for the rest of the run (default: the cap it ran under)")
    rs.add_argument("--progress-nudge", type=int, default=None, metavar="N",
                    help="steps without a todo change before the model is asked to update its "
                         "plan (default: what the run recorded; 0 disables)")
    rs.add_argument("--trust-project-plugins", action="store_true",
                    help="let a skill from <workdir>/.harness/skills register its plugin for the "
                         "rest of the run (default: what the run recorded)")
    rs.add_argument("--force", action="store_true",
                    help="take over a run whose lock is still held by a live process "
                         "(default: refuse, so two processes cannot append to one trajectory)")
    prompt_args(rs)      # accepted so the refusal can explain itself, never applied
    rs.set_defaults(fn=_resume)

    sk = sub.add_parser("skills", help="list discovered skills (what the agent can load)")
    sk.add_argument("--workdir", default=".", help="project whose .harness/skills is searched")
    sk.add_argument("--skills", action="append", metavar="DIR", help=skills_help)
    sk.add_argument("--filter", default=None, help="keyword filter, like skill_list")
    sk.set_defaults(fn=_skills)

    l = sub.add_parser("tools", help="list registered tools (what the agent can discover)")
    l.add_argument("--workdir", default=".", help="root the fs_* tools would be given")
    l.add_argument("--tools", action="append", metavar="MODULE|PATH", help=tools_help)
    l.add_argument("--filter", default=None, help="keyword filter, like toolbelt_list")
    l.set_defaults(fn=_tools)

    pr = sub.add_parser("prompt", help="print the system prompt a run would start with")
    prompt_args(pr)
    pr.add_argument("--skills", action="append", metavar="DIR", help=skills_help)
    pr.add_argument("--sources", action="store_true",
                    help="print where each layer of the prompt came from instead of the prompt")
    pr.set_defaults(fn=_prompt)

    p_replay = sub.add_parser("replay", help="re-drive a recorded run against today's tools")
    p_replay.add_argument("run_id")
    p_replay.add_argument("--workdir", default=".", help="root for fs_* and run_shell tools")
    p_replay.add_argument("--tools", action="append", metavar="MODULE|PATH", help=tools_help)
    p_replay.add_argument("--new-run-id", dest="new_run_id", default=None,
                          help="run id for the replay (default: <run_id>-replay)")
    p_replay.add_argument("--deny-tool", action="append", metavar="NAME",
                          help="refuse this tool, replacing the recorded policy. Repeatable.")
    p_replay.add_argument("--deny-shell-pattern", action="append", metavar="REGEX",
                          help="refuse run_shell commands matching this regex. Repeatable.")
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
