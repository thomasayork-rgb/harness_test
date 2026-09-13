"""Trajectory logging.

One JSONL file per run. Record types:

  header  - run_id, model, harness_version, step_cap, task, ts, and
            prompt_sources: which prompt files the system message was built
            from (see harness.prompts)
  step    - one per tool call (discovery and todo calls included) or per
            text-only turn; carries reasoning, tool, args, result preview,
            token usage, todo snapshot, an artifact reference, and
            call_index: the position of this call in its model turn, so turn
            boundaries survive the round trip (see harness.replay)
  footer  - terminal status, step count, final answer

The header (and each resume record) also carries ``policy`` - what the
tool-call policy in force denied, or null - and ``invocation``: how the segment
was launched (endpoint, provider, workdir, tool modules, request options), so a
resume can default to it. Never the API key.
  resume  - a seam between two segments of the same run: what the previous
            segment ended with, and the model and config the next one starts
            with (see harness.resume)

A resumed run appends to the same file, so the shape is

    header  step*  footer  [resume  step*  footer]*

with one footer per segment and the step counter running straight through.
The last footer is the run's current answer; the earlier ones are history.

Full tool results are written to ``artifacts/`` beside the JSONL so
``trace --step N`` can show the whole thing without bloating the log.
"""
from __future__ import annotations

import calendar
import json
import re
import time
from pathlib import Path
from typing import Any

HARNESS_VERSION = "0.1.0"

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


class TrajectoryWriter:
    def __init__(self, run_dir: Path, preview_chars: int = 400) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "trajectory.jsonl"
        self.artifacts = self.run_dir / "artifacts"
        self.preview_chars = preview_chars
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(exist_ok=True)

    def _write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def header(self, *, run_id: str, model: str, step_cap: int, task: str, config: dict,
               policy: Any = None, prompt_sources: list[dict] | None = None,
               invocation: dict | None = None) -> None:
        self._write({
            "type": "header",
            "run_id": run_id,
            "ts": _now(),
            "model": model,
            "harness_version": HARNESS_VERSION,
            "step_cap": step_cap,
            "task": task,
            "config": config,
            "policy": policy,
            "prompt_sources": prompt_sources or [],
            "invocation": invocation or {},
        })

    def write_artifact(self, step: int, tool: str | None, text: str) -> str:
        name = f"step_{step:04d}_{_SAFE.sub('_', tool or 'text')}.txt"
        (self.artifacts / name).write_text(text, encoding="utf-8")
        return f"artifacts/{name}"

    def step(
        self,
        *,
        run_id: str,
        step: int,
        elapsed_ms: int,
        reasoning: str,
        tool: str | None,
        args: Any,
        result: str,
        artifact: str | None,
        tokens_in: int | None,
        tokens_out: int | None,
        todo_snapshot: list[dict],
        kind: str,
        call_index: int = 0,
    ) -> None:
        self._write({
            "type": "step",
            "run_id": run_id,
            "step": step,
            "ts": _now(),
            "elapsed_ms": elapsed_ms,
            "reasoning": reasoning,
            "tool": tool,
            "args": args,
            "kind": kind,
            "call_index": call_index,
            "result_preview": result[: self.preview_chars],
            "result_bytes": len(result.encode("utf-8")),
            "artifact": artifact,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "todo_snapshot": todo_snapshot,
        })

    def resume(self, *, run_id: str, model: str, from_status: str, from_step: int,
               from_detail: str | None, step_cap: int, config: dict, note: str,
               policy: Any = None, invocation: dict | None = None) -> None:
        self._write({
            "type": "resume",
            "run_id": run_id,
            "ts": _now(),
            "harness_version": HARNESS_VERSION,
            "model": model,
            "from_status": from_status,
            "from_step": from_step,
            "from_detail": from_detail,
            "step_cap": step_cap,
            "config": config,
            "policy": policy,
            "invocation": invocation or {},
            "note": note,
        })

    def footer(self, *, run_id: str, status: str, steps: int, final: dict | None, detail: str | None = None) -> None:
        self._write({
            "type": "footer",
            "run_id": run_id,
            "ts": _now(),
            "status": status,
            "steps": steps,
            "final": final,
            "detail": detail,
        })


def read_trajectory(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "trajectory.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no trajectory at {path}")
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _epoch(ts: str | None) -> float | None:
    try:
        return calendar.timegm(time.strptime(ts or "", "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, TypeError):
        return None


def format_prompt_sources(sources: list[dict] | None) -> str:
    """One line naming every layer the system prompt was built from."""
    if not sources:
        return "(not recorded)"
    return " + ".join(f"{s.get('source')} [{s.get('role')}] {s.get('chars')} chars" for s in sources)


def last(records: list[dict], type_: str) -> dict:
    """The last record of a type, or {}. A resumed run has one footer per
    segment; the last one is the run's current answer."""
    for rec in reversed(records):
        if rec.get("type") == type_:
            return rec
    return {}


def setting(records: list[dict], key: str, default: Any = None) -> Any:
    """The value a trajectory ended under for a header/resume field. A resumed
    run records its own settings at the seam, so the last segment wins."""
    value = default
    for rec in records:
        if rec.get("type") in ("header", "resume") and key in rec:
            value = rec[key]
    return value


def _wall_seconds(records: list[dict]) -> float | None:
    """Seconds inside segments: header/resume to the footer that closed it.

    Summing per segment rather than first-to-last keeps the time a run sat
    waiting to be resumed out of the number.
    """
    total = 0.0
    start: float | None = None
    seen = False
    for rec in records:
        if rec.get("type") in ("header", "resume"):
            start = _epoch(rec.get("ts"))
        elif rec.get("type") == "footer" and start is not None:
            end = _epoch(rec.get("ts"))
            if end is None:
                return None
            total += end - start
            start, seen = None, True
    return round(total, 1) if seen else None


def summarize(records: list[dict]) -> dict:
    """Run stats from the JSONL: status, step kinds, per-tool counts, tokens, elapsed.

    Segment-aware: a resumed run is summarised as one run, with the totals
    covering every segment and the status taken from the last footer.
    """
    header = next((r for r in records if r["type"] == "header"), None) or {}
    footer = last(records, "footer")
    resumes = [r for r in records if r.get("type") == "resume"]
    steps = [r for r in records if r["type"] == "step"]

    kinds: dict[str, int] = {}
    tools: dict[str, int] = {}
    tokens_in = tokens_out = elapsed_ms = 0
    for r in steps:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        name = r["tool"] or "(text only)"
        tools[name] = tools.get(name, 0) + 1
        tokens_in += r.get("tokens_in") or 0
        tokens_out += r.get("tokens_out") or 0
        elapsed_ms += r.get("elapsed_ms") or 0

    return {
        "run_id": header.get("run_id") or footer.get("run_id"),
        "model": (resumes[-1] if resumes else header).get("model"),
        "harness_version": header.get("harness_version"),
        "task": header.get("task"),
        "step_cap": (resumes[-1] if resumes else header).get("step_cap"),
        "policy": (resumes[-1] if resumes else header).get("policy"),
        "prompt_sources": header.get("prompt_sources") or [],
        "segments": 1 + len(resumes),
        "resumed_from": [r.get("from_status") for r in resumes],
        "status": footer.get("status", "incomplete"),
        "detail": footer.get("detail"),
        "steps": footer.get("steps", len(steps)),
        "final": footer.get("final"),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
        "tools": dict(sorted(tools.items(), key=lambda kv: (-kv[1], kv[0]))),
        "errors": kinds.get("error", 0),
        "denied": kinds.get("denied", 0),
        "rejected_finals": kinds.get("final_rejected", 0),
        "text_only": kinds.get("text_only", 0),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "elapsed_ms": elapsed_ms,
        "wall_s": _wall_seconds(records),
    }


def format_summary(records: list[dict]) -> str:
    """One screen of run stats. ``harness trace <run_id> --summary``."""
    s = summarize(records)
    lines = [f"run {s['run_id']}  model={s['model']}  harness {s['harness_version']}  cap={s['step_cap']}"]
    if s["task"]:
        lines.append(f"task: {str(s['task'])[:200]}")
    if s["segments"] > 1:
        lines.append(f"segments: {s['segments']}  resumed from: {', '.join(s['resumed_from'])}")
    lines.append(f"status: {s['status']}  steps: {s['steps']}" + (f"  ({s['detail']})" if s["detail"] else ""))
    wall = "?" if s["wall_s"] is None else f"{s['wall_s']:g}"
    lines.append(f"elapsed: {s['elapsed_ms'] / 1000:.1f} s in steps, {wall} s wall")
    lines.append(f"tokens: {s['tokens_in']} in / {s['tokens_out']} out")
    lines.append(f"errors: {s['errors']}  rejected finals: {s['rejected_finals']}  "
                 f"text-only turns: {s['text_only']}  denied: {s['denied']}")
    if s["policy"]:
        lines.append(f"policy: {json.dumps(s['policy'], sort_keys=True)}")
    lines.append("prompt: " + format_prompt_sources(s["prompt_sources"]))
    lines.append("kinds: " + (", ".join(f"{k} {n}" for k, n in s["kinds"].items()) or "(none)"))
    lines.append("tools:")
    width = max((len(t) for t in s["tools"]), default=0)
    for name, n in s["tools"].items():
        lines.append(f"  {name:<{width}}  {n}")
    if not s["tools"]:
        lines.append("  (none)")
    if s["final"]:
        lines.append(f"final [{s['final'].get('status')}]: {str(s['final'].get('content'))[:500]}")
    return "\n".join(lines)


def turn_start(steps: list[dict], rec: dict) -> int | None:
    """The step that opened this step's model turn, when that is not this step.

    One turn can issue several tool calls, and only the first carries the
    turn's reasoning and its token usage - so a later call has nothing of its
    own to show, and a reader needs pointing at where it went."""
    index = rec.get("call_index") or 0
    if not index:
        return None
    start = rec["step"] - index
    if any(r["step"] == start and not (r.get("call_index") or 0) for r in steps):
        return start
    return None


def format_trace(records: list[dict], run_dir: Path, step: int | None = None) -> str:
    """Readable trace. With ``step`` set, print that step in full (artifact included)."""
    lines: list[str] = []
    steps = [r for r in records if r["type"] == "step"]

    if step is not None:
        rec = next((r for r in steps if r["step"] == step), None)
        if rec is None:
            return f"no step {step} in trajectory"
        shared = turn_start(steps, rec)
        lines.append(f"step {rec['step']}  [{rec['kind']}]  {rec['tool'] or '(text only)'}  {rec['elapsed_ms']} ms")
        if shared is not None:
            lines.append(f"turn: call {rec['call_index'] + 1} of the turn that started at step {shared}; "
                         f"reasoning and token usage are recorded on step {shared}")
            lines.append(f"tokens in/out: recorded on step {shared}")
        else:
            lines.append(f"tokens in/out: {rec['tokens_in']} / {rec['tokens_out']}")
        lines.append("")
        lines.append("reasoning:")
        lines.append(rec["reasoning"] or (f"(recorded on step {shared})" if shared is not None else "(none)"))
        lines.append("")
        lines.append("args:")
        lines.append(json.dumps(rec["args"], indent=2, ensure_ascii=False))
        lines.append("")
        lines.append("result:")
        if rec.get("artifact"):
            art = Path(run_dir) / rec["artifact"]
            lines.append(art.read_text(encoding="utf-8") if art.exists() else rec["result_preview"])
        else:
            lines.append(rec["result_preview"])
        lines.append("")
        lines.append("todos:")
        for t in rec["todo_snapshot"]:
            lines.append(f"  [{t['status']}] {t['id']}: {t['content']}")
        return "\n".join(lines)

    # In file order, so the seams of a resumed run appear where they happened.
    for rec in records:
        kind = rec.get("type")
        if kind == "header":
            lines.append(f"run {rec['run_id']}  model={rec['model']}  cap={rec['step_cap']}  {rec['ts']}")
            lines.append(f"task: {rec['task'][:200]}")
            lines.append("prompt: " + format_prompt_sources(rec.get("prompt_sources")))
            lines.append("")
        elif kind == "step":
            tool = rec["tool"] or "(text only)"
            reasoning = (rec["reasoning"] or "").strip().replace("\n", " ")
            if len(reasoning) > 160:
                reasoning = reasoning[:157] + "..."
            if not reasoning:
                shared = turn_start(steps, rec)
                if shared is not None:
                    reasoning = f"(continues the turn of step {shared})"
            lines.append(f"{rec['step']:>4}  {rec['kind']:<15} {tool:<22} {rec['elapsed_ms']:>6} ms  {reasoning}")
        elif kind == "footer":
            lines.append("")
            lines.append(f"status: {rec['status']}  steps: {rec['steps']}"
                         + (f"  ({rec['detail']})" if rec.get("detail") else ""))
            if rec.get("final"):
                lines.append(f"final [{rec['final'].get('status')}]: {str(rec['final'].get('content'))[:500]}")
        elif kind == "resume":
            lines.append("")
            lines.append(f"-- resumed at step {rec['from_step']} after {rec['from_status']}"
                         f"  model={rec['model']}  cap={rec['step_cap']}  {rec['ts']}")
            lines.append("")
    return "\n".join(lines)
