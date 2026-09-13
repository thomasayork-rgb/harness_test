"""Trajectory logging.

One JSONL file per run. Record types:

  header  - run_id, model, harness_version, step_cap, task, ts
  step    - one per tool call (discovery and todo calls included) or per
            text-only turn; carries reasoning, tool, args, result preview,
            token usage, todo snapshot and an artifact reference
  footer  - terminal status, step count, final answer

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

    def header(self, *, run_id: str, model: str, step_cap: int, task: str, config: dict) -> None:
        self._write({
            "type": "header",
            "run_id": run_id,
            "ts": _now(),
            "model": model,
            "harness_version": HARNESS_VERSION,
            "step_cap": step_cap,
            "task": task,
            "config": config,
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
            "result_preview": result[: self.preview_chars],
            "result_bytes": len(result.encode("utf-8")),
            "artifact": artifact,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "todo_snapshot": todo_snapshot,
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


def summarize(records: list[dict]) -> dict:
    """Run stats from the JSONL: status, step kinds, per-tool counts, tokens, elapsed."""
    header = next((r for r in records if r["type"] == "header"), None) or {}
    footer = next((r for r in records if r["type"] == "footer"), None) or {}
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

    start, end = _epoch(header.get("ts")), _epoch(footer.get("ts"))
    return {
        "run_id": header.get("run_id") or footer.get("run_id"),
        "model": header.get("model"),
        "harness_version": header.get("harness_version"),
        "task": header.get("task"),
        "step_cap": header.get("step_cap"),
        "status": footer.get("status", "incomplete"),
        "detail": footer.get("detail"),
        "steps": footer.get("steps", len(steps)),
        "final": footer.get("final"),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
        "tools": dict(sorted(tools.items(), key=lambda kv: (-kv[1], kv[0]))),
        "errors": kinds.get("error", 0),
        "rejected_finals": kinds.get("final_rejected", 0),
        "text_only": kinds.get("text_only", 0),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "elapsed_ms": elapsed_ms,
        "wall_s": None if start is None or end is None else round(end - start, 1),
    }


def format_summary(records: list[dict]) -> str:
    """One screen of run stats. ``harness trace <run_id> --summary``."""
    s = summarize(records)
    lines = [f"run {s['run_id']}  model={s['model']}  harness {s['harness_version']}  cap={s['step_cap']}"]
    if s["task"]:
        lines.append(f"task: {str(s['task'])[:200]}")
    lines.append(f"status: {s['status']}  steps: {s['steps']}" + (f"  ({s['detail']})" if s["detail"] else ""))
    wall = "?" if s["wall_s"] is None else f"{s['wall_s']:g}"
    lines.append(f"elapsed: {s['elapsed_ms'] / 1000:.1f} s in steps, {wall} s wall")
    lines.append(f"tokens: {s['tokens_in']} in / {s['tokens_out']} out")
    lines.append(f"errors: {s['errors']}  rejected finals: {s['rejected_finals']}  text-only turns: {s['text_only']}")
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


def format_trace(records: list[dict], run_dir: Path, step: int | None = None) -> str:
    """Readable trace. With ``step`` set, print that step in full (artifact included)."""
    lines: list[str] = []
    header = next((r for r in records if r["type"] == "header"), None)
    footer = next((r for r in records if r["type"] == "footer"), None)
    steps = [r for r in records if r["type"] == "step"]

    if step is not None:
        rec = next((r for r in steps if r["step"] == step), None)
        if rec is None:
            return f"no step {step} in trajectory"
        lines.append(f"step {rec['step']}  [{rec['kind']}]  {rec['tool'] or '(text only)'}  {rec['elapsed_ms']} ms")
        lines.append(f"tokens in/out: {rec['tokens_in']} / {rec['tokens_out']}")
        lines.append("")
        lines.append("reasoning:")
        lines.append(rec["reasoning"] or "(none)")
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

    if header:
        lines.append(f"run {header['run_id']}  model={header['model']}  cap={header['step_cap']}  {header['ts']}")
        lines.append(f"task: {header['task'][:200]}")
        lines.append("")
    for rec in steps:
        tool = rec["tool"] or "(text only)"
        reasoning = (rec["reasoning"] or "").strip().replace("\n", " ")
        if len(reasoning) > 160:
            reasoning = reasoning[:157] + "..."
        lines.append(f"{rec['step']:>4}  {rec['kind']:<15} {tool:<22} {rec['elapsed_ms']:>6} ms  {reasoning}")
    if footer:
        lines.append("")
        lines.append(f"status: {footer['status']}  steps: {footer['steps']}" + (f"  ({footer['detail']})" if footer.get("detail") else ""))
        if footer.get("final"):
            lines.append(f"final [{footer['final'].get('status')}]: {str(footer['final'].get('content'))[:500]}")
    return "\n".join(lines)
