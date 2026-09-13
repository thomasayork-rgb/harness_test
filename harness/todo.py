"""Todo list bookkeeping.

The gate in the runtime asks one question: are there open todos? Everything
here exists to make that question answerable and to keep the list well formed.

An item also carries an optional ``notes`` string - the outcome of that step,
in the agent's own words - capped at NOTES_MAX characters. A note survives an
update that does not mention it, so the model can move an item to ``completed``
without retyping what it found; an explicit empty string clears it. Notes ride
along in every ``todo_snapshot`` and in the footer, which is what makes a
finished run auditable per todo rather than only per step.
"""
from __future__ import annotations

from typing import Any

VALID_STATUS = ("pending", "in_progress", "completed", "cancelled")
OPEN_STATUS = ("pending", "in_progress")
NOTES_MAX = 500


def validate_todos(items: Any) -> str | None:
    """Validate a whole todo payload before anything is persisted."""
    if not isinstance(items, list):
        return "todos must be a list"
    seen: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return f"todos[{i}] must be an object"
        tid = item.get("id")
        if not isinstance(tid, str) or not tid.strip():
            return f"todos[{i}].id must be a non-empty string"
        if tid in seen:
            return f"duplicate todo id: {tid}"
        seen.add(tid)
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            return f"todos[{i}].content must be a non-empty string"
        if item.get("status") not in VALID_STATUS:
            return f"todos[{i}].status must be one of {list(VALID_STATUS)}"
        notes = item.get("notes")
        if notes is not None and not isinstance(notes, str):
            return f"todos[{i}].notes must be a string"
        if isinstance(notes, str) and len(notes) > NOTES_MAX:
            return f"todos[{i}].notes is {len(notes)} chars; the limit is {NOTES_MAX}"
    return None


def _clean(item: dict) -> dict:
    """One stored item: the three required fields, plus a note if there is one."""
    out = {"id": item["id"], "content": item["content"], "status": item["status"]}
    notes = item.get("notes")
    if isinstance(notes, str) and notes.strip():
        out["notes"] = notes.strip()
    return out


def apply_update(current: list[dict], update: list[dict], merge: bool) -> list[dict]:
    """merge=True updates/adds by id and keeps order; merge=False replaces the list.

    On a merge, a note the update does not mention is carried over: closing an
    item should not cost the evidence recorded when it was opened. ``notes: ""``
    is how a note is deliberately cleared.
    """
    if not merge:
        return [_clean(t) for t in update]
    by_id = {t["id"]: dict(t) for t in current}
    order = [t["id"] for t in current]
    for raw in update:
        item = _clean(raw)
        previous = by_id.get(item["id"])
        if previous is None:
            order.append(item["id"])
        elif "notes" not in item and "notes" not in raw and previous.get("notes"):
            item["notes"] = previous["notes"]
        by_id[item["id"]] = item
    return [by_id[i] for i in order]


def signature(todos: list[dict]) -> tuple:
    """What "the plan moved" means: which items exist, their status, their notes.

    The progress nudge watches this and nothing else - rewording the content of
    an item is not progress, and neither is calling todo_write with the list it
    already had.
    """
    return tuple((t.get("id"), t.get("status"), t.get("notes", "")) for t in todos)


def open_ids(todos: list[dict]) -> list[str]:
    return [t["id"] for t in todos if t.get("status") in OPEN_STATUS]
