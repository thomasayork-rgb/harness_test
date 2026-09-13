"""Todo list bookkeeping.

The gate in the runtime asks one question: are there open todos? Everything
here exists to make that question answerable and to keep the list well formed.
"""
from __future__ import annotations

from typing import Any

VALID_STATUS = ("pending", "in_progress", "completed", "cancelled")
OPEN_STATUS = ("pending", "in_progress")


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
    return None


def apply_update(current: list[dict], update: list[dict], merge: bool) -> list[dict]:
    """merge=True updates/adds by id and keeps order; merge=False replaces the list."""
    clean = [{"id": t["id"], "content": t["content"], "status": t["status"]} for t in update]
    if not merge:
        return clean
    by_id = {t["id"]: dict(t) for t in current}
    order = [t["id"] for t in current]
    for t in clean:
        if t["id"] in by_id:
            by_id[t["id"]] = t
        else:
            by_id[t["id"]] = t
            order.append(t["id"])
    return [by_id[i] for i in order]


def open_ids(todos: list[dict]) -> list[str]:
    return [t["id"] for t in todos if t.get("status") in OPEN_STATUS]
