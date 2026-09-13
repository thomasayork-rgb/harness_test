"""Context budget.

Two knobs:

  result_chars  - per tool result, in context. Anything longer is cut and
                  replaced with a pointer to the full artifact.
  total_chars   - whole conversation. When exceeded, the oldest evictable tool
                  results are replaced with a one-line placeholder until the
                  conversation fits.

Never evicted: system prompt, the task, assistant turns (reasoning lives
there), tool results flagged ``_protected`` (todo_write results, so the model
always sees its current list), and the results of the most recent turn - the
model has to be able to read what it just asked for, or its only move is to
ask again. That means the budget is a target, not a ceiling: with a large
system prompt and a small budget there may be nothing left to evict.

Messages carry internal keys prefixed with ``_``; the transport strips them.
"""
from __future__ import annotations

EVICTED = "[result evicted from context to stay within budget; full result at {artifact}]"
TRUNCATED = "\n... [truncated at {n} chars; full result at {artifact}]"


def _size(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            total += len(tc.get("function", {}).get("arguments") or "")
    return total


def _last_turn_start(messages: list[dict]) -> int:
    """Index of the last assistant message: everything after it is the results
    of the turn in flight."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return len(messages)


class ContextBudget:
    def __init__(self, result_chars: int = 2000, total_chars: int = 60000) -> None:
        self.result_chars = result_chars
        self.total_chars = total_chars

    def truncate_result(self, text: str, artifact: str | None) -> str:
        if len(text) <= self.result_chars:
            return text
        return text[: self.result_chars] + TRUNCATED.format(n=self.result_chars, artifact=artifact or "artifacts/")

    def enforce(self, messages: list[dict]) -> int:
        """Evict oldest unprotected tool results in place. Returns count evicted.

        The current turn's results are kept whatever the budget says: evicting
        the answer to the call the model just made leaves it nothing to act on.
        """
        evicted = 0
        if _size(messages) <= self.total_chars:
            return 0
        current_turn = _last_turn_start(messages)
        for index, m in enumerate(messages):
            if index >= current_turn:
                break
            if m.get("role") != "tool" or m.get("_protected") or m.get("_evicted"):
                continue
            m["content"] = EVICTED.format(artifact=m.get("_artifact") or "artifacts/")
            m["_evicted"] = True
            evicted += 1
            if _size(messages) <= self.total_chars:
                break
        return evicted

    def size(self, messages: list[dict]) -> int:
        return _size(messages)
