"""Context budget.

Two knobs:

  result_chars  - per tool result, in context. Anything longer is cut and
                  replaced with a pointer to the full artifact.
  total_chars   - whole conversation. When exceeded, the oldest evictable tool
                  results are replaced with a one-line placeholder until the
                  conversation fits.

Never evicted: system prompt, the task, assistant turns (reasoning lives
there), and tool results flagged ``_protected`` (todo_write results, so the
model always sees its current list).

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


class ContextBudget:
    def __init__(self, result_chars: int = 2000, total_chars: int = 60000) -> None:
        self.result_chars = result_chars
        self.total_chars = total_chars

    def truncate_result(self, text: str, artifact: str | None) -> str:
        if len(text) <= self.result_chars:
            return text
        return text[: self.result_chars] + TRUNCATED.format(n=self.result_chars, artifact=artifact or "artifacts/")

    def enforce(self, messages: list[dict]) -> int:
        """Evict oldest unprotected tool results in place. Returns count evicted."""
        evicted = 0
        if _size(messages) <= self.total_chars:
            return 0
        for m in messages:
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
