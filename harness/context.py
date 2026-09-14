"""Context budget.

Two knobs:

  result_chars  - per tool result, in context. Anything longer is cut and
                  replaced with a pointer to the full artifact.
  total_chars   - whole conversation. When exceeded, the oldest evictable tool
                  results are replaced with a one-line placeholder until the
                  conversation fits.

Never evicted: system prompt, the task, assistant turns (reasoning lives
there), tool results flagged ``_protected`` (the current todo list, the text of
a loaded skill), and the results of the most recent turn - the model has to be
able to read what it just asked for, or its only move is to ask again. That
means the budget is a target, not a ceiling: with a large system prompt and a
small budget there may be nothing left to evict.

Protection is for the *current* state, not for every version of it. A result
that carries ``_supersedes`` claims a slot - "the todo list", "the text of
skill X" - and only the newest occupant of a slot keeps its text: ``supersede``
demotes the older ones to a line naming their artifact. Without that, forty
todo_write calls leave forty copies of the list in context, every one of them
protected, and the budget cannot be met however much else is evicted.

``protected_size`` counts what protection holds, separately from what eviction
can still reach, so the loop can say so when the two no longer add up.

Messages carry internal keys prefixed with ``_``; the transport strips them.
"""
from __future__ import annotations

EVICTED = "[result evicted from context to stay within budget; full result at {artifact}]"
TRUNCATED = "\n... [truncated at {n} chars; full result at {artifact}]"
SUPERSEDED = "[superseded by a later {label}; this one is at {artifact}]"


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

    def supersede(self, messages: list[dict]) -> int:
        """Demote every result a newer one has replaced. Returns the count.

        A result carrying ``_supersedes`` occupies a slot: the todo list, the
        text of one skill. The newest occupant is the state; the earlier ones
        describe a state that has moved on, so they lose their protection and
        become a line pointing at the artifact that still holds them in full.
        Already evicted or already superseded results are left alone.
        """
        demoted = 0
        seen: set[str] = set()
        for m in reversed(messages):
            slot = m.get("_supersedes")
            if not slot or m.get("role") != "tool":
                continue
            if slot not in seen:
                seen.add(slot)          # the newest occupant keeps its text
                continue
            if m.get("_superseded") or m.get("_evicted"):
                continue
            m["content"] = SUPERSEDED.format(label=m.get("_label") or slot,
                                             artifact=m.get("_artifact") or "artifacts/")
            m["_protected"] = False
            m["_superseded"] = True
            demoted += 1
        return demoted

    def protected_size(self, messages: list[dict]) -> int:
        """Chars in tool results eviction may not touch.

        Counted apart from the rest because nothing the budget does can reduce
        it: when this alone is over ``total_chars``, only the model can free
        context (unload a skill, keep the plan short).
        """
        return sum(len(m.get("content") or "") for m in messages
                   if m.get("role") == "tool" and m.get("_protected"))

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
