"""Tool-call policy: a veto, run before every dispatch.

``AgentRuntime`` takes an optional ``policy``: any callable

    policy(name: str, args: Any) -> str | None

called before each tool call, including the meta-tools. Return ``None`` to let
the call through, or a sentence saying why not. A denial never raises: it
becomes the tool's result, so the model reads it and can adapt, and the step is
recorded in the trajectory with ``kind: "denied"`` - distinct from ``error``,
because the tool did not fail, it did not run.

``ToolPolicy`` is the concrete one the CLI exposes::

    ToolPolicy(deny_tools=["run_shell"], deny_shell_patterns=[r"rm\\s+-rf", r"\\bcurl\\b"])

Names are matched exactly; shell patterns are Python regexes searched against
the ``command`` argument of the shell tools. ``args`` may be whatever the model
produced - a dict, or a raw string when it emitted invalid JSON - so a policy
must not assume it is a dict.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

DENIED_TOOL = "tool '{name}' is denied by policy"
DENIED_COMMAND = "command denied by policy: it matches {pattern}"


class ToolPolicy:
    """Deny tools by name and shell commands by regex."""

    def __init__(
        self,
        deny_tools: Iterable[str] = (),
        deny_shell_patterns: Iterable[str] = (),
        shell_tools: Iterable[str] = ("run_shell",),
        command_arg: str = "command",
    ) -> None:
        self.deny_tools = sorted(set(deny_tools))
        self.shell_tools = set(shell_tools)
        self.command_arg = command_arg
        self.patterns: list[re.Pattern] = []
        for raw in deny_shell_patterns:
            try:
                self.patterns.append(re.compile(raw))
            except re.error as e:
                raise ValueError(f"bad shell pattern {raw!r}: {e}") from None

    def __bool__(self) -> bool:
        return bool(self.deny_tools or self.patterns)

    def __call__(self, name: str, args: Any) -> str | None:
        if name in self.deny_tools:
            return DENIED_TOOL.format(name=name)
        if name in self.shell_tools and self.patterns:
            command = args.get(self.command_arg) if isinstance(args, dict) else args
            if isinstance(command, str):
                for rx in self.patterns:
                    if rx.search(command):
                        return DENIED_COMMAND.format(pattern=repr(rx.pattern))
        return None

    def describe(self) -> dict:
        """What the trajectory header records, so a trace says what was in force."""
        return {"deny_tools": list(self.deny_tools),
                "deny_shell_patterns": [p.pattern for p in self.patterns],
                "shell_tools": sorted(self.shell_tools)}
