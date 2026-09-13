"""Tool registry.

The registry holds every tool the harness knows about. Nothing in it is sent to
the model until the agent activates it with ``toolbelt_add``. The runtime keeps
the list of active *names* in run state; schemas are looked up here at request
time, never duplicated into state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

ToolFn = Callable[..., Any]

_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON-schema object describing the arguments
    fn: ToolFn

    def schema(self) -> dict:
        """Full OpenAI-style tool schema, sent only once the tool is active."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def brief(self) -> dict:
        """One-line listing entry. This is all the model sees before activation."""
        first_line = self.description.strip().splitlines()[0] if self.description else ""
        return {"name": self.name, "description": first_line[:120]}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        if spec.parameters.get("type") != "object":
            raise ValueError(f"tool {spec.name}: parameters schema must be an object")
        self._tools[spec.name] = spec
        return spec

    def tool(self, name: str, description: str, parameters: dict | None = None):
        """Decorator form: ``@registry.tool("fs_read", "Read a file", {...})``."""
        params = parameters or {"type": "object", "properties": {}, "required": []}

        def wrap(fn: ToolFn) -> ToolFn:
            self.register(ToolSpec(name=name, description=description, parameters=params, fn=fn))
            return fn

        return wrap

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list(self, filter: str | None = None) -> list[dict]:
        needle = (filter or "").strip().lower()
        out = []
        for name in self.names():
            spec = self._tools[name]
            if needle and needle not in name.lower() and needle not in spec.description.lower():
                continue
            out.append(spec.brief())
        return out

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def validate_args(parameters: dict, args: Any) -> str | None:
    """Cheap top-level validation. Returns an error string, or None if OK.

    Deliberately shallow: required keys, unknown keys when the schema forbids
    them, and top-level types. Anything deeper is the tool's own problem and
    surfaces as a tool error result, not a crash.
    """
    if not isinstance(args, dict):
        return f"arguments must be a JSON object, got {type(args).__name__}"
    props: dict = parameters.get("properties", {}) or {}
    required = parameters.get("required", []) or []
    missing = [k for k in required if k not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    if parameters.get("additionalProperties") is False:
        unknown = [k for k in args if k not in props]
        if unknown:
            return f"unknown argument(s): {', '.join(unknown)}"
    for key, value in args.items():
        spec = props.get(key)
        if not spec or "type" not in spec:
            continue
        expected = spec["type"]
        types = expected if isinstance(expected, list) else [expected]
        ok = False
        for t in types:
            if t == "null" and value is None:
                ok = True
            elif t in _JSON_TYPES and isinstance(value, _JSON_TYPES[t]):
                # bool is a subclass of int; don't let True pass as integer
                if t in ("integer", "number") and isinstance(value, bool):
                    continue
                ok = True
        if not ok:
            return f"argument '{key}' must be {expected}, got {type(value).__name__}"
    return None
