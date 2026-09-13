"""Loading tool modules named with ``--tools``.

A plugin is an ordinary Python module, given either as a dotted name on
``sys.path`` or as a path to a ``.py`` file. It must export one of:

  ``registry``            a ``ToolRegistry`` whose tools are merged in
  ``register(registry)``  called with the run's registry; a second parameter
                          receives the resolved workdir

Nothing else is imported for you and nothing is auto-discovered: what the
agent can call is exactly what the command line asked for.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import os
import re
import sys
from pathlib import Path

from .registry import ToolRegistry

_SAFE = re.compile(r"[^A-Za-z0-9_]+")


class PluginError(Exception):
    """Bad ``--tools`` argument. The CLI turns this into a usage error."""


def _is_path(spec: str) -> bool:
    return spec.endswith(".py") or "/" in spec or os.sep in spec


def load_module(spec: str):
    """Import a dotted module name or a ``.py`` file path."""
    if not _is_path(spec):
        try:
            return importlib.import_module(spec)
        except Exception as e:  # noqa: BLE001 - any import failure is the user's answer
            raise PluginError(f"{spec}: import failed: {type(e).__name__}: {e}") from e

    path = Path(spec).expanduser().resolve()
    if not path.is_file():
        raise PluginError(f"{spec}: no such file")
    # Two plugin files may share a stem; keep their module names distinct.
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    name = f"harness_plugin_{_SAFE.sub('_', path.stem)}_{digest}"
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise PluginError(f"{spec}: not importable as a Python module")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001
        del sys.modules[name]
        raise PluginError(f"{spec}: import failed: {type(e).__name__}: {e}") from e
    return module


def _accepts_workdir(fn) -> bool:
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    return len([p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 2


def load_tools(registry: ToolRegistry, spec: str, workdir: Path) -> list[str]:
    """Load one ``--tools`` spec into ``registry``. Returns the names it added."""
    module = load_module(spec)
    exported = getattr(module, "registry", None)
    register = getattr(module, "register", None)
    before = set(registry.names())
    try:
        if isinstance(exported, ToolRegistry):
            registry.merge(exported)
        elif callable(register):
            if _accepts_workdir(register):
                register(registry, Path(workdir))
            else:
                register(registry)
        else:
            raise PluginError(
                f"{spec}: exports neither a `registry` (ToolRegistry) nor a `register(registry)` function")
    except PluginError:
        raise
    except Exception as e:  # noqa: BLE001 - duplicate names, bad schemas, anything
        raise PluginError(f"{spec}: {type(e).__name__}: {e}") from e
    return sorted(set(registry.names()) - before)


def load_all(registry: ToolRegistry, specs: list[str] | None, workdir: Path) -> list[str]:
    added: list[str] = []
    for spec in specs or []:
        added.extend(load_tools(registry, spec, workdir))
    return added
