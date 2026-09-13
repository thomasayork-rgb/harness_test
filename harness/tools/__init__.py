from pathlib import Path

from ..registry import ToolRegistry
from .basic import register_basic_tools
from .edit import register_edit_tools
from .scratch import register_scratch_tools
from .search import register_search_tools


def register_default_tools(registry: ToolRegistry, workdir: Path) -> None:
    """Every workdir-rooted built-in tool. This is what the CLI uses.

    The scratch pad is not here: it is rooted in the run directory, which only
    exists once there is a run, so the CLI adds it after the runtime is built.
    """
    register_basic_tools(registry, workdir)
    register_search_tools(registry, workdir)
    register_edit_tools(registry, workdir)


__all__ = ["register_basic_tools", "register_search_tools", "register_edit_tools",
           "register_scratch_tools", "register_default_tools"]
