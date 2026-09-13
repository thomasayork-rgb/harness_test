from pathlib import Path

from ..registry import ToolRegistry
from .basic import register_basic_tools
from .edit import register_edit_tools
from .search import register_search_tools


def register_default_tools(registry: ToolRegistry, workdir: Path) -> None:
    """Every built-in tool, rooted to ``workdir``. This is what the CLI uses."""
    register_basic_tools(registry, workdir)
    register_search_tools(registry, workdir)
    register_edit_tools(registry, workdir)


__all__ = ["register_basic_tools", "register_search_tools", "register_edit_tools",
           "register_default_tools"]
