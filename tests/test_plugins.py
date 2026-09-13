"""--tools plugin loading, and the `harness tools` listing."""
import json

import pytest

from harness.cli import main
from harness.mockserver import MockOpenAIServer
from harness.plugins import PluginError, load_all, load_tools
from harness.registry import ToolRegistry
from harness.tools import register_default_tools
from harness.trajectory import read_trajectory
from harness.transport import call

REGISTRY_PLUGIN = '''
from harness import ToolRegistry

registry = ToolRegistry()


@registry.tool("coin_flip", "Flip a deterministic coin for a seed.\\nDetail line ignored in listings.",
               {"type": "object", "properties": {"seed": {"type": "integer"}}, "required": ["seed"]})
def coin_flip(seed: int) -> dict:
    return {"seed": seed, "side": "heads" if seed % 2 == 0 else "tails"}
'''

REGISTER_PLUGIN = '''
def register(registry):
    @registry.tool("greet", "Greet someone by name.",
                   {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
    def greet(name: str) -> str:
        return "hello " + name
'''

WORKDIR_PLUGIN = '''
from pathlib import Path


def register(registry, workdir):
    @registry.tool("workdir_name", "Report the workdir the plugin was given.",
                   {"type": "object", "properties": {}, "required": []})
    def workdir_name() -> dict:
        return {"name": Path(workdir).name}
'''

CLASHING_PLUGIN = '''
def register(registry):
    @registry.tool("fs_read", "Shadow a built-in tool.", {"type": "object", "properties": {}, "required": []})
    def fs_read() -> str:
        return "nope"
'''

BROKEN_PLUGIN = "raise RuntimeError('plugin blew up at import')\n"
EMPTY_PLUGIN = "VALUE = 1\n"


def write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_registry_export_and_register_function(tmp_path):
    r = ToolRegistry()
    register_default_tools(r, tmp_path)
    added = load_all(r, [write(tmp_path, "p_registry.py", REGISTRY_PLUGIN),
                         write(tmp_path, "p_register.py", REGISTER_PLUGIN)], tmp_path)
    assert added == ["coin_flip", "greet"]
    assert r.get("coin_flip").fn(seed=4) == {"seed": 4, "side": "heads"}
    assert r.get("greet").fn(name="world") == "hello world"
    # the one-line rule still applies to plugin descriptions
    assert {"name": "coin_flip", "description": "Flip a deterministic coin for a seed."} in r.list("coin")


def test_register_may_take_the_workdir(tmp_path):
    work = tmp_path / "someproject"
    work.mkdir()
    r = ToolRegistry()
    assert load_tools(r, write(tmp_path, "p_workdir.py", WORKDIR_PLUGIN), work) == ["workdir_name"]
    assert r.get("workdir_name").fn() == {"name": "someproject"}


def test_dotted_module_name(tmp_path, monkeypatch):
    write(tmp_path, "harness_plugin_dotted.py", REGISTER_PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    r = ToolRegistry()
    assert load_tools(r, "harness_plugin_dotted", tmp_path) == ["greet"]


def test_plugin_errors_are_explicit(tmp_path):
    r = ToolRegistry()
    register_default_tools(r, tmp_path)
    with pytest.raises(PluginError, match="no such file"):
        load_tools(r, str(tmp_path / "missing.py"), tmp_path)
    with pytest.raises(PluginError, match="import failed: ModuleNotFoundError"):
        load_tools(r, "no_such_module_anywhere", tmp_path)
    with pytest.raises(PluginError, match="import failed: RuntimeError: plugin blew up"):
        load_tools(r, write(tmp_path, "p_broken.py", BROKEN_PLUGIN), tmp_path)
    with pytest.raises(PluginError, match="exports neither"):
        load_tools(r, write(tmp_path, "p_empty.py", EMPTY_PLUGIN), tmp_path)
    with pytest.raises(PluginError, match="already registered: fs_read"):
        load_tools(r, write(tmp_path, "p_clash.py", CLASHING_PLUGIN), tmp_path)
    assert r.get("fs_read").fn.__name__ == "fs_read"  # built-in survived the clash


def test_tools_subcommand_lists_builtins_and_plugins(tmp_path, capsys):
    rc = main(["tools", "--workdir", str(tmp_path)])
    out = capsys.readouterr()
    assert rc == 0
    assert "fs_search" in out.out and "run_shell" in out.out and "coin_flip" not in out.out
    assert "scratch_write" in out.out               # rooted in the run directory at run time
    assert "10 tool(s)" in out.err and "toolbelt_add" in out.err

    rc = main(["tools", "--workdir", str(tmp_path),
               "--tools", write(tmp_path, "p_registry.py", REGISTRY_PLUGIN), "--filter", "coin"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out.strip() == "coin_flip  Flip a deterministic coin for a seed."
    assert "loaded 1 tool(s) from --tools: coin_flip" in out.err

    rc = main(["tools", "--workdir", str(tmp_path), "--tools", str(tmp_path / "gone.py")])
    assert rc == 64 and "no such file" in capsys.readouterr().err


def test_run_calls_a_plugin_tool_over_http(tmp_path, capsys):
    plugin = write(tmp_path, "p_register.py", REGISTER_PLUGIN)
    runs = tmp_path / "runs"
    script = [
        {"content": "Finding the plugin tool.", "tool_calls": [call("toolbelt_list", {"filter": "greet"})]},
        {"content": "Activating it.", "tool_calls": [call("toolbelt_add", {"names": ["greet"]})]},
        {"content": "Calling it.", "tool_calls": [call("greet", {"name": "harness"})]},
        {"content": "Done.", "tool_calls": [
            call("todo_write", {"todos": [{"id": "1", "content": "greet", "status": "completed"}]}),
            call("final_answer", {"status": "completed", "content": "greeted"})]},
    ]
    with MockOpenAIServer(script) as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "greet the harness", "--model", "m",
                   "--endpoint", server.base_url, "--workdir", str(tmp_path),
                   "--tools", plugin, "--run-id", "plug"])
    assert rc == 0
    steps = [s for s in read_trajectory(runs / "plug") if s["type"] == "step"]
    assert [(s["tool"], s["kind"]) for s in steps] == [
        ("toolbelt_list", "ok"), ("toolbelt_add", "ok"), ("greet", "ok"),
        ("todo_write", "ok"), ("final_answer", "final_accepted")]
    assert json.loads(steps[0]["result_preview"]) == [{"name": "greet", "description": "Greet someone by name."}]
    assert steps[2]["result_preview"] == "hello harness"
    # the plugin tool was sent to the model only after activation
    assert "greet" not in {t["function"]["name"] for t in server.requests[0]["body"]["tools"]}
    assert "greet" in {t["function"]["name"] for t in server.requests[2]["body"]["tools"]}


def test_bad_tools_option_on_run_is_a_usage_error(tmp_path, capsys):
    rc = main(["--runs-dir", str(tmp_path), "run", "--task", "t", "--model", "m",
               "--endpoint", "http://127.0.0.1:1/v1", "--workdir", str(tmp_path),
               "--tools", str(tmp_path / "nothing.py")])
    assert rc == 64 and "no such file" in capsys.readouterr().err


def test_two_plugin_files_with_the_same_name_coexist(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = write(tmp_path / "a", "tools.py", REGISTER_PLUGIN)
    second = write(tmp_path / "b", "tools.py", REGISTRY_PLUGIN)
    r = ToolRegistry()
    assert load_all(r, [first, second], tmp_path) == ["greet", "coin_flip"]
    assert r.get("greet").fn(name="x") == "hello x"
    assert r.get("coin_flip").fn(seed=3) == {"seed": 3, "side": "tails"}
