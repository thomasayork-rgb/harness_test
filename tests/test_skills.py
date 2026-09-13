"""Skills: the frontmatter format, discovery, and the three meta-tools.

Every skill directory here is a fixture under tmp_path. The conftest fixture
points HOME at a throwaway directory, so nothing reads the machine's own
~/.config/harness/skills.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.cli import main
from harness.context import EVICTED
from harness.mockserver import MockOpenAIServer
from harness.registry import ToolRegistry, ToolSpec
from harness.runtime import META_NAMES, SKILL_META_NAMES, AgentRuntime, RuntimeConfig
from harness.skills import (NoFrontmatter, SkillError, discover, load_skill,
                            parse_frontmatter, search_dirs)
from harness.trajectory import format_summary, read_trajectory, summarize
from harness.transport import FakeTransport, call

REPO_ROOT = Path(__file__).resolve().parents[1]

INVESTIGATE = """---
description: Answer a question about a codebase with evidence.
  Quote paths and line numbers for every claim.
tools: [fs_search, fs_read]
---
# Investigate

1. fs_glob for the shape of the tree.
2. fs_search for the exact text.
3. fs_read the file before you quote it.
"""

TIDY = """---
name: tidy-up
description: Leave the workdir clean.
---
Delete nothing you did not create.
"""


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def skills_dir(tmp_path, name="skills", **files) -> Path:
    """A skills directory: ``investigate=INVESTIGATE`` writes investigate/SKILL.md."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    for skill, body in files.items():
        write(root / skill.replace("_", "-") / "SKILL.md", body)
    return root


def todo(status="completed", id="1"):
    return call("todo_write", {"todos": [{"id": id, "content": "do the work", "status": status}]})


FINISH = call("final_answer", {"status": "completed", "content": "done"})


def registry_with_files():
    r = ToolRegistry()
    for name in ("fs_search", "fs_read", "run_shell"):
        r.register(ToolSpec(name, f"Pretend {name}.", {"type": "object", "properties": {}, "required": []},
                            lambda name=name: f"{name} ran"))
    return r


def tool_names(request):
    return {t["function"]["name"] for t in request["tools"]}


def run_with_skills(tmp_path, script, dirs, run_id="skills", config=None, registry=None):
    """A scripted run whose runtime was given whatever ``dirs`` hold."""
    found = discover(dirs, tmp_path)
    fake = FakeTransport(script)
    rt = AgentRuntime(registry if registry is not None else registry_with_files(), fake,
                      tmp_path / "runs", "fake", config or RuntimeConfig(), run_id=run_id, skills=found)
    return rt.run("a task"), fake, found


def steps_of(res):
    return [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]


# ---- format ---------------------------------------------------------------


def test_frontmatter_parses_scalars_lists_and_continuations():
    meta, body = parse_frontmatter(INVESTIGATE)
    assert meta["description"].splitlines() == ["Answer a question about a codebase with evidence.",
                                                "Quote paths and line numbers for every claim."]
    assert meta["tools"] == ["fs_search", "fs_read"]
    assert body.startswith("# Investigate") and "fs_glob" in body

    meta, body = parse_frontmatter("---\nname: x\ntools:\n  - a\n  - b\nplugin: './p.py'\n---\nbody\n")
    assert meta == {"name": "x", "tools": ["a", "b"], "plugin": "./p.py"} and body == "body"

    with pytest.raises(NoFrontmatter, match="no '---' frontmatter"):
        parse_frontmatter("# just markdown\n")
    with pytest.raises(SkillError, match="is not closed"):
        parse_frontmatter("---\nname: x\ndescription: d\n")
    with pytest.raises(SkillError, match="line 2: not a 'key: value'"):
        parse_frontmatter("---\njust prose\n---\n")


def test_a_skill_takes_its_name_from_the_file_and_needs_a_description(tmp_path):
    directory = write(tmp_path / "s" / "investigate" / "SKILL.md", INVESTIGATE)
    skill = load_skill(directory)
    assert skill.name == "investigate" and skill.tools == ["fs_search", "fs_read"]
    assert skill.summary() == "Answer a question about a codebase with evidence."
    assert skill.brief() == {"name": "investigate", "description": skill.summary()}
    assert skill.chars == len(skill.body) and skill.plugin_path is None

    flat = write(tmp_path / "s" / "tidy.md", TIDY)
    assert load_skill(flat).name == "tidy-up"           # frontmatter name wins over the stem
    assert load_skill(write(tmp_path / "s" / "plain.md", "---\ndescription: d\n---\nb\n")).name == "plain"

    with pytest.raises(SkillError, match="needs a 'description'"):
        load_skill(write(tmp_path / "s" / "nodesc.md", "---\nname: nodesc\n---\nbody\n"))
    with pytest.raises(SkillError, match="cannot read skill"):
        load_skill(tmp_path / "s" / "missing.md")


# ---- discovery ------------------------------------------------------------


def test_discovery_merges_directories_and_reports_clashes(tmp_path):
    first = skills_dir(tmp_path, "first", investigate=INVESTIGATE)
    write(first / "notes.md", "# not a skill, no frontmatter\n")
    second = skills_dir(tmp_path, "second")
    write(second / "investigate" / "SKILL.md", "---\ndescription: The newer investigate.\n---\nnewer\n")
    write(second / "tidy.md", TIDY)

    found = discover([first, second])
    assert found.names() == ["investigate", "tidy-up"]
    assert found.get("investigate").body == "newer"          # the later directory won
    assert len(found.clashes) == 1
    assert "skill 'investigate'" in found.clashes[0]
    assert str(second / "investigate" / "SKILL.md") in found.clashes[0]
    assert str(first / "investigate" / "SKILL.md") in found.clashes[0]
    assert found.errors == []                                # notes.md was never a skill
    assert found.list("tidy") == [{"name": "tidy-up", "description": "Leave the workdir clean."}]
    assert found.describe() == {"dirs": [str(first), str(second)], "names": ["investigate", "tidy-up"]}
    assert len(found) == 2 and "tidy-up" in found and discover([]).names() == []


def test_a_broken_skill_file_is_reported_not_silently_dropped(tmp_path):
    root = skills_dir(tmp_path, "s", good=TIDY)
    write(root / "broken" / "SKILL.md", "---\nname: broken\n---\nno description\n")
    write(root / "bare" / "SKILL.md", "# a skill directory with no frontmatter\n")

    found = discover([root])
    assert found.names() == ["tidy-up"]
    assert len(found.errors) == 2
    assert any("needs a 'description'" in e for e in found.errors)
    assert any("no '---' frontmatter" in e for e in found.errors)


def test_search_dirs_order_is_config_env_project_then_flags(tmp_path, monkeypatch):
    config = skills_dir(tmp_path, "config-skills", one=TIDY)
    env_a = skills_dir(tmp_path, "env-a", two=TIDY)
    env_b = skills_dir(tmp_path, "env-b", three=TIDY)
    project = tmp_path / "project"
    (project / ".harness" / "skills").mkdir(parents=True)
    flag = skills_dir(tmp_path, "flag", four=TIDY)

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    write(tmp_path / "xdg" / "harness" / "skills" / "x" / "SKILL.md", TIDY)
    monkeypatch.setenv("HARNESS_SKILLS", os.pathsep.join([str(env_a), str(env_b), " "]))

    dirs = search_dirs([flag], workdir=project)
    assert dirs == [tmp_path / "xdg" / "harness" / "skills", env_a, env_b,
                    project / ".harness" / "skills", flag]
    assert search_dirs([flag, flag], workdir=project) == dirs   # the same directory once

    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.delenv("HARNESS_SKILLS")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    write(tmp_path / "home" / ".config" / "harness" / "skills" / "h" / "SKILL.md", TIDY)
    home_skills = tmp_path / "home" / ".config" / "harness" / "skills"
    assert search_dirs([tmp_path / "nowhere"]) == [home_skills]   # a missing directory is not searched
    assert config.is_dir()                                   # not searched: it is nobody's location


# ---- the meta-tools, through the runtime ----------------------------------


def test_list_then_load_puts_the_skill_in_context_and_activates_its_tools(tmp_path):
    root = skills_dir(tmp_path, "skills", investigate=INVESTIGATE, tidy=TIDY)
    script = [
        {"content": "What skills are there?", "tool_calls": [call("skill_list", {})]},
        {"content": "investigate matches; loading it.", "tool_calls": [call("skill_load", {"name": "investigate"})]},
        {"content": "Following it.", "tool_calls": [call("fs_search", {})]},
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ]
    res, fake, found = run_with_skills(tmp_path, script, [root])
    assert res.status == "completed"

    steps = steps_of(res)
    assert [(s["tool"], s["kind"]) for s in steps] == [
        ("skill_list", "ok"), ("skill_load", "ok"), ("fs_search", "ok"),
        ("todo_write", "ok"), ("final_answer", "final_accepted")]
    assert json.loads(steps[0]["result_preview"]) == [
        {"name": "investigate", "description": "Answer a question about a codebase with evidence."},
        {"name": "tidy-up", "description": "Leave the workdir clean."}]

    loaded = (res.run_dir / steps[1]["artifact"]).read_text(encoding="utf-8")
    assert loaded.startswith("skill 'investigate' loaded (")
    assert "activated fs_search, fs_read" in loaded
    assert found.get("investigate").body in loaded

    # guarantee 1 still: skills are meta, the tools they declare are not in
    # context until the load activates them
    assert tool_names(fake.requests[0]) == META_NAMES | SKILL_META_NAMES
    assert tool_names(fake.requests[2]) == META_NAMES | SKILL_META_NAMES | {"fs_search", "fs_read"}
    # and the whole skill reached the model as the tool result
    result = [m for m in fake.requests[2]["messages"] if m["role"] == "tool"][-1]
    assert "fs_glob for the shape of the tree." in result["content"]

    state = json.loads((res.run_dir / "state.json").read_text())
    assert state["loaded_skills"] == ["investigate"]
    assert state["active_tools"] == ["fs_search", "fs_read"]


def test_a_run_without_skills_starts_with_exactly_the_six_meta_tools(tmp_path):
    empty = tmp_path / "no-skills"
    empty.mkdir()
    script = [{"content": "Trying a skill tool anyway.", "tool_calls": [call("skill_list", {})]},
              {"content": "Done.", "tool_calls": [todo(), FINISH]}]
    res, fake, found = run_with_skills(tmp_path, script, [empty], run_id="bare")
    assert len(found) == 0 and res.status == "completed"
    assert tool_names(fake.requests[0]) == META_NAMES
    assert not (META_NAMES & SKILL_META_NAMES)
    steps = steps_of(res)
    assert steps[0]["kind"] == "error" and "unknown tool 'skill_list'" in steps[0]["result_preview"]
    # searched, and empty: the header says so rather than staying silent
    assert read_trajectory(res.run_dir)[0]["skills"] == {"dirs": [str(empty)], "names": []}


def test_unknown_oversized_and_repeated_loads(tmp_path):
    root = skills_dir(tmp_path, "skills", investigate=INVESTIGATE)
    write(root / "huge" / "SKILL.md", "---\ndescription: A very long skill.\n---\n" + "x" * 500)
    script = [
        {"content": "Loading something that is not there.", "tool_calls": [call("skill_load", {"name": "nope"})]},
        {"content": "The big one, then.", "tool_calls": [call("skill_load", {"name": "huge"})]},
        {"content": "investigate it is.", "tool_calls": [call("skill_load", {"name": "investigate"})]},
        {"content": "Again, by accident.", "tool_calls": [call("skill_load", {"name": "investigate"})]},
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ]
    res, fake, _ = run_with_skills(tmp_path, script, [root], run_id="edges",
                                   config=RuntimeConfig(skill_chars=200))
    steps = steps_of(res)
    assert [s["kind"] for s in steps[:4]] == ["error", "error", "ok", "ok"]
    assert "unknown skill 'nope'. Available: huge, investigate" in steps[0]["result_preview"]
    assert "is 500 chars, over the 200 char limit" in steps[1]["result_preview"]
    assert "--skill-chars" in steps[1]["result_preview"]
    assert steps[3]["result_preview"] == "skill 'investigate' is already loaded; its text is above in this conversation."

    state = json.loads((res.run_dir / "state.json").read_text())
    assert state["loaded_skills"] == ["investigate"]          # the refused ones left nothing behind
    assert state["active_tools"] == ["fs_search", "fs_read"]
    # the second load added no second copy of the text
    tool_msgs = [m for m in fake.requests[4]["messages"] if m["role"] == "tool"]
    assert sum("fs_glob for the shape" in m["content"] for m in tool_msgs) == 1


def test_unload_frees_the_context_and_leaves_the_tools_active(tmp_path):
    root = skills_dir(tmp_path, "skills", investigate=INVESTIGATE)
    script = [
        {"content": "Loading.", "tool_calls": [call("skill_load", {"name": "investigate"})]},
        {"content": "Done with it.", "tool_calls": [call("skill_unload", {"name": "investigate"})]},
        {"content": "Unloading it twice.", "tool_calls": [call("skill_unload", {"name": "investigate"})]},
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ]
    res, fake, found = run_with_skills(tmp_path, script, [root], run_id="unload")
    steps = steps_of(res)
    assert [s["kind"] for s in steps[:3]] == ["ok", "ok", "error"]
    freed = json.loads(steps[1]["result_preview"])
    assert freed["unloaded"] == "investigate" and freed["loaded"] == []
    assert freed["context_freed_chars"] > len(found.get("investigate").body) - 200
    assert "is not loaded" in steps[2]["result_preview"]

    last = fake.requests[-1]["messages"]
    skill_msg = [m for m in last if m["role"] == "tool"][0]
    assert skill_msg["content"] == "[skill 'investigate' unloaded; its text is out of context. skill_load reads it again.]"
    assert "fs_glob for the shape" not in json.dumps(last)
    state = json.loads((res.run_dir / "state.json").read_text())
    assert state["loaded_skills"] == [] and state["active_tools"] == ["fs_search", "fs_read"]


def test_a_loaded_skill_is_neither_truncated_nor_evicted(tmp_path):
    """The model was told to follow the skill; a budget that quietly removed it
    would leave it following something it can no longer read."""
    root = tmp_path / "skills"
    body = "\n".join(f"Step {i}: do the thing carefully." for i in range(40))
    write(root / "long" / "SKILL.md", f"---\ndescription: A long guide.\n---\n{body}")
    r = ToolRegistry()
    r.register(ToolSpec("big", "Return a lot.", {"type": "object", "properties": {}, "required": []},
                        lambda: "B" * 900))
    script = [
        {"content": "Loading the guide.", "tool_calls": [call("skill_load", {"name": "long"})]},
        {"content": "Activating.", "tool_calls": [call("toolbelt_add", {"names": ["big"]})]},
    ] + [{"content": f"call {i}", "tool_calls": [call("big", {})]} for i in range(5)] + [
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ]
    cfg = RuntimeConfig(result_context_chars=200, context_budget_chars=2500)
    res, fake, _ = run_with_skills(tmp_path, script, [root], run_id="evict", config=cfg, registry=r)
    assert res.status == "completed"

    tool_msgs = [m for m in fake.requests[-1]["messages"] if m["role"] == "tool"]
    assert any(m["content"].startswith(EVICTED[:20]) for m in tool_msgs)   # other results went
    skill_msg = tool_msgs[0]
    assert body in skill_msg["content"]                                    # whole, untruncated
    assert "truncated at 200" not in skill_msg["content"]


def test_a_skill_can_bring_its_own_tools_through_a_plugin(tmp_path):
    root = tmp_path / "skills"
    write(root / "counting" / "tools.py", '''
def register(registry, workdir):
    @registry.tool("count_lines", "Count the lines in a string.",
                   {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    def count_lines(text: str) -> dict:
        return {"lines": len(text.splitlines()), "workdir": workdir.name}
''')
    write(root / "counting" / "SKILL.md",
          "---\ndescription: Count things.\ntools: [count_lines]\nplugin: ./tools.py\n---\nUse count_lines.\n")
    write(root / "broken-plugin" / "SKILL.md",
          "---\ndescription: Bring a plugin that is not there.\nplugin: ./gone.py\n---\nnothing\n")
    script = [
        {"content": "Loading a skill that brings a tool.", "tool_calls": [call("skill_load", {"name": "counting"})]},
        {"content": "Using it.", "tool_calls": [call("count_lines", {"text": "a\nb\nc"})]},
        {"content": "And the broken one.", "tool_calls": [call("skill_load", {"name": "broken-plugin"})]},
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ]
    work = tmp_path / "project"
    work.mkdir()
    found = discover([root], work)
    fake = FakeTransport(script)
    rt = AgentRuntime(registry_with_files(), fake, tmp_path / "runs", "fake", run_id="plug", skills=found)
    res = rt.run("count things")

    steps = steps_of(res)
    assert [s["kind"] for s in steps[:3]] == ["ok", "ok", "error"]
    assert "plugin: registered count_lines." in steps[0]["result_preview"]
    assert json.loads(steps[1]["result_preview"]) == {"lines": 3, "workdir": "project"}
    assert "declares a plugin that will not load" in steps[2]["result_preview"]
    assert "no such file" in steps[2]["result_preview"]
    state = json.loads((res.run_dir / "state.json").read_text())
    assert state["loaded_skills"] == ["counting"]


def test_the_header_and_the_summary_say_which_skills_were_there_and_which_were_read(tmp_path):
    root = skills_dir(tmp_path, "skills", investigate=INVESTIGATE, tidy=TIDY)
    script = [
        {"content": "Loading.", "tool_calls": [call("skill_load", {"name": "investigate"})]},
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ]
    res, _, _ = run_with_skills(tmp_path, script, [root], run_id="recorded")
    records = read_trajectory(res.run_dir)
    assert records[0]["skills"] == {"dirs": [str(root)], "names": ["investigate", "tidy-up"]}

    s = summarize(records)
    assert s["skills_available"] == ["investigate", "tidy-up"] and s["skills_loaded"] == ["investigate"]
    assert s["skill_dirs"] == [str(root)]
    assert "skills: loaded investigate  (2 discovered in 1 directory)" in format_summary(records)


# ---- the CLI --------------------------------------------------------------


def test_harness_skills_lists_what_the_agent_could_load(tmp_path, capsys, monkeypatch):
    first = skills_dir(tmp_path, "first", investigate=INVESTIGATE, tidy=TIDY)
    second = skills_dir(tmp_path, "second")
    write(second / "tidy" / "SKILL.md", "---\nname: tidy-up\ndescription: The project's own tidy.\n---\nmine\n")
    work = tmp_path / "project"
    (work / ".harness" / "skills").mkdir(parents=True)

    assert main(["skills", "--skills", str(first), "--skills", str(second), "--workdir", str(work)]) == 0
    out = capsys.readouterr()
    assert out.out.splitlines() == [
        f"investigate  Answer a question about a codebase with evidence.  ({first})",
        f"tidy-up      The project's own tidy.  ({second})"]
    assert "2 skill(s) in 3 directories" in out.err            # the empty project one was searched
    assert f"skill 'tidy-up': using {second / 'tidy' / 'SKILL.md'}" in out.err
    assert str(first / "tidy" / "SKILL.md") in out.err         # the clash names both, once
    assert out.err.count("skill 'tidy-up'") == 1

    assert main(["skills", "--skills", str(first), "--filter", "codebase"]) == 0
    assert capsys.readouterr().out.strip().startswith("investigate")
    assert main(["skills", "--workdir", str(work)]) == 0       # nothing anywhere: no output, no crash
    assert capsys.readouterr().out == ""

    # a --skills directory that is not there is said out loud, not shrugged off
    assert main(["skills", "--skills", str(tmp_path / "gone")]) == 0
    assert f"skills: --skills {tmp_path / 'gone'}: no such directory" in capsys.readouterr().err


def test_the_shipped_examples_are_loadable_skills():
    found = discover([REPO_ROOT / "examples" / "skills"])
    assert found.names() == ["code-change", "final-report", "investigate"]
    assert found.errors == [] and found.clashes == []
    assert found.get("code-change").tools == ["fs_search", "fs_read", "fs_edit", "run_shell"]
    assert "fs_edit" in found.get("code-change").body
    assert all(s.chars < 12000 for s in found.skills.values())   # under the default --skill-chars


def test_cli_run_with_skills_over_http(tmp_path):
    """A real process: the SKILLS section reaches the model only because skills
    were found, and the skill the model loaded reaches it as a tool result."""
    root = skills_dir(tmp_path, "skills", investigate=INVESTIGATE)
    work = tmp_path / "project"
    work.mkdir()
    (work / "config.ini").write_text("[server]\nPORT = 8080\n", encoding="utf-8")
    runs = tmp_path / "runs"
    script = [
        {"content": "Any skills for this?", "tool_calls": [call("skill_list", {})]},
        {"content": "investigate matches; loading it.", "tool_calls": [call("skill_load", {"name": "investigate"})]},
        {"content": "Following it: search first.",
         "tool_calls": [call("fs_search", {"pattern": "(?i)port"})]},
        {"content": "Done with the guide.", "tool_calls": [call("skill_unload", {"name": "investigate"})]},
        {"content": "Closing.", "tool_calls": [
            call("todo_write", {"todos": [{"id": "find", "content": "find the port", "status": "completed"}]}),
            call("final_answer", {"status": "completed", "content": "8080, config.ini:2"})]},
    ]
    with MockOpenAIServer(script, model="mock-model") as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run", "--task", "find the port",
             "--model", "mock-model", "--endpoint", server.base_url, "--workdir", str(work),
             "--skills", str(root), "--run-id", "skilled"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
        bodies = [r["body"] for r in server.requests]

    assert proc.returncode == 0, proc.stderr
    system = bodies[0]["messages"][0]["content"]
    assert "SKILLS. Written guides" in system and "skill_load(name)" in system
    assert {t["function"]["name"] for t in bodies[0]["tools"]} == META_NAMES | SKILL_META_NAMES
    # the skill text was in context for the search turn, and gone after the unload
    assert "fs_glob for the shape of the tree." in json.dumps(bodies[2]["messages"])
    assert "fs_glob for the shape of the tree." not in json.dumps(bodies[4]["messages"])
    assert "unloaded" in json.dumps(bodies[4]["messages"])
    records = read_trajectory(runs / "skilled")
    # fs_search was never activated by hand: loading the skill declared it
    assert "toolbelt_add" not in [r.get("tool") for r in records]
    assert records[0]["skills"] == {"dirs": [str(root)], "names": ["investigate"]}
    assert records[0]["invocation"]["skills"] == [str(root)]
    assert summarize(records)["skills_loaded"] == ["investigate"]


def test_resume_keeps_the_skills_the_run_was_launched_with(tmp_path):
    root = skills_dir(tmp_path, "skills", investigate=INVESTIGATE)
    work, runs = tmp_path / "project", tmp_path / "runs"
    work.mkdir()
    first = [{"content": "Planning.", "tool_calls": [
        call("todo_write", {"todos": [{"id": "1", "content": "read the guide", "status": "in_progress"}]})]}]
    with MockOpenAIServer(first, model="mock-model") as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "mock-model",
                   "--endpoint", server.base_url, "--workdir", str(work), "--run-id", "res",
                   "--skills", str(root), "--step-cap", "1"])
    assert rc == 3

    rest = [{"content": "Loading the skill after the resume.",
             "tool_calls": [call("skill_load", {"name": "investigate"})]},
            {"content": "Closing.", "tool_calls": [
                call("todo_write", {"todos": [{"id": "1", "content": "read the guide",
                                               "status": "completed"}]})]},
            {"content": "Done.", "tool_calls": [FINISH]}]
    with MockOpenAIServer(rest, model="mock-model") as server:
        rc = main(["--runs-dir", str(runs), "resume", "res", "--endpoint", server.base_url,
                   "--step-cap", "8"])
        tools = [{t["function"]["name"] for t in r["body"]["tools"]} for r in server.requests]
    assert rc == 0

    records = read_trajectory(runs / "res")
    steps = [r for r in records if r["type"] == "step"]
    assert [(s["tool"], s["kind"]) for s in steps] == [
        ("todo_write", "ok"), ("skill_load", "ok"), ("todo_write", "ok"), ("final_answer", "final_accepted")]
    assert SKILL_META_NAMES <= tools[0]                       # the resumed segment still had them
    assert "fs_glob for the shape of the tree." in (runs / "res" / steps[1]["artifact"]).read_text()
    seam = next(r for r in records if r["type"] == "resume")
    assert seam["invocation"]["skills"] == [str(root)]        # recorded again, for the next resume


def test_a_recorded_skills_run_replays_as_a_skills_run(tmp_path):
    """replay rebuilds the skill set from the header, the way it rebuilds the
    policy: otherwise a recorded skill_load comes back "unknown tool" and every
    later step drifts."""
    from harness.replay import ReplayTransport, compare, replay

    root = skills_dir(tmp_path, "skills", investigate=INVESTIGATE)
    script = [
        {"content": "Loading.", "tool_calls": [call("skill_load", {"name": "investigate"})]},
        {"content": "Using what it activated.", "tool_calls": [call("fs_search", {})]},
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ]
    res, _, _ = run_with_skills(tmp_path, script, [root], run_id="tape")
    assert [s["kind"] for s in steps_of(res)] == ["ok", "ok", "ok", "final_accepted"]

    transport = ReplayTransport(res.run_dir)
    assert transport.skills.names() == ["investigate"]
    again = replay(res.run_dir, registry_with_files(), runs_dir=tmp_path / "runs")
    assert again.status == "completed" and compare(res.run_dir, again.run_dir) == []


def test_a_skill_with_a_plugin_can_be_unloaded_and_loaded_again(tmp_path):
    """The second load must not collide with the tools the first one registered."""
    root = tmp_path / "skills"
    write(root / "counting" / "tools.py", '''
def register(registry):
    @registry.tool("count_lines", "Count lines.", {"type": "object", "properties": {}, "required": []})
    def count_lines() -> int:
        return 1
''')
    write(root / "counting" / "SKILL.md",
          "---\ndescription: Count things.\ntools: [count_lines]\nplugin: ./tools.py\n---\nUse count_lines.\n")
    script = [
        {"content": "Loading.", "tool_calls": [call("skill_load", {"name": "counting"})]},
        {"content": "Freeing the context.", "tool_calls": [call("skill_unload", {"name": "counting"})]},
        {"content": "Needed it after all.", "tool_calls": [call("skill_load", {"name": "counting"})]},
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ]
    res, _, _ = run_with_skills(tmp_path, script, [root], run_id="replug")
    steps = steps_of(res)
    assert [s["kind"] for s in steps[:3]] == ["ok", "ok", "ok"]
    assert "plugin: registered count_lines." in steps[0]["result_preview"]
    assert "plugin: already registered." in steps[2]["result_preview"]
    assert "already active: count_lines" in steps[2]["result_preview"]
    assert json.loads((res.run_dir / "state.json").read_text())["loaded_skills"] == ["counting"]


COUNTING_PLUGIN = '''
def register(registry, workdir):
    @registry.tool("count_lines", "Count the lines in a string.",
                   {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    def count_lines(text: str) -> dict:
        return {"lines": len(text.splitlines())}
'''
COUNTING_SKILL = "---\ndescription: Count things.\ntools: [count_lines]\nplugin: ./tools.py\n---\nUse count_lines.\n"


def project_with_plugin_skill(tmp_path) -> Path:
    """A workdir whose own .harness/skills carries a skill that brings code."""
    work = tmp_path / "project"
    root = work / ".harness" / "skills"
    write(root / "counting" / "tools.py", COUNTING_PLUGIN)
    write(root / "counting" / "SKILL.md", COUNTING_SKILL)
    write(root / "plain" / "SKILL.md", "---\ndescription: No plugin here.\n---\nJust text.\n")
    return work


def test_a_project_skill_with_a_plugin_is_refused_unless_trusted(tmp_path):
    """A plugin under <workdir>/.harness/skills is the project's own code. By
    default nothing is loaded - not the text either - and the tool never
    reaches the registry; a run that opted in gets both."""
    work = project_with_plugin_skill(tmp_path)
    found = discover(search_dirs([], work), work)
    assert found.is_project_skill(found.get("counting")) and found.is_project_skill(found.get("plain"))
    elsewhere = discover([skills_dir(tmp_path, "mine", investigate=INVESTIGATE)], work)
    assert not elsewhere.is_project_skill(elsewhere.get("investigate"))

    registry = registry_with_files()
    fake = FakeTransport([
        {"content": "Loading the project's skill.", "tool_calls": [call("skill_load", {"name": "counting"})]},
        {"content": "The plain one, then.", "tool_calls": [call("skill_load", {"name": "plain"})]},
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ])
    rt = AgentRuntime(registry, fake, tmp_path / "runs", "fake", run_id="untrusted", skills=found)
    res = rt.run("count things")
    steps = steps_of(res)
    assert [s["kind"] for s in steps[:2]] == ["error", "ok"]
    assert "--trust-project-plugins" in steps[0]["result_preview"]
    assert "It was not loaded" in steps[0]["result_preview"]
    assert "count_lines" not in registry
    assert "Use count_lines." not in json.dumps(fake.requests[1]["messages"])
    state = json.loads((res.run_dir / "state.json").read_text())
    assert state["loaded_skills"] == ["plain"]

    registry = registry_with_files()
    fake = FakeTransport([
        {"content": "Loading the project's skill.", "tool_calls": [call("skill_load", {"name": "counting"})]},
        {"content": "Using it.", "tool_calls": [call("count_lines", {"text": "a\nb"})]},
        {"content": "Done.", "tool_calls": [todo(), FINISH]},
    ])
    rt = AgentRuntime(registry, fake, tmp_path / "runs", "fake", RuntimeConfig(trust_project_plugins=True),
                      run_id="trusted", skills=discover(search_dirs([], work), work))
    res = rt.run("count things")
    steps = steps_of(res)
    assert [s["kind"] for s in steps[:2]] == ["ok", "ok"]
    assert "plugin: registered count_lines." in steps[0]["result_preview"]
    assert json.loads(steps[1]["result_preview"]) == {"lines": 2}


def test_harness_skills_marks_a_project_plugin_as_gated(tmp_path, capsys):
    work = project_with_plugin_skill(tmp_path)
    assert main(["skills", "--workdir", str(work), "--filter", "count"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("counting  Count things.")
    assert line.endswith("[plugin: needs --trust-project-plugins]")
    assert main(["skills", "--workdir", str(work), "--filter", "plugin"]) == 0
    assert "[plugin:" not in capsys.readouterr().out          # the plain skill carries no marker


def test_cli_refuses_a_project_plugin_without_the_flag_over_http(tmp_path):
    """A real process, twice over the same script: the project's plugin is
    refused by default and registered with --trust-project-plugins, and the
    header records which it was."""
    work = project_with_plugin_skill(tmp_path)
    runs = tmp_path / "runs"

    def script():
        return [
            {"content": "Loading the project's skill.", "tool_calls": [call("skill_load", {"name": "counting"})]},
            {"content": "Closing.", "tool_calls": [
                call("todo_write", {"todos": [{"id": "c", "content": "count", "status": "completed"}]}),
                call("final_answer", {"status": "completed", "content": "done"})]},
        ]

    def run(run_id, *flags):
        with MockOpenAIServer(script(), model="mock-model") as server:
            proc = subprocess.run(
                [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run", "--task", "count",
                 "--model", "mock-model", "--endpoint", server.base_url, "--workdir", str(work),
                 "--run-id", run_id, *flags],
                cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
                env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
        assert proc.returncode == 0, proc.stderr
        records = read_trajectory(runs / run_id)
        return records[0], [r for r in records if r["type"] == "step"]

    header, steps = run("refused")
    assert steps[0]["kind"] == "error" and "--trust-project-plugins" in steps[0]["result_preview"]
    assert header["config"]["trust_project_plugins"] is False
    header, steps = run("trusted", "--trust-project-plugins")
    assert steps[0]["kind"] == "ok" and "plugin: registered count_lines." in steps[0]["result_preview"]
    assert header["config"]["trust_project_plugins"] is True


def test_resume_can_opt_into_a_project_plugin(tmp_path):
    """A run that refused the project's plugin is resumed with the flag: the
    same skill now loads, and the tool it brought is callable."""
    work = project_with_plugin_skill(tmp_path)
    runs = tmp_path / "runs"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    first = [{"content": "Loading the project's skill.", "tool_calls": [call("skill_load", {"name": "counting"})]}]
    with MockOpenAIServer(first, model="mock-model") as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run", "--task", "count",
             "--model", "mock-model", "--endpoint", server.base_url, "--workdir", str(work),
             "--run-id", "capped", "--step-cap", "1"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 3, proc.stderr                 # step_cap: resumable

    rest = [
        {"content": "Trying again, trusted now.", "tool_calls": [call("skill_load", {"name": "counting"})]},
        {"content": "Using it.", "tool_calls": [call("count_lines", {"text": "a\nb"})]},
        {"content": "Closing.", "tool_calls": [
            call("todo_write", {"todos": [{"id": "c", "content": "count", "status": "completed"}]}),
            call("final_answer", {"status": "completed", "content": "2 lines"})]},
    ]
    with MockOpenAIServer(rest, model="mock-model") as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "resume", "capped",
             "--endpoint", server.base_url, "--step-cap", "10", "--trust-project-plugins"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    steps = [r for r in read_trajectory(runs / "capped") if r["type"] == "step"]
    assert [s["kind"] for s in steps[:3]] == ["error", "ok", "ok"]
    assert "plugin: registered count_lines." in steps[1]["result_preview"]
    assert json.loads(steps[2]["result_preview"]) == {"lines": 2}
