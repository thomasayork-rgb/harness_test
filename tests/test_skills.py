"""Skills: the frontmatter format, discovery, and the three meta-tools.

Every skill directory here is a fixture under tmp_path. The conftest fixture
points HOME at a throwaway directory, so nothing reads the machine's own
~/.config/harness/skills.
"""
import os
from pathlib import Path

import pytest

from harness.skills import (NoFrontmatter, SkillError, discover, load_skill,
                            parse_frontmatter, search_dirs)

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
