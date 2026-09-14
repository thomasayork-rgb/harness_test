"""The code map: what scaffolding generates, and what it must never touch.

The fixture project has two packages that import each other, a subpackage, a
directory of modules that is not a package, and a file that does not parse - so
every branch of the scan has something to find. The properties that matter are
that the frontmatter says what is true about the tree, that the prose under it
survives a refresh byte for byte, and that the same tree always produces the
same bytes.
"""
import json
import shutil
from pathlib import Path

from harness.cli import main
from harness.codemap import (INDEX_FILE, MAP_RELATIVE, SECTIONS, TEMPLATE, blob_sha, scaffold,
                             scan, section, stale, stale_against_tree)
from harness.frontmatter import parse_frontmatter

from .gitfixture import commit_all, git, sample_project, write

MAP = MAP_RELATIVE.as_posix()


def meta_of(repo: Path, package: str) -> dict:
    text = (repo / MAP_RELATIVE / f"{package}.md").read_text(encoding="utf-8")
    return parse_frontmatter(text)[0]


def body_of(repo: Path, package: str) -> str:
    text = (repo / MAP_RELATIVE / f"{package}.md").read_text(encoding="utf-8")
    return parse_frontmatter(text)[1]


def files_under(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ---- the scan --------------------------------------------------------------


def test_packages_subpackages_and_namespace_directories(tmp_path):
    project = scan(sample_project(tmp_path / "p"))
    assert project.names() == ["alpha", "alpha.nested", "beta"]
    assert project.unpackaged == ["scripts/"]              # .py files, no __init__.py
    assert [m.rel for m in project.packages["beta"].modules] == [
        "beta/__init__.py", "beta/broken.py", "beta/util.py"]
    assert project.select("alpha") == [project.packages["alpha"], project.packages["alpha.nested"]]
    assert project.select("nothing") == []


def test_the_frontmatter_says_what_is_true_about_the_tree(tmp_path):
    repo = sample_project(tmp_path / "p")
    scaffold(repo)

    alpha = meta_of(repo, "alpha")
    assert alpha["map_version"] == "1" and alpha["package"] == "alpha"
    assert alpha["files"] == ["alpha/__init__.py", "alpha/core.py"]
    assert alpha["imports"] == {"internal": ["beta.util"], "external": ["os"]}
    assert alpha["public"] == {"alpha/__init__.py": ["Engine"],          # __all__ wins
                               "alpha/core.py": ["Engine", "VERSION", "build"]}
    # count is files outside the package; a caller's number is how many names of
    # the package it imports - deep.py names alpha.core and alpha.nested
    assert alpha["inbound_refs"] == {"count": "2",
                                     "callers": {"alpha.nested.deep": "2", "scripts.tool": "1"}}
    assert "errors" not in alpha

    nested = meta_of(repo, "alpha.nested")
    assert nested["imports"] == {"internal": ["alpha.core"], "external": []}
    assert nested["inbound_refs"] == {"count": "0"}        # no callers key when there are none
    assert nested["public"] == {"alpha/nested/deep.py": ["dive"]}   # the empty __init__ is left out

    beta = meta_of(repo, "beta")
    # 6 lines of util.py, 1 of broken.py, 0 of the empty __init__: the file that
    # does not parse is still a file
    assert beta["loc"] == "7" and beta["files"] == ["beta/__init__.py", "beta/broken.py",
                                                    "beta/util.py"]
    assert beta["imports"] == {"internal": [], "external": ["json", "pathlib"]}
    assert beta["public"] == {"beta/util.py": ["helper"]}
    assert beta["inbound_refs"] == {"count": "1", "callers": {"alpha.core": "1"}}
    assert beta["errors"] == ["beta/broken.py"]            # it still counts everywhere else
    assert set(beta["generated_from"]) == set(beta["files"])


def test_a_package_does_not_import_itself(tmp_path):
    """``imports.internal`` is what a reader has to go and look at elsewhere,
    so a package's own modules and subpackages are not in it - however much of
    itself it imports."""
    repo = sample_project(tmp_path / "p")
    write(repo / "alpha" / "core.py",
          "from alpha.nested.deep import dive\nfrom alpha import core\nfrom beta.util import helper\n")
    scaffold(repo)
    assert meta_of(repo, "alpha")["imports"]["internal"] == ["beta.util"]
    # the subpackage's own `from . import ...` is gone; what its parent holds is not
    assert meta_of(repo, "alpha.nested")["imports"]["internal"] == ["alpha.core"]
    # ... and the package is still counted as a caller of the ones it imports
    assert meta_of(repo, "alpha.nested")["inbound_refs"] == {"count": "1",
                                                            "callers": {"alpha.core": "1"}}


def test_generated_from_holds_the_sha_git_would_print(tmp_path):
    repo = sample_project(tmp_path / "p")
    scaffold(repo)
    recorded = meta_of(repo, "beta")["generated_from"]
    for line in git(repo, "ls-tree", "-r", "HEAD", "--", "beta").splitlines():
        _, _, rest = line.partition(" blob ")
        sha, _, path = rest.partition("\t")
        assert recorded[path] == sha
    assert recorded["beta/util.py"] == blob_sha((repo / "beta" / "util.py").read_bytes())


# ---- writing the files -----------------------------------------------------


def test_a_new_map_file_is_frontmatter_plus_the_template(tmp_path):
    repo = sample_project(tmp_path / "p")
    written = scaffold(repo)
    assert [action for action, _ in written] == ["wrote"] * 4        # three packages and the index
    assert [rel for _, rel in written] == [f"{MAP}/alpha.md", f"{MAP}/alpha.nested.md",
                                           f"{MAP}/beta.md", f"{MAP}/{INDEX_FILE}"]

    text = (repo / MAP_RELATIVE / "alpha.md").read_text(encoding="utf-8")
    assert text.startswith("---\nmap_version: 1\npackage: alpha\n")
    assert text.endswith("## Depended on by\n") and not text.endswith("\n\n")
    assert body_of(repo, "alpha") == TEMPLATE
    assert [f"## {name}" for name in SECTIONS] == TEMPLATE.split("\n\n")


def test_scaffolding_twice_writes_the_same_bytes(tmp_path):
    repo = sample_project(tmp_path / "p")
    scaffold(repo)
    before = files_under(repo / MAP_RELATIVE)
    again = scaffold(repo)
    assert [action for action, _ in again] == ["unchanged"] * 4
    assert files_under(repo / MAP_RELATIVE) == before

    # ... and a second project built from the same files maps to the same bytes
    other = sample_project(tmp_path / "q")
    scaffold(other)
    assert files_under(other / MAP_RELATIVE) == before


PROSE = """## Purpose

The engine package: `Engine.start` is the whole of it.

## Entry points

- `build(name)` -> Engine

## Invariants

## Gotchas

`_private` is not exported, and nothing should start depending on it.

## Depends on

beta.util

## Depended on by

scripts/tool.py"""


def test_a_refresh_updates_the_facts_and_keeps_the_prose(tmp_path):
    repo = sample_project(tmp_path / "p")
    scaffold(repo)
    path = repo / MAP_RELATIVE / "alpha.md"
    write(path, path.read_text(encoding="utf-8").replace(TEMPLATE, PROSE))
    before = path.read_text(encoding="utf-8")

    core = repo / "alpha" / "core.py"
    write(core, core.read_text(encoding="utf-8") + "\n\ndef stop():\n    return None\n")
    actions = dict((rel, action) for action, rel in scaffold(repo))
    assert actions[f"{MAP}/alpha.md"] == "refreshed"
    assert actions[f"{MAP}/beta.md"] == "unchanged"          # nothing about beta changed

    alpha = meta_of(repo, "alpha")
    assert alpha["public"]["alpha/core.py"] == ["Engine", "VERSION", "build", "stop"]
    assert alpha["generated_from"]["alpha/core.py"] == blob_sha(core.read_bytes())
    assert int(alpha["loc"]) > int(parse_frontmatter(before)[0]["loc"])
    assert body_of(repo, "alpha") == PROSE                  # byte for byte
    assert path.read_text(encoding="utf-8").endswith(PROSE + "\n")


def test_prose_in_a_file_without_frontmatter_is_kept_as_the_body(tmp_path):
    repo = sample_project(tmp_path / "p")
    write(repo / MAP_RELATIVE / "beta.md", "## Purpose\n\nHand written, no fence.\n")
    scaffold(repo)
    assert body_of(repo, "beta") == "## Purpose\n\nHand written, no fence."
    assert meta_of(repo, "beta")["package"] == "beta"


def test_the_index_groups_by_top_level_package_and_orders_by_inbound(tmp_path):
    repo = sample_project(tmp_path / "p")
    scaffold(repo)
    index = (repo / MAP_RELATIVE / INDEX_FILE).read_text(encoding="utf-8")
    assert index == ("# Map index\n"
                     "\n## alpha\n"
                     "- alpha (inbound 2): (unmapped) [stale]\n"
                     "- alpha.nested (inbound 0): (unmapped) [stale]\n"
                     "\n## beta\n"
                     "- beta (inbound 1): (unmapped) [stale]\n"
                     "\n## unpackaged\n"
                     "- scripts/\n")

    # a Purpose someone wrote shows up, and stops being stale
    path = repo / MAP_RELATIVE / "beta.md"
    write(path, path.read_text(encoding="utf-8").replace(
        "## Purpose", "## Purpose\n\nJSON helpers for alpha.\nA second line nobody sees here.", 1))
    scaffold(repo)
    index = (repo / MAP_RELATIVE / INDEX_FILE).read_text(encoding="utf-8")
    assert "- beta (inbound 1): JSON helpers for alpha.\n" in index
    assert "A second line" not in index and "beta (inbound 1): JSON helpers for alpha. [stale]" not in index


def test_section_reads_one_heading_at_a_time():
    body = "## Purpose\n\nWhat it is for.\n\n## Gotchas\n\nNone yet.\n"
    assert section(body, "Purpose") == "What it is for."
    assert section(body, "Gotchas") == "None yet."
    assert section(body, "Invariants") == ""
    assert section("## Purpose\n\n## Gotchas\n", "Purpose") == ""


# ---- the command line ------------------------------------------------------


def test_map_scaffold_from_the_cli(tmp_path, capsys):
    repo = sample_project(tmp_path / "p")
    assert main(["map", "scaffold", "--project", str(repo)]) == 0
    out = capsys.readouterr()
    assert f"wrote      {MAP}/alpha.md" in out.out and f"wrote      {MAP}/{INDEX_FILE}" in out.out
    assert "commit before running tasks." in out.err

    assert main(["map", "scaffold", "--project", str(repo), "--package", "alpha"]) == 0
    out = capsys.readouterr().out
    assert f"{MAP}/alpha.md" in out and f"{MAP}/alpha.nested.md" in out
    assert f"{MAP}/beta.md" not in out                      # not selected
    assert f"{MAP}/{INDEX_FILE}" in out                     # the index is always whole
    assert "- beta (inbound 1)" in (repo / MAP_RELATIVE / INDEX_FILE).read_text(encoding="utf-8")


def test_map_scaffold_refuses_an_unknown_package_and_a_non_repository(tmp_path, capsys):
    repo = sample_project(tmp_path / "p")
    assert main(["map", "scaffold", "--project", str(repo), "--package", "gamma"]) == 64
    assert "no package 'gamma'" in capsys.readouterr().err
    plain = tmp_path / "plain"
    plain.mkdir()
    assert main(["map", "scaffold", "--project", str(plain)]) == 64
    assert "not a git repository" in capsys.readouterr().err


def test_the_map_skips_worktrees_and_its_own_directory(tmp_path):
    """A run's worktree under the project is another checkout of it, not more
    code to map; docs/ is where the map itself lives."""
    repo = sample_project(tmp_path / "p")
    git(repo, "worktree", "add", str(repo / "wt"), "-b", "task/x")
    write(repo / "docs" / "helper.py", "def doc_helper():\n    return 1\n")
    project = scan(repo)
    assert project.names() == ["alpha", "alpha.nested", "beta"]
    assert all("wt/" not in rel for package in project.packages.values() for rel in package.files)
    assert project.unpackaged == ["scripts/"]


# ---- staleness -------------------------------------------------------------


def fill_purpose(repo: Path, package: str, text: str = "Does the work.") -> Path:
    path = repo / MAP_RELATIVE / f"{package}.md"
    return write(path, path.read_text(encoding="utf-8").replace(
        "## Purpose", f"## Purpose\n\n{text}", 1))


def mapped_project(tmp_path) -> Path:
    """A project whose map is scaffolded, written and committed: clean."""
    repo = sample_project(tmp_path / "p")
    scaffold(repo)
    for package in ("alpha", "alpha.nested", "beta"):
        fill_purpose(repo, package, f"What {package} is for.")
    commit_all(repo, "map the project")
    return repo


def test_a_committed_map_is_clean_and_a_committed_change_makes_it_stale(tmp_path):
    repo = mapped_project(tmp_path)
    assert stale(repo) == []

    core = repo / "alpha" / "core.py"
    write(core, core.read_text(encoding="utf-8") + "\n\ndef stop():\n    return None\n")
    assert stale(repo) == []                    # HEAD is the question, not the editor
    commit_all(repo, "add stop()")
    assert stale(repo) == [{"package": "alpha",
                            "reasons": ["changed since the map was written: alpha/core.py"]}]

    scaffold(repo, "alpha")
    commit_all(repo, "refresh the map")
    assert stale(repo) == []


def test_a_new_file_a_removed_file_a_missing_map_and_an_empty_purpose(tmp_path):
    repo = mapped_project(tmp_path)
    write(repo / "alpha" / "extra.py", "def extra():\n    return 1\n")
    (repo / "alpha" / "nested" / "deep.py").unlink()
    (repo / MAP_RELATIVE / "beta.md").unlink()
    commit_all(repo, "move things around")

    report = {entry["package"]: entry["reasons"] for entry in stale(repo)}
    assert report["alpha"] == ["not in the map: alpha/extra.py"]
    assert report["alpha.nested"] == ["in the map but gone: alpha/nested/deep.py"]
    assert report["beta"] == ["no map file"]

    scaffold(repo)
    commit_all(repo, "refresh")
    assert [e["package"] for e in stale(repo)] == ["beta"]          # the new map has no Purpose
    assert stale(repo)[0]["reasons"] == ["no Purpose"]


def test_area_restricts_and_an_unknown_area_says_so(tmp_path):
    repo = mapped_project(tmp_path)
    (repo / MAP_RELATIVE / "beta.md").unlink()
    write(repo / "alpha" / "nested" / "extra.py", "x = 1\n")
    commit_all(repo, "break two of them")

    assert [e["package"] for e in stale(repo)] == ["alpha.nested", "beta"]
    assert [e["package"] for e in stale(repo, ["alpha"])] == ["alpha.nested"]   # subpackages count
    assert stale(repo, ["beta"]) == [{"package": "beta", "reasons": ["no map file"]}]
    assert stale(repo, ["gamma"]) == [{"package": "gamma",
                                       "reasons": ["not a package in this project"]}]


def test_map_stale_from_the_cli_with_json_and_exit_codes(tmp_path, capsys):
    repo = mapped_project(tmp_path)
    assert main(["map", "stale", "--project", str(repo)]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "map is clean" and "3 package(s)" in out.err

    core = repo / "alpha" / "core.py"
    write(core, core.read_text(encoding="utf-8") + "\n\ndef stop():\n    return None\n")
    commit_all(repo, "add stop()")

    assert main(["map", "stale", "--project", str(repo)]) == 1
    assert "alpha  changed since the map was written: alpha/core.py" in capsys.readouterr().out

    assert main(["map", "stale", "--project", str(repo), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report == {"clean": False, "stale": [
        {"package": "alpha", "reasons": ["changed since the map was written: alpha/core.py"]}]}

    assert main(["map", "stale", "--project", str(repo), "--area", "beta", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"clean": True, "stale": []}


def test_stale_against_tree_needs_no_git_at_all(tmp_path):
    repo = mapped_project(tmp_path)
    tree = tmp_path / "copy"                    # a plain directory: no repository anywhere
    shutil.copytree(repo, tree, ignore=shutil.ignore_patterns(".git"))
    assert not (tree / ".git").exists()

    assert stale_against_tree(tree) == []
    write(tree / "beta" / "util.py", "import json\n\n\ndef helper(sep):\n    return json.dumps(sep)\n")
    assert stale_against_tree(tree) == [
        {"package": "beta", "reasons": ["changed since the map was written: beta/util.py"]}]
    assert stale_against_tree(tree, ["alpha"]) == []

    # the map may live in the project while the code being judged is a worktree
    assert stale_against_tree(repo, ["beta"], tree_root=tree) == [
        {"package": "beta", "reasons": ["changed since the map was written: beta/util.py"]}]
    assert stale_against_tree(repo, ["beta"], tree_root=repo) == []


def test_a_package_that_was_never_committed_says_that(tmp_path):
    repo = mapped_project(tmp_path)
    write(repo / "gamma" / "__init__.py", "")
    write(repo / "gamma" / "thing.py", "def thing():\n    return 1\n")
    scaffold(repo)
    fill_purpose(repo, "gamma", "Brand new.")
    assert stale(repo, ["gamma"]) == [
        {"package": "gamma", "reasons": ["no files for this package in HEAD; nothing is committed yet"]}]
    commit_all(repo, "add gamma")
    assert stale(repo, ["gamma"]) == []


def test_map_stale_on_a_repository_with_no_commits_is_a_usage_error(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-b", "main")
    write(empty / "pkg" / "__init__.py", "")
    assert main(["map", "stale", "--project", str(empty)]) == 64
    assert "map: git ls-tree" in capsys.readouterr().err
