"""The project and task prompt layers, and the skills bundled with the harness.

A project run is told two things no other run is told: how this repository is
worked on, with the head of its code map index, and what this task calls
finished. Both are layers of the system prompt, both are recorded as sources,
and the first one has a budget - the index of a large repository is not
something to paste into every request.

The three bundled skills (orient, map, handoff) are the other half of that:
they are the guides those instructions point at, and they exist only for a
project run.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from harness.cli import main
from harness.codemap import INDEX_FILE, MAP_RELATIVE, scaffold
from harness.mockserver import MockOpenAIServer
from harness.prompts import (INDEX_REST, PROJECT_UNMAPPED, TASK_LAYER, index_excerpt,
                             project_layer, resolve_system_prompt, task_layer)
from harness.skills import BUNDLED, bundled_dir, discover, search_dirs
from harness.tasks import load as load_task
from harness.trajectory import read_trajectory
from harness.transport import call

from .gitfixture import commit_all, git_repo, sample_project, write

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "sekret"
BUNDLED_NAMES = ["handoff", "map", "orient"]


def wide_project(path: Path) -> Path:
    """A repository with 25 packages, each importing every package below it: so
    pkg00 is the most depended-on and pkg24 the least, in one straight line."""
    files = {}
    for n in range(25):
        files[f"pkg{n:02d}/__init__.py"] = ""
        imports = "".join(f"import pkg{m:02d}\n" for m in range(n))
        files[f"pkg{n:02d}/mod.py"] = imports + f"\n\ndef thing{n:02d}():\n    return {n}\n"
    files["tasks/edge.md"] = ("---\nstatus: todo\narea: [pkg24]\n"
                              "done_when: [pkg24 does the thing, the map for pkg24 is current]\n"
                              "branch: null\nrun_id: null\n---\n\nTeach pkg24 to do the thing.\n")
    repo = git_repo(path, files, message="wide project")
    scaffold(repo)
    for n in range(25):                      # a Purpose apiece, so no line says [stale]
        map_file = repo / MAP_RELATIVE / f"pkg{n:02d}.md"
        write(map_file, map_file.read_text(encoding="utf-8").replace(
            "## Purpose", f"## Purpose\n\nPackage {n:02d} of the wide project.", 1))
    scaffold(repo)
    commit_all(repo, "map the wide project")
    return repo


def packages_in(text: str) -> list[str]:
    """The packages an index excerpt names, in the order it names them."""
    return [line.split()[1] for line in text.splitlines()
            if line.startswith("- pkg") and "(inbound" in line]


# ---- the project layer -----------------------------------------------------


def test_the_layer_quotes_the_top_of_the_index_plus_the_task_area(tmp_path):
    repo = wide_project(tmp_path / "wide")
    task = load_task(repo, "edge")
    text, sources = resolve_system_prompt(project=repo, task=task.prompt_meta())

    layer = next(s for s in sources if s["role"] == "project")
    assert layer["source"] == str(repo / "docs" / "map" / INDEX_FILE)
    assert [s["role"] for s in sources] == ["base", "project", "task"]
    assert sum(s["chars"] for s in sources) < len(text)      # joined with blank lines

    # the twenty most depended-on, in index order, and pkg24 because the task is about it
    quoted, _ = project_layer(repo, task.area)
    assert quoted in text and len(quoted) == layer["chars"]
    assert packages_in(quoted) == [f"pkg{n:02d}" for n in range(20)] + ["pkg24"]
    assert "- pkg00 (inbound 24): Package 00 of the wide project." in quoted
    assert "pkg20" not in quoted and "pkg23" not in quoted
    assert quoted.rstrip().endswith(INDEX_REST)
    for instruction in ("docs/map/index.md first", "load the `map` skill",
                        "before final_answer", "harness commits your work"):
        assert instruction in quoted.lower()

    # the excerpt is the index's own lines, not a paraphrase of them
    index = (repo / MAP_RELATIVE / INDEX_FILE).read_text(encoding="utf-8")
    assert index_excerpt(index, ["pkg24"]) == [line for line in text.splitlines()
                                               if line.startswith("- pkg") and "(inbound" in line]


def test_the_layer_is_capped_by_dropping_excerpt_lines_from_the_bottom(tmp_path):
    repo = wide_project(tmp_path / "wide")
    full, _ = project_layer(repo)
    assert len(full) > 1200

    small, _ = project_layer(repo, limit=1200)
    assert len(small) <= 1200
    assert packages_in(small) == packages_in(full)[:len(packages_in(small))]
    assert small.rstrip().endswith(INDEX_REST)               # the pointer survives the cut

    # ... and the instructions survive even a cap nothing could fit under
    tiny, _ = project_layer(repo, limit=10)
    assert packages_in(tiny) == [] and tiny.rstrip().endswith(INDEX_REST)


def test_a_project_with_no_map_is_told_to_make_one(tmp_path):
    repo = sample_project(tmp_path / "p")                    # no docs/map/ at all
    text, source = project_layer(repo)
    assert text == PROJECT_UNMAPPED
    assert source == str(repo / "docs" / "map" / INDEX_FILE)
    assert "has no code map" in text and "load the `map` skill" in text.lower()
    assert "INDEX.md has not been scaffolded" in text


def test_the_task_layer_is_the_checks_the_run_has_to_show(tmp_path):
    repo = sample_project(tmp_path / "p")
    task = load_task(repo, "first")
    text, source = task_layer(task.prompt_meta())
    assert text == (f"{TASK_LAYER}\n- alpha starts the engine\n- the map for alpha is current")
    assert source == str(task.path)

    # a task that says nothing about finishing adds no layer
    write(task.path, "---\nstatus: todo\narea: [alpha]\n---\n\nJust do it.\n")
    assert task_layer(load_task(repo, "first").prompt_meta()) is None
    _, sources = resolve_system_prompt(project=repo, task=load_task(repo, "first").prompt_meta())
    assert [s["role"] for s in sources] == ["base", "project"]


def test_harness_prompt_shows_the_project_and_task_layers(tmp_path, capsys):
    repo = wide_project(tmp_path / "wide")
    assert main(["prompt", "--project", str(repo), "--task-id", "edge"]) == 0
    out = capsys.readouterr().out
    assert "- pkg00 (inbound 24)" in out and INDEX_REST in out
    assert f"{TASK_LAYER}\n- pkg24 does the thing" in out
    assert "SKILLS." in out                                  # a project run has the bundled ones

    assert main(["prompt", "--project", str(repo), "--task-id", "edge", "--sources"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines[:3]] == ["base", "project", "task"]
    assert str(repo / "docs" / "map" / INDEX_FILE) in lines[1]
    assert str(repo / "tasks" / "edge.md") in lines[2]

    assert main(["prompt", "--task-id", "edge"]) == 64
    assert "--task-id needs --project" in capsys.readouterr().err
    assert main(["prompt", "--project", str(repo), "--task-id", "nope"]) == 64
    assert "no task 'nope'" in capsys.readouterr().err


def test_the_layers_reach_the_model_on_a_project_run(tmp_path):
    repo = wide_project(tmp_path / "wide")
    runs = tmp_path / "runs"
    script = [{"content": "Planning.", "tool_calls": [call("todo_write", {"todos": [
                  {"id": "a", "content": "do it", "status": "completed"}]}, id="t")]},
              {"content": "Reporting.", "tool_calls": [call("final_answer", {
                  "status": "completed", "content": "done"}, id="f")]}]
    with MockOpenAIServer(script, model="mock-model", api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "run", "--model", "mock-model",
                   "--endpoint", server.base_url, "--api-key", SECRET,
                   "--project", str(repo), "--task-id", "edge", "--run-id", "layers",
                   "--project-prompt-chars", "1500"])
        system = server.requests[0]["body"]["messages"][0]["content"]
    assert rc == 0

    assert "PROJECT. You are working in a git worktree" in system
    assert "- pkg00 (inbound 24)" in system and INDEX_REST in system
    assert f"{TASK_LAYER}\n- pkg24 does the thing" in system
    header = read_trajectory(runs / "layers")[0]
    roles = [s["role"] for s in header["prompt_sources"]]
    assert roles == ["base", "project", "task"]
    project = next(s for s in header["prompt_sources"] if s["role"] == "project")
    assert project["chars"] <= 1500                          # the cap the run was given


# ---- the bundled skills ----------------------------------------------------


def test_the_bundled_skills_ship_with_the_harness(tmp_path):
    found = discover([bundled_dir()])
    assert found.names() == BUNDLED_NAMES
    assert [found.source_of(found.get(name)) for name in found.names()] == [BUNDLED] * 3
    assert found.describe() == {"dirs": [BUNDLED], "names": BUNDLED_NAMES}
    assert found.get("orient").tools == ["fs_read", "fs_search", "fs_glob"]
    assert found.get("map").tools == ["fs_read", "fs_search", "fs_edit", "fs_write"]
    assert found.get("handoff").tools == []
    for name in BUNDLED_NAMES:
        assert len(found.get(name).summary()) <= 160
        assert found.get(name).chars < 12000                 # loadable under --skill-chars


def test_they_are_searched_only_for_a_project_run(tmp_path):
    assert search_dirs([], tmp_path) == []
    assert search_dirs([], tmp_path, bundled=True) == [bundled_dir()]
    # ... and last in precedence, so a project's own `map` skill wins
    own = tmp_path / "skills" / "map"
    write(own / "SKILL.md", "---\ndescription: The project's own map skill.\n---\n\nours\n")
    found = discover(search_dirs([tmp_path / "skills"], tmp_path, bundled=True))
    assert found.get("map").path == own / "SKILL.md"
    assert "shadowing" in " ".join(found.warnings())


def test_harness_skills_shows_them_for_a_project_and_not_otherwise(tmp_path, capsys):
    repo = sample_project(tmp_path / "p")
    assert main(["skills", "--workdir", str(repo)]) == 0
    out = capsys.readouterr()
    assert "orient" not in out.out and "0 skill(s)" in out.err

    assert main(["skills", "--project", str(repo)]) == 0
    out = capsys.readouterr()
    assert "orient   Find your way around a mapped repository" in out.out
    assert f"({BUNDLED})" in out.out and "3 skill(s)" in out.err
    assert out.err.rstrip().endswith(f"  {BUNDLED}")         # the location, not the install path


def test_a_project_run_can_load_a_bundled_skill_and_a_resume_still_can(tmp_path):
    repo = sample_project(tmp_path / "p")
    runs = tmp_path / "runs"
    script = [
        {"content": "What guides are there?", "tool_calls": [call("skill_list", {}, id="sl")]},
        {"content": "Loading the map guide.",
         "tool_calls": [call("skill_load", {"name": "map"}, id="load")]},
        {"content": "Planning.", "tool_calls": [call("todo_write", {"todos": [
            {"id": "a", "content": "map it", "status": "in_progress"}]}, id="t")]},
    ]
    with MockOpenAIServer(script, model="mock-model", api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "map the project",
                   "--model", "mock-model", "--endpoint", server.base_url, "--api-key", SECRET,
                   "--project", str(repo), "--run-id", "bundled", "--step-cap", "3"])
        tools = [t["function"]["name"] for t in server.requests[-1]["body"]["tools"]]
    assert rc == 3

    records = read_trajectory(runs / "bundled")
    header = records[0]
    assert header["skills"] == {"dirs": [BUNDLED], "names": BUNDLED_NAMES}
    assert header["invocation"]["skills"] == [BUNDLED]       # no install path in the record
    steps = [r for r in records if r["type"] == "step"]
    assert [s["tool"] for s in steps] == ["skill_list", "skill_load", "todo_write"]
    listed = json.loads((runs / "bundled" / steps[0]["artifact"]).read_text())
    assert [s["name"] for s in listed] == BUNDLED_NAMES
    assert "Map one package" in (runs / "bundled" / steps[1]["artifact"]).read_text()
    assert {"fs_read", "fs_search", "fs_edit", "fs_write"} <= set(tools)   # the skill's tools

    rest = [{"content": "Closing.", "tool_calls": [call("todo_write", {"todos": [
                {"id": "a", "content": "map it", "status": "completed"}]}, id="t2")]},
            {"content": "Reporting.", "tool_calls": [call("final_answer", {
                "status": "completed", "content": "mapped"}, id="f")]}]
    with MockOpenAIServer(rest, model="mock-model", api_key=SECRET) as server:
        rc = main(["--runs-dir", str(runs), "resume", "bundled", "--endpoint", server.base_url,
                   "--api-key", SECRET, "--step-cap", "20"])
        tools = [t["function"]["name"] for t in server.requests[0]["body"]["tools"]]
    assert rc == 0
    assert "skill_load" in tools                             # still a run that has skills
    seam = next(r for r in read_trajectory(runs / "bundled") if r["type"] == "resume")
    assert seam["invocation"]["skills"] == [BUNDLED]


def test_the_packaged_cli_finds_the_bundled_skills_from_anywhere(tmp_path):
    """Installed or not, they come from inside the package - so a run started
    in a directory that knows nothing about the harness still has them."""
    repo = sample_project(tmp_path / "p")
    proc = subprocess.run([sys.executable, "-m", "harness", "skills", "--project", str(repo)],
                          cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
                          env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    assert proc.returncode == 0, proc.stderr
    assert [line.split()[0] for line in proc.stdout.splitlines()] == BUNDLED_NAMES
    assert proc.stdout.count(f"({BUNDLED})") == 3
