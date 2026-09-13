"""The system prompt: a base, a global file, per-run appends, and provenance.

Every test here points HOME, XDG_CONFIG_HOME and HARNESS_SYSTEM_PROMPT at
temporary directories. Nothing reads or writes the real home directory.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.cli import main
from harness.mockserver import MockOpenAIServer
from harness.runtime import META_NAMES
from harness.prompts import (GLOBAL_ENV, SYSTEM_PROMPT, PromptError, global_prompt_path,
                             resolve_system_prompt)
from harness.trajectory import format_summary, format_trace, read_trajectory, summarize
from harness.transport import call

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILTIN = SYSTEM_PROMPT.strip()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """A throwaway home, and no global prompt unless a test writes one."""
    home = tmp_path / "home"
    (home / ".config" / "harness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(GLOBAL_ENV, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    return home


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def plan_and_finish(status="in_progress"):
    return call("todo_write", {"todos": [{"id": "1", "content": "answer", "status": status}]})


FINISH = call("final_answer", {"status": "completed", "content": "ok"})


def test_the_builtin_prompt_stays_within_its_budget_and_teaches_the_mechanics():
    """Local models pay for every token of this on every single request, so it
    has a hard ceiling - and it still has to name every lever the loop has."""
    assert len(SYSTEM_PROMPT) < 3500, f"built-in prompt is {len(SYSTEM_PROMPT)} chars"
    for name in META_NAMES:
        assert name in SYSTEM_PROMPT, f"{name} is not explained to the model"
    for topic in ("scratch_write", "denied", "in_progress", "fs_edit", "run_shell", "evicted"):
        assert topic in SYSTEM_PROMPT, f"{topic} is not explained to the model"


def test_the_example_global_prompt_loads_as_house_rules(isolated_home):
    """examples/global-system.md is shipped to be dropped in as a global prompt."""
    example = (REPO_ROOT / "examples" / "global-system.md").read_text(encoding="utf-8")
    write(isolated_home / ".config" / "harness" / "system.md", example)

    text, sources = resolve_system_prompt()
    assert text == f"{BUILTIN}\n\n{example.strip()}"
    assert "Never run a destructive shell command" in text
    assert [s["role"] for s in sources] == ["base", "global"]
    assert sources[1]["chars"] == len(example.strip())


def test_prompt_precedence_is_flag_then_env_then_xdg_then_home(tmp_path, monkeypatch, isolated_home):
    home_file = write(isolated_home / ".config" / "harness" / "system.md", "HOME RULES")
    xdg_file = write(tmp_path / "xdg" / "harness" / "system.md", "XDG RULES")
    env_file = write(tmp_path / "named.md", "ENV RULES")
    base_file = write(tmp_path / "base.md", "BASE PROMPT")

    text, sources = resolve_system_prompt()
    assert text == f"{BUILTIN}\n\nHOME RULES"
    assert sources == [{"source": "builtin", "role": "base", "chars": len(BUILTIN)},
                       {"source": str(home_file), "role": "global", "chars": 10}]

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    text, sources = resolve_system_prompt()
    assert text == f"{BUILTIN}\n\nXDG RULES" and sources[1]["source"] == str(xdg_file)

    monkeypatch.setenv(GLOBAL_ENV, str(env_file))
    text, sources = resolve_system_prompt()
    assert text == f"{BUILTIN}\n\nENV RULES" and sources[1]["source"] == str(env_file)

    # the flag replaces the base entirely; the global still lands on top of it
    text, sources = resolve_system_prompt(base_file)
    assert text == "BASE PROMPT\n\nENV RULES"
    assert sources == [{"source": str(base_file), "role": "base", "chars": 11},
                       {"source": str(env_file), "role": "global", "chars": 9}]

    # only the first location that exists is used, so a stale env var falls through
    monkeypatch.setenv(GLOBAL_ENV, str(tmp_path / "gone.md"))
    assert global_prompt_path() == xdg_file
    assert resolve_system_prompt()[0].endswith("XDG RULES")


def test_appends_follow_the_global_prompt_in_order(tmp_path, isolated_home):
    write(isolated_home / ".config" / "harness" / "system.md", "HOUSE\n")
    one, two = write(tmp_path / "one.md", "ONE"), write(tmp_path / "two.md", "TWO\n\n")

    text, sources = resolve_system_prompt(appends=[one, two])
    assert text == "\n\n".join([BUILTIN, "HOUSE", "ONE", "TWO"])
    assert [s["role"] for s in sources] == ["base", "global", "append", "append"]
    assert [s["source"] for s in sources[2:]] == [str(one), str(two)]

    text, sources = resolve_system_prompt(appends=[two, one])
    assert text == "\n\n".join([BUILTIN, "HOUSE", "TWO", "ONE"])

    # --no-global-prompt skips a file that is there and readable
    text, sources = resolve_system_prompt(appends=[one], use_global=False)
    assert text == f"{BUILTIN}\n\nONE"
    assert [s["role"] for s in sources] == ["base", "append"]
    assert global_prompt_path() is not None


def test_a_prompt_file_that_cannot_be_read_is_a_usage_error(tmp_path, capsys):
    missing = tmp_path / "nope.md"
    with pytest.raises(PromptError, match="cannot read prompt file"):
        resolve_system_prompt(missing)
    with pytest.raises(PromptError, match="cannot read prompt file"):
        resolve_system_prompt(appends=[missing])

    assert main(["prompt", "--system-prompt", str(missing)]) == 64
    assert "cannot read prompt file" in capsys.readouterr().err

    rc = main(["--runs-dir", str(tmp_path / "runs"), "run", "--task", "t", "--model", "m",
               "--endpoint", "http://127.0.0.1:1/v1", "--workdir", str(tmp_path / "work"),
               "--append-system-prompt", str(missing)])
    assert rc == 64 and "cannot read prompt file" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()          # the run never started


def test_prompt_subcommand_prints_the_prompt_or_where_it_came_from(tmp_path, capsys, isolated_home):
    global_file = write(isolated_home / ".config" / "harness" / "system.md", "HOUSE RULES")
    extra = write(tmp_path / "extra.md", "EXTRA")

    assert main(["prompt"]) == 0
    out = capsys.readouterr().out
    assert out.rstrip("\n") == f"{BUILTIN}\n\nHOUSE RULES"

    assert main(["prompt", "--append-system-prompt", str(extra), "--sources"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows[0] == ["base", "builtin", str(len(BUILTIN)), "chars"]
    assert rows[1] == ["global", str(global_file), "11", "chars"]
    assert rows[2] == ["append", str(extra), "5", "chars"]
    assert rows[3] == ["total:", str(len(BUILTIN) + len("HOUSE RULES") + len("EXTRA") + 4), "chars"]

    assert main(["prompt", "--no-global-prompt"]) == 0
    assert capsys.readouterr().out.rstrip("\n") == BUILTIN


def test_cli_run_sends_the_layered_prompt_and_records_its_provenance(tmp_path, isolated_home, monkeypatch):
    """A real process: the global prompt is found through the environment, the
    layers reach the endpoint as one system message, and the header says so."""
    xdg = tmp_path / "xdg"
    global_file = write(xdg / "harness" / "system.md", "HOUSE: never run destructive commands.")
    extra = write(tmp_path / "extra.md", "RUN: answer in one sentence.")
    runs, work = tmp_path / "runs", tmp_path / "project"
    work.mkdir()

    script = [{"content": "Planning.", "tool_calls": [plan_and_finish("completed")]},
              {"content": "Done.", "tool_calls": [FINISH]}]
    with MockOpenAIServer(script, model="mock-model") as server:
        proc = subprocess.run(
            [sys.executable, "-m", "harness", "--runs-dir", str(runs), "run", "--task", "t",
             "--model", "mock-model", "--endpoint", server.base_url, "--workdir", str(work),
             "--run-id", "layered", "--append-system-prompt", str(extra)],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT), HOME=str(isolated_home),
                     XDG_CONFIG_HOME=str(xdg)))
        sent = [r["body"]["messages"][0] for r in server.requests]

    assert proc.returncode == 0, proc.stderr
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == "\n\n".join(
        [BUILTIN, "HOUSE: never run destructive commands.", "RUN: answer in one sentence."])
    assert all(m == sent[0] for m in sent)           # the same system message every turn

    records = read_trajectory(runs / "layered")
    assert records[0]["prompt_sources"] == [
        {"source": "builtin", "role": "base", "chars": len(BUILTIN)},
        {"source": str(global_file), "role": "global", "chars": 38},
        {"source": str(extra), "role": "append", "chars": 28}]
    assert summarize(records)["prompt_sources"] == records[0]["prompt_sources"]

    line = next(l for l in format_trace(records, runs / "layered").splitlines() if l.startswith("prompt:"))
    assert "builtin [base] " in line and f"{global_file} [global] 38 chars" in line
    assert f"{extra} [append] 28 chars" in line
    assert line in format_summary(records)


def test_resume_keeps_the_recorded_prompt_and_refuses_prompt_flags(tmp_path, capsys, isolated_home):
    global_file = write(isolated_home / ".config" / "harness" / "system.md", "HOUSE RULES v1")
    runs, work = tmp_path / "runs", tmp_path / "project"
    work.mkdir()

    with MockOpenAIServer([{"content": "Planning.", "tool_calls": [plan_and_finish()]}],
                          model="mock-model") as server:
        rc = main(["--runs-dir", str(runs), "run", "--task", "t", "--model", "mock-model",
                   "--endpoint", server.base_url, "--workdir", str(work), "--run-id", "pr",
                   "--step-cap", "1"])
        recorded = server.requests[0]["body"]["messages"][0]["content"]
    assert rc == 3 and recorded.endswith("HOUSE RULES v1")

    # the house rules change under the run; a resumed run must not pick them up
    global_file.write_text("HOUSE RULES v2", encoding="utf-8")
    rest = [{"content": "Closing.", "tool_calls": [plan_and_finish("completed")]},
            {"content": "Done.", "tool_calls": [FINISH]}]
    with MockOpenAIServer(rest, model="mock-model") as server:
        rc = main(["--runs-dir", str(runs), "resume", "pr", "--endpoint", server.base_url,
                   "--workdir", str(work), "--step-cap", "6"])
        resumed = server.requests[0]["body"]["messages"][0]["content"]
    assert rc == 0 and resumed == recorded

    for flag in (["--no-global-prompt"], ["--system-prompt", str(global_file)],
                 ["--append-system-prompt", str(global_file)]):
        rc = main(["--runs-dir", str(runs), "resume", "pr", "--endpoint", "http://127.0.0.1:1/v1",
                   *flag])
        assert rc == 64
        err = capsys.readouterr().err
        assert f"resume: {flag[0]} cannot be used on resume" in err
