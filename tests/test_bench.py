"""harness bench: a file of tasks, run back to back under one set of flags."""
import json

from harness.cli import main
from harness.mockserver import MockOpenAIServer
from harness.trajectory import read_trajectory
from harness.transport import call

DONE = call("final_answer", {"status": "completed", "content": "read it"})


def tasks_file(tmp_path, entries):
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return str(path)


def test_bench_runs_every_task_and_tabulates_them(tmp_path, capsys):
    runs = tmp_path / "runs"
    alpha, beta = tmp_path / "alpha", tmp_path / "beta"
    alpha.mkdir(), beta.mkdir()
    (alpha / "note.txt").write_text("alpha note\n", encoding="utf-8")

    script = [
        # task one: four steps, finishes
        {"content": "Activating fs_read.", "tool_calls": [call("toolbelt_add", {"names": ["fs_read"]})],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"content": "Planning and reading in one turn.",
         "tool_calls": [call("todo_write", {"todos": [{"id": "1", "content": "read", "status": "completed"}]}),
                        call("fs_read", {"path": "note.txt"})],
         "usage": {"prompt_tokens": 200, "completion_tokens": 20}},
        {"content": "Finishing.", "tool_calls": [DONE], "usage": {"prompt_tokens": 300, "completion_tokens": 30}},
        # task two: never finishes, trips the shared step cap of 5
        {"content": "Looking.", "tool_calls": [call("toolbelt_list", {}), call("toolbelt_list", {})],
         "usage": {"prompt_tokens": 400, "completion_tokens": 40}},
        {"content": "Looking again.", "tool_calls": [call("toolbelt_list", {}), call("toolbelt_list", {})],
         "usage": {"prompt_tokens": 500, "completion_tokens": 50}},
        {"content": "And again.", "tool_calls": [call("toolbelt_list", {}), call("toolbelt_list", {})],
         "usage": {"prompt_tokens": 600, "completion_tokens": 60}},
    ]
    path = tasks_file(tmp_path, [
        {"task": "read the note", "workdir": str(alpha), "run_id": "alpha-run"},
        {"task": "wander about", "workdir": str(beta), "run_id": "beta-run"},
    ])

    with MockOpenAIServer(script, model="mock-model") as server:
        rc = main(["--runs-dir", str(runs), "bench", path, "--model", "mock-model",
                   "--endpoint", server.base_url, "--step-cap", "5"])
    out = capsys.readouterr()
    assert rc == 1, out.out                      # one task did not complete

    rows = [line.split() for line in out.out.splitlines() if line.strip()]
    assert rows[0] == ["run", "id", "status", "steps", "tokens", "in", "tokens", "out", "elapsed"]
    assert rows[1][:5] == ["alpha-run", "completed", "4", "600", "60"]
    assert rows[2][:5] == ["beta-run", "step_cap", "5", "1500", "150"]
    assert rows[-1][:4] == ["2", "task(s):", "1", "completed,"]
    assert "[2/2] beta-run" in out.err

    # each task ran in its own workdir, with its own run directory
    steps = [s for s in read_trajectory(runs / "alpha-run") if s["type"] == "step"]
    assert [(s["tool"], s["kind"]) for s in steps][-1] == ("final_answer", "final_accepted")
    assert "alpha note" in steps[2]["result_preview"]
    assert read_trajectory(runs / "beta-run")[-1]["status"] == "step_cap"

    log = [json.loads(line) for line in (runs / "bench.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == 2
    assert [r["run_id"] for r in log] == ["alpha-run", "beta-run"]
    assert [r["status"] for r in log] == ["completed", "step_cap"]
    assert [r["steps"] for r in log] == [4, 5]
    assert [(r["tokens_in"], r["tokens_out"]) for r in log] == [(600, 60), (1500, 150)]
    assert log[0]["task"] == "read the note" and log[0]["workdir"] == str(alpha)
    assert log[0]["final"] == {"status": "completed", "content": "read it"}
    assert log[1]["final"] is None and log[1]["denied"] == 0

    # a second bench appends to the same log rather than replacing it
    again = tasks_file(tmp_path, [{"task": "read it again", "workdir": str(alpha), "run_id": "gamma-run"}])
    with MockOpenAIServer(script[:3], model="mock-model") as server:
        rc = main(["--runs-dir", str(runs), "bench", again, "--model", "mock-model",
                   "--endpoint", server.base_url, "--step-cap", "5"])
    assert rc == 0
    log = (runs / "bench.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 3 and json.loads(log[-1])["run_id"] == "gamma-run"


def test_bench_rejects_a_tasks_file_it_cannot_use(tmp_path, capsys):
    def bench(path):
        return main(["--runs-dir", str(tmp_path / "runs"), "bench", path, "--model", "m",
                     "--endpoint", "http://127.0.0.1:1/v1", "--workdir", str(tmp_path)])

    assert bench(str(tmp_path / "nope.jsonl")) == 64
    assert "bench: " in capsys.readouterr().err

    assert bench(str(_write(tmp_path, "bad.jsonl", '{"task": "fine"}\n{not json}\n'))) == 64
    assert "bad.jsonl:2: not valid JSON" in capsys.readouterr().err

    assert bench(str(_write(tmp_path, "noTask.jsonl", '{"workdir": "."}\n'))) == 64
    assert 'noTask.jsonl:1: each line needs a non-empty "task" string' in capsys.readouterr().err

    assert bench(str(_write(tmp_path, "empty.jsonl", "\n\n"))) == 64
    assert "empty.jsonl: no tasks" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()          # nothing ran


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path
