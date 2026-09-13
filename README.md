# harness

A minimal ReAct agent runtime with four guarantees the loop enforces, not the prompt:

1. **Lazy toolbelt.** Every tool is registered; none is in context until the agent calls `toolbelt_add`. The first request carries only the six meta-tools.
2. **Todo gate.** `final_answer` is rejected while any todo is `pending` or `in_progress` (or before a todo list exists). Plain assistant text cannot end a run.
3. **Think-before-act.** The assistant's text on each tool-calling turn is captured as that step's `reasoning`. No retry if it's empty. Private reasoning channels are never read.
4. **Trajectory export.** One JSONL per run: header, one record per tool call (discovery and todo calls included), footer. Full tool results go to `artifacts/`; the JSONL carries a preview. `harness trace` reads it back.

Plus a context budget so a local model survives a 60-step run: per-result truncation in context, oldest-result eviction past a total budget, reasoning and todo state never evicted.

Zero dependencies. Python 3.10+. Talks to any OpenAI-compatible `/v1/chat/completions` endpoint (llama.cpp server, vLLM, LM Studio, Ollama, or the real thing).

## Run

```bash
python -m harness run \
  --task "Find the config file and report the port" \
  --model qwen2.5-coder-32b \
  --endpoint http://localhost:8080/v1 \
  --workdir ./project

python -m harness tools                       # what the agent can discover
python -m harness trace <run_id>              # readable trace
python -m harness trace <run_id> --step 7     # one step in full, artifact included
python -m harness trace <run_id> --summary    # status, kinds, per-tool counts, tokens, elapsed
python -m harness replay <run_id> --workdir ./project   # re-run a recording against today's tools
```

`--api-key` or `HARNESS_API_KEY` for hosted endpoints. Runs land in `./runs/<run_id>/` (`--runs-dir` to move). Exit codes for `run`: `0` completed, `1` blocked/failed, `2` transport_error, `3` step_cap, `4` stalled. For `replay`: `0` identical to the recording, `1` drifted. A bad command line is `64`, an unreadable run directory `66`.

Options: `--step-cap 250`, `--result-chars 2000`, `--context-chars 60000`, `--preview-chars 400`, `--no-todo-gate`, `--timeout 120`, `--tools mypkg.tools` (repeatable), `--extra-body '{"temperature": 0}'` (merged into every request; may not set `model`, `messages`, `tools`, `tool_choice`).

## Run directory

```
runs/<run_id>/
  trajectory.jsonl     header / step... / footer
  state.json           persisted RunState (active tools, todos, messages, final)
  artifacts/           step_0007_final_answer.txt — full result per step
```

Step record fields: `step, ts, elapsed_ms, reasoning, tool, args, kind, call_index, result_preview, result_bytes, artifact, tokens_in, tokens_out, todo_snapshot`. `call_index` is the position of the call within its model turn, so turn boundaries survive the round trip (see Replay). `kind` ∈ `ok | error | final_accepted | final_rejected | text_only`. When one turn issues several tool calls, usage is recorded on the first and `null` on the rest — never double-counted. `todo_snapshot` is the state after the step.

## Failure semantics

| Event | Behaviour |
|---|---|
| bad tool args, unknown/inactive tool, tool raises | error string returned as the tool result; loop continues |
| `final_answer` with open todos | rejection naming the open ids; loop continues |
| turn with no tool call | recorded as `text_only`; nudge appended; 3 in a row → `stalled` |
| transport error | one retry, then `transport_error` |
| step cap | `step_cap`, `final` is `null` |

All of it is visible in the trajectory.

## Built-in tools

| tool | what it does |
|---|---|
| `fs_list` | list a directory |
| `fs_read` | read a text file, optional offset/limit |
| `fs_write` | write a text file, creating parents |
| `fs_search` | regex over file contents; name or path glob filter, result cap, line numbers |
| `fs_glob` | find files by glob pattern |
| `fs_edit` | exact-string replace, must be unique unless `replace_all`; returns a unified diff |
| `run_shell` | run a command in the workdir |

All rooted to `--workdir`: a path that resolves outside it is refused, never clamped. `fs_search` and `fs_glob` skip binary files, files over 2 MB, and `.git`/cache/vendor directories. `harness tools [--filter kw]` prints the list the agent would see.

## Add tools

```python
from harness import ToolRegistry

registry = ToolRegistry()

@registry.tool("grep", "Search files for a pattern.",
               {"type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"]})
def grep(pattern: str, path: str = "."):
    ...
    return {"matches": [...]}   # str or JSON-serialisable
```

Return values are serialised for the model; exceptions become error results. Top-level argument types and required keys are validated before the call. The description's first line is what the model sees in `toolbelt_list`, so make it count.

`--tools` loads your own module: a dotted name or a path to a `.py` file, repeatable. The module exports either a `registry` (a `ToolRegistry`, merged in) or a `register(registry)` function; if `register` takes a second parameter it receives the resolved workdir, so plugin tools can be rooted like the built-ins.

```bash
python -m harness tools --tools ./my_tools.py           # did it load?
python -m harness run --tools ./my_tools.py --tools mypkg.extra_tools ...
```

Name collisions with existing tools are an error, not a silent override, and nothing is auto-discovered: the agent gets exactly what the command line asked for. Or drive the runtime directly:

```python
from pathlib import Path
from harness import AgentRuntime, RuntimeConfig, ChatCompletionsTransport

rt = AgentRuntime(registry, ChatCompletionsTransport("http://localhost:8080/v1"),
                  Path("runs"), model="qwen2.5-coder-32b", config=RuntimeConfig(step_cap=100))
result = rt.run("...")
```

## Replay and the mock server

`ReplayTransport` turns a recorded `trajectory.jsonl` back into a script: recorded reasoning and tool calls are served in order, the tools run for real, and `compare()` diffs the new trajectory against the recording on everything the model or the tools decided (timing, ids and artifact bodies excluded).

```python
from harness.replay import compare, replay

result = replay("runs/20240101-120000-abc123", registry)
assert compare("runs/20240101-120000-abc123", result.run_dir) == []
```

Same from the CLI: `harness replay <run_id> --workdir ./project` — exit `0` identical, `1` drifted with the differing steps printed. A replay re-runs side effects, so point it at a pristine workdir; replaying against the directory the recording already edited is itself reported as drift.

`MockOpenAIServer` serves scripted chat completions over real HTTP (stdlib `http.server`), which is how the tests exercise `ChatCompletionsTransport` end to end:

```python
from harness import MockOpenAIServer
from harness.transport import call

with MockOpenAIServer([{"content": "Listing.", "tool_calls": [call("toolbelt_list", {})]}]) as server:
    ...  # --endpoint server.base_url
server.requests      # every request body and header it received
```

Script entries are response dicts (`content`, `tool_calls`, `usage`), raw wire payloads, callables, or `HttpError(status)`; an exhausted script answers 503 rather than hanging.

## Tests

```bash
pip install pytest
python -m pytest -q
```

`tests/test_runtime.py::test_scripted_end_to_end_matches_jsonl_step_for_step` drives a fake transport through list → add → inspect → todo → rejected final → close → accepted final and asserts the JSONL step for step. Use `harness.transport.FakeTransport` the same way to test your own tools without a model.

`tests/test_e2e_http.py` runs `python -m harness run` as a subprocess against the mock server and asserts the exit code, the run directory, the trajectory, the artifacts, and the guarantees on the wire. `tests/test_tools_fs.py` exercises each file tool through the runtime, error paths included; `tests/test_plugins.py` covers `--tools`; `tests/test_replay.py` records a run, replays it, and asserts the trajectories match step for step.

## Layout

```
harness/
  registry.py     ToolRegistry, ToolSpec, validate_args
  runtime.py      AgentRuntime, RuntimeConfig, meta-tools, gate, loop
  todo.py         validation and merge
  context.py      ContextBudget
  transport.py    ChatCompletionsTransport, FakeTransport
  replay.py       ReplayTransport, replay, compare
  mockserver.py   MockOpenAIServer (scripted, stdlib http.server)
  plugins.py      --tools module loading
  trajectory.py   TrajectoryWriter, read_trajectory, format_trace, summarize
  prompts.py      system prompt
  cli.py          run / tools / trace / replay
  tools/basic.py  fs_list, fs_read, fs_write, run_shell (rooted to --workdir)
  tools/search.py fs_search, fs_glob
  tools/edit.py   fs_edit
  tools/paths.py  workdir rooting and the directory skip list
tests/
```
