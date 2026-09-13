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

python -m harness trace <run_id>            # readable trace
python -m harness trace <run_id> --step 7   # one step in full, artifact included
```

`--api-key` or `HARNESS_API_KEY` for hosted endpoints. Runs land in `./runs/<run_id>/` (`--runs-dir` to move). Exit codes: `0` completed, `1` blocked/failed, `2` transport_error, `3` step_cap, `4` stalled.

Options: `--step-cap 250`, `--result-chars 2000`, `--context-chars 60000`, `--preview-chars 400`, `--no-todo-gate`, `--timeout 120`.

## Run directory

```
runs/<run_id>/
  trajectory.jsonl     header / step... / footer
  state.json           persisted RunState (active tools, todos, messages, final)
  artifacts/           step_0007_final_answer.txt — full result per step
```

Step record fields: `step, ts, elapsed_ms, reasoning, tool, args, kind, result_preview, result_bytes, artifact, tokens_in, tokens_out, todo_snapshot`. `kind` ∈ `ok | error | final_accepted | final_rejected | text_only`. When one turn issues several tool calls, usage is recorded on the first and `null` on the rest — never double-counted. `todo_snapshot` is the state after the step.

## Failure semantics

| Event | Behaviour |
|---|---|
| bad tool args, unknown/inactive tool, tool raises | error string returned as the tool result; loop continues |
| `final_answer` with open todos | rejection naming the open ids; loop continues |
| turn with no tool call | recorded as `text_only`; nudge appended; 3 in a row → `stalled` |
| transport error | one retry, then `transport_error` |
| step cap | `step_cap`, `final` is `null` |

All of it is visible in the trajectory.

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

To wire your own registry into the CLI, see `harness/cli.py::_run` — it's ten lines. Or drive the runtime directly:

```python
from pathlib import Path
from harness import AgentRuntime, RuntimeConfig, ChatCompletionsTransport

rt = AgentRuntime(registry, ChatCompletionsTransport("http://localhost:8080/v1"),
                  Path("runs"), model="qwen2.5-coder-32b", config=RuntimeConfig(step_cap=100))
result = rt.run("...")
```

## Tests

```bash
pip install pytest
python -m pytest -q
```

`tests/test_runtime.py::test_scripted_end_to_end_matches_jsonl_step_for_step` drives a fake transport through list → add → inspect → todo → rejected final → close → accepted final and asserts the JSONL step for step. Use `harness.transport.FakeTransport` the same way to test your own tools without a model.

## Layout

```
harness/
  registry.py     ToolRegistry, ToolSpec, validate_args
  runtime.py      AgentRuntime, RuntimeConfig, meta-tools, gate, loop
  todo.py         validation and merge
  context.py      ContextBudget
  transport.py    ChatCompletionsTransport, FakeTransport
  trajectory.py   TrajectoryWriter, read_trajectory, format_trace
  prompts.py      system prompt
  cli.py          run / trace
  tools/basic.py  fs_list, fs_read, fs_write, run_shell (rooted to --workdir)
tests/
```
