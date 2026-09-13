# harness

A minimal ReAct agent runtime with four guarantees the loop enforces, not the prompt:

1. **Lazy toolbelt.** Every tool is registered; none is in context until the agent calls `toolbelt_add`. The first request carries only the six meta-tools.
2. **Todo gate.** `final_answer` is rejected while any todo is `pending` or `in_progress` (or before a todo list exists). Plain assistant text cannot end a run.
3. **Think-before-act.** The assistant's text on each tool-calling turn is captured as that step's `reasoning`. No retry if it's empty. Private reasoning channels are never read.
4. **Trajectory export.** One JSONL per run: header, one record per tool call (discovery and todo calls included), footer. Full tool results go to `artifacts/`; the JSONL carries a preview. `harness trace` reads it back.

Plus a context budget so a local model survives a 60-step run: per-result truncation in context, oldest-result eviction past a total budget, reasoning and todo state never evicted.

Zero dependencies. Python 3.10+. Talks to any OpenAI-compatible `/v1/chat/completions` endpoint (llama.cpp server, vLLM, LM Studio, Ollama, or the real thing), or to the Anthropic Messages API with `--provider anthropic`.

## Run

```bash
python -m harness run \
  --task "Find the config file and report the port" \
  --model qwen2.5-coder-32b \
  --endpoint http://localhost:8080/v1 \
  --workdir ./project

python -m harness resume <run_id> --endpoint http://localhost:8080/v1 --step-cap 400
python -m harness tools                       # what the agent can discover
python -m harness trace <run_id>              # readable trace
python -m harness trace <run_id> --step 7     # one step in full, artifact included
python -m harness trace <run_id> --summary    # status, kinds, per-tool counts, tokens, elapsed
python -m harness replay <run_id> --workdir ./project   # re-run a recording against today's tools
```

`--api-key` or `HARNESS_API_KEY` for hosted endpoints. Runs land in `./runs/<run_id>/` (`--runs-dir` to move). Exit codes for `run` and `resume`: `0` completed, `1` blocked/failed, `2` transport_error, `3` step_cap, `4` stalled. For `replay`: `0` identical to the recording, `1` drifted. A bad command line — including a run that cannot be resumed — is `64`, an unreadable run directory `66`.

Options: `--step-cap 250`, `--result-chars 2000`, `--context-chars 60000`, `--preview-chars 400`, `--no-todo-gate`, `--timeout 120`, `--tools mypkg.tools` (repeatable), `--extra-body '{"temperature": 0}'` (merged into every request; may not set `model`, `messages`, `tools`, `tool_choice`), `--provider openai|anthropic`, `--max-tokens 4096` (anthropic only), `--deny-tool NAME` and `--deny-shell-pattern REGEX` (both repeatable).

## Run directory

```
runs/<run_id>/
  trajectory.jsonl     header / step... / footer  (see Resume for a resumed run)
  state.json           persisted RunState (active tools, todos, messages, final)
  artifacts/           step_0007_final_answer.txt — full result per step
  scratch/             notes the agent wrote with scratch_write
```

Step record fields: `step, ts, elapsed_ms, reasoning, tool, args, kind, call_index, result_preview, result_bytes, artifact, tokens_in, tokens_out, todo_snapshot`. `call_index` is the position of the call within its model turn, so turn boundaries survive the round trip (see Replay). `kind` ∈ `ok | error | denied | final_accepted | final_rejected | text_only`. When one turn issues several tool calls, usage is recorded on the first and `null` on the rest — never double-counted. `todo_snapshot` is the state after the step. The header records the `config` the run started under and the `policy` in force, or `null`.

## Failure semantics

| Event | Behaviour |
|---|---|
| bad tool args, unknown/inactive tool, tool raises | error string returned as the tool result; loop continues |
| `final_answer` with open todos | rejection naming the open ids; loop continues |
| turn with no tool call | recorded as `text_only`; nudge appended; 3 in a row → `stalled` |
| transport error | one retry, then `transport_error` |
| step cap | `step_cap`, `final` is `null` |
| tool call denied by policy | denial string as the tool result, `kind: denied`; loop continues |

All of it is visible in the trajectory. When a turn is cut short — the cap trips between two calls of it, or `final_answer` is accepted with calls queued behind it — the calls that never ran are answered in the persisted transcript with a "not executed" result, so every `tool_call` has a matching tool message and the conversation can be handed back to a provider.

`transport_error`, `step_cap` and `stalled` are interruptions, not answers: those runs can be resumed.

## Resume

```bash
python -m harness resume <run_id> --endpoint http://localhost:8080/v1 [--step-cap 400]
```

Continues a run that ended with `transport_error`, `step_cap` or `stalled`. `completed`, `blocked` and `failed` are answers, not interruptions, and `running` means another process still owns the run: all four are refused with exit `64`. The run keeps its id, its directory and its step counter; `state.json` supplies the conversation, the active tools and the todos, and the trajectory header supplies the config (`--step-cap` raises the cap, and must, if the run is already at it). `--model` defaults to the one the run recorded; `--tools`, `--provider` and the policy flags are given again, since they are not recorded state.

The trajectory is appended to, never replaced:

```
header  step*  footer  [resume  step*  footer]*
```

One footer per segment — each says how that segment ended — with a `resume` record between segments carrying `from_status`, `from_step`, `from_detail`, and the `model`, `step_cap`, `config` and `policy` the next segment runs under. The last footer is the run's answer; earlier ones are history. `read_trajectory` returns the lot; `summarize` totals every segment and reads its status from the last footer (plus `segments` and `resumed_from`, and wall time summed per segment so the hours a run sat waiting are not counted); `format_trace` prints the seam where it happened; `ReplayTransport` reads a resumed recording as one straight run under the configuration the last segment ended with, so a raised cap replays as a raised cap.

Resuming appends one user message saying what happened, which also leaves the transcript ending on a user turn however the segment died. From Python:

```python
from harness.resume import resume
result = resume("runs/20240101-120000-abc123", registry, transport, step_cap=400)
```

## Built-in tools

| tool | what it does |
|---|---|
| `fs_list` | list a directory |
| `fs_read` | read a text file, optional offset/limit |
| `fs_write` | write a text file, creating parents |
| `fs_search` | regex over file contents; name or path glob filter, result cap, line numbers |
| `fs_glob` | find files by glob pattern |
| `fs_edit` | exact-string replace, must be unique unless `replace_all`; returns a unified diff |
| `run_shell` | run a command in the workdir; a non-zero exit is a result, a timeout is an error |
| `scratch_write` / `scratch_read` / `scratch_list` | notes under the run directory that survive context eviction |

The `fs_*` tools are rooted to `--workdir`: a path that resolves outside it is refused, never clamped. `run_shell` starts there but is not a sandbox — a shell command can still walk out, which is what `--deny-shell-pattern` is for. `fs_search` and `fs_glob` skip binary files, files over 2 MB, and `.git`/cache/vendor directories. `harness tools [--filter kw]` prints the list the agent would see.

The scratch pad is per run, not per workdir: `scratch_write` saves a short note under `runs/<run_id>/scratch/`, and `scratch_read` gets it back whatever the context budget did in between. The *result* of a scratch call is an ordinary tool result — truncated and evictable like any other — but the note on disk is not, so an agent that writes down what it found can still answer for it forty steps later. Notes the user should keep belong in the workdir, via `fs_write`. Because they are rooted in the run directory, the CLI registers them once the run exists; `harness replay` does not (a recording that used them replays as an unknown tool, which shows up as drift).

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

Return values are serialised for the model; exceptions become error results. Top-level argument types and required keys are validated before the call. The description's first line is what the model sees in `toolbelt_list`, so make it count — it is also the only prose a `toolbelt_list` filter matches, alongside the name, so a filter can never hit text the model cannot see.

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

## Providers

`--provider openai` (default) is `ChatCompletionsTransport`: `POST <endpoint>/chat/completions`, `Authorization: Bearer`. `--provider anthropic` is `AnthropicMessagesTransport`: `POST <endpoint>/messages`, `x-api-key`, `anthropic-version: 2023-06-01`, and `max_tokens` on every request (`--max-tokens`, default 4096).

```bash
python -m harness run --provider anthropic --endpoint https://api.anthropic.com/v1 \
  --model <model-id> --api-key "$KEY" --task "..." --workdir ./project
```

The runtime speaks one message shape; the transport translates. Out: system messages become the top-level `system` parameter, tool schemas become `{name, description, input_schema}`, assistant tool calls become `tool_use` blocks, and all the tool results of one turn go back as `tool_result` blocks in a single user message, so parallel calls stay parallel. In: `text` blocks become the step's reasoning, `tool_use` blocks become tool calls, and `usage` becomes `{prompt_tokens, completion_tokens}` with cached input counted as input. HTTP failures map to `transport_error` with the same one retry. `--extra-body` works for both providers.

Extended thinking is **not supported**. `thinking` and `redacted_thinking` blocks are private reasoning, and guarantee 3 says the harness never reads them — they are dropped, never turned into reasoning and never written to the trajectory. Since the API wants those blocks echoed back on later turns of a tool-use conversation, asking for thinking through `--extra-body` is refused up front rather than producing a conversation the API will later reject.

## Tool-call policy

`AgentRuntime(..., policy=...)` takes any callable run before every dispatch, meta-tools included:

```python
policy(name: str, args: Any) -> str | None    # a sentence to deny, None to allow
```

A denial is not an error — the tool did not fail, it never ran — so it gets its own `kind: denied`, and the message becomes the tool result the model reads, which keeps the loop alive and lets it pick another route. `args` is whatever the model produced, so a policy must not assume it is a dict. A policy that raises is contained as an error result rather than taking the run down. What is in force is recorded in the header (and in each `resume` record), and `trace --summary` counts denials separately from errors.

`harness.policy.ToolPolicy` is the one the CLI exposes, on `run` and `resume`:

```bash
python -m harness run --deny-tool fs_edit --deny-shell-pattern 'rm\s+-rf' --deny-shell-pattern '\bcurl\b' ...
```

Tool names are matched exactly; shell patterns are Python regexes searched against the `command` argument of `run_shell`. A policy is not part of a recording: `replay(..., policy=...)` re-applies one, otherwise a call the original run denied runs for real and the comparison reports drift.

## Replay and the mock server

`ReplayTransport` turns a recorded `trajectory.jsonl` back into a script: recorded reasoning and tool calls are served in order, the tools run for real, and `compare()` diffs the new trajectory against the recording on everything the model or the tools decided (timing, ids and artifact bodies excluded).

```python
from harness.replay import compare, replay

result = replay("runs/20240101-120000-abc123", registry)
assert compare("runs/20240101-120000-abc123", result.run_dir) == []
```

Same from the CLI: `harness replay <run_id> --workdir ./project` — exit `0` identical, `1` drifted with the differing steps printed. A replay re-runs side effects, so point it at a pristine workdir; replaying against the directory the recording already edited is itself reported as drift.

`MockOpenAIServer` serves scripted chat completions over real HTTP (stdlib `http.server`), which is how the tests exercise `ChatCompletionsTransport` end to end; `MockAnthropicServer` is the same script served as Messages API content blocks:

```python
from harness import MockOpenAIServer
from harness.transport import call

with MockOpenAIServer([{"content": "Listing.", "tool_calls": [call("toolbelt_list", {})]}]) as server:
    ...  # --endpoint server.base_url
server.requests      # every request body and header it received
```

Script entries are response dicts (`content`, `tool_calls`, `usage`), raw wire payloads, callables, or `HttpError(status)`; an exhausted script answers 503 rather than hanging. A `reasoning_content` (OpenAI) or `thinking` (Anthropic) key puts a private reasoning channel in the response, which the tests use to prove the harness never reads one.

## Tests

```bash
pip install pytest
python -m pytest -q
```

`tests/test_runtime.py::test_scripted_end_to_end_matches_jsonl_step_for_step` drives a fake transport through list → add → inspect → todo → rejected final → close → accepted final and asserts the JSONL step for step. Use `harness.transport.FakeTransport` the same way to test your own tools without a model.

`tests/test_e2e_http.py` runs `python -m harness run` as a subprocess against the mock server and asserts the exit code, the run directory, the trajectory, the artifacts, and the guarantees on the wire. `tests/test_tools_fs.py` exercises each file tool through the runtime, error paths included; `tests/test_plugins.py` covers `--tools`; `tests/test_replay.py` records a run, replays it, and asserts the trajectories match step for step; `tests/test_resume.py` caps a run, kills one with a transport error, stalls one, resumes each and checks the seams; `tests/test_anthropic.py` asserts the Messages API wire format request by request; `tests/test_policy.py` and `tests/test_scratch.py` drive the denials and the pad through the runtime.

## Layout

```
harness/
  registry.py      ToolRegistry, ToolSpec, validate_args
  runtime.py       AgentRuntime, RuntimeConfig, meta-tools, gate, loop, resume()
  todo.py          validation and merge
  context.py       ContextBudget
  transport.py     ChatCompletionsTransport, FakeTransport
  anthropic.py     AnthropicMessagesTransport (stdlib only, no SDK)
  policy.py        ToolPolicy: the pre-dispatch veto
  resume.py        load a run directory and rebuild the runtime around it
  replay.py        ReplayTransport, replay, compare
  mockserver.py    MockOpenAIServer, MockAnthropicServer (scripted, stdlib http.server)
  plugins.py       --tools module loading
  trajectory.py    TrajectoryWriter, read_trajectory, format_trace, summarize
  prompts.py       system prompt
  cli.py           run / resume / tools / trace / replay
  tools/basic.py   fs_list, fs_read, fs_write, run_shell (rooted to --workdir)
  tools/search.py  fs_search, fs_glob
  tools/edit.py    fs_edit
  tools/scratch.py scratch_write, scratch_read, scratch_list (rooted to the run dir)
  tools/paths.py   workdir rooting and the directory skip list
tests/
```
