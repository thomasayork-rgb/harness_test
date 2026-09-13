# harness

Minimal zero-dependency ReAct agent runtime for any OpenAI-compatible chat-completions endpoint, or the Anthropic Messages API with `--provider anthropic`. Four guarantees are enforced by the loop, never by the prompt:

1. **Lazy toolbelt.** Only the meta-tools are in context until the model calls `toolbelt_add`: the six core ones, plus `skill_list`/`skill_load`/`skill_unload` when a run discovered skills.
2. **Todo gate.** `final_answer` is rejected while any todo is `pending`/`in_progress`, or before a todo list exists.
3. **Think-before-act.** The assistant text on a tool-calling turn is that step's `reasoning`. Private reasoning channels (`reasoning_content`, `reasoning`, `thinking`) are never read.
4. **Trajectory export.** One JSONL per run (header, one record per tool call, `note` records where the loop spoke up on its own, footer with the final todos; a `resume` record and another footer per resumed segment); full results in `artifacts/`.

`README.md` documents the command surface, the JSONL record fields, resume semantics and the failure-semantics table. This file is about how to work on the harness and how to use it while you do.

## Rules

- `harness/` is **stdlib only** and **Python 3.10+** (no `tomllib`, no `except*`, no 3.11-only syntax). Tests use pytest only.
- Preserve the four guarantees and the failure semantics. A change to either is a documented decision, not a side effect.
- Never surface a private reasoning channel as `reasoning` or write it to the trajectory or artifacts.
- A tool's description **first line** is all the model sees in `toolbelt_list`; put detail on a second line for `toolbelt_inspect`. Schemas are validated before the call; raise errors the model can act on. The same rule holds for a skill: the first line of its `description` is all `skill_list` shows.
- File tools are rooted to the workdir through `harness/tools/paths.py`: refuse escapes, never clamp. Scratch tools are rooted to the run directory. Error messages show paths relative to the root, never absolute host paths. `run_shell` is rooted but not sandboxed; `--deny-shell-pattern` exists for that.
- Every tool and every failure path is tested **through the runtime**: a scripted `FakeTransport` run through `toolbelt_add`, the call, and `final_answer`, asserting the JSONL records and artifacts. Anything CLI-facing also gets an end-to-end test over HTTP against a mock server.
- No network in tests except mock servers you start on localhost. Never probe the environment for credentials or try to reach a live model.
- Any new option, subcommand or tool is documented in `README.md` in the same terse style, and `python -m harness --help` stays coherent.
- Checks before every commit:

```bash
python3 -m pytest -q
python3 -m compileall -q harness tests
```

- Commit in logical units with descriptive messages. `runs/`, `__pycache__/` and `.pytest_cache/` are gitignored; never commit a run directory. Do not push or open pull requests unless asked.

## Using the harness while developing

### Scripted run (unit level)

```python
from harness.runtime import AgentRuntime
from harness.trajectory import read_trajectory
from harness.transport import FakeTransport, TransportError, call

fake = FakeTransport([
    {"content": "Listing tools.",  "tool_calls": [call("toolbelt_list", {})]},
    {"content": "Activating.",     "tool_calls": [call("toolbelt_add", {"names": ["fs_search"]})]},
    {"content": "Planning.",       "tool_calls": [call("todo_write", {"todos": [
        {"id": "find", "content": "find the port", "status": "in_progress"}]})]},
    {"content": "Searching.",      "tool_calls": [call("fs_search", {"pattern": "(?i)port"})]},
    {"content": "Closing, done.",  "tool_calls": [
        call("todo_write", {"todos": [{"id": "find", "content": "find the port", "status": "completed"}]}),
        call("final_answer", {"status": "completed", "content": "8080"})]},
])
rt = AgentRuntime(registry, fake, runs_dir, "fake")      # registry: a ToolRegistry with your tools registered
res = rt.run("Find the port")                            # policy=ToolPolicy(...) to test denials
steps = [r for r in read_trajectory(res.run_dir) if r["type"] == "step"]
```

Assert on each step's `tool`, `kind` (`ok | error | denied | final_accepted | final_rejected | text_only`), `result_preview` and artifact file, and on `fake.requests[i]["tools"]` for what was in context on request `i`. A script entry that is an `Exception` instance is raised instead of returned (`TransportError(...)` for a transport failure). Several `tool_calls` in one entry make one turn with several steps; `call()` generates a unique id for each, so pass an explicit `id=` only when a test wants to name a call.

### Real process over HTTP (integration level)

```python
from harness.mockserver import HttpError, MockAnthropicServer, MockOpenAIServer

with MockOpenAIServer(script, model="mock-model", api_key="k") as server:
    subprocess.run([sys.executable, "-m", "harness", "--runs-dir", str(runs), "run",
                    "--task", task, "--model", "mock-model", "--endpoint", server.base_url,
                    "--workdir", str(work), "--api-key", "k"], capture_output=True, text=True)
```

Same script shape as `FakeTransport`; `server.requests` records every wire request so you can assert what reached the endpoint. `MockAnthropicServer` serves the same script as Messages API content blocks for `--provider anthropic`. An `HttpError(500)` entry yields a transport error. See `tests/test_e2e_http.py` and `tests/test_anthropic.py`.

### Reading a run back, resuming, replaying

```bash
python -m harness --runs-dir RUNS trace RUN_ID              # readable trace
python -m harness --runs-dir RUNS trace RUN_ID --step 7     # one step in full, artifact included
python -m harness --runs-dir RUNS trace RUN_ID --summary    # status, kinds, per-tool counts, tokens
python -m harness --runs-dir RUNS resume RUN_ID [--step-cap N]  # after transport_error, step_cap, stalled
python -m harness --runs-dir RUNS replay RUN_ID             # exit 0 identical, 1 drift
python -m harness --runs-dir RUNS bench TASKS.jsonl ...     # a file of tasks, one table, bench.jsonl
python -m harness tools [--tools SPEC]                      # what the agent can discover
python -m harness skills [--skills DIR] [--workdir W]      # what the agent can load
python -m harness prompt [--sources]                        # the system prompt a run would start with
```

`--runs-dir` is a global option and goes **before** the subcommand. `resume` defaults `--endpoint`, `--provider`, `--workdir`, `--tools`, `--extra-body`, `--max-tokens`, `--timeout` and the policy to what the run recorded in its trajectory header (`invocation`); an explicit flag overrides. The API key is never recorded, and prompt flags are refused. Replaying a run that edited files needs a pristine workdir. For experiments, point `--runs-dir` and `--workdir` at scratch space, never at the repo.

### Skills

A skill is `skills/<name>/SKILL.md` (or `skills/<name>.md`): frontmatter with `description`
(required; first line is what `skill_list` shows), optional `tools:` to activate on load and
optional `plugin:` (a `--tools` module relative to the skill file), then a body. Write fixture
skills under `tmp_path` and hand the runtime a `SkillSet`:
`AgentRuntime(registry, fake, runs, "fake", skills=discover([skills_dir], workdir))` - the three
skill meta-tools appear only when that set is non-empty, so every existing test that asserts the
first payload is exactly `META_NAMES` still holds. From the CLI, `--skills DIR` merges with
`$HARNESS_SKILLS`, the config directory and `<workdir>/.harness/skills` (later wins, clashes
reported on stderr), `harness skills` prints what was found, and `harness run --skills
examples/skills` drives the shipped ones. A loaded skill's text is protected from eviction and
exempt from `--result-chars`, which is why it has its own `--skill-chars` ceiling; `skill_unload`
puts a marker in its place. The header records the directories and names, `resume` restores them
from `invocation`, and `replay` rediscovers them, so a recorded skills run is still a regression
test. A project skill (one found in `<workdir>/.harness/skills`) that declares a `plugin` is refused
at load unless the run has `--trust-project-plugins`: that plugin is the project's own code, and
the harness never runs it on the operator's say-so alone.

### Plugins and policy

`--tools path/to/module.py` or `--tools dotted.module` (repeatable). The module exports either `registry` (a `ToolRegistry`) or `register(registry[, workdir])`. Nothing is auto-discovered. `--deny-tool NAME` and `--deny-shell-pattern REGEX` (repeatable) veto calls before dispatch; a denial is a `denied` step the model can read, not a crash.

## Layout

```
harness/
  registry.py      ToolRegistry, ToolSpec, validate_args
  runtime.py       AgentRuntime, RuntimeConfig, meta-tools, todo gate, the loop, resume()
  todo.py          todo validation and merge
  context.py       ContextBudget: per-result truncation, oldest-result eviction
  transport.py     Transport protocol, ChatCompletionsTransport, FakeTransport, call()
  anthropic.py     AnthropicMessagesTransport (stdlib only)
  policy.py        ToolPolicy: the pre-dispatch veto
  resume.py        load a run directory and rebuild the runtime around it
  replay.py        ReplayTransport, replay(), compare()
  plugins.py       --tools loading
  skills.py        SKILL.md frontmatter, discovery, SkillSet (--skills)
  mockserver.py    MockOpenAIServer, MockAnthropicServer for tests
  trajectory.py    TrajectoryWriter, read_trajectory, format_trace, summarize
  prompts.py       prompt layers (built-in, global file, appends) and the nudge
  cli.py           run / resume / bench / skills / tools / prompt / trace / replay
  tools/           paths.py (rooting), basic.py, search.py, edit.py, scratch.py
examples/          global-system.md: an example global prompt
                   skills/: investigate, code-change, final-report
tests/             one module per area; the *_http and anthropic tests are the only real-HTTP tests
```
