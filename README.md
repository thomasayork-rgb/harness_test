# harness

A minimal ReAct agent runtime with four guarantees the loop enforces, not the prompt:

1. **Lazy toolbelt.** Every tool is registered; none is in context until the agent calls `toolbelt_add`. The first request carries only the meta-tools: the six core ones, and three more for skills when a run discovered any.
2. **Todo gate.** `final_answer` is rejected while any todo is `pending` or `in_progress` (or before a todo list exists). Plain assistant text cannot end a run.
3. **Think-before-act.** The assistant's text on each tool-calling turn is captured as that step's `reasoning`. No retry if it's empty. Private reasoning channels are never read.
4. **Trajectory export.** One JSONL per run: header, one record per tool call (discovery, skill and todo calls included), a `note` record wherever the loop spoke up on its own, footer. Full tool results go to `artifacts/`; the JSONL carries a preview. `harness trace` reads it back.

Plus a context budget so a local model survives a 60-step run: per-result truncation in context, oldest-result eviction past a total budget; reasoning, todo state and the results of the turn in flight are never evicted. Protection covers the current state, not every version of it: a new `todo_write` result supersedes the earlier ones, and so does a second `skill_load` of the same skill — each older copy becomes a one-line pointer to its artifact, since a protected copy of a list that has moved on is context nothing can reclaim.

Zero dependencies. Python 3.10+. Talks to any OpenAI-compatible `/v1/chat/completions` endpoint (llama.cpp server, vLLM, LM Studio, Ollama, or the real thing), or to the Anthropic Messages API with `--provider anthropic`.

## Run

```bash
python -m harness run \
  --task "Find the config file and report the port" \
  --model <model-id> \
  --endpoint http://localhost:8080/v1 \
  --workdir ./project

python -m harness run --project ./repo --task "..." --model m --endpoint http://localhost:8080/v1
python -m harness resume <run_id> --step-cap 400        # same flags as before, from the recording
python -m harness bench tasks.jsonl --model m --endpoint http://localhost:8080/v1
python -m harness map scaffold --project ./repo         # write or refresh docs/map/
python -m harness worktree list --project ./repo        # the worktrees this runs dir holds
python -m harness worktree prune --project ./repo       # remove the ones of finished runs
python -m harness tools                       # what the agent can discover
python -m harness skills                      # what the agent can load
python -m harness prompt [--sources]          # the system prompt a run would start with
python -m harness trace <run_id>              # readable trace
python -m harness trace <run_id> --step 7     # one step in full, artifact included
python -m harness trace <run_id> --summary    # status, kinds, per-tool counts, tokens, elapsed
python -m harness replay <run_id> --workdir ./project   # re-run a recording against today's tools
```

`--api-key` or `HARNESS_API_KEY` for hosted endpoints. Runs land in `./runs/<run_id>/` (`--runs-dir` to move). Exit codes for `run` and `resume`: `0` completed, `1` blocked/failed, `2` transport_error, `3` step_cap, `4` stalled, `130` interrupted (Ctrl-C). For `replay`: `0` identical to the recording, `1` drifted. For `bench`: `0` if every task completed, `1` otherwise. A bad command line — including a run that cannot be resumed — is `64`, an unreadable run directory `66`.

Options: `--step-cap 250`, `--result-chars 2000`, `--context-chars 60000`, `--preview-chars 400`, `--no-todo-gate`, `--timeout 120`, `--tools mypkg.tools` (repeatable), `--skills ./skills` (repeatable), `--skill-chars 12000`, `--progress-nudge 12`, `--trust-project-plugins`, `--extra-body '{"temperature": 0}'` (merged into every request; may not set `model`, `messages`, `tools`, `tool_choice`), `--provider openai|anthropic`, `--max-tokens 4096` (anthropic only), `--deny-tool NAME` and `--deny-shell-pattern REGEX` (both repeatable), `--system-prompt FILE`, `--append-system-prompt FILE` (repeatable) and `--no-global-prompt` (see System prompt).

`--project PATH` points a run at a git repository instead of a directory: it gets a worktree of its own at `<runs-dir>/<run_id>/wt`, cut from HEAD on branch `task/<run_id>`, and that worktree is the workdir (so `--project` and `--workdir` are mutually exclusive). The project must be committed first — the worktree is cut from HEAD, so uncommitted work is invisible to the agent, and uncommitted `docs/map/` most of all; changes under `tasks/` are ignored and `--allow-dirty` overrides the check. A project run also denies the git that reaches past its worktree (`git push`, `checkout`, `switch`, `reset --hard`, `worktree`, `branch -D`) as ordinary policy denials, which `--allow-git` lifts. `harness map scaffold --project PATH [--package PKG]` writes the project's code map: `docs/map/<dotted.package>.md` per package — generated frontmatter (files, loc, imports, public names, who imports it, and the git blob sha each of those was read from) above prose a reader writes, which a refresh keeps byte for byte — plus `docs/map/INDEX.md`, one line per package with the first line of its Purpose. It is deterministic: same tree, same bytes, no timestamps. `--no-keep-worktree` removes the worktree after a finished run, keeping the branch; a worktree with uncommitted changes is left alone. The header records a `project` block (`path`, `base_sha`, `branch`, `worktree`, `task_id`), so `resume` continues in the same worktree — or, if it is gone, cuts it again from the branch and tells the model that whatever was uncommitted in it is gone.

## Run directory

```
runs/<run_id>/
  trajectory.jsonl     header / step... / footer  (see Resume for a resumed run)
  state.json           persisted RunState (active tools, loaded skills, todos, messages, final)
  artifacts/           step_0007_final_answer.txt — full result per step
  scratch/             notes the agent wrote with scratch_write
  wt/                  the worktree of a --project run: its workdir, on branch task/<run_id>
  run.lock             the pid of the process in the loop; removed when it leaves
```

Step record fields: `step, ts, elapsed_ms, reasoning, tool, args, kind, call_index, result_preview, result_bytes, artifact, tokens_in, tokens_out, todo_snapshot`. `call_index` is the position of the call within its model turn, so turn boundaries survive the round trip (see Replay). `kind` ∈ `ok | error | denied | final_accepted | final_rejected | text_only`. When one turn issues several tool calls, usage is recorded on the first and `null` on the rest — never double-counted. `todo_snapshot` is the state after the step, notes included. The header records the `config` the run started under, the `policy` in force or `null`, the `prompt_sources` the system message was built from, `skills` (the directories searched and the names found), `project` (the repository, base commit, branch and worktree of a `--project` run, or `null`), and `invocation`: the endpoint, provider, workdir, `--tools` modules, skill directories and request options the run was launched with, which is what `resume` defaults to. The API key is never recorded.

Two other record types sit in the same file. A `note` record — `{"type": "note", "step", "kind", "text"}` — is something the loop said to the model that is not a step: the progress nudge (`kind: "progress_nudge"`) and the budget warning (`kind: "budget"`, once per crossing, when protected results alone exceed `--context-chars`). It does not count toward the step cap; `format_trace` prints it at the seam and `trace --summary` counts it. The footer carries `todos`: the todo list as it stood when the run ended, with its notes, whether or not there was a final answer.

## Failure semantics

| Event | Behaviour |
|---|---|
| bad tool args, unknown/inactive tool, tool raises | error string returned as the tool result; loop continues |
| `final_answer` with open todos | rejection naming the open ids; loop continues |
| turn with no tool call | recorded as `text_only`; nudge appended; 3 in a row → `stalled` |
| transport error | retried (default: once), waiting 1 s before the first retry and 4 s before any later one — or what a `Retry-After` on a 429/529 asked for, capped at 60 s — then `transport_error` |
| step cap | `step_cap`, `final` is `null` |
| Ctrl-C (`KeyboardInterrupt`) | `interrupted`, footer written, the calls of the turn in flight answered "not executed"; exit `130` |
| tool call denied by policy | denial string as the tool result, `kind: denied`; loop continues |
| plan unchanged for `--progress-nudge` steps | one user message asking for the plan; a `note` record, not a step |
| protected results alone over `--context-chars` | one user message saying eviction cannot help; a `budget` note, once per crossing |

All of it is visible in the trajectory. When a turn is cut short — the cap trips between two calls of it, or `final_answer` is accepted with calls queued behind it — the calls that never ran are answered in the persisted transcript with a "not executed" result, so every `tool_call` has a matching tool message and the conversation can be handed back to a provider.

`transport_error`, `step_cap`, `stalled` and `interrupted` are interruptions, not answers: those runs can be resumed.

## Resume

```bash
python -m harness resume <run_id> [--step-cap 400] [--force]
```

Continues a run that ended with `transport_error`, `step_cap`, `stalled` or `interrupted`. `completed`, `blocked` and `failed` are answers, not interruptions: they are refused with exit `64`.

A run whose state still says `running` was never closed, and `run.lock` — the pid of the process that was in the loop, written for the length of `run`/`resume` and removed on the way out, Ctrl-C included — says whether anyone still is. While that pid is alive the resume is refused (exit `64`), because two processes appending to one trajectory is not something to guess at; `--force` takes it over anyway. A lock nobody holds is stale — a hard kill, a machine that went away — so the run is picked up as `interrupted`, with the seam recording which of the two it was. A segment killed outright never wrote a footer, so in that one case the `resume` record follows its last step directly.

The run keeps its id, its directory and its step counter; `state.json` supplies the conversation, the active tools and the todos, and the trajectory header supplies the config (`--step-cap` raises the cap, and must, if the run is already at it). `--model` defaults to the one the run recorded, and `--endpoint`, `--provider`, `--workdir`, `--tools`, `--skills`, `--extra-body`, `--max-tokens`, `--timeout` and the policy flags default to the `invocation` and `policy` the last segment recorded, so `harness resume <run_id>` on its own continues the run as it was; an explicit flag overrides. The skill directories come back as the run searched them, so a resumed segment can still `skill_load`; `--progress-nudge` is the one loop knob a resume can change on its own. The API key is never recorded, so `--api-key` (or `HARNESS_API_KEY`) is given again. Prompt flags are refused with exit `64`: the system prompt is part of the conversation in `state.json` and is never re-resolved.

The trajectory is appended to, never replaced:

```
header  step*  footer  [resume  step*  footer]*
```

One footer per segment — each says how that segment ended — with a `resume` record between segments carrying `from_status`, `from_step`, `from_detail`, and the `model`, `step_cap`, `config`, `policy` and `invocation` the next segment runs under — so the segment after that defaults to them in turn. The last footer is the run's answer; earlier ones are history. `read_trajectory` returns the lot; `summarize` totals every segment and reads its status from the last footer (plus `segments` and `resumed_from`, and wall time summed per segment so the hours a run sat waiting are not counted); `format_trace` prints the seam where it happened; `ReplayTransport` reads a resumed recording as one straight run under the configuration the last segment ended with, so a raised cap replays as a raised cap.

Resuming appends one user message saying what happened, which also leaves the transcript ending on a user turn however the segment died. From Python:

```python
from harness.resume import resume
result = resume("runs/20240101-120000-abc123", registry, transport, step_cap=400)
```

## System prompt

Three layers, joined with a blank line:

| layer | where it comes from |
|---|---|
| base | the built-in prompt, or `--system-prompt FILE` instead of it |
| global | the first of `$HARNESS_SYSTEM_PROMPT`, `$XDG_CONFIG_HOME/harness/system.md`, `~/.config/harness/system.md` that exists — unless `--no-global-prompt` |
| append | each `--append-system-prompt FILE`, in the order given |

The built-in prompt teaches the mechanics the loop enforces (discovery, the todo gate, think-before-act, reading
errors, denials, eviction and the scratch pad, the file-tool patterns, what `final_answer.content` is for) and
nothing about your house. House rules — never run destructive shell commands, prefer `fs_edit` over `fs_write`,
report paths relative to the workdir — belong in the global prompt, which applies to every run on the machine.
`examples/global-system.md` is one to copy to `~/.config/harness/system.md` and edit. Only the first existing
location is used; a machine has one set of house rules, not three.

```bash
python -m harness prompt                      # the effective prompt for these flags
python -m harness prompt --sources            # where each layer came from, and how big it is
python -m harness run --append-system-prompt ./task-notes.md ...
```

```
base    builtin  2567 chars
global  /home/you/.config/harness/system.md  941 chars
total: 3510 chars
```

A file that is named and cannot be read is a usage error (exit `64`), never a silently missing layer. The layers
are recorded in the trajectory header as `prompt_sources` — `{"source": "builtin" | "<path>", "role": "base" |
"global" | "append", "chars": n}` — and printed by `trace` and `trace --summary`, so a trace says what the model
was told and not just what it did. A resumed run replays the system message in `state.json` and never re-reads a
file, which is why prompt flags on `resume` are refused rather than quietly ignored.

## Benchmarks

```bash
python -m harness bench tasks.jsonl --model m --endpoint http://localhost:8080/v1 --step-cap 60
```

One task per JSONL line — `{"task": "...", "workdir": "...", "run_id": "..."}`, the last two optional — run
sequentially under the same flags as `run` (`--workdir` is the default for lines without one). Each task is an
ordinary run with its own run directory, so a task that dies can be resumed, traced and replayed like any other.

```
run id                  status     steps  tokens in  tokens out  elapsed
20240101-120000-a1b2c3  completed      7       4200         310    8.4 s
20240101-120009-d4e5f6  step_cap      60      51000        2400   77.1 s

2 task(s): 1 completed, 1 not (step_cap)
```

A row per task is appended to `bench.jsonl` next to the run directories: `run_id, task, workdir, status, steps,
tokens_in, tokens_out, elapsed_ms, wall_s, errors, denied, final`. Exit `0` if every task completed, `1`
otherwise.

## Skills

A skill is a markdown document that teaches the model how to do one kind of work, discovered lazily
and loaded on demand, in the same spirit as the toolbelt. Only names and one-line descriptions are
held until the model asks for one.

```
skills/investigate/SKILL.md          or   skills/investigate.md
```

```markdown
---
name: code-change                    # optional; defaults to the directory or file stem
description: Change code in a project safely: find it, read it, edit it exactly, run the tests.
tools: [fs_search, fs_read, fs_edit, run_shell]     # activated when the skill is loaded
plugin: ./tools.py                   # optional --tools module, relative to this file
---
The body is the skill: what the model should read before doing this kind of work.
```

The frontmatter is parsed with the stdlib alone by `harness.frontmatter` — `key: value` scalars,
`[a, b]` inline lists, `- item` block lists, indented continuation lines, one-level-or-deeper nested
mappings and `"quoted keys"`. Values are strings; `harness.frontmatter.dump` writes the canonical
form back (keys in order, lists inline, quotes only where a value needs them), which is what the
shipped skills are written in. `description` is required and its **first line**
is all `skill_list` shows, so make it count; put the detail in the body. A `.md` file with no
frontmatter is not a skill and is skipped; a `<name>/SKILL.md` that cannot be parsed is reported on
stderr, once, and left out.

Discovery merges every location that exists, lowest precedence first:

| where | what for |
|---|---|
| `$XDG_CONFIG_HOME/harness/skills`, else `~/.config/harness/skills` | skills for every run on this machine |
| `$HARNESS_SKILLS` (`os.pathsep`-separated) | skills for this shell |
| `<workdir>/.harness/skills` | skills that live with the project |
| `--skills DIR` (repeatable, on `run`, `bench` and `resume`) | skills for this run |

A name found twice is a clash: the last directory wins, and the clash is printed on stderr rather
than silently decided — as is a `--skills` directory that is not there, and a skill file that could
not be read. Nothing is loaded into context at discovery.

```bash
python -m harness skills --skills ./skills --workdir ./project [--filter kw]
python -m harness run --skills ./skills ...
```

Three meta-tools exist **only when at least one skill was discovered**, so a run without skills
starts with exactly the six core ones:

| tool | what it does |
|---|---|
| `skill_list(filter)` | name and first description line per skill, like `toolbelt_list` |
| `skill_load(name)` | the whole skill text as the result; activates its `tools`, registers its `plugin` |
| `skill_unload(name)` | replaces that text with a one-line marker, to free the context |

A loaded skill is the one result that is never truncated by `--result-chars` and never evicted by
the context budget: the model was told to follow it, so it has to still be there. The price is a
size limit — a skill larger than `--skill-chars` (default 12000) is refused at load, with its size
in the error. Loading a skill twice is a no-op that says so; unloading leaves the tools it activated
active (`toolbelt_remove` drops them). The header records what was discovered, `skill_load` and
`skill_unload` are ordinary steps, and `trace --summary` says which skills the run actually read.

A `plugin` is code. A skill from your own directories runs it the moment the skill is loaded; a
skill from `<workdir>/.harness/skills` came with the project the run was pointed at, so its plugin
is refused unless the run was started with `--trust-project-plugins`. The refusal loads nothing —
not the text either — so the model is never told to follow instructions that rely on tools it
cannot have. `harness skills` marks such skills.

When skills are present the built-in prompt gains a short SKILLS section telling the model to list
skills before planning and to follow what it loads; a run without skills is sent the prompt it was
sent before, byte for byte.

`examples/skills/` ships three to copy or point `--skills` at: `investigate` (answer a question
about a codebase with evidence), `code-change` (find it, read it, edit it exactly, run the project's
tests) and `final-report` (what belongs in `final_answer.content`).

## Plan, notes and the progress nudge

A todo item takes an optional `notes` string (500 chars): the outcome of that step, in the agent's
own words — the value it found, the file it changed, the command it ran and what it said. On a merge
a note survives an update that does not mention it, so closing an item costs nothing it recorded
when it opened it; `notes: ""` clears one. Notes ride in every `todo_snapshot`, in `state.json`, and
in the footer.

```json
{"id": "find", "content": "find the port", "status": "completed", "notes": "8080, conf/config.ini:2"}
```

Once a todo list exists, `--progress-nudge N` steps (default 12, `0` disables) with no change to any
todo's status or notes append one user message asking the model to bring its plan up to date, and
reset the counter. Text-only turns count toward it. It is recorded as a `note` record, it is not a
step, and it does not count toward the step cap. A plan that keeps moving is never nudged.

```
  12  ok              run_shell                  28 ms  Running the project's own check.
      -- progress_nudge after step 12: 5 steps since your todo list last changed. Update it now: ...
  13  ok              todo_write                  1 ms  Right - two of those are done and the list does not say so.
```

`trace --summary` ends with the plan the run finished on:

```
todos:
  [completed] find: find the port the service really uses
      note: 8080: app/config.ini:2 and app/server.py:1; README.md:3 says 9090
  [completed] fix: fix the port in README.md
      note: README.md:3, 9090 -> 8080; diff is one line
```

## Built-in tools

| tool | what it does |
|---|---|
| `fs_list` | list a directory, skipping `.git`/cache/vendor directories |
| `fs_read` | read a text file, optional offset/limit; a file with NUL bytes is refused as binary |
| `fs_write` | write a text file, creating parents |
| `fs_search` | regex over file contents; name or path glob filter, result cap, line numbers |
| `fs_glob` | find files by glob pattern |
| `fs_edit` | exact-string replace, must be unique unless `replace_all`; returns a unified diff |
| `run_shell` | run a command in the workdir; a non-zero exit is a result, a timeout is an error |
| `scratch_write` / `scratch_read` / `scratch_list` | notes under the run directory that survive context eviction |

The `fs_*` tools are rooted to `--workdir`: a path that resolves outside it is refused, never clamped. `run_shell` starts there but is not a sandbox — a shell command can still walk out, which is what `--deny-shell-pattern` is for. What it does not get is the harness's own configuration: the child environment is this process's minus every `HARNESS_*` variable, so `env` in a command cannot write `HARNESS_API_KEY` into a result, an artifact or the trajectory. Each command runs in its own session, and a timeout kills that whole process group — what the command backgrounded dies with it instead of outliving the run. `fs_search` and `fs_glob` skip binary files, files over 2 MB, and `.git`/cache/vendor directories, and `fs_list` skips those same directories. `harness tools [--filter kw]` prints the list the agent would see.

The scratch pad is per run, not per workdir: `scratch_write` saves a short note under `runs/<run_id>/scratch/`, and `scratch_read` gets it back whatever the context budget did in between. The *result* of a scratch call is an ordinary tool result — truncated and evictable like any other — but the note on disk is not, so an agent that writes down what it found can still answer for it forty steps later. Notes the user should keep belong in the workdir, via `fs_write`. Because they are rooted in the run directory, the CLI registers them once the run exists, and `harness replay` registers a pad in the replay's own run directory, so a recording that took notes replays as a run that takes notes.

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
                  Path("runs"), model="<model-id>", config=RuntimeConfig(step_cap=100))
result = rt.run("...")
```

## Providers

`--provider openai` (default) is `ChatCompletionsTransport`: `POST <endpoint>/chat/completions`, `Authorization: Bearer`. `--provider anthropic` is `AnthropicMessagesTransport`: `POST <endpoint>/messages`, `x-api-key`, `anthropic-version: 2023-06-01`, and `max_tokens` on every request (`--max-tokens`, default 4096).

```bash
python -m harness run --provider anthropic --endpoint https://api.anthropic.com/v1 \
  --model <model-id> --api-key "$KEY" --task "..." --workdir ./project
```

The runtime speaks one message shape; the transport translates. Out: system messages become the top-level `system` parameter, tool schemas become `{name, description, input_schema}`, assistant tool calls become `tool_use` blocks, and all the tool results of one turn go back as `tool_result` blocks in a single user message, so parallel calls stay parallel. In: `text` blocks become the step's reasoning, `tool_use` blocks become tool calls, and `usage` becomes `{prompt_tokens, completion_tokens}` with cached input counted as input. HTTP failures map to `transport_error` with the same retries and the same backoff, `Retry-After` included. `--extra-body` works for both providers.

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

Tool names are matched exactly; shell patterns are Python regexes searched against the `command` argument of `run_shell`. A policy is part of a recording: `replay` rebuilds the one the header describes, so a denied call stays denied instead of running for real and dragging every later step into drift. The skill directories are rebuilt from the header for the same reason. `replay(..., policy=...)` and `harness replay --deny-tool ...` replay under different rules; `policy=ToolPolicy()` under none.

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

Script entries are response dicts (`content`, `tool_calls`, `usage`), raw wire payloads, callables, or `HttpError(status, message, retry_after=...)` — which sends `Retry-After`, so the backoff can be tested; an exhausted script answers 503 rather than hanging. A `reasoning_content` (OpenAI) or `thinking` (Anthropic) key puts a private reasoning channel in the response, which the tests use to prove the harness never reads one.

## Tests

```bash
pip install pytest
python -m pytest -q
```

`tests/test_runtime.py::test_scripted_end_to_end_matches_jsonl_step_for_step` drives a fake transport through list → add → inspect → todo → rejected final → close → accepted final and asserts the JSONL step for step. Use `harness.transport.FakeTransport` the same way to test your own tools without a model.

`tests/test_e2e_http.py` runs `python -m harness run` as a subprocess against the mock server and asserts the exit code, the run directory, the trajectory, the artifacts, and the guarantees on the wire. `tests/test_tools_fs.py` exercises each file tool through the runtime, error paths included; `tests/test_plugins.py` covers `--tools`; `tests/test_replay.py` records a run, replays it, and asserts the trajectories match step for step; `tests/test_resume.py` caps a run, kills one with a transport error, stalls one, Ctrl-Cs one (a real SIGINT to a real `harness run`), takes one over from a stale lock and refuses one held by a live pid, resuming each and checking the seams; `tests/test_anthropic.py` asserts the Messages API wire format request by request; `tests/test_policy.py` and `tests/test_scratch.py` drive the denials and the pad through the runtime; `tests/test_prompts.py` covers the prompt layers, the provenance and `harness prompt` (with `HOME`, `XDG_CONFIG_HOME` and `HARNESS_SYSTEM_PROMPT` pointed at temporary directories, never the real ones); `tests/test_bench.py` benches two tasks over the mock server; `tests/test_skills.py` covers the frontmatter, discovery and precedence, the three meta-tools through the runtime, `harness skills`, and a CLI run with `--skills` over HTTP; `tests/test_progress.py` covers todo notes, the nudge and the footer todos.

`tests/test_e2e_complex.py` is the one that reads as the whole point: one real `python -m harness run` in which the model lists its skills, loads two of them, plans four todos, does the work with the file tools, records each outcome in a note, is nudged once for ignoring its plan, brings the plan back and finishes — asserted step for step, with the footer todos and the `trace --summary` a reader would run afterwards.

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
  project.py       --project: worktrees, the dirty rules, the git denials, worktree list|prune
  codemap.py       docs/map/: the scan, the generated frontmatter, INDEX.md
  frontmatter.py   the --- block: parse, dump, split (shared by skills and the project files)
  skills.py        SKILL.md frontmatter, discovery, SkillSet
  trajectory.py    TrajectoryWriter, read_trajectory, format_trace, summarize
  prompts.py       the prompt layers: built-in, global file, appends
  cli.py           run / resume / bench / map / worktree / skills / tools / prompt / trace / replay
  tools/basic.py   fs_list, fs_read, fs_write, run_shell (rooted to --workdir)
  tools/search.py  fs_search, fs_glob
  tools/edit.py    fs_edit
  tools/scratch.py scratch_write, scratch_read, scratch_list (rooted to the run dir)
  tools/paths.py   workdir rooting and the directory skip list
examples/
  global-system.md an example global prompt: house rules for every run
  skills/          investigate, code-change, final-report: skills to copy
tests/
```
