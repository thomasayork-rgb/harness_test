# examples

`global-system.md` — house rules for every run on a machine. Copy it to
`~/.config/harness/system.md` and edit; `harness prompt --sources` shows it loaded.

`skills/` — three skills to copy or to point `--skills` at:

| skill | for |
|---|---|
| `investigate` | answer a question about a codebase, with evidence |
| `code-change` | find it, read it, edit it exactly, run the project's tests |
| `final-report` | what belongs in `final_answer.content` |

```bash
python -m harness skills --skills examples/skills
python -m harness run --skills examples/skills --task "..." --model m --endpoint ...
```

Three more ship inside the package, in `harness/skills_bundled/`: `orient` (read the code map index,
then the map files, then the code), `map` (write the prose half of one map file) and `handoff` (what
`final_answer` must contain on a project task). They are not here because a `--project` run is given
them without asking — `harness skills --project ./repo` lists them — but they are written the same
way and are worth reading as examples.
