---
name: investigate
description: Answer a question about a codebase with evidence, not recollection.
  Glob, search, read, then quote paths and line numbers.
tools: [fs_glob, fs_search, fs_read, scratch_write]
---
# Investigate a codebase

You are answering a question about code you have not read. Everything you say
in the final answer has to be traceable to a file you actually opened.

## Order of work

1. **Shape.** `fs_glob` for the layout before anything else: `**/*.py`,
   `**/*.md`, `**/*.toml`. Never guess a path or a file name.
2. **Narrow.** `fs_search` for the exact text - a function name, a setting, an
   error string. Prefer a precise pattern over a broad one; a search that
   returns forty files has told you nothing.
3. **Read.** `fs_read` every file you intend to quote, with `offset`/`limit`
   when it is long. A match line out of `fs_search` is a pointer, not a quote.
4. **Write it down.** `scratch_write` the path, the line and the value the
   moment you find it. Old results are evicted from context; your note is not.

## When the search comes back empty

Widen once - a shorter pattern, a case-insensitive one, a different spelling -
then change tactics: list the directory, read the entry point, follow the
imports. Repeating a failing search with the same pattern is not progress.

## What the answer looks like

- The claim, then the evidence: `path/to/file.py:42`, and the line itself.
- Every claim carries a location. If you could not find something, say so and
  say where you looked, rather than guessing plausibly.
- One todo per question you were asked; record the finding in its `notes` as
  you close it, so the plan itself shows the evidence.
