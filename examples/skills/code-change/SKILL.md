---
name: code-change
description: Change code in a project safely: find it, read it, edit it exactly, run the tests.
tools: [fs_search, fs_read, fs_edit, run_shell]
---
# Make a change to a project

A change nobody verified is a guess. Finish with the diff and the test output,
or with an honest account of why you could not run the tests.

## Order of work

1. **Find it.** `fs_search` for the symbol or the string you are changing.
   Search for the call sites too: a change with one caller and a change with
   twelve are different tasks.
2. **Read it.** `fs_read` the whole region you are about to touch, plus
   whatever it calls. Never edit a file you have only searched.
3. **Edit it.** `fs_edit` replaces an exact string, and the old string must
   match once and only once - include the lines above and below to make it
   unique. The result is a unified diff: read it. That diff is what you
   changed, whatever you meant to change.
4. **Check it.** Run the project's own tests with `run_shell`, from the
   project's root - `python3 -m pytest -q`, `npm test`, `make check`, whatever
   the repository actually uses (look for the command before inventing one).
   A non-zero exit is a result, not a crash: read the failure and fix it.

## Rules

- One todo per change, `in_progress` while you work on it, `completed` with a
  note saying what you edited and what the tests said.
- `fs_write` replaces a whole file; on a file that already exists, prefer
  `fs_edit` so you cannot silently drop the parts you did not read.
- If a test fails for a reason your change did not cause, say so in the final
  answer rather than editing the test to make it quiet.

## What the answer looks like

The diff (or the exact edits), the command you ran, its exit status and the
relevant lines of its output, and anything you deliberately left alone.
