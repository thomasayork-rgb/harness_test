---
name: final-report
description: What a good final_answer.content contains, and what it must never contain.
---

# Write the final answer

`final_answer(status, content)` is the whole deliverable. It is read by someone
who did not watch the run, cannot see your tool results, and will act on what
it says. The trajectory is the log; this is the report.

## Content, in this order

1. **The answer, first.** One or two sentences that answer exactly what was
   asked. Not "I investigated the config" - the port is 8080.
2. **The evidence.** Paths, line numbers, values, the commands you ran and
   what they printed. Enough that the reader can check you without asking.
3. **What changed.** Every file you wrote or edited, and what the change was.
4. **What you could not do.** The parts you skipped, the checks that did not
   run, the assumptions you had to make. This is the most valuable paragraph
   in the report and the one most often missing.

## Status

- `completed` - the task is done and verified.
- `blocked` - something outside your control stopped you: a denied call, a
  missing credential, a file that is not there. Say what, precisely.
- `failed` - you could not do it. Say how far you got.

A run whose todos are all closed but whose work was not done is still
`failed`; the gate checks the plan, not the truth.

## Never

- Never claim a command you did not run, or a value you did not read.
- Never report a file as changed without having seen the diff.
- Never pad with a summary of your own process. What you found, not how it
  felt to look.
