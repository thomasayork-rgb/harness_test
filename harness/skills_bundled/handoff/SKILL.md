---
name: handoff
description: "What final_answer must say on a project task: branch, files, map, tests, what is not done."
---

# Hand the work over

Your run ends on a branch someone else will read. They did not watch you work,
and the branch is all they get. `final_answer.content` is the handover note,
and it has to stand on its own.

## What it must contain

1. **The branch**, by name, and the one-line summary of what it does to the
   repository.
2. **The files you changed**, each with what changed in it. Paths relative to
   the repository root, as the tools report them.
3. **The map files you updated**, and any package you changed whose map you did
   not update - with the reason.
4. **The tests you ran**: the exact command, its exit status, and the lines of
   output that matter. A test you could not run is a sentence saying which and
   why, not silence.
5. **Each of the task's checks**, with the evidence for it: the path, the value,
   the command and what it said.
6. **What you did not do**: what you left alone, what you could not reach, what
   you think the next run should pick up.

## How it ends

`status` is `completed` only when the task's checks are met and you can show
it. `blocked` is for something outside your control - a denied command, a
dependency that is not there, a decision that is not yours. `failed` is for
work you could not do. Say which, and why, in the first sentence.

## Git

The harness commits your worktree on the task branch when the run ends,
whatever the status, so uncommitted work is not lost and you do not have to
drive git. Commit mid-run yourself only when the history is worth splitting.
Never push, never switch branches, never reset: that worktree is shared with
the repository it was cut from, and those reach past your run.
