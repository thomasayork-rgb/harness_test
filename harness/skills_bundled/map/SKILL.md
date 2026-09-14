---
name: map
description: "Write the prose half of one package's map file: purpose, entry points, invariants, gotchas."
tools: [fs_read, fs_search, fs_edit, fs_write]
---

# Map one package

A map file is `docs/map/<dotted.package>.md`: generated frontmatter above
prose. The frontmatter is not yours - `harness map scaffold` writes it and will
overwrite whatever you put there. The prose is yours, and a refresh keeps it
byte for byte.

Map one package at a time, and only packages you have read.

## The sections

- **Purpose.** One sentence first: what this package is for, in the project's
  terms. That first line is what the index shows for the package, so it has to
  stand alone. Then a short paragraph if the package needs one.
- **Entry points.** The two or three names a reader should start at, and what
  each is the way in to. Not a list of everything public; the frontmatter
  already has that.
- **Invariants.** What must stay true, and what breaks if it stops being true.
  The rule a change would silently violate is the reason this file exists.
- **Gotchas.** What surprised you. An ordering that matters, a name that means
  something other than it says, a case handled somewhere unexpected.
- **Depends on** / **Depended on by.** Why, not what. `imports` and
  `inbound_refs` in the frontmatter already say what; this says what would have
  to change with it, and which callers would notice.

## How to find out

1. `fs_read` the map file first: it may already have prose worth keeping.
2. `fs_read` every file in its `files` list. A package you have not read is a
   package you cannot map.
3. `fs_search` for the package's own name, and for the public names in its
   frontmatter, to find the callers. `inbound_refs.callers` says who imports
   it; the search says what they do with it.
4. `fs_edit` the sections one at a time, matching the exact heading line so the
   edit is unique. `fs_write` only for a map file that does not exist yet.

## What not to write

- Nothing the frontmatter already says: no file lists, no line counts, no
  import lists, no restating `public`.
- Nothing you have not read. No speculation about what a function "probably"
  does, and no TODOs for the next reader.
- No history, no dates, no names. The git log holds that; a map file that
  mentions when it was written is wrong the next day.
