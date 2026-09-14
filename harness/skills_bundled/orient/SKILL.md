---
name: orient
description: "Find your way around a mapped repository before changing it: index, map files, then code."
tools: [fs_read, fs_search, fs_glob]
---

# Orient yourself in a mapped repository

This repository keeps a code map: `docs/map/INDEX.md`, and one file per
package under `docs/map/`. Reading it first is cheaper than reading the code,
and it tells you what the code cannot: what a package is for, what must stay
true, and what will surprise you.

## Order of work

1. **The index.** `fs_read docs/map/INDEX.md`. One line per package, grouped by
   top-level package, most depended-on first, with the first line of its
   Purpose. `(inbound n)` is how many files outside the package import it: the
   high numbers are what the project is built on, and what a change to them
   costs. `[stale]` or `(unmapped)` means that line is not to be trusted.
2. **The map files that matter.** For the packages your task touches, and the
   ones they depend on, `fs_read docs/map/<dotted.package>.md`. The frontmatter
   is generated - files, line count, imports, public names, who imports this
   package - and the prose under it was written by someone who read the code.
3. **The entry points.** Only now open code, and open what the map's `Entry
   points` section named first. `fs_search` for a symbol you saw in `public`
   to find where it is used.
4. **Write down what you learned.** `scratch_write` the paths and the facts you
   will need later. Tool results are evicted as a run grows; your notes are not.

## What the map does not answer

- Whether it is current. A package whose map file is stale describes the code
  as it was, so check `generated_from` against the file you are reading, and
  say so in your final answer if they disagree.
- Anything about a directory with no `__init__.py`: those are listed under
  `## unpackaged` in the index and have no map file.
- What a function does. That is the code, and the map is there to tell you
  which code to read.
