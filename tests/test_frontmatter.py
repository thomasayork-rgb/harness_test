"""The shared frontmatter parser and writer.

``harness.frontmatter`` is what reads a SKILL.md, a map file and a task file,
so the property that matters is the round trip: what ``dump`` writes,
``parse_frontmatter`` reads back unchanged, and what it reads back ``dump``
writes byte for byte. The map files of a project are regenerated on every
scaffold, so "byte for byte" is what keeps a scaffold that changed nothing
from touching anything.
"""
from pathlib import Path

import pytest

from harness.frontmatter import (FrontmatterError, NoFrontmatter, dump, parse_frontmatter,
                                 split_frontmatter)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Everything the format can say, in the canonical form ``dump`` writes.
RICH = '''---
name: rich-example
description: Frontmatter: scalars, lists and mappings.
  A second line, kept as a continuation.
tools: [fs_read, fs_edit]
steps: [read the file, write it back again]
imports:
  internal: [harness.registry]
  external: [os, pathlib]
public:
  "harness/tools/basic.py": [child_env, kill_group]
inbound_refs:
  count: 4
  callers:
    harness.cli: 3
    tests.test_tools_fs: 1
branch: null
run_id: 123
done: true
empty: []
blank: ""
quoted: "a value with: a colon and [brackets]"
---

# Body

Prose below the fence, kept as it was written.
'''

META = {
    "name": "rich-example",
    "description": "Frontmatter: scalars, lists and mappings.\nA second line, kept as a continuation.",
    "tools": ["fs_read", "fs_edit"],
    "steps": ["read the file", "write it back again"],
    "imports": {"internal": ["harness.registry"], "external": ["os", "pathlib"]},
    "public": {"harness/tools/basic.py": ["child_env", "kill_group"]},
    "inbound_refs": {"count": "4", "callers": {"harness.cli": "3", "tests.test_tools_fs": "1"}},
    "branch": "null",
    "run_id": "123",
    "done": "true",
    "empty": [],
    "blank": "",
    "quoted": "a value with: a colon and [brackets]",
}


def test_parse_reads_scalars_lists_nesting_and_quoted_keys():
    meta, body = parse_frontmatter(RICH)
    assert meta == META
    assert body == "# Body\n\nProse below the fence, kept as it was written."


def test_round_trip_is_identity_both_ways():
    meta, body = parse_frontmatter(RICH)
    assert dump(meta, body) == RICH                      # canonical in, canonical out
    assert parse_frontmatter(dump(meta, body)) == (meta, body)
    assert dump(meta, body) == dump(meta, body)          # deterministic


def test_values_stay_strings_so_the_consumer_decides():
    meta, _ = parse_frontmatter(RICH)
    assert meta["branch"] == "null" and meta["run_id"] == "123" and meta["done"] == "true"
    # ... and dump writes those back for the types they came from
    assert dump({"branch": None, "run_id": 123, "done": True, "off": False}) == (
        "---\nbranch: null\nrun_id: 123\ndone: true\noff: false\n---\n")
    assert parse_frontmatter(dump({"branch": None, "run_id": 123}))[0] == {
        "branch": "null", "run_id": "123"}


def test_dump_keeps_key_order_and_quotes_only_what_needs_it():
    text = dump({"z": "last", "a": "first", "path": "docs/map/x.md", "hash": "a#b",
                 "spaced": " padded ", "commas": ["a, b", "c"]})
    assert text == ("---\n"
                    "z: last\n"
                    "a: first\n"
                    "path: docs/map/x.md\n"
                    'hash: "a#b"\n'
                    'spaced: " padded "\n'
                    'commas: ["a, b", c]\n'
                    "---\n")
    assert parse_frontmatter(text)[0]["commas"] == ["a, b", "c"]
    assert parse_frontmatter(text)[0]["spaced"] == " padded "


def test_dump_escapes_quotes_and_backslashes_inside_a_quoted_scalar():
    meta = {"weird": 'a "quoted" \\ value: yes', "key with space": "v"}
    text = dump(meta)
    assert text == ('---\n'
                    'weird: "a \\"quoted\\" \\\\ value: yes"\n'
                    '"key with space": v\n'
                    '---\n')
    assert parse_frontmatter(text)[0] == meta


def test_a_block_list_and_an_empty_key_still_parse_as_before():
    meta, _ = parse_frontmatter("---\nname: x\ntools:\n  - a\n  - b\nnothing:\nplugin: './p.py'\n---\nbody\n")
    assert meta == {"name": "x", "tools": ["a", "b"], "nothing": [], "plugin": "./p.py"}


def test_a_block_list_normalises_to_an_inline_one_and_then_holds_still():
    """A hand-written file is read, then rewritten in canonical form; the
    second pass changes nothing, which is what makes a refresh idempotent."""
    written = "---\nsteps:\n  - one\n  - two\n---\n\nbody\n"
    once = dump(*parse_frontmatter(written))
    assert once == "---\nsteps: [one, two]\n---\n\nbody\n"
    assert dump(*parse_frontmatter(once)) == once


def test_split_frontmatter_hands_back_the_body_byte_for_byte():
    text = "---\nk: v\n---\n\n# Title  \n\nline with trailing space  \n\n\n"
    block, body = split_frontmatter(text)
    assert block == "k: v\n"
    assert body == "\n# Title  \n\nline with trailing space  \n\n\n"
    # what a writer does with it: new block, same prose
    assert dump({"k": "w"}, body) == "---\nk: w\n---\n\n# Title  \n\nline with trailing space  \n"


def test_errors_name_the_line():
    with pytest.raises(NoFrontmatter, match="no '---' frontmatter"):
        parse_frontmatter("# just markdown\n")
    with pytest.raises(FrontmatterError, match="is not closed"):
        parse_frontmatter("---\nname: x\ndescription: d\n")
    with pytest.raises(FrontmatterError, match="line 2: not a 'key: value'"):
        parse_frontmatter("---\njust prose\n---\n")
    with pytest.raises(FrontmatterError, match="line 4: not a 'key: value'"):
        parse_frontmatter("\n---\nname: x\n= nonsense\n---\n")
    with pytest.raises(FrontmatterError, match="line 3: unexpected indented line"):
        parse_frontmatter("---\nname: [a]\n  stray: 1\n---\n")


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "examples" / "skills").glob("*/SKILL.md")),
                         ids=lambda p: p.parent.name)
def test_shipped_skills_are_in_canonical_form(path):
    """Every skill the repository ships is what ``dump`` would write, so the
    writer and the files cannot drift apart unnoticed."""
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    assert dump(meta, body) == text
