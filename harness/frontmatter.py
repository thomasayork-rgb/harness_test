"""Frontmatter: the ``---`` block at the top of a markdown file.

One parser, shared by everything in the harness that keeps structured data
above a prose body - skills (``SKILL.md``), and the map files and task files of
a project. Written with the stdlib alone, so the format is exactly what is
documented here and nothing more:

  ``key: value``            a scalar; the value is a string
  ``key: [a, b]``           an inline list
  ``key:`` + ``- item``     a block list (the items may be indented or not)
  ``key:`` + ``sub: value`` a nested mapping (indented under the key)
  indented continuation     more lines of a scalar that already has a value
  ``"key with spaces":``    a quoted key, at any level

Values are strings, lists of strings, or mappings of those: the parser types
nothing. ``null``, ``true`` and ``123`` come back as the strings they look
like, and the consumer decides what they mean - a map file's ``loc`` is an int
to ``codemap``, a task's ``branch: null`` is "not set" to ``tasks``.

``dump()`` is the inverse and is deterministic: same mapping, same bytes, keys
in the mapping's own order. It is what writes a map file, so a scaffold that
changes nothing rewrites nothing. ``split_frontmatter()`` is the raw splitter
underneath both: it hands back the block and the body untouched, so a writer
can regenerate the block and keep an agent's prose byte for byte.

A round trip is exact for what this format can express::

    parse_frontmatter(dump(meta, body)) == (meta, body.strip())

A scalar is quoted by ``dump`` when it would not survive otherwise: when it is
empty, has leading or trailing space, opens with a quote, or contains ``[``,
``]``, ``:`` or ``#`` (and, inside an inline list, ``,``). What cannot survive
is blank lines and trailing space inside a multi-line value, and a multi-line
value whose first line is empty: those are written as continuation lines, and
continuation lines are stripped on the way back in.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

FENCE = "---"

# A key that needs no quoting. Anything else - a path, a name with a space - is
# written ``"like this"`` and read back the same way.
_PLAIN_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_KEY = re.compile(r'^(?:"((?:[^"\\]|\\.)*)"|([A-Za-z_][A-Za-z0-9_.-]*))\s*:\s*(.*)$')
_ITEM = re.compile(r"^-\s+(.*)$")

# Characters that make a scalar ambiguous to read back: the list brackets, the
# key separator, a comment hash.
_QUOTE_CHARS = "[]:#"


class FrontmatterError(Exception):
    """A block this parser cannot read, with the line to report."""


class NoFrontmatter(FrontmatterError):
    """No ``---`` block at all: a markdown file that never had frontmatter."""


# ---- reading --------------------------------------------------------------


def _unescape(text: str) -> str:
    out: list[str] = []
    escape = False
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
        elif ch == "\\":
            escape = True
        else:
            out.append(ch)
    return "".join(out)


def _scalar(raw: str) -> str:
    """One value: stripped, and unquoted if it is wrapped in matching quotes."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        return _unescape(inner) if text[0] == '"' else inner
    return text


def _split_items(inner: str) -> list[str]:
    """Split an inline list on commas that are not inside a quoted item."""
    items: list[str] = []
    buf: list[str] = []
    quoted = escape = False
    for ch in inner:
        if escape:
            buf.append(ch)
            escape = False
        elif quoted and ch == "\\":
            buf.append(ch)
            escape = True
        elif ch == '"':
            quoted = not quoted
            buf.append(ch)
        elif ch == "," and not quoted:
            items.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    items.append("".join(buf))
    return [_scalar(part) for part in items if part.strip()]


def _inline_list(raw: str) -> list[str] | None:
    """``[a, b]`` as a list, or None when this value is not an inline list."""
    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1].strip()
    return _split_items(inner) if inner else []


def _indent(raw: str) -> int:
    return len(raw) - len(raw.lstrip())


def _block(text: str) -> tuple[list[tuple[int, str]], str]:
    """``(numbered block lines, raw body)``. Line numbers are 1-based in the
    original text, so an error names the line the author would look at."""
    lines = text.lstrip("\ufeff").splitlines(keepends=True)
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].strip() != FENCE:
        raise NoFrontmatter(f"no '{FENCE}' frontmatter block")
    start = i + 1
    for j in range(start, len(lines)):
        if lines[j].strip() == FENCE:
            return ([(n + 1, lines[n].rstrip("\n")) for n in range(start, j)],
                    "".join(lines[j + 1:]))
    raise FrontmatterError(f"frontmatter block is not closed with '{FENCE}'")


def split_frontmatter(text: str) -> tuple[str, str]:
    """``(frontmatter_text, body_text)``, both raw.

    The splitter underneath ``parse_frontmatter``, for a writer that wants to
    regenerate the block and leave the body exactly as the agent wrote it.
    """
    block, body = _block(text)
    return "".join(line + "\n" for _, line in block), body


def _parse_mapping(lines: list[tuple[int, str]], i: int, end: int,
                   indent: int) -> tuple[dict, int]:
    """One mapping at ``indent``, from ``i``. Returns it and where it ended."""
    out: dict[str, Any] = {}
    key: str | None = None
    while i < end:
        lineno, raw = lines[i]
        if not raw.strip():
            i += 1
            continue
        content = raw.strip()
        item = _ITEM.match(content)
        if item is not None and key is not None and isinstance(out.get(key), list):
            out[key].append(_scalar(item.group(1)))
            i += 1
            continue
        depth = _indent(raw)
        if depth < indent:
            break                                  # this block is over
        if depth > indent:
            if key is not None and isinstance(out.get(key), str):
                out[key] = (out[key] + "\n" + content).strip()
                i += 1
                continue
            raise FrontmatterError(
                f"line {lineno}: unexpected indented line: {content[:60]!r}")
        found = _KEY.match(content)
        if not found:
            raise FrontmatterError(
                f"line {lineno}: not a 'key: value' frontmatter line: {content[:60]!r}")
        quoted, plain, value = found.group(1), found.group(2), found.group(3).strip()
        key = _unescape(quoted) if quoted is not None else plain
        i += 1
        if value:
            inline = _inline_list(value)
            out[key] = inline if inline is not None else _scalar(value)
            continue
        # An empty value opens a block: a list, a mapping, or nothing at all.
        # The first indented line decides which.
        j = i
        while j < end and not lines[j][1].strip():
            j += 1
        opens_mapping = (j < end and _indent(lines[j][1]) > indent
                         and not _ITEM.match(lines[j][1].strip()))
        if opens_mapping:
            out[key], i = _parse_mapping(lines, j, end, _indent(lines[j][1]))
        else:
            out[key] = []                          # a '- item' block may follow
    return out, i


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """``(metadata, body)``. Raises FrontmatterError on a block this parser
    cannot read, NoFrontmatter when there is no block at all."""
    lines, body = _block(text)
    meta, _ = _parse_mapping(lines, 0, len(lines), 0)
    return meta, body.strip()


# ---- writing --------------------------------------------------------------


def _text(value: Any) -> str:
    """A scalar as the block spells it. ``null``/``true``/``123`` come back
    from the parser as those strings; the consumer decodes them."""
    if value is None:
        return "null"
    if isinstance(value, bool):                    # before int: bool is an int
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return str(value)


def _quoted(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _quote(text: str, extra: str = "") -> str:
    """A scalar, quoted only when it would not read back as itself."""
    risky = _QUOTE_CHARS + extra
    if (not text or text != text.strip() or text[:1] in "\"'"
            or any(ch in text for ch in risky)):
        return _quoted(text)
    return text


def _key_text(key: Any) -> str:
    """A key: bare when the plain-key shape allows it, quoted when not."""
    name = str(key)
    return name if _PLAIN_KEY.match(name) else _quoted(name)


def _entries(meta: Mapping[str, Any], depth: int) -> list[str]:
    pad = "  " * depth
    out: list[str] = []
    for key, value in meta.items():
        name = _key_text(key)
        if isinstance(value, Mapping):
            out.append(f"{pad}{name}:")
            out.extend(_entries(value, depth + 1))
        elif isinstance(value, (list, tuple)):
            out.append(f"{pad}{name}: [" + ", ".join(_quote(_text(v), extra=",") for v in value) + "]")
        else:
            text = _text(value)
            first, *rest = text.split("\n")
            if rest:
                # a multi-line value: continuation lines, indented under the key
                out.append(f"{pad}{name}: {first}")
                out.extend(f"{pad}  {line.strip()}" for line in rest)
            else:
                out.append(f"{pad}{name}: {_quote(first)}")
    return out


def dump(meta: Mapping[str, Any], body: str = "") -> str:
    """The file ``parse_frontmatter`` would read back: fenced block, blank
    line, body, one trailing newline. Deterministic - keys keep the order the
    mapping has, lists are always inline."""
    lines = [FENCE, *_entries(meta, 0), FENCE]
    text = "\n".join(lines) + "\n"
    trimmed = (body or "").strip("\n")
    return text + (f"\n{trimmed}\n" if trimmed else "")
