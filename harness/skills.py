"""Skills: markdown guides the agent discovers and loads on demand.

A skill teaches the model how to do one kind of work. It is discovered lazily,
in the same spirit as the toolbelt: only names and one-line descriptions are
held until the model calls ``skill_load``, and only then does the text enter
the conversation.

Format - ``skills/<name>/SKILL.md`` or ``skills/<name>.md``::

    ---
    name: code-change            # optional; defaults to the directory or file stem
    description: Change code in this project safely.
    tools: [fs_search, fs_read]  # optional; activated when the skill is loaded
    plugin: ./tools.py           # optional; a --tools module, relative to this file
    ---
    The body is the skill: whatever the model should read before doing the work.

The frontmatter is parsed by ``harness.frontmatter``, not by a YAML library:
``key: value`` scalars, ``[a, b]`` inline lists, ``- item`` block lists, and
indented continuation lines for a multi-line value. Anything else in the block
is an error naming the line, because a skill the author thought they wrote is
worse than none. That parser is shared with the project map and task files;
here its errors come back as ``SkillError``, naming the skill file.

Discovery merges every location that exists, lowest precedence first::

    harness/skills_bundled            (--project runs only: orient, map, handoff)
    $XDG_CONFIG_HOME/harness/skills  (else ~/.config/harness/skills)
    $HARNESS_SKILLS                  (os.pathsep-separated)
    <workdir>/.harness/skills        (skills that live with a project)
    --skills DIR                     (repeatable; a later one wins)

A name found twice is a clash: the last directory wins and the clash is
reported, once, rather than silently deciding. A top-level ``.md`` file with no
frontmatter is not a skill and is skipped in silence; one that has frontmatter
but no description is an error, reported the same way.

The bundled skills ship inside the package and are only searched for a
``--project`` run: they are about working in a mapped repository, and a run
pointed at a plain directory has no map, no task and no branch. They are the
lowest precedence there is, so a project that ships its own ``map`` skill wins.
Wherever a location is named - the header, ``harness skills``, the invocation a
resume reads back - theirs is the word ``bundled`` rather than wherever pip
happened to put the package.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import frontmatter

SKILL_FILE = "SKILL.md"
SKILLS_ENV = "HARNESS_SKILLS"
XDG_ENV = "XDG_CONFIG_HOME"
CONFIG_RELATIVE = Path("harness") / "skills"
PROJECT_RELATIVE = Path(".harness") / "skills"
FENCE = frontmatter.FENCE

# The skills that ship with the harness, and what they are called in a record:
# the word, not the install path, so a trajectory says the same thing on every
# machine and a resume can find them again.
BUNDLED = "bundled"
BUNDLED_DIR = Path(__file__).resolve().parent / "skills_bundled"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def bundled_dir() -> Path:
    """Where the skills shipped with the harness live."""
    return BUNDLED_DIR


def resolve_dir(raw: Any) -> Path:
    """A skills directory from a flag or a recording. ``bundled`` is the one
    name that is not a path: it is wherever this package is installed."""
    return bundled_dir() if str(raw) == BUNDLED else Path(raw)


def label_for(directory: Any) -> str:
    """What a directory is called in the header, in ``harness skills`` and in
    the invocation a resume reads back."""
    try:
        return BUNDLED if Path(directory).resolve() == bundled_dir() else str(directory)
    except OSError:                              # a path we cannot resolve is not the bundle
        return str(directory)


class SkillError(Exception):
    """A skill file that cannot be used, with the reason to report."""


class NoFrontmatter(SkillError):
    """No ``---`` block at all: a markdown file that was never a skill."""


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """``(metadata, body)`` for a skill file. Raises SkillError on a block this
    parser cannot read, NoFrontmatter when there is no block at all.

    The parser is ``harness.frontmatter``; this wrapper only re-labels its
    errors, so what a skill file is allowed to say stays one definition.
    """
    try:
        return frontmatter.parse_frontmatter(text)
    except frontmatter.NoFrontmatter as e:
        raise NoFrontmatter(str(e)) from None
    except frontmatter.FrontmatterError as e:
        raise SkillError(str(e)) from None


def _string_list(value: Any, what: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise SkillError(f"{what} must be a list of names")


@dataclass
class Skill:
    """One loaded skill file. ``body`` is what the model reads."""

    name: str
    description: str
    body: str
    path: Path
    source: Path                                  # the skills directory it came from
    tools: list[str] = field(default_factory=list)
    plugin: str | None = None

    @property
    def chars(self) -> int:
        return len(self.body)

    @property
    def plugin_path(self) -> Path | None:
        """The ``--tools`` module this skill registers, resolved against itself."""
        return (self.path.parent / self.plugin).resolve() if self.plugin else None

    def summary(self) -> str:
        """The description's first line: all the model sees before loading."""
        first = self.description.strip().splitlines()[0] if self.description.strip() else ""
        return first[:160]

    def brief(self) -> dict:
        return {"name": self.name, "description": self.summary()}

    def matches(self, needle: str) -> bool:
        return needle in self.name.lower() or needle in self.summary().lower()


def load_skill(path: Any, source: Any = None) -> Skill:
    """Read one skill file. Raises SkillError naming the file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SkillError(f"{path}: cannot read skill: {e.strerror or e}") from None
    try:
        meta, body = parse_frontmatter(text)
    except NoFrontmatter as e:
        raise NoFrontmatter(f"{path}: {e}") from None
    except SkillError as e:
        raise SkillError(f"{path}: {e}") from None

    stem = path.parent.name if path.name == SKILL_FILE else path.stem
    name = meta.get("name") or stem
    if not isinstance(name, str) or not _NAME.match(name):
        raise SkillError(f"{path}: bad skill name {name!r}")
    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillError(f"{path}: a skill needs a 'description' in its frontmatter")
    plugin = meta.get("plugin")
    if plugin is not None and not isinstance(plugin, str):
        raise SkillError(f"{path}: 'plugin' must be a path relative to the skill file")
    try:
        tools = _string_list(meta.get("tools"), "'tools'")
    except SkillError as e:
        raise SkillError(f"{path}: {e}") from None
    return Skill(name=name, description=description.strip(), body=body, path=path,
                 source=Path(source) if source is not None else path.parent,
                 tools=tools, plugin=plugin or None)


@dataclass
class SkillSet:
    """What one run discovered: the directories searched and the skills in them."""

    dirs: list[Path] = field(default_factory=list)
    skills: dict[str, Skill] = field(default_factory=dict)
    clashes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    workdir: Path | None = None

    def __len__(self) -> int:
        return len(self.skills)

    def __contains__(self, name: object) -> bool:
        return name in self.skills

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def names(self) -> list[str]:
        return list(self.skills)

    def list(self, filter: str | None = None) -> list[dict]:
        """Name + first description line per skill, filtered on exactly that text."""
        needle = (filter or "").strip().lower()
        return [s.brief() for s in self.skills.values() if not needle or s.matches(needle)]

    def is_project_skill(self, skill: Skill) -> bool:
        """True for a skill found in ``<workdir>/.harness/skills``: it came with
        the project the run works on, not from the operator's own directories,
        so anything it can execute is that project's code."""
        if self.workdir is None:
            return False
        try:
            return skill.source.resolve() == (Path(self.workdir) / PROJECT_RELATIVE).resolve()
        except OSError:
            return False

    def warnings(self) -> list[str]:
        """Everything discovery wants to say out loud, once."""
        return list(self.errors) + list(self.clashes)

    def locations(self) -> list[str]:
        """The directories searched, as they are named in a record."""
        return [label_for(d) for d in self.dirs]

    def source_of(self, skill: Skill) -> str:
        """Where one skill came from, as a reader should see it."""
        return label_for(skill.source)

    def describe(self) -> dict:
        """What the trajectory header records about discovery."""
        return {"dirs": self.locations(), "names": self.names()}


def _candidates(directory: Path) -> list[tuple[Path, bool]]:
    """``(path, required)`` per skill file in a directory. ``required`` marks a
    ``<name>/SKILL.md``: a directory laid out as a skill that fails to parse is
    a mistake worth reporting, a stray ``notes.md`` is not."""
    out: list[tuple[Path, bool]] = []
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return out
    for entry in entries:
        if entry.is_dir():
            nested = entry / SKILL_FILE
            if nested.is_file():
                out.append((nested, True))
        elif entry.is_file() and entry.suffix == ".md":
            out.append((entry, False))
    return out


def discover(dirs: Iterable[Any], workdir: Any = None) -> SkillSet:
    """Every skill in ``dirs``, lowest precedence first: a later directory wins
    a name clash, and the clash is recorded rather than hidden."""
    searched: list[Path] = []
    skills: dict[str, Skill] = {}
    clashes: list[str] = []
    errors: list[str] = []
    for raw in dirs:
        directory = resolve_dir(raw)
        if not directory.is_dir():
            continue
        searched.append(directory)
        for path, required in _candidates(directory):
            try:
                skill = load_skill(path, directory)
            except NoFrontmatter as e:
                if required:
                    errors.append(str(e))
                continue
            except SkillError as e:
                errors.append(str(e))
                continue
            previous = skills.get(skill.name)
            if previous is not None:
                clashes.append(f"skill '{skill.name}': using {skill.path}, shadowing {previous.path}")
            skills[skill.name] = skill
    ordered = {name: skills[name] for name in sorted(skills)}
    return SkillSet(dirs=searched, skills=ordered, clashes=clashes, errors=errors,
                    workdir=Path(workdir) if workdir is not None else None)


def config_dir(env: Mapping[str, str] | None = None) -> Path:
    """``$XDG_CONFIG_HOME/harness/skills``, else ``~/.config/harness/skills``."""
    env = os.environ if env is None else env
    xdg = env.get(XDG_ENV)
    if xdg:
        return Path(xdg).expanduser() / CONFIG_RELATIVE
    home = env.get("HOME")
    base = Path(home).expanduser() if home else Path.home()
    return base / ".config" / CONFIG_RELATIVE


def search_dirs(flags: Iterable[Any] = (), workdir: Any = None,
                env: Mapping[str, str] | None = None, bundled: bool = False) -> list[Path]:
    """Every skills directory that exists, lowest precedence first.

    bundled < config < ``$HARNESS_SKILLS`` < ``<workdir>/.harness/skills`` <
    ``--skills``: the more specific the location, the later it is searched, and
    the later a directory is searched the more it wins. The skills that ship
    with the harness are the least specific there is, and are searched only
    when the caller asks (a ``--project`` run does).
    """
    env = os.environ if env is None else env
    candidates: list[Path] = [bundled_dir()] if bundled else []
    candidates.append(config_dir(env))
    for part in (env.get(SKILLS_ENV) or "").split(os.pathsep):
        if part.strip():
            candidates.append(Path(part.strip()).expanduser())
    if workdir is not None:
        candidates.append(Path(workdir) / PROJECT_RELATIVE)
    candidates.extend(Path(f).expanduser() for f in flags or ())

    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.is_dir():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out
