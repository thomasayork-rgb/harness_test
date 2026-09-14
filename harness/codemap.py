"""The code map: one markdown file per package, under ``docs/map/``.

A map file is half generated and half written::

    docs/map/harness.tools.md

    ---                        <- generated: files, loc, imports, public names,
    map_version: 1                who imports this package, and the blob sha of
    package: harness.tools        every file the frontmatter was generated from
    ...
    ---

    ## Purpose                 <- the agent's prose, preserved byte for byte
    ...

The frontmatter is what a machine can see without understanding anything: the
file list, the line count, the import graph, the public names, the inbound
references. The body is what only a reader can say: what the package is for,
where to enter it, what must stay true, what will surprise you. Scaffolding
writes the first and never touches the second, so running it again after a
refactor updates the facts and keeps the prose.

``generated_from`` is what makes staleness answerable: the git blob sha of each
file as it was when the frontmatter was written, computed in Python
(``sha1("blob <len>\\0" + bytes)``) so it means the same thing as ``git
ls-tree``. Nothing here records a timestamp, so the same tree always produces
the same bytes, and a scaffold that changed nothing rewrites nothing.

Nothing here imports or executes the project: every file is read as text and
parsed with ``ast``. A file that does not parse is listed under ``errors`` and
otherwise counted like any other.
"""
from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .frontmatter import FrontmatterError, NoFrontmatter, dump, parse_frontmatter, split_frontmatter
from .tools.paths import SKIP_DIRS

MAP_RELATIVE = Path("docs") / "map"
INDEX_FILE = "INDEX.md"
MAP_VERSION = 1
INIT = "__init__.py"

# docs/ is where the map itself lives; the rest is the walking tools' own noise
# list. A directory with its own .git is another repository (or one of this
# project's worktrees), not part of this project's code.
SKIP = frozenset(SKIP_DIRS) | {"docs"}

# The sections a new map file starts with, in this order. Each is a question the
# frontmatter cannot answer.
SECTIONS = ("Purpose", "Entry points", "Invariants", "Gotchas", "Depends on", "Depended on by")
TEMPLATE = "\n\n".join(f"## {name}" for name in SECTIONS)
UNMAPPED = "(unmapped)"
CALLERS_SHOWN = 5


def blob_sha(data: bytes) -> str:
    """The sha ``git ls-tree`` would print for these bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def dotted(root: Path, path: Path) -> str:
    """A path as a dotted name relative to the project root."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts.pop()
    return ".".join(parts)


def walk_dirs(root: Path) -> list[Path]:
    """Every directory of the project worth looking at, deterministic order."""
    out: list[Path] = []
    for current, dirnames, _ in os.walk(root):
        here = Path(current)
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP and not d.startswith(".")
                             and not (here / d / ".git").exists())
        out.append(here)
    return sorted(out)


@dataclass
class Module:
    """One ``.py`` file: what it imports, what it exports, whether it parses."""

    path: Path
    rel: str
    dotted: str
    package: str | None                  # the dotted package it belongs to, if any
    data: bytes
    loc: int
    imports: list[str] = field(default_factory=list)
    public: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def sha(self) -> str:
        return blob_sha(self.data)


@dataclass
class Package:
    """One package: its own files (a subpackage is its own entry)."""

    dotted: str
    directory: Path
    modules: list[Module]
    inbound: int = 0
    callers: dict = field(default_factory=dict)

    @property
    def top(self) -> str:
        return self.dotted.split(".")[0]

    @property
    def files(self) -> list[str]:
        return sorted(m.rel for m in self.modules)

    @property
    def loc(self) -> int:
        return sum(m.loc for m in self.modules)

    @property
    def errors(self) -> list[str]:
        return sorted(m.rel for m in self.modules if m.error)

    def map_file(self, root: Path) -> Path:
        return root / MAP_RELATIVE / f"{self.dotted}.md"


def _imports_of(tree: ast.AST, package: str | None) -> list[str]:
    """Every module this file imports, as a dotted name. A relative import is
    resolved against the package the file is in, so the map speaks one
    language."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if not node.level:
                if node.module:
                    names.append(node.module)
                continue
            base = (package or "").split(".") if package else []
            up = node.level - 1
            base = base[:len(base) - up] if up <= len(base) else []
            parts = base + ([node.module] if node.module else [])
            if parts:
                names.append(".".join(parts))
    return sorted(set(names))


def _public_of(tree: ast.Module) -> list[str]:
    """Module-level names that are not underscored. A literal ``__all__`` wins:
    a package that says what it exports is answering this question itself."""
    declared: list[str] | None = None
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                found.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    declared = [e.value for e in node.value.elts
                                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                elif not target.id.startswith("_"):
                    found.add(target.id)
    return sorted(set(declared)) if declared is not None else sorted(found)


def read_module(root: Path, path: Path, package: str | None) -> Module:
    """One file, read and parsed. A file that will not parse is still a file:
    it counts toward loc and generated_from, and it is named in ``errors``."""
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    module = Module(path=path, rel=path.relative_to(root).as_posix(), dotted=dotted(root, path),
                    package=package, data=data, loc=len(text.splitlines()))
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as e:
        module.error = f"{type(e).__name__}: {e}"
        return module
    module.imports = _imports_of(tree, package)
    module.public = _public_of(tree)
    return module


@dataclass
class Project:
    """Every package of a project, and every file that imports one."""

    root: Path
    packages: dict[str, Package]
    modules: list[Module]
    unpackaged: list[str]

    def names(self) -> list[str]:
        return list(self.packages)

    def select(self, package: str | None = None) -> list[Package]:
        """The packages a command works on: one subtree, or all of them."""
        if not package:
            return list(self.packages.values())
        return [p for name, p in self.packages.items()
                if name == package or name.startswith(package + ".")]


def scan(root: Any) -> Project:
    """Read the project: packages, modules, imports, inbound references."""
    root = Path(root).resolve()
    dirs = walk_dirs(root)
    package_dirs = {d: dotted(root, d) for d in dirs if (d / INIT).is_file()}

    modules: list[Module] = []
    by_package: dict[str, list[Module]] = {name: [] for name in package_dirs.values()}
    unpackaged: list[str] = []
    for directory in dirs:
        files = sorted(p for p in directory.glob("*.py") if p.is_file())
        if not files:
            continue
        package = package_dirs.get(directory)
        if package is None:
            rel = directory.relative_to(root).as_posix()
            unpackaged.append(f"{rel}/" if rel != "." else "./")
        for path in files:
            module = read_module(root, path, package)
            modules.append(module)
            if package is not None:
                by_package[package].append(module)

    packages = {name: Package(dotted=name, directory=directory, modules=by_package[name])
                for directory, name in sorted(package_dirs.items(), key=lambda kv: kv[1])}
    for name, package in packages.items():
        callers: dict[str, int] = {}
        for module in modules:
            if module.package == name:
                continue                         # a package importing itself is not inbound
            hits = sum(1 for imported in module.imports
                       if imported == name or imported.startswith(name + "."))
            if hits:
                callers[module.dotted] = hits
        package.inbound = len(callers)
        package.callers = dict(sorted(callers.items(), key=lambda kv: (-kv[1], kv[0]))[:CALLERS_SHOWN])
    return Project(root=root, packages=packages, modules=modules, unpackaged=sorted(unpackaged))


def frontmatter_for(project: Project, package: Package) -> dict:
    """The generated half of a map file, in one fixed key order."""
    tops = {name.split(".")[0] for name in project.packages}
    internal: set[str] = set()
    external: set[str] = set()
    public: dict[str, list[str]] = {}
    for module in sorted(package.modules, key=lambda m: m.rel):
        for imported in module.imports:
            top = imported.split(".")[0]
            if top in tops:
                internal.add(imported)           # the module, not the symbol
            else:
                external.add(top)                # stdlib and third party: the top level
        if module.public:
            public[module.rel] = module.public

    meta: dict[str, Any] = {
        "map_version": MAP_VERSION,
        "package": package.dotted,
        "files": package.files,
        "loc": package.loc,
        "imports": {"internal": sorted(internal), "external": sorted(external)},
    }
    if public:
        meta["public"] = public
    inbound: dict[str, Any] = {"count": package.inbound}
    if package.callers:
        inbound["callers"] = dict(package.callers)
    meta["inbound_refs"] = inbound
    meta["generated_from"] = {m.rel: m.sha for m in sorted(package.modules, key=lambda m: m.rel)}
    if package.errors:
        meta["errors"] = package.errors
    return meta


def section(body: str, name: str) -> str:
    """The text under one ``## Heading``, stripped. ``""`` when it is empty or
    the heading is not there."""
    wanted = f"## {name}".lower()
    out: list[str] = []
    collecting = False
    for line in (body or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## "):
            if collecting:
                break
            collecting = stripped.lower() == wanted
            continue
        if collecting:
            out.append(line)
    return "\n".join(out).strip()


def purpose_of(path: Path) -> str | None:
    """The Purpose section of a map file, or None when there is no file. An
    empty Purpose is ``""``: the file exists and says nothing yet."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        _, body = parse_frontmatter(text)
    except (NoFrontmatter, FrontmatterError):    # a block we cannot read still has prose
        body = text
    return section(body, "Purpose")


def body_of(path: Path) -> str:
    """The prose of an existing map file, or the template for a new one. A file
    whose frontmatter cannot be read keeps its whole text as prose: the map is
    the agent's writing, and writing is not something to drop on a parse error."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return TEMPLATE
    try:
        _, body = split_frontmatter(text)
    except FrontmatterError:
        return text.strip("\n")
    return body.strip("\n") or TEMPLATE


def index_text(project: Project) -> str:
    """``docs/map/INDEX.md``: every package, grouped by its top-level package,
    the most depended-on first, with the first line of its Purpose."""
    rows: dict[str, list[tuple[int, str, str, bool]]] = {}
    for name, package in project.packages.items():
        purpose = purpose_of(package.map_file(project.root))
        first = (purpose or "").splitlines()[0].strip() if purpose else ""
        rows.setdefault(package.top, []).append(
            (package.inbound, name, first or UNMAPPED, not first))
    lines = ["# Map index"]
    groups = sorted(rows.items(), key=lambda kv: (-max(r[0] for r in kv[1]), kv[0]))
    for top, entries in groups:
        lines.append("")
        lines.append(f"## {top}")
        for inbound, name, first, stale in sorted(entries, key=lambda r: (-r[0], r[1])):
            lines.append(f"- {name} (inbound {inbound}): {first}" + (" [stale]" if stale else ""))
    if project.unpackaged:
        lines.append("")
        lines.append("## unpackaged")
        lines.extend(f"- {d}" for d in project.unpackaged)
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> str:
    """Write only when the bytes change, and say what happened."""
    existed = path.exists()
    if existed and path.read_text(encoding="utf-8") == text:
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "refreshed" if existed else "wrote"


def scaffold(root: Any, package: str | None = None,
             project: Project | None = None) -> list[tuple[str, str]]:
    """Write or refresh the map files and the index.

    Returns ``(action, relative path)`` per file, where action is ``wrote``,
    ``refreshed`` or ``unchanged``. The index always covers the whole project,
    even when ``package`` narrows what is refreshed: an index that lists half a
    project is worse than one that is a little out of date.
    """
    root = Path(root).resolve()
    project = project or scan(root)
    out: list[tuple[str, str]] = []
    for pkg in project.select(package):
        path = pkg.map_file(root)
        text = dump(frontmatter_for(project, pkg), body_of(path))
        out.append((_write(path, text), path.relative_to(root).as_posix()))
    index = root / MAP_RELATIVE / INDEX_FILE
    out.append((_write(index, index_text(project)), index.relative_to(root).as_posix()))
    return out
