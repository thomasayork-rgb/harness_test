"""Fixture git repositories, hermetic from the machine they run on.

Every git call here passes an identity on the command line and runs with
``GIT_CONFIG_GLOBAL=/dev/null`` and ``GIT_CONFIG_NOSYSTEM=1``, so a developer's
own ``~/.gitconfig`` - a default branch name, a commit template, a signing key -
cannot change what these tests see. ``tests/conftest.py`` puts those two
variables in the environment for every test, so the git the harness itself
shells out to is hermetic in the same way.

``sample_project`` is the fixture the project, map and task tests share: two
packages that import each other, a nested subpackage, a directory of modules
that is not a package, a file that does not parse, two tasks and a project
config.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_TERMINAL_PROMPT": "0"}
IDENTITY = ("-c", "user.name=test", "-c", "user.email=test@local")


def git(repo, *args, check: bool = True) -> str:
    """One git command in ``repo``, with a fixture identity and no user config."""
    proc = subprocess.run(["git", *IDENTITY, *[str(a) for a in args]], cwd=str(repo),
                          capture_output=True, text=True, timeout=60,
                          env=dict(os.environ, **GIT_ENV))
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(str(a) for a in args)} failed in {repo}: "
                             f"{proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_all(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        write(root / rel, text)
    return root


def commit_all(repo: Path, message: str = "change") -> str:
    """Stage everything and commit it. Returns the new HEAD sha."""
    git(repo, "add", "-A")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


def git_repo(path: Path, files: dict | None = None, message: str = "initial") -> Path:
    """A repository at ``path`` with one commit holding ``files``."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-b", "main")
    write_all(path, files or {"README.md": "# fixture\n"})
    commit_all(path, message)
    return path


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


def branches(repo: Path) -> list[str]:
    return sorted(line.strip() for line in
                  git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
                  if line.strip())


# ---- the shared fixture project -------------------------------------------

PROJECT_FILES = {
    "alpha/__init__.py": '"""The alpha package."""\nfrom .core import Engine\n\n__all__ = ["Engine"]\n',
    "alpha/core.py": (
        "import os\nfrom beta.util import helper\n\n"
        "VERSION = \"1\"\n_SECRET = \"hidden\"\n\n\n"
        "class Engine:\n    def start(self):\n        return helper(os.sep)\n\n\n"
        "def build(name):\n    return Engine()\n\n\n"
        "def _private():\n    return None\n"),
    "alpha/nested/__init__.py": "",
    "alpha/nested/deep.py": (
        "from ..core import Engine\nfrom . import __name__ as here\n\n\n"
        "def dive():\n    return Engine()\n"),
    "beta/__init__.py": "",
    "beta/util.py": (
        "import json\nimport pathlib\n\n\n"
        "def helper(sep):\n    return json.dumps({\"sep\": sep, \"root\": str(pathlib.Path('.'))})\n"),
    "beta/broken.py": "def oops(\n",
    "scripts/tool.py": "import alpha.core\n\nprint(alpha.core.VERSION)\n",
    "tasks/first.md": ("---\nstatus: todo\narea: [alpha]\n"
                       "done_when: [alpha starts the engine, the map for alpha is current]\n"
                       "branch: null\nrun_id: null\n---\n\nMake the engine start.\n"),
    "tasks/second.md": ("---\nstatus: todo\narea: [beta]\ndone_when: [the helper is helpful]\n"
                        "branch: null\nrun_id: null\n---\n\nMake the helper helpful.\n"),
    ".harness/project.json": '{"test": "python3 -c \\"import sys; sys.exit(0)\\"", "timeout": 30}\n',
    "README.md": "# sample project\n",
}


def sample_project(path: Path) -> Path:
    """The fixture project: packages with cross imports, a namespace directory,
    a file that does not parse, tasks, and a project config."""
    return git_repo(path, dict(PROJECT_FILES), message="sample project")
