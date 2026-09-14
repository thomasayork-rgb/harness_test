"""Test-wide isolation from the machine the tests run on.

The CLI now resolves a global system prompt from ``$HARNESS_SYSTEM_PROMPT``,
``$XDG_CONFIG_HOME/harness/system.md`` or ``~/.config/harness/system.md``, so a
developer who has one would otherwise change what every CLI-driven test sends
to the mock server. Every test gets a throwaway home and no global prompt; a
test that wants one writes it there itself.

The same goes for git: ``--project`` shells out to it, so every test runs with
the machine's own git configuration switched off (see tests/gitfixture.py).
"""
import pytest


@pytest.fixture(autouse=True)
def _no_real_home(tmp_path, monkeypatch):
    home = tmp_path / "isolated-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("HARNESS_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    return home
