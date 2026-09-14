"""Resume an interrupted run.

A run that ended with ``transport_error``, ``step_cap`` or ``stalled`` stopped
for a reason outside the task: the endpoint fell over, the budget ran out, the
model went quiet. Everything needed to carry on is already on disk - the
conversation and tool state in ``state.json``, the configuration, the policy
and the invocation (endpoint, provider, workdir, tool modules; never the API
key) in the trajectory header - so resuming is rebuilding the runtime around
that state and calling ``resume()`` instead of ``run()``::

    from harness.resume import resume

    result = resume("runs/20240101-120000-abc123", registry, transport, step_cap=400)

The run keeps its id, its directory, and its trajectory: the new segment is
appended after a ``resume`` record and the step counter carries on. Statuses
that are answers rather than interruptions (``completed``, ``blocked``,
``failed``) are refused.

A run whose state still says ``running`` was never closed: either a process
still owns it, or one died without writing a footer. The lock file says which.
While the pid in it is alive the run is refused - two processes appending to
one trajectory is not a thing to guess at - and ``force=True`` is how an
operator who knows better says so. A lock nobody holds is stale, so the run is
picked up as an interruption like any other.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .policy import from_description
from .registry import ToolRegistry
from .runtime import (LOCK_FILE, RESUMABLE, AgentRuntime, ResumeError, RunResult, RunState,
                      effective_config, pid_alive, read_lock)
from .trajectory import last, read_trajectory, setting
from .transport import Transport


def recorded_invocation(records: list[dict]) -> dict:
    """How the last segment was launched: endpoint, provider, workdir, tool
    modules, request options. ``{}`` for a run recorded before this existed."""
    return setting(records, "invocation") or {}


def recorded_skill_dirs(records: list[dict]) -> list[str]:
    """The skill directories the last segment searched. ``[]`` for a run that
    had none, or one recorded before skills existed."""
    return list(recorded_invocation(records).get("skills") or [])


def recorded_policy(records: list[dict]) -> Any:
    """The tool-call policy the last segment ran under, rebuilt, or None."""
    return from_description(setting(records, "policy"))


def load(run_dir: Any) -> tuple[RunState, list[dict]]:
    """``(state, records)`` for a run directory. Raises FileNotFoundError if
    either half of the run is missing, ResumeError if the state is unreadable."""
    path = Path(run_dir)
    state_path = path / "state.json"
    records = read_trajectory(path)
    if not state_path.exists():
        raise FileNotFoundError(f"no state at {state_path}")
    try:
        return RunState.load(state_path), records
    except (ValueError, TypeError) as e:
        raise ResumeError(f"{state_path}: cannot read run state: {e}") from None


def claim(run_dir: Path, state: RunState, force: bool) -> str | None:
    """Settle a run whose state still says ``running``, and say what happened.

    Returns the detail for the resume note, or None if the run was closed
    properly and its footer already says. Raises ResumeError while another
    process is demonstrably still on it.
    """
    if state.status != "running":
        return None
    holder = read_lock(run_dir)
    if holder is not None and pid_alive(holder):
        if not force:
            raise ResumeError(
                f"run {state.run_id} is still running: process {holder} holds "
                f"{LOCK_FILE} in {run_dir}. Wait for it to finish, or pass --force "
                "if you know that process is gone.")
        state.status = "interrupted"
        return f"resumed with --force while pid {holder} still held {LOCK_FILE}"
    state.status = "interrupted"
    return (f"the process that held this run (pid {holder}) is gone" if holder is not None
            else "the run stopped without writing a footer, and nothing holds its lock")


def prepare(
    run_dir: Any,
    registry: ToolRegistry,
    transport: Transport,
    *,
    model: str | None = None,
    step_cap: int | None = None,
    policy: Any = None,
    invocation: dict | None = None,
    skills: Any = None,
    progress_nudge_steps: int | None = None,
    trust_project_plugins: bool | None = None,
    force: bool = False,
) -> tuple[AgentRuntime, str | None]:
    """Rebuild the runtime for a resumable run. Returns it with the detail of
    the footer that closed the previous segment, for ``AgentRuntime.resume``.

    ``invocation`` is what the new segment runs under, recorded at the seam for
    the next resume; it defaults to what the recording already says. ``skills``
    is the skill set the next segment can load from - a run that could load
    skills before must still be able to after. ``force`` takes over a run whose
    lock is still held (see ``claim``)."""
    path = Path(run_dir)
    state, records = load(path)
    detail = claim(path, state, force)
    if state.status not in RESUMABLE:
        raise ResumeError(
            f"run {state.run_id} ended with status '{state.status}'; only "
            f"{', '.join(RESUMABLE)} can be resumed")
    config = effective_config(records)
    if step_cap is not None:
        config.step_cap = step_cap
    if progress_nudge_steps is not None:
        config.progress_nudge_steps = progress_nudge_steps
    if trust_project_plugins is not None:
        config.trust_project_plugins = trust_project_plugins
    runtime = AgentRuntime(registry, transport, path.parent, model or state.model, config,
                           run_id=path.name, state=state, policy=policy,
                           invocation=invocation or recorded_invocation(records),
                           skills=skills)
    # a run that was never closed has no footer of its own; any footer in the
    # file belongs to an earlier segment and would misname what stopped this one
    return runtime, detail if detail is not None else last(records, "footer").get("detail")


def resume(
    run_dir: Any,
    registry: ToolRegistry,
    transport: Transport,
    *,
    model: str | None = None,
    step_cap: int | None = None,
    policy: Any = None,
    invocation: dict | None = None,
    skills: Any = None,
    progress_nudge_steps: int | None = None,
    trust_project_plugins: bool | None = None,
    force: bool = False,
) -> RunResult:
    """Continue a run in place. The trajectory grows; it is not replaced."""
    runtime, detail = prepare(run_dir, registry, transport, model=model, step_cap=step_cap,
                              policy=policy, invocation=invocation, skills=skills,
                              progress_nudge_steps=progress_nudge_steps,
                              trust_project_plugins=trust_project_plugins, force=force)
    return runtime.resume(detail)
