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
``failed``) are refused, as is a run some other process still has open
(``running``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .policy import from_description
from .registry import ToolRegistry
from .runtime import (RESUMABLE, AgentRuntime, ResumeError, RunResult, RunState,
                      effective_config)
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
) -> tuple[AgentRuntime, str | None]:
    """Rebuild the runtime for a resumable run. Returns it with the detail of
    the footer that closed the previous segment, for ``AgentRuntime.resume``.

    ``invocation`` is what the new segment runs under, recorded at the seam for
    the next resume; it defaults to what the recording already says. ``skills``
    is the skill set the next segment can load from - a run that could load
    skills before must still be able to after."""
    path = Path(run_dir)
    state, records = load(path)
    if state.status not in RESUMABLE:
        raise ResumeError(
            f"run {state.run_id} ended with status '{state.status}'; only "
            f"{', '.join(RESUMABLE)} can be resumed")
    config = effective_config(records)
    if step_cap is not None:
        config.step_cap = step_cap
    runtime = AgentRuntime(registry, transport, path.parent, model or state.model, config,
                           run_id=path.name, state=state, policy=policy,
                           invocation=invocation or recorded_invocation(records),
                           skills=skills)
    return runtime, last(records, "footer").get("detail")


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
) -> RunResult:
    """Continue a run in place. The trajectory grows; it is not replaced."""
    runtime, detail = prepare(run_dir, registry, transport, model=model, step_cap=step_cap,
                              policy=policy, invocation=invocation, skills=skills)
    return runtime.resume(detail)
