"""Point of No Return (PoNR).

The central concept of IPFD, and the one most often faked by naive tooling.

Definition
----------
The PoNR is the first timestep from which the task is *irrecoverable* -- i.e. no
subsequent state ever returns to recoverable. Crucially, "irrecoverable" cannot be
read off a passive log: you only know a state was doomed by *trying to recover from
it and failing*. So PoNR is defined operationally against a **recovery probe**:

    recovery_success[t] == True  <=>  the best-effort recovery controller,
                                      started from the saved sim state at step t,
                                      reaches task success within a fixed budget.

This module computes PoNR from that boolean array. Producing the array requires a
simulator that can save/restore state (see ``ipfd.adapters.isaac_lab``); the array
is the clean interface between "needs a GPU" and "runs in CI".

We report PoNR under the recovery controller we actually ran. It is a *sound upper
bound* on the true (optimal-control) PoNR: a better controller can only push PoNR
later, never earlier. We say "irrecoverable under the provided recovery oracle",
not "provably irrecoverable", and the README is explicit about this.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["point_of_no_return", "RecoveryProbe"]


def point_of_no_return(recovery_success: np.ndarray | None) -> int | None:
    """First timestep from which recovery never again succeeds.

    Args:
        recovery_success: ``(T,)`` boolean array, or ``None`` if no probe was run.

    Returns:
        The PoNR index, or ``None`` if the task stayed recoverable at some point at
        or after every timestep (i.e. it never became permanently doomed), or if no
        probe data is available.
    """
    if recovery_success is None:
        return None
    rec = np.asarray(recovery_success, dtype=bool)
    if rec.size == 0 or rec.all():
        return None
    # Walk backwards: PoNR is the index just after the last recoverable step.
    last_recoverable = np.max(np.nonzero(rec)[0]) if rec.any() else -1
    ponr = last_recoverable + 1
    return int(ponr) if ponr < rec.size else None


@runtime_checkable
class RecoveryProbe(Protocol):
    """Interface a simulator adapter implements to generate ``recovery_success``.

    ``can_recover`` is called on the *saved state* at each candidate timestep. The
    adapter is responsible for restoring that state, running the recovery controller
    for a fixed budget, and returning whether the task was reached.
    """

    def can_recover(self, saved_state: object) -> bool:  # pragma: no cover - interface
        ...
