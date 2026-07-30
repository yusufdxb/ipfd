"""Point of No Return (PoNR).

Definition
----------
The PoNR is the first timestep from which the task is *irrecoverable*, meaning no
subsequent state ever returns to recoverable. Crucially, "irrecoverable" cannot be
read off a passive log: you only know a state was doomed by *trying to recover from
it and failing*. So PoNR is defined operationally against a **recovery probe**:

    recovery_success[t] == True  <=>  the best-effort recovery controller,
                                      started from the saved sim state at step t,
                                      reaches task success within a fixed budget.

This module computes PoNR from that boolean array. Producing the array requires a
simulator that can save/restore state (see ``ipfd.adapters.isaac_lab``); the array
is the clean interface between "needs a GPU" and "runs in CI".

We report PoNR under the recovery controller we actually ran. A stronger recovery
controller can push the measured timestep later. When positive recovery verdicts
are physically sound, the oracle-relative timestep is therefore a lower bound on
the optimal-control PoNR timestep. A failed attempt is not proof of physical
irrecoverability. Real probes can be noisy, so repeated aggregation reports a confidence value
alongside the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "point_of_no_return",
    "point_of_no_return_repeated",
    "aggregate_repeated_probes",
    "ProbeStatistics",
    "RecoveryProbe",
]


@dataclass(frozen=True)
class ProbeStatistics:
    """Aggregated repeated-probe evidence at each candidate timestep.

    ``verdict`` is deliberately conservative: a candidate is marked recoverable
    unless the observed false fraction reaches ``min_confidence``.  This prevents
    one noisy probe from manufacturing an early PoNR. ``confidence`` is the
    fraction of repeats supporting the returned verdict and ``repeat_count``
    records how much evidence was available.
    """

    verdict: np.ndarray
    confidence: np.ndarray
    repeat_count: np.ndarray
    false_fraction: np.ndarray


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


def _confidence(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number in [0.5, 1.0]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0.5, 1.0]") from exc
    if not np.isfinite(result) or not 0.5 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0.5, 1.0]")
    return result


def _boolean_vector(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind == "b":
        return arr
    if arr.dtype.kind in "iu" and np.isin(arr, (0, 1)).all():
        return arr.astype(bool, copy=False)
    raise ValueError(f"{name} must contain only booleans or integer 0/1 values")


def aggregate_repeated_probes(
    probe_verdicts: np.ndarray,
    *,
    min_repeats: int = 3,
    min_confidence: float = 0.8,
) -> ProbeStatistics:
    """Aggregate repeated recovery probes conservatively.

    Args:
        probe_verdicts: Boolean array shaped ``(candidates, repeats)``. Every
            candidate has the same number of repeats; NaN and missing values are
            rejected.
        min_repeats: Minimum repeats required before a false verdict is trusted.
        min_confidence: Required fraction of repeats agreeing on irrecoverable.

    A candidate is considered irrecoverable only when at least ``min_repeats``
    probes exist and at least ``min_confidence`` of them are false. Otherwise it
    remains conservatively recoverable.
    """
    arr = np.asarray(probe_verdicts)
    if arr.ndim != 2:
        raise ValueError(f"probe_verdicts must be 2-D, got shape {arr.shape}")
    min_repeats = _positive_int(min_repeats, "min_repeats")
    min_confidence = _confidence(min_confidence, "min_confidence")
    if arr.dtype.kind != "b" and not (
        arr.dtype.kind in "iu" and np.isin(arr, (0, 1)).all()
    ):
        raise ValueError("probe_verdicts must contain only booleans or integer 0/1 values")
    arr = arr.astype(bool, copy=False)
    repeats = np.full(arr.shape[0], arr.shape[1], dtype=int)
    false_fraction = (~arr).mean(axis=1) if arr.shape[1] else np.zeros(arr.shape[0])
    confidence = np.maximum(false_fraction, 1.0 - false_fraction)
    if arr.shape[1] == 0:
        confidence.fill(0.0)
    irrecoverable = (repeats >= min_repeats) & (false_fraction >= min_confidence)
    verdict = ~irrecoverable
    return ProbeStatistics(verdict, confidence, repeats, false_fraction)


def point_of_no_return_repeated(
    probe_verdicts: np.ndarray,
    *,
    min_repeats: int = 3,
    min_confidence: float = 0.8,
) -> int | None:
    """Compute PoNR from repeated probes using conservative aggregation."""
    stats = aggregate_repeated_probes(
        probe_verdicts, min_repeats=min_repeats, min_confidence=min_confidence
    )
    return point_of_no_return(stats.verdict)


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
    rec = _boolean_vector(recovery_success, "recovery_success")
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
