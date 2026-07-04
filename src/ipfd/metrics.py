"""The minimal metric set.

Four numbers, each answering one operational question an engineer asks when a
policy silently degrades. All are reported in seconds where they are durations,
using the rollout's ``dt``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "time_to_failure",
    "failure_lead_time",
    "ponr_lead_time",
    "false_continuity_rate",
    "drift_magnitude_at_collapse",
    "silent_doom_window",
]


def time_to_failure(t_failure: int | None, dt: float) -> float | None:
    """Seconds from episode start to *externally observable* failure.

    ``None`` if the episode succeeded (no observable failure).
    """
    return None if t_failure is None else t_failure * dt


def failure_lead_time(t_alarm: int | None, t_failure: int | None, dt: float) -> float | None:
    """Seconds by which the detector alarm *precedes observable failure*.

    Positive = the tool would have paged you before anything looked wrong.
    ``None`` if there was no alarm or no observable failure.
    """
    if t_alarm is None or t_failure is None:
        return None
    return (t_failure - t_alarm) * dt


def ponr_lead_time(t_alarm: int | None, t_ponr: int | None, dt: float) -> float | None:
    """Seconds between the alarm and the point of no return.

    Positive  = alarm fired *before* the trajectory became irrecoverable
                (actionable warning: you could still have intervened).
    Negative  = alarm fired *after* doom was sealed (too late to matter).
    ``None`` if either event is undefined.
    """
    if t_alarm is None or t_ponr is None:
        return None
    return (t_ponr - t_alarm) * dt


def silent_doom_window(t_ponr: int | None, t_failure: int | None, dt: float) -> float | None:
    """Seconds the system spent doomed-but-not-yet-visibly-failed.

    The gap between PoNR and observable failure. This is the interval during which
    a naive success-rate evaluation still counts the episode as "fine".
    """
    if t_ponr is None or t_failure is None:
        return None
    return max(0.0, (t_failure - t_ponr) * dt)


def false_continuity_rate(
    imminence: np.ndarray,
    t_ponr: int | None,
    t_failure: int | None,
    threshold: float = 0.5,
) -> float | None:
    """Fraction of the doomed window in which the detector stayed *quiet*.

    Over ``[t_ponr, t_failure)`` -- the interval where the task is already
    irrecoverable -- this is the fraction of timesteps whose imminence score sits
    *below* ``threshold``. High false continuity means the policy sailed on looking
    confident and healthy while already doomed: the failure mode this tool targets.

    ``None`` if the doomed window is undefined or empty.
    """
    if t_ponr is None or t_failure is None or t_failure <= t_ponr:
        return None
    window = imminence[t_ponr:t_failure]
    if window.size == 0:
        return None
    return float(np.mean(window < threshold))


def drift_magnitude_at_collapse(drift: np.ndarray, t_ponr: int | None) -> float | None:
    """Raw representation-drift value at the point of no return.

    ``None`` if there is no PoNR or no drift signal.
    """
    if t_ponr is None or drift.size == 0 or t_ponr >= drift.size:
        return None
    return float(drift[t_ponr])
