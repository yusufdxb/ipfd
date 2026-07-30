"""Causal actionability metrics for IPFD alarms.

An alarm is useful only when it follows a known disturbance and precedes the
irrecoverable interval. Recovery probes are often sampled at a stride, so this
module reports an uncertainty interval instead of pretending PoNR is more
precise than the evidence supports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .report import FailureDebugReport

__all__ = [
    "ActionabilityReport",
    "PointOfNoReturnInterval",
    "evaluate_actionability",
    "point_of_no_return_interval",
]


@dataclass(frozen=True)
class PointOfNoReturnInterval:
    """Bounds on PoNR when recovery probes were sampled every N steps."""

    earliest: int
    latest: int
    probe_stride: int


@dataclass(frozen=True)
class ActionabilityReport:
    """Causal classification of one detector alarm against one disturbance."""

    disturbance_onset: int
    ponr_earliest: int | None
    ponr_latest: int | None
    t_alarm: int | None
    t_failure: int | None
    window_status: str
    alarm_relation: str
    valid_actionable_warning: bool
    naive_ponr_lead_time_s: float | None
    alarm_delay_from_disturbance_s: float | None

    def to_dict(self) -> dict:
        """Return JSON-compatible scalar fields."""
        return asdict(self)


def point_of_no_return_interval(
    report: FailureDebugReport, probe_stride: int
) -> PointOfNoReturnInterval | None:
    """Return the evidence interval containing PoNR for a strided probe."""
    if probe_stride < 1:
        raise ValueError(f"probe_stride must be >= 1, got {probe_stride}")
    if report.t_ponr is None:
        return None
    latest = int(report.t_ponr)
    earliest = max(0, latest - probe_stride + 1)
    return PointOfNoReturnInterval(earliest, latest, probe_stride)


def evaluate_actionability(
    report: FailureDebugReport,
    *,
    disturbance_onset: int,
    probe_stride: int = 1,
) -> ActionabilityReport:
    """Classify whether an alarm is causally actionable.

    ``disturbance_onset`` is the first timestep at which a known perturbation
    was applied. A warning is actionable only when it is not before that onset
    and is definitely before the earliest possible PoNR. An alarm inside the
    PoNR uncertainty interval is explicitly marked ambiguous.
    """
    if disturbance_onset < 0 or disturbance_onset >= report.T:
        raise ValueError(f"disturbance_onset must be in [0, {report.T}), got {disturbance_onset}")
    interval = point_of_no_return_interval(report, probe_stride)
    t_alarm = report.t_alarm

    if interval is None:
        failure_bound = report.t_failure if report.t_failure is not None else report.T
        window_status = "unlabeled_disturbance" if disturbance_onset < failure_bound else "no_irrecoverability"
        if t_alarm is None:
            relation = "no_alarm"
        elif t_alarm < disturbance_onset:
            relation = "pre_disturbance"
        else:
            relation = "no_ponr"
        valid = False
        earliest = latest = None
    elif disturbance_onset >= interval.latest:
        window_status = "empty"
        earliest, latest = interval.earliest, interval.latest
        if t_alarm is None:
            relation = "no_alarm"
        elif t_alarm < disturbance_onset:
            relation = "pre_disturbance"
        else:
            relation = "too_late"
        valid = False
    else:
        window_status = "available"
        earliest, latest = interval.earliest, interval.latest
        if t_alarm is None:
            relation = "no_alarm"
        elif t_alarm < disturbance_onset:
            relation = "pre_disturbance"
        elif t_alarm < earliest:
            relation = "definitely_actionable"
        elif t_alarm <= latest:
            relation = "ambiguous_within_ponr_interval"
        else:
            relation = "too_late"
        valid = relation == "definitely_actionable"

    delay = None if t_alarm is None else (t_alarm - disturbance_onset) * report.dt
    return ActionabilityReport(
        disturbance_onset=disturbance_onset,
        ponr_earliest=earliest,
        ponr_latest=latest,
        t_alarm=t_alarm,
        t_failure=report.t_failure,
        window_status=window_status,
        alarm_relation=relation,
        valid_actionable_warning=valid,
        naive_ponr_lead_time_s=report.ponr_lead_time_s,
        alarm_delay_from_disturbance_s=delay,
    )
