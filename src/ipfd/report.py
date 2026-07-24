"""Assemble a Failure Debug Report from a rollout.

This is the top-level analysis entry point: given a :class:`Rollout`, it runs the
detectors, locates the point of no return, computes the metric set, and packages
everything into a serializable :class:`FailureDebugReport` plus a human-readable
summary. Pure NumPy -- no simulator, no GPU.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from . import detectors, metrics
from .ponr import point_of_no_return
from .types import Rollout

__all__ = ["AnalysisConfig", "FailureDebugReport", "build_report"]


@dataclass
class AnalysisConfig:
    """Tunable knobs for the analysis pass. Defaults are sane for ~60 Hz control."""

    baseline_window: int = 20
    drift_ref_window: int = 10
    drift_metric: str = "cosine"
    alarm_threshold: float = 0.5
    alarm_persistence: int = 3
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class FailureDebugReport:
    """Everything an engineer needs to see why an episode silently failed.

    Signal arrays are kept out of the dataclass fields for JSON compactness and
    attached separately on the instance (``self.signals``); :meth:`to_dict` emits
    only the scalars by default.
    """

    success: bool
    T: int
    dt: float
    seed: int | None
    t_ponr: int | None
    t_failure: int | None
    t_alarm: int | None
    time_to_failure_s: float | None
    failure_lead_time_s: float | None
    ponr_lead_time_s: float | None
    silent_doom_window_s: float | None
    false_continuity_rate: float | None
    drift_at_collapse: float | None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        try:
            s = json.dumps(self.to_dict(), indent=indent, default=_json_default, allow_nan=False)
        except ValueError as exc:
            raise ValueError(f"report JSON must contain only finite numeric values: {exc}") from exc
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    def summary(self) -> str:
        """A terse, plain-English readout for the terminal."""
        lines = ["=== IPFD Failure Debug Report ==="]
        lines.append(f"outcome            : {'SUCCESS' if self.success else 'FAILURE'}")
        lines.append(f"steps              : {self.T}  (dt={self.dt:.4f}s)")
        lines.append(f"point of no return : {_fmt_step(self.t_ponr, self.dt)}")
        lines.append(f"observable failure : {_fmt_step(self.t_failure, self.dt)}")
        lines.append(f"detector alarm     : {_fmt_step(self.t_alarm, self.dt)}")
        lines.append(f"time to failure    : {_fmt_s(self.time_to_failure_s)}")
        lines.append(f"failure lead time  : {_fmt_s(self.failure_lead_time_s)}  (alarm before visible failure)")
        lines.append(f"PoNR lead time     : {_fmt_s(self.ponr_lead_time_s)}  (alarm vs irrecoverable; +ve = actionable)")
        lines.append(f"silent-doom window : {_fmt_s(self.silent_doom_window_s)}  (doomed but looked fine)")
        if self.false_continuity_rate is not None:
            lines.append(f"false continuity   : {self.false_continuity_rate:.0%}  (of doomed window, detector stayed quiet)")
        lines.append(f"drift @ collapse   : {_fmt_num(self.drift_at_collapse)}")
        lines.append(_verdict(self))
        return "\n".join(lines)


def _fmt_step(t: int | None, dt: float) -> str:
    return "n/a" if t is None else f"step {t} ({t * dt:.2f}s)"


def _fmt_s(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}s"


def _fmt_num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _json_default(o: object) -> object:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _verdict(r: FailureDebugReport) -> str:
    if r.success:
        return "verdict            : nominal."
    if r.ponr_lead_time_s is not None and r.ponr_lead_time_s < 0:
        return "verdict            : SILENT COLLAPSE -- alarm fired only AFTER the trajectory was doomed."
    if r.false_continuity_rate is not None and r.false_continuity_rate >= 0.5:
        return "verdict            : SILENT COLLAPSE -- policy stayed confident through most of the doomed window."
    return "verdict            : failure caught with lead time."


def build_report(rollout: Rollout, config: AnalysisConfig | None = None) -> FailureDebugReport:
    """Run the full analysis pass and return a populated report.

    The per-step signal arrays used to build the report are attached to the
    returned object as ``report.signals`` for the visualizer to consume.
    """
    cfg = config or AnalysisConfig()

    imminence = detectors.failure_imminence_score(
        rollout, weights=cfg.weights, baseline_window=cfg.baseline_window
    )
    drift_raw = detectors.representation_drift(
        rollout.embeddings, ref_window=cfg.drift_ref_window, metric=cfg.drift_metric
    )
    t_alarm = detectors.first_alarm(imminence, cfg.alarm_threshold, cfg.alarm_persistence)
    t_ponr = point_of_no_return(rollout.recovery_success)

    report = FailureDebugReport(
        success=rollout.success,
        T=rollout.T,
        dt=rollout.dt,
        seed=rollout.seed,
        t_ponr=t_ponr,
        t_failure=rollout.t_failure,
        t_alarm=t_alarm,
        time_to_failure_s=metrics.time_to_failure(rollout.t_failure, rollout.dt),
        failure_lead_time_s=metrics.failure_lead_time(t_alarm, rollout.t_failure, rollout.dt),
        ponr_lead_time_s=metrics.ponr_lead_time(t_alarm, t_ponr, rollout.dt),
        silent_doom_window_s=metrics.silent_doom_window(t_ponr, rollout.t_failure, rollout.dt),
        false_continuity_rate=metrics.false_continuity_rate(
            imminence, t_ponr, rollout.t_failure, cfg.alarm_threshold
        ),
        drift_at_collapse=metrics.drift_magnitude_at_collapse(drift_raw, t_ponr),
        meta=dict(rollout.meta),
    )
    # Attach raw signals for visualization (not part of the serialized dict).
    report.signals = {  # type: ignore[attr-defined]
        "imminence": imminence,
        "drift": drift_raw,
        "action_variance": detectors.action_variance_score(
            rollout.actions, baseline_window=cfg.baseline_window
        ),
        "entropy_collapse": detectors.entropy_collapse_score(
            rollout.entropy, baseline_window=cfg.baseline_window
        ),
        "config": cfg,
    }
    return report
