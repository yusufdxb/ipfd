"""IPFD: Isaac Policy Failure Debugger.

Localize the oracle-relative transition after which a tested recovery controller
no longer succeeds, and compare it with detector and visible-failure timing.

The public surface is intentionally small:

    from ipfd import Rollout, build_report, plot_timeline
    from ipfd.adapters.synthetic import make_silent_failure_rollout

    rollout = make_silent_failure_rollout(seed=0)
    report = build_report(rollout)
    print(report.summary())
    plot_timeline(rollout, report, "timeline.png")
"""

from __future__ import annotations

from .actionability import ActionabilityReport, PointOfNoReturnInterval, evaluate_actionability
from .evidence_gate import EvidenceCriteria, EvidenceGateResult, evaluate_evidence
from .ponr import (
    ProbeStatistics,
    aggregate_repeated_probes,
    point_of_no_return,
    point_of_no_return_repeated,
)
from .replay import load_rollout, save_rollout
from .report import AnalysisConfig, FailureDebugReport, build_report
from .types import Rollout
from .viz import plot_timeline

__version__ = "1.1.0.dev0"

__all__ = [
    "Rollout",
    "AnalysisConfig",
    "FailureDebugReport",
    "build_report",
    "plot_timeline",
    "point_of_no_return",
    "point_of_no_return_repeated",
    "aggregate_repeated_probes",
    "ProbeStatistics",
    "ActionabilityReport",
    "PointOfNoReturnInterval",
    "evaluate_actionability",
    "EvidenceCriteria",
    "EvidenceGateResult",
    "evaluate_evidence",
    "save_rollout",
    "load_rollout",
    "__version__",
]
