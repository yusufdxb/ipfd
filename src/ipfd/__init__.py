"""IPFD -- Isaac Policy Failure Debugger.

Detect and visualize the exact moment a policy becomes irrecoverably doomed,
before failure is externally observable, from an Isaac Lab rollout.

The public surface is intentionally small:

    from ipfd import Rollout, build_report, plot_timeline
    from ipfd.adapters.synthetic import make_silent_failure_rollout

    rollout = make_silent_failure_rollout(seed=0)
    report = build_report(rollout)
    print(report.summary())
    plot_timeline(rollout, report, "timeline.png")
"""

from __future__ import annotations

from .ponr import point_of_no_return
from .report import AnalysisConfig, FailureDebugReport, build_report
from .types import Rollout
from .viz import plot_timeline

__version__ = "1.0.0"

__all__ = [
    "Rollout",
    "AnalysisConfig",
    "FailureDebugReport",
    "build_report",
    "plot_timeline",
    "point_of_no_return",
    "__version__",
]
