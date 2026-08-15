"""IPFD: counterfactual fidelity auditing for robot simulation.

Audit which numerical trajectories and downstream decisions a declared simulator
snapshot-and-restore protocol supports. The v1 rollout-analysis imports remain
available for historical artifact compatibility, but they are not the v2 mission.

The primary public surface is intentionally small:

    from ipfd import DecisionContract, audit, check_adapter

    result = audit(
        adapter=my_adapter,
        protocol="expanded_runtime_state",
        branch_step=120,
        horizons=[1, 5, 10, 30, 90],
        decision="task_success",
    )
    print(result.verdict)

Historical rollout-analysis imports remain available for artifact compatibility.
"""

from __future__ import annotations

__version__ = "2.0.0.dev0"

from .actionability import ActionabilityReport, PointOfNoReturnInterval, evaluate_actionability
from .adapter_check import (
    AdapterCheck,
    AdapterCheckReport,
    AdapterCheckStatus,
    check_adapter,
)
from .audit import AuditResult, DecisionContract, audit
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


def plot_timeline(rollout: Rollout, report: FailureDebugReport, path: str) -> str:
    """Lazily render a legacy rollout timeline without loading Matplotlib on import."""

    from .viz import plot_timeline as render_timeline

    return render_timeline(rollout, report, path)

__all__ = [
    "Rollout",
    "audit",
    "AuditResult",
    "DecisionContract",
    "check_adapter",
    "AdapterCheck",
    "AdapterCheckReport",
    "AdapterCheckStatus",
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
