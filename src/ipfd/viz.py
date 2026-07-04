"""Rollout timeline visualization.

A single stacked-panel figure per rollout: key observation feature, action norm,
policy entropy, representation drift, and the combined imminence score. Three
vertical markers tell the whole story at a glance -- point of no return (when it
became doomed), detector alarm (when the tool noticed), and observable failure
(when it finally looked broken). Matplotlib only; ``Agg`` backend for headless use.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .report import FailureDebugReport  # noqa: E402
from .types import Rollout  # noqa: E402

__all__ = ["plot_timeline"]


def plot_timeline(rollout: Rollout, report: FailureDebugReport, path: str) -> str:
    """Render the failure-debug timeline for ``rollout`` to ``path`` (PNG)."""
    signals = getattr(report, "signals", {})
    t = rollout.time_s()

    panels: list[tuple[str, np.ndarray]] = [
        ("obs[1] (object-in-workspace)", rollout.observations[:, 1]),
        ("action L2 norm", np.linalg.norm(rollout.actions, axis=1)),
    ]
    if rollout.entropy is not None:
        panels.append(("policy entropy (confidence)", rollout.entropy))
    if signals.get("drift") is not None and signals["drift"].size == rollout.T:
        panels.append(("representation drift", signals["drift"]))
    panels.append(("failure imminence score", signals.get("imminence", np.zeros(rollout.T))))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, y) in zip(axes, panels, strict=False):
        ax.plot(t, y, lw=1.4, color="#1f77b4")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.25)
        _mark(ax, report.t_ponr, rollout.dt, "#d62728", "PoNR")
        _mark(ax, report.t_alarm, rollout.dt, "#ff7f0e", "alarm")
        _mark(ax, report.t_failure, rollout.dt, "#7f7f7f", "failure")

    axes[-1].axhline(
        signals.get("config").alarm_threshold if signals.get("config") else 0.5,
        ls=":", color="k", lw=1, alpha=0.6,
    )
    axes[-1].set_xlabel("time (s)")
    handles = [
        Line2D([0], [0], color=c, lw=1.6, label=lbl)
        for t_idx, c, lbl in [
            (report.t_ponr, "#d62728", "PoNR"),
            (report.t_alarm, "#ff7f0e", "alarm"),
            (report.t_failure, "#7f7f7f", "failure"),
        ]
        if t_idx is not None
    ]
    if handles:
        axes[0].legend(handles=handles, loc="upper left", fontsize=7, ncol=3)
    title = f"IPFD timeline  |  {'SUCCESS' if report.success else 'SILENT FAILURE'}  |  seed={report.seed}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _mark(ax, t_idx: int | None, dt: float, color: str, label: str) -> None:
    if t_idx is None:
        return
    ax.axvline(t_idx * dt, color=color, lw=1.6, alpha=0.85, label=label)
