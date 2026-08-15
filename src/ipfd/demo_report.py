"""Single-page visual report for the counterfactual fidelity demo.

The renderer intentionally consumes plain mappings and sequences so a report can
be recreated directly from the JSON summary emitted by the demo. It derives the
trajectory error from the supplied positions rather than trusting a second copy
of the same evidence.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

__all__ = ["render_demo_report"]


_REFERENCE = "#0072B2"
_RESTORED = "#D55E00"
_ERROR = "#6F4E7C"
_TOLERANCE = "#009E73"
_GRID = "#CBD5E1"
_TEXT = "#17212B"
_MUTED = "#52616B"
_STATUS_STYLE = {
    "PASS": ("#DDF2E8", "#075E45"),
    "WARN": ("#FFF1C2", "#765500"),
    "DEGRADED": ("#FFE5B5", "#754A00"),
    "FAIL": ("#FADBD8", "#922B21"),
    "UNAVAILABLE": ("#E8EDF2", "#485563"),
    "N/A": ("#F1F4F6", "#52616B"),
}


def render_demo_report(summary: Mapping[str, Any], output: Path) -> Path:
    """Render a deterministic, headless PNG from a serializable demo summary.

    Required schema::

        {
          "system": "MuJoCo filtered-contact example",
          "focus_protocol": "minimal_visible",
          "trajectory": {
            "steps": [0, 1, ...],
            "reference_position": [0.0, ...],
            "restored_position": [0.0, ...],
            "tolerance": 1e-4,
            "reference_contact_steps": [4, ...],
            "restored_contact_steps": [4, ...],
            "reference_decision": "stable",
            "restored_decision": "unstable"
          },
          "protocols": [{
            "name": "minimal_visible",
            "omitted_capabilities": ["solver warm-start state"],
            "fidelity": {
              "l0_restore": "PASS",
              "l1_one_step": "PASS",
              "l2_by_horizon": {"1": "PASS", "30": "FAIL"},
              "l3_decision": "FAIL"
            }
          }]
        }

    ``title`` and ``trajectory.position_label`` are optional. Fidelity statuses
    are PASS, WARN, DEGRADED, FAIL, UNAVAILABLE, or N/A. Horizon keys may be
    integers or integer strings.
    """
    data = _validate_summary(summary)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelcolor": _TEXT,
        "axes.edgecolor": _GRID,
        "axes.titlecolor": _TEXT,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "text.color": _TEXT,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(12, 10), dpi=144)
        grid = fig.add_gridspec(3, 1, height_ratios=(2.5, 2.1, 2.9), hspace=0.52)
        trajectory_ax = fig.add_subplot(grid[0])
        error_ax = fig.add_subplot(grid[1], sharex=trajectory_ax)
        fidelity_ax = fig.add_subplot(grid[2])

        _draw_trajectory(trajectory_ax, data)
        _draw_error(error_ax, data)
        _draw_fidelity_grid(fidelity_ax, data)

        fig.suptitle(data["title"], fontsize=17, fontweight="bold", x=0.07, ha="left", y=0.975)
        fig.text(
            0.07,
            0.945,
            f'{data["system"]}  |  Focus protocol: {data["focus_protocol"]}',
            color=_MUTED,
            fontsize=9.5,
            ha="left",
        )
        fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.075)
        fig.savefig(
            output,
            format="png",
            dpi=144,
            metadata={"Software": "IPFD", "Title": data["title"]},
        )
        plt.close(fig)

    return output


def _draw_trajectory(ax: Any, data: dict[str, Any]) -> None:
    trajectory = data["trajectory"]
    steps = trajectory["steps"]
    reference = trajectory["reference_position"]
    restored = trajectory["restored_position"]
    branch_step = steps[0]

    ax.plot(steps, reference, color=_REFERENCE, lw=2.4, label=f'Reference: {trajectory["reference_decision"]}')
    ax.plot(
        steps,
        restored,
        color=_RESTORED,
        lw=2.1,
        ls=(0, (5, 2)),
        label=f'Restored: {trajectory["restored_decision"]}',
    )
    ax.axvline(branch_step, color=_TEXT, lw=1.2, ls=":")
    ax.annotate(
        "restore / branch",
        xy=(branch_step, reference[0]),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=8,
        color=_TEXT,
        arrowprops={"arrowstyle": "-|>", "color": _TEXT, "lw": 0.8},
    )
    ax.set_title("A. Identical continuation after the restore boundary", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(trajectory["position_label"])
    ax.grid(axis="y", color=_GRID, alpha=0.55, lw=0.7)
    ax.legend(loc="best", frameon=False, fontsize=9)
    _show_left_boundary(ax, steps)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_error(ax: Any, data: dict[str, Any]) -> None:
    trajectory = data["trajectory"]
    steps = trajectory["steps"]
    error = np.abs(trajectory["reference_position"] - trajectory["restored_position"])
    tolerance = trajectory["tolerance"]

    ax.plot(steps, error, color=_ERROR, lw=2.2, label="Absolute trajectory error")
    ax.axhline(tolerance, color=_TOLERANCE, lw=1.5, ls="--", label=f"Declared tolerance: {tolerance:g}")
    ax.fill_between(steps, tolerance, error, where=error > tolerance, color=_RESTORED, alpha=0.16)
    exceeds = np.flatnonzero(error > tolerance)
    if exceeds.size:
        first = int(exceeds[0])
        ax.axvline(steps[first], color=_RESTORED, lw=1.1, ls=":")
        ax.annotate(
            f"first exceeds tolerance: step {steps[first]:g}",
            xy=(steps[first], error[first]),
            xytext=(8, 18),
            textcoords="offset points",
            fontsize=8,
            color=_RESTORED,
            arrowprops={"arrowstyle": "->", "color": _RESTORED, "lw": 0.8},
        )

    contact_handles: list[Line2D] = []
    if trajectory["reference_contact_steps"].size:
        ax.scatter(
            trajectory["reference_contact_steps"],
            np.full(trajectory["reference_contact_steps"].shape, 0.04),
            marker="^",
            s=34,
            color=_REFERENCE,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=4,
        )
        contact_handles.append(
            Line2D([], [], color=_REFERENCE, marker="^", ls="None", label="Reference contact transition")
        )
    if trajectory["restored_contact_steps"].size:
        ax.scatter(
            trajectory["restored_contact_steps"],
            np.full(trajectory["restored_contact_steps"].shape, 0.13),
            marker="v",
            s=34,
            color=_RESTORED,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=4,
        )
        contact_handles.append(
            Line2D([], [], color=_RESTORED, marker="v", ls="None", label="Restored contact transition")
        )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + contact_handles, labels + [item.get_label() for item in contact_handles], loc="best", frameon=False)
    ax.set_title("B. Divergence onset and semantic contact evidence", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Steps after restore")
    ax.set_ylabel("Absolute error")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=_GRID, alpha=0.55, lw=0.7)
    _show_left_boundary(ax, steps)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_fidelity_grid(ax: Any, data: dict[str, Any]) -> None:
    protocols = data["protocols"]
    horizons = sorted({horizon for protocol in protocols for horizon in protocol["l2_by_horizon"]})
    columns = [("L0", "restore"), ("L1", "one step")]
    columns.extend(("L2", f"h={horizon}") for horizon in horizons)
    columns.append(("L3", "decision"))

    row_count = len(protocols)
    column_count = len(columns)
    label_width = 2.9
    cell_width = 1.25
    header_y = row_count + 0.55

    ax.set_xlim(0, label_width + column_count * cell_width)
    ax.set_ylim(-1.45, row_count + 1.15)
    ax.axis("off")
    ax.set_title("C. Fidelity claim frontier by snapshot protocol", loc="left", fontsize=11, fontweight="bold", pad=8)

    for column_index, (level, detail) in enumerate(columns):
        x = label_width + (column_index + 0.5) * cell_width
        ax.text(x, header_y, f"{level}\n{detail}", ha="center", va="center", fontsize=8, fontweight="bold")

    for row_index, protocol in enumerate(protocols):
        y = row_count - row_index - 0.5
        ax.text(0.02, y, protocol["name"], ha="left", va="center", fontsize=8.7, fontweight="bold")
        statuses = [protocol["l0_restore"], protocol["l1_one_step"]]
        statuses.extend(protocol["l2_by_horizon"].get(horizon, "N/A") for horizon in horizons)
        statuses.append(protocol["l3_decision"])
        for column_index, status in enumerate(statuses):
            x = label_width + column_index * cell_width
            face, foreground = _STATUS_STYLE[status]
            ax.add_patch(Rectangle((x, y - 0.34), cell_width - 0.08, 0.68, facecolor=face, edgecolor="white"))
            ax.text(
                x + (cell_width - 0.08) / 2,
                y,
                status,
                color=foreground,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
            )

    disclosures = []
    for protocol in protocols:
        omissions = protocol["omitted_capabilities"]
        description = ", ".join(omissions) if omissions else "none disclosed"
        disclosures.append(f'{protocol["name"]} omits: {description}')
    disclosure = "Capability disclosure: " + "; ".join(disclosures) + "."
    wrapped = "\n".join(textwrap.wrap(disclosure, width=142, subsequent_indent="    "))
    ax.text(0.02, -0.55, wrapped, ha="left", va="top", color=_MUTED, fontsize=8)


def _show_left_boundary(ax: Any, steps: np.ndarray) -> None:
    if steps.size == 1:
        ax.set_xlim(float(steps[0]) - 0.5, float(steps[0]) + 0.5)
        return
    span = float(steps[-1] - steps[0])
    ax.set_xlim(float(steps[0]) - 0.025 * span, float(steps[-1]) + 0.025 * span)


def _validate_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    system = _text(summary, "system")
    focus_protocol = _text(summary, "focus_protocol")
    title = summary.get("title", "Can this restored branch support the same decision?")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

    raw_trajectory = _mapping(summary, "trajectory")
    steps = _numeric_vector(raw_trajectory, "steps")
    reference = _numeric_vector(raw_trajectory, "reference_position")
    restored = _numeric_vector(raw_trajectory, "restored_position")
    if not (steps.size == reference.size == restored.size):
        raise ValueError("trajectory steps and position arrays must have equal lengths")
    if steps.size > 1 and np.any(np.diff(steps) <= 0):
        raise ValueError("trajectory.steps must be strictly increasing")
    tolerance = raw_trajectory.get("tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance):
        raise ValueError("trajectory.tolerance must be a finite number")
    if tolerance < 0:
        raise ValueError("trajectory.tolerance must be non-negative")

    reference_contacts = _numeric_vector(raw_trajectory, "reference_contact_steps", allow_empty=True)
    restored_contacts = _numeric_vector(raw_trajectory, "restored_contact_steps", allow_empty=True)
    reference_decision = _text(raw_trajectory, "reference_decision", prefix="trajectory.")
    restored_decision = _text(raw_trajectory, "restored_decision", prefix="trajectory.")
    position_label = raw_trajectory.get("position_label", "Position")
    if not isinstance(position_label, str) or not position_label.strip():
        raise ValueError("trajectory.position_label must be a non-empty string")

    raw_protocols = summary.get("protocols")
    if not isinstance(raw_protocols, Sequence) or isinstance(raw_protocols, (str, bytes)) or not raw_protocols:
        raise ValueError("protocols must be a non-empty sequence")
    protocols = [_validate_protocol(protocol, index) for index, protocol in enumerate(raw_protocols)]
    names = [protocol["name"] for protocol in protocols]
    if len(names) != len(set(names)):
        raise ValueError("protocol names must be unique")
    if focus_protocol not in names:
        raise ValueError("focus_protocol must name one of the supplied protocols")

    return {
        "title": title.strip(),
        "system": system,
        "focus_protocol": focus_protocol,
        "trajectory": {
            "steps": steps,
            "reference_position": reference,
            "restored_position": restored,
            "tolerance": float(tolerance),
            "reference_contact_steps": reference_contacts,
            "restored_contact_steps": restored_contacts,
            "reference_decision": reference_decision,
            "restored_decision": restored_decision,
            "position_label": position_label.strip(),
        },
        "protocols": protocols,
    }


def _validate_protocol(value: Any, index: int) -> dict[str, Any]:
    prefix = f"protocols[{index}]."
    if not isinstance(value, Mapping):
        raise ValueError(f"protocols[{index}] must be a mapping")
    name = _text(value, "name", prefix=prefix)
    omissions = value.get("omitted_capabilities")
    if not isinstance(omissions, Sequence) or isinstance(omissions, (str, bytes)):
        raise ValueError(f"{prefix}omitted_capabilities must be a sequence of strings")
    normalized_omissions: list[str] = []
    for omission in omissions:
        if not isinstance(omission, str) or not omission.strip():
            raise ValueError(f"{prefix}omitted_capabilities must contain non-empty strings")
        normalized_omissions.append(omission.strip())

    fidelity = _mapping(value, "fidelity", prefix=prefix)
    raw_horizons = fidelity.get("l2_by_horizon")
    if not isinstance(raw_horizons, Mapping) or not raw_horizons:
        raise ValueError(f"{prefix}fidelity.l2_by_horizon must be a non-empty mapping")
    horizons: dict[int, str] = {}
    for horizon, status in raw_horizons.items():
        try:
            normalized_horizon = int(horizon)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix}fidelity.l2_by_horizon keys must be positive integers") from exc
        if normalized_horizon <= 0 or str(normalized_horizon) != str(horizon):
            raise ValueError(f"{prefix}fidelity.l2_by_horizon keys must be positive integers")
        horizons[normalized_horizon] = _status(status, f"{prefix}fidelity.l2_by_horizon[{horizon}]")

    return {
        "name": name,
        "omitted_capabilities": normalized_omissions,
        "l0_restore": _status(fidelity.get("l0_restore"), f"{prefix}fidelity.l0_restore"),
        "l1_one_step": _status(fidelity.get("l1_one_step"), f"{prefix}fidelity.l1_one_step"),
        "l2_by_horizon": horizons,
        "l3_decision": _status(fidelity.get("l3_decision"), f"{prefix}fidelity.l3_decision"),
    }


def _mapping(source: Mapping[str, Any], key: str, *, prefix: str = "") -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{prefix}{key} must be a mapping")
    return value


def _text(source: Mapping[str, Any], key: str, *, prefix: str = "") -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}{key} must be a non-empty string")
    return value.strip()


def _numeric_vector(source: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> np.ndarray:
    value = source.get(key)
    if isinstance(value, (str, bytes)):
        raise ValueError(f"trajectory.{key} must be a sequence of finite numbers")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trajectory.{key} must be a sequence of finite numbers") from exc
    if result.ndim != 1 or (not allow_empty and result.size == 0) or not np.all(np.isfinite(result)):
        qualifier = "a one-dimensional sequence of finite numbers"
        if not allow_empty:
            qualifier = "a non-empty " + qualifier
        raise ValueError(f"trajectory.{key} must be {qualifier}")
    return result


def _status(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a fidelity status")
    normalized = value.strip().upper()
    if normalized not in _STATUS_STYLE:
        allowed = ", ".join(_STATUS_STYLE)
        raise ValueError(f"{path} must be one of: {allowed}")
    return normalized
