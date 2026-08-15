"""Machine-readable and concise human-readable audit outputs."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import AuditConfig
from .contracts import ContractVerdict, to_builtin
from .provenance import sha256_file
from .regression import compare_audit_files

__all__ = ["contract_document", "write_audit_outputs"]


def contract_document() -> dict[str, Any]:
    """Return the stable machine-readable definition of L0 through L3."""

    return {
        "schema_version": 1,
        "name": "IPFD Replay Fidelity Contract",
        "central_principle": "Equality at restoration time does not establish downstream counterfactual validity.",
        "verdicts": [
            ContractVerdict.SUPPORTED.value,
            ContractVerdict.UNSUPPORTED.value,
            ContractVerdict.INSUFFICIENT_EVIDENCE.value,
        ],
        "levels": {
            "L0": {
                "name": "restore equality",
                "compares": [
                    "exposed scene state",
                    "policy observations",
                    "privileged observations",
                    "task-manager state",
                    "controller targets",
                    "sensor state",
                    "simulation counters",
                ],
                "permitted_meaning": "Only the measured exposed state agrees at the restoration boundary.",
                "forbidden_implication": "L0 does not imply trajectory or decision fidelity.",
            },
            "L1": {
                "name": "one-step dynamics fidelity",
                "compares": [
                    "next state",
                    "next observation",
                    "contact state",
                    "task-manager outputs",
                    "termination state",
                    "reward where relevant",
                ],
                "separation": "Numerical differences and semantic differences are reported separately.",
            },
            "L2": {
                "name": "finite-horizon open-loop trajectory fidelity",
                "required_horizons": [1, 5, 10, 30, 90],
                "requires": "Identical recorded actions for uninterrupted and restored instances.",
                "measurements": [
                    "first numerical divergence",
                    "first observation divergence",
                    "first contact divergence",
                    "maximum state error",
                    "terminal state error",
                    "divergence-growth curve",
                ],
            },
            "L3": {
                "name": "downstream decision fidelity",
                "primary": True,
                "requires": "A user-declared decision function.",
                "measurement": "Whether uninterrupted and restored branches produce the same declared conclusion.",
            },
        },
        "tolerance_policy": {
            "universal_tolerance": False,
            "raw_measurements_preserved": True,
            "environment_specific_tolerances_required": True,
            "per_field_overrides_supported": True,
            "units_are_part_of_the_declared_contract": True,
        },
        "scope_fields": [
            "simulator",
            "simulator_version",
            "environment",
            "task",
            "snapshot_protocol",
            "continuation_mode",
            "horizon",
            "action_source",
            "decision_function",
            "tolerances",
            "hardware_and_software_provenance",
        ],
        "universal_simulator_verdicts_permitted": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(to_builtin(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _overall(configurations: list[dict[str, Any]]) -> str:
    results = {item["result"] for item in configurations}
    if ContractVerdict.UNSUPPORTED.value in results:
        return ContractVerdict.UNSUPPORTED.value
    if ContractVerdict.INSUFFICIENT_EVIDENCE.value in results:
        return ContractVerdict.INSUFFICIENT_EVIDENCE.value
    return ContractVerdict.SUPPORTED.value


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    configurations = list(result["configurations"])
    return {
        "schema_version": 1,
        "overall_result": _overall(configurations),
        "interpretation": "Scoped empirical support only. This is not a universal simulator verdict.",
        "configuration_count": len(configurations),
        "branch_record_count": len(result["records"]),
        "failure_reproducer_produced": result.get("minimal_reproducer") is not None,
        "configurations": configurations,
    }


def _markdown(summary: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        "# IPFD replay-fidelity audit",
        "",
        f"Contract result: **{summary['overall_result']}**",
        "",
        (
            "This verdict applies only to the listed simulator, version, environment, task, protocol, "
            "continuation, horizon, action source, decision function, tolerance, and provenance."
        ),
        "",
        "L0 equality never implies L2 trajectory fidelity or L3 decision fidelity.",
        "",
        "| Simulator | Protocol | Horizon | Decision | L0 | L1 | L2 first divergence | L3 disagreement | Result |",
        "|---|---|---:|---|---|---|---:|---|---|",
    ]
    for item in summary["configurations"]:
        scope = item["scope"]
        levels = item["levels"]
        divergence = levels["L2"].get(
            "first_divergence", levels["L2"]["first_numerical_divergence"]
        )
        lines.append(
            "| {simulator} {version} | {protocol} | {horizon} | {decision} | {l0} | {l1} | {l2} | {l3} | {result} |".format(
                simulator=scope["simulator"],
                version=scope["simulator_version"],
                protocol=scope["snapshot_protocol"],
                horizon=scope["horizon"],
                decision=scope["decision_function"],
                l0=levels["L0"]["passed"],
                l1=levels["L1"]["passed"],
                l2=divergence if divergence is not None else "none observed",
                l3=levels["L3"]["decision_disagreement"],
                result=item["result"],
            )
        )
    lines.extend(
        [
            "",
            "## Evidence files",
            "",
            "- `audit_summary.json`: scoped contract conclusions",
            "- `per_branch_records.jsonl`: raw paired branch measurements",
            "- `fidelity_contract.json`: the contract evaluated",
            "- `provenance.json`: source, software, hardware class, and adapter inventory",
            "- `divergence.svg`: numerical error growth",
        ]
    )
    if result.get("minimal_reproducer") is not None:
        lines.append("- `minimal_reproducer.json`: automatically reduced failing case")
    lines.append("")
    return "\n".join(lines)


def _divergence_svg(records: list[dict[str, Any]]) -> str:
    by_step: dict[int, float] = defaultdict(float)
    for record in records:
        curve = record.get("levels", {}).get("L2", {}).get("divergence_growth_curve")
        if not isinstance(curve, list):
            continue
        for point in curve:
            step = int(point["step"])
            by_step[step] = max(by_step[step], float(point["state_max_abs"]))
    width, height = 800, 360
    left, right, top, bottom = 72, 24, 32, 56
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_step = max(by_step, default=1)
    max_error = max(by_step.values(), default=0.0)
    scale_error = max(max_error, 1e-15)
    points = []
    for step, error in sorted(by_step.items()):
        x = left + plot_width * step / max_step
        y = top + plot_height * (1.0 - error / scale_error)
        points.append(f"{x:.2f},{y:.2f}")
    polyline = " ".join(points)
    note = (
        "No per-step curve retained"
        if not points
        else f"largest raw field error {max_error:.6g}; mixed units"
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
  viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Replay divergence growth</title>
  <desc id="desc">Largest raw field error across audited branches by continuation step.
    Values may mix units and are diagnostic only.</desc>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#334155"/>
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#334155"/>
  <polyline points="{html.escape(polyline)}" fill="none" stroke="#dc2626" stroke-width="2.5"/>
  <text x="{width / 2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="16">Replay divergence growth</text>
  <text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-family="sans-serif" font-size="13">Continuation step</text>
  <text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})"
    text-anchor="middle" font-family="sans-serif" font-size="13">Largest raw field error (mixed units)</text>
  <text x="{left + 8}" y="{top + 18}" font-family="sans-serif" font-size="12" fill="#475569">{html.escape(note)}</text>
</svg>
"""


def write_audit_outputs(result: dict[str, Any], config: AuditConfig) -> dict[str, Any]:
    """Write the required artifact set and an optional regression report."""

    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    summary = _summary(result)
    _write_json(output / "audit_summary.json", summary)
    with (output / "per_branch_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in result["records"]:
            stream.write(json.dumps(to_builtin(record), sort_keys=True) + "\n")
    _write_json(output / "fidelity_contract.json", contract_document())
    _write_json(output / "provenance.json", result["provenance"])
    (output / "REPORT.md").write_text(_markdown(summary, result), encoding="utf-8")
    (output / "divergence.svg").write_text(_divergence_svg(result["records"]), encoding="utf-8")
    if result.get("minimal_reproducer") is not None:
        _write_json(output / "minimal_reproducer.json", result["minimal_reproducer"])
    elif (output / "minimal_reproducer.json").is_file():
        (output / "minimal_reproducer.json").unlink()

    regression_report = None
    if config.regression is not None:
        baseline_value = config.regression.get("baseline_summary")
        if not isinstance(baseline_value, str) or not baseline_value:
            raise ValueError("regression.baseline_summary must be a non-empty path")
        baseline = Path(baseline_value)
        if not baseline.is_absolute():
            baseline = (config.source_path.parent / baseline).resolve()
        regression_report = compare_audit_files(baseline, output / "audit_summary.json")
        _write_json(output / "regression_report.json", regression_report)
    elif (output / "regression_report.json").is_file():
        (output / "regression_report.json").unlink()
    artifact_names = [
        "per_branch_records.jsonl",
        "fidelity_contract.json",
        "provenance.json",
        "REPORT.md",
        "divergence.svg",
        "minimal_reproducer.json",
        "regression_report.json",
    ]
    summary["artifacts"] = {
        name: {
            "sha256": sha256_file(output / name),
            "bytes": (output / name).stat().st_size,
        }
        for name in artifact_names
        if (output / name).is_file()
    }
    _write_json(output / "audit_summary.json", summary)
    return {"summary": summary, "regression_report": regression_report}
