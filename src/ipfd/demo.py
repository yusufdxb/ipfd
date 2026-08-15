"""Reproducible MuJoCo counterfactual-fidelity demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .demo_report import render_demo_report
from .fidelity.audit import audit_configuration
from .fidelity.config import AuditConfig, BranchState
from .fidelity.contracts import TrajectoryRecord, to_builtin

__all__ = ["main", "make_demo_adapter", "run_demo"]

_DEMO_CONFIG_TEXT = resources.files("ipfd").joinpath("demo_config.yaml").read_text(
    encoding="utf-8"
)
_DEMO_SPEC = yaml.safe_load(_DEMO_CONFIG_TEXT)
if not isinstance(_DEMO_SPEC, dict) or _DEMO_SPEC.get("schema_version") != 1:
    raise RuntimeError("bundled demo_config.yaml is invalid")
_HORIZONS = tuple(int(value) for value in _DEMO_SPEC["horizons"])
_SEEDS = tuple(int(value) for value in _DEMO_SPEC["seed_groups"])
_BRANCH_STEP = int(_DEMO_SPEC["branch_step"])
_POSITION_TOLERANCE_M = float(_DEMO_SPEC["tolerances"]["position_m"])
_VELOCITY_TOLERANCE_MPS = float(_DEMO_SPEC["tolerances"]["velocity_mps"])
_SETTINGS: dict[str, Any] = {
    "kind": "mujoco",
    "simulator": "MuJoCo",
    "regime": str(_DEMO_SPEC["regime"]),
    "timestep": float(_DEMO_SPEC["timestep_seconds"]),
    "frame_skip": int(_DEMO_SPEC["frame_skip"]),
    "activation_preload_steps": int(_DEMO_SPEC["action_schedule"]["preload_steps"]),
    "post_preload_control": float(_DEMO_SPEC["action_schedule"]["continuation_control"]),
    "position_bound": float(_DEMO_SPEC["position_bound"]),
    "initial_position_jitter": float(_DEMO_SPEC["initial_position_jitter"]),
}


def make_demo_adapter() -> Any:
    """Return the bundled full-state adapter for ``ipfd adapter-check``."""

    from .adapters.mujoco_replay import MuJoCoReplayAdapter

    settings = dict(_SETTINGS)
    settings["minimal_capture_actuator_activation"] = True
    return MuJoCoReplayAdapter(
        settings,
        snapshot_protocol="integration_with_warmstart",
        continuation_mode="exact_action",
    )


def _tolerances() -> dict[str, dict[str, Any]]:
    position_fields = {
        "qpos": {"absolute": _POSITION_TOLERANCE_M, "unit": "m"},
        "mover_position": {"absolute": _POSITION_TOLERANCE_M, "unit": "m"},
        "qvel": {"absolute": _VELOCITY_TOLERANCE_MPS, "unit": "m/s"},
    }
    task_fields = {
        "forward_progress": {"absolute": _POSITION_TOLERANCE_M, "unit": "m"},
        "speed": {"absolute": _VELOCITY_TOLERANCE_MPS, "unit": "m/s"},
    }
    return {
        "default": {"absolute": 0.0, "relative": 0.0},
        "scene_state": {"absolute": 0.0, "relative": 0.0, "fields": position_fields},
        "policy_observations": {"absolute": _VELOCITY_TOLERANCE_MPS, "relative": 0.0},
        "privileged_observations": {"absolute": 0.0, "relative": 0.0},
        "task_state": {"absolute": 0.0, "relative": 0.0},
        "controller_targets": {"absolute": 0.0, "relative": 0.0},
        "sensor_state": {"absolute": _VELOCITY_TOLERANCE_MPS, "relative": 0.0},
        "counters": {"absolute": 0.0, "relative": 0.0},
        "contact_state": {"absolute": 0.001, "relative": 0.0},
        "task_outputs": {"absolute": 0.0, "relative": 0.0, "fields": task_fields},
        "termination": {"absolute": 0.0, "relative": 0.0},
        "reward": {"absolute": _VELOCITY_TOLERANCE_MPS, "relative": 0.0},
    }


def _config(protocol: str, branch_step: int) -> AuditConfig:
    return AuditConfig(
        source_path=Path("<bundled-ipfd-demo>"),
        adapter=dict(_SETTINGS),
        simulator_version="runtime",
        environment=str(_DEMO_SPEC["environment"]),
        task=str(_DEMO_SPEC["task"]),
        snapshot_protocol=protocol,
        branch_states=tuple(
            BranchState(
                id=f"demo-{protocol}-seed-{seed}-step-{branch_step}",
                step=branch_step,
                seed=seed,
                cluster=f"seed-{seed}",
            )
            for seed in _SEEDS
        ),
        horizons=_HORIZONS,
        continuation_mode="exact_action",
        action_source="deterministic",
        decision_functions=("remains_in_contact",),
        tolerances=_tolerances(),
        independent_cluster_key="seed",
        output_directory=Path("."),
        minimum_independent_clusters=len(_SEEDS),
        reduction={"enabled": False},
        regression=None,
        raw={},
    )


def _run_contract(
    protocol: str,
    branch_step: int,
    *,
    capture_activation: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .adapters.mujoco_replay import MuJoCoReplayAdapter

    settings = dict(_SETTINGS)
    settings["minimal_capture_actuator_activation"] = (
        protocol != "minimal_visible"
        if capture_activation is None
        else capture_activation
    )
    adapter = MuJoCoReplayAdapter(
        settings,
        snapshot_protocol=protocol,
        continuation_mode="exact_action",
    )
    try:
        result = audit_configuration(adapter, _config(protocol, branch_step))
        provenance = dict(adapter.provenance())
    finally:
        adapter.close()
    return result, provenance


def _focus_trajectory() -> dict[str, Any]:
    from .adapters.mujoco_replay import MuJoCoReplayAdapter

    settings = dict(_SETTINGS)
    settings["minimal_capture_actuator_activation"] = False
    adapter = MuJoCoReplayAdapter(
        settings,
        snapshot_protocol="minimal_visible",
        continuation_mode="exact_action",
    )
    try:
        adapter.reset(_SEEDS[0])
        for step in range(_BRANCH_STEP):
            adapter.step(adapter.action(step, "deterministic", (0, 1)))
        snapshot = adapter.capture((0,))
        adapter.restore(snapshot, (1,))
        boundary = adapter.observe((0, 1))
        qpos = boundary.scene_state["qpos"]
        qvel = boundary.scene_state["qvel"]
        reference_position = [float(qpos[0, 0])]
        restored_position = [float(qpos[1, 0])]
        reference_velocity = [float(qvel[0, 0])]
        restored_velocity = [float(qvel[1, 0])]
        reference_contact = [bool(boundary.task_state["contact_active"][0])]
        restored_contact = [bool(boundary.task_state["contact_active"][1])]
        records = []
        actions = []
        for continuation_step in range(1, max(_HORIZONS) + 1):
            action = adapter.action(
                _BRANCH_STEP + continuation_step - 1,
                "deterministic",
                (0, 1),
            )
            record = adapter.step(action)
            records.append(record)
            actions.append(to_builtin(action))
            state = record.observation.scene_state
            reference_position.append(float(state["qpos"][0, 0]))
            restored_position.append(float(state["qpos"][1, 0]))
            reference_velocity.append(float(state["qvel"][0, 0]))
            restored_velocity.append(float(state["qvel"][1, 0]))
            reference_contact.append(bool(record.contact_state["active"][0]))
            restored_contact.append(bool(record.contact_state["active"][1]))
        reference = TrajectoryRecord(steps=records, actions=actions, env_id=0)
        restored = TrajectoryRecord(steps=records, actions=actions, env_id=1)
        reference_decision = adapter.decision(reference, "remains_in_contact")
        restored_decision = adapter.decision(restored, "remains_in_contact")
    finally:
        adapter.close()

    def transition_steps(values: list[bool]) -> list[int]:
        return [index for index in range(1, len(values)) if values[index] != values[index - 1]]

    position_error = np.abs(np.asarray(reference_position) - np.asarray(restored_position))
    velocity_error = np.abs(np.asarray(reference_velocity) - np.asarray(restored_velocity))
    contact_disagreement = next(
        (
            index
            for index, (reference_value, restored_value) in enumerate(
                zip(reference_contact, restored_contact, strict=True)
            )
            if reference_value != restored_value
        ),
        None,
    )
    numerical_divergence = next(
        (
            index
            for index, (position_value, velocity_value) in enumerate(
                zip(position_error, velocity_error, strict=True)
            )
            if position_value > _POSITION_TOLERANCE_M
            or velocity_value > _VELOCITY_TOLERANCE_MPS
        ),
        None,
    )
    return {
        "steps": list(range(max(_HORIZONS) + 1)),
        "reference_position": reference_position,
        "restored_position": restored_position,
        "reference_velocity": reference_velocity,
        "restored_velocity": restored_velocity,
        "reference_contact": reference_contact,
        "restored_contact": restored_contact,
        "reference_contact_steps": transition_steps(reference_contact),
        "restored_contact_steps": transition_steps(restored_contact),
        "reference_decision": "remains in contact" if reference_decision else "lifts off",
        "restored_decision": "remains in contact" if restored_decision else "lifts off",
        "decision_agreement": reference_decision == restored_decision,
        "decision_direction": (
            "MATCH"
            if reference_decision == restored_decision
            else "REFERENCE_TRUE_RESTORED_FALSE"
            if reference_decision
            else "REFERENCE_FALSE_RESTORED_TRUE"
        ),
        "first_numerical_divergence_step": numerical_divergence,
        "first_contact_disagreement_step": contact_disagreement,
        "maximum_position_error_m": float(np.max(position_error)),
        "maximum_velocity_error_mps": float(np.max(velocity_error)),
        "captured_components": list(snapshot.captured_components),
        "unavailable_components": list(snapshot.unavailable_components),
        "boundary_unavailable": list(boundary.unavailable),
    }


def _configuration_by_horizon(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(item["scope"]["horizon"]): item
        for item in result["configurations"]
        if item["scope"]["decision_function"] == "remains_in_contact"
    }


def _protocol_row(name: str, result: dict[str, Any], omissions: list[str]) -> dict[str, Any]:
    by_horizon = _configuration_by_horizon(result)
    final = by_horizon[max(_HORIZONS)]
    statuses: dict[str, str] = {}
    for horizon in _HORIZONS:
        levels = by_horizon[horizon]["levels"]
        if bool(levels["L2"]["passed"]):
            status = "PASS"
        elif not bool(levels["L3"]["decision_disagreement"]):
            status = "DEGRADED"
        else:
            status = "FAIL"
        statuses[str(horizon)] = status
    return {
        "name": name,
        "omitted_capabilities": omissions,
        "fidelity": {
            "l0_restore": "PASS" if bool(final["levels"]["L0"]["passed"]) else "FAIL",
            "l1_one_step": "PASS" if bool(final["levels"]["L1"]["passed"]) else "FAIL",
            "l2_by_horizon": statuses,
            "l3_decision": (
                "FAIL" if bool(final["levels"]["L3"]["decision_disagreement"]) else "PASS"
            ),
        },
    }


def _decision_result(result: dict[str, Any]) -> dict[str, Any]:
    records = [
        record
        for record in result["records"]
        if int(record["horizon"]) == max(_HORIZONS)
    ]
    comparisons = [
        record["levels"]["L3"]["decisions"]["remains_in_contact"]
        for record in records
    ]
    reference_values = [bool(item["reference"]) for item in comparisons]
    restored_values = [bool(item["restored"]) for item in comparisons]

    def label(values: list[bool]) -> str:
        if all(values):
            return "remains in contact"
        if not any(values):
            return "lifts off"
        return f"mixed ({sum(values)}/{len(values)} remain in contact)"

    agreement = all(bool(item["agreement"]) for item in comparisons)
    directions = {
        "MATCH"
        if reference == restored
        else "REFERENCE_TRUE_RESTORED_FALSE"
        if reference
        else "REFERENCE_FALSE_RESTORED_TRUE"
        for reference, restored in zip(reference_values, restored_values, strict=True)
    }
    return {
        "reference": reference_values,
        "restored": restored_values,
        "reference_label": label(reference_values),
        "restored_label": label(restored_values),
        "agreement": agreement,
        "direction": next(iter(directions)) if len(directions) == 1 else "MIXED",
        "verdict": "MATCH" if agreement else "FAIL_CLOSED",
    }


def _sensitivity_results() -> list[dict[str, Any]]:
    from .adapters.mujoco_replay import MuJoCoReplayAdapter

    results: list[dict[str, Any]] = []
    for control in _DEMO_SPEC["sensitivity_controls"]:
        settings = dict(_SETTINGS)
        settings["post_preload_control"] = float(control)
        settings["minimal_capture_actuator_activation"] = False
        adapter = MuJoCoReplayAdapter(
            settings,
            snapshot_protocol="minimal_visible",
            continuation_mode="exact_action",
        )
        try:
            adapter.reset(_SEEDS[0])
            for step in range(_BRANCH_STEP):
                adapter.step(adapter.action(step, "deterministic", (0, 1)))
            adapter.restore(adapter.capture((0,)), (1,))
            records = [
                adapter.step(
                    adapter.action(
                        _BRANCH_STEP + continuation_step,
                        "deterministic",
                        (0, 1),
                    )
                )
                for continuation_step in range(max(_HORIZONS))
            ]
            reference = adapter.decision(
                TrajectoryRecord(steps=records, actions=[], env_id=0),
                "remains_in_contact",
            )
            restored = adapter.decision(
                TrajectoryRecord(steps=records, actions=[], env_id=1),
                "remains_in_contact",
            )
        finally:
            adapter.close()
        results.append(
            {
                "continuation_control": float(control),
                "reference": bool(reference),
                "restored": bool(restored),
                "agreement": bool(reference == restored),
            }
        )
    return results


def _software_provenance() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "configuration": {
            "resource": "ipfd/demo_config.yaml",
            "sha256": hashlib.sha256(_DEMO_CONFIG_TEXT.encode("utf-8")).hexdigest(),
        },
        "software": {
            "python": platform.python_version(),
            "ipfd": version("ipfd"),
            "numpy": version("numpy"),
            "mujoco": version("mujoco"),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(to_builtin(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_demo(output_directory: str | Path) -> dict[str, Any]:
    """Run the control, narrow failure, and improved restore experiments."""

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    control, control_provenance = _run_contract("minimal_visible", 0)
    minimal, minimal_provenance = _run_contract("minimal_visible", _BRANCH_STEP)
    activation_ablation, activation_ablation_provenance = _run_contract(
        "minimal_visible",
        _BRANCH_STEP,
        capture_activation=True,
    )
    integration, integration_provenance = _run_contract(
        "integration_with_warmstart", _BRANCH_STEP
    )
    focus = _focus_trajectory()
    control_decision = _decision_result(control)
    minimal_decision = _decision_result(minimal)
    activation_ablation_decision = _decision_result(activation_ablation)
    integration_decision = _decision_result(integration)
    sensitivity = _sensitivity_results()
    narrow_omissions = [
        "actuator activation state",
        "solver acceleration warm-start",
    ]
    protocols = [
        _protocol_row("minimal_visible (control, branch=0)", control, narrow_omissions),
        _protocol_row("minimal_visible", minimal, narrow_omissions),
        _protocol_row(
            "minimal + activation (ablation)",
            activation_ablation,
            ["solver acceleration warm-start"],
        ),
        _protocol_row("integration_with_warmstart", integration, []),
    ]
    evidence = {
        "schema_version": 1,
        "experiment": "filtered_actuator_floor_contact",
        "branch_step": _BRANCH_STEP,
        "horizons": list(_HORIZONS),
        "independent_seed_groups": list(_SEEDS),
        "continuation": "identical deterministic actions",
        "decision_contract": {
            "name": "remains_in_contact",
            "definition": "true only when floor contact is active at every continuation step",
            "true_label": "remains in contact",
            "false_label": "lifts off",
            "required_horizon": max(_HORIZONS),
        },
        "tolerances": _tolerances(),
        "focus_trajectory": focus,
        "decision_results": {
            "control_minimal_visible": control_decision,
            "preloaded_minimal_visible": minimal_decision,
            "minimal_visible_plus_activation": activation_ablation_decision,
            "integration_with_warmstart": integration_decision,
        },
        "sensitivity": sensitivity,
        "audits": {
            "control_minimal_visible": control,
            "preloaded_minimal_visible": minimal,
            "minimal_visible_plus_activation": activation_ablation,
            "preloaded_integration_with_warmstart": integration,
        },
        "provenance": {
            "execution": _software_provenance(),
            "control": control_provenance,
            "minimal_visible": minimal_provenance,
            "minimal_visible_plus_activation": activation_ablation_provenance,
            "integration_with_warmstart": integration_provenance,
        },
    }
    evidence_path = output / "evidence.json"
    _write_json(evidence_path, evidence)
    focus_verdict = "MATCH" if focus["decision_agreement"] else "FAIL_CLOSED"
    control_row, minimal_row, activation_row, integration_row = protocols
    expectation_met = bool(
        focus_verdict == "FAIL_CLOSED"
        and not minimal_decision["agreement"]
        and control_decision["agreement"]
        and activation_ablation_decision["agreement"]
        and integration_decision["agreement"]
        and control_row["fidelity"]["l0_restore"] == "PASS"
        and control_row["fidelity"]["l1_one_step"] == "PASS"
        and all(
            status == "PASS"
            for status in control_row["fidelity"]["l2_by_horizon"].values()
        )
        and minimal_row["fidelity"]["l0_restore"] == "PASS"
        and minimal_row["fidelity"]["l1_one_step"] == "PASS"
        and minimal_row["fidelity"]["l2_by_horizon"]
        == {"1": "PASS", "5": "PASS", "10": "PASS", "30": "DEGRADED", "90": "FAIL"}
        and activation_row["fidelity"]["l3_decision"] == "PASS"
        and all(
            status == "PASS"
            for status in activation_row["fidelity"]["l2_by_horizon"].values()
        )
        and integration_row["fidelity"]["l0_restore"] == "PASS"
        and integration_row["fidelity"]["l1_one_step"] == "PASS"
        and integration_row["fidelity"]["l3_decision"] == "PASS"
        and all(
            status == "PASS"
            for status in integration_row["fidelity"]["l2_by_horizon"].values()
        )
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "title": "Restore-time equality can still change the downstream conclusion",
        "system": "MuJoCo filtered-actuator floor-contact experiment",
        "focus_protocol": "minimal_visible",
        "experiment": {
            "branch_step": _BRANCH_STEP,
            "horizons": list(_HORIZONS),
            "seed_groups": list(_SEEDS),
            "identical_continuation_actions": True,
            "position_tolerance_m": _POSITION_TOLERANCE_M,
            "velocity_tolerance_mps": _VELOCITY_TOLERANCE_MPS,
        },
        "trajectory": {
            "steps": focus["steps"],
            "reference_position": focus["reference_position"],
            "restored_position": focus["restored_position"],
            "tolerance": _POSITION_TOLERANCE_M,
            "reference_contact_steps": focus["reference_contact_steps"],
            "restored_contact_steps": focus["restored_contact_steps"],
            "reference_decision": focus["reference_decision"],
            "restored_decision": focus["restored_decision"],
            "position_label": "Vertical position (m)",
        },
        "protocols": protocols,
        "decision_results": evidence["decision_results"],
        "sensitivity": sensitivity,
        "demo_expectation_met": expectation_met,
        "measured_result": {
            "first_numerical_divergence_step": focus["first_numerical_divergence_step"],
            "first_contact_disagreement_step": focus["first_contact_disagreement_step"],
            "maximum_position_error_m": focus["maximum_position_error_m"],
            "maximum_velocity_error_mps": focus["maximum_velocity_error_mps"],
            "reference_decision": focus["reference_decision"],
            "restored_decision": focus["restored_decision"],
            "decision_direction": focus["decision_direction"],
            "verdict": focus_verdict,
        },
        "capability_disclosure": {
            "minimal_visible_captured": focus["captured_components"],
            "minimal_visible_unavailable": focus["unavailable_components"],
            "l0_interpretation": (
                "PASS covers measured exposed fields only; unavailable causal state is not inferred equal."
            ),
        },
        "engineering_decision": (
            "Do not use the narrow preloaded snapshot for a 90-step contact conclusion; "
            "capture actuator activation or MuJoCo integration state, or shorten the supported horizon."
            if not focus["decision_agreement"]
            else "The designed decision reversal was not observed; do not use this run as failure evidence."
        ),
    }
    report_path = render_demo_report(summary, output / "report.png")
    summary["artifacts"] = {
        "evidence.json": {"sha256": _sha256(evidence_path), "bytes": evidence_path.stat().st_size},
        "report.png": {"sha256": _sha256(report_path), "bytes": report_path.stat().st_size},
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "artifacts": {
            name: {"sha256": _sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in ("summary.json", "evidence.json", "report.png")
        },
    }
    _write_json(output / "artifact_manifest.json", manifest)
    return summary


def _terminal_report(summary: dict[str, Any], output: Path) -> str:
    by_name = {item["name"]: item for item in summary["protocols"]}
    control = by_name["minimal_visible (control, branch=0)"]
    minimal = by_name["minimal_visible"]
    activation = by_name["minimal + activation (ablation)"]
    integration = by_name["integration_with_warmstart"]
    decisions = summary["decision_results"]
    integration_decision = decisions["integration_with_warmstart"]
    activation_decision = decisions["minimal_visible_plus_activation"]
    lines = [
        "IPFD COUNTERFACTUAL FIDELITY DEMO",
        "=================================",
        "",
        "System: MuJoCo filtered-actuator sphere in floor contact",
        f"Branch step: {_BRANCH_STEP} after actuator preload",
        "Continuation: identical controls in uninterrupted and restored branches",
        "Decision: does the object remain in contact through h=90?",
        "",
        "CONTROL CASE",
        f"minimal_visible at branch=0: {control['fidelity']['l3_decision']}",
        "The omitted actuator state is zero here, so the narrow restore is sufficient.",
        "",
        "RESTORE BOUNDARY",
        f"minimal_visible              {minimal['fidelity']['l0_restore']} (measured exposed fields)",
        f"minimal + activation         {activation['fidelity']['l0_restore']} (exact derived fields)",
        f"integration_with_warmstart   {integration['fidelity']['l0_restore']}",
        "Unavailable under minimal_visible: actuator activation, solver warm-start",
        "",
        "FINITE-HORIZON REPLAY",
        "                             h=1    h=5   h=10   h=30   h=90",
    ]
    for name, row in (
        ("minimal_visible", minimal),
        ("minimal + activation", activation),
        ("integration_with_warmstart", integration),
    ):
        values = [row["fidelity"]["l2_by_horizon"][str(horizon)] for horizon in _HORIZONS]
        lines.append(f"{name:<29}" + " ".join(f"{value:>6}" for value in values))
    measured = summary["measured_result"]
    lines.extend(
        [
            "",
            "DOWNSTREAM DECISION AT h=90",
            "minimal_visible:",
            f"  reference = {measured['reference_decision']}",
            f"  restored  = {measured['restored_decision']}",
            f"  direction = {measured['decision_direction']}",
            f"  verdict   = {measured['verdict']}",
            "minimal + actuator activation (warm-start still omitted):",
            f"  reference = {activation_decision['reference_label']}",
            f"  restored  = {activation_decision['restored_label']}",
            f"  verdict   = {activation_decision['verdict']}",
            "integration_with_warmstart:",
            f"  reference = {integration_decision['reference_label']}",
            f"  restored  = {integration_decision['restored_label']}",
            f"  verdict   = {integration_decision['verdict']}",
            "",
            "MEASURED DIVERGENCE",
            f"  first numerical tolerance crossing: step {measured['first_numerical_divergence_step']}",
            f"  first contact disagreement:          step {measured['first_contact_disagreement_step']}",
            f"  maximum position error:              {measured['maximum_position_error_m']:.6g} m",
            f"  maximum velocity error:              {measured['maximum_velocity_error_mps']:.6g} m/s",
            "",
            "SENSITIVITY (one declared seed, h=90)",
            *[
                "  control {control:.2f}: reference={reference}, restored={restored}, {verdict}".format(
                    control=item["continuation_control"],
                    reference="contact" if item["reference"] else "lift-off",
                    restored="contact" if item["restored"] else "lift-off",
                    verdict="MATCH" if item["agreement"] else "MISMATCH",
                )
                for item in summary["sensitivity"]
            ],
            "",
            "CONCLUSION",
            *(
                [
                    "The restored qpos, qvel, observation, and control matched at t=0, but the",
                    "narrow snapshot omitted filtered-actuator state. A later contact conclusion",
                    "changed. Capturing activation removes the decision mismatch in this ablation;",
                    "integration state also passes every tested level and horizon.",
                ]
                if measured["verdict"] == "FAIL_CLOSED"
                else [
                    "The designed decision reversal was not observed on this run.",
                    "Treat the demo as a regression and do not cite it as failure evidence.",
                ]
            ),
            "",
            f"Artifacts: {output}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the bundled MuJoCo demo and write JSON plus PNG evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ipfd-demo-results"),
        help="artifact directory (default: ./ipfd-demo-results)",
    )
    parser.add_argument("--json", type=Path, help="also copy summary.json to this path")
    args = parser.parse_args(argv)
    try:
        summary = run_demo(args.output)
    except ImportError as exc:
        print(
            "IPFD_DEMO_ERROR: MuJoCo is required. Install with "
            '`python -m pip install -e ".[mujoco]"` or `pip install "ipfd[mujoco]"`. '
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(f"IPFD_DEMO_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    output = args.output.resolve()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "summary.json", args.json)
    print(_terminal_report(summary, output))
    return 0 if bool(summary["demo_expectation_met"]) else 1
