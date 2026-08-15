"""Read-only conversion of the preserved Isaac Lab study into the v2 contract."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import AuditConfig
from .contracts import ContractVerdict
from .provenance import collect_provenance, sha256_file

__all__ = ["run_archived_isaac_audit"]

_SCENE_FIELDS = (
    "robot_root_pose",
    "robot_root_velocity",
    "joint_position",
    "joint_velocity",
    "object_pose",
    "object_velocity",
)
_COUNTER_FIELDS = ("episode_length", "sim_step_counter", "common_step_counter")


def _resolve_records_path(config: AuditConfig) -> Path:
    value = config.adapter.get("records_path")
    if not isinstance(value, str) or not value:
        raise ValueError("isaac_lab_archive adapter requires records_path")
    path = Path(value)
    if not path.is_absolute():
        path = (config.source_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"archived branch records are unavailable: {path}")
    expected = config.adapter.get("expected_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("isaac_lab_archive adapter requires a 64-character expected_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"archived branch records digest mismatch: expected {expected}, got {actual}")
    return path


def _difference_result(values: dict[str, float], tolerance: float) -> dict[str, Any]:
    fields = {
        name: {
            "max_abs": value,
            "threshold": tolerance,
            "within_tolerance": value <= tolerance,
        }
        for name, value in sorted(values.items())
    }
    maximum = max(values.values(), default=0.0)
    return {
        "passed": all(item["within_tolerance"] for item in fields.values()),
        "maximum_absolute_error": maximum,
        "absolute_tolerance": tolerance,
        "relative_tolerance": None,
        "fields": fields,
        "raw_reference_and_candidate_values_retained": False,
    }


def _record(config: AuditConfig, raw: dict[str, Any]) -> dict[str, Any]:
    immediate = {str(key): float(value) for key, value in raw["immediate_mismatch"].items()}
    first_step = {str(key): float(value) for key, value in raw["first_step_mismatch"].items()}
    scene_tol = config.tolerance("scene_state")[0]
    obs_tol = config.tolerance("policy_observations")[0]
    contact_tol = config.tolerance("contact_state")[0]
    target_tol = config.tolerance("controller_targets")[0]
    counter_tol = config.tolerance("counters")[0]

    l0_categories = {
        "scene_state": _difference_result({name: immediate[name] for name in _SCENE_FIELDS}, scene_tol),
        "policy_observations": _difference_result({"observation": immediate["observation"]}, obs_tol),
        "privileged_observations": {
            "passed": None,
            "unavailable": True,
            "reason": "not retained by archived study",
        },
        "task_state": _difference_result({"command": immediate["command"]}, scene_tol),
        "controller_targets": _difference_result({"targets": immediate["targets"]}, target_tol),
        "sensor_state": _difference_result({"contact_force": immediate["contact_force"]}, contact_tol),
        "counters": _difference_result({name: immediate[name] for name in _COUNTER_FIELDS}, counter_tol),
    }
    measured_l0 = [value for value in l0_categories.values() if value.get("passed") is not None]
    l0_passed = all(bool(value["passed"]) for value in measured_l0)

    l1_state = _difference_result({name: first_step[name] for name in _SCENE_FIELDS}, scene_tol)
    l1_observation = _difference_result({"observation": first_step["observation"]}, obs_tol)
    l1_contact = _difference_result({"contact_force": first_step["contact_force"]}, contact_tol)
    l1_task = _difference_result(
        {
            "command": first_step["command"],
            "targets": first_step["targets"],
            **{name: first_step[name] for name in _COUNTER_FIELDS},
        },
        scene_tol,
    )
    l1 = {
        "verdict": ContractVerdict.INSUFFICIENT_EVIDENCE.value,
        "passed": None,
        "numerical_passed": bool(l1_state["passed"] and l1_observation["passed"]),
        "semantic_passed": bool(l1_contact["passed"]),
        "numerical": {"next_state": l1_state, "next_observation": l1_observation},
        "semantic": {"contact_state": l1_contact, "task_outputs": l1_task},
        "unavailable": ["termination_state", "reward"],
    }
    l2_passed = (
        raw.get("first_numerical_divergence_step") is None
        and raw.get("first_observation_divergence_step") is None
        and raw.get("first_contact_divergence_step") is None
    )
    l2 = {
        "verdict": ContractVerdict.SUPPORTED.value if l2_passed else ContractVerdict.UNSUPPORTED.value,
        "passed": l2_passed,
        "identical_actions": raw.get("first_action_divergence_step") is None,
        "first_numerical_divergence": raw.get("first_numerical_divergence_step"),
        "first_observation_divergence": raw.get("first_observation_divergence_step"),
        "first_contact_divergence": raw.get("first_contact_divergence_step"),
        "maximum_state_error": float(raw["maximum_trajectory_error"]),
        "terminal_state_error": float(raw["terminal_trajectory_error"]),
        "divergence_growth_curve": None,
        "unavailable": ["per-step divergence-growth curve"],
    }
    agreement = bool(raw["decision_match"])
    decision_name = str(raw["predicate"])
    l3 = {
        "verdict": ContractVerdict.SUPPORTED.value if agreement else ContractVerdict.UNSUPPORTED.value,
        "passed": agreement,
        "decision_disagreement": not agreement,
        "decisions": {
            decision_name: {
                "reference": bool(raw["reference_decision"]),
                "restored": bool(raw["candidate_decision"]),
                "agreement": agreement,
                "verdict": ContractVerdict.SUPPORTED.value if agreement else ContractVerdict.UNSUPPORTED.value,
            }
        },
    }
    return {
        "schema_version": 1,
        "branch_id": str(raw["branch_id"]),
        "branch_step": int(raw["branch_step"]),
        "seed": int(raw["base_seed"]),
        "cluster": str(raw["base_seed"]),
        "horizon": int(raw["horizon"]),
        "action_source": config.action_source,
        "continuation_mode": config.continuation_mode,
        "snapshot_protocol": config.snapshot_protocol,
        "disturbance": raw["disturbance"],
        "phase": raw["phase"],
        "levels": {
            "L0": {
                "verdict": ContractVerdict.SUPPORTED.value if l0_passed else ContractVerdict.UNSUPPORTED.value,
                "passed": l0_passed,
                "measured_exposed_state_only": True,
                "categories": l0_categories,
            },
            "L1": l1,
            "L2": l2,
            "L3": l3,
        },
    }


def _aggregate(config: AuditConfig, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        decision_name = next(iter(record["levels"]["L3"]["decisions"]))
        groups[(record["horizon"], decision_name)].append(record)
    configurations: list[dict[str, Any]] = []
    for (horizon, decision_name), group in sorted(groups.items()):
        clusters = sorted({record["cluster"] for record in group})
        enough = len(clusters) >= config.minimum_independent_clusters
        disagreements = sum(record["levels"]["L3"]["decision_disagreement"] for record in group)
        first_divergences = [
            record["levels"]["L2"]["first_numerical_divergence"]
            for record in group
            if record["levels"]["L2"]["first_numerical_divergence"] is not None
        ]
        l0_passed = all(record["levels"]["L0"]["passed"] for record in group)
        l2_passed = all(record["levels"]["L2"]["passed"] for record in group)
        level_values = (l0_passed, None, l2_passed, disagreements == 0)
        if not enough:
            result = ContractVerdict.INSUFFICIENT_EVIDENCE.value
        elif any(value is False for value in level_values):
            result = ContractVerdict.UNSUPPORTED.value
        else:
            result = ContractVerdict.INSUFFICIENT_EVIDENCE.value
        scope = {
            "simulator": "Isaac Lab",
            "simulator_version": config.simulator_version,
            "environment": config.environment,
            "task": config.task,
            "snapshot_protocol": config.snapshot_protocol,
            "continuation_mode": config.continuation_mode,
            "horizon": horizon,
            "action_source": config.action_source,
            "decision_function": decision_name,
            "tolerances": config.tolerances,
            "independent_cluster_key": config.independent_cluster_key,
            "hardware_and_software_provenance": "provenance.json",
            "evidence_mode": "immutable_archived_pairs",
        }
        configurations.append(
            {
                "comparison_key": {
                    "environment": config.environment,
                    "task": config.task,
                    "continuation_mode": config.continuation_mode,
                    "horizon": horizon,
                    "action_source": config.action_source,
                    "decision_function": decision_name,
                },
                "scope": scope,
                "result": result,
                "independent_clusters": clusters,
                "minimum_independent_clusters": config.minimum_independent_clusters,
                "levels": {
                    "L0": {
                        "passed": l0_passed if enough else None,
                    },
                    "L1": {"passed": None},
                    "L2": {
                        "passed": l2_passed if enough else None,
                        "first_numerical_divergence": min(first_divergences) if first_divergences else None,
                        "maximum_state_error": max(record["levels"]["L2"]["maximum_state_error"] for record in group),
                    },
                    "L3": {
                        "passed": disagreements == 0 if enough else None,
                        "decision_disagreement": bool(disagreements) if enough else None,
                        "disagreements": disagreements,
                        "comparisons": len(group),
                    },
                },
            }
        )
    return configurations


def run_archived_isaac_audit(config: AuditConfig) -> dict[str, Any]:
    """Audit preserved paired results without altering or regenerating them."""

    path = _resolve_records_path(config)
    selected_seeds = {state.seed for state in config.branch_states}
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid archived JSONL at line {line_number}: {exc}") from exc
            if raw.get("protocol") != config.snapshot_protocol:
                continue
            if raw.get("continuation") != config.continuation_mode:
                continue
            if raw.get("predicate") not in config.decision_functions:
                continue
            if int(raw.get("horizon", -1)) not in config.horizons:
                continue
            if int(raw.get("base_seed", -1)) not in selected_seeds:
                continue
            records.append(_record(config, raw))
    if not records:
        raise RuntimeError("no archived records matched the declared audit scope")
    configurations = _aggregate(config, records)
    failures = [record for record in records if record["levels"]["L3"]["decision_disagreement"]]
    failures.sort(key=lambda item: (item["horizon"], item["branch_step"], item["branch_id"]))
    minimal = None
    if failures:
        failure = failures[0]
        decision_name = next(iter(failure["levels"]["L3"]["decisions"]))
        decision = failure["levels"]["L3"]["decisions"][decision_name]
        minimal = {
            "schema_version": 1,
            "status": "ARCHIVED_RECORD_REDUCED_BUT_NOT_SELF_CONTAINED",
            "minimal": {
                "branch_id": failure["branch_id"],
                "branch_step": failure["branch_step"],
                "seed": failure["seed"],
                "horizon": failure["horizon"],
                "decision_name": decision_name,
            },
            "record": failure,
            "expected_decision": {decision_name: decision["reference"]},
            "restored_decision": {decision_name: decision["restored"]},
            "first_divergence_point": {
                "numerical": failure["levels"]["L2"]["first_numerical_divergence"],
                "observation": failure["levels"]["L2"]["first_observation_divergence"],
                "contact": failure["levels"]["L2"]["first_contact_divergence"],
            },
            "captured_snapshot": None,
            "identical_actions": None,
            "missing": [
                "captured snapshot values",
                "recorded action sequence values",
                "per-step divergence-growth curve",
            ],
            "note": "The immutable study retained paired metrics, not a self-contained snapshot/action reproducer.",
        }
    adapter_provenance = {
        "adapter": "isaac_lab_archive",
        "simulator": "Isaac Lab",
        "simulator_version": config.simulator_version,
        "records_sha256": sha256_file(path),
        "records_bytes": path.stat().st_size,
        "archival_commit": config.adapter.get("archival_commit"),
        "source_mode": "read_only_immutable_evidence",
        "unsupported_restoration_claims": [
            "The archive does not establish complete PhysX snapshot restoration.",
            "This conversion does not rerun the simulator.",
        ],
    }
    return {
        "schema_version": 1,
        "configurations": configurations,
        "records": records,
        "minimal_reproducer": minimal,
        "provenance": collect_provenance(
            adapter=adapter_provenance,
            config_path=config.source_path,
            repo_root=Path(__file__).resolve().parents[3],
            ignored_status_paths=(
                Path(__file__).resolve().parents[3] / "results" / "v2",
                config.output_directory,
            ),
        ),
    }
