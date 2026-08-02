"""L0 through L3 replay-fidelity audit runner."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .comparison import compare_values, extract_env, maximum_error
from .config import AuditConfig, BranchState
from .contracts import ContractVerdict, ObservationRecord, ReplayAdapter, StepRecord, TrajectoryRecord, to_builtin
from .minimizer import minimize_failure
from .provenance import collect_provenance

__all__ = ["audit_configuration", "create_adapter", "run_audit"]


def create_adapter(config: AuditConfig) -> ReplayAdapter:
    """Instantiate the adapter named by a validated audit configuration."""

    kind = str(config.adapter["kind"])
    if kind == "mujoco":
        from ..adapters.mujoco_replay import MuJoCoReplayAdapter

        return MuJoCoReplayAdapter(
            config.adapter,
            snapshot_protocol=config.snapshot_protocol,
            continuation_mode=config.continuation_mode,
        )
    if kind == "isaac_lab":
        from ..adapters.isaac_replay import IsaacLabReplayAdapter

        return IsaacLabReplayAdapter.from_config(
            config.adapter,
            snapshot_protocol=config.snapshot_protocol,
            continuation_mode=config.continuation_mode,
        )
    raise ValueError(f"unknown live adapter kind: {kind!r}")


def _compare_observation(config: AuditConfig, observation: ObservationRecord) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    comparable_fields = 0
    passed = True
    for category, values in observation.categories().items():
        absolute, relative = config.tolerance(category)
        result = compare_values(
            extract_env(values, 0),
            extract_env(values, 1),
            absolute=absolute,
            relative=relative,
        )
        categories[category] = result
        comparable_fields += int(result["comparable_fields"])
        passed = passed and bool(result["passed"])
    if comparable_fields == 0:
        verdict = ContractVerdict.INSUFFICIENT_EVIDENCE.value
    else:
        verdict = (
            ContractVerdict.SUPPORTED.value
            if passed
            else ContractVerdict.UNSUPPORTED.value
        )
    return {
        "verdict": verdict,
        "passed": passed if comparable_fields else None,
        "measured_exposed_state_only": True,
        "comparable_fields": comparable_fields,
        "unavailable": list(observation.unavailable),
        "categories": categories,
    }


def _compare_step(config: AuditConfig, step: StepRecord) -> dict[str, Any]:
    def compare(category: str, values: Mapping[str, Any]) -> dict[str, Any]:
        absolute, relative = config.tolerance(category)
        return compare_values(
            extract_env(values, 0),
            extract_env(values, 1),
            absolute=absolute,
            relative=relative,
        )

    state = compare("scene_state", step.observation.scene_state)
    observation = compare("policy_observations", step.observation.policy_observations)
    contact = compare("contact_state", step.contact_state)
    task_outputs = compare("task_outputs", step.task_outputs)
    termination = compare("termination", step.terminated)
    reward = compare("reward", step.reward)
    numerical_passed = bool(state["passed"] and observation["passed"] and reward["passed"])
    semantic_passed = bool(contact["passed"] and task_outputs["passed"] and termination["passed"])
    comparable_fields = sum(
        int(item["comparable_fields"])
        for item in (state, observation, contact, task_outputs, termination, reward)
    )
    passed = numerical_passed and semantic_passed
    verdict = (
        ContractVerdict.INSUFFICIENT_EVIDENCE.value
        if comparable_fields == 0
        else ContractVerdict.SUPPORTED.value
        if passed
        else ContractVerdict.UNSUPPORTED.value
    )
    return {
        "verdict": verdict,
        "passed": passed if comparable_fields else None,
        "numerical_passed": numerical_passed if comparable_fields else None,
        "semantic_passed": semantic_passed if comparable_fields else None,
        "numerical": {
            "next_state": state,
            "next_observation": observation,
            "reward": reward,
        },
        "semantic": {
            "contact_state": contact,
            "task_outputs": task_outputs,
            "termination": termination,
        },
    }


def _actions_identical(actions: Any) -> bool:
    try:
        array = np.asarray(actions)
    except (TypeError, ValueError):
        return False
    if array.ndim == 0 or array.shape[0] < 2:
        return False
    return bool(np.array_equal(array[0], array[1]))


def _run_once(
    adapter: ReplayAdapter,
    config: AuditConfig,
    branch: BranchState,
    horizon: int,
    *,
    decision_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    reset = getattr(adapter, "reset", None)
    action = getattr(adapter, "action", None)
    if not callable(reset) or not callable(action):
        raise TypeError("live audit adapters must implement reset(seed) and action(step, source, env_ids)")
    reset(branch.seed)
    for step_index in range(branch.step):
        branch_actions = action(step_index, config.action_source, (0, 1))
        if not _actions_identical(branch_actions):
            raise RuntimeError("branch preparation actions are not identical across the paired environments")
        adapter.step(branch_actions)

    snapshot = adapter.capture((0,))
    adapter.restore(snapshot, (1,))
    restored_observation = adapter.observe((0, 1))
    l0 = _compare_observation(config, restored_observation)

    steps: list[StepRecord] = []
    actions: list[Any] = []
    action_identity = True
    growth: list[dict[str, Any]] = []
    first_numerical: int | None = None
    first_observation: int | None = None
    first_contact: int | None = None
    maximum_state_error = 0.0
    terminal_state_error = 0.0
    l1: dict[str, Any] | None = None

    for continuation_step in range(1, horizon + 1):
        values = action(branch.step + continuation_step - 1, config.action_source, (0, 1))
        identical = _actions_identical(values)
        action_identity = action_identity and identical
        if not identical and config.continuation_mode in {"exact_action", "open_loop", "restored", "cold"}:
            raise RuntimeError("open-loop continuation actions are not identical")
        step_record = adapter.step(values)
        steps.append(step_record)
        actions.append(to_builtin(values))
        step_comparison = _compare_step(config, step_record)
        if continuation_step == 1:
            l1 = step_comparison
        state = step_comparison["numerical"]["next_state"]
        observation = step_comparison["numerical"]["next_observation"]
        contact = step_comparison["semantic"]["contact_state"]
        state_error = maximum_error(state)
        observation_error = maximum_error(observation)
        contact_error = maximum_error(contact)
        maximum_state_error = max(maximum_state_error, state_error)
        terminal_state_error = state_error
        if first_numerical is None and not state["passed"]:
            first_numerical = continuation_step
        if first_observation is None and not observation["passed"]:
            first_observation = continuation_step
        if first_contact is None and not contact["passed"]:
            first_contact = continuation_step
        growth.append(
            {
                "step": continuation_step,
                "state_max_abs": state_error,
                "observation_max_abs": observation_error,
                "contact_max_abs": contact_error,
            }
        )

    if l1 is None:  # pragma: no cover - config rejects a zero horizon
        raise RuntimeError("one-step record was not produced")
    l2_passed = first_numerical is None and first_observation is None and first_contact is None
    l2 = {
        "verdict": (
            ContractVerdict.SUPPORTED.value
            if l2_passed
            else ContractVerdict.UNSUPPORTED.value
        ),
        "passed": l2_passed,
        "identical_actions": action_identity,
        "first_numerical_divergence": first_numerical,
        "first_observation_divergence": first_observation,
        "first_contact_divergence": first_contact,
        "maximum_state_error": maximum_state_error,
        "terminal_state_error": terminal_state_error,
        "divergence_growth_curve": growth,
    }

    reference_trajectory = TrajectoryRecord(steps=steps, actions=actions, env_id=0)
    restored_trajectory = TrajectoryRecord(steps=steps, actions=actions, env_id=1)
    decisions: dict[str, Any] = {}
    for name in decision_names or config.decision_functions:
        reference_decision = bool(adapter.decision(reference_trajectory, name))
        restored_decision = bool(adapter.decision(restored_trajectory, name))
        decisions[name] = {
            "reference": reference_decision,
            "restored": restored_decision,
            "agreement": reference_decision == restored_decision,
            "verdict": (
                ContractVerdict.SUPPORTED.value
                if reference_decision == restored_decision
                else ContractVerdict.UNSUPPORTED.value
            ),
        }
    l3_passed = all(bool(item["agreement"]) for item in decisions.values())
    l3 = {
        "verdict": (
            ContractVerdict.SUPPORTED.value
            if l3_passed
            else ContractVerdict.UNSUPPORTED.value
        ),
        "passed": l3_passed,
        "decision_disagreement": not l3_passed,
        "decisions": decisions,
    }
    return {
        "schema_version": 1,
        "branch_id": branch.id,
        "branch_step": branch.step,
        "seed": branch.seed,
        "cluster": branch.cluster,
        "horizon": horizon,
        "action_source": config.action_source,
        "continuation_mode": config.continuation_mode,
        "snapshot_protocol": config.snapshot_protocol,
        "snapshot_inventory": {
            "captured_components": list(snapshot.captured_components),
            "unavailable_components": list(snapshot.unavailable_components),
            "metadata": to_builtin(snapshot.metadata),
        },
        "levels": {"L0": l0, "L1": l1, "L2": l2, "L3": l3},
        "captured_snapshot": snapshot.to_dict(),
        "identical_actions": actions,
    }


def _aggregate(config: AuditConfig, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def aggregate_passed(values: Sequence[bool | None]) -> bool | None:
        if any(value is False for value in values):
            return False
        if any(value is None for value in values) or not values:
            return None
        return True

    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        for decision_name in config.decision_functions:
            groups[(int(record["horizon"]), decision_name)].append(record)
    configurations: list[dict[str, Any]] = []
    for (horizon, decision_name), group in sorted(groups.items()):
        clusters = sorted({str(record["cluster"]) for record in group})
        enough = len(clusters) >= config.minimum_independent_clusters
        l0_values = [record["levels"]["L0"]["passed"] for record in group]
        l1_values = [record["levels"]["L1"]["passed"] for record in group]
        l2_values = [record["levels"]["L2"] for record in group]
        decisions = [record["levels"]["L3"]["decisions"][decision_name] for record in group]
        l0_passed = aggregate_passed(l0_values)
        l1_passed = aggregate_passed(l1_values)
        first_divergences = [
            int(value["first_numerical_divergence"])
            for value in l2_values
            if value["first_numerical_divergence"] is not None
        ]
        decision_disagreement = any(not bool(value["agreement"]) for value in decisions)
        l2_passed = all(bool(value["passed"]) for value in l2_values)
        level_values = (l0_passed, l1_passed, l2_passed, not decision_disagreement)
        if not enough:
            result = ContractVerdict.INSUFFICIENT_EVIDENCE.value
        elif any(value is False for value in level_values):
            result = ContractVerdict.UNSUPPORTED.value
        elif any(value is None for value in level_values):
            result = ContractVerdict.INSUFFICIENT_EVIDENCE.value
        else:
            result = ContractVerdict.SUPPORTED.value
        scope = {
            "simulator": str(adapter_value(config.adapter, "simulator", config.adapter["kind"])),
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
                    "L0": {"passed": l0_passed if enough else None},
                    "L1": {"passed": l1_passed if enough else None},
                    "L2": {
                        "passed": l2_passed if enough else None,
                        "first_numerical_divergence": min(first_divergences) if first_divergences else None,
                        "maximum_state_error": max(float(value["maximum_state_error"]) for value in l2_values),
                    },
                    "L3": {
                        "passed": not decision_disagreement if enough else None,
                        "decision_disagreement": decision_disagreement if enough else None,
                        "disagreements": sum(not bool(value["agreement"]) for value in decisions),
                        "comparisons": len(decisions),
                    },
                },
            }
        )
    return configurations


def adapter_value(adapter: Mapping[str, Any], key: str, default: Any) -> Any:
    value = adapter.get(key, default)
    return default if value is None else value


def audit_configuration(adapter: ReplayAdapter, config: AuditConfig) -> dict[str, Any]:
    """Execute all branches and horizons, then reduce the first L2 or L3 failure."""

    records: list[dict[str, Any]] = []
    first_failure: tuple[BranchState, int, dict[str, Any], str, str] | None = None
    for branch in config.branch_states:
        for horizon in config.horizons:
            record = _run_once(adapter, config, branch, horizon)
            records.append(record)
            l3_failures = [
                name
                for name, decision in record["levels"]["L3"]["decisions"].items()
                if not decision["agreement"]
            ]
            if first_failure is None and l3_failures:
                first_failure = (branch, horizon, record, "L3", l3_failures[0])
            elif first_failure is None and not record["levels"]["L2"]["passed"]:
                first_failure = (branch, horizon, record, "L2", config.decision_functions[0])

    configurations = _aggregate(config, records)
    minimal_reproducer: dict[str, Any] | None = None
    reduction_enabled = bool(config.reduction.get("enabled", True))
    if first_failure is not None and reduction_enabled:
        branch, horizon, failure_record, failure_kind, decision_name = first_failure
        dimensions_fn = getattr(adapter, "reduction_dimensions", None)
        dimensions = dict(dimensions_fn()) if callable(dimensions_fn) else {}
        configure_reduction = getattr(adapter, "configure_reduction", None)

        def evaluate(candidate: dict[str, Any]) -> Mapping[str, Any]:
            if callable(configure_reduction):
                configure_reduction(candidate)
            candidate_branch = replace(branch, step=int(candidate["branch_step"]))
            rerun = _run_once(
                adapter,
                config,
                candidate_branch,
                int(candidate["horizon"]),
                decision_names=(str(candidate["decision_name"]),),
            )
            if failure_kind == "L3":
                fails = bool(rerun["levels"]["L3"]["decision_disagreement"])
            else:
                fails = not bool(rerun["levels"]["L2"]["passed"])
            public_record = {key: value for key, value in rerun.items() if key not in {"captured_snapshot", "identical_actions"}}
            return {
                "fails": fails,
                "record": public_record,
                "captured_snapshot": rerun["captured_snapshot"],
                "identical_actions": rerun["identical_actions"],
                "adapter_provenance": to_builtin(adapter.provenance()),
            }

        reduction = minimize_failure(
            {
                "branch_id": branch.id,
                "branch_step": branch.step,
                "seed": branch.seed,
                "cluster": branch.cluster,
                "horizon": horizon,
                "action_sequence_length": horizon,
                "decision_name": decision_name,
                "failure_kind": failure_kind,
                "fails": True,
                "record": failure_record,
                "captured_snapshot": failure_record["captured_snapshot"],
                "identical_actions": failure_record["identical_actions"],
                "adapter_provenance": to_builtin(adapter.provenance()),
            },
            evaluate,
            branch_steps=tuple(config.reduction.get("branch_steps", ())),
            decision_names=config.decision_functions if failure_kind == "L3" else (),
            adapter_dimensions=dimensions,
            max_trials=int(config.reduction.get("max_trials", 100)),
        )
        minimal_record = reduction.get("minimal_result") or failure_record
        minimal_decisions = minimal_record["levels"]["L3"]["decisions"]
        minimal_evidence = reduction.get("minimal_evidence", {})
        minimal_reproducer = {
            "schema_version": 1,
            "scope": configurations[0]["scope"] if configurations else {},
            "captured_snapshot": minimal_evidence.get("captured_snapshot"),
            "identical_actions": minimal_evidence.get("identical_actions"),
            "expected_decision": {
                name: value["reference"] for name, value in minimal_decisions.items()
            },
            "restored_decision": {
                name: value["restored"] for name, value in minimal_decisions.items()
            },
            "first_divergence_point": {
                "numerical": minimal_record["levels"]["L2"]["first_numerical_divergence"],
                "observation": minimal_record["levels"]["L2"]["first_observation_divergence"],
                "contact": minimal_record["levels"]["L2"]["first_contact_divergence"],
            },
            "failure_kind": failure_kind,
            "reduction": reduction,
            "required_assets_and_versions": to_builtin(adapter.provenance()),
        }
    public_records = [
        {key: value for key, value in record.items() if key not in {"captured_snapshot", "identical_actions"}}
        for record in records
    ]
    return {
        "schema_version": 1,
        "configurations": configurations,
        "records": public_records,
        "minimal_reproducer": minimal_reproducer,
    }


def run_audit(config: AuditConfig, *, adapter: ReplayAdapter | None = None) -> dict[str, Any]:
    """Run one live audit and attach provenance. Output writing is separate."""

    owned_adapter = adapter is None
    selected_adapter = adapter or create_adapter(config)
    try:
        initial_adapter_provenance = selected_adapter.provenance()
        actual_version = initial_adapter_provenance.get("simulator_version")
        if actual_version is None:
            raise RuntimeError(
                "adapter provenance must report simulator_version for a live audit"
            )
        if str(actual_version) != config.simulator_version:
            raise RuntimeError(
                "declared simulator_version does not match the live adapter: "
                f"declared {config.simulator_version!r}, actual {actual_version!r}"
            )
        result = audit_configuration(selected_adapter, config)
        result["provenance"] = collect_provenance(
            adapter=selected_adapter.provenance(),
            config_path=config.source_path,
            repo_root=Path(__file__).resolve().parents[3],
            ignored_status_paths=(
                Path(__file__).resolve().parents[3] / "results" / "v2",
                config.output_directory,
            ),
        )
        return result
    finally:
        if owned_adapter:
            close = getattr(selected_adapter, "close", None)
            if callable(close):
                close()
