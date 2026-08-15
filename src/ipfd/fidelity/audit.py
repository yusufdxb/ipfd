"""L0 through L3 replay-fidelity audit runner."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
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
    if kind == "python":
        factory_name = config.adapter.get("factory")
        if not isinstance(factory_name, str) or ":" not in factory_name:
            raise ValueError("adapter.factory must use trusted.module:factory syntax")
        module_name, attribute_name = factory_name.split(":", 1)
        if not module_name or not attribute_name:
            raise ValueError("adapter.factory must use trusted.module:factory syntax")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as first_error:
            cwd = str(Path.cwd())
            if cwd not in sys.path:
                sys.path.insert(0, cwd)
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                raise first_error from None
        factory = getattr(module, attribute_name)
        if not callable(factory):
            raise TypeError("adapter.factory target must be callable")
        kwargs = config.adapter.get("kwargs", {})
        if not isinstance(kwargs, Mapping):
            raise TypeError("adapter.kwargs must be a mapping")
        adapter = factory(**dict(kwargs))
        if not isinstance(adapter, ReplayAdapter):
            raise TypeError("adapter.factory did not return a ReplayAdapter")
        return adapter
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
            field_tolerances=config.field_tolerances(category),
        )
        categories[category] = result
        comparable_fields += int(result["comparable_fields"])
        passed = passed and bool(result["passed"])
    required_evidence = bool(
        int(categories["scene_state"]["comparable_fields"]) > 0
        and int(categories["policy_observations"]["comparable_fields"]) > 0
    )
    if not required_evidence:
        verdict = ContractVerdict.INSUFFICIENT_EVIDENCE.value
    else:
        verdict = (
            ContractVerdict.SUPPORTED.value
            if passed
            else ContractVerdict.UNSUPPORTED.value
        )
    return {
        "verdict": verdict,
        "passed": passed if required_evidence else None,
        "measured_exposed_state_only": True,
        "comparable_fields": comparable_fields,
        "required_evidence": {
            "scene_state_comparable_fields": int(
                categories["scene_state"]["comparable_fields"]
            ),
            "policy_observations_comparable_fields": int(
                categories["policy_observations"]["comparable_fields"]
            ),
            "sufficient": required_evidence,
        },
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
            field_tolerances=config.field_tolerances(category),
        )

    state = compare("scene_state", step.observation.scene_state)
    observation = compare("policy_observations", step.observation.policy_observations)
    contact = compare("contact_state", step.contact_state)
    task_outputs = compare("task_outputs", step.task_outputs)
    termination = compare("termination", step.terminated)
    reward = compare("reward", step.reward)
    semantic_events = compare("semantic", step.semantic)
    required_evidence = bool(
        int(state["comparable_fields"]) > 0
        and int(observation["comparable_fields"]) > 0
    )
    numerical_passed = bool(state["passed"] and observation["passed"] and reward["passed"])
    semantic_passed = bool(
        contact["passed"]
        and task_outputs["passed"]
        and termination["passed"]
        and semantic_events["passed"]
    )
    passed = numerical_passed and semantic_passed
    verdict = (
        ContractVerdict.INSUFFICIENT_EVIDENCE.value
        if not required_evidence
        else ContractVerdict.SUPPORTED.value
        if passed
        else ContractVerdict.UNSUPPORTED.value
    )
    return {
        "verdict": verdict,
        "passed": passed if required_evidence else None,
        "numerical_passed": numerical_passed if required_evidence else None,
        "semantic_passed": semantic_passed if required_evidence else None,
        "required_evidence": {
            "next_state_comparable_fields": int(state["comparable_fields"]),
            "next_observation_comparable_fields": int(observation["comparable_fields"]),
            "sufficient": required_evidence,
        },
        "numerical": {
            "next_state": state,
            "next_observation": observation,
            "reward": reward,
        },
        "semantic": {
            "contact_state": contact,
            "task_outputs": task_outputs,
            "termination": termination,
            "events": semantic_events,
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


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        to_builtin(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _prefix_sha256(values: Sequence[Any]) -> list[str]:
    digest = hashlib.sha256()
    result: list[str] = []
    for value in values:
        payload = _canonical_bytes(value)
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
        result.append(digest.hexdigest())
    return result


def _applied_actions_match(requested: Any, applied: Any) -> bool | None:
    if applied is None:
        return None
    return bool(
        _actions_identical(applied)
        and _canonical_sha256(requested) == _canonical_sha256(applied)
    )


def _divergent_fields(step_comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain compact, signed threshold-crossing evidence for debugging."""

    events: list[dict[str, Any]] = []
    for channel in ("numerical", "semantic"):
        for category, comparison in step_comparison[channel].items():
            for field, metrics in comparison["fields"].items():
                if bool(metrics.get("within_tolerance", False)):
                    continue
                events.append(
                    {
                        "channel": channel,
                        "category": category,
                        "field": field,
                        "max_abs": metrics.get("max_abs"),
                        "signed_candidate_minus_reference": metrics.get("signed_at_max"),
                        "reference_at_max": metrics.get(
                            "reference_at_max",
                            metrics.get("reference_at_first_difference"),
                        ),
                        "candidate_at_max": metrics.get(
                            "candidate_at_max",
                            metrics.get("candidate_at_first_difference"),
                        ),
                        "threshold": metrics.get("threshold"),
                        "unit": metrics.get("unit"),
                        "comparison": metrics.get("comparison", "numeric_tolerance"),
                    }
                )
    return events


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
    preparation_action_delivery: bool | None = True
    for step_index in range(branch.step):
        branch_actions = action(step_index, config.action_source, (0, 1))
        if not _actions_identical(branch_actions):
            raise RuntimeError("branch preparation actions are not identical across the paired environments")
        preparation_record = adapter.step(branch_actions)
        preparation_delivery = _applied_actions_match(
            branch_actions, preparation_record.applied_actions
        )
        if preparation_delivery is False:
            preparation_action_delivery = False
        elif preparation_delivery is None and preparation_action_delivery is not False:
            preparation_action_delivery = None

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
    first_task_output: int | None = None
    first_termination: int | None = None
    first_reward: int | None = None
    first_semantic_event: int | None = None
    first_action_delivery_mismatch: int | None = None
    evidence_sufficient = True
    action_delivery: bool | None = preparation_action_delivery
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
        delivered = _applied_actions_match(values, step_record.applied_actions)
        if delivered is False:
            action_delivery = False
            if first_action_delivery_mismatch is None:
                first_action_delivery_mismatch = continuation_step
        elif delivered is None and action_delivery is not False:
            action_delivery = None
        step_comparison["action_delivery"] = {
            "requested_identical": identical,
            "applied_actions_disclosed": delivered is not None,
            "applied_matches_request": delivered,
        }
        if delivered is None:
            step_comparison["verdict"] = ContractVerdict.INSUFFICIENT_EVIDENCE.value
            step_comparison["passed"] = None
        elif delivered is False:
            step_comparison["verdict"] = ContractVerdict.UNSUPPORTED.value
            step_comparison["passed"] = False
        if continuation_step == 1:
            l1 = step_comparison
        state = step_comparison["numerical"]["next_state"]
        observation = step_comparison["numerical"]["next_observation"]
        contact = step_comparison["semantic"]["contact_state"]
        task_output = step_comparison["semantic"]["task_outputs"]
        termination = step_comparison["semantic"]["termination"]
        semantic_event = step_comparison["semantic"]["events"]
        reward = step_comparison["numerical"]["reward"]
        state_error = maximum_error(state)
        observation_error = maximum_error(observation)
        contact_error = maximum_error(contact)
        maximum_state_error = max(maximum_state_error, state_error)
        terminal_state_error = state_error
        evidence_sufficient = evidence_sufficient and step_comparison["passed"] is not None
        if first_numerical is None and not state["passed"]:
            first_numerical = continuation_step
        if first_observation is None and not observation["passed"]:
            first_observation = continuation_step
        if first_contact is None and not contact["passed"]:
            first_contact = continuation_step
        if first_task_output is None and not task_output["passed"]:
            first_task_output = continuation_step
        if first_termination is None and not termination["passed"]:
            first_termination = continuation_step
        if first_reward is None and not reward["passed"]:
            first_reward = continuation_step
        if first_semantic_event is None and not semantic_event["passed"]:
            first_semantic_event = continuation_step
        growth.append(
            {
                "step": continuation_step,
                "state_max_abs": state_error,
                "observation_max_abs": observation_error,
                "contact_max_abs": contact_error,
                "divergent_fields": _divergent_fields(step_comparison),
            }
        )

    if l1 is None:  # pragma: no cover - config rejects a zero horizon
        raise RuntimeError("one-step record was not produced")
    divergence_steps = {
        "state": first_numerical,
        "observation": first_observation,
        "contact": first_contact,
        "task_output": first_task_output,
        "termination": first_termination,
        "reward": first_reward,
        "semantic_event": first_semantic_event,
        "action_delivery": first_action_delivery_mismatch,
    }
    earliest_divergence = min(
        (value for value in divergence_steps.values() if value is not None),
        default=None,
    )
    l2_passed: bool | None = (
        False
        if action_delivery is False
        else None
        if not evidence_sufficient or action_delivery is None
        else earliest_divergence is None
    )
    l2 = {
        "verdict": (
            ContractVerdict.INSUFFICIENT_EVIDENCE.value
            if l2_passed is None
            else ContractVerdict.SUPPORTED.value
            if l2_passed is True
            else ContractVerdict.UNSUPPORTED.value
        ),
        "passed": l2_passed,
        "identical_actions": action_identity,
        "identical_requested_actions": action_identity,
        "preparation_action_delivery": preparation_action_delivery,
        "applied_action_delivery": action_delivery,
        "first_action_delivery_mismatch": first_action_delivery_mismatch,
        "first_numerical_divergence": first_numerical,
        "first_observation_divergence": first_observation,
        "first_contact_divergence": first_contact,
        "first_task_output_divergence": first_task_output,
        "first_termination_divergence": first_termination,
        "first_reward_divergence": first_reward,
        "first_semantic_event_divergence": first_semantic_event,
        "first_divergence": earliest_divergence,
        "first_divergence_by_channel": divergence_steps,
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
        "snapshot_sha256": _canonical_sha256(snapshot.to_dict()),
        "action_prefix_sha256": _prefix_sha256(actions),
        "trajectory_prefix_sha256": _prefix_sha256(
            [step.to_dict() for step in steps]
        ),
        "levels": {"L0": l0, "L1": l1, "L2": l2, "L3": l3},
        "captured_snapshot": snapshot.to_dict(),
        "identical_actions": actions,
    }


def _validate_horizon_reruns(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject horizon results built from different snapshots or action prefixes."""

    by_branch: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_branch[str(record["branch_id"])].append(record)
    for branch_id, group in by_branch.items():
        snapshots = {str(record["snapshot_sha256"]) for record in group}
        if len(snapshots) != 1:
            raise RuntimeError(
                f"branch {branch_id!r} produced different snapshots across horizon reruns"
            )
        longest = max(group, key=lambda item: int(item["horizon"]))
        reference_prefixes = longest["action_prefix_sha256"]
        reference_trajectory_prefixes = longest["trajectory_prefix_sha256"]
        for record in group:
            horizon = int(record["horizon"])
            prefixes = record["action_prefix_sha256"]
            if prefixes != reference_prefixes[:horizon]:
                raise RuntimeError(
                    f"branch {branch_id!r} produced different continuation action prefixes "
                    f"across horizon reruns"
                )
            trajectory_prefixes = record["trajectory_prefix_sha256"]
            if trajectory_prefixes != reference_trajectory_prefixes[:horizon]:
                raise RuntimeError(
                    f"branch {branch_id!r} produced different paired trajectory prefixes "
                    f"across horizon reruns; no coherent fidelity frontier can be claimed"
                )


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
        def earliest(
            field: str,
            level_values: Sequence[Mapping[str, Any]],
        ) -> int | None:
            values = [
                int(value[field])
                for value in level_values
                if value.get(field) is not None
            ]
            return min(values) if values else None

        first_divergence = earliest("first_divergence", l2_values)
        first_numerical = earliest("first_numerical_divergence", l2_values)
        decision_disagreement = any(not bool(value["agreement"]) for value in decisions)
        l2_passed = aggregate_passed([value["passed"] for value in l2_values])
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
                        "applied_action_delivery": (
                            aggregate_passed(
                                [value["applied_action_delivery"] for value in l2_values]
                            )
                            if enough
                            else None
                        ),
                        "first_action_delivery_mismatch": earliest(
                            "first_action_delivery_mismatch", l2_values
                        ),
                        "first_divergence": first_divergence,
                        "first_numerical_divergence": first_numerical,
                        "first_observation_divergence": earliest(
                            "first_observation_divergence", l2_values
                        ),
                        "first_contact_divergence": earliest(
                            "first_contact_divergence", l2_values
                        ),
                        "first_task_output_divergence": earliest(
                            "first_task_output_divergence", l2_values
                        ),
                        "first_termination_divergence": earliest(
                            "first_termination_divergence", l2_values
                        ),
                        "first_reward_divergence": earliest(
                            "first_reward_divergence", l2_values
                        ),
                        "first_semantic_event_divergence": earliest(
                            "first_semantic_event_divergence", l2_values
                        ),
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

    _validate_horizon_reruns(records)

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
        declared_version = config.simulator_version
        if declared_version in {"installed", "runtime"}:
            effective_config = replace(config, simulator_version=str(actual_version))
        elif str(actual_version) != declared_version:
            raise RuntimeError(
                "declared simulator_version does not match the live adapter: "
                f"declared {declared_version!r}, actual {actual_version!r}"
            )
        else:
            effective_config = config
        result = audit_configuration(selected_adapter, effective_config)
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
