"""Executable conformance checks for live replay adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from .audit import DecisionContract, _provenance_errors
from .fidelity.comparison import compare_values, extract_env
from .fidelity.contracts import (
    ContractVerdict,
    ObservationRecord,
    ReplayAdapter,
    Snapshot,
    StepRecord,
    TrajectoryRecord,
    to_builtin,
)

__all__ = [
    "AdapterCheck",
    "AdapterCheckReport",
    "AdapterCheckStatus",
    "check_adapter",
]


class AdapterCheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class AdapterCheck:
    name: str
    status: AdapterCheckStatus
    detail: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "evidence": to_builtin(self.evidence),
        }


@dataclass(frozen=True)
class AdapterCheckReport:
    checks: tuple[AdapterCheck, ...]

    @property
    def verdict(self) -> ContractVerdict:
        if any(item.status is AdapterCheckStatus.FAIL for item in self.checks):
            return ContractVerdict.UNSUPPORTED
        if any(
            item.status is AdapterCheckStatus.INSUFFICIENT_EVIDENCE
            for item in self.checks
        ):
            return ContractVerdict.INSUFFICIENT_EVIDENCE
        return ContractVerdict.SUPPORTED

    @property
    def passed(self) -> bool:
        return self.verdict is ContractVerdict.SUPPORTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "verdict": self.verdict.value,
            "passed": self.passed,
            "checks": [item.to_dict() for item in self.checks],
        }


def _fingerprint(value: Any) -> str:
    return json.dumps(
        to_builtin(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _snapshot(adapter: ReplayAdapter, env_ids: Sequence[int]) -> Snapshot:
    value = adapter.capture(env_ids)
    if not isinstance(value, Snapshot):
        raise TypeError("capture() must return Snapshot")
    return value


def _observation(adapter: ReplayAdapter, env_ids: Sequence[int]) -> ObservationRecord:
    value = adapter.observe(env_ids)
    if not isinstance(value, ObservationRecord):
        raise TypeError("observe() must return ObservationRecord")
    return value


def _actions_identical(actions: Any) -> bool:
    try:
        array = np.asarray(actions)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim >= 1 and array.shape[0] >= 2 and np.array_equal(array[0], array[1]))


def _counter_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        result: dict[str, np.ndarray] = {}
        for name, item in value.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            result.update(_counter_arrays(item, path))
        return result
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return {}
    if array.dtype.kind not in "biuf" or array.size == 0:
        return {}
    return {prefix or "value": array.astype(np.float64, copy=False)}


def _evaluate_decision(
    adapter: ReplayAdapter,
    decision: str | DecisionContract,
    record: TrajectoryRecord,
) -> bool:
    value = (
        decision.evaluate(record)
        if isinstance(decision, DecisionContract)
        else adapter.decision(record, decision)
    )
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"decision must return bool, got {type(value).__name__}")
    return bool(value)


def check_adapter(
    adapter: ReplayAdapter,
    *,
    decision: str | DecisionContract,
    seed: int = 0,
    action_source: str = "recorded",
) -> AdapterCheckReport:
    """Exercise the paired adapter contract without claiming simulator completeness.

    The checks mutate and reset the supplied adapter.  They do not close it.  A
    pass is scoped to the sampled seed, action, and one-step decision trajectory.
    """

    checks: list[AdapterCheck] = []

    def add(
        name: str,
        status: AdapterCheckStatus,
        detail: str,
        **evidence: Any,
    ) -> None:
        checks.append(AdapterCheck(name, status, detail, evidence))

    required_methods = (
        "reset",
        "action",
        "capture",
        "restore",
        "observe",
        "step",
        "decision",
        "provenance",
    )
    missing = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    if missing:
        add(
            "required_interface",
            AdapterCheckStatus.FAIL,
            "the live runner cannot use this adapter",
            missing_methods=missing,
        )
        return AdapterCheckReport(tuple(checks))
    add(
        "required_interface",
        AdapterCheckStatus.PASS,
        "all methods used by the live runner are callable",
        methods=list(required_methods),
    )

    try:
        provenance = adapter.provenance()
        errors = _provenance_errors(provenance)
    except Exception as exc:  # adapter boundary: preserve the concrete exception
        add(
            "provenance_completeness",
            AdapterCheckStatus.FAIL,
            "provenance() raised",
            exception=f"{type(exc).__name__}: {exc}",
        )
    else:
        add(
            "provenance_completeness",
            AdapterCheckStatus.PASS if not errors else AdapterCheckStatus.FAIL,
            "required capability disclosures are present" if not errors else "required capability disclosures are incomplete",
            errors=errors,
        )

    try:
        adapter.reset(seed)
        first = _snapshot(adapter, (0,))
        second = _snapshot(adapter, (0,))
        deterministic = _fingerprint(first.to_dict()) == _fingerprint(second.to_dict())
        explicit_inventory = not isinstance(first.unavailable_components, (str, bytes)) and isinstance(
            first.unavailable_components, Sequence
        )
    except Exception as exc:
        add(
            "deterministic_repeated_capture",
            AdapterCheckStatus.FAIL,
            "repeated capture raised",
            exception=f"{type(exc).__name__}: {exc}",
        )
        add(
            "explicit_unavailable_state",
            AdapterCheckStatus.INSUFFICIENT_EVIDENCE,
            "no valid snapshot inventory was available",
        )
    else:
        add(
            "deterministic_repeated_capture",
            AdapterCheckStatus.PASS if deterministic else AdapterCheckStatus.FAIL,
            (
                "unchanged simulator state produced identical snapshots"
                if deterministic
                else "unchanged simulator state produced different snapshots"
            ),
        )
        add(
            "explicit_unavailable_state",
            AdapterCheckStatus.PASS if explicit_inventory else AdapterCheckStatus.FAIL,
            (
                "the snapshot carries an explicit unavailable-component sequence"
                if explicit_inventory
                else "unavailable_components is not an explicit sequence"
            ),
            unavailable_components=list(first.unavailable_components) if explicit_inventory else None,
        )

    try:
        adapter.reset(seed)
        captured = _snapshot(adapter, (0,))
        before = _fingerprint(captured.to_dict())
        actions = adapter.action(0, action_source, (0, 1))
        identical = _actions_identical(actions)
        step = adapter.step(actions)
        if not isinstance(step, StepRecord):
            raise TypeError("step() must return StepRecord")
        by_value = before == _fingerprint(captured.to_dict())
    except Exception as exc:
        add(
            "capture_by_value",
            AdapterCheckStatus.FAIL,
            "capture-by-value exercise raised",
            exception=f"{type(exc).__name__}: {exc}",
        )
        add(
            "identical_paired_action_request",
            AdapterCheckStatus.INSUFFICIENT_EVIDENCE,
            "the paired action and step exercise did not complete",
        )
        add(
            "identical_action_delivery",
            AdapterCheckStatus.INSUFFICIENT_EVIDENCE,
            "the paired action and step exercise did not complete",
        )
    else:
        add(
            "capture_by_value",
            AdapterCheckStatus.PASS if by_value else AdapterCheckStatus.FAIL,
            (
                "a later simulator step did not mutate the captured snapshot"
                if by_value
                else "the captured snapshot changed after a simulator step"
            ),
        )
        add(
            "identical_paired_action_request",
            AdapterCheckStatus.PASS if identical else AdapterCheckStatus.FAIL,
            (
                "the runner generated one identical requested action row per branch"
                if identical
                else "the adapter generated different action values for the paired branches"
            ),
        )
        applied = step.applied_actions
        if applied is None:
            delivery_status = AdapterCheckStatus.INSUFFICIENT_EVIDENCE
            delivery_detail = "step() did not disclose the actions actually applied"
            delivery_equal = None
        else:
            delivery_equal = bool(
                _actions_identical(applied)
                and _fingerprint(applied) == _fingerprint(actions)
            )
            delivery_status = (
                AdapterCheckStatus.PASS if delivery_equal else AdapterCheckStatus.FAIL
            )
            delivery_detail = (
                "step() recorded the exact requested action for both branches"
                if delivery_equal
                else "step() recorded applied actions that differ from the paired request"
            )
        add(
            "identical_action_delivery",
            delivery_status,
            delivery_detail,
            requested_actions=actions,
            applied_actions=applied,
        )

    try:
        adapter.reset(seed)
        compatible = _snapshot(adapter, (0,))
        incompatible = replace(compatible, protocol=f"{compatible.protocol}__ipfd_incompatible__")
        rejected = False
        try:
            adapter.restore(incompatible, (1,))
        except Exception:
            rejected = True
    except Exception as exc:
        add(
            "protocol_mismatch_rejection",
            AdapterCheckStatus.FAIL,
            "the mismatch exercise could not run",
            exception=f"{type(exc).__name__}: {exc}",
        )
    else:
        add(
            "protocol_mismatch_rejection",
            AdapterCheckStatus.PASS if rejected else AdapterCheckStatus.FAIL,
            (
                "restore rejected a snapshot with an incompatible protocol"
                if rejected
                else "restore accepted a snapshot with an incompatible protocol"
            ),
        )

    try:
        adapter.reset(seed)
        source = _snapshot(adapter, (0,))
        adapter.restore(source, (1,))
        observed = _observation(adapter, (0, 1))
        comparisons = {
            category: compare_values(
                extract_env(values, 0),
                extract_env(values, 1),
                absolute=0.0,
            )
            for category, values in observed.categories().items()
        }
        comparable = sum(int(item["comparable_fields"]) for item in comparisons.values())
        required_comparable = bool(
            int(comparisons["scene_state"]["comparable_fields"]) > 0
            and int(comparisons["policy_observations"]["comparable_fields"]) > 0
        )
        restore_equal = required_comparable and all(
            bool(item["passed"]) for item in comparisons.values()
        )
    except Exception as exc:
        add(
            "compatible_restore_boundary",
            AdapterCheckStatus.FAIL,
            "a declared-compatible restore raised",
            exception=f"{type(exc).__name__}: {exc}",
        )
    else:
        status = (
            AdapterCheckStatus.PASS
            if restore_equal
            else AdapterCheckStatus.FAIL
            if required_comparable
            else AdapterCheckStatus.INSUFFICIENT_EVIDENCE
        )
        add(
            "compatible_restore_boundary",
            status,
            (
                "available exposed state matches exactly after restore"
                if restore_equal
                else "available exposed state did not establish an exact restore-boundary match"
            ),
            comparable_fields=comparable,
            scene_state_fields=int(
                comparisons["scene_state"]["comparable_fields"]
            ),
            policy_observation_fields=int(
                comparisons["policy_observations"]["comparable_fields"]
            ),
        )

    try:
        adapter.reset(seed)
        before_observe = _snapshot(adapter, (0, 1))
        first_observation = _observation(adapter, (0, 1))
        second_observation = _observation(adapter, (0, 1))
        after_observe = _snapshot(adapter, (0, 1))
        repeated_equal = _fingerprint(first_observation.to_dict()) == _fingerprint(
            second_observation.to_dict()
        )
        capture_equal = _fingerprint(before_observe.to_dict()) == _fingerprint(after_observe.to_dict())
        side_effect_free = repeated_equal and capture_equal
    except Exception as exc:
        add(
            "observe_side_effects",
            AdapterCheckStatus.FAIL,
            "the observation side-effect exercise raised",
            exception=f"{type(exc).__name__}: {exc}",
        )
    else:
        add(
            "observe_side_effects",
            AdapterCheckStatus.PASS if side_effect_free else AdapterCheckStatus.FAIL,
            (
                "repeated observe calls were stable and did not change captured state"
                if side_effect_free
                else "observe changed its result or subsequently captured state"
            ),
            repeated_observation_equal=repeated_equal,
            captured_state_equal=capture_equal,
        )

    try:
        adapter.reset(seed)
        source = _snapshot(adapter, (0,))
        adapter.restore(source, (1,))
        before_step = _observation(adapter, (0, 1))
        actions = adapter.action(0, action_source, (0, 1))
        if not _actions_identical(actions):
            raise RuntimeError("paired actions are not identical")
        step_record = adapter.step(actions)
        if not isinstance(step_record, StepRecord):
            raise TypeError("step() must return StepRecord")
        before_counters = _counter_arrays(before_step.counters)
        after_counters = _counter_arrays(step_record.observation.counters)
        common = sorted(set(before_counters) & set(after_counters))
        nondecreasing = True
        advanced = False
        for name in common:
            before_value = before_counters[name]
            after_value = after_counters[name]
            if before_value.shape != after_value.shape:
                nondecreasing = False
                continue
            delta = after_value - before_value
            nondecreasing = nondecreasing and bool(np.all(delta >= 0.0))
            advanced = advanced or bool(np.any(delta > 0.0))
        record = TrajectoryRecord(
            steps=[step_record],
            actions=[to_builtin(actions)],
            env_id=0,
        )
        first_decision = _evaluate_decision(adapter, decision, record)
        second_decision = _evaluate_decision(adapter, decision, record)
    except Exception as exc:
        add(
            "automatic_reset_not_observed",
            AdapterCheckStatus.INSUFFICIENT_EVIDENCE,
            "the one-step counter exercise did not complete",
            exception=f"{type(exc).__name__}: {exc}",
        )
        add(
            "deterministic_semantic_decision",
            AdapterCheckStatus.FAIL,
            "the semantic decision exercise raised",
            exception=f"{type(exc).__name__}: {exc}",
        )
    else:
        counter_evidence = bool(common) and nondecreasing and advanced
        add(
            "automatic_reset_not_observed",
            AdapterCheckStatus.PASS if counter_evidence else AdapterCheckStatus.INSUFFICIENT_EVIDENCE,
            (
                "at least one exposed counter advanced and none decreased during the sampled step"
                if counter_evidence
                else "exposed counters could not rule out an automatic reset in the sampled step"
            ),
            counter_paths=common,
            sampled_steps=1,
        )
        deterministic_decision = first_decision == second_decision
        add(
            "deterministic_semantic_decision",
            AdapterCheckStatus.PASS if deterministic_decision else AdapterCheckStatus.FAIL,
            (
                "the decision returned the same Boolean result twice for one immutable trajectory"
                if deterministic_decision
                else "the decision changed for the same trajectory"
            ),
            result=first_decision,
        )

    return AdapterCheckReport(tuple(checks))
