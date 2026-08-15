"""Automatic reduction for finite-horizon replay failures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["minimize_failure"]


def minimize_failure(
    initial: Mapping[str, Any],
    evaluate: Callable[[dict[str, Any]], Mapping[str, Any]],
    *,
    branch_steps: Sequence[int] = (),
    decision_names: Sequence[str] = (),
    adapter_dimensions: Mapping[str, Sequence[Any]] | None = None,
    max_trials: int = 100,
) -> dict[str, Any]:
    """Reduce a failing replay case while preserving its declared failure.

    ``evaluate`` must return a mapping with ``fails`` and may include the full
    rerun record. Horizon and action prefix are always reduced together because
    open-loop replay consumes exactly one recorded action per control step.
    Adapter-specific dimensions are attempted only when the adapter advertises
    concrete candidates.
    """

    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")
    current = dict(initial)
    if not bool(current.get("fails")):
        raise ValueError("initial case must be failing")
    trials: list[dict[str, Any]] = []
    trial_count = 0

    def attempt(candidate: dict[str, Any], dimension: str) -> bool:
        nonlocal current, trial_count
        if trial_count >= max_trials:
            return False
        result = dict(evaluate(candidate))
        trial_count += 1
        preserved = bool(result.get("fails"))
        trials.append(
            {
                "dimension": dimension,
                "candidate": _small_candidate(candidate),
                "failure_preserved": preserved,
                "result": result,
            }
        )
        if preserved:
            current = {**candidate, **result, "fails": True}
        return preserved

    original_horizon = int(current["horizon"])
    for horizon in range(1, original_horizon):
        candidate = {**current, "horizon": horizon, "action_sequence_length": horizon}
        if attempt(candidate, "continuation_horizon_and_action_prefix"):
            break
        if trial_count >= max_trials:
            break

    for branch_step in sorted({int(value) for value in branch_steps if 0 <= int(value) < int(current["branch_step"])}):
        candidate = {**current, "branch_step": branch_step}
        if attempt(candidate, "branch_timing"):
            break
        if trial_count >= max_trials:
            break

    dimensions = dict(adapter_dimensions or {})
    attempted_dimensions = {
        "number_of_active_entities": "active_entities",
        "disturbance_schedule": "disturbance_schedule",
        "snapshot_state_components": "snapshot_components",
    }
    unsupported: list[dict[str, str]] = []
    if decision_names:
        for decision_name in decision_names:
            if decision_name == current.get("decision_name"):
                continue
            candidate = {**current, "decision_name": str(decision_name)}
            if attempt(candidate, "decision_predicate"):
                break
            if trial_count >= max_trials:
                break
    else:
        unsupported.append(
            {
                "dimension": "decision_predicate",
                "status": "NOT_APPLICABLE_TO_NUMERICAL_FAILURE",
            }
        )
    for report_name, candidate_name in attempted_dimensions.items():
        candidates = dimensions.get(candidate_name, ())
        if not candidates:
            unsupported.append(
                {
                    "dimension": report_name,
                    "status": "NOT_SUPPORTED_BY_ADAPTER",
                }
            )
            continue
        for value in candidates:
            candidate = {**current, candidate_name: value}
            if attempt(candidate, report_name):
                break
            if trial_count >= max_trials:
                break

    return {
        "schema_version": 1,
        "algorithm": "bounded_prefix_and_dimension_reduction",
        "initial": _small_candidate(initial),
        "minimal": _small_candidate(current),
        "minimal_result": current.get("record"),
        "minimal_evidence": {
            key: current.get(key)
            for key in ("captured_snapshot", "identical_actions", "adapter_provenance")
            if key in current
        },
        "trials": trials,
        "attempts_not_supported": unsupported,
        "trial_count": trial_count,
        "max_trials": max_trials,
        "budget_exhausted": trial_count >= max_trials,
    }


def _small_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "branch_id",
        "branch_step",
        "seed",
        "cluster",
        "horizon",
        "action_sequence_length",
        "decision_name",
        "snapshot_components",
        "active_entities",
        "disturbance_schedule",
        "failure_kind",
        "fails",
    )
    return {field: value[field] for field in fields if field in value}
