"""Decision-relative analysis for restored counterfactual branches.

This module does not claim that a simulator snapshot is complete. It compares
the decision produced by an uninterrupted continuation with the decision from a
restored continuation under an explicitly named protocol.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import sqrt
from statistics import NormalDist
from typing import Any

__all__ = [
    "comparison_records",
    "summarize_branch_validity",
    "wilson_interval",
]


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _decision_match(comparison: Mapping[str, Any]) -> bool:
    if "recovery_verdict_match" in comparison:
        return bool(comparison["recovery_verdict_match"])
    return bool(comparison["success_match"])


def comparison_records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten a recovery-oracle report into paired branch comparisons."""
    records: list[dict[str, Any]] = []
    for run in report.get("runs", []):
        for checkpoint_name, checkpoint in run.get("checkpoints", {}).items():
            roundtrip = checkpoint.get("immediate_roundtrip", {})
            for continuation, comparison in checkpoint.get("comparisons", {}).items():
                records.append(
                    {
                        "seed": int(run["seed"]),
                        "repeat": int(run.get("repeat", 0)),
                        "disturbance": str(run.get("disturbance", "unknown")),
                        "checkpoint": str(checkpoint_name),
                        "phase": str(checkpoint.get("task_phase", "unknown")),
                        "continuation": str(continuation),
                        "decision_match": _decision_match(comparison),
                        "reference_success": bool(comparison["reference_success"]),
                        "candidate_success": bool(comparison["candidate_success"]),
                        "first_material_divergence_step": comparison.get(
                            "first_material_divergence_step"
                        ),
                        "object_pose_max_abs_30": float(
                            comparison.get("object_pose_max_abs_30", 0.0)
                        ),
                        "immediate_obs_max_abs": float(
                            roundtrip.get(
                                "exact_obs_max_abs"
                                if continuation == "exact_action"
                                else "policy_obs_max_abs",
                                float("nan"),
                            )
                        ),
                    }
                )
    return records


def _group_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(records)
    total = len(values)
    if total == 0:
        return {
            "comparisons": 0,
            "agreements": 0,
            "disagreements": 0,
            "agreement_rate": None,
            "wilson_95": None,
        }
    agreements = sum(bool(record["decision_match"]) for record in values)
    lower, upper = wilson_interval(agreements, total)
    false_recoverable = sum(
        not bool(record["reference_success"]) and bool(record["candidate_success"])
        for record in values
    )
    false_unrecoverable = sum(
        bool(record["reference_success"]) and not bool(record["candidate_success"])
        for record in values
    )
    return {
        "comparisons": total,
        "agreements": agreements,
        "disagreements": total - agreements,
        "agreement_rate": agreements / total,
        "wilson_95": [lower, upper],
        "false_recoverable": false_recoverable,
        "false_unrecoverable": false_unrecoverable,
    }


def _summaries_by(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    return {name: _group_summary(group) for name, group in sorted(groups.items())}


def _summaries_by_phase_and_continuation(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["phase"], record["continuation"])].append(record)
    nested: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (phase, continuation), group in sorted(groups.items()):
        nested[phase][continuation] = _group_summary(group)
    return dict(nested)


def _validity_gate(
    records: list[dict[str, Any]],
    *,
    maximum_disagreement_rate: float,
    confidence: float,
) -> dict[str, Any]:
    """Apply a fail-closed empirical gate to phase and continuation strata.

    The bound is comparison-level and is not a publication-grade certificate
    when branches within a seed are correlated. That limitation is carried in
    the returned result.
    """
    if not 0.0 <= maximum_disagreement_rate < 1.0:
        raise ValueError("maximum_disagreement_rate must be in [0, 1)")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")

    z = NormalDist().inv_cdf(confidence)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["phase"], record["continuation"])].append(record)

    cells: list[dict[str, Any]] = []
    for (phase, continuation), group in sorted(groups.items()):
        total = len(group)
        disagreements = sum(not bool(record["decision_match"]) for record in group)
        proportion = disagreements / total
        denominator = 1.0 + z * z / total
        center = (proportion + z * z / (2.0 * total)) / denominator
        radius = (
            z
            * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
            / denominator
        )
        disagreement_upper = min(1.0, center + radius)
        cells.append(
            {
                "phase": phase,
                "continuation": continuation,
                "comparisons": total,
                "disagreements": disagreements,
                "disagreement_rate": proportion,
                "one_sided_upper_bound": disagreement_upper,
                "passes": disagreement_upper <= maximum_disagreement_rate,
            }
        )

    return {
        "maximum_disagreement_rate": maximum_disagreement_rate,
        "confidence": confidence,
        "cells": cells,
        "passing_cells": sum(bool(cell["passes"]) for cell in cells),
        "fail_closed": True,
        "statistical_limitation": (
            "The bound treats paired comparisons as independent. Repeated checkpoints "
            "within a seed are correlated, so this audit can falsify universal validity "
            "but cannot certify deployment validity."
        ),
    }


def summarize_branch_validity(
    report: Mapping[str, Any],
    *,
    source_label: str,
    source_sha256: str,
    maximum_disagreement_rate: float = 0.05,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize one paired uninterrupted-versus-restored branch experiment."""
    records = comparison_records(report)
    if not records:
        raise ValueError("report contains no paired branch comparisons")

    exact_records = [record for record in records if record["continuation"] == "exact_action"]
    material_exact = sum(
        record["first_material_divergence_step"] is not None for record in exact_records
    )
    finite_roundtrips = [
        record["immediate_obs_max_abs"]
        for record in records
        if record["immediate_obs_max_abs"] == record["immediate_obs_max_abs"]
    ]
    mismatches = [record for record in records if not record["decision_match"]]

    return {
        "source": {
            "label": source_label,
            "sha256": source_sha256,
            "input_schema_version": report.get("schema_version"),
            "input_classification": report.get("classification"),
        },
        "result": (
            "FALSIFIED_UNIVERSAL_DECISION_FIDELITY"
            if mismatches
            else "NO_DISAGREEMENT_OBSERVED_NOT_CERTIFIED"
        ),
        "overall": _group_summary(records),
        "by_continuation": _summaries_by(records, "continuation"),
        "by_phase": _summaries_by(records, "phase"),
        "by_seed": _summaries_by(records, "seed"),
        "by_phase_and_continuation": _summaries_by_phase_and_continuation(records),
        "trajectory_checks": {
            "exact_action_comparisons": len(exact_records),
            "exact_action_material_divergences": material_exact,
            "maximum_exact_action_object_pose_error_m": max(
                (record["object_pose_max_abs_30"] for record in exact_records),
                default=None,
            ),
            "full_restore_immediate_observation_checks": len(finite_roundtrips),
            "full_restore_immediate_observation_exact": sum(
                value == 0.0 for value in finite_roundtrips
            ),
        },
        "mismatch_examples": mismatches[:12],
        "validity_gate": _validity_gate(
            records,
            maximum_disagreement_rate=maximum_disagreement_rate,
            confidence=confidence,
        ),
        "interpretation_limit": (
            "A mismatch rejects universal decision fidelity for this protocol. Agreement "
            "does not prove physical recoverability, complete state restoration, or "
            "validity outside the sampled task, policy, phase, and continuation."
        ),
    }
