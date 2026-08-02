"""Simulator-neutral comparison of replay-fidelity audit summaries.

The comparator intentionally consumes plain mappings.  It does not import an
adapter or the audit runner, so saved summaries remain comparable when either
implementation changes.

Each audit must contain a ``configurations`` list.  Every entry has this shape::

    {
        "scope": {"simulator": "...", "task": "...", ...},
        "result": "SUPPORTED" | "UNSUPPORTED" | "INSUFFICIENT_EVIDENCE",
        "levels": {
            "L0": {"passed": true | false | null},
            "L1": {"passed": true | false | null},
            "L2": {"first_numerical_divergence": 12 | null},
            "L3": {"decision_disagreement": true | false | null},
        },
    }

``null`` means that the audit did not contain enough evidence for that field.
Scopes are matched by their complete JSON value unless both entries declare the
same optional ``comparison_key``. The explicit key allows version or protocol
comparisons whose raw scopes necessarily differ. The original scope mappings
are copied into the report so simulator, version, environment, and other labels
are not collapsed into a synthetic identifier.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = ["compare_audits", "compare_audit_files"]

_CONTRACT_RESULTS = frozenset(
    {"SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"}
)
_LEVEL_NAMES = ("L0", "L1", "L2", "L3")


def compare_audits(baseline: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, Any]:
    """Compare two audit summaries and return a scoped regression report.

    Added and removed scopes are reported, but only scopes present in both
    audits participate in level and decision comparisons.  A missing candidate
    scope is therefore not silently treated as ``UNSUPPORTED``.
    """

    baseline_entries = _validate_audit(baseline, "baseline")
    candidate_entries = _validate_audit(candidate, "candidate")
    baseline_by_scope = {
        _json_key(entry["comparison_key"]): entry for entry in baseline_entries
    }
    candidate_by_scope = {
        _json_key(entry["comparison_key"]): entry for entry in candidate_entries
    }

    comparisons: list[dict[str, Any]] = []
    supported_regressions: list[dict[str, Any]] = []
    for scope_key, baseline_entry in baseline_by_scope.items():
        candidate_entry = candidate_by_scope.get(scope_key)
        if candidate_entry is None:
            continue

        baseline_levels = baseline_entry["levels"]
        candidate_levels = candidate_entry["levels"]
        l0_change = _compare_optional_values(
            baseline_levels["L0"]["passed"], candidate_levels["L0"]["passed"]
        )
        l1_change = _compare_optional_values(
            baseline_levels["L1"]["passed"], candidate_levels["L1"]["passed"]
        )
        decision_change = _compare_optional_values(
            baseline_levels["L3"]["decision_disagreement"],
            candidate_levels["L3"]["decision_disagreement"],
        )
        baseline_divergence = baseline_levels["L2"]["first_numerical_divergence"]
        candidate_divergence = candidate_levels["L2"]["first_numerical_divergence"]
        divergence_change = _compare_divergence(
            baseline_divergence,
            candidate_divergence,
            baseline_levels["L2"].get("passed"),
            candidate_levels["L2"].get("passed"),
        )
        became_unsupported = (
            baseline_entry["result"] == "SUPPORTED"
            and candidate_entry["result"] == "UNSUPPORTED"
        )

        comparison = {
            "comparison_key": copy.deepcopy(baseline_entry["comparison_key"]),
            "baseline_scope": copy.deepcopy(baseline_entry["scope"]),
            "candidate_scope": copy.deepcopy(candidate_entry["scope"]),
            "scope_changes": _scope_changes(
                baseline_entry["scope"], candidate_entry["scope"]
            ),
            "baseline_result": baseline_entry["result"],
            "candidate_result": candidate_entry["result"],
            "l0": {
                "baseline_passed": baseline_levels["L0"]["passed"],
                "candidate_passed": candidate_levels["L0"]["passed"],
                "changed": l0_change,
            },
            "l1": {
                "baseline_passed": baseline_levels["L1"]["passed"],
                "candidate_passed": candidate_levels["L1"]["passed"],
                "changed": l1_change,
            },
            "first_divergence": {
                "baseline_step": baseline_divergence,
                "candidate_step": candidate_divergence,
                "baseline_no_divergence_observed": (
                    baseline_divergence is None and baseline_levels["L2"].get("passed") is True
                ),
                "candidate_no_divergence_observed": (
                    candidate_divergence is None and candidate_levels["L2"].get("passed") is True
                ),
                "change": divergence_change,
            },
            "decision_disagreement": {
                "baseline": baseline_levels["L3"]["decision_disagreement"],
                "candidate": candidate_levels["L3"]["decision_disagreement"],
                "changed": decision_change,
            },
            "previously_supported_became_unsupported": became_unsupported,
        }
        comparisons.append(comparison)
        if became_unsupported:
            supported_regressions.append(
                {
                    "comparison_key": copy.deepcopy(baseline_entry["comparison_key"]),
                    "baseline_scope": copy.deepcopy(baseline_entry["scope"]),
                    "candidate_scope": copy.deepcopy(candidate_entry["scope"]),
                }
            )

    added_scopes = [
        copy.deepcopy(entry["scope"])
        for key, entry in candidate_by_scope.items()
        if key not in baseline_by_scope
    ]
    removed_scopes = [
        copy.deepcopy(entry["scope"])
        for key, entry in baseline_by_scope.items()
        if key not in candidate_by_scope
    ]
    divergence_counts = {name: 0 for name in ("earlier", "later", "same", "unavailable")}
    for comparison in comparisons:
        divergence_counts[comparison["first_divergence"]["change"]] += 1

    return {
        "schema_version": 1,
        "summary": {
            "matched_configurations": len(comparisons),
            "added_configurations": len(added_scopes),
            "removed_configurations": len(removed_scopes),
            "l0_changed": _aggregate_changes(item["l0"]["changed"] for item in comparisons),
            "l1_changed": _aggregate_changes(item["l1"]["changed"] for item in comparisons),
            "first_divergence_change_counts": divergence_counts,
            "decision_disagreement_changed": _aggregate_changes(
                item["decision_disagreement"]["changed"] for item in comparisons
            ),
            "previously_supported_became_unsupported": bool(supported_regressions),
        },
        "comparisons": comparisons,
        "supported_to_unsupported": supported_regressions,
        "added_scopes": added_scopes,
        "removed_scopes": removed_scopes,
    }


def compare_audit_files(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Load two JSON audit summaries and pass them to :func:`compare_audits`."""

    baseline = _read_audit_file(Path(baseline_path), "baseline")
    candidate = _read_audit_file(Path(candidate_path), "candidate")
    return compare_audits(baseline, candidate)


def _read_audit_file(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {label} audit file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} audit file {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} audit file {path} must contain a JSON object")
    return payload


def _validate_audit(audit: Mapping[str, object], label: str) -> list[dict[str, Any]]:
    if not isinstance(audit, Mapping):
        raise ValueError(f"{label} audit must be a mapping")
    if audit.get("schema_version") != 1:
        raise ValueError(f"{label} audit schema_version must be 1")
    raw_entries = audit.get("configurations")
    if not isinstance(raw_entries, list):
        raise ValueError(f"{label} audit configurations must be a list")

    entries: list[dict[str, Any]] = []
    scope_keys: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        path = f"{label}.configurations[{index}]"
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{path} must be a mapping")
        scope = raw_entry.get("scope")
        if not isinstance(scope, Mapping) or not scope:
            raise ValueError(f"{path}.scope must be a nonempty mapping")
        if any(not isinstance(key, str) for key in scope):
            raise ValueError(f"{path}.scope keys must be strings")
        _json_key(scope, path=f"{path}.scope")
        comparison_key = raw_entry.get("comparison_key", scope)
        if comparison_key is None or comparison_key == "":
            raise ValueError(f"{path}.comparison_key must not be null or empty")
        scope_key = _json_key(comparison_key, path=f"{path}.comparison_key")
        if scope_key in scope_keys:
            raise ValueError(f"{label} audit contains a duplicate comparison key at index {index}")
        scope_keys.add(scope_key)

        result = raw_entry.get("result")
        if result not in _CONTRACT_RESULTS:
            raise ValueError(
                f"{path}.result must be one of {sorted(_CONTRACT_RESULTS)}, got {result!r}"
            )
        levels = raw_entry.get("levels")
        if not isinstance(levels, Mapping):
            raise ValueError(f"{path}.levels must be a mapping")
        missing_levels = set(_LEVEL_NAMES) - set(levels)
        if missing_levels:
            raise ValueError(f"{path}.levels is missing {sorted(missing_levels)}")

        normalized_levels: dict[str, dict[str, object]] = {}
        for level_name in ("L0", "L1"):
            level = _level_mapping(levels[level_name], f"{path}.levels.{level_name}")
            passed = level.get("passed")
            if passed is not None and not isinstance(passed, bool):
                raise ValueError(f"{path}.levels.{level_name}.passed must be bool or null")
            if "passed" not in level:
                raise ValueError(f"{path}.levels.{level_name}.passed is required")
            normalized_levels[level_name] = {"passed": passed}

        l2 = _level_mapping(levels["L2"], f"{path}.levels.L2")
        if "first_numerical_divergence" not in l2:
            raise ValueError(f"{path}.levels.L2.first_numerical_divergence is required")
        divergence = l2["first_numerical_divergence"]
        if divergence is not None and (
            isinstance(divergence, bool) or not isinstance(divergence, int) or divergence < 0
        ):
            raise ValueError(
                f"{path}.levels.L2.first_numerical_divergence must be a nonnegative integer or null"
            )
        l2_passed = l2.get("passed")
        if l2_passed is not None and not isinstance(l2_passed, bool):
            raise ValueError(f"{path}.levels.L2.passed must be bool or null when provided")
        normalized_levels["L2"] = {
            "first_numerical_divergence": divergence,
            "passed": l2_passed,
        }

        l3 = _level_mapping(levels["L3"], f"{path}.levels.L3")
        if "decision_disagreement" not in l3:
            raise ValueError(f"{path}.levels.L3.decision_disagreement is required")
        disagreement = l3["decision_disagreement"]
        if disagreement is not None and not isinstance(disagreement, bool):
            raise ValueError(f"{path}.levels.L3.decision_disagreement must be bool or null")
        normalized_levels["L3"] = {"decision_disagreement": disagreement}

        entries.append(
            {
                "scope": copy.deepcopy(dict(scope)),
                "comparison_key": copy.deepcopy(comparison_key),
                "result": result,
                "levels": normalized_levels,
            }
        )
    return entries


def _level_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _json_key(value: object, *, path: str = "comparison key") -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain only finite JSON values: {exc}") from exc


def _compare_optional_values(baseline: object, candidate: object) -> bool | None:
    if baseline is None or candidate is None:
        return None
    return baseline != candidate


def _scope_changes(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for field in sorted(set(baseline) | set(candidate)):
        baseline_present = field in baseline
        candidate_present = field in candidate
        baseline_value = baseline.get(field)
        candidate_value = candidate.get(field)
        if baseline_present == candidate_present and baseline_value == candidate_value:
            continue
        changes.append(
            {
                "field": field,
                "baseline_present": baseline_present,
                "candidate_present": candidate_present,
                "baseline": copy.deepcopy(baseline_value),
                "candidate": copy.deepcopy(candidate_value),
            }
        )
    return changes


def _compare_divergence(
    baseline: int | None,
    candidate: int | None,
    baseline_passed: object = None,
    candidate_passed: object = None,
) -> str:
    if baseline is None and candidate is None:
        return "same" if baseline_passed is True and candidate_passed is True else "unavailable"
    if baseline is not None and candidate is None:
        return "later" if candidate_passed is True else "unavailable"
    if baseline is None and candidate is not None:
        return "earlier" if baseline_passed is True else "unavailable"
    assert baseline is not None and candidate is not None
    if candidate < baseline:
        return "earlier"
    if candidate > baseline:
        return "later"
    return "same"


def _aggregate_changes(changes: Iterable[bool | None]) -> bool | None:
    values = list(changes)
    if any(value is True for value in values):
        return True
    if values and all(value is False for value in values):
        return False
    return None
