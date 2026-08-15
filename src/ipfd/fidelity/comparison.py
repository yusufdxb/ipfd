"""Numerical and semantic comparison primitives for L0 through L3."""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from typing import Any

import numpy as np

__all__ = ["compare_values", "extract_env", "maximum_error"]


def extract_env(value: Any, env_index: int) -> Any:
    """Select one environment from every array leaf in a nested record."""

    if isinstance(value, Mapping):
        return {str(key): extract_env(item, env_index) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(extract_env(item, env_index) for item in value)
    if isinstance(value, list):
        return [extract_env(item, env_index) for item in value]
    array = _as_array(value)
    if array is None or array.ndim == 0:
        return value
    if not 0 <= env_index < array.shape[0]:
        raise IndexError(f"environment index {env_index} outside leading shape {array.shape}")
    return array[env_index]


def _as_array(value: Any) -> np.ndarray | None:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in "biufc":
        return None
    return array


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, name))
        return result
    return {prefix or "value": value}


def _numeric_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "comparable": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    if reference.size == 0:
        return {"comparable": True, "max_abs": 0.0, "rms": 0.0, "reference_scale": 0.0}
    ref = reference.astype(np.float64, copy=False)
    cand = candidate.astype(np.float64, copy=False)
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        return {"comparable": False, "reason": "nonfinite_value"}
    signed_difference = cand - ref
    difference = np.abs(signed_difference)
    maximum_index = int(np.argmax(difference))
    return {
        "comparable": True,
        "max_abs": float(np.max(difference)),
        "rms": float(sqrt(float(np.mean(np.square(difference))))),
        "reference_scale": float(np.max(np.abs(ref))),
        "signed_at_max": float(signed_difference.reshape(-1)[maximum_index]),
        "mean_signed": float(np.mean(signed_difference)),
        "reference_at_max": float(ref.reshape(-1)[maximum_index]),
        "candidate_at_max": float(cand.reshape(-1)[maximum_index]),
    }


def compare_values(
    reference: Any,
    candidate: Any,
    *,
    absolute: float,
    relative: float = 0.0,
    field_tolerances: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare nested values while retaining raw per-field error measurements."""

    if absolute < 0.0 or relative < 0.0:
        raise ValueError("tolerances must be non-negative")
    ref_fields = _flatten(reference)
    cand_fields = _flatten(candidate)
    ref_names = set(ref_fields)
    cand_names = set(cand_fields)
    field_results: dict[str, dict[str, Any]] = {}
    passed = ref_names == cand_names
    max_abs = 0.0
    for name in sorted(ref_names & cand_names):
        field_policy = (field_tolerances or {}).get(name, {})
        field_absolute = float(field_policy.get("absolute", absolute))
        field_relative = float(field_policy.get("relative", relative))
        field_unit = field_policy.get("unit")
        if (
            not np.isfinite(field_absolute)
            or not np.isfinite(field_relative)
            or field_absolute < 0.0
            or field_relative < 0.0
        ):
            raise ValueError(f"tolerances for field {name!r} must be non-negative")
        ref_array = _as_array(ref_fields[name])
        cand_array = _as_array(cand_fields[name])
        if ref_array is not None and cand_array is not None:
            if ref_array.dtype.kind == "b" or cand_array.dtype.kind == "b":
                same_shape = ref_array.shape == cand_array.shape
                equal = same_shape and bool(np.array_equal(ref_array, cand_array))
                differing = (
                    np.flatnonzero(ref_array.reshape(-1) != cand_array.reshape(-1))
                    if same_shape
                    else np.asarray([], dtype=int)
                )
                difference_index = int(differing[0]) if differing.size else None
                metrics = {
                    "comparable": same_shape,
                    "reason": None if same_shape else "shape_mismatch",
                    "reference_shape": list(ref_array.shape),
                    "candidate_shape": list(cand_array.shape),
                    "semantic_equal": equal,
                    "within_tolerance": equal,
                    "comparison": "exact_boolean",
                    "max_abs": 0.0 if equal else 1.0,
                    "unit": field_unit,
                    "reference_at_first_difference": (
                        bool(ref_array.reshape(-1)[difference_index])
                        if difference_index is not None
                        else None
                    ),
                    "candidate_at_first_difference": (
                        bool(cand_array.reshape(-1)[difference_index])
                        if difference_index is not None
                        else None
                    ),
                }
                max_abs = max(max_abs, float(metrics["max_abs"]))
                field_results[name] = metrics
                passed = passed and equal
                continue
            metrics = _numeric_metrics(ref_array, cand_array)
            if metrics.get("comparable"):
                threshold = field_absolute + field_relative * float(metrics["reference_scale"])
                metrics["threshold"] = threshold
                metrics["absolute_tolerance"] = field_absolute
                metrics["relative_tolerance"] = field_relative
                metrics["unit"] = field_unit
                metrics["within_tolerance"] = float(metrics["max_abs"]) <= threshold
                max_abs = max(max_abs, float(metrics["max_abs"]))
            else:
                metrics["within_tolerance"] = False
        else:
            equal = ref_fields[name] == cand_fields[name]
            if isinstance(equal, np.ndarray):
                equal = bool(np.all(equal))
            metrics = {
                "comparable": True,
                "semantic_equal": bool(equal),
                "within_tolerance": bool(equal),
            }
        field_results[name] = metrics
        passed = passed and bool(metrics["within_tolerance"])
    return {
        "passed": passed,
        "comparable_fields": len(ref_names & cand_names),
        "absolute_tolerance": absolute,
        "relative_tolerance": relative,
        "field_tolerances": {
            str(name): {
                "absolute": float(policy.get("absolute", absolute)),
                "relative": float(policy.get("relative", relative)),
                **({"unit": policy["unit"]} if isinstance(policy.get("unit"), str) else {}),
            }
            for name, policy in sorted((field_tolerances or {}).items())
        },
        "maximum_absolute_error": max_abs,
        "missing_in_candidate": sorted(ref_names - cand_names),
        "missing_in_reference": sorted(cand_names - ref_names),
        "fields": field_results,
    }


def maximum_error(comparison: Mapping[str, Any]) -> float:
    """Return the maximum raw numerical error from a comparison result."""

    return float(comparison.get("maximum_absolute_error", 0.0))
