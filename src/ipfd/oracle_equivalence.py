"""Pure analysis helpers for recovery-oracle equivalence experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["compare_traces", "classify_oracle", "terminal_outcome"]


def terminal_outcome(
    heights: Sequence[float],
    *,
    rest_height: float,
    lift_threshold: float,
    terminated: bool = False,
) -> tuple[bool, int]:
    """Return final lift success and the start of its terminal success suffix.

    A transient threshold crossing is not recovery. A run succeeds only when its
    final observed state is lifted and it did not terminate. The outcome step is
    the first sample in the final contiguous lifted suffix. Failed runs report
    their horizon as the outcome step.
    """
    values = np.asarray(heights, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("heights must contain at least one sample")
    lifted = values > float(rest_height) + float(lift_threshold)
    success = bool(lifted[-1]) and not terminated
    if not success:
        return False, int(values.size - 1)
    start = int(values.size - 1)
    while start > 0 and bool(lifted[start - 1]):
        start -= 1
    return True, start


def _max_error(reference: np.ndarray, candidate: np.ndarray, limit: int = 30) -> float:
    a = np.asarray(reference)
    b = np.asarray(candidate)
    n = min(len(a), len(b), limit)
    if n == 0:
        return float("inf")
    return float(np.max(np.abs(a[:n].astype(float) - b[:n].astype(float))))


def _first_error_step(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    threshold: float,
    limit: int = 30,
) -> int | None:
    a = np.asarray(reference)
    b = np.asarray(candidate)
    n = min(len(a), len(b), limit)
    if n == 0:
        return 0
    error = np.abs(a[:n].astype(float) - b[:n].astype(float))
    material = error > threshold
    if material.ndim > 1:
        material = material.reshape(n, -1).any(axis=1)
    indices = np.flatnonzero(material)
    return int(indices[0]) if indices.size else None


def _first_true(values: Any) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool).reshape(-1))
    return int(indices[0]) if indices.size else None


def compare_traces(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare one restored continuation with its uninterrupted reference."""
    ref_success = bool(reference["success"])
    cand_success = bool(candidate["success"])
    first_divergence = {
        "observation": _first_error_step(
            reference["observations"], candidate["observations"], threshold=1e-3
        ),
        "joint_position": _first_error_step(
            reference["joint_pos"], candidate["joint_pos"], threshold=1e-4
        ),
        "joint_velocity": _first_error_step(
            reference["joint_vel"], candidate["joint_vel"], threshold=1e-3
        ),
        "object_pose": _first_error_step(
            reference["object_pose"], candidate["object_pose"], threshold=1e-4
        ),
        "object_velocity": _first_error_step(
            reference["object_vel"], candidate["object_vel"], threshold=1e-3
        ),
    }
    material_steps = [step for step in first_divergence.values() if step is not None]
    return {
        "success_match": ref_success == cand_success,
        "recovery_verdict_match": ref_success == cand_success,
        "reference_success": ref_success,
        "candidate_success": cand_success,
        "outcome_time_delta": abs(int(reference["outcome_step"]) - int(candidate["outcome_step"])),
        "success_time_difference": abs(
            int(reference["outcome_step"]) - int(candidate["outcome_step"])
        ),
        "horizon_delta": abs(int(reference["horizon"]) - int(candidate["horizon"])),
        "obs_max_abs_30": _max_error(reference["observations"], candidate["observations"]),
        "joint_pos_max_abs_30": _max_error(reference["joint_pos"], candidate["joint_pos"]),
        "joint_vel_max_abs_30": _max_error(reference["joint_vel"], candidate["joint_vel"]),
        "object_pose_max_abs_30": _max_error(reference["object_pose"], candidate["object_pose"]),
        "object_vel_max_abs_30": _max_error(reference["object_vel"], candidate["object_vel"]),
        "reward_max_abs_30": _max_error(reference["rewards"], candidate["rewards"]),
        "termination_match": np.array_equal(
            np.asarray(reference["dones"], dtype=bool), np.asarray(candidate["dones"], dtype=bool)
        ),
        "termination_difference": _first_true(reference["dones"])
        != _first_true(candidate["dones"]),
        "reference_termination_step": _first_true(reference["dones"]),
        "candidate_termination_step": _first_true(candidate["dones"]),
        "first_material_divergence": first_divergence,
        "first_material_divergence_step": min(material_steps) if material_steps else None,
        "action_max_abs_30": _max_error(reference["actions"], candidate["actions"]),
    }


def classify_oracle(runs: Sequence[Mapping[str, Any]], expected_repeats: int) -> tuple[str, list[str]]:
    """Apply the approved VALID/CONDITIONALLY VALID/INVALID/UNRESOLVED contract."""
    reasons: list[str] = []
    if len(runs) != expected_repeats:
        return "UNRESOLVED", [f"completed {len(runs)}/{expected_repeats} repeats"]
    if any(not bool(run.get("controls_non_degenerate", False)) for run in runs):
        return "UNRESOLVED", ["an uninterrupted C0/C1/C2 control was degenerate"]

    comparisons = [
        (mode, name, comp)
        for run in runs
        for name, checkpoint in run["checkpoints"].items()
        for mode, comp in checkpoint["comparisons"].items()
    ]
    exact_mismatches = sum(
        not bool(comp["success_match"]) for mode, _name, comp in comparisons if mode == "exact_action"
    )
    policy_mismatches = sum(
        not bool(comp["success_match"]) for mode, _name, comp in comparisons if mode == "policy"
    )
    if exact_mismatches >= 2:
        return "INVALID", [f"exact-action outcome mismatch reproduced {exact_mismatches} times"]
    if policy_mismatches:
        return "INVALID", [f"policy replay outcome mismatch occurred {policy_mismatches} times"]
    if exact_mismatches:
        return "UNRESOLVED", ["one non-reproduced exact-action outcome mismatch"]

    for run in runs:
        recovery_name = "pre_success_contact" if "pre_success_contact" in run["checkpoints"] else "post_grasp"
        for mode in ("exact_action", "policy"):
            c1 = run["checkpoints"][recovery_name]["comparisons"][mode]["candidate_success"]
            c2 = run["checkpoints"]["post_teleport"]["comparisons"][mode]["candidate_success"]
            if not c1 or c2:
                return "INVALID", [f"{mode} did not preserve the recoverable C1/unrecoverable C2 boundary"]
    boundary_steps = [int(run["boundary_step"]) for run in runs]
    if max(boundary_steps) - min(boundary_steps) > 1:
        return "INVALID", ["recovery boundary varied by more than one policy step"]

    locality = max(float(run["primary_locality_max_abs"]) for run in runs)
    if locality > 1e-6:
        return "INVALID", [f"probe reset moved the primary by {locality:.3e}"]
    for mode, name, comp in comparisons:
        allowed = 1 if mode == "exact_action" else 5
        if int(comp["outcome_time_delta"]) > allowed:
            return "INVALID", [f"{mode} {name} outcome timing delta exceeded {allowed} steps"]

    exact = [comp for mode, _name, comp in comparisons if mode == "exact_action"]
    trajectory_valid = all(
        comp["obs_max_abs_30"] <= 1e-3
        and comp["joint_pos_max_abs_30"] <= 1e-4
        and comp["joint_vel_max_abs_30"] <= 1e-3
        and comp["object_pose_max_abs_30"] <= 1e-4
        and comp["object_vel_max_abs_30"] <= 1e-3
        and comp["reward_max_abs_30"] <= 1e-3
        and bool(comp["termination_match"])
        for comp in exact
    )
    if trajectory_valid:
        reasons.append("outcomes, timing, boundary, locality, and exact-action trajectories met contract")
        return "VALID", reasons
    reasons.append("outcomes and boundary held, but exact-action trajectory equivalence exceeded tolerance")
    return "CONDITIONALLY_VALID", reasons
