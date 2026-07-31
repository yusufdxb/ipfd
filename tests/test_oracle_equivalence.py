from __future__ import annotations

import numpy as np

from ipfd.oracle_equivalence import classify_oracle, compare_traces, terminal_outcome


def _trace(success: bool = True) -> dict:
    values = np.zeros((4, 2), dtype=float)
    return {
        "success": success,
        "outcome_step": 2 if success else 3,
        "horizon": 4,
        "observations": values,
        "joint_pos": values,
        "joint_vel": values,
        "object_pose": values,
        "object_vel": values,
        "rewards": np.zeros(4),
        "dones": np.zeros(4, dtype=bool),
        "actions": values,
    }


def _run() -> dict:
    checkpoints = {}
    for name, success in (("pre_contact", True), ("pre_success_contact", True), ("post_teleport", False)):
        ref = _trace(success)
        checkpoints[name] = {
            "comparisons": {
                "exact_action": compare_traces(ref, _trace(success)),
                "policy": compare_traces(ref, _trace(success)),
            }
        }
    return {
        "controls_non_degenerate": True,
        "primary_locality_max_abs": 0.0,
        "boundary_step": 12,
        "checkpoints": checkpoints,
    }


def test_terminal_outcome_rejects_transient_lift():
    assert terminal_outcome([0.0, 0.07, 0.01], rest_height=0.0, lift_threshold=0.06) == (False, 2)


def test_terminal_outcome_reports_stable_suffix_start():
    assert terminal_outcome([0.0, 0.07, 0.08], rest_height=0.0, lift_threshold=0.06) == (True, 1)


def test_compare_traces_detects_outcome_mismatch():
    result = compare_traces(_trace(True), _trace(False))
    assert not result["success_match"]
    assert not result["recovery_verdict_match"]


def test_compare_traces_reports_first_material_divergence_step():
    candidate = _trace(True)
    candidate["object_pose"][2, 0] = 0.01
    result = compare_traces(_trace(True), candidate)
    assert result["first_material_divergence"]["object_pose"] == 2
    assert result["first_material_divergence_step"] == 2


def test_compare_traces_reports_termination_step_difference():
    reference = _trace(False)
    candidate = _trace(False)
    reference["dones"][2] = True
    candidate["dones"][3] = True
    result = compare_traces(reference, candidate)
    assert result["termination_difference"] is True
    assert result["reference_termination_step"] == 2
    assert result["candidate_termination_step"] == 3


def test_classify_valid_equivalence():
    status, _ = classify_oracle([_run(), _run()], expected_repeats=2)
    assert status == "VALID"


def test_classify_reproduced_exact_action_mismatch_invalid():
    runs = [_run(), _run()]
    for run in runs:
        run["checkpoints"]["pre_contact"]["comparisons"]["exact_action"]["success_match"] = False
    status, reasons = classify_oracle(runs, expected_repeats=2)
    assert status == "INVALID"
    assert "reproduced" in reasons[0]


def test_classify_trajectory_drift_conditionally_valid():
    run = _run()
    run["checkpoints"]["pre_success_contact"]["comparisons"]["exact_action"]["obs_max_abs_30"] = 0.2
    status, _ = classify_oracle([run], expected_repeats=1)
    assert status == "CONDITIONALLY_VALID"
