from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ipfd.branch_study import (
    HORIZONS,
    PROTOCOL_A,
    PROTOCOL_B,
    DecisionSignals,
    DisturbanceSchedule,
    PhaseSignals,
    PhaseTracker,
    SeedBundle,
    assert_schedule_equivalence,
    decision_predicates,
    horizon_reached,
    validate_horizons,
    validate_protocol_bookkeeping,
    validate_seed_bundles,
)


def _schedule(*, start: int = 50) -> DisturbanceSchedule:
    return DisturbanceSchedule(
        kind="gripper_open_interruption",
        start_step=start,
        duration_steps=8,
        magnitude=(1.0,),
        target="gripper_action",
        random_values=(),
    )


def test_disturbance_schedule_equivalence_is_exact():
    left = _schedule()
    right = _schedule()
    assert left.sha256 == right.sha256
    assert_schedule_equivalence(left, right)
    with pytest.raises(AssertionError):
        assert_schedule_equivalence(left, _schedule(start=51))


def test_horizon_semantics_count_post_branch_actions():
    assert validate_horizons(HORIZONS) == HORIZONS
    assert not horizon_reached(branch_step=10, current_step=10, horizon=1)
    assert horizon_reached(branch_step=10, current_step=11, horizon=1)
    assert horizon_reached(branch_step=10, current_step=100, horizon=90)
    with pytest.raises(ValueError):
        validate_horizons((1, 5, 3))


def test_seed_metadata_rejects_correlated_pseudo_replicates():
    bases = (101, 211, 307, 401, 503)
    bundles = [
        SeedBundle.derive(base, disturbance) for base in bases for disturbance in ("object_teleport", "gripper_open_interruption")
    ]
    validate_seed_bundles(bundles)
    with pytest.raises(ValueError):
        validate_seed_bundles([bundles[0]] * 5)


def test_phase_assignment_uses_contact_and_temporal_windows():
    tracker = PhaseTracker()
    emitted = tracker.update(
        PhaseSignals(
            step=0,
            object_rise_m=0.0,
            finger_aperture_m=0.08,
            ee_object_distance_m=0.4,
            left_object_force_n=0.0,
            right_object_force_n=0.0,
        )
    )
    assert emitted == ["free_space_pre_manipulation"]

    approach_emissions = []
    for step, distance in ((1, 0.17), (2, 0.15), (3, 0.13)):
        approach_emissions.extend(
            tracker.update(
                PhaseSignals(
                    step=step,
                    object_rise_m=0.0,
                    finger_aperture_m=0.08,
                    ee_object_distance_m=distance,
                    left_object_force_n=0.0,
                    right_object_force_n=0.0,
                )
            )
        )
    assert "approach" in approach_emissions

    emitted = tracker.update(
        PhaseSignals(
            step=4,
            object_rise_m=0.0,
            finger_aperture_m=0.05,
            ee_object_distance_m=0.1,
            left_object_force_n=1.0,
            right_object_force_n=0.0,
        )
    )
    assert emitted == ["first_contact"]

    grasp_emissions = []
    for step in (5, 6, 7, 8):
        grasp_emissions.extend(
            tracker.update(
                PhaseSignals(
                    step=step,
                    object_rise_m=0.006 if step >= 7 else 0.0,
                    finger_aperture_m=0.04,
                    ee_object_distance_m=0.08,
                    left_object_force_n=1.0,
                    right_object_force_n=1.0,
                )
            )
        )
    assert "stable_grasp" in grasp_emissions
    assert "initial_lift" in grasp_emissions


def test_sustained_success_requires_window_pose_and_no_termination():
    good = [DecisionSignals(0.05, 0.08, 0.04, 1.0, 1.0) for _ in range(5)]
    result = decision_predicates(good)
    assert result["sustained_lift"]
    assert result["stable_grasp"]
    assert not result["final_height"]

    broken_pose = good[:-1] + [DecisionSignals(0.05, 0.2, 0.04, 1.0, 1.0)]
    assert not decision_predicates(broken_pose)["sustained_lift"]
    terminated = good[:-1] + [DecisionSignals(0.07, 0.08, 0.04, 1.0, 1.0, True)]
    assert not any(decision_predicates(terminated).values())


def test_snapshot_protocol_bookkeeping_is_fail_closed():
    validate_protocol_bookkeeping()
    assert PROTOCOL_A != PROTOCOL_B


def test_primary_runner_returns_nonzero_before_runtime_on_missing_checkpoint(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_snapshot_protocol_study.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--asset-root",
            "https://invalid.example/unused",
            "--output-dir",
            str(tmp_path / "result"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "checkpoint does not exist" in completed.stderr
