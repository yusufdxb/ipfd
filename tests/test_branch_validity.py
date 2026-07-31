from __future__ import annotations

import pytest

from ipfd.branch_validity import comparison_records, summarize_branch_validity, wilson_interval


def _report(*, exact_match: bool = True, policy_match: bool = True) -> dict:
    def comparison(match: bool) -> dict:
        return {
            "success_match": match,
            "reference_success": True,
            "candidate_success": match,
            "first_material_divergence_step": 1,
            "object_pose_max_abs_30": 0.2,
        }

    return {
        "schema_version": 2,
        "classification": "GENERALLY_INVALID",
        "runs": [
            {
                "seed": 3,
                "repeat": 0,
                "disturbance": "gripper_interrupt",
                "checkpoints": {
                    "mid_contact": {
                        "task_phase": "mid_contact",
                        "immediate_roundtrip": {
                            "exact_obs_max_abs": 0.0,
                            "policy_obs_max_abs": 0.0,
                        },
                        "comparisons": {
                            "exact_action": comparison(exact_match),
                            "policy": comparison(policy_match),
                        },
                    }
                },
            }
        ],
    }


def test_wilson_interval_contains_observed_rate():
    lower, upper = wilson_interval(8, 10)
    assert lower < 0.8 < upper


@pytest.mark.parametrize(("successes", "total"), [(-1, 1), (2, 1), (0, 0)])
def test_wilson_interval_rejects_invalid_counts(successes: int, total: int):
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def test_comparison_records_preserve_branch_context():
    records = comparison_records(_report())
    assert len(records) == 2
    assert records[0]["phase"] == "mid_contact"
    assert records[0]["seed"] == 3
    assert records[0]["immediate_obs_max_abs"] == 0.0


def test_summary_falsifies_universal_validity_on_one_disagreement():
    summary = summarize_branch_validity(
        _report(exact_match=False),
        source_label="test",
        source_sha256="a" * 64,
    )
    assert summary["result"] == "FALSIFIED_UNIVERSAL_DECISION_FIDELITY"
    assert summary["overall"]["disagreements"] == 1
    assert summary["by_continuation"]["exact_action"]["false_unrecoverable"] == 1
    assert summary["validity_gate"]["passing_cells"] == 0


def test_zero_observed_disagreements_is_not_certification():
    summary = summarize_branch_validity(
        _report(),
        source_label="test",
        source_sha256="b" * 64,
    )
    assert summary["result"] == "NO_DISAGREEMENT_OBSERVED_NOT_CERTIFIED"
    assert summary["validity_gate"]["passing_cells"] == 0
