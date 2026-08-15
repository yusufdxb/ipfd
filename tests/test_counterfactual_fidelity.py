from __future__ import annotations

import hashlib
import json
import lzma
import math
from pathlib import Path

import pytest

from ipfd.fidelity import (
    BranchComparison,
    ErrorDirection,
    FidelityGateStatus,
    FrontierStatus,
    compare_restore_protocols,
    evaluate_fidelity_gate,
    fidelity_envelope,
    first_untrusted_horizon,
    load_branch_comparisons,
    normalize_branch_comparison,
    seed_cluster_bootstrap,
    summarize_by_seed,
    summarize_error_direction,
    validate_branch_comparisons,
)


def _comparison(
    *,
    cluster: str = "seed-1",
    branch: str = "branch-1",
    protocol: str = "expanded",
    continuation: str = "exact_action",
    disturbance: str = "gripper_open",
    phase: str = "contact",
    predicate: str = "sustained_lift",
    horizon: int = 1,
    reference: bool = True,
    restored: bool = True,
    schedule_id: str | None = "schedule-1",
    schedule_equivalent: bool | None = True,
    replicate: str = "0",
) -> BranchComparison:
    return BranchComparison(
        cluster_id=cluster,
        branch_id=branch,
        protocol=protocol,
        continuation=continuation,
        disturbance=disturbance,
        phase=phase,
        predicate=predicate,
        horizon=horizon,
        actual_continuation_steps=horizon,
        reference_decision=reference,
        candidate_decision=restored,
        branch_step=4,
        schedule_id=schedule_id,
        schedule_equivalent=schedule_equivalent,
        replicate_id=replicate,
    )


def _record(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "cluster_id": "seed-1",
        "branch_id": "branch-1",
        "protocol": "expanded",
        "continuation": "exact_action",
        "disturbance": "gripper_open",
        "phase": "contact",
        "predicate": "sustained_lift",
        "horizon": 3,
        "actual_continuation_steps": 3,
        "reference_decision": True,
        "restored_decision": True,
        "decision_match": True,
        "branch_step": 4,
        "schedule_id": "schedule-1",
        "schedule_equivalent": True,
    }
    value.update(overrides)
    return value


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_zero_disagreement_is_descriptive_not_a_certificate() -> None:
    records = [
        _comparison(cluster=f"seed-{seed}", branch=f"b-{seed}", horizon=horizon) for seed in range(5) for horizon in (1, 3, 5)
    ]

    envelope = fidelity_envelope(records, bootstrap_samples=50)[0]
    frontier = first_untrusted_horizon(
        envelope,
        max_disagreement=0.0,
        minimum_independent_seeds=5,
    )

    assert [point.disagreements for point in envelope.points] == [0, 0, 0]
    assert all(point.to_dict()["interpretation"] == "no disagreement observed" for point in envelope.points)
    assert frontier.status is FrontierStatus.ALL_TESTED_WITHIN_TOLERANCE
    assert frontier.last_acceptable_tested_horizon == 5
    assert frontier.claim_strength == "DESCRIPTIVE_ONLY"


def test_single_late_horizon_disagreement_brackets_observed_frontier() -> None:
    records = []
    for horizon in (1, 3):
        for seed in range(5):
            records.append(
                _comparison(
                    cluster=f"seed-{seed}",
                    branch=f"b-{seed}",
                    horizon=horizon,
                    restored=not (horizon == 3 and seed == 0),
                )
            )

    envelope = fidelity_envelope(records, bootstrap_samples=50)[0]
    frontier = first_untrusted_horizon(
        envelope,
        max_disagreement=0.1,
        minimum_independent_seeds=5,
    )

    assert envelope.first_observed_disagreement_horizon == 3
    assert frontier.status is FrontierStatus.BRACKETED
    assert frontier.last_acceptable_tested_horizon == 1
    assert frontier.first_rejected_tested_horizon == 3


def test_non_monotonic_pass_fail_pass_is_not_forced_into_a_frontier() -> None:
    records = [
        _comparison(
            cluster=f"seed-{seed}",
            branch=f"b-{seed}",
            horizon=horizon,
            restored=horizon != 3,
        )
        for seed in range(3)
        for horizon in (1, 3, 5)
    ]
    envelope = fidelity_envelope(records, bootstrap_samples=50)[0]

    frontier = first_untrusted_horizon(
        envelope,
        max_disagreement=0.25,
        minimum_independent_seeds=3,
    )
    gate = evaluate_fidelity_gate(
        envelope,
        horizon=5,
        max_disagreement=0.25,
        minimum_independent_seeds=3,
    )

    assert frontier.status is FrontierStatus.NON_MONOTONIC
    assert gate.status is FidelityGateStatus.NON_MONOTONIC_EVIDENCE
    assert gate.accepted is False


def test_error_direction_separates_optimistic_and_pessimistic_errors() -> None:
    records = [
        _comparison(branch="agree", reference=True, restored=True),
        _comparison(branch="false-recoverable", reference=False, restored=True),
        _comparison(branch="false-unrecoverable", reference=True, restored=False),
    ]

    profile = summarize_error_direction(records)

    assert records[1].error_direction is ErrorDirection.FALSE_RECOVERABLE
    assert records[2].error_direction is ErrorDirection.FALSE_UNRECOVERABLE
    assert profile.comparisons == 3
    assert profile.disagreements == 2
    assert profile.false_recoverable == 1
    assert profile.false_unrecoverable == 1


def test_protocol_comparison_counts_fixed_introduced_and_shared_errors() -> None:
    outcomes = [
        ("both-agree", True, True),
        ("fixed", False, True),
        ("introduced", True, False),
        ("both-disagree", False, False),
    ]
    records = [
        _comparison(branch=branch, protocol=protocol, restored=a if protocol == "A" else b)
        for branch, a, b in outcomes
        for protocol in ("A", "B")
    ]

    result = compare_restore_protocols(
        records,
        protocol_a="A",
        protocol_b="B",
        samples=100,
        random_seed=9,
    )

    assert result.pairs == 4
    assert result.both_agree == 1
    assert result.fixed_by_protocol_b == 1
    assert result.introduced_by_protocol_b == 1
    assert result.both_disagree == 1
    assert result.protocol_a_profile.disagreements == 2
    assert result.protocol_b_profile.disagreements == 2
    assert result.branch_rate_difference_b_minus_a == 0.0
    assert result.relative_disagreement_reduction == 0.0
    assert result.schedule_identity_verified is True


def test_protocol_comparison_rejects_missing_partner_and_schedule_mismatch() -> None:
    with pytest.raises(ValueError, match="incomplete restore-protocol pair"):
        compare_restore_protocols(
            [_comparison(protocol="A")],
            protocol_a="A",
            protocol_b="B",
            samples=10,
        )

    mismatched = [
        _comparison(protocol="A", schedule_id="schedule-A"),
        _comparison(protocol="B", schedule_id="schedule-B"),
    ]
    with pytest.raises(ValueError, match="schedules differ"):
        compare_restore_protocols(
            mismatched,
            protocol_a="A",
            protocol_b="B",
            samples=10,
        )


def test_protocol_comparison_fails_closed_without_schedule_evidence() -> None:
    with pytest.raises(ValueError, match="schedule equivalence is unavailable"):
        compare_restore_protocols(
            [
                _comparison(protocol="A", schedule_id=None, schedule_equivalent=None),
                _comparison(protocol="B", schedule_id=None, schedule_equivalent=None),
            ],
            protocol_a="A",
            protocol_b="B",
            samples=10,
        )


def test_insufficient_seed_evidence_fails_frontier_and_gate_closed() -> None:
    records = [
        _comparison(cluster="seed-1", branch="b-1", horizon=1),
        _comparison(cluster="seed-2", branch="b-2", horizon=1),
    ]
    envelope = fidelity_envelope(records, bootstrap_samples=20)[0]

    frontier = first_untrusted_horizon(
        envelope,
        max_disagreement=1.0,
        minimum_independent_seeds=3,
    )
    gate = evaluate_fidelity_gate(
        envelope,
        horizon=1,
        max_disagreement=1.0,
        minimum_independent_seeds=3,
    )

    assert frontier.status is FrontierStatus.INSUFFICIENT_EVIDENCE
    assert gate.status is FidelityGateStatus.INSUFFICIENT_INDEPENDENT_SEEDS
    assert gate.accepted is False


def test_gate_rejects_unseen_stratum_and_untested_horizon() -> None:
    unseen = evaluate_fidelity_gate(
        None,
        horizon=3,
        max_disagreement=0.05,
        minimum_independent_seeds=1,
    )
    envelope = fidelity_envelope([_comparison(horizon=1)], bootstrap_samples=10)[0]
    outside = evaluate_fidelity_gate(
        envelope,
        horizon=2,
        max_disagreement=0.05,
        minimum_independent_seeds=1,
    )

    assert unseen.status is FidelityGateStatus.UNSEEN_STRATUM
    assert unseen.scope is None
    assert outside.status is FidelityGateStatus.OUTSIDE_TESTED_HORIZON
    assert unseen.accepted is outside.accepted is False


def test_gate_rejects_changing_cluster_coverage_across_horizons() -> None:
    records = [
        _comparison(cluster="a", branch="a-1", horizon=1),
        _comparison(cluster="b", branch="b-1", horizon=1),
        _comparison(cluster="b", branch="b-2", horizon=3),
        _comparison(cluster="c", branch="c-2", horizon=3),
    ]
    envelope = fidelity_envelope(records, bootstrap_samples=10)[0]

    gate = evaluate_fidelity_gate(
        envelope,
        horizon=3,
        max_disagreement=1.0,
        minimum_independent_seeds=2,
    )

    assert gate.status is FidelityGateStatus.INCOMPLETE_STRATUM_COVERAGE
    assert gate.accepted is False


def test_uneven_seed_sizes_do_not_become_independent_trials() -> None:
    records = [
        _comparison(
            cluster="large",
            branch=f"large-{index}",
            restored=index == 9,
        )
        for index in range(10)
    ]
    records.append(_comparison(cluster="small", branch="small-0", restored=True))

    summary = summarize_by_seed(records)
    interval = seed_cluster_bootstrap(records, samples=200, random_seed=17)

    assert summary.comparisons == 11
    assert summary.disagreement_rate == pytest.approx(9 / 11)
    assert summary.independent_seed_count == 2
    assert summary.seed_mean_disagreement_rate == pytest.approx(0.45)
    assert interval.estimate == pytest.approx(0.45)
    assert interval.resampling_unit == "independent_seed"


def test_gate_uses_equal_seed_mean_not_branch_pooled_rate() -> None:
    records = [_comparison(cluster="large", branch=f"large-{index}", restored=True) for index in range(100)]
    records.append(_comparison(cluster="small", branch="small", restored=False))
    envelope = fidelity_envelope(records, bootstrap_samples=20)[0]

    gate = evaluate_fidelity_gate(
        envelope,
        horizon=1,
        max_disagreement=0.05,
        minimum_independent_seeds=2,
    )

    assert envelope.points[0].disagreement_rate == pytest.approx(1 / 101)
    assert envelope.points[0].summary.seed_mean_disagreement_rate == pytest.approx(0.5)
    assert gate.status is FidelityGateStatus.REJECT_HIGH_DISAGREEMENT
    assert gate.rate_basis == "EQUAL_SEED_MEAN_DISAGREEMENT_RATE"


def test_core_grouping_cannot_pool_protocol_continuation_or_predicate() -> None:
    with pytest.raises(ValueError, match="group_by must preserve"):
        fidelity_envelope([_comparison()], group_by=("phase",), bootstrap_samples=10)


def test_seed_cluster_bootstrap_is_deterministic_for_fixed_seed() -> None:
    records = [_comparison(cluster=f"seed-{seed}", branch=f"b-{seed}", restored=seed % 2 == 0) for seed in range(5)]

    first = seed_cluster_bootstrap(records, samples=500, random_seed=123)
    again = seed_cluster_bootstrap(records, samples=500, random_seed=123)
    other = seed_cluster_bootstrap(records, samples=500, random_seed=124)

    assert first.to_dict() == again.to_dict()
    assert first.random_seed == 123
    assert other.random_seed == 124


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_extreme_finite_thresholds_are_supported(threshold: float) -> None:
    envelope = fidelity_envelope([_comparison(restored=False)], bootstrap_samples=10)[0]
    frontier = first_untrusted_horizon(
        envelope,
        max_disagreement=threshold,
        minimum_independent_seeds=1,
    )

    expected = (
        FrontierStatus.NO_TESTED_HORIZON_WITHIN_TOLERANCE if threshold == 0.0 else FrontierStatus.ALL_TESTED_WITHIN_TOLERANCE
    )
    assert frontier.status is expected


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf, True])
def test_invalid_thresholds_fail_closed(threshold: float) -> None:
    envelope = fidelity_envelope([_comparison()], bootstrap_samples=10)[0]
    with pytest.raises(ValueError, match="max_disagreement"):
        first_untrusted_horizon(
            envelope,
            max_disagreement=threshold,
            minimum_independent_seeds=1,
        )


def test_minimum_seed_count_must_be_a_strict_positive_integer() -> None:
    envelope = fidelity_envelope([_comparison()], bootstrap_samples=10)[0]
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="minimum_independent_seeds"):
            first_untrusted_horizon(
                envelope,
                max_disagreement=0.05,
                minimum_independent_seeds=invalid,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("horizon", [True, 0, -1, 1.5, "3"])
def test_loader_rejects_malformed_horizons(tmp_path: Path, horizon: object) -> None:
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [_record(horizon=horizon, actual_continuation_steps=horizon)])

    with pytest.raises((TypeError, ValueError), match="horizon"):
        load_branch_comparisons(path)


@pytest.mark.parametrize("value", [0, 1, "true", None, 1.0])
def test_loader_requires_json_boolean_decisions(tmp_path: Path, value: object) -> None:
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [_record(reference_decision=value)])

    with pytest.raises(TypeError, match="reference_decision"):
        load_branch_comparisons(path)


def test_loader_rejects_inconsistent_declared_decision_match(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [_record(reference_decision=True, restored_decision=False, decision_match=True)])

    with pytest.raises(ValueError, match="decision_match contradicts"):
        load_branch_comparisons(path)


@pytest.mark.parametrize(
    "missing",
    [
        "cluster_id",
        "branch_id",
        "protocol",
        "continuation",
        "disturbance",
        "phase",
        "predicate",
        "horizon",
        "actual_continuation_steps",
        "reference_decision",
        "restored_decision",
    ],
)
def test_loader_rejects_missing_required_fields(tmp_path: Path, missing: str) -> None:
    record = _record()
    del record[missing]
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises((TypeError, ValueError), match="missing required field"):
        load_branch_comparisons(path)


def test_loader_rejects_empty_input_and_blank_records(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        load_branch_comparisons(empty)

    blank = tmp_path / "blank.jsonl"
    blank.write_text(json.dumps(_record()) + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank JSONL record"):
        load_branch_comparisons(blank)


def test_loader_rejects_duplicate_composite_keys(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    record = _record()
    duplicate_with_changed_outcome = _record(restored_decision=False, decision_match=False)
    _write_jsonl(path, [record, duplicate_with_changed_outcome])

    with pytest.raises(ValueError, match="duplicate branch comparison key"):
        load_branch_comparisons(path)


def test_loader_supports_archived_aliases_without_silent_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    record = _record()
    record["base_seed"] = record.pop("cluster_id")
    record["candidate_decision"] = record.pop("restored_decision")
    record["schedule_sha256"] = record.pop("schedule_id")
    _write_jsonl(path, [record])

    loaded = load_branch_comparisons(path)

    assert loaded.records[0].cluster_id == "seed-1"
    assert loaded.records[0].candidate_decision is True
    assert loaded.source_sha256

    record["cluster_id"] = "different-seed"
    _write_jsonl(path, [record])
    with pytest.raises(ValueError, match="conflicting aliases"):
        load_branch_comparisons(path)


def test_loader_reads_xz_and_records_container_and_content_hashes(tmp_path: Path) -> None:
    raw = (json.dumps(_record()) + "\n").encode()
    path = tmp_path / "records.jsonl.xz"
    path.write_bytes(lzma.compress(raw))

    loaded = load_branch_comparisons(path)

    assert len(loaded.records) == 1
    assert loaded.compression == "xz"
    assert loaded.content_sha256 == hashlib.sha256(raw).hexdigest()
    assert loaded.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_expands_native_l3_audit_records(tmp_path: Path) -> None:
    native = {
        "schema_version": 1,
        "branch_id": "branch-1",
        "branch_step": 4,
        "cluster": "seed-1",
        "seed": 1,
        "snapshot_protocol": "full_state",
        "continuation_mode": "exact_action",
        "horizon": 3,
        "levels": {
            "L3": {
                "decisions": {
                    "success": {
                        "reference": True,
                        "restored": False,
                        "agreement": False,
                    }
                }
            }
        },
    }
    path = tmp_path / "native.jsonl"
    _write_jsonl(path, [native])

    loaded = load_branch_comparisons(path)

    assert loaded.input_schema == "native_l3_audit_v1"
    assert loaded.records[0].predicate == "success"
    assert loaded.records[0].disturbance == "UNSPECIFIED"
    assert loaded.records[0].candidate_decision is False


def test_public_validation_rejects_non_boolean_decisions_and_non_integer_horizon() -> None:
    non_boolean = _comparison()
    object.__setattr__(non_boolean, "reference_decision", 1)
    with pytest.raises((TypeError, ValueError), match="decisions must be booleans"):
        validate_branch_comparisons([non_boolean])

    fractional_horizon = _comparison()
    object.__setattr__(fractional_horizon, "horizon", 1.5)
    object.__setattr__(fractional_horizon, "actual_continuation_steps", 1.5)
    with pytest.raises((TypeError, ValueError), match="horizon"):
        validate_branch_comparisons([fractional_horizon])


def test_normalizer_rejects_nonequivalent_schedule() -> None:
    with pytest.raises(ValueError, match="schedule is not equivalent"):
        normalize_branch_comparison(_record(schedule_equivalent=False))
