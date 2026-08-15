from __future__ import annotations

import json

import pytest

from ipfd.fidelity.regression import compare_audit_files, compare_audits


def _configuration(
    *,
    version: str = "1.0",
    protocol: str = "expanded-runtime",
    result: str = "SUPPORTED",
    l0: bool | None = True,
    l1: bool | None = True,
    divergence: int | None = 30,
    disagreement: bool | None = False,
) -> dict:
    return {
        "scope": {
            "simulator": "reference-sim",
            "simulator_version": version,
            "environment": "contact-case",
            "snapshot_protocol": protocol,
            "decision_function": "stable-contact",
        },
        "result": result,
        "levels": {
            "L0": {"passed": l0},
            "L1": {"passed": l1},
            "L2": {"first_numerical_divergence": divergence},
            "L3": {"decision_disagreement": disagreement},
        },
    }


def _audit(*configurations: dict) -> dict:
    return {"schema_version": 1, "configurations": list(configurations)}


def test_compare_audits_reports_all_required_regression_questions():
    baseline = _configuration(result="SUPPORTED", divergence=30, disagreement=False)
    candidate = _configuration(
        result="UNSUPPORTED",
        l0=False,
        l1=False,
        divergence=10,
        disagreement=True,
    )

    report = compare_audits(_audit(baseline), _audit(candidate))

    assert report["summary"] == {
        "matched_configurations": 1,
        "added_configurations": 0,
        "removed_configurations": 0,
        "l0_changed": True,
        "l1_changed": True,
        "first_divergence_change_counts": {
            "earlier": 1,
            "later": 0,
            "same": 0,
            "unavailable": 0,
        },
        "decision_disagreement_changed": True,
        "previously_supported_became_unsupported": True,
    }
    comparison = report["comparisons"][0]
    assert comparison["first_divergence"]["change"] == "earlier"
    assert comparison["previously_supported_became_unsupported"] is True
    assert report["supported_to_unsupported"][0]["baseline_scope"] == baseline["scope"]


@pytest.mark.parametrize(
    ("baseline_step", "candidate_step", "expected"),
    [(10, 30, "later"), (10, 10, "same"), (None, 10, "unavailable"), (10, None, "unavailable")],
)
def test_first_divergence_change_is_explicit(baseline_step, candidate_step, expected):
    baseline = _configuration(divergence=baseline_step)
    candidate = _configuration(divergence=candidate_step)
    comparison = compare_audits(_audit(baseline), _audit(candidate))["comparisons"][0]
    assert comparison["first_divergence"]["change"] == expected


def test_missing_measurements_do_not_claim_no_change():
    baseline = _configuration(l0=None, l1=True, disagreement=None)
    candidate = _configuration(l0=True, l1=True, disagreement=False)

    summary = compare_audits(_audit(baseline), _audit(candidate))["summary"]

    assert summary["l0_changed"] is None
    assert summary["l1_changed"] is False
    assert summary["decision_disagreement_changed"] is None


def test_added_and_removed_scopes_preserve_raw_labels():
    removed = _configuration(protocol="scene-only")
    added = _configuration(protocol="full-integration")

    report = compare_audits(_audit(removed), _audit(added))

    assert report["comparisons"] == []
    assert report["removed_scopes"] == [removed["scope"]]
    assert report["added_scopes"] == [added["scope"]]
    assert report["summary"]["previously_supported_became_unsupported"] is False


def test_scope_mapping_order_does_not_prevent_matching():
    baseline = _configuration()
    candidate = _configuration()
    candidate["scope"] = dict(reversed(list(candidate["scope"].items())))
    candidate["levels"]["L2"]["first_numerical_divergence"] = 5

    report = compare_audits(_audit(baseline), _audit(candidate))

    assert report["summary"]["matched_configurations"] == 1
    assert report["comparisons"][0]["candidate_scope"] == candidate["scope"]


def test_explicit_comparison_key_pairs_different_version_scopes():
    baseline = _configuration(version="1.0", divergence=30)
    candidate = _configuration(
        version="2.0", protocol="full-integration", divergence=10
    )
    baseline["comparison_key"] = "contact-case-expanded-runtime"
    candidate["comparison_key"] = "contact-case-expanded-runtime"

    report = compare_audits(_audit(baseline), _audit(candidate))

    comparison = report["comparisons"][0]
    assert comparison["comparison_key"] == "contact-case-expanded-runtime"
    assert comparison["baseline_scope"]["simulator_version"] == "1.0"
    assert comparison["candidate_scope"]["simulator_version"] == "2.0"
    assert comparison["scope_changes"] == [
        {
            "field": "simulator_version",
            "baseline_present": True,
            "candidate_present": True,
            "baseline": "1.0",
            "candidate": "2.0",
        },
        {
            "field": "snapshot_protocol",
            "baseline_present": True,
            "candidate_present": True,
            "baseline": "expanded-runtime",
            "candidate": "full-integration",
        },
    ]
    assert comparison["first_divergence"]["change"] == "earlier"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda audit: audit.update(schema_version=2), "schema_version must be 1"),
        (lambda audit: audit.pop("configurations"), "configurations must be a list"),
        (lambda audit: audit["configurations"][0].pop("scope"), "scope must be a nonempty mapping"),
        (lambda audit: audit["configurations"][0].update(result="MAYBE"), "result must be one of"),
        (lambda audit: audit["configurations"][0]["levels"]["L0"].update(passed=1), "passed must be bool or null"),
        (
            lambda audit: audit["configurations"][0]["levels"]["L2"].update(first_numerical_divergence=-1),
            "must be a nonnegative integer or null",
        ),
    ],
)
def test_invalid_audit_schema_is_rejected(mutate, message):
    audit = _audit(_configuration())
    mutate(audit)
    with pytest.raises(ValueError, match=message):
        compare_audits(audit, _audit(_configuration()))


def test_duplicate_and_non_json_scopes_are_rejected():
    configuration = _configuration()
    with pytest.raises(ValueError, match="duplicate comparison key"):
        compare_audits(_audit(configuration, configuration), _audit(_configuration()))

    configuration = _configuration()
    configuration["scope"]["tolerance"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON values"):
        compare_audits(_audit(configuration), _audit(_configuration()))


def test_compare_audit_files_loads_json(tmp_path):
    baseline = _audit(_configuration(divergence=30))
    candidate = _audit(_configuration(divergence=90))
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    report = compare_audit_files(baseline_path, candidate_path)

    assert report["comparisons"][0]["first_divergence"]["change"] == "later"


def test_compare_audit_files_rejects_non_object_json(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text("[]", encoding="utf-8")
    candidate_path.write_text(json.dumps(_audit(_configuration())), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        compare_audit_files(baseline_path, candidate_path)
