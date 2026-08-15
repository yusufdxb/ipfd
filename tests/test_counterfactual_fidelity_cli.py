from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfd.cli import main


def _record(*, seed: int, horizon: int, restored: bool = True, protocol: str = "expanded") -> dict[str, object]:
    return {
        "cluster_id": f"seed-{seed}",
        "branch_id": f"branch-{seed}",
        "protocol": protocol,
        "continuation": "exact_action",
        "disturbance": "gripper_open",
        "phase": "contact",
        "predicate": "sustained_lift",
        "horizon": horizon,
        "actual_continuation_steps": horizon,
        "reference_decision": True,
        "restored_decision": restored,
        "decision_match": restored,
        "schedule_id": "schedule-1",
        "schedule_equivalent": True,
    }


def _write_records(path: Path) -> bytes:
    payload = "".join(json.dumps(_record(seed=seed, horizon=horizon)) + "\n" for seed in (1, 2) for horizon in (1, 3)).encode()
    path.write_bytes(payload)
    return payload


def test_fidelity_cli_text_separates_branch_and_seed_evidence(tmp_path, capsys) -> None:
    path = tmp_path / "records.jsonl"
    _write_records(path)

    status = main(
        [
            "fidelity",
            str(path),
            "--minimum-independent-seeds",
            "2",
            "--bootstrap-samples",
            "30",
            "--bootstrap-seed",
            "11",
        ]
    )
    captured = capsys.readouterr()

    assert status == 0
    assert captured.err == ""
    assert "COUNTERFACTUAL FIDELITY AUDIT" in captured.out
    assert "Branch comparisons analyzed: 4" in captured.out
    assert "Independent seed groups observed: 2" in captured.out
    assert "no disagreement observed" in captured.out
    assert "Seed groups, not branch comparisons, are the independent units" in captured.out
    assert "No disagreement observed is not proof of validity" in captured.out


def test_fidelity_cli_json_records_manifest_effective_grouping_and_gates(tmp_path, capsys) -> None:
    path = tmp_path / "records.jsonl"
    raw = _write_records(path)
    output = tmp_path / "audit.json"

    status = main(
        [
            "fidelity",
            str(path),
            "--group-by",
            "phase",
            "--minimum-independent-seeds",
            "2",
            "--bootstrap-samples",
            "30",
            "--bootstrap-seed",
            "11",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert status == 0
    assert captured.out == captured.err == ""
    assert payload["record_count"] == 4
    assert payload["independent_seed_count"] == 2
    assert payload["configuration"]["requested_group_by"] == ["phase"]
    assert payload["configuration"]["effective_group_by"] == [
        "phase",
        "protocol",
        "continuation",
        "predicate",
    ]
    assert payload["evidence_manifest"]["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert payload["evidence_manifest"]["analysis"]["configuration"]["bootstrap_seed"] == 11
    assert payload["envelopes"][0]["points"][0]["gate"]["status"] == "ACCEPT_OBSERVED_ENVELOPE"
    assert payload["envelopes"][0]["claim_strength"] == "DESCRIPTIVE_ONLY"


@pytest.mark.parametrize(
    "extra_args",
    [
        [],
        ["--max-disagreement", "1.1"],
        ["--minimum-independent-seeds", "0"],
        ["--bootstrap-samples", "0"],
        ["--protocol", "unseen"],
    ],
)
def test_fidelity_cli_data_and_configuration_failures_return_two(
    tmp_path,
    capsys,
    extra_args: list[str],
) -> None:
    path = tmp_path / "records.jsonl"
    if extra_args:
        _write_records(path)
    else:
        path.write_bytes(b"")

    status = main(["fidelity", str(path), *extra_args])
    captured = capsys.readouterr()

    assert status == 2
    assert captured.out == ""
    assert "IPFD_FIDELITY_ERROR" in captured.err


def test_fidelity_cli_rejects_malformed_json_with_exit_two(tmp_path, capsys) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    status = main(["fidelity", str(path)])
    captured = capsys.readouterr()

    assert status == 2
    assert "invalid JSON" in captured.err


def test_fidelity_cli_json_is_deterministic_for_fixed_bootstrap_seed(tmp_path, capsys) -> None:
    path = tmp_path / "records.jsonl"
    _write_records(path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    common = [
        "fidelity",
        str(path),
        "--minimum-independent-seeds",
        "2",
        "--bootstrap-samples",
        "50",
        "--bootstrap-seed",
        "91",
        "--format",
        "json",
    ]

    assert main([*common, "--output", str(first)]) == 0
    assert main([*common, "--output", str(second)]) == 0
    capsys.readouterr()

    assert first.read_bytes() == second.read_bytes()


def test_fidelity_cli_refuses_to_overwrite_source_evidence(tmp_path, capsys) -> None:
    path = tmp_path / "records.jsonl"
    original = _write_records(path)

    status = main(["fidelity", str(path), "--output", str(path)])
    captured = capsys.readouterr()

    assert status == 2
    assert "must not overwrite" in captured.err
    assert path.read_bytes() == original


def test_fidelity_cli_reports_paired_protocol_delta(tmp_path, capsys) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        _record(
            seed=seed,
            horizon=horizon,
            protocol=protocol,
            restored=not (protocol == "basic" and seed == 1 and horizon == 3),
        )
        for seed in (1, 2)
        for horizon in (1, 3)
        for protocol in ("basic", "expanded")
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    output = tmp_path / "audit.json"

    status = main(
        [
            "fidelity",
            str(path),
            "--minimum-independent-seeds",
            "2",
            "--bootstrap-samples",
            "30",
            "--compare-protocols",
            "basic,expanded",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )
    capsys.readouterr()
    comparison = json.loads(output.read_text())["protocol_comparison"]

    assert status == 0
    assert comparison["pairs"] == 4
    assert comparison["paired_outcomes"]["fixed_by_protocol_b"] == 1
    assert comparison["paired_outcomes"]["introduced_by_protocol_b"] == 0
    assert comparison["schedule_identity_verified"] is True


def test_fidelity_cli_hashes_related_provenance_without_exposing_path(tmp_path, capsys) -> None:
    path = tmp_path / "records.jsonl"
    _write_records(path)
    provenance = tmp_path / "study_provenance.json"
    provenance_payload = b'{"runtime":"declared-test-runtime"}\n'
    provenance.write_bytes(provenance_payload)
    output = tmp_path / "audit.json"

    status = main(
        [
            "fidelity",
            str(path),
            "--minimum-independent-seeds",
            "2",
            "--bootstrap-samples",
            "10",
            "--provenance",
            str(provenance),
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )
    capsys.readouterr()
    related = json.loads(output.read_text())["configuration"]["related_provenance"]

    assert status == 0
    assert related == {
        "logical_name": "study_provenance.json",
        "sha256": hashlib.sha256(provenance_payload).hexdigest(),
        "bytes": len(provenance_payload),
    }
    assert str(tmp_path) not in output.read_text()
