from __future__ import annotations

import hashlib
import importlib.util
import json
from importlib import resources
from pathlib import Path

import pytest

from ipfd.demo import main, run_demo

_MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="MuJoCo is not installed")
def test_demo_proves_delayed_semantic_failure_and_richer_protocol_control(tmp_path):
    output = tmp_path / "demo"

    result = run_demo(output)

    protocols = {item["name"]: item for item in result["protocols"]}
    narrow = protocols["minimal_visible"]["fidelity"]
    improved = protocols["integration_with_warmstart"]["fidelity"]
    activation = protocols["minimal + activation (ablation)"]["fidelity"]
    control = protocols["minimal_visible (control, branch=0)"]["fidelity"]
    assert control["l3_decision"] == "PASS"
    assert narrow["l0_restore"] == "PASS"
    assert narrow["l1_one_step"] == "PASS"
    assert [narrow["l2_by_horizon"][str(value)] for value in (1, 5, 10)] == [
        "PASS",
        "PASS",
        "PASS",
    ]
    assert narrow["l2_by_horizon"]["30"] == "DEGRADED"
    assert narrow["l2_by_horizon"]["90"] == "FAIL"
    assert narrow["l3_decision"] == "FAIL"
    assert all(value == "PASS" for value in improved["l2_by_horizon"].values())
    assert improved["l3_decision"] == "PASS"
    assert activation["l0_restore"] == "FAIL"
    assert all(value == "PASS" for value in activation["l2_by_horizon"].values())
    assert activation["l3_decision"] == "PASS"
    measured = result["measured_result"]
    assert 10 <= measured["first_numerical_divergence_step"] <= 15
    assert 60 <= measured["first_contact_disagreement_step"] <= 70
    assert measured["reference_decision"] == "remains in contact"
    assert measured["restored_decision"] == "lifts off"
    assert measured["verdict"] == "FAIL_CLOSED"
    assert result["demo_expectation_met"] is True
    assert [item["agreement"] for item in result["sensitivity"]] == [True, False, True]
    assert "actuator_activations" in result["capability_disclosure"]["minimal_visible_unavailable"]
    assert (output / "summary.json").is_file()
    assert (output / "evidence.json").is_file()
    assert (output / "report.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="MuJoCo is not installed")
def test_demo_rerun_has_identical_canonical_artifact_hashes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_demo(first)
    run_demo(second)

    for name in ("summary.json", "evidence.json", "report.png", "artifact_manifest.json"):
        assert _sha256(first / name) == _sha256(second / name)
    manifest = json.loads((first / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, identity in manifest["artifacts"].items():
        assert identity["sha256"] == _sha256(first / name)


@pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="MuJoCo is not installed")
def test_demo_cli_prints_decision_and_copies_summary(tmp_path, capsys):
    output = tmp_path / "artifacts"
    summary_copy = tmp_path / "demo.json"

    assert main(["--output", str(output), "--json", str(summary_copy)]) == 0

    terminal = capsys.readouterr().out
    assert "RESTORE BOUNDARY" in terminal
    assert "reference = remains in contact" in terminal
    assert "restored  = lifts off" in terminal
    assert "FAIL_CLOSED" in terminal
    assert summary_copy.read_bytes() == (output / "summary.json").read_bytes()


def test_documented_demo_config_matches_packaged_source_of_truth():
    packaged = resources.files("ipfd").joinpath("demo_config.yaml").read_bytes()
    documented = Path("examples/demo/config.yaml").read_bytes()

    assert documented == packaged


def test_demo_missing_mujoco_returns_actionable_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "ipfd.demo._run_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("not installed")),
    )

    assert main(["--output", str(tmp_path / "demo")]) == 2

    error = capsys.readouterr().err
    assert "IPFD_DEMO_ERROR" in error
    assert "ipfd[mujoco]" in error
