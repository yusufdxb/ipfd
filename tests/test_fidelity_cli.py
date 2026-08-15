from __future__ import annotations

import json

from ipfd.cli import main


def test_audit_configuration_failure_returns_nonzero(tmp_path, capsys):
    exit_status = main(["audit", "--config", str(tmp_path / "missing.yaml")])

    assert exit_status == 2
    assert "IPFD_AUDIT_ERROR" in capsys.readouterr().err


def test_adapter_check_import_failure_returns_runtime_error_code(capsys):
    exit_status = main(["adapter-check", "not-a-target", "--decision", "success"])

    assert exit_status == 2
    assert "IPFD_ADAPTER_CHECK_ERROR" in capsys.readouterr().err


def _summary(result: str, divergence: int | None, disagreement: bool) -> dict:
    return {
        "schema_version": 1,
        "configurations": [
            {
                "comparison_key": {"environment": "demo", "horizon": 90},
                "scope": {"environment": "demo", "snapshot_protocol": result.lower()},
                "result": result,
                "levels": {
                    "L0": {"passed": result == "SUPPORTED"},
                    "L1": {"passed": result == "SUPPORTED"},
                    "L2": {"passed": divergence is None, "first_numerical_divergence": divergence},
                    "L3": {"decision_disagreement": disagreement},
                },
            }
        ],
    }


def test_compare_is_compact_and_regressions_return_one(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_summary("SUPPORTED", None, False)), encoding="utf-8")
    candidate.write_text(json.dumps(_summary("UNSUPPORTED", 12, True)), encoding="utf-8")

    assert main(["compare", str(baseline), str(candidate)]) == 1
    rendered = capsys.readouterr().out
    assert "IPFD AUDIT COMPARISON" in rendered
    assert "none -> 12" in rendered
    assert len(rendered.splitlines()) < 20

    assert main(["compare", str(baseline), str(candidate), "--json"]) == 1
    assert '"previously_supported_became_unsupported": true' in capsys.readouterr().out
