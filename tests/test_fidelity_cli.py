from __future__ import annotations

from ipfd.cli import main


def test_audit_configuration_failure_returns_nonzero(tmp_path, capsys):
    exit_status = main(["audit", "--config", str(tmp_path / "missing.yaml")])

    assert exit_status == 2
    assert "IPFD_AUDIT_ERROR" in capsys.readouterr().err
