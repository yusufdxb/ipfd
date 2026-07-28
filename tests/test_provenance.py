import subprocess

import pytest

from ipfd.provenance import source_provenance


def test_source_provenance_records_commit_and_dirty_state(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        output = "a" * 40 + "\n" if "rev-parse" in command else " M file.py\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert source_provenance(tmp_path) == {
        "git_commit": "a" * 40,
        "git_dirty": True,
    }


def test_source_provenance_rejects_invalid_commit(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="not-a-commit\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="invalid commit"):
        source_provenance(tmp_path)
