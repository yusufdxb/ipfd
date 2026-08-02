from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ipfd.cli import main
from ipfd.fidelity.archive import run_archived_isaac_audit
from ipfd.fidelity.config import load_config
from ipfd.fidelity.matrix import run_audit_matrix
from ipfd.fidelity.reporting import _overall, write_audit_outputs


def _raw_record(horizon: int) -> dict:
    mismatch = {
        "robot_root_pose": 0.0,
        "robot_root_velocity": 0.0,
        "joint_position": 0.0,
        "joint_velocity": 0.0,
        "object_pose": 0.0,
        "object_velocity": 0.0,
        "observation": 0.0,
        "command": 0.0,
        "targets": 0.0,
        "contact_force": 0.0,
        "episode_length": 0.0,
        "sim_step_counter": 0.0,
        "common_step_counter": 0.0,
    }
    fails = horizon == 90
    return {
        "branch_id": f"seed-7-horizon-{horizon}",
        "branch_step": 20,
        "base_seed": 7,
        "horizon": horizon,
        "protocol": "expanded_runtime_state",
        "continuation": "exact_action",
        "predicate": "task_success",
        "disturbance": "fixture",
        "phase": "contact",
        "immediate_mismatch": mismatch,
        "first_step_mismatch": mismatch,
        "first_numerical_divergence_step": 2 if fails else None,
        "first_observation_divergence_step": 2 if fails else None,
        "first_contact_divergence_step": None,
        "first_action_divergence_step": None,
        "maximum_trajectory_error": 0.25 if fails else 0.0,
        "terminal_trajectory_error": 0.25 if fails else 0.0,
        "reference_decision": True,
        "candidate_decision": not fails,
        "decision_match": not fails,
    }


def _archive_config(tmp_path: Path) -> Path:
    records = tmp_path / "archive.jsonl"
    records.write_text(
        "".join(json.dumps(_raw_record(horizon)) + "\n" for horizon in (1, 5, 10, 30, 90)),
        encoding="utf-8",
    )
    digest = hashlib.sha256(records.read_bytes()).hexdigest()
    config = {
        "schema_version": 1,
        "adapter": {
            "kind": "isaac_lab_archive",
            "records_path": records.name,
            "expected_sha256": digest,
            "archival_commit": "fixture",
        },
        "simulator_version": "fixture",
        "environment": "archive-fixture",
        "task": "archive-fixture",
        "snapshot_protocol": "expanded_runtime_state",
        "branch_states": [{"id": "seed-7", "step": 0, "seed": 7, "cluster": "seed-7"}],
        "horizons": [1, 5, 10, 30, 90],
        "continuation_mode": "exact_action",
        "action_source": "archived_actions",
        "decision_functions": ["task_success"],
        "tolerances": {"default": {"absolute": 0.0, "relative": 0.0}},
        "independent_cluster_key": "seed",
        "minimum_independent_clusters": 1,
        "output_directory": "audit-output",
        "reduction": {"enabled": True},
    }
    path = tmp_path / "archive.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _matrix_config(tmp_path: Path, child: Path) -> Path:
    config = {
        "schema_version": 1,
        "adapter": {"kind": "matrix", "cases": [child.name]},
        "simulator_version": "mixed",
        "environment": "matrix",
        "task": "matrix",
        "snapshot_protocol": "mixed",
        "branch_states": [{"step": 0, "seed": 0}],
        "horizons": [1, 5, 10, 30, 90],
        "continuation_mode": "per-child",
        "action_source": "per-child",
        "decision_functions": ["per_child"],
        "tolerances": {"default": {"absolute": 0.0, "relative": 0.0}},
        "independent_cluster_key": "per-child",
        "output_directory": "matrix-output",
    }
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_archive_reporting_matrix_and_cli_end_to_end(tmp_path, capsys):
    child_path = _archive_config(tmp_path)
    child = load_config(child_path)
    result = run_archived_isaac_audit(child)

    assert len(result["records"]) == 5
    assert result["configurations"][-1]["levels"]["L3"]["decision_disagreement"] is True
    assert result["minimal_reproducer"]["status"] == "ARCHIVED_RECORD_REDUCED_BUT_NOT_SELF_CONTAINED"

    outputs = write_audit_outputs(result, child)
    assert outputs["summary"]["overall_result"] == "UNSUPPORTED"
    assert set(outputs["summary"]["artifacts"]) >= {
        "per_branch_records.jsonl",
        "fidelity_contract.json",
        "provenance.json",
        "REPORT.md",
        "divergence.svg",
        "minimal_reproducer.json",
    }
    assert _overall([{"result": "SUPPORTED"}]) == "SUPPORTED"
    assert _overall([{"result": "INSUFFICIENT_EVIDENCE"}]) == "INSUFFICIENT_EVIDENCE"

    result["records"][0]["levels"]["L2"]["divergence_growth_curve"] = [
        {"step": 1, "state_max_abs": 0.1}
    ]
    result["minimal_reproducer"] = None
    (child.output_directory / "regression_report.json").write_text("stale", encoding="utf-8")
    rewritten = write_audit_outputs(result, child)
    assert rewritten["summary"]["failure_reproducer_produced"] is False
    assert not (child.output_directory / "minimal_reproducer.json").exists()
    assert not (child.output_directory / "regression_report.json").exists()
    assert "max raw state error 0.1" in (child.output_directory / "divergence.svg").read_text()

    matrix_path = _matrix_config(tmp_path, child_path)
    matrix = run_audit_matrix(load_config(matrix_path))
    assert matrix["result"] == "COMPLETED_WITH_UNSUPPORTED_SCOPES"
    assert matrix["cases"][0]["overall_result"] == "UNSUPPORTED"
    assert not matrix["cases"][0]["output_directory"].startswith("/")

    assert main(["audit", "--config", str(matrix_path)]) == 0
    assert "COMPLETED_WITH_UNSUPPORTED_SCOPES" in capsys.readouterr().out
