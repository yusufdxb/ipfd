from pathlib import Path

import numpy as np
import pytest
import yaml

from ipfd.fidelity.comparison import compare_values, extract_env, maximum_error
from ipfd.fidelity.config import load_config
from ipfd.fidelity.contracts import ContractVerdict, Snapshot, to_builtin
from ipfd.fidelity.minimizer import minimize_failure


def test_contract_verdicts_are_the_only_public_statuses():
    assert {item.value for item in ContractVerdict} == {
        "SUPPORTED",
        "UNSUPPORTED",
        "INSUFFICIENT_EVIDENCE",
    }


def test_snapshot_serialization_keeps_missing_state_visible():
    snapshot = Snapshot(
        protocol="scene_only",
        values={"q": np.array([[1.0, 2.0]])},
        captured_components=("qpos",),
        unavailable_components=("solver warm-start",),
    )
    assert snapshot.to_dict()["values"] == {"q": [[1.0, 2.0]]}
    assert snapshot.to_dict()["unavailable_components"] == ["solver warm-start"]
    assert to_builtin(np.float64(1.25)) == 1.25


def test_comparison_preserves_raw_errors_and_environment_selection():
    values = {"q": np.array([[1.0, 2.0], [1.0, 2.01]])}
    result = compare_values(
        extract_env(values, 0),
        extract_env(values, 1),
        absolute=0.001,
    )
    assert result["passed"] is False
    assert result["maximum_absolute_error"] == pytest.approx(0.01)
    assert result["fields"]["q"]["threshold"] == 0.001


def test_comparison_supports_unit_specific_field_tolerances():
    result = compare_values(
        {"qpos": np.array([0.0]), "qvel": np.array([0.0])},
        {"qpos": np.array([2.0e-4]), "qvel": np.array([5.0e-3])},
        absolute=0.0,
        field_tolerances={
            "qpos": {"absolute": 1.5e-4, "unit": "m"},
            "qvel": {"absolute": 1.0e-2, "unit": "m/s"},
        },
    )

    assert result["passed"] is False
    assert result["fields"]["qpos"]["within_tolerance"] is False
    assert result["fields"]["qvel"]["within_tolerance"] is True
    assert result["fields"]["qvel"]["absolute_tolerance"] == 1.0e-2
    assert result["fields"]["qpos"]["unit"] == "m"


def test_boolean_semantics_are_exact_even_under_numeric_tolerance():
    result = compare_values(
        {"terminated": np.array([False])},
        {"terminated": np.array([True])},
        absolute=1.0,
    )

    assert result["passed"] is False
    assert result["fields"]["terminated"]["comparison"] == "exact_boolean"
    assert result["fields"]["terminated"]["reference_at_first_difference"] is False
    assert result["fields"]["terminated"]["candidate_at_first_difference"] is True


def test_comparison_handles_nested_semantic_and_invalid_numeric_values():
    nested = {"tuple": (np.array([[1], [2]]),), "list": [np.array([[3], [4]])]}
    selected = extract_env(nested, 1)
    assert selected["tuple"][0].tolist() == [2]
    assert selected["list"][0].tolist() == [4]
    assert extract_env("scalar", 0) == "scalar"
    with pytest.raises(IndexError, match="environment index"):
        extract_env(np.zeros((1, 2)), 2)

    semantic = compare_values(
        {"mode": "contact", "missing": 1},
        {"mode": "free", "extra": 1},
        absolute=0.0,
    )
    assert semantic["passed"] is False
    assert semantic["fields"]["mode"]["semantic_equal"] is False
    assert semantic["missing_in_candidate"] == ["missing"]
    assert semantic["missing_in_reference"] == ["extra"]

    shape = compare_values(np.zeros(2), np.zeros(3), absolute=0.0)
    assert shape["fields"]["value"]["reason"] == "shape_mismatch"
    nonfinite = compare_values(np.array([np.nan]), np.array([np.nan]), absolute=0.0)
    assert nonfinite["fields"]["value"]["reason"] == "nonfinite_value"
    empty = compare_values(np.array([]), np.array([]), absolute=0.0)
    assert empty["passed"] is True
    assert maximum_error({}) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        compare_values(1.0, 1.0, absolute=-1.0)


def test_config_requires_declared_tolerance_and_all_contract_horizons(tmp_path: Path):
    path = tmp_path / "audit.yaml"
    path.write_text(
        """schema_version: 1
adapter: {kind: fake}
simulator_version: test
environment: env
task: task
snapshot_protocol: full
branch_states: [{step: 0, seed: 1}]
horizons: [1, 5, 10]
continuation_mode: exact_action
action_source: recorded
decision_functions: [success]
tolerances: {default: {absolute: 0.0, relative: 0.0}}
independent_cluster_key: seed
output_directory: out
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="required control-step horizons"):
        load_config(path)


def _valid_config_mapping() -> dict:
    return {
        "schema_version": 1,
        "adapter": {"kind": "fake"},
        "simulator_version": "1",
        "environment": "env",
        "task": "task",
        "snapshot_protocol": "full",
        "branch_states": [{"step": 0, "seed": 1}],
        "horizons": [1, 5, 10, 30, 90],
        "continuation_mode": "exact_action",
        "action_source": "recorded",
        "decision_functions": ["success"],
        "tolerances": {"default": {"absolute": 0.0, "relative": 0.0}},
        "independent_cluster_key": "seed",
        "output_directory": "out",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version=2), "schema_version"),
        (lambda value: value.update(adapter=[]), "adapter must be"),
        (lambda value: value["adapter"].update(kind=""), "kind must be"),
        (lambda value: value.update(decision_functions=[]), "decision_functions"),
        (lambda value: value.update(branch_states=[]), "branch_states"),
        (lambda value: value.update(branch_states=[1]), "must be a mapping"),
        (lambda value: value.update(branch_states=[{"step": -1, "seed": 1}]), "step must be"),
        (lambda value: value.update(branch_states=[{"step": 0, "seed": True}]), "seed must be"),
        (lambda value: value.update(horizons=[]), "horizons must be"),
        (lambda value: value.update(horizons=[1, 5, 10, 30, True]), "every horizon"),
        (lambda value: value.update(tolerances=[]), "tolerances must be"),
        (lambda value: value.update(tolerances={"default": 1}), "must be a mapping"),
        (
            lambda value: value.update(tolerances={"default": {"absolute": -1}}),
            "non-negative",
        ),
        (lambda value: value.update(tolerances={"state": {}}), "default is required"),
        (lambda value: value.update(minimum_independent_clusters=0), "minimum_independent"),
    ],
)
def test_config_validation_is_fail_closed(tmp_path, mutation, message):
    value = _valid_config_mapping()
    mutation(value)
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_config_rejects_non_mapping_and_invalid_yaml(tmp_path):
    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(non_mapping)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("mapping: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_config(invalid)


def test_config_preserves_absolute_output_directory(tmp_path):
    value = _valid_config_mapping()
    value["output_directory"] = str((tmp_path / "absolute-output").resolve())
    path = tmp_path / "absolute.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    assert load_config(path).output_directory == tmp_path / "absolute-output"


def test_minimizer_reduces_horizon_and_reports_unsupported_dimensions():
    def evaluate(candidate):
        return {"fails": candidate["horizon"] >= 3, "record": {"horizon": candidate["horizon"]}}

    result = minimize_failure(
        {
            "branch_id": "b",
            "branch_step": 10,
            "seed": 1,
            "cluster": "1",
            "horizon": 8,
            "action_sequence_length": 8,
            "decision_name": "success",
            "failure_kind": "L3",
            "fails": True,
        },
        evaluate,
        decision_names=("success",),
        max_trials=20,
    )
    assert result["minimal"]["horizon"] == 3
    assert result["minimal"]["action_sequence_length"] == 3
    assert {item["dimension"] for item in result["attempts_not_supported"]} == {
        "number_of_active_entities",
        "disturbance_schedule",
        "snapshot_state_components",
    }


def test_minimizer_exercises_all_supported_reduction_dimensions():
    def evaluate(candidate):
        return {
            "fails": True,
            "record": {"horizon": candidate["horizon"]},
            "captured_snapshot": {"state": 1},
            "identical_actions": [[0.0]],
        }

    result = minimize_failure(
        {
            "branch_id": "b",
            "branch_step": 5,
            "seed": 1,
            "cluster": "1",
            "horizon": 2,
            "action_sequence_length": 2,
            "decision_name": "a",
            "failure_kind": "L3",
            "fails": True,
        },
        evaluate,
        branch_steps=(0,),
        decision_names=("a", "b"),
        adapter_dimensions={
            "active_entities": (1,),
            "disturbance_schedule": ("none",),
            "snapshot_components": (("qpos",),),
        },
        max_trials=10,
    )
    assert result["minimal"]["horizon"] == 1
    assert result["minimal"]["branch_step"] == 0
    assert result["minimal"]["decision_name"] == "b"
    assert result["attempts_not_supported"] == []
    assert result["minimal_evidence"]["captured_snapshot"] == {"state": 1}
    assert {trial["dimension"] for trial in result["trials"]} >= {
        "continuation_horizon_and_action_prefix",
        "branch_timing",
        "decision_predicate",
        "number_of_active_entities",
        "disturbance_schedule",
        "snapshot_state_components",
    }


def test_minimizer_rejects_nonfailure_and_invalid_budget():
    with pytest.raises(ValueError, match="at least 1"):
        minimize_failure({"fails": True}, lambda value: value, max_trials=0)
    with pytest.raises(ValueError, match="must be failing"):
        minimize_failure({"fails": False}, lambda value: value)
