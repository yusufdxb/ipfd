import json
from copy import deepcopy

import pytest

from ipfd.evidence_gate import EvidenceCriteria, evaluate_evidence
from ipfd.evidence_gate_cli import main as evidence_gate_main

CHECKPOINT = "a" * 64
TASK = "Isaac-Lift-Cube-Franka-v0"
RUNTIME = {"isaaclab": "4.5.22", "isaacsim": "6.0.0", "torch": "2.7.0"}
SOFTWARE = {
    "ipfd_version": "1.1.0.dev0",
    "git_commit": "b" * 40,
    "git_dirty": False,
}


def good():
    competence = {
        "schema": "ipfd.competence.v1",
        "status": "complete",
        "success_rate": 0.9,
        "n_episodes": 64,
        "sustain_steps": 10,
        "success_definition": "sustained_final_lift_v1",
        "checkpoint_sha256": CHECKPOINT,
        "task": TASK,
        "runtime": dict(RUNTIME),
        "software": dict(SOFTWARE),
    }
    runs = []
    for seed in range(5):
        common = {
            "status": "complete",
            "seed": seed,
            "fault_injection_triggered": True,
            "checkpoint_sha256": CHECKPOINT,
            "task": TASK,
            "runtime": dict(RUNTIME),
            "software": dict(SOFTWARE),
            "recovery_predicate": "physical_pick_lift_v1",
            "probe_repeats": 3,
            "probe_stride": 8,
            "probe_budget": 90,
            "probe_min_false_fraction": 0.8,
            "reset_boundary_primary_pose_delta_m": 0.0,
        }
        runs.append(
            common
            | {
                "expected_outcome": "ponr",
                "disturbance_onset": 50,
                "t_ponr": 56,
                "rollout_sha256": f"{2 * seed:064x}",
                "raw_probe_verdicts": {
                    str(step): [True, True, True] for step in range(0, 56, 8)
                }
                | {"56": [False, False, False]},
            }
        )
        runs.append(
            common
            | {
                "expected_outcome": "no_ponr",
                "disturbance_onset": 50,
                "t_ponr": None,
                "rollout_sha256": f"{2 * seed + 1:064x}",
                "raw_probe_verdicts": {
                    str(step): [True, True, True] for step in range(0, 57, 8)
                },
            }
        )
    multiseed = {
        "schema": "ipfd.multiseed.v1",
        "status": "complete",
        "runs": runs,
    }
    actionability = {
        "schema": "ipfd.actionability.v1",
        "status": "complete",
        "source": "isaac_lab",
        "checkpoint_sha256": CHECKPOINT,
        "task": TASK,
        "runtime": dict(RUNTIME),
        "software": dict(SOFTWARE),
        "cases": [
            {
                "case_id": f"case-{index}",
                "seed": index,
                "rollout_sha256": f"{index:064x}",
                "expected_relation": relation,
                "alarm_relation": relation,
            }
            for index, relation in enumerate(
                [
                    "no_alarm",
                    "pre_disturbance",
                    "definitely_actionable",
                    "ambiguous_within_ponr_interval",
                    "too_late",
                ]
                * 4
            )
        ],
    }
    return competence, multiseed, actionability


def test_gate_passes_complete_auditable_evidence():
    result = evaluate_evidence(
        *good(),
        criteria=EvidenceCriteria(required_git_commit="b" * 40),
    )
    assert result.passed
    assert result.metrics["n_seeds"] == 5
    assert result.metrics["actionability_cases"] == 20


def test_gate_fails_missing_evidence():
    result = evaluate_evidence(None, *good()[1:])
    assert not result.passed and "competence" in result.missing


def test_gate_fails_weak_competence():
    artifacts = list(good())
    artifacts[0]["success_rate"] = 0.2
    result = evaluate_evidence(*artifacts)
    assert not result.passed and not result.checks["competence"]


def test_gate_rejects_transient_or_under_sustained_competence():
    artifacts = list(good())
    artifacts[0]["success_definition"] = "max_lift_once"
    artifacts[0]["sustain_steps"] = 1
    result = evaluate_evidence(*artifacts)
    assert not result.passed
    assert any("sustained_final_lift_v1" in error for error in result.errors)


def test_gate_rejects_out_of_range_aggregate():
    artifacts = list(good())
    artifacts[0]["success_rate"] = 1.2
    result = evaluate_evidence(*artifacts)
    assert not result.passed
    assert any("success_rate" in error for error in result.errors)


def test_gate_rejects_underpowered_multiseed_and_actionability():
    competence, multiseed, actionability = deepcopy(good())
    multiseed["runs"] = multiseed["runs"][:2]
    actionability["cases"] = actionability["cases"][:4]
    result = evaluate_evidence(competence, multiseed, actionability)
    assert not result.passed
    assert not result.checks["multiseed"]
    assert not result.checks["actionability"]


def test_gate_rejects_legacy_predicate_and_missing_raw_probes():
    competence, multiseed, actionability = deepcopy(good())
    multiseed["runs"][0]["recovery_predicate"] = "height_only_legacy"
    multiseed["runs"][1]["raw_probe_verdicts"] = {}
    result = evaluate_evidence(competence, multiseed, actionability)
    assert not result.passed
    assert any("physical predicate" in error for error in result.errors)
    assert any("raw_probe_verdicts" in error for error in result.errors)


def test_gate_rejects_t_ponr_inconsistent_with_raw_probes_and_duplicate_outcome():
    competence, multiseed, actionability = deepcopy(good())
    multiseed["runs"][0]["t_ponr"] = 50
    duplicate = deepcopy(multiseed["runs"][2])
    duplicate["rollout_sha256"] = "f" * 64
    multiseed["runs"].append(duplicate)
    result = evaluate_evidence(competence, multiseed, actionability)
    assert not result.passed
    assert any("raw-probe-derived" in error for error in result.errors)
    assert any("same seed and expected_outcome" in error for error in result.errors)


def test_gate_rejects_mismatched_runtime_and_unlocalized_ponr():
    competence, multiseed, actionability = deepcopy(good())
    multiseed["runs"][0]["runtime"]["torch"] = "other"
    multiseed["runs"][2]["t_ponr"] = 99
    result = evaluate_evidence(competence, multiseed, actionability)
    assert not result.passed
    assert any("runtime must match" in error for error in result.errors)
    assert any("within 10 steps" in error for error in result.errors)


def test_gate_rejects_dirty_or_mismatched_source_provenance():
    competence, multiseed, actionability = deepcopy(good())
    competence["software"]["git_dirty"] = True
    multiseed["runs"][0]["software"]["git_commit"] = "c" * 40
    result = evaluate_evidence(competence, multiseed, actionability)
    assert not result.passed
    assert any("git_dirty must be false" in error for error in result.errors)
    assert any("software must match" in error for error in result.errors)


def test_gate_rejects_actionability_without_relation_coverage():
    competence, multiseed, actionability = deepcopy(good())
    for case in actionability["cases"]:
        case["expected_relation"] = "definitely_actionable"
        case["alarm_relation"] = "definitely_actionable"
    result = evaluate_evidence(competence, multiseed, actionability)
    assert not result.passed
    assert any("cover required expected relations" in error for error in result.errors)


def test_gate_rejects_synthetic_actionability_contract():
    competence, multiseed, actionability = deepcopy(good())
    actionability["source"] = "synthetic_contract"
    result = evaluate_evidence(competence, multiseed, actionability)
    assert not result.passed
    assert any("synthetic contract cases are not evidence" in error for error in result.errors)


def test_criteria_reject_invalid_thresholds():
    with pytest.raises(ValueError):
        EvidenceCriteria(min_success_rate=float("nan"))
    with pytest.raises(ValueError):
        EvidenceCriteria(min_seeds=0)
    with pytest.raises(ValueError):
        EvidenceCriteria(min_success_rate="0.8")
    with pytest.raises(ValueError):
        EvidenceCriteria(max_reset_boundary_delta_m=float("inf"))
    with pytest.raises(ValueError):
        EvidenceCriteria(min_probe_false_fraction=0.49)
    with pytest.raises(ValueError):
        EvidenceCriteria(required_git_commit="not-a-commit")


def test_evidence_gate_cli_writes_passing_report(tmp_path, capsys):
    names = ("competence", "multiseed", "actionability")
    paths = {}
    for name, artifact in zip(names, good(), strict=True):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        paths[name] = path
    report = tmp_path / "nested" / "report.json"

    assert evidence_gate_main(
        [
            "--competence",
            str(paths["competence"]),
            "--multiseed",
            str(paths["multiseed"]),
            "--actionability",
            str(paths["actionability"]),
            "--expected-git-commit",
            "b" * 40,
            "--report",
            str(report),
        ]
    ) == 0
    assert json.loads(report.read_text())["passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_evidence_gate_cli_fails_closed_on_missing_file(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    assert evidence_gate_main(
        [
            "--competence",
            str(missing),
            "--multiseed",
            str(missing),
            "--actionability",
            str(missing),
        ]
    ) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is False
    assert any("could not be read" in error for error in result["errors"])


def test_evidence_gate_cli_rejects_invalid_criteria(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        evidence_gate_main(
            [
                "--competence",
                str(tmp_path / "a"),
                "--multiseed",
                str(tmp_path / "b"),
                "--actionability",
                str(tmp_path / "c"),
                "--min-seeds",
                "0",
            ]
        )
