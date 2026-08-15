from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ipfd.fidelity.audit import audit_configuration, create_adapter, run_audit
from ipfd.fidelity.config import AuditConfig, BranchState, load_config
from ipfd.fidelity.contracts import (
    ObservationRecord,
    ReplayAdapter,
    Snapshot,
    StepRecord,
    TrajectoryRecord,
)


class HiddenStateAdapter:
    def __init__(self):
        self.x = np.zeros(2)
        self.hidden = np.zeros(2)

    def reset(self, seed):
        self.x[:] = 0.0
        self.hidden[:] = 0.0

    def action(self, step, source, env_ids):
        return np.ones((2, 1))

    def capture(self, env_ids):
        return Snapshot(
            protocol="visible_only",
            values={"x": self.x[list(env_ids)].copy()},
            captured_components=("visible position",),
            unavailable_components=("hidden integrator",),
        )

    def restore(self, snapshot, env_ids):
        self.x[list(env_ids)] = snapshot.values["x"]
        self.hidden[list(env_ids)] = 0.0

    def observe(self, env_ids):
        ids = list(env_ids)
        return ObservationRecord(
            scene_state={"x": self.x[ids].copy()},
            policy_observations={"x": self.x[ids].copy()},
            task_state={},
            counters={"step": np.zeros(len(ids), dtype=int)},
            unavailable=("true hidden integrator state",),
        )

    def step(self, actions):
        applied = np.asarray(actions).copy()
        self.hidden += 0.01
        self.x += applied[:, 0] + self.hidden
        observed = self.observe((0, 1))
        return StepRecord(
            observation=observed,
            contact_state={"active": self.x > 100.0},
            task_outputs={"above_threshold": self.x > 2.025},
            terminated={"done": np.zeros(2, dtype=bool)},
            reward={"reward": self.x.copy()},
            applied_actions=applied,
        )

    def decision(self, record: TrajectoryRecord, name: str) -> bool:
        assert name == "above_threshold"
        return bool(record.steps[-1].task_outputs["above_threshold"][record.env_id])

    def provenance(self):
        return {
            "adapter": "hidden-state-test",
            "state_components_captured": ["visible position"],
            "state_components_unavailable": ["hidden integrator"],
        }


class SparseEvidenceAdapter(HiddenStateAdapter):
    def observe(self, env_ids):
        return ObservationRecord(
            scene_state={},
            policy_observations={},
            counters={"step": np.zeros(len(env_ids), dtype=int)},
            unavailable=("all physical state",),
        )

    def step(self, actions):
        return StepRecord(
            observation=self.observe((0, 1)),
            contact_state={},
        )

    def decision(self, record: TrajectoryRecord, name: str) -> bool:
        return True


class SemanticDivergenceAdapter(HiddenStateAdapter):
    def restore(self, snapshot, env_ids):
        super().restore(snapshot, env_ids)
        self.hidden[list(env_ids)] = self.hidden[0]

    def step(self, actions):
        applied = np.asarray(actions).copy()
        self.x += np.asarray(actions)[:, 0]
        observed = self.observe((0, 1))
        return StepRecord(
            observation=observed,
            contact_state={"active": np.zeros(2, dtype=bool)},
            task_outputs={"success": np.array([False, True])},
            terminated={"done": np.array([False, True])},
            reward={"reward": np.array([0.0, 1.0])},
            semantic={"mode": np.array([False, True])},
            applied_actions=applied,
        )

    def decision(self, record: TrajectoryRecord, name: str) -> bool:
        return True


class NondeterministicRerunAdapter(HiddenStateAdapter):
    def __init__(self):
        super().__init__()
        self.rerun = 0
        self.after_restore = False

    def reset(self, seed):
        self.rerun += 1
        self.after_restore = False
        return super().reset(seed)

    def restore(self, snapshot, env_ids):
        super().restore(snapshot, env_ids)
        self.after_restore = True

    def step(self, actions):
        record = super().step(actions)
        if not self.after_restore:
            return record
        self.x += self.rerun * 1.0e-4
        return StepRecord(
            observation=self.observe((0, 1)),
            contact_state=record.contact_state,
            task_outputs=record.task_outputs,
            terminated=record.terminated,
            reward=record.reward,
            applied_actions=record.applied_actions,
        )


def make_config(tmp_path: Path) -> AuditConfig:
    return AuditConfig(
        source_path=tmp_path / "audit.yaml",
        adapter={"kind": "fake", "simulator": "FakeSim"},
        simulator_version="1",
        environment="hidden-state",
        task="threshold",
        snapshot_protocol="visible_only",
        branch_states=tuple(
            BranchState(id=f"seed-{seed}", step=1, seed=seed, cluster=str(seed))
            for seed in (1, 2, 3)
        ),
        horizons=(1, 5, 10, 30, 90),
        continuation_mode="exact_action",
        action_source="recorded",
        decision_functions=("above_threshold",),
        tolerances={"default": {"absolute": 0.0, "relative": 0.0}},
        independent_cluster_key="seed",
        output_directory=tmp_path / "out",
        minimum_independent_clusters=3,
        reduction={"enabled": True, "max_trials": 5},
        regression=None,
        raw={},
    )


def test_audit_separates_restore_equality_trajectory_and_decision(tmp_path):
    result = audit_configuration(HiddenStateAdapter(), make_config(tmp_path))
    horizon_one = result["configurations"][0]
    assert horizon_one["levels"]["L0"]["passed"] is True
    assert horizon_one["levels"]["L1"]["passed"] is False
    assert horizon_one["levels"]["L2"]["first_numerical_divergence"] == 1
    assert horizon_one["levels"]["L3"]["decision_disagreement"] is True
    assert horizon_one["result"] == "UNSUPPORTED"
    assert result["minimal_reproducer"]["failure_kind"] == "L3"
    assert result["minimal_reproducer"]["reduction"]["minimal"]["horizon"] == 1


def test_missing_physical_channels_are_insufficient_evidence(tmp_path):
    result = audit_configuration(SparseEvidenceAdapter(), make_config(tmp_path))
    record = result["records"][0]

    assert record["levels"]["L0"]["passed"] is None
    assert record["levels"]["L1"]["passed"] is None
    assert record["levels"]["L2"]["passed"] is None
    assert result["configurations"][0]["result"] == "INSUFFICIENT_EVIDENCE"


def test_l2_tracks_task_termination_reward_and_semantic_divergence(tmp_path):
    result = audit_configuration(SemanticDivergenceAdapter(), make_config(tmp_path))
    l2 = result["records"][0]["levels"]["L2"]

    assert l2["first_numerical_divergence"] is None
    assert l2["first_task_output_divergence"] == 1
    assert l2["first_termination_divergence"] == 1
    assert l2["first_reward_divergence"] == 1
    assert l2["first_semantic_event_divergence"] == 1
    assert l2["first_divergence"] == 1
    assert l2["passed"] is False


def test_horizon_frontier_rejects_noncoherent_trajectory_reruns(tmp_path):
    with pytest.raises(RuntimeError, match="different paired trajectory prefixes"):
        audit_configuration(NondeterministicRerunAdapter(), make_config(tmp_path))


def test_live_audit_rejects_declared_simulator_version_mismatch(tmp_path):
    adapter = HiddenStateAdapter()
    adapter.provenance = lambda: {
        "adapter": "hidden-state-test",
        "simulator_version": "2",
    }

    with pytest.raises(RuntimeError, match="declared simulator_version"):
        run_audit(make_config(tmp_path), adapter=adapter)


def test_live_audit_can_bind_scope_to_installed_simulator_version(tmp_path):
    config = replace(make_config(tmp_path), simulator_version="installed")
    config.source_path.write_text("schema_version: 1\n", encoding="utf-8")
    adapter = HiddenStateAdapter()
    adapter.provenance = lambda: {
        "adapter": "hidden-state-test",
        "simulator_version": "1",
    }

    result = run_audit(config, adapter=adapter)

    assert result["configurations"][0]["scope"]["simulator_version"] == "1"


def test_config_driven_audit_can_load_a_trusted_python_factory():
    config = load_config(Path("examples/custom_adapter.yaml"))

    adapter = create_adapter(config)

    assert isinstance(adapter, ReplayAdapter)
    assert adapter.provenance()["environment"] == "point-mass-v1"
