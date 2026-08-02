from pathlib import Path

import numpy as np
import pytest

from ipfd.fidelity.audit import audit_configuration, run_audit
from ipfd.fidelity.config import AuditConfig, BranchState
from ipfd.fidelity.contracts import ObservationRecord, Snapshot, StepRecord, TrajectoryRecord


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
        self.hidden += 0.01
        self.x += np.asarray(actions)[:, 0] + self.hidden
        observed = self.observe((0, 1))
        return StepRecord(
            observation=observed,
            contact_state={"active": self.x > 100.0},
            task_outputs={"above_threshold": self.x > 2.025},
            terminated={"done": np.zeros(2, dtype=bool)},
            reward={"reward": self.x.copy()},
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


def test_live_audit_rejects_declared_simulator_version_mismatch(tmp_path):
    adapter = HiddenStateAdapter()
    adapter.provenance = lambda: {
        "adapter": "hidden-state-test",
        "simulator_version": "2",
    }

    with pytest.raises(RuntimeError, match="declared simulator_version"):
        run_audit(make_config(tmp_path), adapter=adapter)
