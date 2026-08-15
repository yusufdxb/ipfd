"""Small complete IPFD adapter that can be copied into another simulator package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ipfd import audit, check_adapter
from ipfd.fidelity.contracts import ObservationRecord, Snapshot, StepRecord, TrajectoryRecord


class PointMassAdapter:
    """Two deterministic point masses with a complete snapshot contract."""

    snapshot_protocol = "full_state"

    def __init__(self) -> None:
        self.position = np.zeros(2, dtype=np.float64)
        self.control_steps = np.zeros(2, dtype=np.int64)

    def reset(self, seed: int) -> ObservationRecord:
        del seed
        self.position[:] = 0.0
        self.control_steps[:] = 0
        return self.observe((0, 1))

    def action(self, step: int, source: Any, env_ids: Sequence[int]) -> np.ndarray:
        del step, source
        return np.full((len(env_ids), 1), 0.25, dtype=np.float64)

    def capture(self, env_ids: Sequence[int]) -> Snapshot:
        ids = np.asarray(env_ids, dtype=int)
        return Snapshot(
            protocol=self.snapshot_protocol,
            values={
                "position": self.position[ids].copy(),
                "control_steps": self.control_steps[ids].copy(),
            },
            captured_components=("position", "control_step_counter"),
            unavailable_components=("renderer state (not used)",),
            metadata={"environment": "point-mass-v1"},
        )

    def restore(self, snapshot: Snapshot, env_ids: Sequence[int]) -> None:
        if snapshot.protocol != self.snapshot_protocol:
            raise ValueError("snapshot protocol mismatch")
        if snapshot.metadata.get("environment") != "point-mass-v1":
            raise ValueError("snapshot environment mismatch")
        ids = np.asarray(env_ids, dtype=int)
        source_position = np.asarray(snapshot.values["position"])
        source_steps = np.asarray(snapshot.values["control_steps"])
        for row, env_id in enumerate(ids):
            source_row = 0 if len(source_position) == 1 else row
            self.position[env_id] = source_position[source_row]
            self.control_steps[env_id] = source_steps[source_row]

    def observe(self, env_ids: Sequence[int]) -> ObservationRecord:
        ids = np.asarray(env_ids, dtype=int)
        positions = self.position[ids].copy()
        return ObservationRecord(
            scene_state={"position": positions},
            policy_observations={"position": positions.copy()},
            counters={"control_steps": self.control_steps[ids].copy()},
            unavailable=("renderer state (not used)",),
        )

    def step(self, actions: Any) -> StepRecord:
        applied = np.asarray(actions, dtype=np.float64).copy()
        if applied.shape != (2, 1):
            raise ValueError("actions must have shape (2, 1)")
        self.position += applied[:, 0]
        self.control_steps += 1
        return StepRecord(
            observation=self.observe((0, 1)),
            contact_state={"collision": np.zeros(2, dtype=bool)},
            task_outputs={"threshold_reached": self.position >= 1.0},
            terminated={"done": np.zeros(2, dtype=bool)},
            reward={"position": self.position.copy()},
            applied_actions=applied,
        )

    def decision(self, record: TrajectoryRecord, name: str) -> bool:
        if name != "threshold_reached":
            raise KeyError(name)
        value = record.steps[-1].task_outputs[name]
        return bool(np.asarray(value)[record.env_id])

    def provenance(self) -> Mapping[str, object]:
        return {
            "adapter": "PointMassAdapter",
            "simulator": "documented deterministic example",
            "simulator_version": "1",
            "environment": "point-mass-v1",
            "snapshot_protocol": self.snapshot_protocol,
            "state_components_captured": ["position", "control_step_counter"],
            "state_components_unavailable": ["renderer state (not used)"],
            "task_state_captured": [],
            "controller_or_policy_history_captured": False,
            "random_state_handling": "no randomness",
            "solver_state_availability": "no solver state",
            "sensor_refresh_behavior": "no sensors",
            "unsupported_restoration_claims": ["renderer fidelity"],
            "decision_contracts": {
                "threshold_reached": {
                    "definition": "terminal position is at least 1.0",
                    "threshold": 1.0,
                }
            },
        }


def make_adapter() -> PointMassAdapter:
    return PointMassAdapter()


if __name__ == "__main__":
    conformance = check_adapter(make_adapter(), decision="threshold_reached")
    result = audit(
        adapter=make_adapter(),
        protocol="full_state",
        branch_step=2,
        horizons=[1, 5, 10],
        continuation="exact_action",
        decision="threshold_reached",
    )
    print(f"adapter-check: {conformance.verdict.value}")
    print(f"audit:         {result.verdict.value}")
