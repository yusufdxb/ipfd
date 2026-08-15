from __future__ import annotations

import numpy as np
import pytest

from ipfd import DecisionContract, audit, check_adapter
from ipfd.adapter_check import AdapterCheckStatus
from ipfd.fidelity.contracts import ContractVerdict, ObservationRecord, Snapshot, StepRecord, TrajectoryRecord


class PairedAdapter:
    snapshot_protocol = "visible_only"

    def __init__(self) -> None:
        self.x = np.zeros(2)
        self.hidden = np.zeros(2)
        self.counters = np.zeros(2, dtype=np.int64)

    def reset(self, seed: int) -> ObservationRecord:
        self.x[:] = 0.0
        self.hidden[:] = 0.0
        self.counters[:] = 0
        return self.observe((0, 1))

    def action(self, step: int, source: object, env_ids: tuple[int, ...]) -> np.ndarray:
        del step, source
        return np.ones((len(env_ids), 1))

    def capture(self, env_ids: tuple[int, ...]) -> Snapshot:
        ids = list(env_ids)
        return Snapshot(
            protocol=self.snapshot_protocol,
            values={
                "x": self.x[ids].copy(),
                "counters": self.counters[ids].copy(),
            },
            captured_components=("x", "counter"),
            unavailable_components=("hidden integrator",),
            metadata={"environment": "paired-test"},
        )

    def restore(self, snapshot: Snapshot, env_ids: tuple[int, ...]) -> None:
        if snapshot.protocol != self.snapshot_protocol:
            raise ValueError("protocol mismatch")
        ids = list(env_ids)
        self.x[ids] = snapshot.values["x"]
        self.counters[ids] = snapshot.values["counters"]
        self.hidden[ids] = 0.0

    def observe(self, env_ids: tuple[int, ...]) -> ObservationRecord:
        ids = list(env_ids)
        return ObservationRecord(
            scene_state={"x": self.x[ids].copy()},
            policy_observations={"x": self.x[ids].copy()},
            counters={"control_steps": self.counters[ids].copy()},
            unavailable=("hidden integrator",),
        )

    def step(self, actions: object) -> StepRecord:
        values = np.asarray(actions)
        self.hidden += 0.01
        self.x += values[:, 0] + self.hidden
        self.counters += 1
        observation = self.observe((0, 1))
        return StepRecord(
            observation=observation,
            contact_state={"active": np.zeros(2, dtype=bool)},
            task_outputs={"above_threshold": self.x > 2.025},
            terminated={"done": np.zeros(2, dtype=bool)},
            reward={"value": self.x.copy()},
            applied_actions=values.copy(),
        )

    def decision(self, record: TrajectoryRecord, name: str) -> bool:
        if name != "above_threshold":
            raise KeyError(name)
        return bool(record.steps[-1].task_outputs[name][record.env_id])

    def provenance(self) -> dict[str, object]:
        return {
            "adapter": "PairedAdapter",
            "simulator": "TestSim",
            "simulator_version": "1",
            "environment": "paired-test",
            "snapshot_protocol": self.snapshot_protocol,
            "state_components_captured": ["x", "counter"],
            "state_components_unavailable": ["hidden integrator"],
            "task_state_captured": [],
            "controller_or_policy_history_captured": False,
            "random_state_handling": "seeded reset",
            "solver_state_availability": "hidden integrator unavailable",
            "sensor_refresh_behavior": "no sensors",
            "unsupported_restoration_claims": ["hidden integrator fidelity"],
        }


def test_public_audit_exposes_levels_frontier_and_custom_decision() -> None:
    contract = DecisionContract(
        name="above_threshold",
        evaluator=lambda record: bool(
            record.steps[-1].task_outputs["above_threshold"][record.env_id]
        ),
        description="terminal x crosses the declared threshold",
        parameters={"threshold": 2.025},
        true_label="above threshold",
        false_label="below threshold",
    )

    result = audit(
        adapter=PairedAdapter(),
        protocol="visible_only",
        branch_step=1,
        horizons=[1, 5],
        continuation="exact_action",
        decision=contract,
        metadata={"task": "threshold"},
    )

    assert result.verdict is ContractVerdict.UNSUPPORTED
    assert result.levels[1]["L0"]["passed"] is True
    assert result.levels[1]["L1"]["passed"] is False
    assert result.levels[1]["L2"]["first_numerical_divergence"] == 1
    assert result.levels[1]["L3"]["decision_disagreement"] is True
    assert result.fidelity_frontier["first_untrusted_horizon"] == 1
    result_data = result.to_dict()
    assert result_data["comparison_policy"]["default_is_exact_comparison"] is True
    decision = result_data["records"][0]["levels"]["L3"]["decisions"]["above_threshold"]
    assert decision["direction"] == "REFERENCE_TRUE_RESTORED_FALSE"
    assert decision["reference_label"] == "above threshold"
    assert decision["restored_label"] == "below threshold"


def test_public_audit_rejects_incomplete_provenance() -> None:
    adapter = PairedAdapter()
    adapter.provenance = lambda: {"adapter": "incomplete"}  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="adapter provenance is incomplete"):
        audit(
            adapter=adapter,
            branch_step=0,
            horizons=[1],
            decision="above_threshold",
        )


def test_adapter_check_passes_a_conformant_paired_adapter() -> None:
    report = check_adapter(PairedAdapter(), decision="above_threshold")

    assert report.verdict is ContractVerdict.SUPPORTED
    assert report.passed is True
    assert all(check.status is AdapterCheckStatus.PASS for check in report.checks)
    assert report.to_dict()["verdict"] == "SUPPORTED"


class LiveViewAdapter(PairedAdapter):
    def capture(self, env_ids: tuple[int, ...]) -> Snapshot:
        size = len(env_ids)
        return Snapshot(
            protocol=self.snapshot_protocol,
            values={"x": self.x[:size], "counters": self.counters[:size]},
            captured_components=("x", "counter"),
            unavailable_components=("hidden integrator",),
            metadata={"environment": "paired-test"},
        )


def test_adapter_check_detects_a_live_snapshot_view() -> None:
    report = check_adapter(LiveViewAdapter(), decision="above_threshold")
    capture_check = next(item for item in report.checks if item.name == "capture_by_value")

    assert capture_check.status is AdapterCheckStatus.FAIL
    assert report.verdict is ContractVerdict.UNSUPPORTED


class BrokenActionDeliveryAdapter(PairedAdapter):
    def step(self, actions: object) -> StepRecord:
        requested = np.asarray(actions)
        applied = requested.copy()
        applied[1, 0] += 1.0
        record = super().step(applied)
        record.applied_actions = applied
        return record


def test_adapter_check_detects_unequal_applied_actions() -> None:
    report = check_adapter(BrokenActionDeliveryAdapter(), decision="above_threshold")
    delivery = next(item for item in report.checks if item.name == "identical_action_delivery")

    assert delivery.status is AdapterCheckStatus.FAIL
    assert report.verdict is ContractVerdict.UNSUPPORTED


def test_live_audit_rejects_unequal_applied_actions() -> None:
    result = audit(
        adapter=BrokenActionDeliveryAdapter(),
        protocol="visible_only",
        branch_step=0,
        horizons=[1],
        decision="above_threshold",
    )
    l2 = result.levels[1]["L2"]

    assert result.verdict is ContractVerdict.UNSUPPORTED
    assert l2["applied_action_delivery"] is False
    assert l2["first_action_delivery_mismatch"] == 1


def test_decision_contract_copies_parameters_and_audit_rejects_nondeterminism() -> None:
    parameters = {"thresholds": [1.0]}
    state = False

    def alternating(_record: TrajectoryRecord) -> bool:
        nonlocal state
        state = not state
        return state

    contract = DecisionContract(
        name="alternating",
        evaluator=alternating,
        parameters=parameters,
    )
    parameters["thresholds"].append(2.0)
    assert contract.parameters == {"thresholds": [1.0]}

    with pytest.raises(RuntimeError, match="nondeterministic"):
        audit(
            adapter=PairedAdapter(),
            branch_step=0,
            horizons=[1],
            decision=contract,
        )
