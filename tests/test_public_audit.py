from __future__ import annotations

import importlib

import numpy as np
import pytest

from ipfd import DecisionContract, audit, check_adapter
from ipfd.adapter_check import AdapterCheckStatus
from ipfd.fidelity.contracts import ContractVerdict, ObservationRecord, Snapshot, StepRecord, TrajectoryRecord

audit_module = importlib.import_module("ipfd.audit")


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


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"name": ""}, ValueError, "name"),
        ({"evaluator": 1}, TypeError, "callable"),
        ({"description": 1}, TypeError, "description"),
        ({"parameters": []}, TypeError, "parameters"),
        ({"true_label": ""}, ValueError, "true_label"),
        ({"false_label": ""}, ValueError, "false_label"),
        ({"true_label": "same", "false_label": "same"}, ValueError, "labels"),
    ],
)
def test_decision_contract_rejects_ambiguous_definitions(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {"name": "decision", "evaluator": lambda _record: True}
    values.update(kwargs)

    with pytest.raises(exception, match=message):
        DecisionContract(**values)  # type: ignore[arg-type]


def test_decision_contract_rejects_non_boolean_results_and_labels_false() -> None:
    contract = DecisionContract(name="decision", evaluator=lambda _record: 1)  # type: ignore[arg-type,return-value]

    with pytest.raises(TypeError, match="must return bool"):
        contract.evaluate(TrajectoryRecord(steps=[], actions=[], env_id=0))
    assert contract.label(False) == "false"


@pytest.mark.parametrize(
    ("tolerances", "exception", "message"),
    [
        ({"scene_state": {}}, ValueError, "default"),
        ({"default": {}, "": {}}, ValueError, "category names"),
        ({"default": [], "scene_state": {}}, TypeError, "must be a mapping"),
        ({"default": {"absolute": -1.0}}, ValueError, "finite and non-negative"),
        ({"default": {"relative": float("inf")}}, ValueError, "finite and non-negative"),
        ({"default": {"fields": []}}, TypeError, "fields.*mapping"),
        ({"default": {"fields": {"": {}}}}, ValueError, "field tolerance names"),
        ({"default": {"fields": {"x": []}}}, TypeError, "field tolerance default.x"),
        (
            {"default": {"fields": {"x": {"absolute": -1.0}}}},
            ValueError,
            "field tolerances must be finite",
        ),
        (
            {"default": {"fields": {"x": {"unit": ""}}}},
            ValueError,
            "units must be non-empty",
        ),
    ],
)
def test_tolerance_validation_fails_closed(
    tolerances: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        audit_module._normalized_tolerances(tolerances)


def test_tolerance_normalization_preserves_field_units() -> None:
    result = audit_module._normalized_tolerances(
        {
            "default": {"absolute": 0.1, "relative": 0.2},
            "scene_state": {
                "fields": {"qpos": {"absolute": 0.01, "unit": "m"}}
            },
        }
    )

    assert result["scene_state"]["fields"]["qpos"] == {
        "absolute": 0.01,
        "relative": 0.0,
        "unit": "m",
    }


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"branch_step": True}, ValueError, "branch_step"),
        ({"branch_step": -1}, ValueError, "branch_step"),
        ({"seed": True}, TypeError, "seed"),
        ({"horizons": "1"}, TypeError, "horizons"),
        ({"horizons": []}, ValueError, "positive integers"),
        ({"horizons": [True]}, ValueError, "positive integers"),
        ({"continuation": ""}, ValueError, "continuation"),
        ({"continuation": "policy"}, ValueError, "must be one of"),
        ({"action_source": ""}, ValueError, "action_source"),
        ({"decision": ""}, ValueError, "decision"),
        ({"decision": 1}, TypeError, "DecisionContract"),
        ({"protocol": "different"}, ValueError, "does not match"),
        ({"metadata": {"task": ""}}, ValueError, "metadata"),
    ],
)
def test_public_audit_rejects_invalid_arguments(
    overrides: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "adapter": PairedAdapter(),
        "branch_step": 0,
        "horizons": [1],
        "decision": "above_threshold",
    }
    values.update(overrides)

    with pytest.raises(exception, match=message):
        audit(**values)  # type: ignore[arg-type]


def test_public_audit_rejects_missing_interface_method() -> None:
    adapter = PairedAdapter()
    adapter.step = None  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="missing required methods: step"):
        audit(adapter=adapter, branch_step=0, horizons=[1], decision="above_threshold")


def test_frontier_reports_insufficient_and_non_monotonic_support() -> None:
    configurations = [
        {"scope": {"horizon": 1}, "result": "SUPPORTED"},
        {"scope": {"horizon": 5}, "result": "INSUFFICIENT_EVIDENCE"},
        {"scope": {"horizon": 10}, "result": "SUPPORTED"},
        {"scope": {"horizon": 30}, "result": "UNSUPPORTED"},
    ]

    result = audit_module._frontier(configurations)

    assert result["first_untrusted_horizon"] == 5
    assert result["last_supported_horizon_before_failure"] == 1
    assert result["non_monotonic"] is True


def test_adapter_check_stops_on_a_missing_required_method() -> None:
    adapter = PairedAdapter()
    adapter.observe = None  # type: ignore[method-assign]

    report = check_adapter(adapter, decision="above_threshold")

    assert report.verdict is ContractVerdict.UNSUPPORTED
    assert [item.name for item in report.checks] == ["required_interface"]
    assert report.checks[0].evidence["missing_methods"] == ["observe"]


class RaisingCaptureAdapter(PairedAdapter):
    def capture(self, env_ids: tuple[int, ...]) -> Snapshot:
        del env_ids
        raise RuntimeError("capture unavailable")


def test_adapter_check_retains_fail_closed_evidence_when_capture_raises() -> None:
    report = check_adapter(RaisingCaptureAdapter(), decision="above_threshold")
    statuses = {item.name: item.status for item in report.checks}

    assert report.verdict is ContractVerdict.UNSUPPORTED
    assert statuses["deterministic_repeated_capture"] is AdapterCheckStatus.FAIL
    assert statuses["explicit_unavailable_state"] is AdapterCheckStatus.INSUFFICIENT_EVIDENCE
    assert statuses["capture_by_value"] is AdapterCheckStatus.FAIL
    assert statuses["protocol_mismatch_rejection"] is AdapterCheckStatus.FAIL
    assert statuses["compatible_restore_boundary"] is AdapterCheckStatus.FAIL
    assert statuses["observe_side_effects"] is AdapterCheckStatus.FAIL
    assert statuses["automatic_reset_not_observed"] is AdapterCheckStatus.INSUFFICIENT_EVIDENCE
