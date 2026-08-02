"""Focused runtime tests for the asset-free MuJoCo replay adapter."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="MuJoCo is not installed",
)

from ipfd.adapters.mujoco_replay import MuJoCoReplayAdapter
from ipfd.fidelity.contracts import ReplayAdapter, TrajectoryRecord


def _adapter(
    regime: str,
    protocol: str = "integration_with_warmstart",
    continuation: str = "restored",
) -> MuJoCoReplayAdapter:
    return MuJoCoReplayAdapter(
        settings={"regime": regime, "initial_position_jitter": 0.0},
        snapshot_protocol=protocol,
        continuation_mode=continuation,
    )


def _trajectory(adapter: MuJoCoReplayAdapter, steps: int = 90) -> TrajectoryRecord:
    records = []
    actions = []
    for step in range(steps):
        action = adapter.action(step, "recorded", (0, 1))
        actions.append(action[0].copy())
        records.append(adapter.step(action))
    return TrajectoryRecord(steps=records, actions=actions, env_id=0)


def test_adapter_is_import_gated_and_satisfies_protocol():
    adapter = _adapter("free_space")
    try:
        assert isinstance(adapter, ReplayAdapter)
        provenance = adapter.provenance()
        assert provenance["simulator"] == "MuJoCo"
        assert provenance["environment"] == "free_space"
        assert provenance["model_source"] == "embedded_asset_free_mjcf"
        assert "GPU" not in repr(provenance)
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("protocol", "captured", "omitted", "warmstart"),
    [
        (
            "minimal_visible",
            {"simulation_time", "generalized_positions", "generalized_velocities"},
            {"controls", "solver_acceleration_warmstart"},
            False,
        ),
        (
            "full_physics",
            {"simulation_time", "generalized_positions", "generalized_velocities", "plugin_state"},
            {"controls", "solver_acceleration_warmstart"},
            False,
        ),
        (
            "integration_with_warmstart",
            {"controls", "solver_acceleration_warmstart", "applied_generalized_forces"},
            set(),
            True,
        ),
    ],
)
def test_protocol_inventory_is_explicit(protocol, captured, omitted, warmstart):
    adapter = _adapter("sustained_contact", protocol)
    try:
        snapshot = adapter.capture((0,))
        assert captured <= set(snapshot.captured_components)
        assert omitted <= set(snapshot.unavailable_components)
        assert snapshot.metadata["state_size"] == snapshot.values["state_vectors"].shape[1]
        assert snapshot.metadata["component_sizes"]["generalized_positions"] == 1
        provenance = adapter.provenance()
        assert provenance["solver_state_availability"]["warm_start_in_snapshot_protocol"] is warmstart
        assert "derived_contact_manifolds_recomputed_by_mj_forward" in snapshot.unavailable_components
    finally:
        adapter.close()


@pytest.mark.parametrize("protocol", sorted(("minimal_visible", "full_physics", "integration_with_warmstart")))
def test_restore_round_trips_captured_exposed_state(protocol):
    adapter = _adapter("free_space", protocol)
    try:
        adapter.reset(seed=17)
        adapter.step(np.asarray([[0.25], [-0.75]]))
        snapshot = adapter.capture((0,))
        adapter.restore(snapshot, (1,))
        observed = adapter.observe((0, 1))
        np.testing.assert_array_equal(observed.scene_state["qpos"][0], observed.scene_state["qpos"][1])
        np.testing.assert_array_equal(observed.scene_state["qvel"][0], observed.scene_state["qvel"][1])
        np.testing.assert_array_equal(
            observed.policy_observations["state"][0], observed.policy_observations["state"][1]
        )
        np.testing.assert_array_equal(observed.sensor_state["sensordata"][0], observed.sensor_state["sensordata"][1])
    finally:
        adapter.close()


def test_cold_continuation_drops_warmstart_even_for_integration_protocol():
    restored = _adapter("sustained_contact", continuation="restored")
    cold = _adapter("sustained_contact", continuation="cold")
    try:
        for step in range(50):
            restored.step(restored.action(step, "recorded", (0, 1)))
            cold.step(cold.action(step, "recorded", (0, 1)))
        restored.restore(restored.capture((0,)), (1,))
        cold.restore(cold.capture((0,)), (1,))
        assert np.max(np.abs(restored._data[1].qacc_warmstart)) > 0.0
        np.testing.assert_array_equal(cold._data[1].qacc_warmstart, 0.0)
        assert restored.provenance()["solver_state_availability"]["warm_start_preserved_for_continuation"] is True
        assert cold.provenance()["solver_state_availability"]["warm_start_preserved_for_continuation"] is False
    finally:
        restored.close()
        cold.close()


def test_free_space_is_a_90_step_positive_control():
    adapter = _adapter("free_space")
    try:
        adapter.reset(seed=101)
        snapshot = adapter.capture((0,))
        adapter.restore(snapshot, (1,))
        record = _trajectory(adapter)
        assert adapter.decision(record, "within_bounds") is True
        assert adapter.decision(record, "collision") is False
        assert adapter.decision(record, "sustained_contact") is False
        assert adapter.decision(record, "stable_contact") is False
        assert adapter.decision(record, "forward_progress") is True
        final = record.steps[-1].observation.scene_state
        np.testing.assert_array_equal(final["qpos"][0], final["qpos"][1])
        np.testing.assert_array_equal(final["qvel"][0], final["qvel"][1])
    finally:
        adapter.close()


def test_intermittent_contact_has_a_bounded_contact_event_then_separates():
    adapter = _adapter("intermittent_contact")
    try:
        record = _trajectory(adapter)
        active = [bool(step.contact_state["active"][0]) for step in record.steps]
        assert any(active)
        assert not active[-1]
        assert adapter.decision(record, "collision") is True
        assert adapter.decision(record, "sustained_contact") is False
        assert adapter.decision(record, "stable_contact") is False
        assert adapter.decision(record, "within_bounds") is True
    finally:
        adapter.close()


def test_sustained_contact_settles_and_stays_stable_through_step_90():
    adapter = _adapter("sustained_contact")
    try:
        record = _trajectory(adapter)
        active = [bool(step.contact_state["active"][0]) for step in record.steps]
        assert not active[0]
        assert all(active[-20:])
        assert adapter.decision(record, "collision") is True
        assert adapter.decision(record, "sustained_contact") is True
        assert adapter.decision(record, "stable_contact") is True
        assert adapter.decision(record, "within_bounds") is True
    finally:
        adapter.close()


def test_unknown_decision_and_mismatched_protocol_fail_clearly():
    adapter = _adapter("free_space")
    other = _adapter("free_space", "minimal_visible")
    try:
        record = _trajectory(adapter, steps=1)
        with pytest.raises(ValueError, match="unknown MuJoCo decision"):
            adapter.decision(record, "task_success")
        with pytest.raises(ValueError, match="does not match adapter protocol"):
            other.restore(adapter.capture((0,)), (1,))
    finally:
        adapter.close()
        other.close()
