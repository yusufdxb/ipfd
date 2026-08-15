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
    **settings,
) -> MuJoCoReplayAdapter:
    return MuJoCoReplayAdapter(
        settings={"regime": regime, "initial_position_jitter": 0.0, **settings},
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
            {
                "simulation_time",
                "generalized_positions",
                "generalized_velocities",
                "actuator_activations",
                "controls",
            },
            {"solver_acceleration_warmstart"},
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


def test_filtered_contact_exposes_delayed_decision_failure_from_omitted_activation():
    minimal = _adapter(
        "filtered_contact",
        "minimal_visible",
        minimal_capture_actuator_activation=False,
        activation_preload_steps=100,
        post_preload_control=0.55,
        timestep=0.002,
        frame_skip=1,
    )
    integration = _adapter(
        "filtered_contact",
        "integration_with_warmstart",
        activation_preload_steps=100,
        post_preload_control=0.55,
        timestep=0.002,
        frame_skip=1,
    )
    try:
        for adapter in (minimal, integration):
            for step in range(100):
                adapter.step(adapter.action(step, "deterministic", (0, 1)))
            adapter.restore(adapter.capture((0,)), (1,))

        boundary = minimal.observe((0, 1))
        np.testing.assert_array_equal(boundary.scene_state["qpos"][0], boundary.scene_state["qpos"][1])
        np.testing.assert_array_equal(boundary.scene_state["qvel"][0], boundary.scene_state["qvel"][1])
        np.testing.assert_array_equal(
            boundary.policy_observations["state"][0], boundary.policy_observations["state"][1]
        )
        np.testing.assert_array_equal(
            boundary.controller_targets["ctrl"][0], boundary.controller_targets["ctrl"][1]
        )
        assert "actuator_activation_state" in boundary.unavailable
        assert "actuator_activations" in minimal.capture((0,)).unavailable_components
        assert minimal._data[0].act[0] < -0.95
        np.testing.assert_array_equal(minimal._data[1].act, 0.0)

        records = {minimal: [], integration: []}
        errors = {minimal: [], integration: []}
        for continuation_step in range(1, 91):
            absolute_step = 100 + continuation_step - 1
            for adapter in (minimal, integration):
                record = adapter.step(adapter.action(absolute_step, "deterministic", (0, 1)))
                records[adapter].append(record)
                errors[adapter].append(
                    float(
                        np.max(
                            np.abs(
                                record.observation.policy_observations["state"][0]
                                - record.observation.policy_observations["state"][1]
                            )
                        )
                    )
                )

        assert max(errors[minimal][:10]) < 0.01
        assert max(errors[minimal][:30]) < 0.01
        assert errors[minimal][89] > 0.01
        reference = TrajectoryRecord(steps=records[minimal], actions=[], env_id=0)
        restored = TrajectoryRecord(steps=records[minimal], actions=[], env_id=1)
        assert minimal.decision(reference, "remains_in_contact") is True
        assert minimal.decision(restored, "remains_in_contact") is False
        restored_contact = [bool(item.contact_state["active"][1]) for item in records[minimal]]
        assert all(restored_contact[:60])
        assert any(not value for value in restored_contact[60:])

        np.testing.assert_array_equal(errors[integration], 0.0)
        integration_reference = TrajectoryRecord(steps=records[integration], actions=[], env_id=0)
        integration_restored = TrajectoryRecord(steps=records[integration], actions=[], env_id=1)
        assert integration.decision(integration_reference, "remains_in_contact") is True
        assert integration.decision(integration_restored, "remains_in_contact") is True
    finally:
        minimal.close()
        integration.close()


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


def test_restore_rejects_incompatible_integration_configuration():
    source = _adapter("filtered_contact", timestep=0.002)
    target = _adapter("filtered_contact", timestep=0.01)
    try:
        snapshot = source.capture((0,))
        assert snapshot.metadata["model_xml_sha256"]
        assert snapshot.metadata["model_configuration_sha256"]
        with pytest.raises(ValueError, match="integration configuration"):
            target.restore(snapshot, (1,))
        provenance = source.provenance()
        assert provenance["model_configuration"]["timestep"] == 0.002
        assert provenance["task_state_captured"] == ["initial_qpos"]
        assert "contact_active" in provenance["task_state_recomputed_or_measured"]
    finally:
        source.close()
        target.close()
